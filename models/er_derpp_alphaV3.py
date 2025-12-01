import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model

# ============================================================================
# Alpha Controller V3 (保持原汁原味)
# ============================================================================
class AlphaControllerV3(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.ctx_dim = 8
        self.mlp_ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.project_feature = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.readout = nn.Linear(d_model, 1)

    def forward(self, ctx, mu_old, mu_new):
        ctx_emb = self.mlp_ctx(ctx)
        old_emb = self.project_feature(mu_old)
        new_emb = self.project_feature(mu_new)
        tokens = torch.stack([ctx_emb, old_emb, new_emb], dim=1)
        out = self.transformer(tokens)
        h_ctx = out[:, 0, :]
        logit = self.readout(h_ctx)
        s = torch.sigmoid(logit)[0, 0]
        # Range: [0.55, 0.75]
        w_old = 0.55 + 0.20 * s
        w_new = 1.0 - w_old
        return w_old, w_new


# ============================================================================
# ER + DER++ + Meta-Alpha (V3) - Decoupled Edition
# ============================================================================
@register_model("er_derpp_alphaV3")
class ERDERPPAlphaV3(ContinualModel):
    """
    Decoupled DER++ with Meta-Alpha V3.
    Strategy:
    - MSE Loss (Logit Distillation): Treated as a HARD constraint (always alpha * L_mse).
    - CE Loss (Label Replay): Controlled by Meta-Alpha (w_old * beta * L_ce).
    - New Loss: Controlled by Meta-Alpha (w_new * L_new).
    """
    NAME = "er_derpp_alphaV3"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        
        # DER++ Args
        parser.add_argument('--alpha', type=float, default=0.1, help='DER++ MSE weight')
        parser.add_argument('--beta', type=float, default=0.5, help='DER++ CE weight')

        # Meta-Alpha Args
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        parser.add_argument("--total_tasks", type=int, default=5)
        parser.add_argument("--w_reg_strength", type=float, default=0.01)
        parser.add_argument("--meta_lr", type=float, default=1e-3)
        parser.add_argument("--meta_interval", type=int, default=50)
        parser.add_argument("--meta_grad_balance_coef", type=float, default=0.5)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512

        self.controller = AlphaControllerV3(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            nhead=self.args.dam_nhead,
            num_layers=self.args.dam_layers,
        ).to(self.device)

        self.opt = torch.optim.SGD(
            self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.optim_wd,
            momentum=self.args.optim_mom,
        )

        self.opt_cont = torch.optim.Adam(
            self.controller.parameters(),
            lr=self.args.meta_lr,
            weight_decay=1e-5,
        )

        self.current_task_id = 0
        self.global_step = 0
        self.log_steps = 0
        self.log_w_old_sum = 0.0
        self.log_w_new_sum = 0.0
        
        seed = getattr(self.args, "seed", 0)
        self.stats_file = f"er_derpp_alphaV3_stats_seed{seed}.txt"

    def _get_features(self, x):
        return self.net(x, returnt="features")
    
    def forward(self, x):
        return self.net(x)

    def _build_ctx_and_grads(self, loss_new, loss_old, features_new, features_old):
        # Standard context builder
        l_new_val = loss_new.item()
        l_old_val = loss_old.item()
        denom = l_old_val + l_new_val + 1e-8

        head_params = list(self.net.classifier.parameters())
        g_old = torch.autograd.grad(loss_old, head_params, retain_graph=True, allow_unused=True)
        g_new = torch.autograd.grad(loss_new, head_params, retain_graph=True, allow_unused=True)

        g_old_flat = torch.cat([g.view(-1) for g in g_old if g is not None])
        g_new_flat = torch.cat([g.view(-1) for g in g_new if g is not None])

        norm_old = g_old_flat.norm() + 1e-8
        norm_new = g_new_flat.norm() + 1e-8
        cos_theta = (g_old_flat @ g_new_flat) / (norm_old * norm_new)
        
        norm_old_scaled = torch.log(norm_old) / 5.0
        norm_new_scaled = torch.log(norm_new) / 5.0
        t_norm_val = self.current_task_id / max(getattr(self.args, "total_tasks", 5) - 1, 1)

        ctx_vec = torch.tensor(
            [l_old_val/denom, l_new_val/denom, l_new_val-l_old_val, denom,
             t_norm_val, cos_theta.clamp(-1,1).item(), norm_old_scaled.item(), norm_new_scaled.item()],
            device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        
        mu_new = features_new.mean(dim=0, keepdim=True)
        mu_old = features_old.mean(dim=0, keepdim=True)
        return ctx_vec, mu_new, mu_old, norm_old, norm_new

    def end_task(self, dataset):
        if self.log_steps > 0:
            avg_old = self.log_w_old_sum / self.log_steps
            avg_new = self.log_w_new_sum / self.log_steps
            msg = (f"\n[DER++ & AlphaV3] Task {self.current_task_id + 1} | "
                   f"Avg w_old: {avg_old:.4f} | Avg w_new: {avg_new:.4f}")
            print(msg)
            try:
                with open(self.stats_file, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        self.log_steps = 0
        self.log_w_old_sum = 0.0
        self.log_w_new_sum = 0.0
        self.current_task_id += 1
        super().end_task(dataset)

    # ========================================================================
    # 核心修正：Decoupled MSE Optimization
    # ========================================================================
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        # 1. New Task Forward
        feats_new = self._get_features(inputs)
        outputs_new = self.net.classifier(feats_new)
        loss_new = self.loss(outputs_new, labels)

        if not self.buffer.is_empty():
            # 2. Buffer Data (Fix: correct unpacking of 3 items)
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            
            feats_old = self._get_features(buf_inputs)
            outputs_old = self.net.classifier(feats_old)

            # --- DER++ Components ---
            loss_mse = F.mse_loss(outputs_old, buf_logits)
            loss_ce = self.loss(outputs_old, buf_labels)
            
            # --- Controller Input ---
            # Controller 只负责平衡 Label Conflict (CE vs CE)
            # 我们把 beta * CE 作为 Old Loss 喂给 Controller，这样信号是对齐的
            loss_old_for_ctx = self.args.beta * loss_ce
            
            ctx_vec, mu_new, mu_old, _, _ = self._build_ctx_and_grads(
                loss_new, loss_old_for_ctx, feats_new, feats_old
            )

            # --- Controller Decision ---
            with torch.no_grad():
                w_old, w_new = self.controller(ctx_vec, mu_old, mu_new)

            # --- Decoupled Loss Composition (Fix: MSE is HARD constraint) ---
            # 1. New Task (Weighted)
            term_new = w_new * loss_new
            
            # 2. Old Labels (Weighted, Controlled by Alpha Policy)
            term_old_ce = w_old * (self.args.beta * loss_ce)
            
            # 3. Old Logits (Constant Hard Constraint, Unweighted by Policy)
            # 这保证了Task 1不会因为w_old波动而崩塌
            term_mse = self.args.alpha * loss_mse 
            
            loss_main = term_new + term_old_ce + term_mse
            
            # Regularization
            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            target = 0.6
            loss_reg = reg_strength * (w_old - target) ** 2

            loss = loss_main + loss_reg

            self.log_steps += 1
            self.log_w_old_sum += float(w_old.item())
            self.log_w_new_sum += float(w_new.item())
        else:
            loss = loss_new

        loss.backward()
        self.opt.step()

        # Store Logits for DER++
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
            logits=outputs_new.data
        )

        # -------------------------------------
        # 2. Phi-Step (Controller Update)
        # -------------------------------------
        if (self.global_step % self.args.meta_interval == 0) and (not self.buffer.is_empty()):
            self.net.eval()
            self.controller.train()
            self.opt_cont.zero_grad()

            m_inputs, m_labels = inputs, labels
            # Fix: unpack 3 items here too
            m_buf_inputs, m_buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )

            m_f_new = self._get_features(m_inputs)
            m_l_new = self.loss(self.net.classifier(m_f_new), m_labels)
            
            m_f_old = self._get_features(m_buf_inputs)
            # Phi-step 依然只看 CE 平衡，让 Controller 学会基本的 Loss 冲突管理
            m_l_old_ce = self.loss(self.net.classifier(m_f_old), m_buf_labels)
            m_l_old_weighted = self.args.beta * m_l_old_ce

            ctx_meta, mu_n_m, mu_o_m, n_o_m, n_n_m = self._build_ctx_and_grads(
                m_l_new, m_l_old_weighted, m_f_new, m_f_old
            )
            n_o_m, n_n_m = n_o_m.detach(), n_n_m.detach()

            w_o_m, w_n_m = self.controller(ctx_meta, mu_o_m, mu_n_m)
            
            grad_bal = F.relu(w_n_m*n_n_m - w_o_m*n_o_m)**2
            reg = 0.01 * (w_o_m - 0.6)**2
            meta_loss = 0.5 * grad_bal + reg
            
            meta_loss.backward()
            self.opt_cont.step()

        return loss.item()