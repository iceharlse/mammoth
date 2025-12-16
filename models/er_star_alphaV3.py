import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model

# 引入 STAR 组件
from models.star_utils.star_perturber import Perturber, add_perturb_args

# ============================================================================
# Alpha Controller V3 (完全保留你的 V3 版本)
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
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
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
        
        # RunB V3 Logic: w_old in [0.55, 0.75]
        lo, hi = 0.55, 0.75
        w_old = lo + (hi - lo) * s
        w_new = 1.0 - w_old
        return w_old, w_new


# ============================================================================
# ER + STAR + Meta-Alpha (V3)
# ============================================================================
@register_model("er_star_alphaV3")
class ERSTARAlphaV3(ContinualModel):
    """
    ER + STAR + Meta-Alpha V3
    证明 V3 (动态权重) 与 STAR (拓扑正则) 是正交的。
    """
    NAME = "er_star_alphaV3"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        # 1. 加入 STAR 的参数
        add_perturb_args(parser)
        
        # 2. 加入 V3 的参数
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        parser.add_argument("--total_tasks", type=int, default=5)
        parser.add_argument("--w_reg_strength", type=float, default=0.01)
        parser.add_argument("--meta_lr", type=float, default=1e-3)
        parser.add_argument("--meta_interval", type=int, default=50)
        parser.add_argument("--meta_grad_balance_coef", type=float, default=0.5)
        parser.add_argument(
            "--alpha_save_path",
            type=str,
            default=None,
            help="If set, save controller.state_dict() to this path at each end_task."
        )
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512

        # 初始化 V3 Controller
        self.controller = AlphaControllerV3(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            nhead=self.args.dam_nhead,
            num_layers=self.args.dam_layers,
        ).to(self.device)

        # 初始化 STAR Perturber
        self.pert = Perturber(self)

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
        self.stats_file = f"er_star_alphaV3_stats_seed{seed}.txt"

    def _get_features(self, x):
        return self.net(x, returnt="features")

    def forward(self, x):
        return self.net(x)

    # 这里的 Context Builder 和 V3 一模一样
    def _build_ctx_and_grads(self, loss_new, loss_old, features_new, features_old):
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
        # print and log average weights per task
        if self.log_steps > 0:
            avg_old = self.log_w_old_sum / self.log_steps
            avg_new = self.log_w_new_sum / self.log_steps
            msg = (f"\n[Meta-Alpha] Task {self.current_task_id + 1} | "
                   f"Avg w_old: {avg_old:.4f} | Avg w_new: {avg_new:.4f}")
            print(msg)
            try:
                with open(self.stats_file, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

        # 新增：按需保存 controller
        save_path = getattr(self.args, "alpha_save_path", None)
        if save_path is not None:
            try:
                torch.save(self.controller.state_dict(), save_path)
                print(f"[Meta-Alpha] Saved controller to {save_path}")
            except Exception as e:
                print(f"[Meta-Alpha] Failed to save controller: {e}")

        self.log_steps = 0
        self.log_w_old_sum = 0.0
        self.log_w_new_sum = 0.0

        self.current_task_id += 1
        super().end_task(dataset)


    # ========================================================================
    # 核心融合逻辑: Observe
    # ========================================================================
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # ---------------------------------------------
        # Step 1: Theta-Step (Backbone Update)
        # ---------------------------------------------
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        # [STAR 关键植入点]
        # 在计算任何 loss 之前，先让 STAR 对 memory 进行扰动/正则
        # 这会影响后续 self.net 的状态或梯度
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            self.pert(buf_inputs, buf_labels)

        # [V3 关键逻辑]
        # 分别计算 New Loss 和 Old Loss，交给 Controller 赋权
        feats_new = self._get_features(inputs)
        out_new = self.net.classifier(feats_new)
        loss_new = self.loss(out_new, labels)

        if not self.buffer.is_empty():
            # 重新拿 Buffer 数据用于计算 Weighted Loss
            # (注意：STAR 的 er_star.py 里是拼接 inputs，但为了 V3 的加权，我们需要分开算)
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            feats_old = self._get_features(buf_inputs)
            out_old = self.net.classifier(feats_old)
            loss_old = self.loss(out_old, buf_labels)

            # 构建 Context
            ctx_vec, mu_new, mu_old, _, _ = self._build_ctx_and_grads(
                loss_new, loss_old, feats_new, feats_old
            )

            # Controller 决策
            with torch.no_grad():
                w_old, w_new = self.controller(ctx_vec, mu_old, mu_new)

            # 加权 Loss + Alpha 正则
            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            target = 0.6
            loss_main = w_old * loss_old + w_new * loss_new
            loss_reg = reg_strength * (w_old - target) ** 2
            
            # 最终 Loss
            loss = loss_main + loss_reg
            
            self.log_steps += 1
            self.log_w_old_sum += float(w_old.item())
            self.log_w_new_sum += float(w_new.item())
        else:
            loss = loss_new

        loss.backward()
        self.opt.step()

        # 更新 Buffer
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        # ---------------------------------------------
        # Step 2: Phi-Step (Controller Update) - 保持 V3 原样
        # ---------------------------------------------
        if (self.global_step % self.args.meta_interval == 0) and (not self.buffer.is_empty()):
            self.net.eval()
            self.controller.train()
            self.opt_cont.zero_grad()

            m_inputs, m_labels = inputs, labels
            m_buf_inputs, m_buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )

            # Meta Forward
            m_f_new = self._get_features(m_inputs)
            m_l_new = self.loss(self.net.classifier(m_f_new), m_labels)
            m_f_old = self._get_features(m_buf_inputs)
            m_l_old = self.loss(self.net.classifier(m_f_old), m_buf_labels)

            ctx_meta, mu_n_m, mu_o_m, n_o_m, n_n_m = self._build_ctx_and_grads(
                m_l_new, m_l_old, m_f_new, m_f_old
            )
            # Detach norms for loss calculation
            n_o_m = n_o_m.detach()
            n_n_m = n_n_m.detach()

            # Controller Update
            w_o_m, w_n_m = self.controller(ctx_meta, mu_o_m, mu_n_m)
            
            # Grad Balance Loss
            grad_bal = F.relu(w_n_m * n_n_m - w_o_m * n_o_m) ** 2
            reg_term = 0.01 * (w_o_m - 0.6) ** 2
            meta_loss = 0.5 * grad_bal + reg_term

            meta_loss.backward()
            self.opt_cont.step()

        return loss.item()