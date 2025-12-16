import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model

# ============================================================================
# 1. Controller (V8 Logic)
#    - 保持 V8 的逻辑，因为它是唯一跑通过的。
#    - 关键在于 Observe 里的执行逻辑，而不是控制器本身。
# ============================================================================
class AlphaBetaControllerV11(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        alpha_min: float = 0.60, # [Safety] Raised min stability
        alpha_max: float = 0.85, # [Safety] Raised max stability
        num_layers_out: int = 25,
    ):
        super().__init__()
        self.ctx_dim = 8
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.mlp_ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.project_feature = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head_p_stable = nn.Linear(d_model, 1)

        # Head 2: Beta
        self.head_beta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_layers_out)
        )
        # Init Bias: -1.0
        nn.init.constant_(self.head_beta[2].bias, -1.0) 

    def forward(self, ctx, mu_old, mu_new):
        ctx_emb = self.mlp_ctx(ctx)
        old_emb = self.project_feature(mu_old)
        new_emb = self.project_feature(mu_new)

        tokens = torch.stack([ctx_emb, old_emb, new_emb], dim=1)
        out = self.transformer(tokens)
        h_ctx = out[:, 0, :]

        p_stable = torch.sigmoid(self.head_p_stable(h_ctx)).squeeze()
        w_old = p_stable * self.alpha_max + (1.0 - p_stable) * self.alpha_min
        w_new = 1.0 - w_old

        beta_logits = self.head_beta(h_ctx).squeeze()
        beta_profile = torch.sigmoid(beta_logits)

        return w_old, w_new, p_stable, beta_profile


# ============================================================================
# 2. ER + DAM V11 (Gradient Matching Fix)
#    - Solves the "New Task Dominance" by strictly normalizing gradients
# ============================================================================
@register_model("er_dam_cagradV11")
class ER_DAM_V11_Final(ContinualModel):
    NAME = "er_dam_cagradV11"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        
        parser.add_argument("--meta_lr", type=float, default=5e-4) # Keep low LR
        parser.add_argument("--meta_interval", type=int, default=20)
        parser.add_argument("--w_reg_strength", type=float, default=0.01)
        parser.add_argument("--beta_meta_coef", type=float, default=1.0)
        
        parser.add_argument("--alpha_min", type=float, default=0.60)
        parser.add_argument("--alpha_max", type=float, default=0.85)
        
        parser.add_argument("--ctx_ema_beta", type=float, default=0.9)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512

        self.layer_names = []
        for name, param in self.net.named_parameters():
            if param.requires_grad and ("weight" in name) and ("bn" not in name):
                layer_id = ".".join(name.split(".")[:-1])
                if layer_id not in self.layer_names:
                    self.layer_names.append(layer_id)
        
        self.num_layers_out = max(len(self.layer_names), 1)

        self.controller = AlphaBetaControllerV11(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            num_layers_out=self.num_layers_out,
            alpha_min=getattr(self.args, "alpha_min", 0.60),
            alpha_max=getattr(self.args, "alpha_max", 0.85),
        ).to(self.device)

        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        self.opt_cont = torch.optim.Adam(self.controller.parameters(), lr=self.args.meta_lr, weight_decay=1e-5)
        
        self.global_step = 0
        self.current_task_id = 0
        self.ctx_ema = None

    def _get_features(self, x):
        return self.net(x, returnt="features")

    def _get_layer_groups(self):
        groups = {name: [] for name in self.layer_names}
        for name, param in self.net.named_parameters():
            if not param.requires_grad: continue
            for key in self.layer_names:
                if name.startswith(key):
                    groups[key].append(param)
                    break
        return groups

    def _build_ctx(self, loss_new, loss_old, feats_new, feats_old):
        l_n, l_o = loss_new.item(), loss_old.item()
        denom = l_n + l_o + 1e-8
        ctx = torch.tensor([
            l_o / denom, l_n / denom, l_n - l_o, l_n + l_o,
            self.current_task_id / 5.0, 0, 0, 0
        ]).to(self.device).float().unsqueeze(0)
        mu_n = feats_new.mean(dim=0, keepdim=True).detach()
        mu_o = feats_old.mean(dim=0, keepdim=True).detach()
        return ctx, mu_n, mu_o

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1
        
        # --- 1. Theta Step ---
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        feats_new = self._get_features(inputs)
        loss_new = self.loss(self.net.classifier(feats_new), labels)
        
        if self.buffer.is_empty():
            loss_new.backward()
            self.opt.step()
            self.buffer.add_data(not_aug_inputs, labels)
            return loss_new.item()

        buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
        feats_old = self._get_features(buf_inputs)
        loss_old = self.loss(self.net.classifier(feats_old), buf_labels)

        with torch.no_grad():
            ctx, mu_n, mu_o = self._build_ctx(loss_new, loss_old, feats_new, feats_old)
            if self.ctx_ema is None: self.ctx_ema = ctx
            else: self.ctx_ema = self.args.ctx_ema_beta * self.ctx_ema + (1 - self.args.ctx_ema_beta) * ctx
            w_old, w_new, p_stable, beta_profile = self.controller(self.ctx_ema, mu_o, mu_n)
            w_old = w_old.item()
            w_new = w_new.item()
            if math.isnan(w_old): w_old = 0.6; w_new = 0.4

        # Gradients
        params = [p for p in self.net.parameters() if p.requires_grad]
        grads_new = torch.autograd.grad(loss_new, params, retain_graph=True, allow_unused=True)
        grads_old = torch.autograd.grad(loss_old, params, retain_graph=False, allow_unused=True)

        p_to_gn = {p: g for p, g in zip(params, grads_new)}
        p_to_go = {p: g for p, g in zip(params, grads_old)}

        # --- [CRITICAL FIX] GRADIENT MAGNITUDE MATCHING ---
        layer_groups = self._get_layer_groups()
        for p in params: p.grad = None
        
        for i, (layer_name, layer_params) in enumerate(layer_groups.items()):
            if not layer_params: continue
            
            beta_l = beta_profile[i].item() if i < len(beta_profile) else beta_profile[-1].item()
            
            # 1. Calculate Norms for this layer
            gn_norm = 0.0
            go_norm = 0.0
            for p in layer_params:
                gn = p_to_gn.get(p)
                go = p_to_go.get(p)
                if gn is not None: gn_norm += gn.norm().item()**2
                if go is not None: go_norm += go.norm().item()**2
            gn_norm = math.sqrt(gn_norm)
            go_norm = math.sqrt(go_norm)
            
            # 2. Calculate Scaling Factor
            # If New Grad is HUGE (e.g. 100) and Old Grad is tiny (e.g. 1),
            # we scale New Grad down so its magnitude matches Old Grad.
            # This ensures beta_l actually controls the proportion, not just the magnitude.
            if gn_norm > 1e-8 and go_norm > 1e-8:
                scaler = go_norm / gn_norm
                # Only scale DOWN. If new grad is smaller, don't scale up.
                scaler = min(scaler, 1.0) 
            else:
                scaler = 1.0

            for p in layer_params:
                gn = p_to_gn.get(p)
                go = p_to_go.get(p)
                if gn is None: gn = torch.zeros_like(p)
                if go is None: go = torch.zeros_like(p)

                # 3. Apply Scaled Gradient
                grad_stable = w_old * go 
                grad_plastic = (w_new * beta_l) * (gn * scaler) # <--- The Magic
                
                p.grad = grad_stable + grad_plastic

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.buffer.add_data(not_aug_inputs, labels)

        # --- 2. Meta Step ---
        if self.global_step % self.args.meta_interval == 0:
            self._meta_update(inputs, labels, buf_inputs, buf_labels)

        return loss_new.item() + loss_old.item()

    def _meta_update(self, val_inputs, val_labels, buf_inputs, buf_labels):
        self.net.eval()
        self.controller.train()
        self.opt_cont.zero_grad()

        feats_new = self._get_features(val_inputs)
        loss_new = self.loss(self.net.classifier(feats_new), val_labels)
        feats_old = self._get_features(buf_inputs)
        loss_old = self.loss(self.net.classifier(feats_old), buf_labels)

        ctx, mu_n, mu_o = self._build_ctx(loss_new, loss_old, feats_new, feats_old)
        w_old, w_new, p_stable, beta_profile = self.controller(ctx, mu_o, mu_n)

        # Loss 1: Alpha
        with torch.no_grad():
            l_o, l_n = loss_old.item(), loss_new.item()
            target_w_old = l_o / (l_o + l_n + 1e-8)
            target_w_old = max(0.2, min(0.8, target_w_old))
        loss_alpha = F.mse_loss(w_old, torch.tensor(target_w_old).to(self.device))

        # Loss 2: Beta (Ratio)
        params = [p for p in self.net.parameters() if p.requires_grad]
        g_meta_new = torch.autograd.grad(loss_new, params, retain_graph=True, allow_unused=True)
        g_meta_old = torch.autograd.grad(loss_old, params, retain_graph=False, allow_unused=True)
        p_to_idx = {id(p): idx for idx, p in enumerate(params)}
        layer_groups = self._get_layer_groups()
        
        beta_targets = []
        for layer_name in self.layer_names:
            p_list = layer_groups[layer_name]
            if not p_list: 
                beta_targets.append(0.5)
                continue
            gn_sq_sum, go_sq_sum = 0.0, 0.0
            for p in p_list:
                idx = p_to_idx.get(id(p))
                if idx is None: continue
                gn, go = g_meta_new[idx], g_meta_old[idx]
                if gn is not None: gn_sq_sum += gn.norm().item()**2
                if go is not None: go_sq_sum += go.norm().item()**2
            
            denom = gn_sq_sum + go_sq_sum + 1e-8
            beta_star = gn_sq_sum / denom
            beta_targets.append(beta_star)
            
        if beta_targets:
            L = min(len(beta_targets), beta_profile.shape[0])
            loss_beta = F.mse_loss(beta_profile[:L], torch.tensor(beta_targets[:L]).to(self.device))
        else:
            loss_beta = torch.tensor(0.0).to(self.device)

        loss_reg = self.args.w_reg_strength * (p_stable - 0.5)**2
        total_meta_loss = loss_alpha + loss_reg + self.args.beta_meta_coef * loss_beta
        total_meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
        self.opt_cont.step()