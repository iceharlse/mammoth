import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model

# ============================================================================
# 1. Controller (Standard V8)
# ============================================================================
class AlphaBetaControllerV8(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        alpha_min: float = 0.55,
        alpha_max: float = 0.75,
        num_layers_out: int = 25,
    ):
        super().__init__()
        self.ctx_dim = 8
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Embeddings
        self.mlp_ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.project_feature = nn.Linear(feature_dim, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head 1: Alpha (Global Stability)
        self.head_p_stable = nn.Linear(d_model, 1)

        # Head 2: Beta (Layer-wise Plasticity Gate)
        self.head_beta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_layers_out),
        )
        # 初始偏置负一点，让 Beta 稍微保守一点
        nn.init.constant_(self.head_beta[2].bias, -1.0)

    def forward(self, ctx, mu_old, mu_new):
        # Embeddings
        ctx_emb = self.mlp_ctx(ctx)
        old_emb = self.project_feature(mu_old)
        new_emb = self.project_feature(mu_new)

        tokens = torch.stack([ctx_emb, old_emb, new_emb], dim=1)
        out = self.transformer(tokens)
        h_ctx = out[:, 0, :]

        # Alpha
        p_stable = torch.sigmoid(self.head_p_stable(h_ctx)).squeeze()
        w_old = p_stable * self.alpha_max + (1.0 - p_stable) * self.alpha_min
        w_new = 1.0 - w_old

        # Beta
        beta_logits = self.head_beta(h_ctx).squeeze()
        beta_profile = torch.sigmoid(beta_logits)

        return w_old, w_new, p_stable, beta_profile


# ============================================================================
# 2. ER + DAM V8, A-2: MGDA-blended Alpha in theta-step
# ============================================================================
@register_model("er_dam_cagradV8A2")
class ER_DAM_V8_A2(ContinualModel):
    NAME = "er_dam_cagradV8A2"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        # Controller
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)

        # Meta Training
        parser.add_argument("--meta_lr", type=float, default=1e-3)
        parser.add_argument("--meta_interval", type=int, default=20)
        parser.add_argument("--w_reg_strength", type=float, default=0.01)
        parser.add_argument(
            "--beta_meta_coef", type=float, default=1.0
        )  # Weight for Beta loss

        # Alpha Bounds for controller
        parser.add_argument("--alpha_min", type=float, default=0.55)
        parser.add_argument("--alpha_max", type=float, default=0.75)

        parser.add_argument("--ctx_ema_beta", type=float, default=0.9)

        # A-2: MGDA blend coefficient
        parser.add_argument(
            "--mgda_blend",
            type=float,
            default=0.3,
            help=(
                "blend coeff η for MGDA-Alpha: "
                "w_old_final = (1-η) * w_old_meta + η * alpha_mgda"
            ),
        )
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512

        # Identify layers
        self.layer_names = []
        for name, param in self.net.named_parameters():
            if param.requires_grad and ("weight" in name) and ("bn" not in name):
                layer_id = ".".join(name.split(".")[:-1])
                if layer_id not in self.layer_names:
                    self.layer_names.append(layer_id)

        self.num_layers_out = max(len(self.layer_names), 1)

        # Controller
        self.controller = AlphaBetaControllerV8(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            num_layers_out=self.num_layers_out,
            alpha_min=getattr(self.args, "alpha_min", 0.55),
            alpha_max=getattr(self.args, "alpha_max", 0.75),
        ).to(self.device)

        # Optimizers
        self.opt = torch.optim.SGD(
            self.net.parameters(),
            lr=self.args.lr,
            momentum=self.args.optim_mom,
            weight_decay=self.args.optim_wd,
        )
        self.opt_cont = torch.optim.Adam(
            self.controller.parameters(),
            lr=self.args.meta_lr,
            weight_decay=1e-5,
        )

        self.global_step = 0
        self.current_task_id = 0
        self.ctx_ema = None

    # --------------------------------------------------------------
    # utilities
    # --------------------------------------------------------------
    def _get_features(self, x):
        return self.net(x, returnt="features")

    def _get_layer_groups(self):
        groups = {name: [] for name in self.layer_names}
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            for key in self.layer_names:
                if name.startswith(key):
                    groups[key].append(param)
                    break
        return groups

    def _build_ctx(self, loss_new, loss_old, feats_new, feats_old):
        l_n, l_o = loss_new.item(), loss_old.item()
        denom = l_n + l_o + 1e-8
        ctx = torch.tensor(
            [
                l_o / denom,                  # 0: normalized old loss
                l_n / denom,                  # 1: normalized new loss
                l_n - l_o,                    # 2: loss gap
                l_n + l_o,                    # 3: total loss
                self.current_task_id / 5.0,   # 4: normalized task id
                0.0,
                0.0,
                0.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        mu_n = feats_new.mean(dim=0, keepdim=True).detach()
        mu_o = feats_old.mean(dim=0, keepdim=True).detach()
        return ctx, mu_n, mu_o

    # --------------------------------------------------------------
    # main training step (A-2: MGDA-blended Alpha)
    # --------------------------------------------------------------
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

        buf_inputs, buf_labels = self.buffer.get_data(
            self.args.minibatch_size,
            transform=self.transform,
            device=self.device,
        )
        feats_old = self._get_features(buf_inputs)
        loss_old = self.loss(self.net.classifier(feats_old), buf_labels)

        # ---- controller: ctx + Alpha/Beta ----
        with torch.no_grad():
            ctx, mu_n, mu_o = self._build_ctx(loss_new, loss_old, feats_new, feats_old)
            if self.ctx_ema is None:
                self.ctx_ema = ctx
            else:
                self.ctx_ema = (
                    self.args.ctx_ema_beta * self.ctx_ema
                    + (1 - self.args.ctx_ema_beta) * ctx
                )

            w_old_meta_t, w_new_meta_t, p_stable, beta_profile = self.controller(
                self.ctx_ema, mu_o, mu_n
            )

            w_old_meta = float(w_old_meta_t.item())
            w_new_meta = float(w_new_meta_t.item())
            if math.isnan(w_old_meta):
                w_old_meta, w_new_meta = 0.6, 0.4

        # ---- grads for new & old ----
        params = [p for p in self.net.parameters() if p.requires_grad]
        grads_new = torch.autograd.grad(
            loss_new, params, retain_graph=True, allow_unused=True
        )
        grads_old = torch.autograd.grad(
            loss_old, params, retain_graph=False, allow_unused=True
        )

        # --- A-2: MGDA closed-form alpha, blended with controller alpha ---
        with torch.no_grad():
            dot_oo = torch.tensor(0.0, device=self.device)
            dot_nn = torch.tensor(0.0, device=self.device)
            dot_on = torch.tensor(0.0, device=self.device)
            any_grad = False

            for gn, go in zip(grads_new, grads_old):
                if gn is None or go is None:
                    continue
                any_grad = True
                dot_oo += torch.sum(go * go)
                dot_nn += torch.sum(gn * gn)
                dot_on += torch.sum(gn * go)

            if not any_grad:
                alpha_mgda = torch.tensor(0.5, device=self.device)
            else:
                den = dot_oo + dot_nn - 2.0 * dot_on
                if den.abs() < 1e-8:
                    alpha_mgda = torch.tensor(0.5, device=self.device)
                else:
                    alpha_mgda = (dot_nn - dot_on) / (den + 1e-8)
                    alpha_mgda = torch.clamp(alpha_mgda, 0.0, 1.0)

            alpha_mgda_scalar = float(alpha_mgda.item())
            # 稍微夹一下，避免极端 0/1
            alpha_mgda_scalar = max(0.2, min(0.8, alpha_mgda_scalar))

            eta = getattr(self.args, "mgda_blend", 0.3)
            eta = max(0.0, min(1.0, float(eta)))

            w_old_final = (1.0 - eta) * w_old_meta + eta * alpha_mgda_scalar
            w_old_final = max(0.0, min(1.0, w_old_final))
            w_new_final = 1.0 - w_old_final

        # Map grads to params
        p_to_gn = {p: g for p, g in zip(params, grads_new)}
        p_to_go = {p: g for p, g in zip(params, grads_old)}

        # --- GATED MODULATION ---
        layer_groups = self._get_layer_groups()
        for p in params:
            p.grad = None

        for i, (layer_name, layer_params) in enumerate(layer_groups.items()):
            if not layer_params:
                continue

            # Map beta (handle size mismatch gracefully)
            if i < len(beta_profile):
                beta_l = beta_profile[i].item()
            else:
                beta_l = beta_profile[-1].item()

            for p in layer_params:
                gn = p_to_gn.get(p)
                go = p_to_go.get(p)
                if gn is None:
                    gn = torch.zeros_like(p)
                if go is None:
                    go = torch.zeros_like(p)

                # Stable Base + Gated Plasticity (using MGDA-blended weights)
                grad_stable = w_old_final * go
                grad_plastic = (w_new_final * beta_l) * gn
                p.grad = grad_stable + grad_plastic

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.buffer.add_data(not_aug_inputs, labels)

        # --- 2. Meta Step ---
        if self.global_step % self.args.meta_interval == 0:
            self._meta_update(inputs, labels, buf_inputs, buf_labels)

        return loss_new.item() + loss_old.item()

    # --------------------------------------------------------------
    # meta-update: Alpha target = loss-ratio (原版), Beta target = grad-norm ratio
    # --------------------------------------------------------------
    def _meta_update(self, val_inputs, val_labels, buf_inputs, buf_labels):
        self.net.eval()
        self.controller.train()
        self.opt_cont.zero_grad()

        feats_new = self._get_features(val_inputs)
        loss_new = self.loss(self.net.classifier(feats_new), val_labels)
        feats_old = self._get_features(buf_inputs)
        loss_old = self.loss(self.net.classifier(feats_old), buf_labels)

        ctx, mu_n, mu_o = self._build_ctx(loss_new, loss_old, feats_new, feats_old)

        # Forward with gradients w.r.t. controller
        w_old, w_new, p_stable, beta_profile = self.controller(ctx, mu_o, mu_n)

        # --- Meta Loss 1: Alpha (Regret Proxy; same as原V8) ---
        with torch.no_grad():
            l_o, l_n = loss_old.item(), loss_new.item()
            target_w_old = l_o / (l_o + l_n + 1e-8)
            target_w_old = max(0.2, min(0.8, target_w_old))
        loss_alpha = F.mse_loss(
            w_old, torch.tensor(target_w_old, device=self.device, dtype=w_old.dtype)
        )

        # --- Meta Loss 2: Beta (Gradient Norm Ratio; 原V8逻辑不动) ---
        params = [p for p in self.net.parameters() if p.requires_grad]
        g_meta_new = torch.autograd.grad(
            loss_new, params, retain_graph=True, allow_unused=True
        )
        g_meta_old = torch.autograd.grad(
            loss_old, params, retain_graph=False, allow_unused=True
        )

        p_to_idx = {id(p): idx for idx, p in enumerate(params)}
        layer_groups = self._get_layer_groups()

        beta_targets = []

        for layer_name in self.layer_names:
            p_list = layer_groups[layer_name]
            if not p_list:
                beta_targets.append(0.5)
                continue

            gn_sq_sum = 0.0
            go_sq_sum = 0.0

            for p in p_list:
                idx = p_to_idx.get(id(p))
                if idx is None:
                    continue
                gn = g_meta_new[idx]
                go = g_meta_old[idx]
                if gn is not None:
                    val = gn.detach().norm().item()
                    gn_sq_sum += val * val
                if go is not None:
                    val = go.detach().norm().item()
                    go_sq_sum += val * val

            denom = gn_sq_sum + go_sq_sum + 1e-8
            beta_star = gn_sq_sum / denom
            beta_targets.append(beta_star)

        if beta_targets:
            L = min(len(beta_targets), beta_profile.shape[0])
            beta_target_tensor = torch.tensor(
                beta_targets[:L], device=self.device, dtype=beta_profile.dtype
            )
            loss_beta = F.mse_loss(beta_profile[:L], beta_target_tensor)
        else:
            loss_beta = torch.tensor(0.0, device=self.device)

        # Reg for stability on p_stable
        loss_reg = self.args.w_reg_strength * (p_stable - 0.5) ** 2

        total_meta_loss = (
            loss_alpha + loss_reg + self.args.beta_meta_coef * loss_beta
        )
        total_meta_loss.backward()
        self.opt_cont.step()
