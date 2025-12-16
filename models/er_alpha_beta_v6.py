import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


# ================================================================
# 1. Alpha + Beta Controller V6（简化版）
#    - 输入：8 维 ctx + mu_old + mu_new
#    - 输出：
#        w_old, w_new: 全局 old/new 权重
#        p_stable:     标量 [0,1]，控制在 alpha_min/alpha_max 之间插值
#        beta_profile: 每层一个 beta_l ∈ [0,1]，表示“稳定度”
#
#    约定（很重要）：
#      beta_l ≈ 1 -> 这一层更 STABLE（偏向 old-grad）
#      beta_l ≈ 0 -> 这一层更 PLASTIC（偏向 new-grad）
# ================================================================

class AlphaBetaControllerV6(nn.Module):
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
        self.num_layers_out = num_layers_out

        # ctx -> d_model
        self.mlp_ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # feature -> d_model
        self.project_feature = nn.Linear(feature_dim, d_model)

        # transformer over [ctx, mu_old, mu_new]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # p_stable head
        self.head_p_stable = nn.Linear(d_model, 1)

        # beta-profile head
        self.head_beta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_layers_out),
            nn.Sigmoid(),    # beta_l ∈ [0,1]，代表“稳定度”
        )

    def forward(self, ctx, mu_old, mu_new):
        """
        ctx:    [B, 8]
        mu_old: [B, F]
        mu_new: [B, F]

        return:
          w_old, w_new, p_stable, beta_profile
          若 B==1，则返回标量 + 一维 beta_profile
        """
        B = ctx.size(0)

        ctx_emb = self.mlp_ctx(ctx)            # [B, D]
        old_emb = self.project_feature(mu_old) # [B, D]
        new_emb = self.project_feature(mu_new) # [B, D]

        tokens = torch.stack([ctx_emb, old_emb, new_emb], dim=1)  # [B,3,D]
        out = self.transformer(tokens)                             # [B,3,D]
        h_ctx = out[:, 0, :]                                      # [B,D]

        # p_stable ∈ [0,1]
        p_stable = torch.sigmoid(self.head_p_stable(h_ctx)).view(B)

        # p_stable 在 alpha_min / alpha_max 之间插值
        alpha_plastic = self.alpha_min
        alpha_stable = self.alpha_max
        w_old = p_stable * alpha_stable + (1.0 - p_stable) * alpha_plastic
        w_new = 1.0 - w_old

        # 每层 beta_profile（稳定度）
        beta_profile = self.head_beta(h_ctx)   # [B, L]

        if B == 1:
            w_old = w_old[0]
            w_new = w_new[0]
            p_stable = p_stable[0]
            beta_profile = beta_profile[0]

        return w_old, w_new, p_stable, beta_profile


# ================================================================
# 2. ER + AlphaBetaV6（无 CAGrad，按层混合梯度）
# ================================================================

@register_model("er_alpha_beta_v6")
class ERAlphaBetaV6(ContinualModel):
    """
    Experience Replay + AlphaBetaV6:

    - Base:  ER（current batch + buffer batch, CE loss）
    - Controller:
        * 全局 w_old / w_new：old / new 的整体比例
        * 每层 beta_l：这一层的“稳定度”（越大越偏 old-grad）
    - 梯度组合：
        g_final^l = beta_l * w_old * g_old^l
                    + (1 - beta_l) * w_new * g_new^l

    - Meta:
        * 对 alpha 用 V5 风格的 grad-balance surrogate:
            希望 w_old * ||g_old|| >= w_new * ||g_new||
        * 对 beta_l 用 per-layer grad norm:
            beta_l* ≈ ||g_old^l||^2 / (||g_old^l||^2 + ||g_new^l||^2)
    """

    NAME = "er_alpha_beta_v6"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)

        # controller 结构
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        parser.add_argument("--total_tasks", type=int, default=5)

        # meta 部分
        parser.add_argument("--meta_lr", type=float, default=1e-3)
        parser.add_argument("--meta_interval", type=int, default=50)
        parser.add_argument("--meta_grad_balance_coef", type=float, default=0.5)
        parser.add_argument("--w_reg_strength", type=float, default=0.01)
        parser.add_argument("--beta_meta_coef", type=float, default=1.0)

        # ctx EMA（只影响 theta-step 里用的 ctx，用来平滑）
        parser.add_argument("--ctx_ema_beta", type=float, default=0.9)

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # feature dim
        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512

        # 以 param 名字里去掉 ".weight" 后的前缀做 layer id
        self.layer_names = []
        for name, param in self.net.named_parameters():
            if param.requires_grad and ("weight" in name) and ("bn" not in name):
                layer_id = ".".join(name.split(".")[:-1])
                if layer_id not in self.layer_names:
                    self.layer_names.append(layer_id)
        num_layers_out = max(len(self.layer_names), 1)

        # controller
        self.controller = AlphaBetaControllerV6(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            nhead=self.args.dam_nhead,
            num_layers=self.args.dam_layers,
            alpha_min=0.55,
            alpha_max=0.75,
            num_layers_out=num_layers_out,
        ).to(self.device)

        # backbone optimizer
        self.opt = torch.optim.SGD(
            self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.optim_wd,
            momentum=self.args.optim_mom,
        )

        # controller optimizer
        self.opt_cont = torch.optim.Adam(
            self.controller.parameters(),
            lr=self.args.meta_lr,
            weight_decay=1e-5,
        )

        self.global_step = 0
        self.current_task_id = 0

        # ctx EMA（theta-step 用）
        self.ctx_ema = None

        # logging
        self.task_steps = 0
        self.task_w_old_sum = 0.0
        self.task_p_stable_sum = 0.0
        self.task_beta_sum = None
        self.task_beta_count = 0

        seed = getattr(self.args, "seed", 0)
        self.stats_file = f"er_alpha_beta_v6_stats_seed{seed}.txt"

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
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

    def forward(self, x):
        return self.net(x)

    # ------------------------------------------------------------------
    # 构造 ctx + classifier head 的梯度信息（V5 风格）
    # ------------------------------------------------------------------
    def _build_ctx_and_grads(self, loss_new, loss_old,
                             features_new, features_old):
        """
        返回：
          ctx_vec: (1,8)
          mu_new, mu_old: (1,F)
          norm_old, norm_new, cos_theta: scalar (detached)
          l_old_val, l_new_val: float
        """
        l_new_val = float(loss_new.item())
        l_old_val = float(loss_old.item())
        denom_loss = l_old_val + l_new_val + 1e-8

        head_params = list(self.net.classifier.parameters())
        g_old = torch.autograd.grad(
            loss_old, head_params,
            retain_graph=True, allow_unused=True
        )
        g_new = torch.autograd.grad(
            loss_new, head_params,
            retain_graph=True, allow_unused=True
        )

        g_old_flat = torch.cat([g.view(-1) for g in g_old if g is not None])
        g_new_flat = torch.cat([g.view(-1) for g in g_new if g is not None])

        norm_old = g_old_flat.norm() + 1e-8
        norm_new = g_new_flat.norm() + 1e-8
        cos_theta = (g_old_flat @ g_new_flat) / (norm_old * norm_new)
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        norm_old_scaled = torch.log(norm_old) / 5.0
        norm_new_scaled = torch.log(norm_new) / 5.0

        total_tasks = getattr(self.args, "total_tasks", 5) or 5
        t_norm = self.current_task_id / max(total_tasks - 1, 1)

        ctx_vec = torch.tensor(
            [
                l_old_val / denom_loss,          # 0
                l_new_val / denom_loss,          # 1
                l_new_val - l_old_val,           # 2
                denom_loss,                      # 3
                t_norm,                          # 4
                float(cos_theta.item()),         # 5
                float(norm_old_scaled.item()),   # 6
                float(norm_new_scaled.item()),   # 7
            ],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        mu_new = features_new.mean(dim=0, keepdim=True).detach()
        mu_old = features_old.mean(dim=0, keepdim=True).detach()

        return (
            ctx_vec,
            mu_new,
            mu_old,
            norm_old.detach(),
            norm_new.detach(),
            cos_theta.detach(),
            l_old_val,
            l_new_val,
        )

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def end_task(self, dataset):
        if self.task_steps > 0:
            avg_w_old = self.task_w_old_sum / self.task_steps
            avg_p_stable = self.task_p_stable_sum / self.task_steps

            msg = (
                f"[AlphaBetaV6][Task {self.current_task_id + 1}] "
                f"Avg w_old={avg_w_old:.4f} | "
                f"Avg p_stable={avg_p_stable:.4f}"
            )

            if self.task_beta_count > 0 and self.task_beta_sum is not None:
                beta_mean = (self.task_beta_sum / self.task_beta_count).cpu()
                beta_global_mean = beta_mean.mean().item()
                beta_head = ", ".join(
                    [f"{v:.3f}" for v in beta_mean[:5].tolist()]
                )
                msg += (
                    f" | Beta_mean_global={beta_global_mean:.4f} "
                    f"| Beta_head={beta_head}"
                )

            print(msg)
            try:
                with open(self.stats_file, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

        self.task_steps = 0
        self.task_w_old_sum = 0.0
        self.task_p_stable_sum = 0.0
        self.task_beta_sum = None
        self.task_beta_count = 0

        self.current_task_id += 1
        super().end_task(dataset)

    # ------------------------------------------------------------------
    # main observe
    # ------------------------------------------------------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # ===== Task1 warmup：完全不用 alpha/beta，纯 ER =====
        if self.current_task_id == 0:
            self.opt.zero_grad()
            self.net.train()

            feats_new = self._get_features(inputs)
            out_new = self.net.classifier(feats_new)
            loss_new = self.loss(out_new, labels)

            if self.buffer.is_empty():
                loss_new.backward()
                self.opt.step()
                self.buffer.add_data(examples=not_aug_inputs, labels=labels)
                return float(loss_new.item())

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            feats_old = self._get_features(buf_inputs)
            out_old = self.net.classifier(feats_old)
            loss_old = self.loss(out_old, buf_labels)

            loss = 0.5 * (loss_new + loss_old)
            loss.backward()
            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)
            return float(loss.item())

        # ===== Task >= 2：启用 AlphaBeta 策略 =====
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        feats_new = self._get_features(inputs)
        out_new = self.net.classifier(feats_new)
        loss_new = self.loss(out_new, labels)

        if self.buffer.is_empty():
            loss_new.backward()
            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)
            return float(loss_new.item())

        # buffer batch
        buf_inputs, buf_labels = self.buffer.get_data(
            self.args.minibatch_size,
            transform=self.transform,
            device=self.device,
        )
        feats_old = self._get_features(buf_inputs)
        out_old = self.net.classifier(feats_old)
        loss_old = self.loss(out_old, buf_labels)

        # 1) ctx + stats（对 net 不求 grad）
        with torch.no_grad():
            (
                ctx_vec,
                mu_new,
                mu_old,
                norm_old,
                norm_new,
                cos_theta,
                l_old_val,
                l_new_val,
            ) = self._build_ctx_and_grads(loss_new, loss_old, feats_new, feats_old)

            # ctx EMA
            if self.ctx_ema is None:
                self.ctx_ema = ctx_vec
            else:
                beta_ema = getattr(self.args, "ctx_ema_beta", 0.9)
                self.ctx_ema = beta_ema * self.ctx_ema + (1.0 - beta_ema) * ctx_vec

            ctx_used = self.ctx_ema

            # controller（theta-step 不给 controller 求 grad）
            w_old, w_new, p_stable, beta_profile = self.controller(
                ctx_used, mu_old, mu_new
            )

            w_old_val = float(w_old.item())
            w_new_val = float(w_new.item())
            p_stable_val = float(p_stable.item())
            beta_profile_det = beta_profile.detach()

        # 2) 计算 new / old 的完整梯度
        params = [p for p in self.net.parameters() if p.requires_grad]
        g_new_all = torch.autograd.grad(
            loss_new, params, retain_graph=True, allow_unused=True
        )
        g_old_all = torch.autograd.grad(
            loss_old, params, retain_graph=False, allow_unused=True
        )

        # 映射 param -> layer_idx
        layer_groups = self._get_layer_groups()
        param_to_layer = {}
        for layer_idx, layer_name in enumerate(self.layer_names):
            for p in layer_groups[layer_name]:
                param_to_layer[id(p)] = layer_idx

        L = beta_profile_det.numel()

        # 3) 按层组合梯度
        for p, g_n, g_o in zip(params, g_new_all, g_old_all):
            if (g_n is None) and (g_o is None):
                continue
            if g_n is None:
                g_n = torch.zeros_like(g_o)
            if g_o is None:
                g_o = torch.zeros_like(g_n)

            layer_idx = param_to_layer.get(id(p), None)
            if (layer_idx is None) or (layer_idx >= L):
                beta_l = 0.5
            else:
                beta_l = float(beta_profile_det[layer_idx].item())

            # 关键：β = 稳定度 → β 越大越偏 old-grad
            g_final = beta_l * w_old_val * g_o + (1.0 - beta_l) * w_new_val * g_n
            p.grad = g_final

        self.opt.step()

        # buffer 里只需要 (x,y)
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        # 纯标量 loss（用于日志）
        reg_strength = getattr(self.args, "w_reg_strength", 0.01)
        loss_scalar = (
            w_old_val * l_old_val
            + w_new_val * l_new_val
            + reg_strength * (p_stable_val - 0.5) ** 2
        )

        # 记录统计
        self.task_steps += 1
        self.task_w_old_sum += w_old_val
        self.task_p_stable_sum += p_stable_val
        if self.task_beta_sum is None:
            self.task_beta_sum = beta_profile_det.clone()
        else:
            self.task_beta_sum += beta_profile_det
        self.task_beta_count += 1

        # meta 更新
        if (self.global_step % self.args.meta_interval == 0) and (
            not self.buffer.is_empty()
        ):
            self._meta_update(inputs, labels)

        return float(loss_scalar)

    # ------------------------------------------------------------------
    # meta-update for controller
    # ------------------------------------------------------------------
    def _meta_update(self, new_inputs, new_labels):
        if self.buffer.is_empty():
            return

        self.net.eval()
        self.controller.train()
        self.opt_cont.zero_grad()

        new_inputs = new_inputs.to(self.device)
        new_labels = new_labels.to(self.device)

        # new batch
        feats_new = self._get_features(new_inputs)
        out_new = self.net.classifier(feats_new)
        loss_new = self.loss(out_new, new_labels)

        # old batch
        buf_inputs, buf_labels = self.buffer.get_data(
            self.args.minibatch_size,
            transform=self.transform,
            device=self.device,
        )
        feats_old = self._get_features(buf_inputs)
        out_old = self.net.classifier(feats_old)
        loss_old = self.loss(out_old, buf_labels)

        with torch.no_grad():
            (
                ctx_vec,
                mu_new,
                mu_old,
                norm_old,
                norm_new,
                cos_theta,
                l_old_val,
                l_new_val,
            ) = self._build_ctx_and_grads(loss_new, loss_old, feats_new, feats_old)

        # controller（这一步要对 controller 求 grad）
        w_old_meta, w_new_meta, p_stable_meta, beta_profile_meta = self.controller(
            ctx_vec, mu_old, mu_new
        )

        # 1) alpha 的 grad-balance surrogate
        prod_old = w_old_meta * norm_old
        prod_new = w_new_meta * norm_new
        margin = 0.0
        ratio_term = torch.relu(prod_new - prod_old + margin)
        grad_balance = ratio_term ** 2

        # 2) p_stable 正则
        reg_strength = getattr(self.args, "w_reg_strength", 0.01)
        p_reg = reg_strength * (p_stable_meta - 0.5) ** 2

        # 3) beta 的 meta loss（目标是 old-grad 比例）
        params = [p for p in self.net.parameters() if p.requires_grad]
        g_new_all = torch.autograd.grad(
            loss_new, params, retain_graph=True, allow_unused=True
        )
        g_old_all = torch.autograd.grad(
            loss_old, params, retain_graph=False, allow_unused=True
        )

        param_to_idx = {id(p): i for i, p in enumerate(params)}
        layer_groups = self._get_layer_groups()

        beta_targets = []
        for layer_idx, layer_name in enumerate(self.layer_names):
            p_list = layer_groups[layer_name]
            if not p_list:
                beta_targets.append(0.5)
                continue

            g_new_norm_sq = 0.0
            g_old_norm_sq = 0.0
            for p in p_list:
                idx = param_to_idx.get(id(p), None)
                if idx is None:
                    continue
                g_n = g_new_all[idx]
                g_o = g_old_all[idx]
                if g_n is not None:
                    val = g_n.detach().norm().item()
                    g_new_norm_sq += val * val
                if g_o is not None:
                    val = g_o.detach().norm().item()
                    g_old_norm_sq += val * val

            denom = g_new_norm_sq + g_old_norm_sq + 1e-8
            beta_star = g_old_norm_sq / denom    # old-grad 比例 = 稳定度
            beta_targets.append(beta_star)

        if beta_targets:
            beta_targets_t = torch.tensor(
                beta_targets,
                device=self.device,
                dtype=beta_profile_meta.dtype,
            )
            L = min(beta_profile_meta.numel(), beta_targets_t.numel())
            beta_loss = F.mse_loss(beta_profile_meta[:L], beta_targets_t[:L])
        else:
            beta_loss = torch.tensor(0.0, device=self.device)

        lambda_beta = getattr(self.args, "beta_meta_coef", 1.0)
        meta_coef = getattr(self.args, "meta_grad_balance_coef", 0.5)

        meta_loss = meta_coef * grad_balance + p_reg + lambda_beta * beta_loss

        meta_loss.backward()
        self.opt_cont.step()
