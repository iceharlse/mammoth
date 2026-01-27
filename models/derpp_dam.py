import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


# ----------------------------------------------------------------------
# 小 MLP 控制器：输入 8 维 ctx（V5 风格） -> 输出 Δα_raw, Δβ_raw ∈ [-1, 1]
# ----------------------------------------------------------------------
class AlphaBetaController(nn.Module):
    def __init__(self, ctx_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),  # 限制在 [-1, 1]
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: [B, ctx_dim]，这里 B=1
        return self.net(ctx)


@register_model("derpp_dam")
class DerppDamAlphaBetaConstrained(ContinualModel):
    """
    DER++ + α/β 控制器 + β-grad-health 医生 + 显式 DER++ 梯度锥形约束，
    并把 V5 的梯度几何特征 (cosθ, log||g||) 塞回 ctx。
    
    机制：
    - 目标向量: (L_new, L_old)，其中 L_old ≈ beta * CE + alpha * MSE。
    - Anchor 梯度: g_anchor = ∇ L_anchor（标准 DER++ 配方）：
          L_anchor = L_new + beta0 * L_buf_ce + alpha0 * L_buf_mse
    - DAM 插件产生:
          L_total = L_new + beta_eff * L_buf_ce + alpha_eff * L_buf_mse + reg
          g_total = ∇ L_total
    
    约束：
    - 角度: cos(g_total, g_anchor) >= dam_cone_cos
    - 范数: ||g_safe|| <= (1 + dam_cone_norm_eps) * ||g_anchor||
    
    控制：
    - α: 调整 alpha_eff（distill MSE 权重），global trade-off。
    - β: 一方面调整 beta_eff（replay CE 权重），另一方面调节 β-doctor 强度
         (c_eff, rho_eff)，实现 per-layer 的 hierarchical 控制。
    """

    NAME = "derpp_dam"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    # ------------------------ parser ------------------------ #
    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument(
            "--alpha", type=float, required=True,
            help="DER++ distillation (MSE) weight."
        )
        parser.add_argument(
            "--beta", type=float, required=True,
            help="DER++ replay (CE) weight."
        )

        # α/β 控制器的超参
        parser.add_argument(
            "--dam_alpha_gain", type=float, default=0.0,
            help="Max relative deviation for alpha: "
                 "alpha_eff in [(1-gain)*alpha, (1+gain)*alpha]."
        )
        parser.add_argument(
            "--dam_beta_gain", type=float, default=0.0,
            help="Max relative deviation for beta: "
                 "beta_eff in [(1-gain)*beta, (1+gain)*beta], "
                 "and also scales beta-doctor strength."
        )
        parser.add_argument(
            "--dam_hidden", type=int, default=32,
            help="Hidden dim of alpha/beta controller."
        )
        parser.add_argument(
            "--dam_reg", type=float, default=1e-3,
            help="L2 regularization on controller output."
        )

        # β-grad-health 医生的开关和超参
        parser.add_argument(
            "--dam_beta_health", type=int, default=0,
            help="If >0, enable beta grad-health doctor."
        )
        parser.add_argument(
            "--dam_beta_eta", type=float, default=0.01,
            help="EMA update rate for grad norms."
        )
        parser.add_argument(
            "--dam_beta_c", type=float, default=0.1,
            help="Base strength of grad health correction."
        )
        parser.add_argument(
            "--dam_beta_rho", type=float, default=0.2,
            help="Base max grad scaling range [1-rho, 1+rho]."
        )

        # 梯度锥形约束的超参
        parser.add_argument(
            "--dam_cone_cos", type=float, default=0.0,
            help="Minimum cosine between g_total and g_anchor. "
                 "0.0 ≈ 90°, >0 means stronger alignment."
        )
        parser.add_argument(
            "--dam_cone_norm_eps", type=float, default=0.3,
            help="Max relative norm increase: "
                 "||g_safe|| <= (1+eps)*||g_anchor||."
        )

        # 用于 task_progress 的任务数（可选）
        parser.add_argument(
            "--num_tasks", type=int, default=10,
            help="Number of tasks for task_progress in context."
        )

        return parser

    # ------------------------ init ------------------------ #
    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # DER++ 风格的 Buffer
        self.buffer = Buffer(self.args.buffer_size)

        # ctx 维度 = 8（V5 风格）：
        # 0: frac_o, 1: frac_n, 2: log_l_n - log_l_o, 3: log_l_n + log_l_o,
        # 4: task_progress, 5: cosθ, 6: log||g_old||/5, 7: log||g_new||/5
        self.ctx_dim = 8
        self.controller = AlphaBetaController(
            ctx_dim=self.ctx_dim,
            hidden_dim=self.args.dam_hidden,
        ).to(self.device)

        # 控制器优化器（单独）
        self.opt_ctrl = torch.optim.Adam(
            self.controller.parameters(),
            lr=self.args.lr,
            weight_decay=1e-5,
        )

        # β-grad-health：每层的 grad_norm EMA
        self.grad_ema = {}  # name -> tensor scalar

    # ------------------------ helper: head grad stats ------------------------ #
    def _compute_head_grad_stats(self, loss_new, loss_buf_ce):
        """
        在 classifier head 上计算：
          - cosθ(g_old, g_new)
          - log||g_old|| / 5
          - log||g_new|| / 5
        只作为 ctx 特征，不参与反传（detach）。
        """
        # 优先用 net.classifier
        if hasattr(self.net, "classifier"):
            head_params = list(self.net.classifier.parameters())
        else:
            # fallback: 名字里带 'fc' 的参数（基本是最后一层）
            head_params = [
                p for n, p in self.net.named_parameters()
                if "fc" in n and p.requires_grad
            ]

        if not head_params:
            return 0.0, 0.0, 0.0

        # g_old: buffer CE 的梯度
        g_old = torch.autograd.grad(
            loss_buf_ce,
            head_params,
            retain_graph=True,
            allow_unused=True,
        )
        # g_new: 当前 batch CE 的梯度
        g_new = torch.autograd.grad(
            loss_new,
            head_params,
            retain_graph=True,
            allow_unused=True,
        )

        g_old_flat = torch.cat([g.view(-1) for g in g_old if g is not None])
        g_new_flat = torch.cat([g.view(-1) for g in g_new if g is not None])

        if g_old_flat.numel() == 0 or g_new_flat.numel() == 0:
            return 0.0, 0.0, 0.0

        norm_old = g_old_flat.norm() + 1e-8
        norm_new = g_new_flat.norm() + 1e-8
        cos_theta = (g_old_flat @ g_new_flat) / (norm_old * norm_new)
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        log_old = torch.log(norm_old) / 5.0
        log_new = torch.log(norm_new) / 5.0

        return (
            float(cos_theta.detach().item()),
            float(log_old.detach().item()),
            float(log_new.detach().item()),
        )

    # ------------------------ helper: context (V5 ctx8) ------------------------ #
    def _build_context(
        self,
        loss_new: torch.Tensor,
        loss_buf_ce: torch.Tensor,
        loss_buf_mse: torch.Tensor,
        cos_theta: float,
        log_g_old: float,
        log_g_new: float,
    ) -> torch.Tensor:
        """
        构造 [1, 8] 的 ctx：
          0: frac_o (旧 loss 比例, log 归一化)
          1: frac_n (新 loss 比例, log 归一化)
          2: log_l_n - log_l_o
          3: log_l_n + log_l_o
          4: task_progress
          5: cosθ(g_old, g_new)
          6: log||g_old||/5
          7: log||g_new||/5
        其中 l_old 用 anchor 的 alpha0/beta0 加权。
        """
        alpha0 = float(self.args.alpha)
        beta0 = float(self.args.beta)

        l_new = float(loss_new.detach().item())
        l_old = float(
            (beta0 * loss_buf_ce + alpha0 * loss_buf_mse).detach().item()
        )

        log_l_n = math.log1p(max(l_new, 0.0))
        log_l_o = math.log1p(max(l_old, 0.0))
        denom = log_l_n + log_l_o + 1e-8

        frac_o = log_l_o / denom
        frac_n = log_l_n / denom

        # task progress: [0, 1]
        num_tasks = float(max(getattr(self.args, "num_tasks", 10), 1))
        cur_task = getattr(self, "current_task", 0)
        try:
            task_prog = float(cur_task) / max(num_tasks - 1.0, 1.0)
        except Exception:
            task_prog = 0.0

        base_ctx = torch.tensor(
            [
                frac_o,
                frac_n,
                log_l_n - log_l_o,
                log_l_n + log_l_o,
                task_prog,
                float(cos_theta),
                float(log_g_old),
                float(log_g_new),
            ],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 8]

        return base_ctx

    # ------------------------ helper: β-grad-health ------------------------ #
    def _beta_grad_health_update(self, c_eff: float, rho_eff: float):
        """
        在我们手工设置好 .grad（已经做完锥形约束）之后调用。

        对每个参数的梯度按 grad_norm / ema_grad 的比值做一个小的缩放：
            r = g / ema
            scale = clamp(1 + c_eff*(1 - r), 1-rho_eff, 1+rho_eff)
        """
        if getattr(self.args, "dam_beta_health", 0) <= 0:
            return

        eta = float(getattr(self.args, "dam_beta_eta", 0.01))
        c = float(c_eff)
        rho = float(max(rho_eff, 0.0))
        eps = 1e-8

        for name, p in self.net.named_parameters():
            if p.grad is None:
                continue

            g = p.grad.data.norm(2)
            if not torch.isfinite(g):
                continue

            # 初始化 EMA
            if name not in self.grad_ema:
                self.grad_ema[name] = g.detach()
                continue

            ema = self.grad_ema[name]
            new_ema = (1.0 - eta) * ema + eta * g
            self.grad_ema[name] = new_ema.detach()

            # 健康比例
            r = g / (new_ema + eps)

            # 变成缩放系数：r>1 -> scale<1；r<1 -> scale>1
            scale = 1.0 + c * (1.0 - r)
            min_s = 1.0 - rho
            max_s = 1.0 + rho
            scale = torch.clamp(scale, min=min_s, max=max_s)

            p.grad.data.mul_(scale)

    # ------------------------ core: observe ------------------------ #
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        和原版 DER++ 的差异：

        - 多了 controller 产出 alpha_eff/beta_eff（小区间内动）。
        - 多了 β-grad-health，在锥形约束之后微调各层梯度大小。
        - 最关键：在 backbone 的梯度上显式施加了 DER++ anchor 的锥形约束。
        - ctx 使用 V5 风格的 8 维特征（包括 cosθ 和 log||g||）。
        """
        self.net.train()
        self.controller.train()

        self.opt.zero_grad()
        self.opt_ctrl.zero_grad()

        # 收集需要求梯度的参数列表
        net_params = [p for p in self.net.parameters() if p.requires_grad]
        ctrl_params = [p for p in self.controller.parameters() if p.requires_grad]

        # ----------------------
        # 1. 当前 batch：new loss
        # ----------------------
        outputs = self.net(inputs)
        loss_new = self.loss(outputs, labels)

        # buffer 为空：退化成 ER
        if self.buffer.is_empty():
            loss_new.backward()

            grad_clip = getattr(self.args, "grad_clip", 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)

            self.opt.step()

            self.buffer.add_data(
                examples=not_aug_inputs,
                labels=labels,
                logits=outputs.data,
            )
            return loss_new.item()

        # ----------------------
        # 2. MSE distill buffer
        # ----------------------
        buf_inputs_mse, _, buf_logits = self.buffer.get_data(
            self.args.minibatch_size,
            transform=self.transform,
            device=self.device,
        )
        buf_outputs_mse = self.net(buf_inputs_mse)
        loss_buf_mse_raw = F.mse_loss(buf_outputs_mse, buf_logits)

        # ----------------------
        # 3. CE replay buffer
        # ----------------------
        buf_inputs_ce, buf_labels, _ = self.buffer.get_data(
            self.args.minibatch_size,
            transform=self.transform,
            device=self.device,
        )
        buf_outputs_ce = self.net(buf_inputs_ce)
        loss_buf_ce_raw = self.loss(buf_outputs_ce, buf_labels)

        # ----------------------
        # 4. 先算 head 上的梯度几何特征 -> V5 ctx 部分
        # ----------------------
        cos_t, log_go, log_gn = self._compute_head_grad_stats(
            loss_new, loss_buf_ce_raw
        )

        # ----------------------
        # 5. 构造 ctx -> controller -> α_eff / β_eff
        # ----------------------
        with torch.no_grad():
            ctx = self._build_context(
                loss_new,
                loss_buf_ce_raw,
                loss_buf_mse_raw,
                cos_t,
                log_go,
                log_gn,
            )
        delta = self.controller(ctx).squeeze(0)   # [2]
        delta_alpha_raw = delta[0]
        delta_beta_raw = delta[1]

        alpha0 = self.args.alpha
        beta0  = self.args.beta
        gain_a = self.args.dam_alpha_gain
        gain_b = self.args.dam_beta_gain

        # 允许区间：[(1-gain), (1+gain)] * base
        alpha_min = alpha0 * (1.0 - gain_a)
        alpha_max = alpha0 * (1.0 + gain_a)
        beta_min  = beta0  * (1.0 - gain_b)
        beta_max  = beta0  * (1.0 + gain_b)

        alpha_eff = alpha0 * (1.0 + gain_a * delta_alpha_raw)
        beta_eff  = beta0  * (1.0 + gain_b * delta_beta_raw)

        alpha_eff = torch.clamp(alpha_eff, min=alpha_min, max=alpha_max)
        beta_eff  = torch.clamp(beta_eff,  min=beta_min,  max=beta_max)

        # β-controller 同时调节 β-doctor 强度
        beta_factor = 1.0 + gain_b * delta_beta_raw
        # 确保在 [1-gain_b, 1+gain_b] 内，且不会变成负数
        low = 1.0 - gain_b
        high = 1.0 + gain_b
        beta_factor = torch.clamp(beta_factor, min=low, max=high)
        beta_factor_f = float(beta_factor.detach().item())

        c0   = float(getattr(self.args, "dam_beta_c", 0.1))
        rho0 = float(getattr(self.args, "dam_beta_rho", 0.2))
        c_eff   = c0 * beta_factor_f
        rho_eff = rho0 * beta_factor_f

        # ----------------------
        # 6. 定义 anchor loss & total loss
        # ----------------------
        loss_anchor = (
            loss_new
            + beta0  * loss_buf_ce_raw
            + alpha0 * loss_buf_mse_raw
        )

        loss_total = (
            loss_new
            + beta_eff  * loss_buf_ce_raw
            + alpha_eff * loss_buf_mse_raw
        )

        # controller 输出的 L2 正则
        if self.args.dam_reg > 0.0:
            reg = (delta_alpha_raw ** 2 + delta_beta_raw ** 2)
            loss_total = loss_total + self.args.dam_reg * reg

        # ----------------------
        # 7. 用 autograd.grad 计算 g_anchor, g_total
        # ----------------------
        g_anchor = torch.autograd.grad(
            loss_anchor,
            net_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        g_total = torch.autograd.grad(
            loss_total,
            net_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        # 替换 None -> 0
        g_anchor_list = []
        g_total_list = []
        for p, ga, gt in zip(net_params, g_anchor, g_total):
            if ga is None:
                ga = torch.zeros_like(p)
            if gt is None:
                gt = torch.zeros_like(p)
            g_anchor_list.append(ga)
            g_total_list.append(gt)

        # 拼成向量
        flat_anchor = torch.cat([ga.view(-1) for ga in g_anchor_list])
        flat_total  = torch.cat([gt.view(-1) for gt in g_total_list])

        eps = 1e-8
        norm_a = flat_anchor.norm() + eps
        norm_t = flat_total.norm() + eps

        # 如果 anchor 梯度几乎为 0，就直接用 g_total
        if norm_a < 1e-12:
            flat_safe = flat_total.clone()
        else:
            cos_theta = torch.dot(flat_total, flat_anchor) / (norm_t * norm_a)
            cos_min = float(self.args.dam_cone_cos)

            # 角度约束：若 cos < cos_min，则只保留沿 anchor 的分量
            if cos_theta.item() < cos_min:
                proj_coeff = torch.dot(flat_total, flat_anchor) / (norm_a * norm_a)
                flat_safe = proj_coeff * flat_anchor
            else:
                flat_safe = flat_total.clone()

            # 范数约束：||g_safe|| <= (1+eps)*||g_anchor||
            norm_safe = flat_safe.norm()
            max_norm = (1.0 + float(self.args.dam_cone_norm_eps)) * norm_a
            if norm_safe > max_norm:
                flat_safe = flat_safe * (max_norm / (norm_safe + eps))

        # ----------------------
        # 8. 把 flat_safe 分配回各个参数 .grad
        # ----------------------
        offset = 0
        for p in net_params:
            numel = p.numel()
            g_chunk = flat_safe[offset : offset + numel].view_as(p)
            p.grad = g_chunk.detach().clone()
            offset += numel

        # ----------------------
        # 9. β-grad-health 医生（在锥形约束之后）
        # ----------------------
        self._beta_grad_health_update(c_eff=c_eff, rho_eff=rho_eff)

        # ----------------------
        # 10. controller 的梯度：对 controller 参数做 autograd.grad
        # ----------------------
        if ctrl_params:
            g_ctrl = torch.autograd.grad(
                loss_total,
                ctrl_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            for p, g in zip(ctrl_params, g_ctrl):
                if g is None:
                    continue
                p.grad = g.detach().clone()

        # ----------------------
        # 11. grad_clip + step
        # ----------------------
        grad_clip = getattr(self.args, "grad_clip", 0.0)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), grad_clip)

        self.opt.step()
        self.opt_ctrl.step()

        # ----------------------
        # 12. 更新 buffer（和 derpp 一致）
        # ----------------------
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
            logits=outputs.data,
        )

        return loss_total.item()
