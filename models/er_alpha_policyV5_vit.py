import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model
from models.er_alpha_policyV5 import AlphaControllerV5


@register_model("derpp-alpha-policyV5")
class DerppAlphaPolicyV5(ContinualModel):
    """
    DER++ + Alpha-policy V5（更稳的一版整合）

    原始 DER++:
        L_new      = CE(f(x_new), y_new)
        L_ce_old   = CE(f(x_buf), y_buf)
        L_mse      = MSE(f(x_buf), logits_buf)

        L_old_grp  = beta * L_ce_old + alpha * L_mse
        L_derpp    = L_new + L_old_grp

    现在只多一个可学习的标量 lambda ∈ [alpha_min, alpha_max]：
        L = L_new + lambda * L_old_grp

    - use_alpha_policy = 0 或 alpha_min = alpha_max = 1 时，
      完全退化为原版 DER++。
    """
    NAME = "derpp-alpha-policyV5"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    # ------------------------------------------------------------------
    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)

        # DER++ 权重
        parser.add_argument("--alpha", type=float, required=True,
                            help="DER++ MSE distillation weight.")
        parser.add_argument("--beta", type=float, required=True,
                            help="DER++ replay CE weight.")

        # 是否启用 alpha policy
        parser.add_argument("--use_alpha_policy", type=int, default=1,
                            help="1: 启用 V5 controller; 0: 纯 DER++。")

        # controller 结构
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        parser.add_argument("--total_tasks", type=int, default=5)

        # meta / 正则
        parser.add_argument("--w_reg_strength", type=float, default=0.05,
                            help="对 lambda 和 p_stable 的正则强度。")
        parser.add_argument("--meta_lr", type=float, default=1e-3,
                            help="alpha controller 的学习率。")
        parser.add_argument("--meta_interval", type=int, default=50,
                            help="如果 meta_interval_examples == 0，则每多少 step 更新一次 controller。")
        parser.add_argument("--meta_grad_balance_coef", type=float, default=0.5,
                            help="meta loss 中 grad-balance 项系数。")

        # ctx 平滑 & lambda 范围（围绕 1.0）
        parser.add_argument("--ctx_ema_beta", type=float, default=0.9,
                            help="上下文 EMA 系数 (0 = 不用 EMA)。")
        parser.add_argument("--alpha_min", type=float, default=0.8,
                            help="old-group loss 的最小 lambda。")
        parser.add_argument("--alpha_max", type=float, default=1.2,
                            help="old-group loss 的最大 lambda。")

        # 按 sample 数触发 meta-update
        parser.add_argument("--meta_interval_examples", type=int, default=0,
                            help=">0 时：每看到 N 个样本更新一次 controller。")

        parser.add_argument("--log_interval", type=int, default=100,
                            help="多少 step 打一次 log（这里只简单存一下步骤信息，方便你以后扩展）。")

        return parser

    # ------------------------------------------------------------------
    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)

        # 给 controller 用的特征维度
        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512  # resnet18 一般是 512

        # 直接复用你 ER 里的 AlphaControllerV5
        self.controller = AlphaControllerV5(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            nhead=self.args.dam_nhead,
            num_layers=self.args.dam_layers,
            alpha_min=self.args.alpha_min,
            alpha_max=self.args.alpha_max,
        ).to(self.device)

        self.opt_cont = torch.optim.Adam(
            self.controller.parameters(),
            lr=self.args.meta_lr,
            weight_decay=1e-5,
        )

        # 记步数 & task id
        self.current_task_id = 0
        self.global_step = 0

        self.meta_interval_examples = getattr(self.args, "meta_interval_examples", 0)
        self.meta_token_examples = 0

        self.ctx_ema = None
        self.ctx_ema_beta = getattr(self.args, "ctx_ema_beta", 0.9)

        self.log_interval = getattr(self.args, "log_interval", 100)

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.net(x)

    def _get_features(self, x):
        # Mammoth 风格：backbone(x, returnt="features")
        return self.net(x, returnt="features")

    def _get_head_params(self):
        if hasattr(self.net, "classifier"):
            params = [p for p in self.net.classifier.parameters()
                      if p.requires_grad]
            if len(params) > 0:
                return params
        return [p for p in self.net.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # 构造 8 维 ctx + 梯度统计
    # ------------------------------------------------------------------
    def _build_ctx_and_grads(self, loss_new, loss_old,
                             features_new, features_old):
        l_new_val = float(loss_new.item())
        l_old_val = float(loss_old.item())
        denom_loss = l_old_val + l_new_val + 1e-8

        head_params = self._get_head_params()
        g_old = torch.autograd.grad(
            loss_old, head_params,
            retain_graph=True, allow_unused=True
        )
        g_new = torch.autograd.grad(
            loss_new, head_params,
            retain_graph=True, allow_unused=True
        )

        def _flat(gs):
            return torch.cat([g.view(-1) for g in gs if g is not None])

        g_old_flat = _flat(g_old)
        g_new_flat = _flat(g_new)

        norm_old = g_old_flat.norm(p=2) + 1e-8
        norm_new = g_new_flat.norm(p=2) + 1e-8
        cos_theta = (g_old_flat @ g_new_flat) / (norm_old * norm_new)
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        norm_old_scaled = torch.log(norm_old) / 5.0
        norm_new_scaled = torch.log(norm_new) / 5.0

        total_tasks = getattr(self.args, "total_tasks", 5) or 5
        t_norm = self.current_task_id / max(total_tasks - 1, 1)

        ctx_vec = torch.tensor(
            [
                l_old_val / denom_loss,          # 0: l_old_n
                l_new_val / denom_loss,          # 1: l_new_n
                l_new_val - l_old_val,           # 2: diff
                denom_loss,                      # 3: sum
                t_norm,                          # 4: t_norm
                float(cos_theta.item()),         # 5: cosθ
                float(norm_old_scaled.item()),   # 6: log||g_old||
                float(norm_new_scaled.item()),   # 7: log||g_new||
            ],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, 8)

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
    def end_task(self, dataset):
        self.current_task_id += 1
        super().end_task(dataset)

    # ------------------------------------------------------------------
    # 主训练循环：theta-step + phi-step
    # ------------------------------------------------------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        batch_size = inputs.size(0)

        if self.meta_interval_examples > 0:
            self.meta_token_examples += batch_size

        # ================== 1) theta-step：更新 backbone ==================
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        outputs = self.net(inputs)
        loss_new = self.loss(outputs, labels)

        loss = loss_new

        if not self.buffer.is_empty():
            # 与 DER++ 一致：同一批 buffer 用于 CE & MSE
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )

            buf_outputs = self.net(buf_inputs)
            loss_ce = self.loss(buf_outputs, buf_labels)
            loss_mse = F.mse_loss(buf_outputs, buf_logits)

            loss_old_grp = (
                self.args.beta * loss_ce
                + self.args.alpha * loss_mse
            )

            if getattr(self.args, "use_alpha_policy", 1) == 1:
                # 构造 ctx
                feats_new = self._get_features(inputs)
                feats_old = self._get_features(buf_inputs)

                (
                    ctx_vec,
                    mu_new,
                    mu_old,
                    norm_old,
                    norm_new,
                    cos_theta,
                    l_old_val,
                    l_new_val,
                ) = self._build_ctx_and_grads(
                    loss_new, loss_old_grp, feats_new, feats_old
                )

                # EMA 平滑
                if self.ctx_ema_beta > 0.0:
                    if self.ctx_ema is None:
                        self.ctx_ema = ctx_vec.detach()
                    else:
                        beta_ema = self.ctx_ema_beta
                        self.ctx_ema = (
                            beta_ema * self.ctx_ema
                            + (1.0 - beta_ema) * ctx_vec.detach()
                        )
                    ctx_used = self.ctx_ema
                else:
                    ctx_used = ctx_vec

                # θ-step 中 query controller，不反传到 controller
                with torch.no_grad():
                    w_old, w_new_unused, p_stable = self.controller(
                        ctx_used, mu_old, mu_new
                    )

                lambda_old = w_old  # 直接当成 lambda

                loss_main = loss_new + lambda_old * loss_old_grp

                w_reg_strength = getattr(self.args, "w_reg_strength", 0.05)
                reg_lambda = w_reg_strength * (lambda_old - 1.0) ** 2
                reg_p = w_reg_strength * (p_stable - 0.5) ** 2

                loss = loss_main + reg_lambda + reg_p
            else:
                # 纯 DER++
                loss = loss_new + loss_old_grp

        loss.backward()
        self.opt.step()

        # buffer 存 logits：保证还是 DER++ 的蒸馏形式
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
            logits=outputs.detach(),
        )

        # ================== 2) phi-step：更新 controller ==================
        trigger_meta = False
        if self.meta_interval_examples > 0:
            if self.meta_token_examples >= self.meta_interval_examples:
                trigger_meta = True
                self.meta_token_examples = 0
        else:
            if (self.global_step % self.args.meta_interval) == 0:
                trigger_meta = True

        if (getattr(self.args, "use_alpha_policy", 1) == 1
                and trigger_meta and (not self.buffer.is_empty())):
            self.net.eval()
            self.controller.train()
            self.opt_cont.zero_grad()

            # new batch
            meta_inputs = inputs
            meta_labels = labels
            meta_outputs = self.net(meta_inputs)
            meta_loss_new = self.loss(meta_outputs, meta_labels)

            # old-group batch
            m_buf_inputs, m_buf_labels, m_buf_logits = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            m_buf_outputs = self.net(m_buf_inputs)
            m_loss_ce = self.loss(m_buf_outputs, m_buf_labels)
            m_loss_mse = F.mse_loss(m_buf_outputs, m_buf_logits)
            m_loss_old_grp = (
                self.args.beta * m_loss_ce
                + self.args.alpha * m_loss_mse
            )

            m_feats_new = self._get_features(meta_inputs)
            m_feats_old = self._get_features(m_buf_inputs)

            (
                ctx_meta,
                mu_new_meta,
                mu_old_meta,
                norm_old_meta,
                norm_new_meta,
                cos_meta,
                l_old_meta,
                l_new_meta,
            ) = self._build_ctx_and_grads(
                meta_loss_new, m_loss_old_grp,
                m_feats_new, m_feats_old
            )

            head_params = self._get_head_params()

            def _grad_norm(loss_scalar):
                g = torch.autograd.grad(
                    loss_scalar, head_params,
                    retain_graph=True, allow_unused=True
                )
                if not any(v is not None for v in g):
                    return torch.tensor(0.0, device=self.device)
                g_flat = torch.cat([v.view(-1) for v in g if v is not None])
                return g_flat.norm(p=2) + 1e-8

            g_new_norm = _grad_norm(meta_loss_new).detach()
            g_old_norm = _grad_norm(m_loss_old_grp).detach()

            lambda_meta, _w_new_meta, p_stable_meta = self.controller(
                ctx_meta, mu_old_meta, mu_new_meta
            )

            # 希望 lambda * ||g_old|| >= ||g_new||
            G_old = lambda_meta * g_old_norm
            G_new = g_new_norm
            imbalance = torch.relu(G_new - G_old)
            grad_balance = imbalance ** 2

            w_reg_strength = getattr(self.args, "w_reg_strength", 0.05)
            reg_lambda_meta = w_reg_strength * (lambda_meta - 1.0) ** 2
            reg_p_meta = w_reg_strength * (p_stable_meta - 0.5) ** 2

            meta_coef = getattr(self.args, "meta_grad_balance_coef", 0.5)
            meta_loss = meta_coef * grad_balance + reg_lambda_meta + reg_p_meta

            meta_loss.backward()
            self.opt_cont.step()

        return loss.item()
