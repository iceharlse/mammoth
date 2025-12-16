import torch
import torch.nn.functional as F

from models.er_alpha_policyV5 import ERAlphaPolicyV5
from utils.args import ArgumentParser
from models.star_utils.star_perturber import Perturber, add_perturb_args
from models import register_model


@register_model("er_alpha_policyV5_star")
class ERAlphaPolicyV5STAR(ERAlphaPolicyV5):
    """
    ER + Alpha Policy V5 + STAR 组合版（修正后的完整实现）

    设计原则：
      - 继承 V5：保留冲突门控 alpha、meta-update、日志等所有逻辑；
      - 整个训练 step 只算一次 global_step（不会像之前那样 double 计数）；
      - 在 V5 的 theta-step 之前，插入一次 STAR 对 buffer 的扰动，
        模式和你 er_star_alphaV3 里的写法一致。
    """
    NAME = "er_alpha_policyV5_star"
    COMPATIBILITY = ERAlphaPolicyV5.COMPATIBILITY

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        # 先拿到 V5 的所有参数（rehearsal + controller + meta + log 等）
        parser = ERAlphaPolicyV5.get_parser(parser)
        # 再附加 STAR 的扰动参数（和 ErSTAR / er_star_alphaV3 一样）
        add_perturb_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        # 用 V5 的初始化：buffer、controller、opt、日志、meta 配置都在这里
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # 再多一个 STAR 的扰动器
        self.pert = Perturber(self)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        一次完整训练 step：
          1) global_step += 1，更新 meta 计数
          2) 如果 buffer 非空：
               在 buffer 上跑一次 STAR 扰动（perturb）
          3) V5 的 theta-step：
               - 冲突门控 alpha：根据梯度几何算 w_old / w_new
               - 加权 old/new loss + 轻微正则，更新 backbone
               - 写 step 日志
          4) 把当前 batch 加入 buffer
          5) 按 meta_interval / meta_interval_examples 触发 V5 的 meta-step：
               - 重新算 old/new grad 几何
               - 用 grad-balance + p_stable 正则更新 controller
               - 写 meta-step 日志
        """
        # ------------------------------------------------
        # 1. 计步 & 数据搬到 device
        # ------------------------------------------------
        self.global_step += 1

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        batch_size = inputs.size(0)

        # example-based meta 计数（如果启用的话）
        if self.meta_interval_examples > 0:
            self.meta_token_examples += batch_size

        # ------------------------------------------------
        # 2. STAR 扰动：在 buffer 上走一小步
        #    （完全参照你 V3+STAR 的实现方式）
        # ------------------------------------------------
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs_star, buf_labels_star = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,   # 不提前搬到 device，交给 Perturber 处理
            )
            # 在当前 net 和 opt 上做一次 STAR 更新
            self.pert(buf_inputs_star, buf_labels_star)

        # ------------------------------------------------
        # 3. V5 的 theta-step：冲突门控 alpha
        # ------------------------------------------------
        self.net.train()
        self.controller.eval()

        feats_new = self._get_features(inputs)
        out_new = self.net.classifier(feats_new)
        loss_new = self.loss(out_new, labels)

        if not self.buffer.is_empty():
            # 正式的 replay batch（和 STAR 那次可以是不同采样）
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            feats_old = self._get_features(buf_inputs)
            out_old = self.net.classifier(feats_old)
            loss_old = self.loss(out_old, buf_labels)

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
                loss_new, loss_old, feats_new, feats_old
            )

            # EMA 平滑 ctx
            if self.ctx_ema_beta > 0.0:
                if self.ctx_ema is None:
                    self.ctx_ema = ctx_vec.detach()
                else:
                    beta = self.ctx_ema_beta
                    self.ctx_ema = beta * self.ctx_ema + (1.0 - beta) * ctx_vec.detach()
                ctx_used = self.ctx_ema
            else:
                ctx_used = ctx_vec

            # controller 前向（theta-step 不对 controller 反传）
            with torch.no_grad():
                w_old, w_new, p_stable = self.controller(ctx_used, mu_old, mu_new)

            # 主 loss：冲突门控的 old/new 加权
            loss_main = w_old * loss_old + w_new * loss_new
            # 轻微正则：鼓励 p_stable 不要长期极端
            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            reg_term = reg_strength * (p_stable - 0.5) ** 2

            loss = loss_main + reg_term

            # 统计 per-task 平均
            self.task_steps += 1
            self.task_w_old_sum += float(w_old.item())
            self.task_p_stable_sum += float(p_stable.item())
            self.task_cos_sum += float(cos_theta.item())
            self.task_l_old_sum += l_old_val
            self.task_l_new_sum += l_new_val

            # 详细 step log（每 log_interval 记一次）
            if (self.global_step % self.log_interval) == 0:
                l_old_n = ctx_used[0, 0].item()
                l_new_n = ctx_used[0, 1].item()
                diff = ctx_used[0, 2].item()
                l_sum = ctx_used[0, 3].item()
                t_norm = ctx_used[0, 4].item()
                self._log_step(
                    "theta_star",
                    {
                        "w_old": f"{w_old.item():.4f}",
                        "w_new": f"{w_new.item():.4f}",
                        "p_stable": f"{p_stable.item():.4f}",
                        "cos": f"{cos_theta.item():.4f}",
                        "norm_old": f"{norm_old.item():.4f}",
                        "norm_new": f"{norm_new.item():.4f}",
                        "l_old": f"{l_old_val:.4f}",
                        "l_new": f"{l_new_val:.4f}",
                        "l_old_n": f"{l_old_n:.4f}",
                        "l_new_n": f"{l_new_n:.4f}",
                        "diff": f"{diff:.4f}",
                        "l_sum": f"{l_sum:.4f}",
                        "t_norm": f"{t_norm:.4f}",
                    },
                )
        else:
            loss = loss_new

        loss.backward()
        self.opt.step()

        # ------------------------------------------------
        # 4. 把当前真实 batch 存进 buffer
        # ------------------------------------------------
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
        )

        # ------------------------------------------------
        # 5. V5 的 meta-step：完全沿用原有逻辑
        # ------------------------------------------------
        trigger_meta = False
        if self.meta_interval_examples > 0:
            if self.meta_token_examples >= self.meta_interval_examples:
                trigger_meta = True
                self.meta_token_examples = 0
        else:
            if (self.global_step % self.args.meta_interval) == 0:
                trigger_meta = True

        if trigger_meta and (not self.buffer.is_empty()):
            self.net.eval()
            self.controller.train()
            self.opt_cont.zero_grad()

            # 用当前 batch 做 "new"
            meta_inputs = inputs
            meta_labels = labels

            meta_feats_new = self._get_features(meta_inputs)
            meta_out_new = self.net.classifier(meta_feats_new)
            meta_loss_new = self.loss(meta_out_new, meta_labels)

            # 从 buffer 抽一批做 "old"
            m_buf_inputs, m_buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            meta_feats_old = self._get_features(m_buf_inputs)
            meta_out_old = self.net.classifier(meta_feats_old)
            meta_loss_old = self.loss(meta_out_old, m_buf_labels)

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
                meta_loss_new, meta_loss_old,
                meta_feats_new, meta_feats_old
            )

            # meta-step 直接用当前 ctx（不 EMA）
            w_old_meta, w_new_meta, p_stable_meta = self.controller(
                ctx_meta, mu_old_meta, mu_new_meta
            )

            # grad-balance：鼓励 w_old*||g_old|| 不比 w_new*||g_new|| 弱太多
            prod_old = w_old_meta * norm_old_meta
            prod_new = w_new_meta * norm_new_meta
            ratio_term = F.relu(prod_new - prod_old)
            grad_balance = ratio_term ** 2

            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            p_reg = reg_strength * (p_stable_meta - 0.5) ** 2

            meta_coef = getattr(self.args, "meta_grad_balance_coef", 0.5)
            meta_loss = meta_coef * grad_balance + p_reg

            meta_loss.backward()
            self.opt_cont.step()

            self._log_step(
                "meta_star",
                {
                    "meta_loss": f"{meta_loss.item():.6f}",
                    "grad_balance": f"{grad_balance.item():.6f}",
                    "p_reg": f"{p_reg.item():.6f}",
                    "w_old_meta": f"{w_old_meta.item():.4f}",
                    "w_new_meta": f"{w_new_meta.item():.4f}",
                    "p_stable_meta": f"{p_stable_meta.item():.4f}",
                    "cos_meta": f"{cos_meta.item():.4f}",
                    "norm_old_meta": f"{norm_old_meta.item():.4f}",
                    "norm_new_meta": f"{norm_new_meta.item():.4f}",
                    "l_old_meta": f"{l_old_meta:.4f}",
                    "l_new_meta": f"{l_new_meta:.4f}",
                },
            )

        return loss.item()
