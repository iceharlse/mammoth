import torch
import torch.nn.functional as F
from models.er_alpha_policyV3 import ERAlphaPolicyV3, AlphaControllerV3
from models import register_model

@register_model("er_alpha_policyV4")
class ERAlphaPolicyV4(ERAlphaPolicyV3):
    """
    V4: ER + Meta-Alpha + State Momentum (MoCo-Style State Smoothing)
    解决 Batch 32 下梯度噪声过大导致 Controller 决策抖动的问题。
    """
    NAME = "er_alpha_policyV4"

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        
        # --- V4 新增: 状态动量系数 (State Momentum) ---
        # 0.9 表示非常相信历史趋势，0.1 表示相信当前观测
        # 对于小 Batch (32)，建议设高一点 (0.9 或 0.95) 以抵抗噪声
        self.ctx_momentum = getattr(args, "ctx_momentum", 0.9) 
        
        # 用于存储平滑后的 Context (1, 8)
        self.register_buffer("running_ctx", torch.zeros(1, 8))
        self.ctx_initialized = False

    def _update_running_ctx(self, current_ctx):
        """
        MoCo-style 动量更新 Context
        S_tilde = m * S_tilde + (1-m) * S_current
        """
        if not self.ctx_initialized:
            self.running_ctx = current_ctx.detach()
            self.ctx_initialized = True
        else:
            # 动量更新
            self.running_ctx = (self.ctx_momentum * self.running_ctx) + \
                               ((1 - self.ctx_momentum) * current_ctx.detach())
        
        return self.running_ctx

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # ======================
        # 1. theta-step (Backbone Update)
        # ======================
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        # New Task Batch
        feats_new = self._get_features(inputs)
        out_new = self.net.classifier(feats_new)
        loss_new = self.loss(out_new, labels)

        if not self.buffer.is_empty():
            # Old Task Batch
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            feats_old = self._get_features(buf_inputs)
            out_old = self.net.classifier(feats_old)
            loss_old = self.loss(out_old, buf_labels)

            # 获取【瞬时】Context
            raw_ctx, mu_new, mu_old, norm_old, norm_new = \
                self._build_ctx_and_grads(loss_new, loss_old, feats_new, feats_old)

            # --- V4 核心: 使用【动量平滑后】的 Context 输入给 Controller ---
            # 这样 Controller 看到的不再是 Batch 32 的随机噪声，而是平滑后的趋势
            smooth_ctx = self._update_running_ctx(raw_ctx)
            
            # Controller 决策
            with torch.no_grad():
                # 注意：这里我们喂给 Controller 是 smooth_ctx，而不是 raw_ctx
                w_old, w_new = self.controller(smooth_ctx, mu_old, mu_new)

            # Loss Calculation
            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            target = 0.6
            loss_main = w_old * loss_old + w_new * loss_new
            loss_reg = reg_strength * (w_old - target) ** 2
            loss = loss_main + loss_reg

            # Stats logging
            self.log_steps += 1
            self.log_w_old_sum += float(w_old.item())
            self.log_w_new_sum += float(w_new.item())

        else:
            loss = loss_new
            # 第一步也要初始化 running_ctx，避免空指针，虽然此时只用 loss_new
            # (为了代码简洁，这里略过伪造 ctx 的过程，通常第一步不需要 controller)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        # ======================
        # 2. phi-step (Controller Update)
        # ======================
        if (self.global_step % self.args.meta_interval == 0) and (not self.buffer.is_empty()):
            self.net.eval()
            self.controller.train()
            self.opt_cont.zero_grad()

            # Meta Batch (Reuse or Resample)
            # 为了省内存，这里简化重用逻辑，实际上最好 resample
            meta_inputs, meta_labels = inputs, labels
            m_buf_inputs, m_buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            
            # Forward passes for meta-gradients
            m_feats_new = self._get_features(meta_inputs)
            m_out_new = self.net.classifier(m_feats_new)
            m_loss_new = self.loss(m_out_new, meta_labels)
            
            m_feats_old = self._get_features(m_buf_inputs)
            m_out_old = self.net.classifier(m_feats_old)
            m_loss_old = self.loss(m_out_old, m_buf_labels)

            # Meta Context
            m_raw_ctx, m_mu_new, m_mu_old, m_n_old, m_n_new = \
                self._build_ctx_and_grads(m_loss_new, m_loss_old, m_feats_new, m_feats_old)
            
            # --- V4 注意点 ---
            # 在更新 Controller 时，输入给它的也应该是【当前的 Smooth Context】
            # 因为推理时用的是 Smooth，训练时也要保持分布一致
            # 但这里不需要 detach，因为我们要训练 Controller 对 Smooth Input 的反应
            # (不过 running_ctx 本身是 detach 的，所以梯度只传过 Controller 参数)
            m_smooth_ctx = self.running_ctx.detach() 

            w_old_meta, w_new_meta = self.controller(m_smooth_ctx, m_mu_old, m_mu_new)

            # Grad Balance Loss & Reg
            prod_old = w_old_meta * m_n_old.detach()
            prod_new = w_new_meta * m_n_new.detach()
            ratio_term = F.relu(prod_new - prod_old) # Margin 0
            grad_balance = ratio_term ** 2
            
            reg_term = reg_strength * (w_old_meta - target) ** 2
            meta_loss = 0.5 * grad_balance + reg_term
            
            meta_loss.backward()
            self.opt_cont.step()

        return loss.item()