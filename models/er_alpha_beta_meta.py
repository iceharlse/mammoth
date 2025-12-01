import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


# ============================================================================
# 1. Controller
# ============================================================================
class AlphaBetaController(nn.Module):
    def __init__(self, feature_dim: int, num_blocks: int = 4, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_blocks = num_blocks
        
        self.global_mlp = nn.Sequential(nn.Linear(8, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.block_mlp = nn.Sequential(nn.Linear(4, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.feat_proj = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.alpha_head = nn.Linear(d_model, 1) 
        self.beta_head = nn.Linear(d_model, 1)

    def forward(self, global_ctx, block_ctx, mu_old, mu_new):
        if block_ctx.dim() == 2:
            block_ctx = block_ctx.unsqueeze(0)

        e_global = self.global_mlp(global_ctx).unsqueeze(1)    
        e_blocks = self.block_mlp(block_ctx)                   
        e_mu_old = self.feat_proj(mu_old).unsqueeze(1)         
        e_mu_new = self.feat_proj(mu_new).unsqueeze(1)         

        # Tokens: [Global, B1...Bn, MuOld, MuNew]
        tokens = torch.cat([e_global, e_blocks, e_mu_old, e_mu_new], dim=1) 
        out = self.transformer(tokens)

        # Alpha
        s_alpha = torch.sigmoid(self.alpha_head(out[:, 0, :]))[0, 0]
        w_old = 0.5 + 0.45 * s_alpha 
        w_new = 1.0 - w_old

        # Beta (per block)
        # 对应 tokens indices: 1 到 1+num_blocks
        h_beta = out[:, 1 : 1 + self.num_blocks, :] 
        beta = torch.sigmoid(self.beta_head(h_beta).squeeze(-1))[0] 

        return w_old, w_new, beta


# ============================================================================
# 2. Model
# ============================================================================
@register_model("er_alpha_beta_meta")
class ErAlphaBetaMeta(ContinualModel):
    NAME = "er_alpha_beta_meta"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        parser.add_argument("--meta_lr", type=float, default=1e-3)
        parser.add_argument("--meta_interval", type=int, default=20)
        parser.add_argument("--meta_lambda_pr", type=float, default=1.0) # 调高了一点权重以配合 element-wise
        parser.add_argument("--meta_pr_target", type=float, default=-2.0)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512
            
        self.block_names = ["layer1", "layer2", "layer3", "layer4"]
        self.num_blocks = len(self.block_names)
        
        self.param2block = {}
        for name, module in self.net.named_children():
            if name in self.block_names:
                b_idx = self.block_names.index(name)
                for pname, _ in module.named_parameters():
                    self.param2block[f"{name}.{pname}"] = b_idx

        self.controller = AlphaBetaController(
            feature_dim=self.feature_dim,
            num_blocks=self.num_blocks,
            d_model=self.args.dam_d_model,
            nhead=self.args.dam_nhead,
            num_layers=self.args.dam_layers
        ).to(self.device)

        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
        self.opt_cont = torch.optim.Adam(self.controller.parameters(), lr=self.args.meta_lr)

        self.global_step = 0
        self.current_task_id = 0
        
        # Logging accumulators
        self.log_w_old = []
        self.log_beta = []

    def _get_block_features(self, x):
        # Forward pass grabbing intermediates
        out = self.net.conv1(x)
        out = self.net.bn1(out)
        out = self.net.relu(out)
        out = self.net.maxpool(out)

        block_feats = []
        for layer in [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]:
            out = layer(out)
            block_feats.append(F.adaptive_avg_pool2d(out, 1).flatten(1))

        out = self.net.avgpool(out)
        final_feat = torch.flatten(out, 1)
        return block_feats, final_feat

    def _effective_rank(self, z, eps=1e-6):
        B, D = z.size()
        if B < 2: return torch.tensor(0.0, device=z.device)
        z = z - z.mean(dim=0, keepdim=True)
        cov = z.t() @ z 
        tr_C = cov.trace()
        tr_C2 = (cov * cov).sum()
        pr_norm = (tr_C * tr_C) / (tr_C2 + eps) / (D + eps)
        return torch.log(pr_norm + eps)

    def _build_state(self, loss_new, loss_old, feats_new_blocks, feats_old_blocks):
        l_new, l_old = loss_new.item(), loss_old.item()
        denom = l_new + l_old + 1e-8
        
        global_ctx = torch.tensor([
            l_new / denom, l_old / denom, l_new - l_old, denom,
            self.current_task_id / 5.0, 0.0, 0.0, 0.0
        ], device=self.device).unsqueeze(0)

        pr_new = torch.stack([self._effective_rank(f) for f in feats_new_blocks])
        pr_old = torch.stack([self._effective_rank(f) for f in feats_old_blocks])
        
        # 暂时填 0 的 grad norm，简化计算
        g_zeros = torch.zeros_like(pr_new)
        block_ctx = torch.stack([pr_new, pr_old, g_zeros, g_zeros], dim=1) 

        return global_ctx, block_ctx

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        # ==================== Phase 1: θ-step ====================
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()
        
        feats_new_blocks, feat_new = self._get_block_features(inputs)
        loss_new = self.loss(self.net.classifier(feat_new), labels)
        
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            feats_old_blocks, feat_old = self._get_block_features(buf_inputs)
            loss_old = self.loss(self.net.classifier(feat_old), buf_labels)
            
            global_ctx, block_ctx = self._build_state(loss_new, loss_old, feats_new_blocks, feats_old_blocks)
            
            with torch.no_grad():
                w_old, w_new, beta = self.controller(
                    global_ctx, block_ctx, 
                    feat_old.mean(0, keepdim=True), feat_new.mean(0, keepdim=True)
                )

            # --- Logging ---
            self.log_w_old.append(w_old.item())
            self.log_beta.append(beta.detach().cpu().numpy())

            # Apply Beta Mask
            (w_old * loss_old + w_new * loss_new).backward()
            
            with torch.no_grad():
                for name, p in self.net.named_parameters():
                    if p.grad is not None and name in self.param2block:
                        p.grad.mul_(beta[self.param2block[name]])
            self.opt.step()
        else:
            loss_new.backward()
            self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        # ==================== Phase 2: φ-step ====================
        if self.global_step % self.args.meta_interval == 0 and not self.buffer.is_empty():
            self.controller.train()
            self.opt_cont.zero_grad()
            
            # Resample for meta
            meta_buf_inputs, meta_buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            
            with torch.no_grad():
                # Re-compute features/losses on current theta (detached)
                m_blk_new, m_feat_new = self._get_block_features(inputs)
                m_blk_old, m_feat_old = self._get_block_features(meta_buf_inputs)
                m_loss_new = self.loss(self.net.classifier(m_feat_new), labels)
                m_loss_old = self.loss(self.net.classifier(m_feat_old), meta_buf_labels)

            m_glob, m_blk = self._build_state(m_loss_new, m_loss_old, m_blk_new, m_blk_old)
            w_old_m, w_new_m, beta_m = self.controller(
                m_glob, m_blk, m_feat_old.mean(0, keepdim=True), m_feat_new.mean(0, keepdim=True)
            )

            # --- Meta Loss Design (Fixing Differentiation) ---
            # 1. Alignment: w_old 应该响应 loss_old 的强度
            target_w_old = m_loss_old / (m_loss_old + m_loss_new + 1e-8)
            loss_align = F.mse_loss(w_old_m, target_w_old.detach())
            
            # 2. Spectral Protection (Element-wise): 只有 rank 塌缩的层，其 beta 才会被惩罚
            ranks_old = torch.stack([self._effective_rank(f) for f in m_blk_old])
            pr_target = getattr(self.args, "meta_pr_target", -2.0)
            reg_per_block = F.relu(pr_target - ranks_old).detach() # (B,)
            
            # Dot product: collapse_risk_i * beta_i.
            # 如果某层 rank 很好 (reg=0), beta_i 不受此项影响
            loss_stability = (beta_m * reg_per_block).mean() 
            
            # 3. Plasticity Bias: 默认给所有 beta 一个微小的向上推力
            # 这样健康的层会飘向 1，不健康的层会被上面的 stability 项压向 0
            loss_plasticity = -0.05 * beta_m.mean()

            meta_loss = loss_align + self.args.meta_lambda_pr * loss_stability + loss_plasticity
            
            meta_loss.backward()
            self.opt_cont.step()
            
        return loss_new.item()

    def end_task(self, dataset):
        # --- Task Summary Printing ---
        if len(self.log_w_old) > 0:
            avg_w_old = np.mean(self.log_w_old)
            # Stack logs -> (Steps, 4), then mean -> (4,)
            avg_beta = np.mean(np.stack(self.log_beta), axis=0)
            
            print(f"\n[DAM-ParetoCL] Task {self.current_task_id + 1} Summary:")
            print(f"  > Avg w_old : {avg_w_old:.4f}")
            print(f"  > Avg Beta  : {np.array2string(avg_beta, precision=4, separator=', ')}")
            
            # 清空 logs 以备下一个 task
            self.log_w_old = []
            self.log_beta = []

        self.current_task_id += 1
        super().end_task(dataset)