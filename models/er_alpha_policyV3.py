import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


# ============================================================================
# 1. Transformer-based Alpha Controller (V3.5, RunB config)
#    - Input:  8-dim context + mean features of old/new batches
#    - Output: w_old, w_new with w_old in [0.55, 0.75], w_new = 1 - w_old
# ============================================================================
class AlphaControllerV3(nn.Module):
    """
    Alpha controller:
      ctx:    (1, 8)
      mu_old: (1, F)
      mu_new: (1, F)
    Returns:
      w_old, w_new (scalars)
    ctx definition:
      [0] l_old_n  = L_old / (L_old + L_new)
      [1] l_new_n  = L_new / (L_old + L_new)
      [2] diff     = L_new - L_old
      [3] sum      = L_old + L_new
      [4] t_norm   = task_id / (total_tasks - 1)
      [5] cosθ     = cos(g_old, g_new)
      [6] log||g_old||
      [7] log||g_new||
    """
    def __init__(self, feature_dim: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.ctx_dim = 8

        # ctx -> d_model
        self.mlp_ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # feature -> d_model
        self.project_feature = nn.Linear(feature_dim, d_model)

        # tokens = [ctx, mu_old, mu_new] -> transformer
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

        # readout scalar
        self.readout = nn.Linear(d_model, 1)

    def forward(self, ctx, mu_old, mu_new):
        # ctx embedding
        ctx_emb = self.mlp_ctx(ctx)              # (1, D)
        old_emb = self.project_feature(mu_old)   # (1, D)
        new_emb = self.project_feature(mu_new)   # (1, D)

        # tokens: [ctx, old, new]
        tokens = torch.stack([ctx_emb, old_emb, new_emb], dim=1)  # (1, 3, D)

        out = self.transformer(tokens)           # (1, 3, D)
        h_ctx = out[:, 0, :]                     # (1, D)

        logit = self.readout(h_ctx)              # (1,1)
        s = torch.sigmoid(logit)[0, 0]           # scalar in (0,1)

        # RunB: w_old in [0.55, 0.75], w_new = 1 - w_old
        lo, hi = 0.55, 0.75
        w_old = lo + (hi - lo) * s
        w_new = 1.0 - w_old
        return w_old, w_new


# ============================================================================
# 2. ER + Meta-Alpha (Policy V3.5, RunB)
# ============================================================================
@register_model("er_alpha_policyV3")
class ERAlphaPolicyV3(ContinualModel):
    """
    ER + Meta-learned alpha (RunB fixed config)

    - Base: ER
        L_new = CE(current batch)
        L_old = CE(buffer batch)

      Total loss for theta-step:
        L_main = w_old * L_old + w_new * L_new
        L_reg  = λ (w_old - target)^2, target = 0.6
        L      = L_main + L_reg

    - Alpha controller:
        Transformer on (ctx, mu_old, mu_new), ctx is gradient-aware.

    - Training:
        * theta-step: update backbone with current w_old/w_new
        * phi-step:   every meta_interval steps, update controller with
                      grad-balance + mild regularization.
    """
    NAME = "er_alpha_policyV3"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)

        # controller structure
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        parser.add_argument("--total_tasks", type=int, default=5)

        # meta / regularization
        parser.add_argument("--w_reg_strength", type=float, default=0.01,
                            help="regularization strength around w_old target")
        parser.add_argument("--meta_lr", type=float, default=1e-3,
                            help="LR for alpha controller")
        parser.add_argument("--meta_interval", type=int, default=50,
                            help="update controller every N steps")
        parser.add_argument("--meta_grad_balance_coef", type=float, default=0.5,
                            help="coefficient for grad-balance term in meta loss")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # feature dim for classifier head
        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            # default for ResNet18 in Mammoth
            self.feature_dim = 512

        # alpha controller
        self.controller = AlphaControllerV3(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            nhead=self.args.dam_nhead,
            num_layers=self.args.dam_layers,
        ).to(self.device)

        # optimizer for backbone (same as ER)
        self.opt = torch.optim.SGD(
            self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.optim_wd,
            momentum=self.args.optim_mom,
        )

        # optimizer for controller
        self.opt_cont = torch.optim.Adam(
            self.controller.parameters(),
            lr=self.args.meta_lr,
            weight_decay=1e-5,
        )

        self.current_task_id = 0
        self.global_step = 0

        # logging of average weights per task
        self.log_steps = 0
        self.log_w_old_sum = 0.0
        self.log_w_new_sum = 0.0

        # per-run stats file (different for each seed)
        seed = getattr(self.args, "seed", 0)
        self.stats_file = f"er_alpha_policyV3_stats_seed{seed}.txt"

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def _get_features(self, x):
        # Mammoth ResNet supports returnt="features"
        return self.net(x, returnt="features")

    def forward(self, x):
        return self.net(x)

    def _build_ctx_and_grads(self, loss_new, loss_old,
                             features_new, features_old):
        """
        Build context vector and gradient stats on classifier head.
        Returns:
          ctx_vec: (1,8)
          mu_new:  (1,F)
          mu_old:  (1,F)
          norm_old, norm_new: scalars (torch.Tensor)
        """
        l_new_val = loss_new.item()
        l_old_val = loss_old.item()
        denom = l_old_val + l_new_val + 1e-8

        # gradients on classifier head only
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
        cos_theta_clamped = cos_theta.clamp(-1.0, 1.0)

        norm_old_scaled = torch.log(norm_old) / 5.0
        norm_new_scaled = torch.log(norm_new) / 5.0

        total_tasks = getattr(self.args, "total_tasks", 5) or 5
        t_norm_val = self.current_task_id / max(total_tasks - 1, 1)

        ctx_vec = torch.tensor(
            [
                l_old_val / denom,           # 0: l_old_n
                l_new_val / denom,           # 1: l_new_n
                l_new_val - l_old_val,       # 2: diff
                denom,                       # 3: sum
                t_norm_val,                  # 4: t_norm
                cos_theta_clamped.item(),    # 5: cosθ
                norm_old_scaled.item(),      # 6: log||g_old||
                norm_new_scaled.item(),      # 7: log||g_new||
            ],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, 8)

        mu_new = features_new.mean(dim=0, keepdim=True)  # (1, F)
        mu_old = features_old.mean(dim=0, keepdim=True)  # (1, F)

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

        self.log_steps = 0
        self.log_w_old_sum = 0.0
        self.log_w_new_sum = 0.0

        self.current_task_id += 1
        super().end_task(dataset)

    # ----------------------------------------------------------------------
    # Main training loop
    # ----------------------------------------------------------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        Each batch:
          1) theta-step: update backbone with current alpha
          2) every meta_interval steps: phi-step to update controller
        """
        self.global_step += 1
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # ======================
        # 1. theta-step (update backbone)
        # ======================
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        # current batch (new)
        feats_new = self._get_features(inputs)
        out_new = self.net.classifier(feats_new)
        loss_new = self.loss(out_new, labels)

        if not self.buffer.is_empty():
            # buffer batch (old)
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            feats_old = self._get_features(buf_inputs)
            out_old = self.net.classifier(feats_old)
            loss_old = self.loss(out_old, buf_labels)

            # build context (will compute grad norms on head)
            ctx_vec, mu_new, mu_old, norm_old, norm_new = \
                self._build_ctx_and_grads(
                    loss_new, loss_old, feats_new, feats_old
                )

            # query controller (no grad in theta-step)
            with torch.no_grad():
                w_old, w_new = self.controller(ctx_vec, mu_old, mu_new)

            # main loss + mild regularization around target=0.6
            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            target = 0.6
            loss_main = w_old * loss_old + w_new * loss_new
            loss_reg = reg_strength * (w_old - target) ** 2
            loss = loss_main + loss_reg

            # accumulate stats
            self.log_steps += 1
            self.log_w_old_sum += float(w_old.item())
            self.log_w_new_sum += float(w_new.item())
        else:
            # first task: pure new loss
            loss = loss_new

        loss.backward()
        self.opt.step()

        # store to buffer (as in ER)
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
        )

        # ======================
        # 2. phi-step (update controller)
        # ======================
        if (self.global_step % self.args.meta_interval == 0) and (not self.buffer.is_empty()):
            self.net.eval()
            self.controller.train()
            self.opt_cont.zero_grad()

            # meta batch: reuse current new batch + a fresh buffer batch
            meta_inputs = inputs
            meta_labels = labels

            # new
            meta_feats_new = self._get_features(meta_inputs)
            meta_out_new = self.net.classifier(meta_feats_new)
            meta_loss_new = self.loss(meta_out_new, meta_labels)

            # old
            m_buf_inputs, m_buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            meta_feats_old = self._get_features(m_buf_inputs)
            meta_out_old = self.net.classifier(meta_feats_old)
            meta_loss_old = self.loss(meta_out_old, m_buf_labels)

            # context for controller (grad norms used only for ctx)
            ctx_meta, mu_new_meta, mu_old_meta, norm_old_meta, norm_new_meta = \
                self._build_ctx_and_grads(
                    meta_loss_new, meta_loss_old,
                    meta_feats_new, meta_feats_old
                )
            norm_old_meta = norm_old_meta.detach()
            norm_new_meta = norm_new_meta.detach()

            # controller outputs with grad
            w_old_meta, w_new_meta = self.controller(
                ctx_meta, mu_old_meta, mu_new_meta
            )

            # grad-balance term: encourage w_old * ||g_old|| >= w_new * ||g_new||
            prod_old = w_old_meta * norm_old_meta
            prod_new = w_new_meta * norm_new_meta
            margin = 0.0
            ratio_term = F.relu(prod_new - prod_old + margin)
            grad_balance = ratio_term ** 2

            # regularization around target=0.6
            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            target = 0.6
            reg_term = reg_strength * (w_old_meta - target) ** 2

            meta_coef = getattr(self.args, "meta_grad_balance_coef", 0.5)
            meta_loss = meta_coef * grad_balance + reg_term

            meta_loss.backward()
            self.opt_cont.step()

        return loss.item()
