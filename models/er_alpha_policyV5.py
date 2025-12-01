import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


# ============================================================================
# 1. Transformer-based Alpha Controller (V5)
#    - Input: 8-dim context + mean features of old/new batches
#    - Output: w_old, w_new with w_old in [alpha_min, alpha_max]
#      interpreted as mixture between a "plastic" and a "stable" regime.
# ============================================================================
class AlphaControllerV5(nn.Module):
    """
    Alpha controller V5:
      ctx:    (1, 8)
      mu_old: (1, F)
      mu_new: (1, F)

    Returns:
      w_old, w_new, p_stable (scalars)

    We interpret w_old as a convex combination:
      alpha_plastic = alpha_min
      alpha_stable  = alpha_max
      p_stable      = sigmoid(f(ctx, mu_old, mu_new)) in [0, 1]
      w_old         = p_stable * alpha_stable + (1 - p_stable) * alpha_plastic
      w_new         = 1 - w_old
    """
    def __init__(self, feature_dim: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 alpha_min: float = 0.55, alpha_max: float = 0.75):
        super().__init__()
        self.ctx_dim = 8
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # ctx -> d_model
        self.mlp_ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # feature -> d_model
        self.project_feature = nn.Linear(feature_dim, d_model)

        # tokens = [ctx, mu_old, mu_new] -> transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # readout for p_stable (logit)
        self.readout = nn.Linear(d_model, 1)

    def forward(self, ctx, mu_old, mu_new):
        """
        Args:
          ctx:      (1, 8)
          mu_old:   (1, F)
          mu_new:   (1, F)

        Returns:
          w_old, w_new, p_stable
        """
        ctx_emb = self.mlp_ctx(ctx)              # (1, D)
        old_emb = self.project_feature(mu_old)   # (1, D)
        new_emb = self.project_feature(mu_new)   # (1, D)

        tokens = torch.stack([ctx_emb, old_emb, new_emb], dim=1)  # (1, 3, D)
        out = self.transformer(tokens)           # (1, 3, D)
        h_ctx = out[:, 0, :]                     # (1, D)

        logit = self.readout(h_ctx)              # (1,1)
        p_stable = torch.sigmoid(logit)[0, 0]    # scalar in [0,1]

        alpha_plastic = self.alpha_min
        alpha_stable = self.alpha_max

        # convex combination between plastic and stable regimes
        w_old = p_stable * alpha_stable + (1.0 - p_stable) * alpha_plastic
        w_new = 1.0 - w_old

        return w_old, w_new, p_stable


# ============================================================================
# 2. ER + Alpha Policy V5
# ============================================================================
@register_model("er_alpha_policyV5")
class ERAlphaPolicyV5(ContinualModel):
    """
    ER + conflict-gated meta alpha (V5)

    Differences vs V3/V4:
      - Controller outputs a gating scalar p_stable in [0,1], which mixes
        between a "plastic" alpha (alpha_min) and a "stable" alpha (alpha_max).
      - We keep using gradient-geometry-based context, but we apply
        EMA smoothing and log everything needed for later analysis.
      - Meta-loss still uses a grad-balance surrogate, but regularizes
        p_stable around 0.5 instead of hard-pulling w_old to 0.6.
    """
    NAME = "er_alpha_policyV5"
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
                            help="regularization strength for p_stable (around 0.5)")
        parser.add_argument("--meta_lr", type=float, default=1e-3,
                            help="LR for alpha controller")
        parser.add_argument("--meta_interval", type=int, default=50,
                            help="update controller every N steps")
        parser.add_argument("--meta_grad_balance_coef", type=float, default=0.5,
                            help="coefficient for grad-balance term in meta loss")

        # context smoothing & alpha range
        parser.add_argument("--ctx_ema_beta", type=float, default=0.9,
                            help="EMA factor for context (0 = no EMA)")
        parser.add_argument("--alpha_min", type=float, default=0.55,
                            help="minimum w_old (plastic regime)")
        parser.add_argument("--alpha_max", type=float, default=0.75,
                            help="maximum w_old (stable regime)")

        # optional: example-based meta interval (0 = disabled, use meta_interval)
        parser.add_argument("--meta_interval_examples", type=int, default=0,
                            help="if >0, update controller every N seen examples")

        # logging
        parser.add_argument("--log_interval", type=int, default=100,
                            help="steps between detailed log lines")

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # feature dim for classifier head
        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            # default for ResNet18-style backbones
            self.feature_dim = 512

        # controller
        self.controller = AlphaControllerV5(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            nhead=self.args.dam_nhead,
            num_layers=self.args.dam_layers,
            alpha_min=getattr(self.args, "alpha_min", 0.55),
            alpha_max=getattr(self.args, "alpha_max", 0.75),
        ).to(self.device)

        # optimizer for backbone
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

        # task / step counters
        self.current_task_id = 0
        self.global_step = 0

        # example-based meta token
        self.meta_token_examples = 0
        self.meta_interval_examples = getattr(self.args, "meta_interval_examples", 0)

        # EMA for ctx
        self.ctx_ema = None
        self.ctx_ema_beta = getattr(self.args, "ctx_ema_beta", 0.9)

        # logging buffers
        self.log_interval = getattr(self.args, "log_interval", 100)
        self.task_steps = 0
        self.task_w_old_sum = 0.0
        self.task_p_stable_sum = 0.0
        self.task_cos_sum = 0.0
        self.task_l_old_sum = 0.0
        self.task_l_new_sum = 0.0

        # log files
        seed = getattr(self.args, "seed", 0)
        self.stats_file = f"er_alpha_policyV5_stats_seed{seed}.txt"
        self.step_log_file = f"er_alpha_policyV5_steps_seed{seed}.txt"

    # ------------------------------------------------------------------
    # basic utilities
    # ------------------------------------------------------------------
    def forward(self, x):
        return self.net(x)

    def _get_features(self, x):
        # Mammoth-style backbones support returnt="features"
        return self.net(x, returnt="features")

    # ------------------------------------------------------------------
    # context & gradient statistics
    # ------------------------------------------------------------------
    def _build_ctx_and_grads(self, loss_new, loss_old,
                             features_new, features_old):
        """
        Build 8-dim context vector + gradient stats on the classifier head.

        Returns:
          ctx_vec:        (1, 8)
          mu_new:         (1, F)
          mu_old:         (1, F)
          norm_old:       scalar (detached)
          norm_new:       scalar (detached)
          cos_theta:      scalar (detached, in [-1, 1])
          l_old_val:      float
          l_new_val:      float
        """
        l_new_val = float(loss_new.item())
        l_old_val = float(loss_old.item())
        denom_loss = l_old_val + l_new_val + 1e-8

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
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        # scaled log-norms (same trick as V3)
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

        # mean features
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
    # logging helpers
    # ------------------------------------------------------------------
    def _log_step(self, tag, info: dict):
        """
        Append a single line of key=value pairs to the step log file.
        tag: "theta" or "meta"
        """
        try:
            with open(self.step_log_file, "a") as f:
                parts = [f"[{tag}] step={self.global_step} task={self.current_task_id}"]
                for k, v in info.items():
                    parts.append(f"{k}={v}")
                line = " ".join(parts)
                f.write(line + "\n")
        except Exception:
            # don't break training because of logging issues
            pass

    def end_task(self, dataset):
        # print and log per-task averages
        if self.task_steps > 0:
            avg_w_old = self.task_w_old_sum / self.task_steps
            avg_p_stable = self.task_p_stable_sum / self.task_steps
            avg_cos = self.task_cos_sum / self.task_steps
            avg_l_old = self.task_l_old_sum / self.task_steps
            avg_l_new = self.task_l_new_sum / self.task_steps

            msg = (
                f"[V5] Task {self.current_task_id} summary: "
                f"steps={self.task_steps} | "
                f"avg_w_old={avg_w_old:.4f} | "
                f"avg_p_stable={avg_p_stable:.4f} | "
                f"avg_cos={avg_cos:.4f} | "
                f"avg_l_old={avg_l_old:.4f} | "
                f"avg_l_new={avg_l_new:.4f}"
            )
            print(msg)
            try:
                with open(self.stats_file, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

        # reset counters
        self.task_steps = 0
        self.task_w_old_sum = 0.0
        self.task_p_stable_sum = 0.0
        self.task_cos_sum = 0.0
        self.task_l_old_sum = 0.0
        self.task_l_new_sum = 0.0

        self.current_task_id += 1
        super().end_task(dataset)

    # ------------------------------------------------------------------
    # main training loop
    # ------------------------------------------------------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        One training step:
          1) theta-step: update backbone with current alpha policy
          2) occasionally: phi-step to update controller
        """
        self.global_step += 1

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        batch_size = inputs.size(0)

        # example-based meta schedule
        if self.meta_interval_examples > 0:
            self.meta_token_examples += batch_size

        # ======================
        # 1. theta-step
        # ======================
        self.opt.zero_grad()
        self.net.train()
        self.controller.eval()

        feats_new = self._get_features(inputs)
        out_new = self.net.classifier(feats_new)
        loss_new = self.loss(out_new, labels)

        if not self.buffer.is_empty():
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

            # EMA smoothing of ctx
            if self.ctx_ema_beta > 0.0:
                if self.ctx_ema is None:
                    self.ctx_ema = ctx_vec.detach()
                else:
                    beta = self.ctx_ema_beta
                    self.ctx_ema = beta * self.ctx_ema + (1.0 - beta) * ctx_vec.detach()
                ctx_used = self.ctx_ema
            else:
                ctx_used = ctx_vec

            # query controller (no grad in theta-step)
            with torch.no_grad():
                w_old, w_new, p_stable = self.controller(ctx_used, mu_old, mu_new)

            # main loss
            loss_main = w_old * loss_old + w_new * loss_new

            # mild reg: encourage p_stable around 0.5 (balanced by default)
            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            reg_term = reg_strength * (p_stable - 0.5) ** 2

            loss = loss_main + reg_term

            # logging accumulators
            self.task_steps += 1
            self.task_w_old_sum += float(w_old.item())
            self.task_p_stable_sum += float(p_stable.item())
            self.task_cos_sum += float(cos_theta.item())
            self.task_l_old_sum += l_old_val
            self.task_l_new_sum += l_new_val

            # occasional detailed log line
            if (self.global_step % self.log_interval) == 0:
                l_old_n = ctx_used[0, 0].item()
                l_new_n = ctx_used[0, 1].item()
                diff = ctx_used[0, 2].item()
                l_sum = ctx_used[0, 3].item()
                t_norm = ctx_used[0, 4].item()
                self._log_step(
                    "theta",
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

        # store current batch to buffer
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
        )

        # ======================
        # 2. phi-step (meta-update)
        # ======================
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

            # reuse current batch as "new"
            meta_inputs = inputs
            meta_labels = labels

            meta_feats_new = self._get_features(meta_inputs)
            meta_out_new = self.net.classifier(meta_feats_new)
            meta_loss_new = self.loss(meta_out_new, meta_labels)

            # sample a fresh buffer batch as "old"
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

            # no EMA in meta-step: we want current gradient geometry
            w_old_meta, w_new_meta, p_stable_meta = self.controller(
                ctx_meta, mu_old_meta, mu_new_meta
            )

            # grad-balance term: encourage w_old*||g_old|| >= w_new*||g_new||
            prod_old = w_old_meta * norm_old_meta
            prod_new = w_new_meta * norm_new_meta
            margin = 0.0
            ratio_term = F.relu(prod_new - prod_old + margin)
            grad_balance = ratio_term ** 2

            # regularize p_stable around 0.5
            reg_strength = getattr(self.args, "w_reg_strength", 0.01)
            p_reg = reg_strength * (p_stable_meta - 0.5) ** 2

            meta_coef = getattr(self.args, "meta_grad_balance_coef", 0.5)
            meta_loss = meta_coef * grad_balance + p_reg

            meta_loss.backward()
            self.opt_cont.step()

            # log meta-step
            self._log_step(
                "meta",
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
