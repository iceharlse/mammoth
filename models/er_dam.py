
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


# ============================================================================
# ER + DAM (Alpha + Beta + Anchor Cone + Meta updates)
# ----------------------------------------------------------------------------
# Core idea:
#   - Compute separate losses on new batch (stream) and replay batch (buffer).
#   - A transformer-based controller reads gradient-geometry context (losses,
#     cos(theta), log norms) + mean features, outputs:
#        * w_old (alpha-like) in [alpha_min, alpha_max]
#        * beta  (buffer amplification) in [beta_min, beta_max]
#   - Blend controller outputs with ER's "natural" per-sample weighting using
#     gains (dam_alpha_gain / dam_beta_gain).
#   - Apply an anchor-gradient cone constraint:
#        g_safe is a projection/clamp of g_total so that it stays close to the
#        ER anchor gradient g_anchor (fixed ER weights).
#   - Meta-updates periodically adjust controller parameters using a simple
#     surrogate: balance weighted gradient magnitudes of old vs new.
#
# Notes:
#   - Cone constraint is applied on the full parameter gradient (global),
#     so there is no "front layers constrained, back layers free" artifact.
# ============================================================================


class AlphaBetaController(nn.Module):
    """
    Transformer controller:
      inputs:
        ctx   : (1, 8)
        mu_old: (1, F)
        mu_new: (1, F)

      outputs:
        w_old, w_new, beta, p_stable, p_beta
    """
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        alpha_min: float = 0.55,
        alpha_max: float = 0.75,
        beta_min: float = 0.7,
        beta_max: float = 1.5,
    ):
        super().__init__()
        self.ctx_dim = 8
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.mlp_ctx = nn.Sequential(
            nn.Linear(self.ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.project_feature = nn.Linear(feature_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # two readouts: stable gate + beta gate
        self.readout_stable = nn.Linear(d_model, 1)
        self.readout_beta = nn.Linear(d_model, 1)

    def forward(self, ctx, mu_old, mu_new):
        ctx_emb = self.mlp_ctx(ctx)                # (1, D)
        old_emb = self.project_feature(mu_old)     # (1, D)
        new_emb = self.project_feature(mu_new)     # (1, D)

        tokens = torch.stack([ctx_emb, old_emb, new_emb], dim=1)  # (1, 3, D)
        out = self.transformer(tokens)
        h = out[:, 0, :]  # ctx token

        p_stable = torch.sigmoid(self.readout_stable(h))[0, 0]  # scalar
        p_beta = torch.sigmoid(self.readout_beta(h))[0, 0]      # scalar

        # alpha-like w_old
        w_old = p_stable * self.alpha_max + (1.0 - p_stable) * self.alpha_min
        w_new = 1.0 - w_old

        # beta scaling
        beta = self.beta_min + (self.beta_max - self.beta_min) * p_beta

        return w_old, w_new, beta, p_stable, p_beta


def _flat_from_grads(grad_list, params, device):
    flats = []
    for g, p in zip(grad_list, params):
        if g is None:
            flats.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
        else:
            flats.append(g.detach().view(-1).float())
    return torch.cat(flats, dim=0)


def _flat_from_param_grads(params, device):
    flats = []
    for p in params:
        if p.grad is None:
            flats.append(torch.zeros(p.numel(), device=device, dtype=torch.float32))
        else:
            flats.append(p.grad.detach().view(-1).float())
    return torch.cat(flats, dim=0)


def _set_param_grads_from_flat(params, flat, device):
    idx = 0
    for p in params:
        n = p.numel()
        g = flat[idx:idx+n].view_as(p).to(device=device, dtype=p.dtype)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        idx += n


def _project_to_cone_and_cap(g_total, g_anchor, cos_min: float, norm_eps: float):
    """
    Enforce:
      1) cosine(g_safe, g_anchor) >= cos_min   (if cos_min > -1)
      2) ||g_safe|| <= (1+norm_eps)*||g_anchor||
    """
    eps = 1e-12
    a_norm = g_anchor.norm() + eps
    t_norm = g_total.norm() + eps

    if a_norm < 1e-8:
        return g_total

    u = g_anchor / a_norm

    if cos_min is not None and cos_min > -1.0:
        dot = (g_total @ u)
        cos_cur = dot / (t_norm + eps)
        if cos_cur < cos_min:
            g_par = dot * u
            g_perp = g_total - g_par
            b = g_perp.norm() + eps

            if dot <= 0:
                g_total = u * t_norm
            else:
                tan_theta = math.sqrt(max(1.0 / (cos_min * cos_min) - 1.0, 0.0))
                k = (dot / b) * tan_theta
                k = min(float(k), 1.0)
                g_total = g_par + k * g_perp

    max_norm = (1.0 + max(norm_eps, 0.0)) * a_norm
    g_norm = g_total.norm() + eps
    if g_norm > max_norm:
        g_total = g_total * (max_norm / g_norm)

    return g_total


@register_model("er_dam")
class ERDamAlphaBeta(ContinualModel):
    """
    ER + DAM (alpha+beta controller + anchor-gradient cone constraint).
    """
    NAME = "er_dam"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)

        # controller structure
        parser.add_argument("--dam_d_model", type=int, default=64)
        parser.add_argument("--dam_nhead", type=int, default=4)
        parser.add_argument("--dam_layers", type=int, default=2)
        parser.add_argument("--num_tasks", type=int, default=10)

        # alpha/beta ranges
        parser.add_argument("--alpha_min", type=float, default=0.55)
        parser.add_argument("--alpha_max", type=float, default=0.80)
        parser.add_argument("--beta_min", type=float, default=0.7)
        parser.add_argument("--beta_max", type=float, default=1.6)

        # gains to blend controller outputs with ER anchor weights
        parser.add_argument("--dam_alpha_gain", type=float, default=0.7,
                            help="0=use ER anchor weights; 1=use controller w_old")
        parser.add_argument("--dam_beta_gain", type=float, default=0.7,
                            help="0=beta=1; 1=use controller beta")

        # meta learning
        parser.add_argument("--meta_lr", type=float, default=1e-3)
        parser.add_argument("--meta_interval", type=int, default=50)
        parser.add_argument("--meta_interval_examples", type=int, default=0)
        parser.add_argument("--meta_grad_balance_coef", type=float, default=0.5)
        parser.add_argument("--meta_margin", type=float, default=0.0)

        # regularization
        parser.add_argument("--w_reg_strength", type=float, default=0.01,
                            help="regularize p_stable around 0.5")
        parser.add_argument("--beta_reg_strength", type=float, default=0.01,
                            help="regularize beta around 1.0")

        # context EMA
        parser.add_argument("--ctx_ema_beta", type=float, default=0.9)

        # handcuff / anchor cone
        parser.add_argument("--dam_cone_cos", type=float, default=0.0,
                            help="minimum cosine with ER anchor gradient (-1 disables)")
        parser.add_argument("--dam_cone_norm_eps", type=float, default=0.3,
                            help="norm cap: ||g|| <= (1+eps)*||g_anchor||")
        parser.add_argument("--dam_warmup_epochs", type=int, default=1,
                            help="for first N epochs of each task, skip cone constraint")
        parser.add_argument("--dam_clip_grad", type=float, default=5.0,
                            help="gradient clip (0 disables)")

        # logging
        parser.add_argument("--log_interval", type=int, default=100)

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # feature dim
        if hasattr(self.net, "num_features"):
            self.feature_dim = self.net.num_features
        else:
            self.feature_dim = 512

        self.has_classifier = hasattr(self.net, "classifier")
        self.controller = AlphaBetaController(
            feature_dim=self.feature_dim,
            d_model=self.args.dam_d_model,
            nhead=self.args.dam_nhead,
            num_layers=self.args.dam_layers,
            alpha_min=self.args.alpha_min,
            alpha_max=self.args.alpha_max,
            beta_min=self.args.beta_min,
            beta_max=self.args.beta_max,
        ).to(self.device)

        # explicit optimizers (like your V5)
        self.opt = torch.optim.SGD(
            self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.optim_wd,
            momentum=self.args.optim_mom,
        )
        self.opt_cont = torch.optim.Adam(self.controller.parameters(), lr=self.args.meta_lr, weight_decay=1e-5)

        self.current_task_id = 0
        self.global_step = 0
        self.meta_token_examples = 0
        self.ctx_ema = None

        self.log_interval = getattr(self.args, "log_interval", 100)

    def forward(self, x):
        return self.net(x)

    def _get_features(self, x):
        return self.net(x, returnt="features")

    def _safe_head_params(self):
        if self.has_classifier:
            return list(self.net.classifier.parameters())
        return []

    def _build_ctx(self, loss_new, loss_old, features_new, features_old):
        """
        8-dim ctx: [l_old_n, l_new_n, diff, sum, t_norm, cos, log||g_old||, log||g_new||]
        computed on classifier head grads (cheap + informative).
        """
        l_new_val = float(loss_new.item())
        l_old_val = float(loss_old.item())
        denom = l_old_val + l_new_val + 1e-8

        head_params = self._safe_head_params()
        if len(head_params) > 0:
            g_old = torch.autograd.grad(loss_old, head_params, retain_graph=True, allow_unused=True)
            g_new = torch.autograd.grad(loss_new, head_params, retain_graph=True, allow_unused=True)
            g_old_flat = torch.cat([g.view(-1) for g in g_old if g is not None])
            g_new_flat = torch.cat([g.view(-1) for g in g_new if g is not None])
            norm_old = g_old_flat.norm() + 1e-8
            norm_new = g_new_flat.norm() + 1e-8
            cos = (g_old_flat @ g_new_flat) / (norm_old * norm_new)
            cos = cos.clamp(-1.0, 1.0)
            log_old = torch.log(norm_old) / 5.0
            log_new = torch.log(norm_new) / 5.0
        else:
            cos = torch.tensor(0.0, device=self.device)
            log_old = torch.tensor(0.0, device=self.device)
            log_new = torch.tensor(0.0, device=self.device)
            norm_old = torch.tensor(0.0, device=self.device)
            norm_new = torch.tensor(0.0, device=self.device)

        t_norm = self.current_task_id / max((getattr(self.args, "num_tasks", 5) or 5) - 1, 1)

        ctx = torch.tensor(
            [
                l_old_val / denom,
                l_new_val / denom,
                l_new_val - l_old_val,
                denom,
                float(t_norm),
                float(cos.item()),
                float(log_old.item()),
                float(log_new.item()),
            ],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        mu_new = features_new.mean(dim=0, keepdim=True).detach()
        mu_old = features_old.mean(dim=0, keepdim=True).detach()

        return ctx, mu_old, mu_new, norm_old.detach(), norm_new.detach(), cos.detach()

    def _anchor_weights(self, real_bs: int, buf_bs: int):
        total = max(real_bs + buf_bs, 1)
        w_old_anchor = float(buf_bs) / float(total)
        w_new_anchor = 1.0 - w_old_anchor
        return w_old_anchor, w_new_anchor

    def _blend_weights_no_grad(self, w_old_pred, beta_pred, real_bs: int, buf_bs: int):
        w_old_anchor, w_new_anchor = self._anchor_weights(real_bs, buf_bs)

        a_gain = float(getattr(self.args, "dam_alpha_gain", 0.7))
        b_gain = float(getattr(self.args, "dam_beta_gain", 0.7))
        a_gain = max(0.0, min(1.0, a_gain))
        b_gain = max(0.0, min(1.0, b_gain))

        w_old = (1.0 - a_gain) * w_old_anchor + a_gain * float(w_old_pred.item())
        w_old = max(0.0, min(1.0, w_old))
        w_new = 1.0 - w_old

        beta = 1.0 + b_gain * (float(beta_pred.item()) - 1.0)
        beta = max(0.0, beta)

        return w_old, w_new, beta, w_old_anchor, w_new_anchor

    def _blend_weights_tensor(self, w_old_pred_t, beta_pred_t, real_bs: int, buf_bs: int):
        """
        Differentiable blend for meta-step (keeps grad to controller).
        """
        w_old_anchor, _ = self._anchor_weights(real_bs, buf_bs)
        w_old_anchor_t = torch.tensor(w_old_anchor, device=self.device, dtype=w_old_pred_t.dtype)

        a_gain = float(getattr(self.args, "dam_alpha_gain", 0.7))
        b_gain = float(getattr(self.args, "dam_beta_gain", 0.7))
        a_gain = max(0.0, min(1.0, a_gain))
        b_gain = max(0.0, min(1.0, b_gain))

        w_old = (1.0 - a_gain) * w_old_anchor_t + a_gain * w_old_pred_t
        w_old = w_old.clamp(0.0, 1.0)
        w_new = 1.0 - w_old

        beta = 1.0 + b_gain * (beta_pred_t - 1.0)
        beta = F.relu(beta)  # keep non-negative with gradient

        return w_old, w_new, beta, w_old_anchor

    def end_task(self, dataset):
        self.current_task_id += 1
        super().end_task(dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        real_bs = inputs.size(0)

        if getattr(self.args, "meta_interval_examples", 0) and self.args.meta_interval_examples > 0:
            self.meta_token_examples += real_bs

        # ------------------------------------------------------------
        # 1) theta-step: update backbone
        # ------------------------------------------------------------
        self.net.train()
        self.controller.eval()
        self.opt.zero_grad()

        # forward new
        if self.has_classifier:
            feats_new = self._get_features(inputs)
            out_new = self.net.classifier(feats_new)
        else:
            feats_new = None
            out_new = self.net(inputs)
        loss_new = self.loss(out_new, labels)

        if self.buffer.is_empty():
            loss_new.backward()
            if getattr(self.args, "dam_clip_grad", 0.0) and self.args.dam_clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.dam_clip_grad)
            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_bs])
            return float(loss_new.item())

        # replay batch
        buf_inputs, buf_labels = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform, device=self.device
        )
        buf_bs = buf_inputs.size(0)

        if self.has_classifier:
            feats_old = self._get_features(buf_inputs)
            out_old = self.net.classifier(feats_old)
        else:
            feats_old = None
            out_old = self.net(buf_inputs)
        loss_old = self.loss(out_old, buf_labels)

        # ctx (+ EMA)
        ctx, mu_old, mu_new, _, _, _ = self._build_ctx(
            loss_new, loss_old,
            feats_new if feats_new is not None else out_new.detach(),
            feats_old if feats_old is not None else out_old.detach(),
        )

        beta_ema = float(getattr(self.args, "ctx_ema_beta", 0.9))
        if beta_ema > 0.0:
            if self.ctx_ema is None:
                self.ctx_ema = ctx.detach()
            else:
                self.ctx_ema = beta_ema * self.ctx_ema + (1.0 - beta_ema) * ctx.detach()
            ctx_used = self.ctx_ema
        else:
            ctx_used = ctx

        # controller outputs (no grad)
        with torch.no_grad():
            w_old_pred, _, beta_pred, _, _ = self.controller(ctx_used, mu_old, mu_new)

        w_old, w_new, beta, w_old_anchor, w_new_anchor = self._blend_weights_no_grad(
            w_old_pred, beta_pred, real_bs=real_bs, buf_bs=buf_bs
        )

        # losses
        loss_total = w_new * loss_new + (w_old * beta) * loss_old
        loss_anchor = (w_new_anchor * loss_new) + (w_old_anchor * loss_old)

        params = [p for p in self.net.parameters() if p.requires_grad]

        g_anchor_list = torch.autograd.grad(loss_anchor, params, retain_graph=True, allow_unused=True)
        g_anchor = _flat_from_grads(g_anchor_list, params, device=self.device)

        loss_total.backward()
        g_total = _flat_from_param_grads(params, device=self.device)

        # handcuff unless warmup epoch
        warmup_epochs = int(getattr(self.args, "dam_warmup_epochs", 1) or 0)
        use_cone = (epoch is None) or (epoch >= warmup_epochs)
        if use_cone:
            cos_min = float(getattr(self.args, "dam_cone_cos", 0.0))
            norm_eps = float(getattr(self.args, "dam_cone_norm_eps", 0.3))
            if cos_min > -1.0:
                g_safe = _project_to_cone_and_cap(g_total, g_anchor, cos_min=cos_min, norm_eps=norm_eps)
            else:
                g_safe = _project_to_cone_and_cap(g_total, g_anchor, cos_min=-2.0, norm_eps=norm_eps)
            _set_param_grads_from_flat(params, g_safe, device=self.device)

        if getattr(self.args, "dam_clip_grad", 0.0) and self.args.dam_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.dam_clip_grad)

        self.opt.step()

        # store batch
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_bs])

        # ------------------------------------------------------------
        # 2) phi-step: meta-update controller
        # ------------------------------------------------------------
        trigger_meta = False
        if getattr(self.args, "meta_interval_examples", 0) and self.args.meta_interval_examples > 0:
            if self.meta_token_examples >= self.args.meta_interval_examples:
                trigger_meta = True
                self.meta_token_examples = 0
        else:
            if (self.global_step % int(getattr(self.args, "meta_interval", 50))) == 0:
                trigger_meta = True

        if trigger_meta:
            self.net.eval()
            self.controller.train()
            self.opt_cont.zero_grad()

            # meta new
            if self.has_classifier:
                m_feats_new = self._get_features(inputs)
                m_out_new = self.net.classifier(m_feats_new)
            else:
                m_feats_new = None
                m_out_new = self.net(inputs)
            m_loss_new = self.loss(m_out_new, labels)

            # meta old
            m_buf_inputs, m_buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            if self.has_classifier:
                m_feats_old = self._get_features(m_buf_inputs)
                m_out_old = self.net.classifier(m_feats_old)
            else:
                m_feats_old = None
                m_out_old = self.net(m_buf_inputs)
            m_loss_old = self.loss(m_out_old, m_buf_labels)

            # ctx on meta batch (no EMA)
            ctx_m, mu_old_m, mu_new_m, n_old, n_new, _ = self._build_ctx(
                m_loss_new, m_loss_old,
                m_feats_new if m_feats_new is not None else m_out_new.detach(),
                m_feats_old if m_feats_old is not None else m_out_old.detach(),
            )

            w_old_p, _, beta_p, p_stable_m, _ = self.controller(ctx_m, mu_old_m, mu_new_m)
            w_old_m, w_new_m, beta_m, _ = self._blend_weights_tensor(
                w_old_p, beta_p, real_bs=inputs.size(0), buf_bs=m_buf_inputs.size(0)
            )

            # grad-balance surrogate (norms are detached)
            prod_old = (w_old_m * beta_m) * n_old
            prod_new = (w_new_m) * n_new
            margin = float(getattr(self.args, "meta_margin", 0.0))
            gb = F.relu(prod_new - prod_old + margin) ** 2

            w_reg = float(getattr(self.args, "w_reg_strength", 0.01))
            b_reg = float(getattr(self.args, "beta_reg_strength", 0.01))
            p_reg = w_reg * (p_stable_m - 0.5) ** 2
            beta_reg = b_reg * (beta_m - 1.0) ** 2

            coef = float(getattr(self.args, "meta_grad_balance_coef", 0.5))
            meta_loss = coef * gb + p_reg + beta_reg
            meta_loss.backward()
            self.opt_cont.step()

        return float(loss_total.item())
