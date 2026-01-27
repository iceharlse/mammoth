"""
ER + DAM (Alpha+Beta controller + anchor cone) with spectral/probe logging.

This is a drop-in variant of the user's original `er_dam.py` that keeps training behavior intact,
but adds the SAME logging "mouthpiece" as `er_log` so you can draw the mechanism figure:

At each end_task (i.e., after finishing task t), it appends to a CSV:
- Old-task probe accuracy (per probe_task) + oldavg/allavg
- Stage-wise spectral metrics (PR / eRank / nRank) on a fixed per-task probe set
  for ResNet-18 stages:
    stage1 = layer1 (conv2_x)
    stage2 = layer2 (conv3_x)
    stage3 = layer3 (conv4_x)
    stage4 = layer4 (conv5_x)

CSV columns:
  method, seed, after_task, probe_task, stage, n, dim, PR, eRank, nRank, probe_acc

Special rows:
- stage == 'acc'        : per-task probe accuracy only (PR/eRank/nRank empty)
- stage == 'acc_oldavg' : probe_task = -1, old tasks (< after_task) average accuracy
- stage == 'acc_allavg' : probe_task = -2, all seen tasks (<= after_task) average accuracy

Probe set:
- Fixed FIFO subset of NOT-augmented stream samples per task: --probe_size (default 512)
- Stored on CPU during training to avoid GPU memory impact

Notes:
- Logging is evaluation-only (torch.no_grad) and should not affect training gradients.
- If you want to disable all logging: --spectral_log_disable
"""

import os
import csv
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


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

        w_old = p_stable * self.alpha_max + (1.0 - p_stable) * self.alpha_min
        w_new = 1.0 - w_old

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


@register_model("er_dam_log")
class ERDamAlphaBetaLog(ContinualModel):
    """
    ER + DAM (alpha+beta controller + anchor-gradient cone constraint) + logging.
    """
    NAME = "er_dam_log"
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

        # original logging
        parser.add_argument("--log_interval", type=int, default=100)

        # NEW: spectral/probe logging (same as er_log)
        parser.add_argument('--probe_size', type=int, default=512,
                            help='Number of samples per task in the fixed old-task probe set.')
        parser.add_argument('--probe_batch_size', type=int, default=256,
                            help='Batch size for probe forward passes (end_task evaluation).')
        parser.add_argument('--spectral_log_path', type=str, default='',
                            help='Optional path to CSV. If empty, uses <log_dir>/er_dam_log_spectral_probe.csv')
        parser.add_argument('--spectral_log_disable', action='store_true',
                            help='Disable spectral/probe logging.')

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

        # explicit optimizers (same as original)
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

        # --- probe storage (task -> tensors on CPU)
        self._probe_x: Dict[int, torch.Tensor] = {}
        self._probe_y: Dict[int, torch.Tensor] = {}

        # --- CSV setup
        self._csv_path: Optional[str] = None
        self._csv_header = [
            'method', 'seed', 'after_task', 'probe_task', 'stage',
            'n', 'dim', 'PR', 'eRank', 'nRank', 'probe_acc'
        ]
        if not getattr(self.args, 'spectral_log_disable', False):
            self._csv_path = self._resolve_csv_path()
            self._ensure_csv_header()

    # -----------------------
    # Logging path helpers
    # -----------------------
    def _resolve_csv_path(self) -> str:
        if hasattr(self.args, 'spectral_log_path') and isinstance(self.args.spectral_log_path, str) and self.args.spectral_log_path.strip():
            path = self.args.spectral_log_path.strip()
            base = os.path.dirname(path)
            if base:
                os.makedirs(base, exist_ok=True)
            return path

        for attr in ('output_dir', 'log_dir', 'results_dir', 'save_path', 'experiment_path'):
            if hasattr(self.args, attr):
                p = getattr(self.args, attr)
                if isinstance(p, str) and p.strip():
                    os.makedirs(p, exist_ok=True)
                    return os.path.join(p, f'{self.NAME}_spectral_probe.csv')

        return os.path.join(os.getcwd(), f'{self.NAME}_spectral_probe.csv')

    def _ensure_csv_header(self) -> None:
        assert self._csv_path is not None
        is_new = (not os.path.exists(self._csv_path)) or (os.path.getsize(self._csv_path) == 0)
        if is_new:
            with open(self._csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(self._csv_header)

    def _csv_append_rows(self, rows: List[List]) -> None:
        if self._csv_path is None:
            return
        with open(self._csv_path, 'a', newline='') as f:
            csv.writer(f).writerows(rows)

    # -----------------------
    # Probe set (fixed FIFO)
    # -----------------------
    def _maybe_add_to_probe(self, task_id: int, x: torch.Tensor, y: torch.Tensor) -> None:
        if getattr(self.args, 'spectral_log_disable', False):
            return
        target = int(getattr(self.args, 'probe_size', 512))
        if target <= 0:
            return

        x_cpu = x.detach().to('cpu')
        y_cpu = y.detach().to('cpu')

        if task_id not in self._probe_x:
            self._probe_x[task_id] = x_cpu[:0].clone()
            self._probe_y[task_id] = y_cpu[:0].clone()

        cur_n = int(self._probe_x[task_id].shape[0])
        if cur_n >= target:
            return

        take = min(target - cur_n, x_cpu.shape[0])
        if take <= 0:
            return

        self._probe_x[task_id] = torch.cat([self._probe_x[task_id], x_cpu[:take]], dim=0)
        self._probe_y[task_id] = torch.cat([self._probe_y[task_id], y_cpu[:take]], dim=0)

    # -----------------------
    # Spectral metrics
    # -----------------------
    @staticmethod
    def _participation_ratio_and_erank(feat: torch.Tensor) -> Tuple[float, float]:
        n, d = feat.shape
        if n < 2:
            return float('nan'), float('nan')

        x = feat - feat.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(x)
        lam = (s ** 2) / max(n - 1, 1)
        lam = torch.clamp(lam, min=1e-12)

        s1 = lam.sum()
        pr = ((s1 ** 2) / (lam.pow(2).sum())).item()

        p = lam / s1
        h = (-p * torch.log(p)).sum()
        erank = torch.exp(h).item()
        return pr, erank

    def _get_resnet_stages(self):
        net = self.net
        if all(hasattr(net, k) for k in ('layer1', 'layer2', 'layer3', 'layer4')):
            return {'stage1': net.layer1, 'stage2': net.layer2, 'stage3': net.layer3, 'stage4': net.layer4}
        if hasattr(net, 'backbone') and all(hasattr(net.backbone, k) for k in ('layer1', 'layer2', 'layer3', 'layer4')):
            bb = net.backbone
            return {'stage1': bb.layer1, 'stage2': bb.layer2, 'stage3': bb.layer3, 'stage4': bb.layer4}
        return {}

    @torch.no_grad()
    def _forward_collect_stage_feats(self, x: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        stages = self._get_resnet_stages()
        if not stages:
            return {}

        acts: Dict[str, List[torch.Tensor]] = {k: [] for k in stages.keys()}
        handles = []

        def make_hook(name: str):
            def _hook(_m, _inp, out):
                if isinstance(out, torch.Tensor):
                    f = out.mean(dim=(2, 3))  # GAP -> [b, c]
                    acts[name].append(f.detach().to('cpu'))
            return _hook

        for name, mod in stages.items():
            handles.append(mod.register_forward_hook(make_hook(name)))

        self.net.eval()
        n = x.shape[0]
        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size].to(self.device, non_blocking=True)
            _ = self.net(xb)  # triggers hooks

        for h in handles:
            h.remove()

        return {k: torch.cat(v, dim=0) if len(v) else torch.empty(0) for k, v in acts.items()}

    @torch.no_grad()
    def _compute_probe_acc(self, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
        self.net.eval()
        n = x.shape[0]
        correct = 0
        total = 0
        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size].to(self.device, non_blocking=True)
            yb = y[i:i + batch_size].to(self.device, non_blocking=True)
            logits = self.net(xb)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        if total == 0:
            return float('nan')
        return 100.0 * correct / total

    # -----------------------
    # Original model helpers
    # -----------------------
    def forward(self, x):
        return self.net(x)

    def _get_features(self, x):
        return self.net(x, returnt="features")

    def _safe_head_params(self):
        if self.has_classifier:
            return list(self.net.classifier.parameters())
        return []

    def _build_ctx(self, loss_new, loss_old, features_new, features_old):
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
        beta = F.relu(beta)

        return w_old, w_new, beta, w_old_anchor

    # -----------------------
    # Logging at task boundary
    # -----------------------
    def _log_end_task(self, finished_task: int) -> None:
        if getattr(self.args, 'spectral_log_disable', False) or self._csv_path is None:
            return
        if finished_task not in self._probe_x:
            return

        batch_size = int(getattr(self.args, 'probe_batch_size', 256))
        seed = getattr(self.args, 'seed', '')
        method = self.NAME

        probe_tasks = sorted([k for k in self._probe_x.keys() if k <= finished_task])
        rows: List[List] = []

        acc_per_task: Dict[int, float] = {}
        for pt in probe_tasks:
            x = self._probe_x[pt].float()
            y = self._probe_y[pt].long()
            acc = self._compute_probe_acc(x, y, batch_size=batch_size)
            acc_per_task[pt] = acc
            rows.append([method, seed, finished_task, pt, 'acc', x.shape[0], '', '', '', '', acc])

        if finished_task > 0:
            old_tasks = [k for k in probe_tasks if k < finished_task]
            if old_tasks:
                old_avg = sum(acc_per_task[k] for k in old_tasks) / len(old_tasks)
                rows.append([method, seed, finished_task, -1, 'acc_oldavg', '', '', '', '', '', old_avg])

        all_avg = sum(acc_per_task[k] for k in probe_tasks) / len(probe_tasks) if probe_tasks else float('nan')
        rows.append([method, seed, finished_task, -2, 'acc_allavg', '', '', '', '', '', all_avg])

        for pt in probe_tasks:
            x = self._probe_x[pt].float()
            feats_by_stage = self._forward_collect_stage_feats(x, batch_size=batch_size)
            if not feats_by_stage:
                continue
            for stage, feat in feats_by_stage.items():
                if feat.numel() == 0:
                    continue
                n, dim = feat.shape
                pr, er = self._participation_ratio_and_erank(feat)
                nr = (er / float(dim)) if (dim and er == er) else float('nan')
                rows.append([method, seed, finished_task, pt, stage, n, dim, pr, er, nr, acc_per_task.get(pt, '')])

        self._csv_append_rows(rows)

    # -----------------------
    # Train step (unchanged + probe capture)
    # -----------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.global_step += 1

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        real_bs = inputs.size(0)

        # collect fixed probe samples from current stream (NOT augmented)
        self._maybe_add_to_probe(self.current_task_id, not_aug_inputs, labels[:real_bs])

        if getattr(self.args, "meta_interval_examples", 0) and self.args.meta_interval_examples > 0:
            self.meta_token_examples += real_bs

        # 1) theta-step: update backbone
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

        loss_total = w_new * loss_new + (w_old * beta) * loss_old
        loss_anchor = (w_new_anchor * loss_new) + (w_old_anchor * loss_old)

        params = [p for p in self.net.parameters() if p.requires_grad]

        g_anchor_list = torch.autograd.grad(loss_anchor, params, retain_graph=True, allow_unused=True)
        g_anchor = _flat_from_grads(g_anchor_list, params, device=self.device)

        loss_total.backward()
        g_total = _flat_from_param_grads(params, device=self.device)

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

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_bs])

        # 2) phi-step: meta-update controller
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

            if self.has_classifier:
                m_feats_new = self._get_features(inputs)
                m_out_new = self.net.classifier(m_feats_new)
            else:
                m_feats_new = None
                m_out_new = self.net(inputs)
            m_loss_new = self.loss(m_out_new, labels)

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

            ctx_m, mu_old_m, mu_new_m, n_old, n_new, _ = self._build_ctx(
                m_loss_new, m_loss_old,
                m_feats_new if m_feats_new is not None else m_out_new.detach(),
                m_feats_old if m_feats_old is not None else m_out_old.detach(),
            )

            w_old_p, _, beta_p, p_stable_m, _ = self.controller(ctx_m, mu_old_m, mu_new_m)
            w_old_m, w_new_m, beta_m, _ = self._blend_weights_tensor(
                w_old_p, beta_p, real_bs=inputs.size(0), buf_bs=m_buf_inputs.size(0)
            )

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

    def end_task(self, dataset):
        """
        Keep original behavior (increment task id at boundary) but log BEFORE increment
        so after_task matches the task that just finished.
        """
        finished_task = int(self.current_task_id)
        self._log_end_task(finished_task)

        self.current_task_id += 1
        super().end_task(dataset)
