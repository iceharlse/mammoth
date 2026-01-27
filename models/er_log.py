"""
Experience Replay (ER) with lightweight logging for "spectral collapse mechanism" plots.

This model keeps ER training identical to the vanilla ER implementation, but additionally:
- builds a *fixed per-task probe set* (old-task probe) during training from not-augmented inputs
- at each end_task, computes:
    (1) old-task probe accuracy (current head) per probe task and old-task average
    (2) stage-wise spectral metrics on old-task probe features:
        - Participation Ratio (PR)
        - Effective Rank (eRank)
        - Normalized Rank (nRank = eRank / dim)

ResNet-18 stages correspond to:
- Stage 1: layer1 (conv2_x)
- Stage 2: layer2 (conv3_x)
- Stage 3: layer3 (conv4_x)
- Stage 4: layer4 (conv5_x)

Outputs:
- A CSV file (append-only) saved to <log_dir>/<NAME>_spectral_probe.csv
  Columns:
    method,seed,after_task,probe_task,stage,n,dim,PR,eRank,nRank,probe_acc

Notes:
- probe_task = -1 denotes old-task average accuracy (tasks < after_task)
- probe_task = -2 denotes all-seen average accuracy (tasks <= after_task)
- stage = 'acc' rows carry probe_acc; PR/eRank/nRank fields are empty

"""

import os
import csv
from typing import Dict, List, Optional, Tuple

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class ErLog(ContinualModel):
    """Continual learning via Experience Replay, with spectral-collapse logging."""
    NAME = 'er_log'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        # Logging / probe collection knobs (safe defaults)
        parser.add_argument('--probe_size', type=int, default=512,
                            help='Number of samples per task in the fixed old-task probe set.')
        parser.add_argument('--probe_batch_size', type=int, default=256,
                            help='Batch size for probe forward passes (end_task evaluation).')
        parser.add_argument('--spectral_log_path', type=str, default='',
                            help='Optional path to CSV. If empty, uses <log_dir>/er_log_spectral_probe.csv')
        parser.add_argument('--spectral_log_disable', action='store_true',
                            help='Disable spectral/probe logging (training remains ER).')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # Fixed per-task probe sets: task_id -> (x_cpu_uint8/float, y_cpu_long)
        self._probe_x: Dict[int, torch.Tensor] = {}
        self._probe_y: Dict[int, torch.Tensor] = {}
        self._last_task_seen: Optional[int] = None

        # Prepare CSV writer
        self._csv_path = None
        self._csv_header = [
            'method', 'seed', 'after_task', 'probe_task', 'stage',
            'n', 'dim', 'PR', 'eRank', 'nRank', 'probe_acc'
        ]
        if not getattr(self.args, 'spectral_log_disable', False):
            self._csv_path = self._resolve_csv_path()
            self._ensure_csv_header()

    # -----------------------
    # Helpers: task + paths
    # -----------------------
    def _get_current_task_id(self) -> Optional[int]:
        """
        Try common Mammoth attributes to infer current task id.
        Falls back to last_task_seen.
        """
        for attr in ('current_task', 'task', '_current_task', 'cur_task', 't'):
            if hasattr(self, attr):
                v = getattr(self, attr)
                if isinstance(v, int):
                    return v
        return self._last_task_seen

    def _resolve_csv_path(self) -> str:
        """
        Pick a reasonable log directory without assuming a specific Mammoth version.
        """
        # User override
        if hasattr(self.args, 'spectral_log_path') and isinstance(self.args.spectral_log_path, str) and self.args.spectral_log_path.strip():
            path = self.args.spectral_log_path.strip()
            base = os.path.dirname(path)
            if base:
                os.makedirs(base, exist_ok=True)
            return path

        # Try common directories
        for attr in ('output_dir', 'log_dir', 'results_dir', 'save_path', 'experiment_path'):
            if hasattr(self.args, attr):
                p = getattr(self.args, attr)
                if isinstance(p, str) and p.strip():
                    os.makedirs(p, exist_ok=True)
                    return os.path.join(p, f'{self.NAME}_spectral_probe.csv')

        return os.path.join(os.getcwd(), f'{self.NAME}_spectral_probe.csv')

    def _ensure_csv_header(self) -> None:
        assert self._csv_path is not None
        is_new = not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0
        if is_new:
            with open(self._csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(self._csv_header)

    def _csv_append_rows(self, rows: List[List]) -> None:
        if self._csv_path is None:
            return
        with open(self._csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerows(rows)

    # -----------------------
    # Probe set maintenance
    # -----------------------
    def _maybe_add_to_probe(self, task_id: int, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Maintain a fixed per-task probe set of up to probe_size samples.
        Uses FIFO to keep earliest samples (stable across time).
        Stores on CPU to avoid GPU memory.
        """
        if getattr(self.args, 'spectral_log_disable', False):
            return
        target = int(getattr(self.args, 'probe_size', 512))
        if target <= 0:
            return

        # Ensure CPU tensors, detach
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
        """
        feat: [n, d] float tensor on CPU or GPU
        Returns: (PR, eRank)
        PR = (sum λ)^2 / (sum λ^2)
        eRank = exp(H(p)), p = λ / sum λ
        where λ are eigenvalues of covariance (or singular values squared).
        """
        n, d = feat.shape
        if n < 2:
            return float('nan'), float('nan')

        # Center
        x = feat - feat.mean(dim=0, keepdim=True)

        # Singular values (no need to form covariance explicitly)
        # lam = (s^2) / (n-1)
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
        """
        Return mapping stage_name -> module for ResNet18 stages.
        Tries to support wrappers where backbone is stored as .backbone.
        """
        net = self.net
        if all(hasattr(net, k) for k in ('layer1', 'layer2', 'layer3', 'layer4')):
            return {
                'stage1': net.layer1,
                'stage2': net.layer2,
                'stage3': net.layer3,
                'stage4': net.layer4,
            }
        if hasattr(net, 'backbone') and all(hasattr(net.backbone, k) for k in ('layer1', 'layer2', 'layer3', 'layer4')):
            bb = net.backbone
            return {
                'stage1': bb.layer1,
                'stage2': bb.layer2,
                'stage3': bb.layer3,
                'stage4': bb.layer4,
            }
        # If not found, return empty (logging will be skipped gracefully)
        return {}

    @torch.no_grad()
    def _forward_collect_stage_feats(self, x: torch.Tensor, batch_size: int) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Forward x through net and collect stage outputs via hooks.
        Returns:
          feats: dict stage -> [n, dim] GAP features on CPU
          acc: accuracy (%) using current head on x/y must be computed outside (needs labels)
        """
        stages = self._get_resnet_stages()
        if not stages:
            return {}, float('nan')

        acts: Dict[str, List[torch.Tensor]] = {k: [] for k in stages.keys()}
        handles = []

        def make_hook(name: str):
            def _hook(_m, _inp, out):
                # out: [b, c, h, w]
                if isinstance(out, torch.Tensor):
                    # GAP to [b, c]
                    f = out.mean(dim=(2, 3))
                    acts[name].append(f.detach().to('cpu'))
            return _hook

        for name, mod in stages.items():
            handles.append(mod.register_forward_hook(make_hook(name)))

        # Forward in chunks
        self.net.eval()
        n = x.shape[0]
        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size].to(self.device, non_blocking=True)
            _ = self.net(xb)

        for h in handles:
            h.remove()

        feats = {k: torch.cat(v, dim=0) if len(v) else torch.empty(0) for k, v in acts.items()}
        return feats, float('nan')

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
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        if total == 0:
            return float('nan')
        return 100.0 * correct / total

    # -----------------------
    # ER training (unchanged)
    # -----------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        real_batch_size = inputs.shape[0]

        # Track task id + build fixed probe set (uses current-stream samples only)
        t = self._get_current_task_id()
        if isinstance(t, int):
            self._last_task_seen = t
            # NOTE: labels[:real_batch_size] are the current stream labels even after buffer concat
            self._maybe_add_to_probe(t, not_aug_inputs, labels[:real_batch_size])

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    # -----------------------
    # Task boundary logging
    # -----------------------
    def end_task(self, dataset=None):
        """
        Called by the training loop at the end of each task.
        Computes per-task probe acc + stage-wise spectral metrics on the fixed old-task probe set.
        """
        if getattr(self.args, 'spectral_log_disable', False) or self._csv_path is None:
            return

        after_task = self._get_current_task_id()
        if after_task is None:
            # cannot determine task id; skip
            return
        if after_task not in self._probe_x:
            # no probe samples collected; skip
            return

        batch_size = int(getattr(self.args, 'probe_batch_size', 256))
        seed = getattr(self.args, 'seed', '')
        method = self.NAME

        # Compute per-probe-task accuracy + stage metrics
        rows: List[List] = []
        probe_tasks = sorted([k for k in self._probe_x.keys() if k <= after_task])

        # Per task acc
        acc_per_task: Dict[int, float] = {}
        for pt in probe_tasks:
            x = self._probe_x[pt].float()
            y = self._probe_y[pt].long()
            acc = self._compute_probe_acc(x, y, batch_size=batch_size)
            acc_per_task[pt] = acc
            rows.append([method, seed, after_task, pt, 'acc', x.shape[0], '', '', '', '', acc])

        # Old-task average (tasks < after_task) and all-seen average (tasks <= after_task)
        if after_task > 0:
            old_tasks = [k for k in probe_tasks if k < after_task]
            if old_tasks:
                old_avg = sum(acc_per_task[k] for k in old_tasks) / len(old_tasks)
                rows.append([method, seed, after_task, -1, 'acc_oldavg', '', '', '', '', '', old_avg])
        all_avg = sum(acc_per_task[k] for k in probe_tasks) / len(probe_tasks) if probe_tasks else float('nan')
        rows.append([method, seed, after_task, -2, 'acc_allavg', '', '', '', '', '', all_avg])

        # Stage-wise metrics per probe_task
        for pt in probe_tasks:
            x = self._probe_x[pt].float()
            y = self._probe_y[pt].long()

            feats_by_stage, _ = self._forward_collect_stage_feats(x, batch_size=batch_size)
            if not feats_by_stage:
                continue

            for stage, feat in feats_by_stage.items():
                if feat.numel() == 0:
                    continue
                n, dim = feat.shape
                pr, er = self._participation_ratio_and_erank(feat)
                nr = (er / float(dim)) if (dim and er == er) else float('nan')  # er==er checks nan
                rows.append([method, seed, after_task, pt, stage, n, dim, pr, er, nr, acc_per_task.get(pt, '')])

        self._csv_append_rows(rows)
