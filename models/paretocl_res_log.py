import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


class HyperNet(nn.Module):
    """
    HyperNetwork for ParetoCL.
    Generates the classifier weights conditioned on the preference vector alpha.
    Ref: ParetoCL paper.
    """
    def __init__(self, feature_dim: int, total_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.total_classes = total_classes
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head_generator = nn.Linear(hidden_dim, (feature_dim + 1) * total_classes)

        # Good init for HyperNet output layer
        nn.init.normal_(self.head_generator.weight, std=0.01)
        nn.init.constant_(self.head_generator.bias, 0.0)

    def forward(self, alpha: torch.Tensor, n_classes: int):
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        device = next(self.parameters()).device
        alpha = alpha.to(device)

        embedding = self.mlp(alpha)
        raw_head = self.head_generator(embedding)

        raw_head = raw_head.view(-1, self.total_classes, self.feature_dim + 1)

        weights = raw_head[..., : self.feature_dim]
        biases = raw_head[..., -1]

        weights = weights[:, :n_classes, :]
        biases = biases[:, :n_classes]

        if weights.size(0) == 1:
            return weights.squeeze(0), biases.squeeze(0)
        return weights, biases


@register_model("paretocl_res_log")
class ParetoCL_res_log(ContinualModel):
    """
    ParetoCL (ResNet backbone + HyperNet) with the SAME spectral/probe logging mouthpiece as er_log / er_dam_log.

    What gets logged at each end_task:
      - Old-task probe accuracy (%): computed on a fixed per-task probe set.
      - Stage-wise spectral metrics on the same probe set:
          PR, eRank, nRank = eRank / dim
        for ResNet-18 stages:
          stage1=layer1(conv2_x), stage2=layer2(conv3_x), stage3=layer3(conv4_x), stage4=layer4(conv5_x)

    CSV columns:
      method, seed, after_task, probe_task, stage, n, dim, PR, eRank, nRank, probe_acc

    Notes on probe accuracy for ParetoCL:
      - Because logits depend on preference alpha, we use a deterministic "mean preference" alpha
        derived from the Dirichlet concentration: alpha_mean = conc / sum(conc).
      - This makes the probe curve stable and comparable across runs.
      - If you prefer the stochastic min-entropy inference path, set --paretocl_probe_use_inference 1.
    """
    NAME = "paretocl_res_log"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)

        # Default Hyperparams from paper
        parser.add_argument("--hyper_hidden_dim", type=int, default=128)
        parser.add_argument("--paretocl_dirichlet_alpha_stability", type=float, default=1.0)
        parser.add_argument("--paretocl_dirichlet_alpha_plasticity", type=float, default=1.0)
        parser.add_argument("--pref_samples", type=int, default=20)  # Inference samples

        # Existing debug log switch (kept)
        parser.add_argument("--save_paretocl_log", type=int, default=0)

        # NEW: spectral/probe logging (same mouthpiece)
        parser.add_argument('--probe_size', type=int, default=512,
                            help='Number of samples per task in the fixed old-task probe set.')
        parser.add_argument('--probe_batch_size', type=int, default=256,
                            help='Batch size for probe forward passes (end_task evaluation).')
        parser.add_argument('--spectral_log_path', type=str, default='',
                            help='Optional path to CSV. If empty, uses <log_dir>/paretocl_res_log_spectral_probe.csv')
        parser.add_argument('--spectral_log_disable', action='store_true',
                            help='Disable spectral/probe logging.')

        # Probe-accuracy mode for ParetoCL
        parser.add_argument('--paretocl_probe_use_inference', type=int, default=0,
                            help='0: deterministic mean-preference alpha; 1: use ParetoCL inference (min-entropy over K alphas).')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # Feature dim for backbone features
        self.feature_dim = getattr(self.net, "feature_dim", None)
        if self.feature_dim is None:
            # ResNet18 typically 512 after GAP
            self.feature_dim = 512

        self.hypernet = HyperNet(
            feature_dim=self.feature_dim,
            total_classes=self.num_classes,
            hidden_dim=self.args.hyper_hidden_dim,
        )

        conc_stab = float(self.args.paretocl_dirichlet_alpha_stability)
        conc_plas = float(self.args.paretocl_dirichlet_alpha_plasticity)
        self.register_buffer("dirichlet_concentration", torch.tensor([conc_stab, conc_plas], dtype=torch.float))

        self.pref_samples = int(self.args.pref_samples)

        if hasattr(self, "device"):
            self.hypernet.to(self.device)
            self.dirichlet_concentration = self.dirichlet_concentration.to(self.device)

        # Joint optimization
        self.opt = self.get_optimizer(list(self.net.parameters()) + list(self.hypernet.parameters()))

        # --- existing defect log (kept as-is)
        self.do_log = (args.save_paretocl_log == 1)
        self.inference_recorder = []
        self.log_save_path = "paretocl_defect_proof"
        self.log_file_path = None
        if self.do_log:
            if not os.path.exists(self.log_save_path):
                os.makedirs(self.log_save_path, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_file_path = f"{self.log_save_path}/defect_log_{timestamp}.csv"
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'sample_idx', 'min_entropy', 'selected_is_correct',
                                 'oracle_exists', 'regret', 'true_label', 'pred_label'])

        # --- NEW: per-task probe sets on CPU
        self.current_task_id = 0
        self._probe_x: Dict[int, torch.Tensor] = {}
        self._probe_y: Dict[int, torch.Tensor] = {}

        # --- NEW: CSV mouthpiece
        self._csv_path: Optional[str] = None
        self._csv_header = ['method', 'seed', 'after_task', 'probe_task', 'stage',
                            'n', 'dim', 'PR', 'eRank', 'nRank', 'probe_acc']
        if not getattr(self.args, 'spectral_log_disable', False):
            self._csv_path = self._resolve_csv_path()
            self._ensure_csv_header()

    # -----------------------
    # Helpers
    # -----------------------
    def _get_device(self):
        return next(self.parameters()).device

    def _get_n_classes(self) -> int:
        if hasattr(self, "n_seen_classes") and self.n_seen_classes > 0:
            return self.n_seen_classes
        return self.num_classes

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, returnt="features")

    def _logits_with_alpha(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        device = self._get_device()
        x = x.to(device)
        alpha = alpha.to(device)
        n_classes = self._get_n_classes()
        feats = self._features(x)
        weights, biases = self.hypernet(alpha, n_classes)
        return F.linear(feats, weights, biases)

    def flush_stats_to_disk(self):
        if not self.do_log or len(self.inference_recorder) == 0:
            return
        try:
            with open(self.log_file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'sample_idx', 'min_entropy',
                                                      'selected_is_correct', 'oracle_exists', 'regret',
                                                      'true_label', 'pred_label'])
                for row in self.inference_recorder:
                    writer.writerow(row)
            self.inference_recorder = []
        except Exception as e:
            print(f"[Log Error] {e}")

    # -----------------------
    # NEW: probe set (fixed FIFO)
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
    # NEW: CSV mouthpiece
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
    # NEW: spectral metrics
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
            _ = self._features(xb)  # forward through backbone to trigger hooks

        for h in handles:
            h.remove()

        return {k: torch.cat(v, dim=0) if len(v) else torch.empty(0) for k, v in acts.items()}

    # -----------------------
    # Probe accuracy for ParetoCL
    # -----------------------
    @torch.no_grad()
    def _compute_probe_acc(self, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
        self.eval()
        device = self._get_device()
        n = x.shape[0]
        correct = 0
        total = 0

        use_infer = int(getattr(self.args, "paretocl_probe_use_inference", 0)) == 1

        # Deterministic mean preference
        conc = self.dirichlet_concentration.detach().float().to(device)
        alpha_mean = conc / (conc.sum() + 1e-12)

        for i in range(0, n, batch_size):
            xb = x[i:i + batch_size].to(device, non_blocking=True)
            yb = y[i:i + batch_size].to(device, non_blocking=True)

            if use_infer:
                # Use ParetoCL inference: sample K alphas and choose min-entropy
                logits = self.forward(xb)  # eval mode triggers inference path
            else:
                logits = self._logits_with_alpha(xb, alpha_mean)

            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())

        if total == 0:
            return float('nan')
        return 100.0 * correct / total

    # -----------------------
    # Forward (kept compatible)
    # -----------------------
    def forward(self, x: torch.Tensor, alpha: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        device = self._get_device()
        x = x.to(device)

        if alpha is not None:
            return self._logits_with_alpha(x, alpha)

        if self.training and not self.do_log:
            alpha_default = torch.tensor([0.5, 0.5], device=device)
            return self._logits_with_alpha(x, alpha_default)

        # --------- Inference / Defect Probe ---------
        feats = self._features(x)
        n_classes = self._get_n_classes()
        B = feats.size(0)

        dirichlet = Dirichlet(self.dirichlet_concentration)
        alphas = dirichlet.sample((self.pref_samples,)).to(device)

        logits_list = []
        for k in range(self.pref_samples):
            alpha_k = alphas[k]
            W_k, b_k = self.hypernet(alpha_k, n_classes)
            logits_k = F.linear(feats, W_k, b_k)
            logits_list.append(logits_k.unsqueeze(0))

        logits_stack = torch.cat(logits_list, dim=0)  # (K, B, C)

        probs = F.softmax(logits_stack, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (K, B)

        best_k = entropy.argmin(dim=0)  # (B,)

        logits_stack_perm = logits_stack.permute(1, 0, 2)  # (B, K, C)
        batch_indices = torch.arange(B, device=device)
        chosen_logits = logits_stack_perm[batch_indices, best_k, :]

        # optional existing debug logging
        if self.do_log and labels is not None:
            labels = labels.to(device)
            preds_selected = chosen_logits.argmax(dim=1)
            selected_is_correct = (preds_selected == labels).int().cpu()
            all_preds = logits_stack.argmax(dim=2)
            oracle_check = (all_preds == labels.unsqueeze(0)).int()
            oracle_exists = oracle_check.max(dim=0)[0].cpu()
            regret = oracle_exists - selected_is_correct
            min_entropy_vals = entropy.min(dim=0)[0].detach().cpu()

            current_time = time.time()
            for i in range(B):
                self.inference_recorder.append({
                    'timestamp': current_time,
                    'sample_idx': i,
                    'min_entropy': min_entropy_vals[i].item(),
                    'selected_is_correct': selected_is_correct[i].item(),
                    'oracle_exists': oracle_exists[i].item(),
                    'regret': regret[i].item(),
                    'true_label': labels[i].item(),
                    'pred_label': preds_selected[i].item()
                })
            self.flush_stats_to_disk()

        return chosen_logits

    # -----------------------
    # Observe (training) + probe capture
    # -----------------------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        # capture probe BEFORE moving tensors to device (keeps CPU copy)
        self._maybe_add_to_probe(self.current_task_id, not_aug_inputs, labels)

        self.opt.zero_grad()
        device = self._get_device()
        inputs = inputs.to(device)
        labels = labels.to(device)
        not_aug_inputs = not_aug_inputs.to(device)

        dirichlet = Dirichlet(self.dirichlet_concentration)
        alpha = dirichlet.sample().to(device)
        alpha_stab, alpha_plas = alpha[0], alpha[1]

        logits_new = self._logits_with_alpha(inputs, alpha)
        loss_new = self.loss(logits_new, labels)

        loss_replay = torch.tensor(0.0, device=device)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=device
            )
            logits_replay = self._logits_with_alpha(buf_inputs, alpha)
            loss_replay = self.loss(logits_replay, buf_labels)

        loss = alpha_stab * loss_replay + alpha_plas * loss_new
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        return loss.item()

    # -----------------------
    # NEW: end_task logging (same mouthpiece)
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

        # per-probe-task accuracy
        acc_per_task: Dict[int, float] = {}
        for pt in probe_tasks:
            x = self._probe_x[pt].float()
            y = self._probe_y[pt].long()
            acc = self._compute_probe_acc(x, y, batch_size=batch_size)
            acc_per_task[pt] = acc
            rows.append([method, seed, finished_task, pt, 'acc', x.shape[0], '', '', '', '', acc])

        # oldavg and allavg
        if finished_task > 0:
            old_tasks = [k for k in probe_tasks if k < finished_task]
            if old_tasks:
                old_avg = sum(acc_per_task[k] for k in old_tasks) / len(old_tasks)
                rows.append([method, seed, finished_task, -1, 'acc_oldavg', '', '', '', '', '', old_avg])

        all_avg = sum(acc_per_task[k] for k in probe_tasks) / len(probe_tasks) if probe_tasks else float('nan')
        rows.append([method, seed, finished_task, -2, 'acc_allavg', '', '', '', '', '', all_avg])

        # stage metrics per probe_task
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

    def end_task(self, dataset):
        finished_task = int(self.current_task_id)
        self._log_end_task(finished_task)
        self.current_task_id += 1
        super().end_task(dataset)
