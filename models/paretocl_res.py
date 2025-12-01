import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import csv
import os
import time

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models import register_model


class HyperNet(nn.Module):
    """
    HyperNetwork for ParetoCL.
    Generates the classifier weights conditioned on the preference vector alpha.
    [cite_start]Ref: [cite: 487, 491, 533]
    """
    def __init__(self, feature_dim: int, total_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.total_classes = total_classes
        self.hidden_dim = hidden_dim
        
        # [cite_start]MLP with two hidden layers as per paper [cite: 533]
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head_generator = nn.Linear(
            hidden_dim, (feature_dim + 1) * total_classes
        )

        # --- [微修改] 初始化优化 ---
        # HyperNetwork 的输出层初始化非常关键。
        # 我们希望初始生成的分类头接近于随机初始化 (mean=0, std=0.01)，避免训练初期梯度爆炸或消失。
        nn.init.normal_(self.head_generator.weight, std=0.01)
        nn.init.constant_(self.head_generator.bias, 0.0)

    def forward(self, alpha: torch.Tensor, n_classes: int):
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        device = next(self.parameters()).device
        alpha = alpha.to(device)
        
        embedding = self.mlp(alpha)
        raw_head = self.head_generator(embedding)
        
        # Reshape into (Batch, Classes, Feat+Bias)
        raw_head = raw_head.view(-1, self.total_classes, self.feature_dim + 1)
        
        weights = raw_head[..., : self.feature_dim]
        biases = raw_head[..., -1]
        
        # Slice for current seen classes
        weights = weights[:, :n_classes, :]
        biases = biases[:, :n_classes]
        
        if weights.size(0) == 1:
            return weights.squeeze(0), biases.squeeze(0)
        return weights, biases


@register_model("paretocl_res")
class ParetoCL_res(ContinualModel):
    NAME = "paretocl_res"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        # [cite_start]Default Hyperparams from paper [cite: 533]
        parser.add_argument("--hyper_hidden_dim", type=int, default=128)
        parser.add_argument("--paretocl_dirichlet_alpha_stability", type=float, default=1.0)
        parser.add_argument("--paretocl_dirichlet_alpha_plasticity", type=float, default=1.0)
        parser.add_argument("--pref_samples", type=int, default=20) # Inference samples [cite: 534]
        
        # Log switch
        parser.add_argument("--save_paretocl_log", type=int, default=0) 
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.feature_dim = getattr(self.net, "feature_dim", None)
        
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
            
        # [cite_start]Joint optimization [cite: 492]
        self.opt = self.get_optimizer(
            list(self.net.parameters()) + list(self.hypernet.parameters())
        )

        # --- 日志部分 (保留你的调试逻辑) ---
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
                writer.writerow(['timestamp', 'sample_idx', 'min_entropy', 'selected_is_correct', 'oracle_exists', 'regret', 'true_label', 'pred_label'])

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
        
        # [cite_start]Generate weights using HyperNet [cite: 489, 491]
        weights, biases = self.hypernet(alpha, n_classes)
        return F.linear(feats, weights, biases)

    def flush_stats_to_disk(self):
        if not self.do_log or len(self.inference_recorder) == 0:
            return
        try:
            with open(self.log_file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'sample_idx', 'min_entropy', 'selected_is_correct', 'oracle_exists', 'regret', 'true_label', 'pred_label'])
                for row in self.inference_recorder:
                    writer.writerow(row)
            self.inference_recorder = []
        except Exception as e:
            print(f"[Log Error] {e}")

    def forward(self, x: torch.Tensor, alpha: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        device = self._get_device()
        x = x.to(device)

        # Training logic or explicit alpha provided
        if alpha is not None:
            return self._logits_with_alpha(x, alpha)
        
        if self.training and not self.do_log:
            # Standard training forward (usually not used in Mammoth training loop as observe handles it, but good for compatibility)
            alpha_default = torch.tensor([0.5, 0.5], device=device)
            return self._logits_with_alpha(x, alpha_default)

        # [cite_start]--------- Inference / Defect Probe [cite: 495-497] ---------
        feats = self._features(x)
        n_classes = self._get_n_classes()
        B = feats.size(0)

        dirichlet = Dirichlet(self.dirichlet_concentration)
        alphas = dirichlet.sample((self.pref_samples,)).to(device) # Sample K preferences

        # 1. Compute logits for all K alphas
        logits_list = []
        for k in range(self.pref_samples):
            alpha_k = alphas[k]
            W_k, b_k = self.hypernet(alpha_k, n_classes)
            logits_k = F.linear(feats, W_k, b_k)
            logits_list.append(logits_k.unsqueeze(0))

        logits_stack = torch.cat(logits_list, dim=0) # (K, B, C)
        
        # [cite_start]2. Compute Entropy [cite: 497]
        probs = F.softmax(logits_stack, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) # (K, B)
        
        # 3. Min-Entropy Selection
        best_k = entropy.argmin(dim=0) # (B,)
        
        logits_stack_perm = logits_stack.permute(1, 0, 2) # (B, K, C)
        batch_indices = torch.arange(B, device=device)
        chosen_logits = logits_stack_perm[batch_indices, best_k, :]

        # --------- Logging Logic ---------
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

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        device = self._get_device()
        inputs = inputs.to(device)
        labels = labels.to(device)
        not_aug_inputs = not_aug_inputs.to(device)

        # [cite_start]Sample preference for training [cite: 492]
        dirichlet = Dirichlet(self.dirichlet_concentration)
        alpha = dirichlet.sample().to(device)
        alpha_stab, alpha_plas = alpha[0], alpha[1]

        # Compute Loss on New Data
        logits_new = self._logits_with_alpha(inputs, alpha)
        loss_new = self.loss(logits_new, labels)

        # Compute Loss on Replay Buffer
        loss_replay = torch.tensor(0.0, device=device)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=device
            )
            logits_replay = self._logits_with_alpha(buf_inputs, alpha)
            loss_replay = self.loss(logits_replay, buf_labels)

        # [cite_start]Weighted Aggregation [cite: 485]
        loss = alpha_stab * loss_replay + alpha_plas * loss_new
        loss.backward()
        self.opt.step()
        
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        return loss.item()