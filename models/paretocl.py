import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class HyperNet(nn.Module):
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

    def forward(self, alpha: torch.Tensor, n_classes: int):
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
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


class ParetoCL(ContinualModel):
    NAME = "paretocl"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument("--hyper_hidden_dim", type=int, default=128,
                            help="Hidden dimension for the hypernetwork MLP")
        parser.add_argument("--pref_samples", type=int, default=20,
                            help="Number of preference samples for dynamic adaptation")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.feature_dim = getattr(self.net, "feature_dim", None)
        if self.feature_dim is None:
            raise ValueError("Backbone must expose feature_dim attribute")
        self.hypernet = HyperNet(self.feature_dim, self.num_classes, self.args.hyper_hidden_dim)
        self.dirichlet = Dirichlet(torch.ones(2))
        self.pref_samples = self.args.pref_samples

    def generate_logits(self, features: torch.Tensor, alpha: torch.Tensor, n_classes: int):
        weights, biases = self.hypernet(alpha, n_classes)
        logits = torch.matmul(features, weights.t()) + biases
        return logits

    def forward(self, x: torch.Tensor, alpha: torch.Tensor = None):
        n_classes = self.n_seen_classes if self.n_seen_classes > 0 else self.num_classes
        feats = self.net(x, returnt="features")
        if alpha is not None:
            return self.generate_logits(feats, alpha.to(feats.device), n_classes)

        alphas = self.dirichlet.sample((self.pref_samples,)).to(feats.device)
        logits_candidates = []
        for pref in alphas:
            logits_candidates.append(self.generate_logits(feats, pref, n_classes))
        stacked_logits = torch.stack(logits_candidates, dim=0)
        probs = F.softmax(stacked_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        best_indices = entropy.argmin(dim=0)
        chosen_logits = stacked_logits.permute(1, 0, 2)[torch.arange(feats.size(0)), best_indices]
        return chosen_logits

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        alpha_sample = self.dirichlet.sample().to(self.device)

        loss_replay = torch.tensor(0.0, device=self.device)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            replay_pref = torch.tensor([alpha_sample[0], 1 - alpha_sample[0]], device=self.device)
            replay_logits = self.forward(buf_inputs, alpha=replay_pref)
            loss_replay = self.loss(replay_logits, buf_labels)

        plasticity_pref = torch.tensor([1 - alpha_sample[1], alpha_sample[1]], device=self.device)
        outputs = self.forward(inputs, alpha=plasticity_pref)
        loss_new = self.loss(outputs, labels)

        loss = alpha_sample[0] * loss_replay + alpha_sample[1] * loss_new
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        return loss.item()
