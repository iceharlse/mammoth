import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from models import register_model

# =============================================================================
# HyperNet: 动态生成分类头参数
# =============================================================================
class HyperNet(nn.Module):
    def __init__(self, feature_dim: int, total_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.total_classes = total_classes
        
        # MLP: 将 Alpha 映射到隐空间
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Generator: 生成权重和偏置
        self.head_generator = nn.Linear(
            hidden_dim, (feature_dim + 1) * total_classes
        )

    def forward(self, alpha: torch.Tensor, n_classes: int):
        """
        alpha: (K, 2)
        Returns: weights (K, n_classes, feature_dim), biases (K, n_classes)
        """
        # 自动获取当前 HyperNet 参数所在的设备
        device = next(self.parameters()).device
        alpha = alpha.to(device)
        
        # 1. 映射偏好向量
        embedding = self.mlp(alpha) # (K, hidden_dim)
        
        # 2. 生成参数
        params = self.head_generator(embedding) 
        params = params.view(-1, self.total_classes, self.feature_dim + 1)
        
        # 3. 截取当前任务可见的类别
        weights = params[..., : self.feature_dim] # (K, Total_C, D)
        biases = params[..., -1]                  # (K, Total_C)
        
        return weights[:, :n_classes, :], biases[:, :n_classes]


@register_model("paretocl")
class ParetoCL(ContinualModel):
    NAME = "paretocl"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument("--hyper_hidden_dim", type=int, default=128)
        parser.add_argument("--paretocl_dirichlet_alpha", type=float, default=1.0, help="Dirichlet分布参数")
        parser.add_argument("--pref_samples_train", type=int, default=5, help="训练时的采样次数 K")
        parser.add_argument("--pref_samples_test", type=int, default=20, help="推理时的采样次数 K")
        
        # 保留自定义参数
        parser.add_argument("--freeze_backbone", type=int, default=0)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        
        # 自动获取特征维度
        self.feature_dim = getattr(self.net, "feature_dim", None)
        if self.feature_dim is None:
            if hasattr(self.net, 'embed_dim'): self.feature_dim = self.net.embed_dim
            elif hasattr(self.net, 'num_features'): self.feature_dim = self.net.num_features
            else: self.feature_dim = 512 
        
        # HyperNet 组件
        self.hypernet = HyperNet(
            feature_dim=self.feature_dim,
            total_classes=self.num_classes,
            hidden_dim=self.args.hyper_hidden_dim,
        )
        
        # 智能冻结策略 (保留您的代码)
        if getattr(self.args, 'freeze_backbone', 0) == 1:
            for param in self.net.parameters():
                param.requires_grad = False
            # 尝试解冻最后一块
            if hasattr(self.net, 'blocks'):
                for param in self.net.blocks[-1].parameters(): param.requires_grad = True
            elif hasattr(self.net, 'encoder') and hasattr(self.net.encoder, 'layers'):
                for param in self.net.encoder.layers[-1].parameters(): param.requires_grad = True
            # 解冻 Norm 层
            if hasattr(self.net, 'norm'):
                 for param in self.net.norm.parameters(): param.requires_grad = True
            elif hasattr(self.net, 'fc_norm'):
                 for param in self.net.fc_norm.parameters(): param.requires_grad = True
            self.net.eval()

        # 优化器配置
        self.opt = self.get_optimizer(self.parameters())

        # Alpha 分布参数
        self.register_buffer("dirichlet_concentration", torch.tensor(
            [self.args.paretocl_dirichlet_alpha, self.args.paretocl_dirichlet_alpha]
        ))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 Backbone 特征"""
        if hasattr(self.net, 'forward_features'):
             feats = self.net.forward_features(x)
        else:
             feats = self.net(x, returnt="features")
        if feats.dim() == 3: 
            feats = feats[:, 0]
        return feats

    def _compute_logits_stack(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        核心计算逻辑：支持一次性计算多个 Alpha 对应的 Logits
        """
        n_classes = self.num_classes if self.n_seen_classes == 0 else self.n_seen_classes
        feats = self._features(x) # (B, D), 通常在 GPU
        
        # 从 HyperNet 获取 K 组参数
        # 如果 HyperNet 在 CPU，这里的 w, b 也会在 CPU
        w, b = self.hypernet(alpha, n_classes)
        
        # [关键修复] 强制设备对齐
        # 如果 w 在 CPU 而 feats 在 GPU，强行将 w, b 移动到 feats 的设备
        if w.device != feats.device:
            w = w.to(feats.device)
            b = b.to(feats.device)
        
        # 矩阵乘法: (K, C, D) @ (B, D)^T -> (K, C, B) -> (K, B, C)
        logits = torch.einsum('kcd,bd->kbc', w, feats) + b.unsqueeze(1)
        return logits

    def forward(self, x: torch.Tensor, alpha: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        device = x.device
        
        # 1. 训练模式或指定 Alpha
        if alpha is not None:
            if alpha.dim() == 1: alpha = alpha.unsqueeze(0)
            logits = self._compute_logits_stack(x, alpha)
            return logits.squeeze(0)
            
        if self.training:
            # 训练态默认给一个平衡点 (防止被意外调用时 crash)
            default_alpha = torch.tensor([[0.5, 0.5]], device=device)
            return self._compute_logits_stack(x, default_alpha).squeeze(0)

        # 2. 推理模式 (ParetoCL Algorithm 2)
        K = self.args.pref_samples_test
        dirichlet = Dirichlet(self.dirichlet_concentration)
        alphas = dirichlet.sample((K,)).to(device) # (K, 2)

        logits_stack = self._compute_logits_stack(x, alphas) # (K, B, C)

        # 计算熵
        probs = F.softmax(logits_stack, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) # (K, B)
        
        # 选择最佳 Alpha
        best_k_indices = entropy.argmin(dim=0) # (B,)
        
        B = x.size(0)
        logits_stack = logits_stack.permute(1, 0, 2)
        final_logits = logits_stack[torch.arange(B, device=device), best_k_indices]
        
        return final_logits

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        device = inputs.device
        B = inputs.size(0)
        
        # 论文训练时 K=5
        K = self.args.pref_samples_train
        
        # 1. 采样 K 个偏好向量
        dirichlet = Dirichlet(self.dirichlet_concentration)
        alphas = dirichlet.sample((K,)).to(device)

        # 2. 准备数据
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=device
            )
        else:
            buf_inputs, buf_labels = None, None

        # 3. 计算 K 组 Logits
        logits_new_stack = self._compute_logits_stack(inputs, alphas)
        
        if buf_inputs is not None:
            logits_replay_stack = self._compute_logits_stack(buf_inputs, alphas)
        
        # 4. 计算期望损失 (Eq.4)
        total_loss = 0.0
        for k in range(K):
            alpha_stab = alphas[k, 0]
            alpha_plas = alphas[k, 1]
            
            l_new = self.loss(logits_new_stack[k], labels)
            
            l_replay = 0.0
            if buf_inputs is not None:
                l_replay = self.loss(logits_replay_stack[k], buf_labels)
            
            total_loss += (alpha_stab * l_replay + alpha_plas * l_new)

        loss = total_loss / K
        loss.backward()
        self.opt.step()
        
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        return loss.item()