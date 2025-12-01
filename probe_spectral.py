import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.args import ArgumentParser
from models.paretocl import ParetoCL
from datasets import get_dataset

# -------------------------------------------
# ⚠️ 权重路径
CHECKPOINT_PATH = "./checkpoints/paretocl_seq-imagenet-r_None_2000_50_20251122-101206_2fc64851_last.pt"
# -------------------------------------------

sys.path.append(os.getcwd())

def build_native_backbone():
    """导入 Mammoth 原生 ViT"""
    try:
        from backbone.vit import VisionTransformer
        # 使用之前的智能参数适配逻辑
        import inspect
        sig = inspect.signature(VisionTransformer.__init__)
        params = {
            'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
            'depth': 12, 'num_heads': 12, 'num_classes': 0,
            'drop_path_rate': 0.0
        }
        valid_args = {k: v for k, v in params.items() if k in sig.parameters}
        model = VisionTransformer(**valid_args)
        if not hasattr(model, 'feature_dim'): model.feature_dim = 768
        return model
    except Exception as e:
        print(f"❌ Backbone Init Failed: {e}")
        return None

def compute_effective_rank(singular_values):
    """计算有效秩 (Effective Rank)"""
    # 归一化奇异值得到分布 p
    singular_values = singular_values / singular_values.sum()
    # 计算熵 H(p)
    entropy = -(singular_values * torch.log(singular_values + 1e-8)).sum()
    # Effective Rank = exp(H(p))
    return torch.exp(entropy).item()

def probe_spectral():
    print(f"[Info] 启动谱塌陷诊断 (Spectral Collapse Probe)...")

    # 1. 准备参数 (复用之前的稳健配置)
    parser = ArgumentParser()
    try: ParetoCL.get_parser(parser)
    except: pass
    args = parser.parse_args(['--buffer_size', '2000'])
    
    # 核心配置
    args.dataset = 'seq-imagenet-r'
    args.model = 'paretocl'
    args.backbone = 'vit'
    args.num_classes = 200
    args.minibatch_size = 32
    args.batch_size = 32
    args.num_workers = 4
    args.drop_last = False  # <--- [FIX] 解决本次报错
    args.pin_memory = True
    
    # 防报错配置
    args.validation = False
    args.validation_mode = 'current'
    args.joint = 0
    args.transform_type = 'weak'
    args.noise_type = None; args.noise_rate = 0.0
    args.disable_noisy_labels_cache = 1
    args.cache_path_noisy_labels = None
    args.lr = 0.001; args.optimizer = 'adam'; args.optim_wd = 0.0
    args.optim_mom = 0.9; args.optim_nesterov = 0
    args.custom_task_order = None; args.custom_class_order = None
    args.seed = 0; args.permute_classes = 0
    args.label_perc = 1.0; args.label_perc_by_class = 1.0
    args.debug_mode = 0; args.eval_future = 0; args.distributed = 'no'
    args.savecheck = 'no'; args.conf_path = ''; args.notes = ''
    args.wandb_name = 'probe_spectral'; args.disable_log = False
    args.tensorboard = 0
    
    # ParetoCL
    if not hasattr(args, 'hyper_hidden_dim'): args.hyper_hidden_dim = 128
    if not hasattr(args, 'pref_samples_test'): args.pref_samples_test = 20
    if not hasattr(args, 'paretocl_dirichlet_alpha'): args.paretocl_dirichlet_alpha = 1.0
    args.save_paretocl_log = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型与数据
    print(f"[Info] Loading Dataset & Model...")
    mammoth_dataset = get_dataset(args)
    from torchvision import transforms
    dummy_transform = transforms.Compose([transforms.ToTensor()])
    backbone = build_native_backbone()
    loss = torch.nn.CrossEntropyLoss()
    model = ParetoCL(backbone, loss, args, dummy_transform, dataset=mammoth_dataset)
    
    if os.path.exists(CHECKPOINT_PATH):
        loaded = torch.load(CHECKPOINT_PATH, map_location=device)
        sd = loaded.get('model') or loaded.get('model_state_dict') or loaded
        try: model.load_state_dict(sd, strict=True)
        except: model.load_state_dict(sd, strict=False)
    else:
        print("❌ Checkpoint not found!")
        return

    model.to(device)
    model.eval()

    # =========================================================
    # 核心诊断: 收集特征并进行 SVD 分析
    # =========================================================
    
    # 存储每个 Task 的特征矩阵
    task_features = {} 
    
    print(f"\n[Start] 正在提取特征并计算奇异值谱...")
    print(f"{'Task':<5} | {'Samples':<8} | {'Eff. Rank':<10} | {'Top-1 SV':<10} | {'Top-50 SV Ratio':<15}")
    print("-" * 75)

    with torch.no_grad():
        for task_id in range(mammoth_dataset.N_TASKS):
            _, test_loader = mammoth_dataset.get_data_loaders()
            
            feats_list = []
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                # 获取 Backbone 特征
                f = model.net(inputs)
                if f.dim() > 2: f = f[:, 0] # ViT CLS token
                feats_list.append(f.cpu())
            
            # 拼接该 Task 的所有特征: (N, 768)
            feats_matrix = torch.cat(feats_list, dim=0)
            
            # 中心化特征 (PCA 需要)
            feats_centered = feats_matrix - feats_matrix.mean(dim=0, keepdim=True)
            
            # SVD 分解
            # U, S, V = torch.svd(feats_centered)
            # S 是奇异值向量，从大到小排列
            _, S, _ = torch.linalg.svd(feats_centered, full_matrices=False)
            
            # 计算指标
            eff_rank = compute_effective_rank(S)
            top1 = S[0].item()
            # 计算前 50 个奇异值占总能量的比例
            energy_ratio = (S[:50].square().sum() / S.square().sum()).item() * 100
            
            task_features[task_id] = S.numpy()
            
            print(f"{task_id:<5} | {feats_matrix.size(0):<8} | {eff_rank:6.2f}     | {top1:6.1f}     | {energy_ratio:6.2f}%")

    # 绘制谱图
    print(f"\n[Info] 生成谱塌陷对比图 (Task 0 vs Task 9)...")
    plt.figure(figsize=(10, 6))
    
    # 为了对比，我们把奇异值归一化 (除以各自的最大值)
    s_old = task_features[0] / task_features[0].max()
    s_new = task_features[mammoth_dataset.N_TASKS - 1] / task_features[mammoth_dataset.N_TASKS - 1].max()
    
    plt.plot(np.log10(s_old[:100]), label=f'Task 0 (Oldest) - Eff Rank: {compute_effective_rank(torch.tensor(task_features[0])):.1f}', linewidth=2, color='blue')
    plt.plot(np.log10(s_new[:100]), label=f'Task 9 (Newest) - Eff Rank: {compute_effective_rank(torch.tensor(task_features[mammoth_dataset.N_TASKS - 1])):.1f}', linewidth=2, color='red', linestyle='--')
    
    plt.title('Log-Singular Value Spectrum (Top 100 Components)')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Log10 (Normalized Singular Value)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('paretocl_spectral_collapse.png')
    print(f"[Info] 结果图已保存至 paretocl_spectral_collapse.png")

if __name__ == "__main__":
    probe_spectral()