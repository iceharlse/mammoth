import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from utils.args import ArgumentParser
from models.paretocl import ParetoCL
from datasets import get_dataset

# -------------------------------------------
# ⚠️ 请确认这是你 ImageNet-R (49%) 的权重路径
CHECKPOINT_PATH = "./checkpoints/paretocl_seq-imagenet-r_None_2000_50_20251122-101206_2fc64851_last.pt"
# -------------------------------------------

sys.path.append(os.getcwd())

import inspect

def build_native_backbone():
    """
    智能读取并组装 Mammoth 原生 ViT
    不再硬编码参数，而是根据类定义动态适配
    """
    print("[Info] 正在读取 backbone.vit.VisionTransformer 定义...")
    try:
        from backbone.vit import VisionTransformer
        
        # 1. 准备我们想用的标准 ViT-B/16 参数
        # (这是 ImageNet/CIFAR-224 常用的配置)
        desired_params = {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'distilled': False,
            'drop_path_rate': 0.0,
            'ckpt_layer': 0,  # 之前报错的参数
            'num_classes': 0  # 特征提取模式
        }

        # 2. 读取类的构造函数签名
        sig = inspect.signature(VisionTransformer.__init__)
        valid_args = {}
        
        # 3. 智能过滤：只传它支持的参数
        print(f"   -> 检测到构造函数支持参数: {list(sig.parameters.keys())}")
        for param_name in desired_params:
            if param_name in sig.parameters:
                valid_args[param_name] = desired_params[param_name]
        
        print(f"   -> 最终传入参数: {list(valid_args.keys())}")

        # 4. 实例化
        model = VisionTransformer(**valid_args)
        
        # 补全 feature_dim
        if not hasattr(model, 'feature_dim'): 
            model.feature_dim = 768
            
        print("✅ 成功组装 VisionTransformer (参数已自动适配)")
        return model

    except ImportError:
        print(f"❌ 致命错误：无法导入 `backbone.vit`")
        return None
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None

def probe_overconfidence():
    print(f"[Info] 启动过度自信诊断 (Overconfidence Probe)...")
    print(f"[Target] Dataset: ImageNet-R | Backbone: ViT | K=20")

    parser = ArgumentParser()
    try:
        ParetoCL.get_parser(parser)
    except:
        pass

    # =========================================================================
    # 🛠️ 终极参数补全 (修复所有 AttributeError)
    # =========================================================================
    # 1. 基础设置
    args = parser.parse_args(['--buffer_size', '2000'])
    args.dataset = 'seq-imagenet-r'
    args.model = 'paretocl'
    args.backbone = 'vit'
    
    # 2. 核心数据管理 (本次报错修复点)
    args.validation = False           # 关闭验证集
    args.validation_mode = 'current'  # <--- [修复] 必须指定验证模式，即使 validation=False
    args.joint = 0                    # 关闭联合训练
    args.transform_type = 'weak'      # 指定增强类型
    
    # 3. 噪声控制
    args.noise_type = None
    args.noise_rate = 0.0
    args.disable_noisy_labels_cache = 1
    args.cache_path_noisy_labels = None
    
    # 4. 优化器与其他
    args.lr = 0.001
    args.optimizer = 'adam'
    args.optim_wd = 0.0
    args.optim_mom = 0.9
    args.optim_nesterov = 0
    
    # 5. 杂项 (防止打地鼠)
    args.custom_task_order = None
    args.custom_class_order = None
    args.seed = 0
    args.permute_classes = 0
    args.label_perc = 1.0
    args.label_perc_by_class = 1.0
    args.debug_mode = 0
    args.eval_future = 0
    args.distributed = 'no'
    args.savecheck = 'no'
    args.conf_path = ''
    args.notes = ''
    args.wandb_name = 'probe_test'
    args.disable_log = False
    args.tensorboard = 0
    
    # =========================================================================
    # 6. 模型与加载器参数 (修复 num_workers)
    # =========================================================================
    args.num_classes = 200
    args.minibatch_size = 32
    
    # [修复] DataLoader 必须参数
    args.batch_size = 32      
    args.num_workers = 4      # <--- 本次报错缺少的
    args.pin_memory = True    # <--- [预防] 下一个可能报错的
    args.drop_last = False    
    
    # ParetoCL 参数
    if not hasattr(args, 'hyper_hidden_dim'): args.hyper_hidden_dim = 128
    if not hasattr(args, 'pref_samples_test'): args.pref_samples_test = 20
    if not hasattr(args, 'pref_samples_train'): args.pref_samples_train = 5
    if not hasattr(args, 'paretocl_dirichlet_alpha'): args.paretocl_dirichlet_alpha = 1.0
    args.save_paretocl_log = 1 
    # =========================================================================
    # =========================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载数据
    print(f"[Info] Loading Dataset (Mammoth get_dataset)...")
    mammoth_dataset = get_dataset(args)
    
    # 伪 Transform 用于模型初始化
    from torchvision import transforms
    dummy_transform = transforms.Compose([transforms.ToTensor()])

    # 2. 构建模型
    backbone = build_native_backbone()
    if backbone is None: return

    loss = torch.nn.CrossEntropyLoss()
    print(f"[Info] Initializing ParetoCL...")
    model = ParetoCL(backbone, loss, args, dummy_transform, dataset=mammoth_dataset)
    
    # 3. 加载权重
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[Info] Loading Weights: {CHECKPOINT_PATH}")
        loaded_data = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = loaded_data.get('model') or loaded_data.get('model_state_dict') or loaded_data
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ Weights Loaded (Strict)!")
        except RuntimeError as e:
            print(f"⚠️ Strict Failed. Trying Loose Loading...")
            # 过滤掉不匹配的键 (例如 VisionTransformer 的 pos_embed 可能会因为 resize 策略不同而 mismatch)
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    model.to(device)
    model.eval()

    # =========================================================
    # 核心诊断逻辑
    # =========================================================
    K = 20 # 采样次数
    dirichlet = Dirichlet(torch.tensor([1.0, 1.0])) 
    
    total_samples = 0
    standard_correct = 0
    oracle_correct = 0
    overconfident_errors = 0 
    
    print(f"\n[Start] 开始 Oracle 对比测试 (K={K})...")
    print(f"{'Task':<5} | {'Std Acc':<10} | {'Oracle Acc':<10} | {'Gap (Potential)':<15} | {'Overconf Rate':<15}")
    print("-" * 75)

    with torch.no_grad():
        # 遍历每个 Task
        for task_id in range(mammoth_dataset.N_TASKS):
            # 获取该 Task 的测试集
            # 注意：get_data_loaders 会设置内部状态，确保我们拿到的是对应 Task 的数据
            _, test_loader = mammoth_dataset.get_data_loaders()
            
            task_samples = 0
            task_std_corr = 0
            task_ora_corr = 0
            task_overconf = 0
            
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                B = inputs.size(0)
                
                # 1. 采样 K 个 Alpha
                alphas = dirichlet.sample((K,)).to(device)
                
                # 2. 获取 K 组 Logits
                logits_stack = model._compute_logits_stack(inputs, alphas)
                
                # 3. 计算概率和熵
                probs_stack = F.softmax(logits_stack, dim=-1) # (K, B, C)
                entropy_stack = -(probs_stack * torch.log(probs_stack + 1e-8)).sum(dim=-1) # (K, B)
                
                # 4. 预测
                preds_stack = probs_stack.argmax(dim=-1) # (K, B)
                
                # --- 策略 A: ParetoCL (Min Entropy) ---
                selected_indices = entropy_stack.argmin(dim=0) # (B,)
                std_preds = preds_stack[selected_indices, torch.arange(B, device=device)]
                is_std_correct = (std_preds == labels) 
                
                # --- 策略 B: Oracle (Max Potential) ---
                # 只要 K 次里有一次猜对，就算对
                is_oracle_correct = (preds_stack == labels.unsqueeze(0)).any(dim=0)
                
                # --- 统计 ---
                task_samples += B
                task_std_corr += is_std_correct.sum().item()
                task_ora_corr += is_oracle_correct.sum().item()
                
                # 过度自信 = 自己选错了，但在备选里有正确答案
                is_overconfident = (~is_std_correct) & is_oracle_correct
                task_overconf += is_overconfident.sum().item()

            # 汇总 Task
            if task_samples > 0:
                std_acc = 100 * task_std_corr / task_samples
                ora_acc = 100 * task_ora_corr / task_samples
                overconf_rate = 100 * task_overconf / task_samples
                gap = ora_acc - std_acc
                print(f"{task_id:<5} | {std_acc:6.2f}%    | {ora_acc:6.2f}%    | +{gap:5.2f}%         | {overconf_rate:5.2f}%")
                
                total_samples += task_samples
                standard_correct += task_std_corr
                oracle_correct += task_ora_corr
                overconfident_errors += task_overconf

    print("-" * 75)
    if total_samples > 0:
        total_std_acc = 100 * standard_correct / total_samples
        total_ora_acc = 100 * oracle_correct / total_samples
        total_overconf = 100 * overconfident_errors / total_samples
        
        print(f"【Final Result】")
        print(f"Standard Acc (Entropy): {total_std_acc:.2f}%")
        print(f"Oracle Acc (Potential): {total_ora_acc:.2f}%")
        print(f"Performance Gap       : {total_ora_acc - total_std_acc:.2f}%")
        print(f"Overconfidence Rate   : {total_overconf:.2f}%")
    else:
        print("⚠️ 没有处理任何样本，请检查 Dataloader。")

if __name__ == "__main__":
    probe_overconfidence()