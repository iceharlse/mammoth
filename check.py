import sys
import os
import torch
from models.er_star_log import ErSTARLog

# 1. 检查类中是否存在 observe 方法
print(f"Checking class: {ErSTARLog.__name__}")

if 'observe' in ErSTARLog.__dict__:
    print("✅ PASS: 'observe' method found in class definition.")
else:
    print("❌ FAIL: 'observe' method NOT found in class definition!")
    print("   -> 原因：通常是因为缩进错误。请检查 def observe 是否与 def __init__ 垂直对齐。")
    print("   -> 检查：它是否被意外缩进到了上一个函数内部？")
    sys.exit(1)

# 2. 尝试实例化并检查方法
try:
    # 模拟必要的参数
    class MockArgs:
        buffer_size = 100
        minibatch_size = 10
    
    model = ErSTARLog(
        backbone=torch.nn.Linear(10, 10),
        loss=torch.nn.MSELoss(),
        args=MockArgs(),
        transform=None
    )
    print("✅ PASS: Model instantiated successfully.")
except Exception as e:
    print(f"⚠️ Warning: Instantiation failed ({e}), but checking method existence...")

# 3. 再次检查实例方法
if hasattr(model, 'observe'):
    print("✅ PASS: Instance has 'observe' method.")
else:
    print("❌ FAIL: Instance missing 'observe' method.")