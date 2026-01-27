import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import rcParams
import matplotlib.font_manager as fm

# === 1. 字体与环境设置 ===
# 尝试寻找 Times New Roman，找不到则回退
font_names = [f.name for f in fm.fontManager.ttflist]
if 'Times New Roman' in font_names:
    serif_font = ['Times New Roman']
else:
    serif_font = ['DejaVu Serif', 'Liberation Serif', 'serif']

config = {
    "font.family": 'serif',
    "font.serif": serif_font,
    "mathtext.fontset": 'stix',
    "font.size": 12
}
rcParams.update(config)

# === 2. 数据录入 ===
data = [
    [62.61, 62.91, 63.21, 63.05, 63.84, 63.17, 63.54, 63.98, 63.14], # Beta 0.1
    [64.01, 63.67, 63.32, 65.97, 63.92, 64.40, 63.60, 64.23, 63.70], # Beta 0.2
    [62.78, 63.69, 64.73, 64.16, 63.30, 64.76, 63.66, 64.24, 64.01], # Beta 0.3
    [64.07, 64.16, 63.58, 64.54, 63.93, 64.56, 63.12, 63.40, 63.44], # Beta 0.4
    [63.02, 63.52, 63.65, 63.75, 63.76, 64.78, 63.74, 64.72, 64.12], # Beta 0.5
    [63.14, 63.92, 64.14, 64.00, 63.71, 64.71, 63.98, 65.05, 63.76], # Beta 0.6
    [63.14, 64.57, 63.90, 63.45, 63.94, 63.89, 63.60, 64.14, 63.07], # Beta 0.7
    [63.50, 66.03, 64.23, 64.04, 65.27, 65.21, 63.75, 64.56, 63.96], # Beta 0.8
    [63.57, 64.88, 63.70, 63.17, 63.44, 63.86, 63.23, 63.68, 63.94]  # Beta 0.9
]

betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
df = pd.DataFrame(data, index=betas, columns=alphas)

# === 3. 绘图 ===
plt.figure(figsize=(10, 8))

# 使用 "YlGnBu" 配色方案
# vmin/vmax 保持原设定以维持良好的颜色对比度
ax = sns.heatmap(df, annot=True, fmt=".2f",
                 cmap='YlGnBu', 
                 vmin=62.5, vmax=66.1,
                 linewidths=1, linecolor='white',
                 cbar_kws={'label': 'Accuracy (%)', 'shrink': 1.0})

# === 4. 细节调整 ===
plt.xlabel(r'$\alpha$', fontsize=18, fontweight='bold')
plt.ylabel(r'$\beta$', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)

# === 5. 保存 ===
plt.tight_layout()
plt.savefig('heatmap_cifar10_dam_clean.pdf', dpi=300, bbox_inches='tight')
plt.savefig('heatmap_cifar10_dam_clean.png', dpi=300, bbox_inches='tight')

plt.show()