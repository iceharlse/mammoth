import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 1. 录入数据 (来自你的图片)
alphas = np.array([0.1, 0.4, 0.75, 0.8, 0.85, 1.2]) # Columns
betas = np.array([0.1, 0.3, 0.55, 0.6, 0.65, 0.9]) # Rows

# 数据矩阵 (Rows correspond to betas, Cols to alphas)
values = np.array([
    [62.61, 63.05, 63.54, 62.98, 63.14, 62.18], # Beta 0.1
    [62.78, 64.16, 63.66, 64.24, 64.01, 61.37], # Beta 0.3
    [63.02, 63.75, 63.74, 64.72, 64.12, 62.45], # Beta 0.55
    [63.14, 63.00, 63.98, 65.05, 63.76, 63.34], # Beta 0.6
    [63.14, 63.45, 63.60, 64.14, 63.07, 63.03], # Beta 0.65
    [62.57, 62.17, 63.23, 63.68, 63.94, 63.33]  # Beta 0.9
])

# 准备网格数据
X, Y = np.meshgrid(alphas, betas)
points = np.column_stack((X.ravel(), Y.ravel()))
Z = values.ravel()

# 定义绘图辅助函数
def plot_heatmap(ax, grid_x, grid_y, grid_z, title, points, x_range=None, y_range=None):
    # 使用 contourf 进行平滑填充
    c = ax.contourf(grid_x, grid_y, grid_z, levels=50, cmap='viridis')
    
    # 标出原始数据点（黑色小点）
    ax.scatter(points[:, 0], points[:, 1], c='black', s=20, alpha=0.5, label='Sampled Points')
    
    # 标出最高点 (Alpha 0.8, Beta 0.6)
    ax.scatter([0.8], [0.6], c='red', marker='*', s=150, edgecolors='white', label='Peak (0.8, 0.6)')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r'$\alpha$ (Alpha)', fontsize=12)
    ax.set_ylabel(r'$\beta$ (Beta)', fontsize=12)
    plt.colorbar(c, ax=ax, label='Accuracy (%)')
    
    # 如果指定了范围（局部图），则设置坐标轴
    if x_range and y_range:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
    
    ax.legend(loc='lower right', framealpha=0.8)

# --- 图 1：全局插值热力图 ---
plt.figure(figsize=(16, 6))

# 创建全局更密集的网格用于插值
grid_x_global, grid_y_global = np.mgrid[min(alphas):max(alphas):200j, min(betas):max(betas):200j]
grid_z_global = griddata(points, Z, (grid_x_global, grid_y_global), method='cubic')

ax1 = plt.subplot(1, 2, 1)
plot_heatmap(ax1, grid_x_global, grid_y_global, grid_z_global, 
             "Global Parameter Sensitivity (Full Table)", points)

# --- 图 2：局部放大插值热力图 ---
# 目标：Alpha 0.8, Beta 0.6 周围 0.05 范围
# 即 Alpha: [0.75, 0.85], Beta: [0.55, 0.65]

ax2 = plt.subplot(1, 2, 2)
# 为了局部图更平滑，我们在局部范围内生成网格，但依然利用全局数据进行插值以保证边界准确
zoom_x_min, zoom_x_max = 0.75, 0.85
zoom_y_min, zoom_y_max = 0.55, 0.65

grid_x_local, grid_y_local = np.mgrid[zoom_x_min:zoom_x_max:200j, zoom_y_min:zoom_y_max:200j]
grid_z_local = griddata(points, Z, (grid_x_local, grid_y_local), method='cubic')

plot_heatmap(ax2, grid_x_local, grid_y_local, grid_z_local, 
             "Local Zoomed Heatmap (Center: 0.8, 0.6)", points,
             x_range=(zoom_x_min, zoom_x_max),
             y_range=(zoom_y_min, zoom_y_max))

# 标出该区域的极值数值
ax2.text(0.8, 0.61, "65.05%", color='white', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()