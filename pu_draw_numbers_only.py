import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==========================================
# 1. 全局风格设置
# ==========================================
font_names = [f.name for f in fm.fontManager.ttflist]
if 'Times New Roman' in font_names:
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    plt.rcParams['font.family'] = 'serif'

# ---- Font size control: edit BASE_FONT only ----
BASE_FONT = 15 

def _clamp(v, lo=6):
    return max(lo, int(v))

FS_LABEL    = _clamp(BASE_FONT + 1)
FS_TICK     = _clamp(BASE_FONT - 1)
FS_TITLE    = _clamp(BASE_FONT + 5)
FS_SUBTITLE = _clamp(BASE_FONT + 1)
FS_LEGEND   = _clamp(BASE_FONT - 2)
FS_ANNOT    = _clamp(BASE_FONT - 3)

# Only for numeric value labels inside plots (bar labels & point annotations)
FS_VALUE   = _clamp(BASE_FONT + 8)

plt.rcParams.update({
    'font.size': BASE_FONT,
    'axes.titlesize': FS_TITLE,
    'axes.labelsize': FS_LABEL,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_LEGEND,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'axes.linewidth': 1.6,
})

# 统一颜色: Blue, Orange, Green, Red
colors_stages = ["#377eb8", "#ff7f00", "#4daf4a", "#e41a1c"] 
labels_layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
ranks_norm_concept = [0.27, 0.35, 0.30, 0.13] 

# Fixed limits for the Concept Nebula KDE grid
NEBULA_XLIM = (-4.5, 4.5)
NEBULA_YLIM = (-4.0, 4.0)

# ==========================================
# 2. 数据准备 (保持不变)
# ==========================================

# --- Part A: 左侧 Concept 星云 ---
def generate_layer_data(n_samples=2000, layer_idx=0):
    np.random.seed(42 + layer_idx)
    x = np.random.normal(0, 1, n_samples)
    y = np.random.normal(0, 1, n_samples)
    
    if layer_idx == 0: 
        x = x * 1.0 + 0.2 * np.sin(y*2)
        y = y * 0.55 + 0.1 * x**2 
    elif layer_idx == 1: 
        x = x * 1.1
        y = y * 0.8 + 0.1 * np.cos(x) 
    elif layer_idx == 2: 
        x = x * 1.05
        y = y * 0.65 - 0.05 * x**2 
    elif layer_idx == 3: 
        x = x * 1.2
        y = y * 0.15 + 0.15 * np.sin(x * 1.5) 
    return x, y

def kde2d_contour(ax, x, y, *, color, xlim=NEBULA_XLIM, ylim=NEBULA_YLIM,
                 fill_alpha=0.10, line_alpha=0.85, lw=1.8, levels=2, thresh=0.2, gridsize=220):
    if len(x) < 5 or len(y) < 5: return
    if np.std(x) < 1e-8 or np.std(y) < 1e-8: return
    values = np.vstack([x, y])
    kde = gaussian_kde(values)
    xx, yy = np.mgrid[xlim[0]:xlim[1]:complex(gridsize), ylim[0]:ylim[1]:complex(gridsize)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.reshape(kde(positions), xx.shape)
    zmax = float(np.nanmax(zz))
    if not np.isfinite(zmax) or zmax <= 0: return
    zmin = max(1e-12, thresh * zmax)
    n_levels = max(2, int(levels))
    levs = np.linspace(zmin, zmax, n_levels)
    ax.contourf(xx, yy, zz, levels=levs, colors=[color], alpha=fill_alpha, antialiased=True)
    ax.contour(xx, yy, zz, levels=levs, colors=[color], linewidths=lw, alpha=line_alpha)

# --- Part B: 左侧 Diagnosis 柱状图 ---
layers_x = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
eff_rank_data = [17.2, 45.5, 78.0, 70.6] 
norm_rank_data = [0.27, 0.35, 0.30, 0.13] 

# --- Part C: 右侧 Method 折线图 ---
tasks = [1, 2, 3, 4, 5]
stages_list = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

# 1. ER
er_acc_mean = np.array([51.56, 21.88, 1.59, 14.65, 1.56])
er_acc_std  = np.array([5.0, 3.0, 0.5, 2.0, 0.5])
er_ranks = {
    'Stage 1': [0.03851, 0.03132, 0.03631, 0.03148, 0.05209],
    'Stage 2': [0.02562, 0.03132, 0.04035, 0.02806, 0.04457],
    'Stage 3': [0.01815, 0.02218, 0.04391, 0.03148, 0.05209],
    'Stage 4': [0.00612, 0.00783, 0.02112, 0.01979, 0.02596]
}

# 2. ParetoCL
paretocl_acc_mean = np.array([52.15, 32.42, 5.27, 6.84, 1.39])
paretocl_acc_std  = np.array([5.0, 3.5, 0.5, 0.8, 0.3])
paretocl_ranks = {
    'Stage 1': [0.07987, 0.09072, 0.08517, 0.06398, 0.09915],
    'Stage 2': [0.04026, 0.07526, 0.06907, 0.06710, 0.09287],
    'Stage 3': [0.02169, 0.06441, 0.05630, 0.05496, 0.04425],
    'Stage 4': [0.01023, 0.00921, 0.01127, 0.01332, 0.00891]
}

# 3. SCOPE (Ours)
dam_acc_mean = np.array([51.56, 35.94, 45.31, 34.77, 18.76])
dam_acc_std  = np.array([5.0, 6.0, 4.0, 5.0, 1.0])
dam_ranks = {
    'Stage 1': [0.06004, 0.08298, 0.07959, 0.07984, 0.09215],
    'Stage 2': [0.03563, 0.05701, 0.06612, 0.06803, 0.08119],
    'Stage 3': [0.02707, 0.02615, 0.05880, 0.04832, 0.07603],
    'Stage 4': [0.00717, 0.01857, 0.03164, 0.03075, 0.04234]
}

methods_data = [
    ("(b) ER", er_acc_mean, er_acc_std, er_ranks),
    ("(c) ParetoCL", paretocl_acc_mean, paretocl_acc_std, paretocl_ranks),    
    ("(d) +SCOPE (ours)", dam_acc_mean, dam_acc_std, dam_ranks),
]

# ==========================================
# 3. 辅助绘图函数
# ==========================================
def annotate_points(ax, x, y, y_offset=6):
    for xi, yi in zip(x, y):
        label = f"{yi:.5f}" if yi <= 1.0 else f"{yi:.2f}"
        ax.annotate(
            label, (xi, yi),
            textcoords="offset points", xytext=(0, y_offset),
            ha='center', fontsize=FS_VALUE, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white', alpha=0.9)]
        )

def setup_axis_lineplot(ax):
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.85, 5.15)
    
# ==========================================
# 4. 主绘图逻辑 (2行 x 4列)
# ==========================================
fig, axes = plt.subplots(2, 4, figsize=(25, 9), dpi=300)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.94, wspace=0.35, hspace=0.25)

# -------------------------------------------------------------------------
# Column 0: 左侧两张图
# -------------------------------------------------------------------------

# --- [0,0] Top-Left: Concept Nebula ---
ax_nebula = axes[0, 0]
draw_order = [1, 2, 0, 3] 

for original_idx in draw_order:
    x, y = generate_layer_data(3000, layer_idx=original_idx)
    color = colors_stages[original_idx]
    
    lw = 2.8 if original_idx == 3 else 1.8
    alpha_line = 1.0 if original_idx == 3 else 0.8
    kde2d_contour(ax_nebula, x, y, color=color, fill_alpha=0.10, line_alpha=alpha_line, lw=lw, levels=2, thresh=0.2)
    idx = np.random.choice(len(x), 60)
    ax_nebula.scatter(x[idx], y[idx], s=12, color=color, alpha=0.5, edgecolors='none')

ax_nebula.set_title('(a) Spectral Collapse Analysis', fontweight='bold')
ax_nebula.set_xticks([])
ax_nebula.set_yticks([])
ax_nebula.set_xlabel('Feature Dim 1', fontsize=FS_LABEL, color='gray')
ax_nebula.set_ylabel('Feature Dim 2', fontsize=FS_LABEL, color='gray')
for spine in ax_nebula.spines.values():
    spine.set_visible(False)
ax_nebula.set_xlim(-4.5, 4.5)
ax_nebula.set_ylim(-4, 4)
ax_nebula.set_aspect('equal')

legend_elements_nebula = []
for i in range(4):
    legend_elements_nebula.append(Line2D([0], [0], marker='o', color='w', 
                          label=f'{labels_layers[i]} (R$\\approx${ranks_norm_concept[i]})',
                          markerfacecolor=colors_stages[i], markersize=9))
ax_nebula.legend(handles=legend_elements_nebula, loc='upper right', fontsize=FS_LEGEND, 
                 frameon=True, fancybox=True, framealpha=0.9, handletextpad=0.2)


# --- [1,0] Bottom-Left: Diagnosis Bar+Line ---
ax_bar = axes[1, 0]
ax_line_twin = ax_bar.twinx()

bars = ax_bar.bar(layers_x, eff_rank_data, color=colors_stages, width=0.7, zorder=1, alpha=0.9)
ax_line_twin.plot(layers_x, norm_rank_data, color='#333333', marker='o', 
                  linewidth=3, markersize=8, zorder=2)

for bar in bars:
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height:.1f}', ha='center', va='bottom', fontsize=FS_VALUE, color='black')

ax_bar.set_title('Diagnosis: Effective Rank', fontweight='bold', fontsize=FS_SUBTITLE)
ax_bar.set_ylabel('Effective Rank', color='black', fontweight='bold')
ax_bar.set_ylim(0, 90)
ax_bar.tick_params(axis='x', labelsize=FS_TICK)

ax_line_twin.set_ylabel('Norm. Rank', color='#333333', fontweight='bold')
ax_line_twin.tick_params(axis='y', labelcolor='#333333')
ax_line_twin.set_ylim(0, 0.6)

legend_elements_bottom = [
    Line2D([0], [0], color='#333333', lw=3, marker='o', label='Norm. Rank')
]
ax_bar.legend(handles=legend_elements_bottom, loc='upper left', frameon=True, fontsize=FS_LEGEND)


# -------------------------------------------------------------------------
# Columns 1-3: Method Comparisons
# -------------------------------------------------------------------------
markers_list = ['o', 's', '^', 'D']

for i, (method_name, acc_mean, acc_std, ranks_dict) in enumerate(methods_data):
    col_idx = i + 1 
    
    # --- Top Row: Accuracy ---
    ax_acc = axes[0, col_idx]
    setup_axis_lineplot(ax_acc)
    
    ax_acc.plot(tasks, acc_mean, marker='o', color="#314D63", linewidth=3.2, markersize=9)
    ax_acc.fill_between(tasks, acc_mean - acc_std, acc_mean + acc_std, color="#979797", alpha=0.22)
    annotate_points(ax_acc, tasks, acc_mean, y_offset=8)
    
    ax_acc.set_title(f"{method_name}", fontweight='bold')
    ax_acc.set_ylim(-5, 85)
    ax_acc.set_xlabel('') 
    
    if col_idx == 1:
        ax_acc.set_ylabel('Old-task probe accuracy (%)')
    else:
        ax_acc.set_ylabel('')

    # --- Bottom Row: Rank (Modified Section) ---
    ax_rank = axes[1, col_idx]
    setup_axis_lineplot(ax_rank)
    
    for stage_i, stage_name in enumerate(stages_list):
        values = ranks_dict[stage_name]
        
        # 1. 始终绘制折线
        ax_rank.plot(tasks, values, marker=markers_list[stage_i], color=colors_stages[stage_i],
                     label=stage_name, markersize=8.5, linewidth=2.6)
        
        # 2. 仅对 Stage 1 和 Stage 4 绘制数字
        if stage_name == 'Stage 1':
            annotate_points(ax_rank, tasks, values, y_offset=10) # 向上偏
        elif stage_name == 'Stage 4':
            annotate_points(ax_rank, tasks, values, y_offset=-14) # 向下偏

    ax_rank.set_ylim(0.00, 0.10)
    ax_rank.set_xlabel('After training task t')
    
    if col_idx == 1:
        ax_rank.set_ylabel('Norm. effective rank\n(eRank / dim)')
        ax_rank.legend(loc='upper left', framealpha=0.95, edgecolor='gray',
                       fancybox=True, fontsize=FS_LEGEND, handlelength=1.2)
    else:
        ax_rank.set_ylabel('')

# 保存
plt.savefig('figure3.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure3.png', dpi=300, bbox_inches='tight')
plt.show()