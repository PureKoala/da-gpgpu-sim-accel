import matplotlib.pyplot as plt
import numpy as np

# 设置科研常用字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 1. 原始分类数据准备 (基于趋势估算)
years = np.arange(2008, 2025)
data_points = {
    "1024 x 768":  [25, 18, 7, 4, 3, 2, 2, 2, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2],
    "1152 x 864":  [4, 3, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
    "1280 x 800":  [9, 7, 6, 5, 4, 3, 2, 2, 2, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5],
    "1280 x 1024": [25, 20, 15, 11, 8, 7, 6, 5, 4, 1, 2, 1.5, 1, 1, 0.5, 0.5, 0.5],
    "1360 x 768":  [0, 1, 2, 2, 2, 2, 2, 2, 2, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5],
    "1366 x 768":  [0, 5, 15, 12, 18, 20, 22, 21, 18, 8, 15, 12, 10, 8, 6, 4, 3],
    "1440 x 900":  [10, 8, 7, 6, 5, 5, 5, 5, 4, 3, 4, 3, 2, 2, 1.5, 1, 1],
    "1600 x 900":  [0, 2, 3, 4, 5, 5, 6, 6, 5, 1, 2, 2, 2, 1, 1, 1, 0.5],
    "1680 x 1050": [14, 15, 12, 10, 9, 8, 7, 6, 4, 1, 2, 1.5, 1, 1, 0.5, 0.5, 0.5],
    "1920 x 1080": [4, 6, 25, 30, 32, 34, 35, 35, 38, 75, 58, 60, 62, 60, 58, 55, 50],
    "1920 x 1200": [1, 2, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "2560 x 1440": [0, 0, 0, 0.5, 1, 1, 1, 2, 3, 4, 5, 6, 8, 12, 15, 18, 20],
    "3440 x 1440": [0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 1, 2, 3, 3, 3, 3, 4, 5],
    "3840 x 2160": [0, 0, 0, 0, 0, 0, 0, 0.3, 0.5, 1, 1, 1, 1.5, 2, 3, 4, 5]
}

# 2. 计算 Others 类别
labels = list(data_points.keys())
values_array = np.array([data_points[label] for label in labels])
others = np.maximum(100 - np.sum(values_array, axis=0), 0)

all_labels = labels + ["Others"]
all_values = np.vstack([values_array, others])

# 3. 创建科研配图
fig, ax = plt.subplots(figsize=(11, 7), dpi=600)

# 使用 tab20 色板并显式映射红色系到 2K+ 分辨率
# tab20 颜色说明: 6-红, 7-浅红, 13-浅粉/红
tab20 = plt.cm.get_cmap('tab20')
color_indices = [
    0, 1,   # 1024, 1152 (蓝色系)
    2, 3,   # 1280x800, 1280x1024 (橙色系)
    4, 5,   # 1360, 1366 (绿色系)
    8, 9,   # 1440, 1600 (紫色系)
    10, 11, # 1680, 1920x1080 (棕色系)
    19,     # 1920x1200 (青色)
    13,     # 2560 x 1440 (映射为浅粉红)
    7,      # 3440 x 1440 (映射为浅红)
    6,      # 3840 x 2160 (映射为亮红)
    15      # Others (浅灰)
]
colors = [tab20(i) for i in color_indices]

# 4. 绘图
ax.stackplot(years, all_values, labels=all_labels, colors=colors, alpha=0.95, edgecolor='none')

# 5. 精细化美化
# x 参数控制标题水平位置: 0=最左, 0.5=正中, 1=最右, 可以微调如 0.48 或 0.52
ax.set_title('Historical Trend of Monitor Resolutions in Steam Hardware Survey (2008-2024)', 
             fontsize=14, fontweight='bold', loc='center', pad=10, ha='center', x=0.55)
ax.set_ylabel('Market Share (%)', fontsize=12, fontweight='medium')
ax.set_xlabel('Survey Year', fontsize=12, fontweight='medium')

# 轴限制与刻度优化
ax.set_xlim(2008, 2024)
ax.set_ylim(0, 100)
ax.set_xticks(np.arange(2008, 2025, 2))
ax.tick_params(axis='both', which='major', labelsize=10)

# 隐藏顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加轻微的水平参考线
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

# 6. 图例优化：逆序排列使图例顺序与图中色块高度顺序完全一致
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels_leg[::-1], loc='center left', 
          bbox_to_anchor=(1, 0.5), fontsize=11, frameon=False, title="Resolutions", title_fontsize=12)

# 7. 保存高质量图片
plt.tight_layout()
plt.savefig('Steam_Resolution.png', bbox_inches='tight', dpi=300)
plt.savefig('Steam_Resolution.pdf', bbox_inches='tight', transparent=True)