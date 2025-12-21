import matplotlib.pyplot as plt
import numpy as np

# 设置风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 1. 数据准备
# 格式: Year: [NVIDIA_TFLOPS, AMD_TFLOPS]
data_map = {
    2016: [8.9, 5.8],   # GTX 1080M vs RX 480M
    2019: [10.1, 7.9],  # RTX 2080M vs RX 5700M
    2021: [19.0, 12.2], # RTX 3080M vs RX 6800M
    2023: [33.0, 38.5], # RTX 4090M vs RX 7900M (AMD反超)
    2025: [36.5, 38.5]  # Blackwell vs RX 7900M (Continued)
}

years = np.array(sorted(data_map.keys()))
means = []
min_vals = []
max_vals = []

# 计算均值和区间
for y in years:
    vals = data_map[y]
    means.append(np.mean(vals))
    min_vals.append(min(vals))
    max_vals.append(max(vals))

means = np.array(means)
min_vals = np.array(min_vals)
max_vals = np.array(max_vals)

# 2. 绘图
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 计算误差棒长度 (用于绘制区间)
yerr_lower = means - min_vals
yerr_upper = max_vals - means
yerr = [yerr_lower, yerr_upper]

# 绘制均值折线 (深蓝色)
ax.plot(years, means, color='#2C3E50', linewidth=2.5, marker='o', markersize=6, 
        label='Industry Average Trend', zorder=10)

# 绘制垂直区间 (灰色粗线)
ax.errorbar(years, means, yerr=yerr, fmt='none', ecolor='#95A5A6', elinewidth=4, 
            capsize=8, capthick=2, label='Performance Range (NV/AMD)', zorder=5)

# 3. 详细标注
for i, y in enumerate(years):
    val_nv = data_map[y][0]
    val_amd = data_map[y][1]
    
    # 判断谁高谁低，分配颜色
    if val_nv >= val_amd:
        top_txt, bot_txt = f"Nvidia: {val_nv}", f"AMD: {val_amd}"
        top_col, bot_col = '#4b7600', '#9e0b0f' # 绿高红低
    else:
        top_txt, bot_txt = f"AMD: {val_amd}", f"Nvidia: {val_nv}"
        top_col, bot_col = '#9e0b0f', '#4b7600' # 红高绿低
        
    # 顶部标注 (Max)
    ax.annotate(top_txt, (y, max_vals[i]), xytext=(0, 6), textcoords='offset points', 
                ha='center', va='bottom', fontsize=9, color=top_col, fontweight='bold')
    
    # 底部标注 (Min)
    ax.annotate(bot_txt, (y, min_vals[i]), xytext=(0, -8), textcoords='offset points', 
                ha='center', va='top', fontsize=9, color=bot_col, fontweight='bold')
    
    # 均值标注
    ax.annotate(f"Avg: {means[i]:.1f}", (y, means[i]), xytext=(8, 0), textcoords='offset points', 
                ha='left', va='center', fontsize=9, color='#2C3E50', fontstyle='italic')

# 4. 图表美化
ax.set_title('Mobile GPU FP32 Compute Power Range (2016-2025)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
ax.set_xticks(years)
ax.set_xlim(2015, 2026)
ax.set_ylim(0, 45)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)
ax.legend(loc='upper left', frameon=False)

# 去除边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('GPU_Supply_Trend_Comparison.png')
plt.show() # 如果您在本地运行，使用 plt.show()，这里我已经保存为了图片