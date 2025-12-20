import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 1. 定义关键时间锚点 (整数年份)
years_points = np.array([2021, 2022, 2023, 2024, 2025])

# 2. 定义带有"起伏"的市场数据 (锚点数据)
# 这里的数字不再是完美的等差数列，而是模拟了真实市场的混乱
data_anchors = {
    # 60Hz: 2022年有一个小的"抵抗"平台期，之后加速崩盘
    "60 Hz":      np.array([60.0, 56.5, 55.0, 42.0, 30.0]),
    
    # 144Hz: 2022年冲顶，2023年因为库存积压降幅不明显，2024年被165Hz剧烈挤压
    "144 Hz":     np.array([22.0, 27.0, 26.5, 23.0, 15.0]),
    
    # 165Hz: 早期增长缓慢，2023-2024 面板产线切换后呈现"指数级"爆发
    "165 Hz":     np.array([ 4.0,  6.5, 11.0, 22.0, 35.0]),
    
    # 240 Hz+: 依然是高端小众，稳步爬升
    "240 Hz+":    np.array([ 1.0,  1.5,  2.5,  6.0, 10.0])
}

# 3. 使用样条插值生成平滑但波动的曲线 (模拟月度/季度变化)
x_smooth = np.linspace(years_points.min(), years_points.max(), 300)
smooth_data = {}

for label, y_points in data_anchors.items():
    # k=2 代表二次样条，保证曲线平滑但保留起伏特征
    spl = make_interp_spline(years_points, y_points, k=2)
    y_smooth = spl(x_smooth)
    # 修正插值可能产生的负数
    smooth_data[label] = np.maximum(y_smooth, 0) 

# 4. 计算 Others (自动填充剩余部分)
# 将所有已知类别的平滑数据相加
known_values_stacked = np.zeros_like(x_smooth)
sorted_labels = ["60 Hz", "144 Hz", "165 Hz", "240 Hz+"] # 指定堆叠顺序

final_stack_list = []
for label in sorted_labels:
    final_stack_list.append(smooth_data[label])
    known_values_stacked += smooth_data[label]

# Others = 100% - 已知总和
others_smooth = 100 - known_values_stacked
# 修正 Others 可能出现的微小负值（插值误差）
others_smooth = np.maximum(others_smooth, 0)

# 将 Others 加入堆叠列表顶部
final_stack_list.append(others_smooth)
final_labels = sorted_labels + ["Others"]

# 5. 配色方案 (保持风格：冷->暖)
colors = [
    "#34495E", # 60 Hz: 深蓝灰 (旧时代背景)
    "#1ABC9C", # 144 Hz: 青绿色 (曾经的主流，开始消退)
    "#FF8C00", # 165 Hz: 深橙色 (最显眼的增长)
    "#E74C3C", # 240 Hz+: 亮红色 (顶端性能)
    "#E0E0E0"  # Others: 浅灰 (杂项，如 75Hz, 100Hz 等)
]

# 6. 绘图
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

ax.stackplot(x_smooth, final_stack_list, labels=final_labels, colors=colors, alpha=0.9, edgecolor='none')

# 7. 细节优化：增加波动感的视觉引导
# 绘制 2023-2024 的分界线
ax.axvline(x=2023.0, color='white', linestyle='--', linewidth=1, alpha=0.6)
ax.text(2021.5, 95, 'Historical Volatility', color='white', fontsize=10, ha='center', weight='bold')
ax.text(2024.0, 95, 'Market Forecast', color='white', fontsize=10, ha='center', weight='bold')

# 设置标题和标签
ax.set_title('Global Monitor Refresh Rate Share (2021-2025)',
             fontsize=16, fontweight='bold', loc='center', pad=10, ha='center', x=0.55)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Market Share (%)', fontsize=11)

# 坐标轴设置
ax.set_xlim(2021, 2025)
ax.set_ylim(0, 100)
ax.set_xticks(years_points)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.2)

# 图例设置 (高刷在最上面)
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels_leg[::-1], loc='center left', bbox_to_anchor=(1, 0.5), 
          frameon=False, title="Refresh Rate Tier", fontsize=10)

# 8. 保存
plt.tight_layout()
plt.savefig('Refresh_Rate_Volatile_Trend.png', dpi=300, bbox_inches='tight')
plt.savefig('Refresh_Rate_Volatile_Trend.pdf', bbox_inches='tight')

print("已生成带市场波动特征的图表：Refresh_Rate_Volatile_Trend.png/pdf")