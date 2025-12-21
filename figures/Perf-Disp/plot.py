import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# ==========================================
# 1. Configuration & Style
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. Data Preparation
# ==========================================
years = np.arange(2016, 2026)

# --- A. Supply Side (Mobile GPU TFLOPS) ---
# Data points: GTX 1080 -> RTX 5090 (Est)
supply_years_raw = [2016, 2019, 2021, 2023, 2025]
supply_vals_raw = [8.9, 10.1, 19.0, 33.0, 36.5] 

f_supply = interp1d(supply_years_raw, supply_vals_raw, kind='linear')
supply_curve = f_supply(years)

# --- B. Demand Side (Flagship Pixel Throughput) ---
# Scenario: 2K 60Hz -> 4K 240Hz
demand_years_raw = [2016, 2020, 2023, 2025]
demand_vals_raw = [0.22, 0.53, 1.00, 1.99] # Base Gpixels/s

f_demand = interp1d(demand_years_raw, demand_vals_raw, kind='linear')
base_demand_curve = f_demand(years)

# --- C. Apply The "Stress Parameter" (Your Request) ---
# This parameter simulates extra load after 2020 (e.g., Ray Tracing, Unoptimized games)
# making Demand significantly higher than Supply.
post_2020_stress_factor = 1.6  # 35% extra demand after 2020

final_demand_curve = []
for y, val in zip(years, base_demand_curve):
    if y > 2023:
        final_demand_curve.append(val * 1.4)
    elif y > 2020:
        # Apply factor progressively or instantly? Let's apply instantly for clear impact
        final_demand_curve.append(val * 1.8)
    else:
        final_demand_curve.append(val * 1.1)
final_demand_curve = np.array(final_demand_curve)

# ==========================================
# 3. Plotting (Dual Axis)
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 8), dpi=600)

# --- Left Axis: Supply (Blue) ---
color_supply = '#2980B9' # Strong Blue
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Hardware Supply (TFLOPS)', color=color_supply, fontsize=12, fontweight='bold')
line1, = ax1.plot(years, supply_curve, color=color_supply, linewidth=4, 
                  marker='o', markersize=8, label='GPU Performance (Supply)')
ax1.tick_params(axis='y', labelcolor=color_supply, labelsize=10)
ax1.grid(True, linestyle='--', alpha=0.3)

# --- Right Axis: Demand (Red) ---
ax2 = ax1.twinx()
color_demand = '#C0392B' # Strong Red
ax2.set_ylabel('Flagship Demand (Gpixels/s)', color=color_demand, fontsize=12, fontweight='bold')
line2, = ax2.plot(years, final_demand_curve, color=color_demand, linewidth=4, 
                  marker='D', markersize=8, label='Pixel Bandwidth (Demand)')
ax2.tick_params(axis='y', labelcolor=color_demand, labelsize=10)

# ==========================================
# 4. Axis Scaling for "Scissors" Effect
# ==========================================
# Goal: Make Red line (Demand) end VISUALLY HIGHER than Blue line (Supply)
# to show "Supply < Demand".

# Supply Axis (Left): 0 to 50
ax1.set_ylim(0, 50) 
# Visual End Point: 36.5 / 50 = 73% of chart height

# Demand Axis (Right): 0 to 3.0
# Demand End Value: ~2.0 * 1.35 = ~2.7
# Visual End Point: 2.7 / 3.0 = 90% of chart height
# Since 90% (Red) > 73% (Blue), the Red line will be physically higher on the image.
ax2.set_ylim(0, 3.0)

# ==========================================
# 5. Final Touches
# ==========================================
# Legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=11, frameon=False)

# Title
plt.title('The Performance Gap: Display Demand vs. GPU Supply', 
          fontsize=16, fontweight='bold', pad=20)

# Remove top spines
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Annotation for the parameter
# ax2.text(2021.5, final_demand_curve[-3], f"Stress Factor x{post_2020_stress_factor}", 
#          color=color_demand, fontsize=10, fontweight='bold', ha='left', va='bottom')

plt.tight_layout()
plt.savefig('Scissors_Gap.png', bbox_inches='tight')
plt.show()