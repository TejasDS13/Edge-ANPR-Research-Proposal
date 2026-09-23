import os
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------
# PLOT 1: Latency vs. Precision (Figure 2) - Fixed Overlap
# -------------------------------------------------------------
models = ['Faster R-CNN', 'SSD-MobileNetV2', 'YOLOv8-Small', 'Proposed Backbone']
latency = [82.4, 14.1, 16.5, 8.2]      # ms
mAP = [97.4, 88.2, 96.8, 97.1]          # %

plt.figure(figsize=(7.5, 4.5), dpi=300)
colors = ['#e74c3c', '#95a5a6', '#f39c12', '#2ecc71']

# Custom text positions (x_offset, y_offset, alignment) to prevent overlap
# Custom text positions (x_offset, y_offset, alignment) to prevent border clipping
offsets = [
    (-2.0, 0.6, 'right'),   # Faster R-CNN
    (2.0, -0.6, 'left'),    # SSD-MobileNetV2
    (2.0, -0.6, 'left'),    # YOLOv8-Small
    (1.5, 0.6, 'left')      # Proposed Backbone (shifted right & above dot)
]

for i in range(len(models)):
    plt.scatter(latency[i], mAP[i], color=colors[i], s=130, edgecolors='black', linewidth=1.2, zorder=5)
    ox, oy, align = offsets[i]
    plt.annotate(
        models[i], 
        (latency[i] + ox, mAP[i] + oy), 
        fontsize=9, 
        fontweight='bold',
        ha=align,
        va='center'
    )

plt.title("Inference Latency vs. Detection Precision (mAP@0.5)", fontsize=11, fontweight='bold', pad=12)
plt.xlabel("Latency on Edge Platform (ms) [Lower is Better]", fontsize=10)
plt.ylabel("mAP@0.5 (%) [Higher is Better]", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(-5, 95)
plt.ylim(85, 100)
plt.tight_layout()
plt.savefig("figures/detection_latency_tradeoff.png")
plt.close()

# -------------------------------------------------------------
# PLOT 2: Character Error Rate under Perspective Skew (Figure 3)
# -------------------------------------------------------------
methods = ['Tesseract v5', 'EasyOCR', 'CRNN Baseline', 'Proposed Pipeline']
skew_cer = [41.2, 18.5, 14.8, 2.6]

plt.figure(figsize=(7, 4), dpi=300)
bars = plt.bar(methods, skew_cer, color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'], width=0.48, edgecolor='black', linewidth=1.1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.title("Character Error Rate (CER) under Severe Perspective Skew (>35°)", fontsize=11, fontweight='bold', pad=12)
plt.ylabel("Character Error Rate (%) [Lower is Better]", fontsize=10)
plt.ylim(0, 50)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figures/cer_skew_distribution.png")
plt.close()

print("Plots successfully re-generated with clean label positioning!")

# -------------------------------------------------------------
# PLOT 3: Character Error Rate (CER) Range under Skew (Figure 3)
# -------------------------------------------------------------
methods = ['Tesseract v5\n(Segmented)', 'EasyOCR\n(CRAFT+ResNet)', 'CRNN Baseline\n(Unrectified)', 'Proposed Pipeline\n(TPS+CRNN+CMVR)']
midpoints = [42.5, 19.0, 15.0, 2.75]
labels = ['40% – 45%', '18% – 20%', '14% – 16%', '2.0% – 3.5%']
colors = ['#e74c3c', '#e67e22', '#2980b9', '#27ae60']

plt.figure(figsize=(7, 3.8), dpi=300)
bars = plt.bar(methods, midpoints, color=colors, edgecolor='black', alpha=0.85, width=0.45)

plt.ylabel('Estimated CER (%) [Lower is Better]', fontweight='bold', fontsize=9)
plt.title('Projected CER Distribution under Severe Perspective Skew (>35°)', fontweight='bold', fontsize=10, pad=10)
plt.ylim(0, 50)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Range labels exactly matching Table 4.3
for bar, label in zip(bars, labels):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.2, label, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/cer_skew_distribution.png")
plt.close()

print("Figure 3 (cer_skew_distribution.png) updated with consistent ranges!")

import numpy as np
# -------------------------------------------------------------
# PLOT 4: Hardware Power vs. FPS Efficiency (Figure 4)
# -------------------------------------------------------------
devices = ['Raspberry Pi 4B\n(CPU)', 'Jetson Nano\n(Maxwell)', 'Jetson Orin Nano\n(Ampere)']
power_w = [5, 10, 15]
fps = [11, 28, 88]

fig, ax1 = plt.subplots(figsize=(7, 3.5), dpi=300)

x = np.arange(len(devices))
width = 0.35

color1 = '#e74c3c'
color2 = '#2ecc71'

rects1 = ax1.bar(x - width/2, power_w, width, label='Thermal Dissipation (Watts)', color=color1, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Power Consumption (Watts)', color=color1, fontweight='bold', fontsize=9)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 20)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, fps, width, label='Throughput (FPS)', color=color2, edgecolor='black', alpha=0.85)
ax2.set_ylabel('Inference Throughput (FPS)', color=color2, fontweight='bold', fontsize=9)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 105)

ax1.set_xticks(x)
ax1.set_xticklabels(devices, fontsize=9, fontweight='semibold')
plt.title('Edge Hardware: Power Envelope vs. Pipeline Throughput', fontsize=10, fontweight='bold', pad=10)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# Data labels
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h}W', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h} FPS', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/hardware_efficiency.png")
plt.close()

print("Figure 4 (hardware_efficiency.png) successfully generated!")