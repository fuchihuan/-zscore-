import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

artifacts_dir = r"C:\Users\chihuan\.gemini\antigravity-ide\brain\d2afa23f-f21e-4fb8-a40f-186608353862"

df_1h = pd.read_csv('data/crude_oil_1y_1h.csv', index_col=0)
spr = df_1h['Brent'] - df_1h['WTI']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), gridspec_kw={'height_ratios': [1.3, 1]})

# Panel 1: Time series with absolute entry bands
ax1.plot(pd.to_datetime(spr.index, utc=True), spr, color='#2a9d8f', lw=1.2, label='布蘭特 - WTI 實際價差 ($/桶)')
ax1.axhline(spr.mean(), color='#e76f51', lw=2, linestyle='-', label=f'平均價差: ${spr.mean():.2f} /桶')
ax1.axhline(spr.median(), color='#f4a261', lw=1.5, linestyle='--', label=f'中位數價差: ${spr.median():.2f} /桶')

# Highlight zones
ax1.axhspan(7.25, spr.max()+0.5, color='#d62828', alpha=0.15, label='做空價差區間 (價差過高，賣Brent買WTI: >= $7.25)')
ax1.axhspan(spr.min()-0.5, 3.0, color='#06d6a0', alpha=0.15, label='做多價差區間 (價差過低，買Brent賣WTI: <= $3.00)')
ax1.axhspan(3.64, 5.53, color='#457b9d', alpha=0.1, label='常態核心震盪區 (25%~75%: $3.64~$5.53)')

ax1.set_title('過去一年 CME 原油價差 (Brent - WTI) 走勢與收斂套利進場區間', fontsize=14, fontweight='bold', pad=10)
ax1.set_ylabel('價差 (USD / 桶)', fontsize=11)
ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel 2: Distribution histogram
counts, bins, patches = ax2.hist(spr, bins=60, color='#457b9d', edgecolor='white', alpha=0.75, density=True)

# Colorize specific bins
for patch, b_left, b_right in zip(patches, bins[:-1], bins[1:]):
    if b_right <= 3.0:
        patch.set_facecolor('#06d6a0')
    elif b_left >= 7.25:
        patch.set_facecolor('#d62828')
    elif 3.64 <= b_left and b_right <= 5.53:
        patch.set_facecolor('#2a9d8f')

ax2.axvline(spr.mean(), color='#e76f51', lw=2, linestyle='-', label=f'平均價差: ${spr.mean():.2f}')
ax2.axvline(3.0, color='#06d6a0', lw=1.8, linestyle='--', label='做多進場警戒線 ($3.00)')
ax2.axvline(7.25, color='#d62828', lw=1.8, linestyle='--', label='做空進場警戒線 ($7.25)')
ax2.axvline(0.0, color='black', lw=1.2, linestyle=':', label='零軸倒掛線 ($0.00)')

ax2.set_title('價差機率分佈 (綠色: 做多區間 ｜ 藍色: 常態區間 ｜ 紅色: 做空區間)', fontsize=12, fontweight='bold')
ax2.set_xlabel('價差 (USD / 桶)', fontsize=11)
ax2.set_ylabel('機率密度', fontsize=11)
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
dist_path = os.path.join(artifacts_dir, 'crude_oil_spread_distribution.png')
plt.savefig(dist_path, dpi=200, bbox_inches='tight')
print(f"Distribution chart saved to {dist_path}")
