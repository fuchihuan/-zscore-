"""
plot_yearly_spread_stats.py
統計 2020 年以來的 WTI 與 Brent 原油期貨價差走勢
包含每年平均、最高、最低、中位數、標準差，並繪製專業機構級 4 面板視覺化圖表
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# 設定繁體中文字型支援與負號修正
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

CSV_FILE = 'wti_brent_daily_spread_2020_2026.csv'
ARTIFACT_DIR = r'C:\Users\chihuan\.gemini\antigravity-ide\brain\d2afa23f-f21e-4fb8-a40f-186608353862'
OUT_PNG = os.path.join(ARTIFACT_DIR, 'crude_oil_yearly_spread_analysis_2020_2026.png')

def main():
    df = pd.read_csv(CSV_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Year'] = df['Date'].dt.year

    # 1. 每年統計指標
    yearly_stats = []
    for yr, grp in df.groupby('Year'):
        max_idx = grp['Spread_USD'].idxmax()
        min_idx = grp['Spread_USD'].idxmin()
        yearly_stats.append({
            'Year': yr,
            'Days': len(grp),
            'Mean': grp['Spread_USD'].mean(),
            'Median': grp['Spread_USD'].median(),
            'Min': grp['Spread_USD'].min(),
            'Min_Date': df.loc[min_idx, 'Date'].strftime('%Y-%m-%d'),
            'Max': grp['Spread_USD'].max(),
            'Max_Date': df.loc[max_idx, 'Date'].strftime('%Y-%m-%d'),
            'Std': grp['Spread_USD'].std(),
            'WTI_Mean': grp['WTI_Close'].mean(),
            'Brent_Mean': grp['Brent_Close'].mean()
        })
    stats_df = pd.DataFrame(yearly_stats)
    print("=== 各年份價差統計表 ===")
    print(stats_df.to_string(index=False))

    # 全期間指標
    total_mean = df['Spread_USD'].mean()
    total_median = df['Spread_USD'].median()
    total_std = df['Spread_USD'].std()
    total_max = df['Spread_USD'].max()
    total_min = df['Spread_USD'].min()

    # 2. 建立畫布 (4 個子圖)
    fig = plt.figure(figsize=(16, 13), facecolor='#0b0f19')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.2, 1.0], hspace=0.32, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, :])      # 上：兩大油價歷史走勢
    ax2 = fig.add_subplot(gs[1, :])      # 中：每日價差走勢與年度均值
    ax3 = fig.add_subplot(gs[2, 0])      # 下左：各年度平均與高低價差柱狀圖
    ax4 = fig.add_subplot(gs[2, 1])      # 下右：各年度價差盒鬚分佈圖 (Boxplot)

    dark_bg = '#111827'
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors='#94a3b8', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle='--', alpha=0.25, color='#64748b')

    # -------------------------------------------------------------
    # 子圖 1: Brent vs WTI 歷史收盤價走勢
    # -------------------------------------------------------------
    ax1.plot(df['Date'], df['Brent_Close'], label='CME Brent 原油 (BZ)', color='#f472b6', linewidth=1.5, alpha=0.9)
    ax1.plot(df['Date'], df['WTI_Close'], label='CME WTI 輕原油 (CL)', color='#38bdf8', linewidth=1.4, alpha=0.9)
    ax1.set_title('2020～2026 CME 原油期貨雙雄歷史走勢 (WTI vs. Brent)', color='#f8fafc', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('美元 / 桶 ($/Bbl)', color='#cbd5e1', fontsize=11)
    ax1.legend(loc='upper left', facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=10)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.axhline(0, color='#64748b', linestyle=':', alpha=0.5)

    # 標註歷史特殊事件
    ax1.annotate('2020 負油價事件\nWTI 暴跌至 -$37.63', xy=(pd.to_datetime('2020-04-20'), -37.63),
                 xytext=(pd.to_datetime('2020-09-01'), -20),
                 arrowprops=dict(facecolor='#f87171', edgecolor='#f87171', width=1, headwidth=6),
                 color='#fca5a5', fontsize=9, fontweight='bold')

    ax1.annotate('2022 俄烏衝突爆發\n油價雙雙突破 $120', xy=(pd.to_datetime('2022-03-08'), 127.98),
                 xytext=(pd.to_datetime('2022-07-01'), 135),
                 arrowprops=dict(facecolor='#fbbf24', edgecolor='#fbbf24', width=1, headwidth=6),
                 color='#fde68a', fontsize=9, fontweight='bold')

    # -------------------------------------------------------------
    # 子圖 2: 每日價差走勢 (Brent - WTI) 與年度階梯均值
    # -------------------------------------------------------------
    # 為了圖表易讀性，將 2020 年極端負油價當日的 $63 價差截斷顯示標註，Y 軸聚焦於 -$5 ~ +$20
    ax2.plot(df['Date'], df['Spread_USD'], color='#2dd4bf', linewidth=1.3, label='每日實際價差 (Brent - WTI)')
    ax2.axhline(total_mean, color='#fbbf24', linestyle='--', linewidth=1.8, label=f'6.7 年歷史總均值 (${total_mean:.2f})')
    ax2.axhline(total_mean + 2 * total_std, color='#f87171', linestyle=':', linewidth=1.2, label=f'+2σ 做空收斂線 (${total_mean + 2*total_std:.2f})')
    ax2.axhline(total_mean - 2 * total_std, color='#34d399', linestyle=':', linewidth=1.2, label=f'-2σ 做多收斂線 (${total_mean - 2*total_std:.2f})')
    ax2.axhline(0, color='#94a3b8', linestyle='-', linewidth=0.8, alpha=0.6)

    # 繪製各年份年度均值階梯
    for _, row in stats_df.iterrows():
        yr = int(row['Year'])
        sub_d = df[df['Year'] == yr]
        ax2.hlines(y=row['Mean'], xmin=sub_d['Date'].min(), xmax=sub_d['Date'].max(),
                   color='#c084fc', linewidth=2.5, alpha=0.9)
        # 標註各年度均值文字
        mid_d = sub_d['Date'].iloc[len(sub_d)//2]
        ax2.text(mid_d, row['Mean'] + 0.6, f"{yr}均: ${row['Mean']:.2f}",
                 color='#e9d5ff', fontsize=8.5, ha='center', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#581c87', alpha=0.7, edgecolor='none'))

    ax2.set_ylim(-5, 20)
    ax2.set_title('2020～2026 每日價差走勢 (Brent - WTI) 與各年度中樞階梯帶', color='#f8fafc', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel('價差 (USD/桶)', color='#cbd5e1', fontsize=11)
    ax2.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax2.annotate('2020-04-20 負油價衝至 $63.2\n(圖表聚焦常態區截斷)', xy=(pd.to_datetime('2020-04-20'), 19.5),
                 xytext=(pd.to_datetime('2020-07-01'), 16.5),
                 arrowprops=dict(facecolor='#f87171', edgecolor='#f87171', width=1, headwidth=5),
                 color='#fca5a5', fontsize=8.5)

    ax2.annotate('2026-03-31 擴大至 $16.97', xy=(pd.to_datetime('2026-03-31'), 16.97),
                 xytext=(pd.to_datetime('2025-08-01'), 17.5),
                 arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8', width=1, headwidth=5),
                 color='#7dd3fc', fontsize=8.5)

    # -------------------------------------------------------------
    # 子圖 3: 各年度平均、最高、最低價差柱狀圖
    # -------------------------------------------------------------
    years = stats_df['Year'].tolist()
    x = np.arange(len(years))
    bar_width = 0.25

    # 為了避免 2020 年的 63.2 破壞比例，長條圖標註實際值但設置 Y 軸上限
    rects1 = ax3.bar(x - bar_width, stats_df['Max'], bar_width, label='每年最高價差 (Max)', color='#f87171', alpha=0.85)
    rects2 = ax3.bar(x, stats_df['Mean'], bar_width, label='每年平均價差 (Mean)', color='#fbbf24', alpha=0.9)
    rects3 = ax3.bar(x + bar_width, stats_df['Min'], bar_width, label='每年最低價差 (Min)', color='#34d399', alpha=0.85)

    ax3.set_title('各年度價差極值與平均值對照 (2020～2026)', color='#f8fafc', fontsize=12.5, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(y) for y in years], color='#e2e8f0', fontweight='bold')
    ax3.set_ylabel('美元 / 桶 ($)', color='#cbd5e1', fontsize=10.5)
    ax3.set_ylim(-6, 22)
    ax3.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8.5)
    ax3.axhline(0, color='#64748b', linestyle='-', linewidth=0.8)

    # 柱狀上方數值標註
    for i, r in stats_df.iterrows():
        # Mean
        ax3.text(x[i], r['Mean'] + 0.4, f"${r['Mean']:.2f}", ha='center', color='#fbbf24', fontsize=8.5, fontweight='bold')
        # Min
        y_pos_min = r['Min'] - 1.1 if r['Min'] < 0 else r['Min'] - 0.9
        ax3.text(x[i] + bar_width, y_pos_min, f"${r['Min']:.1f}", ha='center', color='#34d399', fontsize=8)
        # Max
        if r['Max'] > 20:
            ax3.text(x[i] - bar_width, 19.5, f"${r['Max']:.1f}*", ha='center', color='#f87171', fontsize=8, fontweight='bold')
        else:
            ax3.text(x[i] - bar_width, r['Max'] + 0.4, f"${r['Max']:.1f}", ha='center', color='#f87171', fontsize=8)

    # -------------------------------------------------------------
    # 子圖 4: 各年度價差盒鬚分佈圖 (Boxplot)
    # -------------------------------------------------------------
    yearly_data = [df[df['Year'] == yr]['Spread_USD'].values for yr in years]
    bplot = ax4.boxplot(yearly_data, patch_artist=True, tick_labels=[str(y) for y in years],
                        showfliers=False, # 隱藏極端離群值以突顯箱體分佈
                        medianprops=dict(color='#fbbf24', linewidth=2),
                        whiskerprops=dict(color='#94a3b8', linewidth=1.2),
                        capprops=dict(color='#94a3b8', linewidth=1.2))

    # 彩色填充箱體
    box_colors = ['#3b82f6', '#06b6d4', '#10b981', '#6366f1', '#ec4899', '#f97316', '#a855f7']
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor('#ffffff')

    ax4.set_title('各年度價差常態分佈箱型圖 (Boxplot 25%~75%)', color='#f8fafc', fontsize=12.5, fontweight='bold', pad=10)
    ax4.set_ylabel('美元 / 桶 ($)', color='#cbd5e1', fontsize=10.5)
    ax4.tick_params(colors='#e2e8f0', labelsize=10)
    ax4.axhline(total_mean, color='#fbbf24', linestyle='--', linewidth=1.2, alpha=0.8, label=f'歷史總均值 ${total_mean:.2f}')
    ax4.legend(loc='upper right', facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)

    # 總體浮水印與說明
    fig.suptitle('CME 原油期貨 (Brent - WTI) 2020～2026 年價差多維統計與走勢全貌分析',
                 color='#ffffff', fontsize=17, fontweight='bold', y=0.985)

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(OUT_PNG, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\n[OK] 圖表已成功儲存至: {OUT_PNG}")

if __name__ == '__main__':
    main()
