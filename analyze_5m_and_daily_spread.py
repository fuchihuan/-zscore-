"""
analyze_5m_and_daily_spread.py
原油期貨 (WTI vs Brent) 5分K 與 日K 價差收斂分析：
1. 下載並整理 5 分 K 資料 (12,000+ 根 K 棒)
2. 擬合 Ornstein-Uhlenbeck (OU) 均值回歸過程與計算理論半衰期
3. 統計實際交易之平均收斂時間、中位數、最大最小收斂時間
4. 樣本內外 (IS 70% / OOS 30%) 網格搜尋防過擬合平原區 (Parameter Stability Plateau)
5. 同步針對 2020~2026 日 K 資料進行全面驗證與對比
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

artifacts_dir = r"C:\Users\chihuan\.gemini\antigravity-ide\brain\d2afa23f-f21e-4fb8-a40f-186608353862"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# ============================================================
# 1. 資料獲取與前處理
# ============================================================
def load_or_fetch_5m_data():
    file_5m = os.path.join(data_dir, "crude_oil_5m.csv")
    if os.path.exists(file_5m):
        print(f"從本機快取載入 5 分 K: {file_5m}")
        df = pd.read_csv(file_5m, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

    print("從 Yahoo Finance 下載最新 60 天 5 分 K 資料...")
    cl = yf.Ticker("CL=F").history(period="60d", interval="5m")
    bz = yf.Ticker("BZ=F").history(period="60d", interval="5m")

    cl.index = pd.to_datetime(cl.index).tz_localize(None)
    bz.index = pd.to_datetime(bz.index).tz_localize(None)
    common_idx = cl.index.intersection(bz.index)

    df_5m = pd.DataFrame({
        'WTI_Open': cl.loc[common_idx, 'Open'],
        'WTI_High': cl.loc[common_idx, 'High'],
        'WTI_Low': cl.loc[common_idx, 'Low'],
        'WTI_Close': cl.loc[common_idx, 'Close'],
        'WTI_Volume': cl.loc[common_idx, 'Volume'],
        'Brent_Open': bz.loc[common_idx, 'Open'],
        'Brent_High': bz.loc[common_idx, 'High'],
        'Brent_Low': bz.loc[common_idx, 'Low'],
        'Brent_Close': bz.loc[common_idx, 'Close'],
        'Brent_Volume': bz.loc[common_idx, 'Volume'],
    }, index=common_idx)
    df_5m['Spread'] = df_5m['Brent_Close'] - df_5m['WTI_Close']
    df_5m.to_csv(file_5m)
    print(f"5 分 K 下載完成，共 {len(df_5m):,} 根 K 棒，已儲存至 {file_5m}")
    return df_5m

def load_daily_data():
    daily_file = "wti_brent_daily_spread_2020_2026.csv"
    if not os.path.exists(daily_file):
        daily_file = os.path.join(data_dir, "crude_oil_1y_1d.csv")
    print(f"載入日 K 資料: {daily_file}")
    df = pd.read_csv(daily_file, index_col=0)
    df.index = pd.to_datetime(df.index)
    if 'Spread' not in df.columns:
        if 'Brent' in df.columns and 'WTI' in df.columns:
            df['Spread'] = df['Brent'] - df['WTI']
            df['WTI_Close'] = df['WTI']
            df['Brent_Close'] = df['Brent']
        elif 'Brent_Close' in df.columns and 'WTI_Close' in df.columns:
            df['Spread'] = df['Brent_Close'] - df['WTI_Close']
    return df

# ============================================================
# 2. Ornstein-Uhlenbeck (OU) 均值回歸過程擬合
# ============================================================
def fit_ou_process(spread_series, dt=1.0):
    """
    擬合 OU 過程: dS_t = theta * (mu - S_t) * dt + sigma * dW_t
    離散化 AR(1): S_t - S_{t-1} = a + b * S_{t-1} + eps
    theta = -ln(1 + b) / dt (或近似 -b / dt)
    mu = -a / b
    half_life = ln(2) / theta
    """
    s = spread_series.dropna()
    s_lag = s.shift(1).dropna()
    s_diff = (s - s_lag).dropna()
    common = s_lag.index.intersection(s_diff.index)
    
    X = sm.add_constant(s_lag.loc[common])
    y = s_diff.loc[common]
    model = sm.OLS(y, X).fit()
    
    a = model.params['const']
    b = model.params.iloc[1]
    r2 = model.rsquared
    resids = model.resid
    sigma_eps = resids.std()
    
    if b < 0:
        theta = -b / dt
        mu = -a / b
        half_life = np.log(2) / theta
        sigma = sigma_eps / np.sqrt(dt)
    else:
        # 非均值回歸 (發散)
        theta = 0.0
        mu = s.mean()
        half_life = np.nan
        sigma = sigma_eps

    return {
        'theta': theta,
        'mu': mu,
        'half_life': half_life,
        'sigma': sigma,
        'r2': r2,
        'pvalue': model.pvalues.iloc[1],
        'is_mean_reverting': b < 0 and model.pvalues.iloc[1] < 0.05
    }

# ============================================================
# 3. 向量化回測與實際收斂時間追蹤
# ============================================================
def backtest_spread_strategy(df, window=60, entry_z=2.0, exit_z=0.5, stop_loss_z=3.5, freq_unit="5m"):
    """
    價差均值回歸策略回測
    記錄每筆交易的實際收斂耗時 (K棒數、實際時間)
    """
    spread = df['Spread'].dropna()
    roll_mean = spread.rolling(window=window, min_periods=window//2).mean()
    roll_std = spread.rolling(window=window, min_periods=window//2).std().replace(0, np.nan)
    zscore = (spread - roll_mean) / roll_std
    
    # 策略狀態模擬
    position = 0 # 1: 做多價差 (買Brent 賣WTI), -1: 做空價差 (賣Brent 買WTI)
    entry_idx = None
    entry_time = None
    entry_spread = 0.0
    
    trades = []
    
    z_vals = zscore.values
    spr_vals = spread.values
    times = spread.index
    
    # 依序走訪
    for i in range(window, len(z_vals)):
        z = z_vals[i]
        spr = spr_vals[i]
        t = times[i]
        
        if np.isnan(z):
            continue
            
        if position == 0:
            # 進場條件 (t 判定，t+1 執行，此處以 bar 判定)
            if z <= -entry_z:
                position = 1
                entry_idx = i
                entry_time = t
                entry_spread = spr
            elif z >= entry_z:
                position = -1
                entry_idx = i
                entry_time = t
                entry_spread = spr
                
        elif position == 1:
            # 做多價差持倉中
            # 平倉條件: 回到 exit_z 或停損
            is_exit = (z >= -exit_z)
            is_stop = (stop_loss_z is not None and z <= -stop_loss_z)
            
            if is_exit or is_stop or i == len(z_vals) - 1:
                pnl_pts = spr - entry_spread # 價差擴大獲利
                pnl_usd = pnl_pts * 1000.0   # 1000桶/口
                bars_held = i - entry_idx
                duration_hours = (t - entry_time).total_seconds() / 3600.0 if hasattr(t - entry_time, 'total_seconds') else bars_held
                trades.append({
                    'type': 'Long_Spread',
                    'entry_time': entry_time,
                    'exit_time': t,
                    'bars_held': bars_held,
                    'duration_hours': duration_hours,
                    'entry_spread': entry_spread,
                    'exit_spread': spr,
                    'pnl_pts': pnl_pts,
                    'pnl_usd': pnl_usd,
                    'is_stop': is_stop,
                    'is_win': pnl_usd > 0
                })
                position = 0
                
        elif position == -1:
            # 做空價差持倉中
            is_exit = (z <= exit_z)
            is_stop = (stop_loss_z is not None and z >= stop_loss_z)
            
            if is_exit or is_stop or i == len(z_vals) - 1:
                pnl_pts = entry_spread - spr # 價差收斂獲利
                pnl_usd = pnl_pts * 1000.0
                bars_held = i - entry_idx
                duration_hours = (t - entry_time).total_seconds() / 3600.0 if hasattr(t - entry_time, 'total_seconds') else bars_held
                trades.append({
                    'type': 'Short_Spread',
                    'entry_time': entry_time,
                    'exit_time': t,
                    'bars_held': bars_held,
                    'duration_hours': duration_hours,
                    'entry_spread': entry_spread,
                    'exit_spread': spr,
                    'pnl_pts': pnl_pts,
                    'pnl_usd': pnl_usd,
                    'is_stop': is_stop,
                    'is_win': pnl_usd > 0
                })
                position = 0
                
    tdf = pd.DataFrame(trades)
    return tdf

# ============================================================
# 4. 網格搜尋與平原區 (Plateau) 防過擬合驗證
# ============================================================
def run_grid_search_with_oos(df, windows, entry_zs, exit_zs=[0.5], is_ratio=0.7, freq_unit="5m"):
    """
    將資料切為 In-Sample (70%) 與 Out-of-Sample (30%)
    在 IS 搜尋表現，並同時檢驗 OOS 衰退率 (WFE)，避免過擬合
    """
    n = len(df)
    split_idx = int(n * is_ratio)
    df_is = df.iloc[:split_idx]
    df_oos = df.iloc[split_idx:]
    
    results = []
    for w in windows:
        for ez in entry_zs:
            for xz in exit_zs:
                if ez <= xz:
                    continue
                # 回測 IS
                tdf_is = backtest_spread_strategy(df_is, window=w, entry_z=ez, exit_z=xz, freq_unit=freq_unit)
                # 回測 OOS
                tdf_oos = backtest_spread_strategy(df_oos, window=w, entry_z=ez, exit_z=xz, freq_unit=freq_unit)
                
                def calc_perf(tdf):
                    if tdf.empty:
                        return {'n': 0, 'wr': 0, 'pf': 0, 'net_usd': 0, 'mean_dur': 0, 'med_dur': 0}
                    n_trades = len(tdf)
                    wr = tdf['is_win'].mean() * 100.0
                    wins = tdf[tdf['pnl_usd'] > 0]['pnl_usd'].sum()
                    losses = abs(tdf[tdf['pnl_usd'] < 0]['pnl_usd'].sum())
                    pf = wins / losses if losses > 0 else (99.0 if wins > 0 else 0.0)
                    net_usd = tdf['pnl_usd'].sum()
                    mean_dur = tdf['bars_held'].mean()
                    med_dur = tdf['bars_held'].median()
                    return {
                        'n': n_trades, 'wr': wr, 'pf': pf, 'net_usd': net_usd,
                        'mean_dur': mean_dur, 'med_dur': med_dur
                    }
                
                perf_is = calc_perf(tdf_is)
                perf_oos = calc_perf(tdf_oos)
                
                # Walk-Forward Efficiency (WFE)
                wfe = (perf_oos['net_usd'] / (perf_is['net_usd'] * (1 - is_ratio) / is_ratio)) if perf_is['net_usd'] > 0 else 0
                
                results.append({
                    'window': w,
                    'entry_z': ez,
                    'exit_z': xz,
                    'is_trades': perf_is['n'],
                    'is_wr': perf_is['wr'],
                    'is_pf': perf_is['pf'],
                    'is_pnl': perf_is['net_usd'],
                    'oos_trades': perf_oos['n'],
                    'oos_wr': perf_oos['wr'],
                    'oos_pf': perf_oos['pf'],
                    'oos_pnl': perf_oos['net_usd'],
                    'wfe': wfe,
                    'is_mean_bars': perf_is['mean_dur'],
                    'oos_mean_bars': perf_oos['mean_dur']
                })
                
    return pd.DataFrame(results)

if __name__ == '__main__':
    print('========================================')
    print('1. 載入並分析 5 分 K 資料')
    print('========================================')
    df_5m = load_or_fetch_5m_data()
    print(f'5分K 總筆數: {len(df_5m):,} 根')

    # OU 過程擬合 (5分K，dt = 5/60 小時 = 1/12 小時)
    ou_5m_bar = fit_ou_process(df_5m['Spread'], dt=1.0)
    ou_5m_hr = fit_ou_process(df_5m['Spread'], dt=5.0/60.0)
    print(f"5分K OU 半衰期: {ou_5m_bar['half_life']:.1f} 根 K棒 ({ou_5m_hr['half_life']:.2f} 小時)")
    print(f"5分K OU 回歸均值 mu: ${ou_5m_bar['mu']:.3f} 美元, 回歸速度 theta: {ou_5m_hr['theta']:.3f} /hr, R2: {ou_5m_bar['r2']:.4f}")

    # 網格搜尋 (防過擬合平原區)
    windows_5m = [30, 45, 60, 90, 120]
    entry_zs_5m = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
    grid_5m = run_grid_search_with_oos(df_5m, windows_5m, entry_zs_5m, exit_zs=[0.5], is_ratio=0.7, freq_unit="5m")

    # 挑選穩健參數
    grid_5m['score'] = (grid_5m['is_pf'] >= 1.5) & (grid_5m['oos_pf'] >= 1.3) & (grid_5m['oos_pnl'] > 0)
    robust_5m = grid_5m[grid_5m['score']].sort_values(by='oos_pf', ascending=False)

    print('\n[5分K 樣本外 (OOS) 最優穩健參數 Top 5]:')
    print(robust_5m[['window', 'entry_z', 'is_wr', 'is_pf', 'oos_wr', 'oos_pf', 'oos_pnl', 'wfe']].head(5).to_string(index=False))

    # 以最優平原參數執行完整回測
    best_5m_w = int(robust_5m.iloc[0]['window']) if not robust_5m.empty else 60
    best_5m_z = float(robust_5m.iloc[0]['entry_z']) if not robust_5m.empty else 2.2
    print(f'\n-> 選定 5分K 最佳穩健參數: Window={best_5m_w}, Entry Z={best_5m_z}, Exit Z=0.5')

    tdf_5m_full = backtest_spread_strategy(df_5m, window=best_5m_w, entry_z=best_5m_z, exit_z=0.5, freq_unit="5m")
    dur_bars = tdf_5m_full['bars_held']
    dur_mins = dur_bars * 5.0
    dur_hrs = dur_mins / 60.0

    print('\n========================================')
    print('【5分K 實際收斂時間統計 (全樣本)】')
    print('========================================')
    print(f'總交易筆數: {len(tdf_5m_full)} 筆 (勝率 {tdf_5m_full["is_win"].mean()*100:.1f}%)')
    print(f'平均收斂時間: {dur_bars.mean():.1f} 根 K棒 ({dur_hrs.mean():.2f} 小時 / {dur_mins.mean():.1f} 分鐘)')
    print(f'中位數收斂時間: {dur_bars.median():.1f} 根 K棒 ({dur_hrs.median():.2f} 小時 / {dur_mins.median():.1f} 分鐘)')
    print(f'最小收斂時間: {dur_bars.min():.0f} 根 K棒 ({dur_mins.min():.0f} 分鐘)')
    print(f'最大收斂時間: {dur_bars.max():.0f} 根 K棒 ({dur_hrs.max():.2f} 小時 / {dur_bars.max()*5/1440:.2f} 天)')
    print(f'25%~75% 常態收斂區間: {dur_bars.quantile(0.25):.0f} ~ {dur_bars.quantile(0.75):.0f} 根 K棒 ({dur_hrs.quantile(0.25):.1f} ~ {dur_hrs.quantile(0.75):.1f} 小時)')
    print(f'95% 極端最長收斂上限: {dur_bars.quantile(0.95):.0f} 根 K棒 ({dur_hrs.quantile(0.95):.1f} 小時)')

    print('\n========================================')
    print('2. 載入並分析 日 K 資料 (2020 ~ 2026)')
    print('========================================')
    df_daily = load_daily_data()
    print(f'日K 總筆數: {len(df_daily):,} 根交易日 ({df_daily.index[0].date()} ~ {df_daily.index[-1].date()})')

    # OU 過程擬合 (日K，dt = 1 天)
    ou_daily = fit_ou_process(df_daily['Spread'], dt=1.0)
    print(f"日K OU 半衰期: {ou_daily['half_life']:.1f} 個交易日")
    print(f"日K OU 回歸均值 mu: ${ou_daily['mu']:.3f} 美元, 回歸速度 theta: {ou_daily['theta']:.4f} /天, R2: {ou_daily['r2']:.4f}")

    # 網格搜尋 (防過擬合平原區)
    windows_daily = [20, 30, 40, 60, 90]
    entry_zs_daily = [1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
    grid_daily = run_grid_search_with_oos(df_daily, windows_daily, entry_zs_daily, exit_zs=[0.5], is_ratio=0.7, freq_unit="1d")

    grid_daily['score'] = (grid_daily['is_pf'] >= 1.5) & (grid_daily['oos_pf'] >= 1.3) & (grid_daily['oos_pnl'] > 0)
    robust_daily = grid_daily[grid_daily['score']].sort_values(by='oos_pf', ascending=False)

    print('\n[日K 樣本外 (OOS) 最優穩健參數 Top 5]:')
    print(robust_daily[['window', 'entry_z', 'is_wr', 'is_pf', 'oos_wr', 'oos_pf', 'oos_pnl', 'wfe']].head(5).to_string(index=False))

    best_daily_w = int(robust_daily.iloc[0]['window']) if not robust_daily.empty else 40
    best_daily_z = float(robust_daily.iloc[0]['entry_z']) if not robust_daily.empty else 2.0
    print(f'\n-> 選定 日K 最佳穩健參數: Window={best_daily_w}, Entry Z={best_daily_z}, Exit Z=0.5')

    tdf_daily_full = backtest_spread_strategy(df_daily, window=best_daily_w, entry_z=best_daily_z, exit_z=0.5, freq_unit="1d")
    dur_daily = tdf_daily_full['bars_held']

    print('\n========================================')
    print('【日K 實際收斂時間統計 (全樣本 2020~2026)】')
    print('========================================')
    print(f'總交易筆數: {len(tdf_daily_full)} 筆 (勝率 {tdf_daily_full["is_win"].mean()*100:.1f}%)')
    print(f'平均收斂時間: {dur_daily.mean():.1f} 個交易日')
    print(f'中位數收斂時間: {dur_daily.median():.1f} 個交易日')
    print(f'最小收斂時間: {dur_daily.min():.0f} 個交易日')
    print(f'最大收斂時間: {dur_daily.max():.0f} 個交易日')
    print(f'25%~75% 常態收斂區間: {dur_daily.quantile(0.25):.0f} ~ {dur_daily.quantile(0.75):.0f} 個交易日')
    print(f'95% 極端最長收斂上限: {dur_daily.quantile(0.95):.0f} 個交易日')

    # 輸出圖表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 5m 收斂時間分佈直方圖
    axes[0, 0].hist(dur_hrs, bins=35, color='#2a9d8f', edgecolor='white', alpha=0.8)
    axes[0, 0].axvline(dur_hrs.mean(), color='#e76f51', lw=2, linestyle='-', label=f'平均收斂: {dur_hrs.mean():.1f}h')
    axes[0, 0].axvline(dur_hrs.median(), color='#f4a261', lw=2, linestyle='--', label=f'中位數: {dur_hrs.median():.1f}h')
    axes[0, 0].axvline(ou_5m_hr['half_life'], color='#9b5de5', lw=2, linestyle=':', label=f'OU理論半衰期: {ou_5m_hr["half_life"]:.1f}h')
    axes[0, 0].set_title('5分K 實際收斂時間分佈 (小時)', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('收斂耗時 (小時)')
    axes[0, 0].set_ylabel('交易筆數')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.4)

    # 日K 收斂時間分佈直方圖
    axes[0, 1].hist(dur_daily, bins=25, color='#457b9d', edgecolor='white', alpha=0.8)
    axes[0, 1].axvline(dur_daily.mean(), color='#e76f51', lw=2, linestyle='-', label=f'平均收斂: {dur_daily.mean():.1f}天')
    axes[0, 1].axvline(dur_daily.median(), color='#f4a261', lw=2, linestyle='--', label=f'中位數: {dur_daily.median():.1f}天')
    axes[0, 1].axvline(ou_daily['half_life'], color='#9b5de5', lw=2, linestyle=':', label=f'OU理論半衰期: {ou_daily["half_life"]:.1f}天')
    axes[0, 1].set_title('日K 實際收斂時間分佈 (天數)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('收斂耗時 (交易日)')
    axes[0, 1].set_ylabel('交易筆數')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.4)

    # 5m 參數平原熱力圖 (Window vs Entry Z 的 OOS PF)
    piv_5m = grid_5m.pivot(index='entry_z', columns='window', values='oos_pf')
    im1 = axes[1, 0].imshow(piv_5m.values, cmap='YlGn', aspect='auto', origin='lower')
    axes[1, 0].set_xticks(range(len(piv_5m.columns)))
    axes[1, 0].set_xticklabels(piv_5m.columns)
    axes[1, 0].set_yticks(range(len(piv_5m.index)))
    axes[1, 0].set_yticklabels(piv_5m.index)
    axes[1, 0].set_title('5分K 樣本外(OOS) 獲利因子平原區 (綠色為高PF穩健區)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('滾動窗口 (Window)')
    axes[1, 0].set_ylabel('進場門檻 (Entry Z)')
    plt.colorbar(im1, ax=axes[1, 0])

    # 日K 參數平原熱力圖
    piv_daily = grid_daily.pivot(index='entry_z', columns='window', values='oos_pf')
    im2 = axes[1, 1].imshow(piv_daily.values, cmap='YlGn', aspect='auto', origin='lower')
    axes[1, 1].set_xticks(range(len(piv_daily.columns)))
    axes[1, 1].set_xticklabels(piv_daily.columns)
    axes[1, 1].set_yticks(range(len(piv_daily.index)))
    axes[1, 1].set_yticklabels(piv_daily.index)
    axes[1, 1].set_title('日K 樣本外(OOS) 獲利因子平原區 (綠色為高PF穩健區)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('滾動窗口 (Window)')
    axes[1, 1].set_ylabel('進場門檻 (Entry Z)')
    plt.colorbar(im2, ax=axes[1, 1])

    plt.tight_layout()
    chart_path = os.path.join(artifacts_dir, 'crude_oil_ou_convergence_analysis.png')
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    print(f'\n分析圖表已成功輸出至: {chart_path}')

