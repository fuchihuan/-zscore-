import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Set matplotlib style and fonts
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

artifacts_dir = r"C:\Users\chihuan\.gemini\antigravity-ide\brain\d2afa23f-f21e-4fb8-a40f-186608353862"

def analyze_and_optimize(csv_path, timeframe_name, contract_multiplier=1000, fee_per_round_turn=10, slippage_per_trade=0.02):
    print(f"\n{'='*70}")
    print(f"--- ANALYZING & OPTIMIZING CRUDE OIL PAIR ({timeframe_name}) ---")
    print(f"{'='*70}")
    
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    wti = df['WTI']
    brent = df['Brent']
    
    # 1. Cointegration test
    coint_stat, p_value, crit_values = coint(brent, wti)
    print(f"Engle-Granger Cointegration Test:")
    print(f"  t-statistic: {coint_stat:.4f}, p-value: {p_value:.5e}")
    print(f"  Critical values: 1%: {crit_values[0]:.4f}, 5%: {crit_values[1]:.4f}, 10%: {crit_values[2]:.4f}")
    if p_value < 0.05:
        print("  -> Cointegrated at 5% significance level! (Strong mean-reverting relationship)")
    else:
        print("  -> P-value > 0.05, testing stationary spread behavior.")
        
    # OLS Regression for hedge ratio
    X = sm.add_constant(wti)
    model = sm.OLS(brent, X).fit()
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    print(f"OLS Hedge Ratio: Brent = {alpha:.4f} + {beta:.4f} * WTI (R-squared: {model.rsquared:.4f})")
    
    # Both spreads: OLS spread & Nominal spread
    ols_spread = brent - (alpha + beta * wti)
    nominal_spread = brent - wti # Standard market Brent-WTI spread ($/barrel)
    
    # Test ADF stationarity of spreads
    adf_ols = adfuller(ols_spread.dropna())
    adf_nom = adfuller(nominal_spread.dropna())
    print(f"ADF Test on OLS Spread: p-value = {adf_ols[1]:.5e}")
    print(f"ADF Test on Nominal (Brent - WTI) Spread: p-value = {adf_nom[1]:.5e}")
    
    # Grid search for best parameters
    if "1h" in timeframe_name.lower():
        windows = [10, 20, 24, 48, 72, 96, 120, 168, 240] # in hours
    else:
        windows = [10, 15, 20, 30, 45, 60] # in days
        
    entry_thresholds = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    exit_thresholds = [0.0, 0.25, 0.5, 0.75]
    stop_losses = [2.5, 3.0, 3.5, 4.0, 99.0]
    
    best_results = []
    
    for spread_type, spread_series, hedge in [("Nominal (1:1)", nominal_spread, 1.0), ("OLS Beta", ols_spread, beta)]:
        for w in windows:
            roll_mean = spread_series.rolling(w).mean()
            roll_std = spread_series.rolling(w).std()
            z_score = ((spread_series - roll_mean) / roll_std).dropna()
            
            common_idx = z_score.index
            s_aligned = spread_series.loc[common_idx]
            
            for z_in in entry_thresholds:
                for z_out in exit_thresholds:
                    if z_out >= z_in:
                        continue
                    for z_sl in stop_losses:
                        if z_sl <= z_in:
                            continue
                            
                        # Simulate trades
                        pos = 0 # 1: Long spread, -1: Short spread
                        trades = []
                        entry_price = 0.0
                        entry_idx = None
                        
                        z_vals = z_score.values
                        spr_vals = s_aligned.values
                        idx_vals = common_idx
                        
                        for i in range(len(z_vals)):
                            z = z_vals[i]
                            spr = spr_vals[i]
                            
                            # Check exits
                            if pos == 1: # Long spread
                                exit_hit = False
                                if z >= -z_out:
                                    exit_hit = True
                                elif z <= -z_sl:
                                    exit_hit = True
                                if exit_hit:
                                    pnl_spr = (spr - entry_price) - slippage_per_trade * 2
                                    dollar_pnl = pnl_spr * contract_multiplier - fee_per_round_turn
                                    trades.append({
                                        'type': 'Long Spread',
                                        'entry_idx': entry_idx,
                                        'exit_idx': idx_vals[i],
                                        'entry_price': entry_price,
                                        'exit_price': spr,
                                        'pnl_pts': pnl_spr,
                                        'dollar_pnl': dollar_pnl
                                    })
                                    pos = 0
                            elif pos == -1: # Short spread
                                exit_hit = False
                                if z <= z_out:
                                    exit_hit = True
                                elif z >= z_sl:
                                    exit_hit = True
                                if exit_hit:
                                    pnl_spr = (entry_price - spr) - slippage_per_trade * 2
                                    dollar_pnl = pnl_spr * contract_multiplier - fee_per_round_turn
                                    trades.append({
                                        'type': 'Short Spread',
                                        'entry_idx': entry_idx,
                                        'exit_idx': idx_vals[i],
                                        'entry_price': entry_price,
                                        'exit_price': spr,
                                        'pnl_pts': pnl_spr,
                                        'dollar_pnl': dollar_pnl
                                    })
                                    pos = 0
                                    
                            # Check entry
                            if pos == 0:
                                if z <= -z_in: # Spread unusually low -> Long spread
                                    pos = 1
                                    entry_price = spr
                                    entry_idx = idx_vals[i]
                                elif z >= z_in: # Spread unusually high -> Short spread
                                    pos = -1
                                    entry_price = spr
                                    entry_idx = idx_vals[i]
                                    
                        if len(trades) >= 5:
                            tdf = pd.DataFrame(trades)
                            total_pnl = tdf['dollar_pnl'].sum()
                            wins = tdf[tdf['dollar_pnl'] > 0]
                            win_rate = len(wins) / len(tdf) * 100
                            gross_profit = wins['dollar_pnl'].sum() if len(wins) > 0 else 0
                            losses = tdf[tdf['dollar_pnl'] < 0]
                            gross_loss = abs(losses['dollar_pnl'].sum()) if len(losses) > 0 else 1e-6
                            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0
                            
                            cum_pnl = tdf['dollar_pnl'].cumsum()
                            peak = cum_pnl.cummax()
                            dd = peak - cum_pnl
                            max_dd = dd.max() if len(dd) > 0 else 0.0
                            
                            # Sharpe
                            mean_trade = tdf['dollar_pnl'].mean()
                            std_trade = tdf['dollar_pnl'].std()
                            trade_sharpe = (mean_trade / std_trade) * np.sqrt(len(tdf)) if std_trade > 0 else 0
                            
                            best_results.append({
                                'spread_type': spread_type,
                                'window': w,
                                'entry_z': z_in,
                                'exit_z': z_out,
                                'stop_z': z_sl,
                                'trade_count': len(tdf),
                                'win_rate': win_rate,
                                'profit_factor': profit_factor,
                                'total_pnl': total_pnl,
                                'max_dd': max_dd,
                                'sharpe': trade_sharpe,
                                'trades_data': trades
                            })
                            
    res_df = pd.DataFrame(best_results)
    if res_df.empty:
        print("No valid results found.")
        return None, None
        
    res_df = res_df.sort_values(by=['profit_factor', 'total_pnl'], ascending=False).reset_index(drop=True)
    print(f"\nTop 10 Parameter Combinations ({timeframe_name}):")
    print(res_df[['spread_type', 'window', 'entry_z', 'exit_z', 'stop_z', 'trade_count', 'win_rate', 'profit_factor', 'total_pnl', 'max_dd']].head(10).to_string(index=False))
    
    top = res_df.iloc[0]
    print(f"\n>>> BEST PARAMETER SET ({timeframe_name}) <<<")
    print(f"Spread Type: {top['spread_type']}")
    print(f"Window: {top['window']}, Entry Z: {top['entry_z']}, Exit Z: {top['exit_z']}, Stop Loss Z: {top['stop_z']}")
    print(f"Trades: {top['trade_count']}, Win Rate: {top['win_rate']:.1f}%, Profit Factor: {top['profit_factor']:.2f}")
    print(f"Total Net PnL: ${top['total_pnl']:,.2f}, Max Drawdown: ${top['max_dd']:,.2f}")
    
    return res_df, top

# Run optimization for both 1h and 1d
res_1h, top_1h = analyze_and_optimize('data/crude_oil_1y_1h.csv', '1-Hour Intraday (5,660 Bars)')
res_1d, top_1d = analyze_and_optimize('data/crude_oil_1y_1d.csv', 'Daily 1D (252 Bars)')

# Generate comprehensive charts for the best 1-hour strategy
print("\nGenerating charts...")
df_1h = pd.read_csv('data/crude_oil_1y_1h.csv', index_col=0)
df_1h.index = pd.to_datetime(df_1h.index, utc=True)

w = int(top_1h['window'])
z_in = top_1h['entry_z']
z_out = top_1h['exit_z']
z_sl = top_1h['stop_z']

if top_1h['spread_type'] == "Nominal (1:1)":
    spr = df_1h['Brent'] - df_1h['WTI']
else:
    X = sm.add_constant(df_1h['WTI'])
    m = sm.OLS(df_1h['Brent'], X).fit()
    spr = df_1h['Brent'] - (m.params.iloc[0] + m.params.iloc[1] * df_1h['WTI'])

r_mean = spr.rolling(w).mean()
r_std = spr.rolling(w).std()
z_series = (spr - r_mean) / r_std

trades_df = pd.DataFrame(top_1h['trades_data'])
trades_df['entry_idx'] = pd.to_datetime(trades_df['entry_idx'], utc=True)
trades_df['exit_idx'] = pd.to_datetime(trades_df['exit_idx'], utc=True)

# Plot 4-panel analysis figure
fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1.8, 1.8, 1.8]})

# Panel 1: Prices of WTI and Brent
ax1 = axes[0]
ax1.plot(df_1h.index, df_1h['Brent'], label='布蘭特原油 Brent (BZ)', color='#e63946', lw=1.5)
ax1.plot(df_1h.index, df_1h['WTI'], label='紐約輕原油 WTI (CL)', color='#457b9d', lw=1.5)
ax1.set_title('過去一年 CME 原油期貨價差配對交易回測 (WTI CL vs Brent BZ 1-Hour Intraday)', fontsize=14, fontweight='bold', pad=10)
ax1.set_ylabel('期貨價格 (USD / 桶)', fontsize=11)
ax1.legend(loc='upper left', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel 2: Price Spread
ax2 = axes[1]
ax2.plot(spr.index, spr, label=f'原油價差 ({top_1h["spread_type"]}: Brent - WTI)', color='#2a9d8f', lw=1.2)
ax2.plot(r_mean.index, r_mean, label=f'滾動均線 (Window={w}h)', color='#e76f51', linestyle='--', lw=1.2)
ax2.set_ylabel('價差 (USD / 桶)', fontsize=11)
ax2.legend(loc='upper left', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel 3: Z-Score with Trading Signals
ax3 = axes[2]
ax3.plot(z_series.index, z_series, label='Z-分數 (Z-Score)', color='#1d3557', lw=1.0)
ax3.axhline(z_in, color='#d62828', linestyle='--', lw=1.2, label=f'做空價差門檻 (+{z_in})')
ax3.axhline(-z_in, color='#2a9d8f', linestyle='--', lw=1.2, label=f'做多價差門檻 (-{z_in})')
ax3.axhline(z_out, color='#f77f00', linestyle=':', lw=1.0, label=f'平倉門檻 (±{z_out})')
ax3.axhline(-z_out, color='#f77f00', linestyle=':', lw=1.0)
ax3.axhline(0, color='gray', linestyle='-', lw=0.8, alpha=0.7)
if z_sl < 90:
    ax3.axhline(z_sl, color='black', linestyle='-.', lw=1.0, label=f'停損線 (±{z_sl})')
    ax3.axhline(-z_sl, color='black', linestyle='-.', lw=1.0)

# Mark entries on Z-Score chart
long_entries = trades_df[trades_df['type'] == 'Long Spread']['entry_idx']
short_entries = trades_df[trades_df['type'] == 'Short Spread']['entry_idx']
ax3.scatter(long_entries, z_series.loc[long_entries], marker='^', color='#2a9d8f', s=45, zorder=5, label='做多進場 (買Brent賣WTI)')
ax3.scatter(short_entries, z_series.loc[short_entries], marker='v', color='#d62828', s=45, zorder=5, label='做空進場 (賣Brent買WTI)')

ax3.set_ylabel('Z-分數', fontsize=11)
ax3.set_ylim(-max(3.5, z_sl+0.5 if z_sl < 90 else 4.0), max(3.5, z_sl+0.5 if z_sl < 90 else 4.0))
ax3.legend(loc='upper left', ncol=3, fontsize=9, frameon=True)
ax3.grid(True, linestyle='--', alpha=0.5)

# Panel 4: Cumulative Net PnL (Equity Curve)
ax4 = axes[3]
equity_series = pd.Series(0.0, index=df_1h.index)
running_pnl = 0.0
for _, tr in trades_df.iterrows():
    running_pnl += tr['dollar_pnl']
    equity_series.loc[tr['exit_idx']:] = running_pnl

ax4.plot(equity_series.index, equity_series, label=f'累計淨損益 (Net PnL): ${top_1h["total_pnl"]:,.2f}', color='#06d6a0', lw=2)
ax4.fill_between(equity_series.index, equity_series, 0, where=(equity_series >= 0), color='#06d6a0', alpha=0.2)
ax4.fill_between(equity_series.index, equity_series, 0, where=(equity_series < 0), color='#ef476f', alpha=0.2)
ax4.set_ylabel('累計損益 (USD)', fontsize=11)
ax4.set_xlabel('日期', fontsize=11)
ax4.legend(loc='upper left', frameon=True)
ax4.grid(True, linestyle='--', alpha=0.5)

# Format x-axis dates
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=30)

plt.tight_layout()
chart_path = os.path.join(artifacts_dir, 'crude_oil_spread_backtest_1y.png')
plt.savefig(chart_path, dpi=200, bbox_inches='tight')
print(f"Chart saved successfully to {chart_path}")

# Also generate a second chart for Daily 1D strategy
print("Generating Daily 1D chart...")
df_1d = pd.read_csv('data/crude_oil_1y_1d.csv', index_col=0)
df_1d.index = pd.to_datetime(df_1d.index, utc=True)
w_1d = int(top_1d['window'])
z_in_1d = top_1d['entry_z']
z_out_1d = top_1d['exit_z']
z_sl_1d = top_1d['stop_z']

if top_1d['spread_type'] == "Nominal (1:1)":
    spr_1d = df_1d['Brent'] - df_1d['WTI']
else:
    X_1d = sm.add_constant(df_1d['WTI'])
    m_1d = sm.OLS(df_1d['Brent'], X_1d).fit()
    spr_1d = df_1d['Brent'] - (m_1d.params.iloc[0] + m_1d.params.iloc[1] * df_1d['WTI'])

r_mean_1d = spr_1d.rolling(w_1d).mean()
r_std_1d = spr_1d.rolling(w_1d).std()
z_series_1d = (spr_1d - r_mean_1d) / r_std_1d

trades_1d = pd.DataFrame(top_1d['trades_data'])
trades_1d['entry_idx'] = pd.to_datetime(trades_1d['entry_idx'], utc=True)
trades_1d['exit_idx'] = pd.to_datetime(trades_1d['exit_idx'], utc=True)

fig_1d, axes_1d = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1.8, 1.8]})
axes_1d[0].plot(df_1d.index, df_1d['Brent'], label='布蘭特原油 Brent (BZ)', color='#e63946', lw=1.5)
axes_1d[0].plot(df_1d.index, df_1d['WTI'], label='紐約輕原油 WTI (CL)', color='#457b9d', lw=1.5)
axes_1d[0].set_title('過去一年 CME 原油期貨日K配對交易 (Daily 1D)', fontsize=14, fontweight='bold', pad=10)
axes_1d[0].set_ylabel('期貨價格 (USD / 桶)')
axes_1d[0].legend(loc='upper left')
axes_1d[0].grid(True, linestyle='--', alpha=0.5)

axes_1d[1].plot(z_series_1d.index, z_series_1d, label='Z-分數', color='#1d3557', lw=1.5)
axes_1d[1].axhline(z_in_1d, color='#d62828', linestyle='--', label=f'做空門檻 (+{z_in_1d})')
axes_1d[1].axhline(-z_in_1d, color='#2a9d8f', linestyle='--', label=f'做多門檻 (-{z_in_1d})')
axes_1d[1].axhline(z_out_1d, color='#f77f00', linestyle=':', label=f'平倉門檻 ({z_out_1d})')
if z_sl_1d < 90:
    axes_1d[1].axhline(z_sl_1d, color='black', linestyle='-.', label=f'停損線 (±{z_sl_1d})')
    axes_1d[1].axhline(-z_sl_1d, color='black', linestyle='-.')
axes_1d[1].scatter(trades_1d[trades_1d['type'] == 'Long Spread']['entry_idx'], z_series_1d.loc[trades_1d[trades_1d['type'] == 'Long Spread']['entry_idx']], marker='^', color='#2a9d8f', s=60, zorder=5, label='做多進場')
axes_1d[1].scatter(trades_1d[trades_1d['type'] == 'Short Spread']['entry_idx'], z_series_1d.loc[trades_1d[trades_1d['type'] == 'Short Spread']['entry_idx']], marker='v', color='#d62828', s=60, zorder=5, label='做空進場')
axes_1d[1].set_ylabel('Z-分數')
axes_1d[1].legend(loc='upper left', ncol=3)
axes_1d[1].grid(True, linestyle='--', alpha=0.5)

equity_1d = pd.Series(0.0, index=df_1d.index)
running_1d = 0.0
for _, tr in trades_1d.iterrows():
    running_1d += tr['dollar_pnl']
    equity_1d.loc[tr['exit_idx']:] = running_1d
axes_1d[2].plot(equity_1d.index, equity_1d, label=f'日K累計淨損益: ${top_1d["total_pnl"]:,.2f}', color='#06d6a0', lw=2)
axes_1d[2].fill_between(equity_1d.index, equity_1d, 0, color='#06d6a0', alpha=0.2)
axes_1d[2].set_ylabel('累計損益 (USD)')
axes_1d[2].set_xlabel('日期')
axes_1d[2].legend(loc='upper left')
axes_1d[2].grid(True, linestyle='--', alpha=0.5)

chart_1d_path = os.path.join(artifacts_dir, 'crude_oil_spread_backtest_daily.png')
fig_1d.savefig(chart_1d_path, dpi=200, bbox_inches='tight')
print(f"Daily chart saved successfully to {chart_1d_path}")

# Save top parameters
res_1h.drop(columns=['trades_data']).head(25).to_csv('data/top_params_1h.csv', index=False)
res_1d.drop(columns=['trades_data']).head(25).to_csv('data/top_params_1d.csv', index=False)
print("Saved top parameters tables to CSV!")
