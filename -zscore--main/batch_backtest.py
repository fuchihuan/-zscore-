import sys
import os
import time
import json
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定 matplotlib 中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(r'c:\Users\chihuan\Desktop\股期配對\-zscore--main')
try:
    from app import run_backtest
except ImportError:
    print("Error importing app.py. Make sure run_backtest is defined.")
    sys.exit(1)

def calc_sharpe(results_df):
    if len(results_df) < 2:
        return 0.0
    daily_returns = results_df['帳戶淨值'].pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    return sharpe

def plot_and_save_results(res_df, trd_df, metrics, sharpe, sym1, sym2, save_dir, group_name):
    os.makedirs(save_dir, exist_ok=True)
    pair_name = f"{sym1}_{sym2}"
    
    # 1. Equity Curve
    plt.figure(figsize=(10, 5))
    plt.plot(res_df.index, res_df['帳戶淨值'], label='Cumulative Equity')
    plt.title(f"{pair_name} ({group_name}) - Equity Curve (Sharpe: {sharpe:.2f})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{pair_name}_equity.png"))
    plt.close()
    
    # 2. Z-Score and Trades
    plt.figure(figsize=(10, 5))
    plt.plot(res_df.index, res_df['Z-Score'], label='Z-Score', color='blue', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--')
    
    # mark trades
    try:
        if '日期' in trd_df.columns and '動作' in trd_df.columns:
            trd_df['日期'] = pd.to_datetime(trd_df['日期'])
            buys = trd_df[trd_df['動作'].str.contains('開倉\(做多')]
            sells = trd_df[trd_df['動作'].str.contains('開倉\(做空')]
            
            if not buys.empty:
                plt.scatter(buys['日期'], [0]*len(buys), marker='^', color='g', label='Long Entry', zorder=5)
            if not sells.empty:
                plt.scatter(sells['日期'], [0]*len(sells), marker='v', color='r', label='Short Entry', zorder=5)
    except Exception as e:
        print(f"Plotting trades failed: {e}")
        
    plt.title(f"{pair_name} - Z-Score & Trades")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{pair_name}_zscore.png"))
    plt.close()

def main():
    print("=" * 60)
    print("  自動化跨產業配對批次回測引擎 (尋找 Sharpe >= 2)")
    print("=" * 60)
    
    # 1. 讀取資料
    try:
        df_ind = pd.read_csv(r'c:\Users\chihuan\Desktop\股期配對\-zscore--main\industry_data.csv')
        df_px = pd.read_csv(r'c:\Users\chihuan\Desktop\股期配對\-zscore--main\stock_prices.csv', index_col=0, parse_dates=True)
        # 過濾 2021 至今
        df_px = df_px.loc['2021-01-01':]
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # 分組
    groups = df_ind.groupby('Industry')['Ticker'].apply(list).to_dict()
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"找到 {len(valid_groups)} 個具備 2 檔以上股票的產業分類。")
    
    # 參數設定
    initial_capital = 1_000_000
    param_grid = [
        {'z_method': '歷史均值(預設)', 'z_window': 20, 'z_entry': 2.0, 'z_exit': 0.0, 'ou': False},
        {'z_method': '歷史均值(預設)', 'z_window': 60, 'z_entry': 2.0, 'z_exit': 0.0, 'ou': False}
    ]
    
    target_pairs = 9999 # 找出所有及格配對
    successful_pairs = []
    
    timestamp = time.strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(r'c:\Users\chihuan\Desktop\股期配對', f'Backtest_Results_{timestamp}')
    
    industries = list(valid_groups.keys())
    random.shuffle(industries)
    
    for ind in industries:
        if len(successful_pairs) >= target_pairs:
            break
            
        tickers = valid_groups[ind]
        # 產生所有配對 (全部找一輪)
        pairs = list(itertools.combinations(tickers, 2))
        random.shuffle(pairs) # 隨機測試
        
        for p in pairs:
            tk1, tk2 = p
            if tk1 not in df_px.columns or tk2 not in df_px.columns:
                continue
                
            s1 = df_px[tk1].dropna()
            s2 = df_px[tk2].dropna()
            common_idx = s1.index.intersection(s2.index)
            if len(common_idx) < 252: # 至少一年資料
                continue
                
            s1, s2 = s1.loc[common_idx], s2.loc[common_idx]
            
            # 測試參數網格
            print(f"Testing {tk1} vs {tk2} in {ind}...", flush=True)
            for prm in param_grid:
                adv_params = {
                    'z_method': prm['z_method'],
                    'kalman_q': 1e-5,
                    'kalman_r': 1e-3,
                    'halflife': prm['z_window'] // 2,
                    'use_ou_filter': prm['ou'],
                    'ou_window': prm['z_window'],
                    'ou_min_reversion': 0.05,
                    'use_beta_trans': False,
                    'beta_window': 60,
                    'beta_trans_max': 0.1
                }
                
                try:
                    res_df, trd_df, metrics = run_backtest(
                        s1, s2, tk1, tk2, 'N/A', 'N/A', 
                        initial_capital, prm['z_entry'], prm['z_exit'], prm['z_window'], 
                        size_mode='依保證金上限最大化（預設）', margin_usage_pct=0.3,
                        run_start_date='2021-01-01', run_end_date=None, 
                        advanced_params=adv_params
                    )
                except Exception as e:
                    print(f"Error in {tk1}_{tk2}: {e}", flush=True)
                    continue
                    
                if res_df is None or trd_df.empty:
                    continue
                    
                sharpe = calc_sharpe(res_df)
                
                # 判斷是否及格
                coint_p = metrics.get('coint_pvalue', 1.0)
                trades = metrics.get('total_trades', 0)
                profit_factor = metrics.get('profit_factor', 0.0)
                win_rate = metrics.get('win_rate', 0.0)
                
                if profit_factor > 3.0 and trades >= 10 and win_rate >= 50.0:
                    print(f"[及格] {ind}: {tk1} vs {tk2} | PF: {profit_factor:.2f} | Trades: {trades} | Coint: {coint_p:.4f}", flush=True)
                    
                    # 儲存
                    plot_and_save_results(res_df, trd_df, metrics, sharpe, tk1, tk2, save_dir, ind)
                    
                    record = {
                        'Industry': ind,
                        'Pair': f"{tk1}_{tk2}",
                        'Sharpe': round(sharpe, 2),
                        'Total Return %': metrics['total_return_pct'],
                        'MDD %': metrics['max_drawdown_pct'],
                        'Win Rate %': win_rate,
                        'Trades': trades,
                        'Coint P-Value': round(coint_p, 4),
                        'Parameters': json.dumps(prm)
                    }
                    successful_pairs.append(record)
                    # 不再 break，繼續尋找該產業其他及格的配對
                    
    # 匯出總表
    if successful_pairs:
        summary_df = pd.DataFrame(successful_pairs)
        summary_path = os.path.join(save_dir, 'summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 成功找到 {len(successful_pairs)} 組黃金配對！")
        print(f"📁 結果已儲存於: {save_dir}")
    else:
        print("\n⚠️ 條件太嚴苛，無法找到足夠的及格配對！請嘗試放寬 Sharpe 或交易次數限制。")

if __name__ == '__main__':
    main()
