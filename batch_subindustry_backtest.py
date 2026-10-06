"""
Batch Sub-Industry Pairs Backtest Runner
========================================
測試所有細產業同業配對標的組合：
- 策略模型: 基礎 Z-Score 均值回歸 (Entry=±2.0, Exit=0.0, Window=20)
- 資金管理: 固定 1,000,000 TWD 保證金，每次僅下最少配對口數 (基本單位 1 組)
- 回測期間: 2022-01-01 至 2026-10-05 (最新歷史數據)
- 篩選門檻: 獲利因子 (Profit Factor) > 3.0
"""

import os
import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import time
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pure_backtest

# Global variables in workers
_prices = None
_ind_df = None

def run_pair_worker(task):
    global _prices, _ind_df
    sub, sym1, sym2 = task
    
    if _prices is None or sym1 not in _prices.columns or sym2 not in _prices.columns:
        return None
        
    s1 = _prices[sym1].dropna()
    s2 = _prices[sym2].dropna()
    common = s1.index.intersection(s2.index)
    if len(common) < 120:
        return None
        
    s1 = s1.loc[common]
    s2 = s2.loc[common]
    
    name1 = str(_ind_df.loc[sym1, 'Name']) if sym1 in _ind_df.index else sym1
    name2 = str(_ind_df.loc[sym2, 'Name']) if sym2 in _ind_df.index else sym2
    
    prod1 = str(_ind_df.loc[sym1, 'PrimaryProduct']) if sym1 in _ind_df.index else ''
    ratio1 = float(_ind_df.loc[sym1, 'PrimaryRatio']) if sym1 in _ind_df.index and pd.notna(_ind_df.loc[sym1, 'PrimaryRatio']) else 0.0
    
    prod2 = str(_ind_df.loc[sym2, 'PrimaryProduct']) if sym2 in _ind_df.index else ''
    ratio2 = float(_ind_df.loc[sym2, 'PrimaryRatio']) if sym2 in _ind_df.index and pd.notna(_ind_df.loc[sym2, 'PrimaryRatio']) else 0.0

    try:
        df_rec, df_trades, stats = pure_backtest.run_backtest(
            s1, s2, sym1, sym2, name1, name2,
            initial_capital=1000000,
            selected_models=['Z-Score (標準)'],
            model_params={'Z-Score (標準)': {'z_entry': 2.0, 'z_exit': 0.0, 'z_window': 20}},
            size_mode='最少配對口數 (僅基本單位)',
            margin_usage_pct=1.0,
            run_start_date='2022-01-01',
            run_end_date='2026-10-05',
            advanced_params={'fixed_pair_multiplier': 1, 'trade_mode': '收斂 (均值回歸)', 'use_grid': False}
        )
    except Exception as e:
        return None

    if stats.get('insufficient_margin_error', False) or df_rec.empty:
        return None

    # Price correlation over recent 250 days
    recent = common[-250:]
    p_corr = float(s1.loc[recent].corr(s2.loc[recent])) if len(recent) > 30 else 0.0
    r_corr = float(s1.loc[recent].pct_change().dropna().corr(s2.loc[recent].pct_change().dropna())) if len(recent) > 30 else 0.0

    return {
        'SubIndustry': sub,
        'Pair': f"{sym1} - {sym2}",
        'Ticker1': sym1,
        'Name1': name1,
        'Prod1': f"{prod1} ({ratio1:.0f}%)" if ratio1 > 0 else prod1,
        'Ticker2': sym2,
        'Name2': name2,
        'Prod2': f"{prod2} ({ratio2:.0f}%)" if ratio2 > 0 else prod2,
        'ProfitFactor': round(stats.get('profit_factor', 0.0), 3),
        'TotalReturn': round(stats.get('total_return', 0.0), 0),
        'TotalReturnPct': round(stats.get('total_return_pct', 0.0), 2),
        'AnnualizedReturn': round(stats.get('annualized_return', 0.0), 2),
        'WinRate': round(stats.get('win_rate', 0.0), 1),
        'TotalTrades': stats.get('total_trades', 0),
        'WinTrades': stats.get('win_trades', 0),
        'LoseTrades': stats.get('lose_trades', 0),
        'AvgWin': round(stats.get('avg_win', 0.0), 0),
        'AvgLose': round(stats.get('avg_lose', 0.0), 0),
        'MaxDrawdownPct': round(stats.get('max_drawdown_pct', 0.0), 2),
        'SharpeRatio': round(stats.get('sharpe_ratio', 0.0), 3),
        'BaseContracts': f"{stats.get('base_s1', 1)} : {stats.get('base_s2', 1)}",
        'BaseMargin': round(stats.get('base_margin', 0.0), 0),
        'PriceCorr': round(p_corr, 3),
        'ReturnCorr': round(r_corr, 3),
        'CointPValue': round(stats.get('coint_pvalue', 1.0), 4)
    }

def main():
    print("==================================================")
    print("🔍 載入台股股期資料庫與細產業產銷組合...")
    prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
    ind_df = pd.read_csv('industry_data.csv').set_index('Ticker')

    # 排除 ETF 與全球總體資產，專注於 247 檔台灣股票期貨標的
    stock_df = ind_df[~ind_df['Category'].str.contains('ETF|全球', na=False)]
    
    sub_dict = {}
    for tk, row in stock_df.iterrows():
        sub = row['SubIndustry']
        if tk in prices.columns:
            sub_dict.setdefault(sub, []).append(tk)

    multi_stock_subs = {k: v for k, v in sub_dict.items() if len(v) >= 2}
    print(f"🏢 符合條件之同細產業族群: {len(multi_stock_subs)} 個")
    
    tasks = []
    for sub, t_list in multi_stock_subs.items():
        for i in range(len(t_list)):
            for j in range(len(t_list)):
                if i != j:
                    tasks.append((sub, t_list[i], t_list[j]))
                    
    print(f"📊 總待測試組合數: {len(tasks)} 組 (涵蓋正反向 OLS 配對)")
    print(f"⚙️ 參數設定: 基礎 Z-Score (Entry=±2.0, Exit=0.0, Window=20), 固定 100 萬保證金, 每筆 1 組最少口數")
    print(f"🚀 開始執行平行回測運算 (12 Workers)...")
    
    global _prices, _ind_df
    _prices = prices
    _ind_df = ind_df

    t0 = time.time()
    results = []
    max_w = 6
    print(f"🚀 開始執行多執行緒回測運算 ({max_w} Workers)...")
    with ThreadPoolExecutor(max_workers=max_w) as executor:
        futures = {executor.submit(run_pair_worker, task): task for task in tasks}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            if done % 150 == 0 or done == total:
                print(f"  進度: {done}/{total} ({done/total*100:.1f}%) | 耗時: {time.time()-t0:.1f}s")
            res = fut.result()
            if res is not None:
                results.append(res)
                
    t1 = time.time()
    print(f"✅ 回測完成！總耗時: {t1-t0:.2f} 秒，有效完成配對: {len(results)} 組")
    
    df_all = pd.DataFrame(results)
    df_all.to_csv('all_subindustry_zscore_backtest.csv', index=False, encoding='utf-8-sig')
    print(f"💾 全部 {len(df_all)} 組回測結果已存至: all_subindustry_zscore_backtest.csv")
    
    # 篩選 獲利因子 > 3.0 (且有實際交易次數 >= 3，避免 1 筆隨機交易造成的假象)
    df_gt3 = df_all[(df_all['ProfitFactor'] > 3.0) & (df_all['TotalTrades'] >= 3)].copy()
    df_gt3.sort_values(by=['ProfitFactor', 'TotalReturn', 'SharpeRatio'], ascending=[False, False, False], inplace=True)
    df_gt3.to_csv('filtered_profit_factor_gt3.csv', index=False, encoding='utf-8-sig')
    print(f"🎯 獲利因子 > 3.0 且交易次數 >= 3 之配對數: {len(df_gt3)} 組！已存至: filtered_profit_factor_gt3.csv")
    
    # 去除對稱重複 (保留表現較優者)
    best_pairs = {}
    for _, row in df_gt3.iterrows():
        pair_tks = tuple(sorted([row['Ticker1'], row['Ticker2']]))
        key = (row['SubIndustry'], pair_tks)
        if key not in best_pairs:
            best_pairs[key] = row
        else:
            if row['TotalReturn'] > best_pairs[key]['TotalReturn']:
                best_pairs[key] = row
                
    df_dedup = pd.DataFrame(list(best_pairs.values()))
    df_dedup.sort_values(by=['ProfitFactor', 'TotalReturn', 'SharpeRatio'], ascending=[False, False, False], inplace=True)
    df_dedup.to_csv('filtered_profit_factor_gt3_dedup.csv', index=False, encoding='utf-8-sig')
    print(f"🏆 經去重最佳化後的精選配對: {len(df_dedup)} 組！已存至: filtered_profit_factor_gt3_dedup.csv")
    
    print("\n================== 獲利因子 > 3.0 TOP 25 配對清單 ==================")
    for i, (_, row) in enumerate(df_dedup.head(25).iterrows()):
        pf_display = "∞ (100%勝率)" if row['ProfitFactor'] == float('inf') or row['LoseTrades'] == 0 else f"{row['ProfitFactor']:.2f}"
        print(f"{i+1:2d}. [{row['SubIndustry']}] {row['Name1']} ({row['Ticker1']}) vs {row['Name2']} ({row['Ticker2']})")
        print(f"    獲利因子: {pf_display} | 總報酬: {row['TotalReturn']:+,.0f} 元 ({row['TotalReturnPct']:+.1f}%) | 勝率: {row['WinRate']:.1f}% ({row['WinTrades']}勝/{row['LoseTrades']}敗) | 交易: {row['TotalTrades']}次 | MDD: {row['MaxDrawdownPct']:.1f}% | 協整p: {row['CointPValue']:.4f}")

if __name__ == '__main__':
    main()
