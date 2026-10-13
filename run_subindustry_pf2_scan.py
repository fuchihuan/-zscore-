"""
Run Sub-Industry Pairs PF > 2.0 Scan (20-day & 60-day Windows)
=============================================================
找出所有細產業同業配對：
- Z-Score 20天 & 60天窗口均值回歸 (Entry=±2.0, Exit=0.0)
- 最小一組口數配對 (基本單位 1 組)
- 回測期間: 2022-01-01 至最新歷史數據
- 篩選門檻: 獲利因子 (Profit Factor) > 2.0 且 交易次數 >= 3
- 統計過往勝率、交易次數、平均/最長/最短收斂天數
- 計算當前最新 Z 分數、停利價差目標與進場操作建議
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import time
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pure_backtest
import statsmodels.api as sm

_prices = None
_ind_df = None

def run_pair_task(task):
    global _prices, _ind_df
    sub, sym1, sym2, window = task
    
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
    prod1_str = f"{prod1} ({ratio1:.0f}%)" if ratio1 > 0 else prod1

    prod2 = str(_ind_df.loc[sym2, 'PrimaryProduct']) if sym2 in _ind_df.index else ''
    ratio2 = float(_ind_df.loc[sym2, 'PrimaryRatio']) if sym2 in _ind_df.index and pd.notna(_ind_df.loc[sym2, 'PrimaryRatio']) else 0.0
    prod2_str = f"{prod2} ({ratio2:.0f}%)" if ratio2 > 0 else prod2

    try:
        df_rec, df_trades, stats = pure_backtest.run_backtest(
            s1, s2, sym1, sym2, name1, name2,
            initial_capital=1000000,
            selected_models=['Z-Score (標準)'],
            model_params={'Z-Score (標準)': {'z_entry': 2.0, 'z_exit': 0.0, 'z_window': window}},
            size_mode='最少配對口數 (僅基本單位)',
            margin_usage_pct=1.0,
            run_start_date='2020-01-01',
            advanced_params={'fixed_pair_multiplier': 1, 'trade_mode': '收斂 (均值回歸)', 'use_grid': False, 'enable_stop_loss': False}
        )
    except Exception as e:
        return None

    if stats.get('insufficient_margin_error', False) or df_rec.empty:
        return None

    pf = stats.get('profit_factor', 0.0)
    total_trades = stats.get('total_trades', 0)
    
    # 門檻: 獲利因子 > 2.0 且 交易次數 >= 3
    if pf < 2.0 or total_trades < 3:
        return None

    # 最新狀態計算
    try:
        X = sm.add_constant(s1)
        ols_model = sm.OLS(s2, X).fit()
        hedge_ratio = float(ols_model.params.iloc[1])
        spread = s2 - hedge_ratio * s1
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        
        latest_spread = float(spread.iloc[-1])
        latest_mean = float(spread_mean.iloc[-1])
        latest_std = float(spread_std.iloc[-1])
        
        if latest_std > 1e-6:
            current_z = (latest_spread - latest_mean) / latest_std
        else:
            current_z = 0.0
            
        p1_latest = float(s1.iloc[-1])
        p2_latest = float(s2.iloc[-1])
    except Exception as e:
        current_z = 0.0
        latest_spread = 0.0
        latest_mean = 0.0
        latest_std = 0.0
        hedge_ratio = 1.0
        p1_latest = 0.0
        p2_latest = 0.0

    # 判斷當前操作狀態與停利目標
    if current_z >= 2.0:
        action = '🔥 放空價差收斂 (賣B買A)'
        room = latest_spread - latest_mean
        target_tp = f"價差回歸至 {latest_mean:.2f} (Z=0, 空間 +{room:.2f} 點)"
    elif current_z <= -2.0:
        action = '🔥 做多價差收斂 (買B賣A)'
        room = latest_mean - latest_spread
        target_tp = f"價差回歸至 {latest_mean:.2f} (Z=0, 空間 +{room:.2f} 點)"
    elif current_z >= 1.2:
        action = '⏳ 價差偏高 (接近進場門檻)'
        room = latest_spread - latest_mean
        target_tp = f"目標中樞 {latest_mean:.2f} (目前偏離 +{room:.2f} 點)"
    elif current_z <= -1.2:
        action = '⏳ 價差偏低 (接近進場門檻)'
        room = latest_mean - latest_spread
        target_tp = f"目標中樞 {latest_mean:.2f} (目前偏離 -{room:.2f} 點)"
    else:
        action = '🎯 常態區間 (Z接近0)'
        target_tp = f"常態中樞 {latest_mean:.2f} (Z={current_z:.2f})"

    # 最近 250 天相關性
    recent = s1.index[-250:]
    p_corr = float(s1.loc[recent].corr(s2.loc[recent])) if len(recent) > 30 else 0.0
    r_corr = float(s1.loc[recent].pct_change().dropna().corr(s2.loc[recent].pct_change().dropna())) if len(recent) > 30 else 0.0

    return {
        'SubIndustry': sub,
        'Pair': f"{sym1} - {sym2}",
        'Ticker1': sym1,
        'Name1': name1,
        'Prod1': prod1_str,
        'Ticker2': sym2,
        'Name2': name2,
        'Prod2': prod2_str,
        'Window': window,
        'ProfitFactor': round(float(pf), 3) if pf != float('inf') else 999.0,
        'TotalReturn': round(stats.get('total_return', 0.0), 0),
        'TotalReturnPct': round(stats.get('total_return_pct', 0.0), 2),
        'AnnualizedReturn': round(stats.get('annualized_return', 0.0), 2),
        'WinRate': round(stats.get('win_rate', 0.0), 1),
        'TotalTrades': total_trades,
        'WinTrades': stats.get('win_trades', 0),
        'LoseTrades': stats.get('lose_trades', 0),
        'AvgWin': round(stats.get('avg_win', 0.0), 0),
        'AvgLose': round(stats.get('avg_lose', 0.0), 0),
        'AvgConvergenceDays': stats.get('avg_convergence_days', 0.0),
        'MinConvergenceDays': stats.get('min_convergence_days', 0),
        'MaxConvergenceDays': stats.get('max_convergence_days', 0),
        'MaxDrawdownPct': round(stats.get('max_drawdown_pct', 0.0), 2),
        'SharpeRatio': round(stats.get('sharpe_ratio', 0.0), 3),
        'BaseContracts': f"{stats.get('base_s1', 1)} : {stats.get('base_s2', 1)}",
        'BaseMargin': round(stats.get('base_margin', 0.0), 0),
        'PriceCorr': round(p_corr, 3),
        'ReturnCorr': round(r_corr, 3),
        'CointPValue': round(stats.get('coint_pvalue', 1.0), 4),
        'HedgeRatio': round(hedge_ratio, 4),
        'LatestPrice1': round(p1_latest, 2),
        'LatestPrice2': round(p2_latest, 2),
        'CurrentSpread': round(latest_spread, 2),
        'SpreadMean': round(latest_mean, 2),
        'SpreadStd': round(latest_std, 2),
        'CurrentZScore': round(current_z, 2),
        'TakeProfitTarget': target_tp,
        'ActionRecommendation': action
    }

def main():
    print("==================================================", flush=True)
    print("🔍 載入台股股期資料庫與同產業配對組合...", flush=True)
    prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
    ind_df = pd.read_csv('industry_data.csv').set_index('Ticker')

    stock_df = ind_df[~ind_df['Category'].str.contains('ETF|全球', na=False)]
    
    sub_dict = {}
    for tk, row in stock_df.iterrows():
        sub = row['SubIndustry']
        if tk in prices.columns:
            sub_dict.setdefault(sub, []).append(tk)

    multi_stock_subs = {k: v for k, v in sub_dict.items() if len(v) >= 2}
    print(f"🏢 符合條件之同細產業族群: {len(multi_stock_subs)} 個", flush=True)

    tasks = []
    for sub, t_list in multi_stock_subs.items():
        for i in range(len(t_list)):
            for j in range(len(t_list)):
                if i != j:
                    sym1 = t_list[i]
                    sym2 = t_list[j]
                    for w in [20, 60]:
                        tasks.append((sub, sym1, sym2, w))

    print(f"📊 總待測試組合數: {len(tasks)} 組 (涵蓋 20日 與 60日 雙天期)", flush=True)
    print(f"⚙️ 篩選門檻: Profit Factor > 2.0 且 交易次數 >= 3", flush=True)

    global _prices, _ind_df
    _prices = prices
    _ind_df = ind_df

    t0 = time.time()
    results = []
    max_w = 6
    print(f"🚀 開始執行多執行緒回測運算 ({max_w} Workers)...", flush=True)

    with ThreadPoolExecutor(max_workers=max_w) as executor:
        futures = {executor.submit(run_pair_task, task): task for task in tasks}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  進度: {done}/{total} ({done/total*100:.1f}%) | 找到優質配對: {len(results)} 組 | 耗時: {time.time()-t0:.1f}s", flush=True)
            res = fut.result()
            if res is not None:
                results.append(res)

    t1 = time.time()
    print(f"\n✅ 掃描運算完成！總耗時: {t1-t0:.2f} 秒，找到 Profit Factor > 2.0 之配對: {len(results)} 組", flush=True)

    if not results:
        print("⚠️ 未找到符合條件之配對！", flush=True)
        return

    df_res = pd.DataFrame(results)
    df_res.sort_values(by=['ProfitFactor', 'TotalReturn', 'WinRate'], ascending=[False, False, False], inplace=True)
    
    df_res.to_csv('subindustry_pf2_convergence_master.csv', index=False, encoding='utf-8-sig')
    print(f"💾 全部 {len(df_res)} 組 PF > 2.0 配對已儲存至: subindustry_pf2_convergence_master.csv", flush=True)

    # 去除對稱重複 (同一對股票同一窗口保留表現較優者)
    best_pairs = {}
    for _, row in df_res.iterrows():
        pair_tks = tuple(sorted([row['Ticker1'], row['Ticker2']]))
        key = (row['SubIndustry'], pair_tks, row['Window'])
        if key not in best_pairs:
            best_pairs[key] = row
        else:
            if row['TotalReturn'] > best_pairs[key]['TotalReturn']:
                best_pairs[key] = row
                
    df_dedup = pd.DataFrame(list(best_pairs.values()))
    df_dedup.sort_values(by=['ProfitFactor', 'TotalReturn', 'WinRate'], ascending=[False, False, False], inplace=True)
    df_dedup.to_csv('subindustry_pf2_convergence_dedup.csv', index=False, encoding='utf-8-sig')
    print(f"🏆 經去重後之獨立精選配對: {len(df_dedup)} 組！已儲存至: subindustry_pf2_convergence_dedup.csv", flush=True)

    signals = df_dedup[df_dedup['CurrentZScore'].abs() >= 2.0]
    print(f"\n🔥 今日已達進場門檻 (|Z| >= 2.0) 的高勝率收斂機會共有: {len(signals)} 組！", flush=True)
    for i, (_, r) in enumerate(signals.head(10).iterrows()):
        print(f"  {i+1:2d}. [{r['SubIndustry']}] {r['Name1']} vs {r['Name2']} (W={r['Window']}日, 口數 {r['BaseContracts']})", flush=True)
        print(f"      當前Z: {r['CurrentZScore']:+.2f} ｜ 動作: {r['ActionRecommendation']} ｜ 停利目標: {r['TakeProfitTarget']}", flush=True)
        print(f"      歷史PF: {r['ProfitFactor']:.2f} ｜ 勝率: {r['WinRate']:.1f}% ｜ 交易: {r['TotalTrades']}次 ｜ 平均收斂: {r['AvgConvergenceDays']}天 (最短{r['MinConvergenceDays']}天/最長{r['MaxConvergenceDays']}天)", flush=True)

if __name__ == '__main__':
    main()
