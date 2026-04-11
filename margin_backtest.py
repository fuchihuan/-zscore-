"""
台灣股票期貨配對交易 - 完整保證金模擬回測
標的: 1513.TW (中興電) vs 6414.TW (樺漢)

核心規則：
- 股票期貨 1 口 = 2 張 = 2000 股
- 保證金比率 = 合約價值的 13.5%（一般標準）
- 合約價值 = 股價 × 2000
- 每日結算：依收盤價計算未實現損益、保證金使用、淨值
- Z-Score 訊號：> +2 做空 Spread, < -2 做多 Spread, 歸零平倉
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import os
from math import gcd

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 參數設定
# ============================================================
INITIAL_MARGIN_CAPITAL = 200_000  # 初始保證金帳戶資金 20 萬
MARGIN_RATE = 0.135               # 股票期貨保證金比率 13.5%
SHARES_PER_CONTRACT = 2000        # 1 口 = 2000 股
ZSCORE_ENTRY = 2.0                # Z-Score 開倉閾值
ZSCORE_EXIT = 0.0                 # Z-Score 平倉閾值 (回歸均值)
ZSCORE_WINDOW = 60                # 滾動窗口天數
TRADING_FEE_RATE = 0.00002        # 期交稅 (十萬分之二, 單邊)
COMMISSION_PER_CONTRACT = 30      # 手續費每口(單邊，概估)

SYM1 = '1513.TW'  # 中興電
SYM2 = '6414.TW'  # 樺漢


def calc_contract_value(price):
    """計算一口股期的合約價值"""
    return price * SHARES_PER_CONTRACT


def calc_margin_per_contract(price):
    """計算一口股期所需保證金"""
    return calc_contract_value(price) * MARGIN_RATE


def calc_trading_cost(price, contracts):
    """計算交易成本：期交稅 + 手續費（單邊）"""
    contract_value = calc_contract_value(price) * contracts
    tax = contract_value * TRADING_FEE_RATE
    commission = COMMISSION_PER_CONTRACT * contracts
    return tax + commission


def find_min_contracts(hedge_ratio):
    """
    根據 Hedge Ratio 計算最少需下幾口才能配對。
    
    Hedge Ratio ≈ 0.96 表示:
    每做 1 口 6414 (S2), 需要 0.96 口 1513 (S1) 來對沖。
    
    我們需要找到最小的整數口數組合:
    contracts_S1 / contracts_S2 ≈ hedge_ratio
    
    注意：Hedge Ratio 是「價格」層面的，但由於兩檔都是1口=2000股，
    所以口數比 ≈ hedge_ratio × (price_S1 / price_S2) 的效應實際上
    已經反映在 Spread 本身的計算中。
    
    對於 Spread = S2 - β × S1:
    - 做多 Spread: 做多 1 口 S2 + 做空 β 口 S1
    - 但口數必須是整數，所以要找近似比率
    """
    # hedge_ratio 是以元計算的: Spread = S2_price - HR * S1_price
    # 但因為兩邊每口股數一樣(2000股)，要對沖的是：
    # 名目金額匹配: contracts_S2 * P_S2 ≈ HR * contracts_S1 * P_S1
    # 但在 spread 模型裡已經用一對一口算: S2 - HR * S1
    # 所以最簡單方式：每邊各做 1 口，用 Spread 的 Z-Score 決定方向
    # 但如果 hedge_ratio 差太多，需要調整口數
    
    # 用分數逼近
    # 我們把 hedge_ratio 轉成分數 p/q
    # 嘗試從 1~10 口找到最接近的整數比
    best_error = float('inf')
    best_s1 = 1
    best_s2 = 1
    
    for s2 in range(1, 11):
        for s1 in range(1, 11):
            # 理想情況: s1/s2 = hedge_ratio
            ratio = s1 / s2
            error = abs(ratio - hedge_ratio)
            if error < best_error:
                best_error = error
                best_s1 = s1
                best_s2 = s2
    
    return best_s1, best_s2


def main():
    if not os.path.exists('stock_prices.csv'):
        print("請先執行 data_fetcher.py 取得歷史資料！")
        return

    # ============================================================
    # 1. 載入資料並計算 Z-Score
    # ============================================================
    print("=" * 70)
    print("  台灣股票期貨配對交易 - 完整保證金模擬回測")
    print("=" * 70)
    
    prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
    S1 = prices[SYM1].dropna()
    S2 = prices[SYM2].dropna()
    
    # 對齊
    common_idx = S1.index.intersection(S2.index)
    S1 = S1[common_idx]
    S2 = S2[common_idx]
    
    # OLS Hedge Ratio
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params[SYM1]
    
    print(f"\n標的: {SYM1} (中興電) vs {SYM2} (樺漢)")
    print(f"Hedge Ratio (β): {hedge_ratio:.4f}")
    
    # Spread & Z-Score
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=ZSCORE_WINDOW).mean()
    spread_std = spread.rolling(window=ZSCORE_WINDOW).std()
    zscore = (spread - spread_mean) / spread_std
    zscore = zscore.dropna()
    
    # ============================================================
    # 2. 計算最少配對口數 & 保證金需求
    # ============================================================
    contracts_s1, contracts_s2 = find_min_contracts(hedge_ratio)
    
    # 取回測期間的起始價格做估算
    start_date = zscore.index[0]
    p1_start = S1[start_date]
    p2_start = S2[start_date]
    
    margin_s1 = calc_margin_per_contract(p1_start) * contracts_s1
    margin_s2 = calc_margin_per_contract(p2_start) * contracts_s2
    total_margin_needed = margin_s1 + margin_s2
    
    print(f"\n{'─' * 60}")
    print(f"  配對口數計算")
    print(f"{'─' * 60}")
    print(f"Hedge Ratio β = {hedge_ratio:.4f}")
    print(f"最少配對口數: {SYM1} {contracts_s1} 口 / {SYM2} {contracts_s2} 口")
    print(f"  (實際口數比 {contracts_s1}/{contracts_s2} = {contracts_s1/contracts_s2:.4f}, 目標 β = {hedge_ratio:.4f})")
    print(f"\n以回測起始日 ({start_date.strftime('%Y-%m-%d')}) 價格試算保證金:")
    print(f"  {SYM1} 股價 {p1_start:.2f}, 1口合約值 = {p1_start:.2f} × 2000 = {calc_contract_value(p1_start):,.0f}")
    print(f"  {SYM1} {contracts_s1}口保證金 = {calc_contract_value(p1_start):,.0f} × {MARGIN_RATE*100:.1f}% × {contracts_s1} = {margin_s1:,.0f}")
    print(f"  {SYM2} 股價 {p2_start:.2f}, 1口合約值 = {p2_start:.2f} × 2000 = {calc_contract_value(p2_start):,.0f}")
    print(f"  {SYM2} {contracts_s2}口保證金 = {calc_contract_value(p2_start):,.0f} × {MARGIN_RATE*100:.1f}% × {contracts_s2} = {margin_s2:,.0f}")
    print(f"\n  >> 單次配對開倉最少保證金需求: {total_margin_needed:,.0f} 元")
    print(f"  >> 初始帳戶資金: {INITIAL_MARGIN_CAPITAL:,.0f} 元")
    
    if total_margin_needed > INITIAL_MARGIN_CAPITAL:
        print(f"\n  !! 警告: 最少保證金 {total_margin_needed:,.0f} > 帳戶資金 {INITIAL_MARGIN_CAPITAL:,.0f}")
        print(f"  帳戶資金不足以開倉！")
    else:
        print(f"  OK 帳戶資金足夠開倉（使用率 {total_margin_needed/INITIAL_MARGIN_CAPITAL*100:.1f}%）")

    # ============================================================
    # 3. 逐日模擬交易
    # ============================================================
    print(f"\n{'─' * 60}")
    print(f"  開始逐日模擬...")
    print(f"{'─' * 60}")
    
    valid_dates = zscore.index
    
    # 交易狀態
    position = 0  # 0=空手, 1=做多Spread(多S2空S1), -1=做空Spread(空S2多S1)
    
    # 開倉資訊
    entry_price_s1 = 0.0
    entry_price_s2 = 0.0
    entry_date = None
    
    # 帳戶紀錄
    cash = INITIAL_MARGIN_CAPITAL  # 可用現金（還沒被凍結的）
    realized_pnl_total = 0.0      # 累計已實現損益
    
    # 每日紀錄
    records = []
    trade_log = []  # 交易明細
    
    for date in valid_dates:
        z = zscore[date]
        p1 = S1[date]
        p2 = S2[date]
        
        # 當日需要的保證金（以當日價格計算）
        margin_s1_today = calc_margin_per_contract(p1) * contracts_s1
        margin_s2_today = calc_margin_per_contract(p2) * contracts_s2
        margin_required_today = margin_s1_today + margin_s2_today
        
        unrealized_pnl = 0.0
        action = ''
        
        # --- 計算持倉未實現損益 ---
        if position != 0:
            # 股期每日損益 = (今日收盤 - 昨日收盤或開倉價) × 股數 × 口數
            # 做多 Spread: 多 S2 空 S1
            #   S2 多方損益 = (P2_today - P2_entry) * 2000 * contracts_s2
            #   S1 空方損益 = (P1_entry - P1_today) * 2000 * contracts_s1
            if position == 1:  # 做多 Spread: Long S2, Short S1
                pnl_s2 = (p2 - entry_price_s2) * SHARES_PER_CONTRACT * contracts_s2
                pnl_s1 = (entry_price_s1 - p1) * SHARES_PER_CONTRACT * contracts_s1
            else:  # position == -1, 做空 Spread: Short S2, Long S1
                pnl_s2 = (entry_price_s2 - p2) * SHARES_PER_CONTRACT * contracts_s2
                pnl_s1 = (p1 - entry_price_s1) * SHARES_PER_CONTRACT * contracts_s1
                
            unrealized_pnl = pnl_s1 + pnl_s2
        
        # --- 開倉/平倉邏輯 ---
        # Z > +2 且空手 → 做空 Spread (空 S2, 多 S1)
        if z > ZSCORE_ENTRY and position == 0:
            if cash >= margin_required_today:
                position = -1
                entry_price_s1 = p1
                entry_price_s2 = p2
                entry_date = date
                # 扣交易成本
                cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
                cash -= cost
                realized_pnl_total -= cost
                action = f'開倉: 空Spread (空{SYM2}×{contracts_s2}, 多{SYM1}×{contracts_s1})'
                trade_log.append({
                    '日期': date.strftime('%Y-%m-%d'),
                    '動作': '開倉-空Spread',
                    f'{SYM1}價': p1,
                    f'{SYM2}價': p2,
                    'Z-Score': z,
                    '交易成本': cost,
                    '損益': 0
                })
            else:
                action = f'!! 保證金不足，無法開倉 (需{margin_required_today:,.0f}, 有{cash:,.0f})'
        
        # Z < -2 且空手 → 做多 Spread (多 S2, 空 S1)
        elif z < -ZSCORE_ENTRY and position == 0:
            if cash >= margin_required_today:
                position = 1
                entry_price_s1 = p1
                entry_price_s2 = p2
                entry_date = date
                cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
                cash -= cost
                realized_pnl_total -= cost
                action = f'開倉: 多Spread (多{SYM2}×{contracts_s2}, 空{SYM1}×{contracts_s1})'
                trade_log.append({
                    '日期': date.strftime('%Y-%m-%d'),
                    '動作': '開倉-多Spread',
                    f'{SYM1}價': p1,
                    f'{SYM2}價': p2,
                    'Z-Score': z,
                    '交易成本': cost,
                    '損益': 0
                })
            else:
                action = f'!! 保證金不足，無法開倉 (需{margin_required_today:,.0f}, 有{cash:,.0f})'
        
        # Z 穿越 0 且有持倉 → 平倉
        elif position != 0 and ((position == 1 and z >= ZSCORE_EXIT) or (position == -1 and z <= ZSCORE_EXIT)):
            # 平倉：實現損益
            cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
            net_pnl = unrealized_pnl - cost
            cash += net_pnl
            realized_pnl_total += net_pnl
            
            direction = '多Spread' if position == 1 else '空Spread'
            action = f'平倉: {direction}, 損益={net_pnl:+,.0f}'
            trade_log.append({
                '日期': date.strftime('%Y-%m-%d'),
                '動作': f'平倉-{direction}',
                f'{SYM1}價': p1,
                f'{SYM2}價': p2,
                'Z-Score': z,
                '交易成本': cost,
                '損益': net_pnl
            })
            
            position = 0
            unrealized_pnl = 0.0
            entry_price_s1 = 0.0
            entry_price_s2 = 0.0
        
        # --- 計算當日帳戶淨值 ---
        # 帳戶淨值 = 可用現金 + 未實現損益
        # (保證金被凍結在持倉中，但淨值包含未實現損益)
        if position != 0:
            frozen_margin = margin_required_today
            equity = cash + unrealized_pnl
        else:
            frozen_margin = 0
            equity = cash
        
        records.append({
            '日期': date,
            f'{SYM1}價': p1,
            f'{SYM2}價': p2,
            'Z-Score': z,
            '持倉': position,
            '未實現損益': unrealized_pnl,
            '累計已實現損益': realized_pnl_total,
            '凍結保證金': frozen_margin,
            '帳戶淨值': equity,
            '動作': action
        })
    
    # ============================================================
    # 4. 產出結果
    # ============================================================
    df_records = pd.DataFrame(records)
    df_records.set_index('日期', inplace=True)
    
    df_trades = pd.DataFrame(trade_log)
    
    print(f"\n{'=' * 70}")
    print(f"  交易明細")
    print(f"{'=' * 70}")
    if not df_trades.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:,.2f}'.format)
        print(df_trades.to_string(index=False))
    
    # 統計
    close_trades = df_trades[df_trades['動作'].str.contains('平倉')]
    win_trades = close_trades[close_trades['損益'] > 0]
    lose_trades = close_trades[close_trades['損益'] <= 0]
    
    total_trades = len(close_trades)
    total_pnl = close_trades['損益'].sum() if total_trades > 0 else 0
    win_rate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0
    avg_win = win_trades['損益'].mean() if len(win_trades) > 0 else 0
    avg_lose = lose_trades['損益'].mean() if len(lose_trades) > 0 else 0
    max_dd = (df_records['帳戶淨值'].cummax() - df_records['帳戶淨值']).max()
    
    print(f"\n{'=' * 70}")
    print(f"  回測統計摘要")
    print(f"{'=' * 70}")
    print(f"  回測期間: {valid_dates[0].strftime('%Y-%m-%d')} ～ {valid_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  初始資金: {INITIAL_MARGIN_CAPITAL:>12,.0f} 元")
    print(f"  最終淨值: {df_records['帳戶淨值'].iloc[-1]:>12,.0f} 元")
    print(f"  累計報酬: {df_records['帳戶淨值'].iloc[-1] - INITIAL_MARGIN_CAPITAL:>+12,.0f} 元 ({(df_records['帳戶淨值'].iloc[-1]/INITIAL_MARGIN_CAPITAL - 1)*100:+.2f}%)")
    print(f"  最大回撤: {max_dd:>12,.0f} 元")
    print(f"  總交易次數(完整來回): {total_trades} 次")
    print(f"  勝率: {win_rate:.1f}%")
    print(f"  平均獲利: {avg_win:>+12,.0f} 元")
    print(f"  平均虧損: {avg_lose:>+12,.0f} 元")
    print(f"  獲利因子: {abs(avg_win/avg_lose):.2f}" if avg_lose != 0 else "  獲利因子: N/A")
    
    # 保證金使用摘要
    margin_when_holding = df_records[df_records['持倉'] != 0]['凍結保證金']
    if not margin_when_holding.empty:
        print(f"\n  持倉期間保證金使用:")
        print(f"    最小凍結保證金: {margin_when_holding.min():>12,.0f} 元")
        print(f"    最大凍結保證金: {margin_when_holding.max():>12,.0f} 元")
        print(f"    平均凍結保證金: {margin_when_holding.mean():>12,.0f} 元")
    
    # ============================================================
    # 5. 繪製圖表
    # ============================================================
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
    
    # (1) 兩檔股價
    ax1 = axes[0]
    ax1.plot(df_records.index, df_records[f'{SYM1}價'], label=f'{SYM1} 中興電', color='#2196F3', linewidth=1)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_records.index, df_records[f'{SYM2}價'], label=f'{SYM2} 樺漢', color='#FF9800', linewidth=1)
    ax1.set_ylabel(f'{SYM1} 價格', color='#2196F3')
    ax1_twin.set_ylabel(f'{SYM2} 價格', color='#FF9800')
    ax1.set_title(f'{SYM1} (中興電) vs {SYM2} (樺漢) 股價走勢', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # (2) Z-Score + 進出場訊號
    ax2 = axes[1]
    ax2.plot(df_records.index, df_records['Z-Score'], color='#9C27B0', linewidth=0.8, label='Z-Score')
    ax2.axhline(ZSCORE_ENTRY, color='red', linestyle='--', alpha=0.7, label=f'+{ZSCORE_ENTRY} 閾值')
    ax2.axhline(-ZSCORE_ENTRY, color='green', linestyle='--', alpha=0.7, label=f'-{ZSCORE_ENTRY} 閾值')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.fill_between(df_records.index, ZSCORE_ENTRY, df_records['Z-Score'],
                     where=df_records['Z-Score'] > ZSCORE_ENTRY, alpha=0.3, color='red')
    ax2.fill_between(df_records.index, -ZSCORE_ENTRY, df_records['Z-Score'],
                     where=df_records['Z-Score'] < -ZSCORE_ENTRY, alpha=0.3, color='green')
    
    # 標記開平倉點
    for _, trade in df_trades.iterrows():
        td = pd.Timestamp(trade['日期'])
        if td in df_records.index:
            z_val = df_records.loc[td, 'Z-Score']
            if '開倉' in trade['動作']:
                ax2.scatter(td, z_val, marker='o', s=60, color='blue', zorder=5, edgecolors='black')
            elif '平倉' in trade['動作']:
                ax2.scatter(td, z_val, marker='x', s=60, color='black', zorder=5, linewidths=2)
    
    ax2.set_ylabel('Z-Score')
    ax2.set_title('Z-Score 與進出場訊號 (●=開倉, X=平倉)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # (3) 持倉狀態
    ax3 = axes[2]
    ax3.fill_between(df_records.index, 0, df_records['持倉'], 
                     where=df_records['持倉'] > 0, alpha=0.5, color='green', label='多Spread')
    ax3.fill_between(df_records.index, 0, df_records['持倉'],
                     where=df_records['持倉'] < 0, alpha=0.5, color='red', label='空Spread')
    ax3.set_ylabel('持倉方向')
    ax3.set_title('持倉狀態 (1=多Spread, -1=空Spread, 0=空手)', fontsize=14, fontweight='bold')
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['空Spread', '空手', '多Spread'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # (4) 未實現損益 & 已實現損益
    ax4 = axes[3]
    ax4.fill_between(df_records.index, 0, df_records['未實現損益'], 
                     where=df_records['未實現損益'] >= 0, alpha=0.4, color='green')
    ax4.fill_between(df_records.index, 0, df_records['未實現損益'],
                     where=df_records['未實現損益'] < 0, alpha=0.4, color='red')
    ax4.plot(df_records.index, df_records['未實現損益'], color='gray', linewidth=0.5, label='未實現損益')
    ax4.plot(df_records.index, df_records['累計已實現損益'], color='blue', linewidth=1.5, label='累計已實現損益')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_ylabel('損益 (TWD)')
    ax4.set_title('損益追蹤 (未實現 + 已實現)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # (5) 帳戶淨值 & 保證金使用
    ax5 = axes[4]
    ax5.plot(df_records.index, df_records['帳戶淨值'], color='#4CAF50', linewidth=1.5, label='帳戶淨值')
    ax5.axhline(INITIAL_MARGIN_CAPITAL, color='gray', linestyle='--', alpha=0.5, label=f'初始資金 {INITIAL_MARGIN_CAPITAL:,}')
    ax5.fill_between(df_records.index, 0, df_records['凍結保證金'], alpha=0.3, color='orange', label='凍結保證金')
    ax5.set_ylabel('金額 (TWD)')
    ax5.set_xlabel('日期')
    ax5.set_title('帳戶淨值 & 凍結保證金', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f'MarginBacktest_{SYM1}_{SYM2}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n圖表已存檔: {output_file}")
    
    # 存交易明細
    df_trades.to_csv('trade_log.csv', index=False, encoding='utf-8-sig')
    df_records.to_csv('daily_records.csv', encoding='utf-8-sig')
    print("交易明細已存: trade_log.csv")
    print("每日紀錄已存: daily_records.csv")
    

if __name__ == "__main__":
    main()
