"""
台灣股票期貨 10 組配對交易批次回測
==================================
總資金 1,000 萬 TWD，每組配對分配 100 萬 TWD
使用收盤價計算，含完整結算轉倉/漲跌停過濾機制

輸出：
  1. 每組配對詳細回測報告（CSV + 圖表）
  2. 整合 1000 萬組合績效（MDD / Sharpe Ratio / 淨值曲線）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import os
import datetime
import calendar
import warnings

warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 常數
# ============================================================
SHARES_PER_CONTRACT = 2000
MARGIN_RATE = 0.135
TRADING_FEE_RATE = 0.00002
COMMISSION_PER_CONTRACT = 30
LIMIT_PCT = 0.10
RISK_FREE_RATE = 0.015  # 年化無風險利率（台灣定存約 1.5%）

CAPITAL_PER_PAIR = 1_000_000   # 每組 100 萬
TOTAL_CAPITAL = 10_000_000     # 總資金 1000 萬

# Z-Score 參數（10組統一）
# Z_ENTRY 可手動調整範圍: 0.5 ~ 3.0（預設 2.0）
Z_ENTRY = 2.0
Z_EXIT = 0.0
Z_WINDOW = 60

# 是否自動最大化口數（根據保證金）
AUTO_MAX_CONTRACTS = True

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_prices.csv')
TAIFEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'taifex_stocks.csv')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'batch_10pairs_v7')

# ============================================================
# 10 組配對定義 (全部 Sharpe>1, Return>100%, MDD<40%)
# ============================================================
PAIR_CONFIGS = [
    {'id': 1,  'sym1': '8039.TW',   'sym2': '3711.TW',   'name1': '台虹',   'name2': '日月光投控','industry': 'PI膜/半導體封測'},
    {'id': 2,  'sym1': '3227.TWO',  'sym2': '2603.TW',   'name1': '原相',   'name2': '長榮',    'industry': 'IC設計/航運'},
    {'id': 3,  'sym1': '3680.TWO',  'sym2': '1216.TW',   'name1': '家登',   'name2': '統一',    'industry': '半導體設備/食品'},
    {'id': 4,  'sym1': '3706.TW',   'sym2': '2059.TW',   'name1': '神達',   'name2': '川湖',    'industry': 'IPC/伺服器機殼'},
    {'id': 5,  'sym1': '6669.TW',   'sym2': '1504.TW',   'name1': '緯穎',   'name2': '東元',    'industry': '伺服器/重電'},
    {'id': 6,  'sym1': '2330.TW',   'sym2': '3653.TW',   'name1': '台積電', 'name2': '健策',    'industry': '晶圓代工/散熱'},
    {'id': 7,  'sym1': '3045.TW',   'sym2': '6269.TW',   'name1': '台灣大', 'name2': '台郡',    'industry': '電信/PCB'},
    {'id': 8,  'sym1': '6414.TW',   'sym2': '6121.TWO',  'name1': '樺漢',   'name2': '新普',    'industry': '工業電腦/電池'},
    {'id': 9,  'sym1': '3376.TW',   'sym2': '2474.TW',   'name1': '新日興', 'name2': '可成',    'industry': '連接器/金屬機殼'},
    {'id': 10, 'sym1': '6547.TWO',  'sym2': '1795.TW',   'name1': '高端疫苗','name2': '美時',    'industry': '生技/製藥'},
]




# ============================================================
# 工具函式
# ============================================================
def detect_limit_days(series):
    """偵測漲跌停日"""
    pct = series.pct_change()
    return (pct >= (LIMIT_PCT - 0.001)) | (pct <= -(LIMIT_PCT - 0.001))


def calc_contract_value(price):
    return price * SHARES_PER_CONTRACT


def calc_margin_per_contract(price):
    return calc_contract_value(price) * MARGIN_RATE


def calc_trading_cost(price, contracts):
    contract_value = calc_contract_value(price) * contracts
    tax = contract_value * TRADING_FEE_RATE
    commission = COMMISSION_PER_CONTRACT * contracts
    return tax + commission


def find_min_contracts(price_s1, price_s2):
    """找最少配對口數（加入口數懲罰，避免盲目放大）"""
    target_ratio = price_s2 / price_s1
    best_score = float('inf')
    best_s1, best_s2 = 1, 1
    for s2 in range(1, 21):
        for s1 in range(1, 21):
            ratio = s1 / s2
            error = abs(ratio - target_ratio)
            score = error + (s1 + s2) * 0.015
            if score < best_score:
                best_score = score
                best_s1, best_s2 = s1, s2
    return best_s1, best_s2


def calc_max_contracts(price_s1, price_s2, capital, margin_rate=MARGIN_RATE):
    """
    根據保證金自動計算最大口數：
    1. 先找最小比例口數 (c1_base, c2_base)
    2. 計算每組最小單位的保證金 = margin_s1*c1_base + margin_s2*c2_base
    3. 倍數 = floor(capital / 每組最小單位保證金)
    4. 最終口數 = c1_base * 倍數, c2_base * 倍數
    """
    c1_base, c2_base = find_min_contracts(price_s1, price_s2)
    margin_per_unit = (
        price_s1 * SHARES_PER_CONTRACT * margin_rate * c1_base +
        price_s2 * SHARES_PER_CONTRACT * margin_rate * c2_base
    )
    if margin_per_unit <= 0:
        return c1_base, c2_base, 1
    multiplier = int(capital / margin_per_unit)
    multiplier = max(1, multiplier)  # 至少 1 倍
    return c1_base * multiplier, c2_base * multiplier, multiplier


def get_third_wednesdays(start_year, end_year):
    """產生指定年份範圍內所有每月第三個禮拜三"""
    settlement_dates = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            cal = calendar.monthcalendar(year, month)
            wednesdays = [week[2] for week in cal if week[2] != 0]
            if len(wednesdays) >= 3:
                third_wed = datetime.date(year, month, wednesdays[2])
                settlement_dates.add(third_wed)
    return settlement_dates


def calc_sharpe_ratio(equity_series, risk_free_rate=RISK_FREE_RATE):
    """計算年化夏普比率"""
    daily_returns = equity_series.pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)


def calc_mdd(equity_series):
    """計算最大回撤金額與百分比"""
    peak = equity_series.cummax()
    drawdown = peak - equity_series
    drawdown_pct = drawdown / peak
    mdd_amount = drawdown.max()
    mdd_pct = drawdown_pct.max() * 100
    # 找到 MDD 發生的區間
    mdd_end_idx = drawdown.idxmax()
    mdd_peak_idx = equity_series.loc[:mdd_end_idx].idxmax()
    return mdd_amount, mdd_pct, mdd_peak_idx, mdd_end_idx


# ============================================================
# 單組配對回測
# ============================================================
def run_single_backtest(S1, S2, sym1, sym2, initial_capital,
                        name1='', name2='',
                        z_entry=Z_ENTRY, z_exit=Z_EXIT, z_window=Z_WINDOW,
                        auto_max=AUTO_MAX_CONTRACTS):
    """
    執行單組配對交易回測（含漲跌停過濾、每月結算轉倉）
    回傳: (df_records, df_trades, stats)
    """
    # 對齊
    common_idx = S1.index.intersection(S2.index)
    S1 = S1[common_idx].copy()
    S2 = S2[common_idx].copy()

    # 漲跌停
    limit_s1 = detect_limit_days(S1)
    limit_s2 = detect_limit_days(S2)
    any_limit = limit_s1 | limit_s2

    # OLS Hedge Ratio
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params[sym1]

    # Spread & Z-Score
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=z_window).mean()
    spread_std = spread.rolling(window=z_window).std()
    zscore = (spread - spread_mean) / spread_std
    zscore = zscore.dropna()

    # 配對口數 — 根據保證金自動最大化
    start_date = zscore.index[0]
    p1_start = S1[start_date]
    p2_start = S2[start_date]
    usable_capital = initial_capital * 0.8
    
    if auto_max:
        contracts_s1, contracts_s2, multiplier = calc_max_contracts(
            p1_start, p2_start, usable_capital
        )
    else:
        contracts_s1, contracts_s2 = find_min_contracts(p1_start, p2_start)
        multiplier = 1

    margin_s1 = calc_margin_per_contract(p1_start) * contracts_s1
    margin_s2 = calc_margin_per_contract(p2_start) * contracts_s2
    total_margin_needed = margin_s1 + margin_s2

    # 結算日
    start_year = zscore.index[0].year
    end_year = zscore.index[-1].year
    settlement_dates = get_third_wednesdays(start_year, end_year)

    # 逐日模擬
    valid_dates = zscore.index
    dates_list = list(valid_dates)
    position = 0
    entry_price_s1 = 0.0
    entry_price_s2 = 0.0
    entry_date = None
    entry_zscore = 0.0
    cash = initial_capital
    realized_pnl_total = 0.0
    records = []
    trade_log = []
    skipped_limits = 0
    settlement_rolls = 0
    total_roll_cost = 0.0
    pending_reopen = None
    trade_counter = 0  # 追蹤交易編號

    for i, date in enumerate(dates_list):
        z = zscore[date]
        p1 = S1[date]
        p2 = S2[date]
        is_limit = any_limit.get(date, False)
        is_settlement = date.date() in settlement_dates

        # 如果目前有持倉，計算現有持倉的所需保證金
        if position != 0:
            margin_s1_today = calc_margin_per_contract(p1) * contracts_s1
            margin_s2_today = calc_margin_per_contract(p2) * contracts_s2
            margin_required_today = margin_s1_today + margin_s2_today
        else:
            margin_required_today = 0

        unrealized_pnl = 0.0
        action = ''
        
        # 動態決定口數的內部函數
        def size_trade():
            if auto_max:
                c1, c2, _ = calc_max_contracts(p1, p2, cash * 0.8) # Batch default usage
            else:
                c1, c2 = find_min_contracts(p1, p2)
            req = calc_margin_per_contract(p1) * c1 + calc_margin_per_contract(p2) * c2
            return c1, c2, req

        # 計算未實現損益
        if position != 0:
            if position == 1:
                pnl_s2 = (p2 - entry_price_s2) * SHARES_PER_CONTRACT * contracts_s2
                pnl_s1 = (entry_price_s1 - p1) * SHARES_PER_CONTRACT * contracts_s1
            else:
                pnl_s2 = (entry_price_s2 - p2) * SHARES_PER_CONTRACT * contracts_s2
                pnl_s1 = (p1 - entry_price_s1) * SHARES_PER_CONTRACT * contracts_s1
            unrealized_pnl = pnl_s1 + pnl_s2

        # 描述函式: 誰做多、誰做空
        def make_desc(pos_dir):
            if pos_dir == 1:  # 多Spread = 做多S2 + 做空S1
                return (f'做空{sym1}({name1}) {contracts_s1}口 / '
                        f'做多{sym2}({name2}) {contracts_s2}口')
            else:  # 空Spread = 做多S1 + 做空S2
                return (f'做多{sym1}({name1}) {contracts_s1}口 / '
                        f'做空{sym2}({name2}) {contracts_s2}口')

        # (A) 結算日後次日重新開倉
        if pending_reopen is not None and position == 0:
            dyn_c1, dyn_c2, dyn_req = size_trade()
            
            if is_limit:
                action = '漲跌停，轉倉開倉延後'
                skipped_limits += 1
            elif dyn_c1 > 0 and cash >= dyn_req:
                contracts_s1, contracts_s2 = dyn_c1, dyn_c2
                margin_required_today = dyn_req
                position = pending_reopen
                entry_price_s1 = p1
                entry_price_s2 = p2
                entry_date = date
                entry_zscore = z
                cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
                cash -= cost
                realized_pnl_total -= cost
                total_roll_cost += cost
                direction = '多Spread' if position == 1 else '空Spread'
                desc = make_desc(position)
                action = f'轉倉開倉: {desc}'
                trade_counter += 1
                trade_log.append({
                    '交易編號': trade_counter, '日期': date.strftime('%Y-%m-%d'),
                    '動作': f'轉倉開倉-{direction}',
                    '部位描述': desc,
                    f'{sym1}({name1})多空': '做空' if position == 1 else '做多',
                    f'{sym2}({name2})多空': '做多' if position == 1 else '做空',
                    f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                    f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                    '保證金佔用': round(margin_required_today, 0),
                    f'{sym1}保證金': round(margin_s1_today, 0),
                    f'{sym2}保證金': round(margin_s2_today, 0),
                    'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0,
                    '開倉日': entry_date.strftime('%Y-%m-%d') if entry_date else '',
                    '持倉天數': 0
                })
                pending_reopen = None
            else:
                action = f'轉倉失敗: 保證金不足'
                pending_reopen = None

        # (B) 結算日強制平倉
        elif is_settlement and position != 0:
            cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
            net_pnl = unrealized_pnl - cost
            cash += net_pnl
            realized_pnl_total += net_pnl
            total_roll_cost += cost
            settlement_rolls += 1
            direction = '多Spread' if position == 1 else '空Spread'
            desc = make_desc(position)
            hold_days = (date - entry_date).days if entry_date else 0
            action = f'結算平倉: {desc}, 損益={net_pnl:+,.0f}'
            trade_counter += 1
            trade_log.append({
                '交易編號': trade_counter, '日期': date.strftime('%Y-%m-%d'),
                '動作': f'結算平倉-{direction}',
                '部位描述': desc,
                f'{sym1}({name1})多空': '做空' if position == 1 else '做多',
                f'{sym2}({name2})多空': '做多' if position == 1 else '做空',
                f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                '保證金佔用': round(margin_required_today, 0),
                f'{sym1}保證金': round(margin_s1_today, 0),
                f'{sym2}保證金': round(margin_s2_today, 0),
                '開倉Z-Score': round(entry_zscore, 2),
                '平倉Z-Score': round(z, 2),
                '交易成本': round(cost, 0),
                '損益': round(net_pnl, 0),
                '開倉日': entry_date.strftime('%Y-%m-%d') if entry_date else '',
                '持倉天數': hold_days
            })
            if (position == 1 and z < z_exit) or (position == -1 and z > z_exit):
                pending_reopen = position
            else:
                pending_reopen = None
            position = 0
            unrealized_pnl = 0.0
            entry_price_s1 = 0.0
            entry_price_s2 = 0.0
            entry_date = None
            entry_zscore = 0.0

        # (C) Z > +entry → 空 Spread
        elif z > z_entry and position == 0 and pending_reopen is None:
            if is_limit:
                action = '漲跌停，跳過開倉'
                skipped_limits += 1
            elif cash >= margin_required_today:
                position = -1
                entry_price_s1 = p1
                entry_price_s2 = p2
                entry_date = date
                entry_zscore = z
                cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
                cash -= cost
                realized_pnl_total -= cost
                desc = make_desc(-1)
                action = f'開倉: {desc}'
                trade_counter += 1
                trade_log.append({
                    '交易編號': trade_counter, '日期': date.strftime('%Y-%m-%d'),
                    '動作': '開倉-空Spread',
                    '部位描述': desc,
                    f'{sym1}({name1})多空': '做多',
                    f'{sym2}({name2})多空': '做空',
                    f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                    f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                    '保證金佔用': round(margin_required_today, 0),
                    f'{sym1}保證金': round(margin_s1_today, 0),
                    f'{sym2}保證金': round(margin_s2_today, 0),
                    'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0,
                    '開倉日': entry_date.strftime('%Y-%m-%d'),
                    '持倉天數': 0
                })
            else:
                action = f'保證金不足'

        # (D) Z < -entry → 多 Spread
        elif z < -z_entry and position == 0 and pending_reopen is None:
            if is_limit:
                action = '漲跌停，跳過開倉'
                skipped_limits += 1
            elif cash >= margin_required_today:
                position = 1
                entry_price_s1 = p1
                entry_price_s2 = p2
                entry_date = date
                entry_zscore = z
                cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
                cash -= cost
                realized_pnl_total -= cost
                desc = make_desc(1)
                action = f'開倉: {desc}'
                trade_counter += 1
                trade_log.append({
                    '交易編號': trade_counter, '日期': date.strftime('%Y-%m-%d'),
                    '動作': '開倉-多Spread',
                    '部位描述': desc,
                    f'{sym1}({name1})多空': '做空',
                    f'{sym2}({name2})多空': '做多',
                    f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                    f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                    '保證金佔用': round(margin_required_today, 0),
                    f'{sym1}保證金': round(margin_s1_today, 0),
                    f'{sym2}保證金': round(margin_s2_today, 0),
                    'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0,
                    '開倉日': entry_date.strftime('%Y-%m-%d'),
                    '持倉天數': 0
                })
            else:
                action = f'保證金不足'

        # (E) 平倉
        elif position != 0 and (
            (position == 1 and z >= z_exit) or (position == -1 and z <= z_exit)
        ):
            if is_limit:
                action = '漲跌停，無法平倉'
                skipped_limits += 1
            else:
                cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
                net_pnl = unrealized_pnl - cost
                cash += net_pnl
                realized_pnl_total += net_pnl
                direction = '多Spread' if position == 1 else '空Spread'
                desc = make_desc(position)
                hold_days = (date - entry_date).days if entry_date else 0
                action = f'平倉: {desc}, 損益={net_pnl:+,.0f}'
                trade_counter += 1
                trade_log.append({
                    '交易編號': trade_counter, '日期': date.strftime('%Y-%m-%d'),
                    '動作': f'平倉-{direction}',
                    '部位描述': desc,
                    f'{sym1}({name1})多空': '做空' if position == 1 else '做多',
                    f'{sym2}({name2})多空': '做多' if position == 1 else '做空',
                    f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                    f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                    '保證金佔用': round(margin_required_today, 0),
                    f'{sym1}保證金': round(margin_s1_today, 0),
                    f'{sym2}保證金': round(margin_s2_today, 0),
                    '開倉Z-Score': round(entry_zscore, 2),
                    '平倉Z-Score': round(z, 2),
                    '交易成本': round(cost, 0),
                    '損益': round(net_pnl, 0),
                    '開倉日': entry_date.strftime('%Y-%m-%d') if entry_date else '',
                    '持倉天數': hold_days
                })
                position = 0
                unrealized_pnl = 0.0
                entry_price_s1 = 0.0
                entry_price_s2 = 0.0
                entry_date = None
                entry_zscore = 0.0

        # 淨值計算
        if position != 0:
            if position == 1:
                pnl_s2 = (p2 - entry_price_s2) * SHARES_PER_CONTRACT * contracts_s2
                pnl_s1 = (entry_price_s1 - p1) * SHARES_PER_CONTRACT * contracts_s1
            else:
                pnl_s2 = (entry_price_s2 - p2) * SHARES_PER_CONTRACT * contracts_s2
                pnl_s1 = (p1 - entry_price_s1) * SHARES_PER_CONTRACT * contracts_s1
            unrealized_pnl = pnl_s1 + pnl_s2
            frozen_margin = margin_required_today
            equity = cash + unrealized_pnl
        else:
            frozen_margin = 0
            equity = cash

        records.append({
            '日期': date,
            f'{sym1}價': p1, f'{sym2}價': p2,
            'Z-Score': z, '持倉': position,
            f'{sym1}口數': contracts_s1 if position != 0 else 0,
            f'{sym2}口數': contracts_s2 if position != 0 else 0,
            '未實現損益': unrealized_pnl,
            '累計已實現損益': realized_pnl_total,
            '凍結保證金': frozen_margin,
            f'{sym1}保證金': margin_s1_today if position != 0 else 0,
            f'{sym2}保證金': margin_s2_today if position != 0 else 0,
            '帳戶淨值': equity,
            '動作': action
        })

    df_records = pd.DataFrame(records)
    df_records.set_index('日期', inplace=True)
    df_trades = pd.DataFrame(trade_log)

    # 統計
    close_trades = df_trades[df_trades['動作'].str.contains('平倉')] if not df_trades.empty else pd.DataFrame()
    stats = {}
    stats['hedge_ratio'] = hedge_ratio
    stats['contracts_s1'] = contracts_s1
    stats['contracts_s2'] = contracts_s2
    stats['multiplier'] = multiplier if auto_max else 1
    stats['total_margin_needed'] = total_margin_needed
    stats['margin_s1'] = margin_s1
    stats['margin_s2'] = margin_s2
    stats['margin_utilization'] = (total_margin_needed / initial_capital * 100) if initial_capital > 0 else 0
    stats['p1_start'] = p1_start
    stats['p2_start'] = p2_start
    stats['start_date'] = valid_dates[0]
    stats['end_date'] = valid_dates[-1]
    stats['initial_capital'] = initial_capital
    stats['final_equity'] = df_records['帳戶淨值'].iloc[-1]
    stats['total_return'] = df_records['帳戶淨值'].iloc[-1] - initial_capital
    stats['total_return_pct'] = (df_records['帳戶淨值'].iloc[-1] / initial_capital - 1) * 100

    # 年化報酬率
    trading_days = len(df_records)
    years = trading_days / 252
    if years > 0 and stats['final_equity'] > 0:
        stats['annualized_return'] = ((stats['final_equity'] / initial_capital) ** (1 / years) - 1) * 100
    else:
        stats['annualized_return'] = 0.0

    # MDD
    mdd_amount, mdd_pct, mdd_peak, mdd_trough = calc_mdd(df_records['帳戶淨值'])
    stats['max_drawdown'] = mdd_amount
    stats['max_drawdown_pct'] = mdd_pct
    stats['mdd_peak_date'] = mdd_peak
    stats['mdd_trough_date'] = mdd_trough

    # Sharpe Ratio
    stats['sharpe_ratio'] = calc_sharpe_ratio(df_records['帳戶淨值'])

    if not close_trades.empty:
        win = close_trades[close_trades['損益'] > 0]
        lose = close_trades[close_trades['損益'] <= 0]
        stats['total_trades'] = len(close_trades)
        stats['win_trades'] = len(win)
        stats['lose_trades'] = len(lose)
        stats['win_rate'] = len(win) / len(close_trades) * 100
        stats['avg_win'] = win['損益'].mean() if len(win) > 0 else 0
        stats['avg_lose'] = lose['損益'].mean() if len(lose) > 0 else 0
        stats['max_win'] = win['損益'].max() if len(win) > 0 else 0
        stats['max_lose'] = lose['損益'].min() if len(lose) > 0 else 0
        stats['profit_factor'] = abs(win['損益'].sum() / lose['損益'].sum()) if lose['損益'].sum() != 0 else float('inf')
        stats['total_realized_pnl'] = close_trades['損益'].sum()
    else:
        for k in ['total_trades', 'win_trades', 'lose_trades', 'win_rate',
                   'avg_win', 'avg_lose', 'max_win', 'max_lose', 'profit_factor', 'total_realized_pnl']:
            stats[k] = 0

    stats['skipped_limits'] = skipped_limits
    stats['settlement_rolls'] = settlement_rolls
    stats['total_roll_cost'] = total_roll_cost

    # 共整合 p-value
    try:
        _, pvalue, _ = coint(S1[common_idx], S2[common_idx])
        stats['coint_pvalue'] = pvalue
    except:
        stats['coint_pvalue'] = None

    return df_records, df_trades, stats


# ============================================================
# 圖表繪製 — 單組
# ============================================================
def plot_single_pair(df_records, df_trades, stats, config, save_dir):
    """繪製單組配對回測圖表"""
    sym1, sym2 = config['sym1'], config['sym2']
    name1, name2 = config['name1'], config['name2']
    pair_id = config['id']

    fig, axes = plt.subplots(5, 1, figsize=(18, 24), sharex=True)
    fig.suptitle(
        f"配對 #{pair_id}: {sym1}({name1}) vs {sym2}({name2}) — {config['industry']}\n"
        f"報酬: {stats['total_return']:+,.0f} ({stats['total_return_pct']:+.1f}%) | "
        f"夏普: {stats['sharpe_ratio']:.2f} | MDD: {stats['max_drawdown']:,.0f} ({stats['max_drawdown_pct']:.1f}%)",
        fontsize=14, fontweight='bold', y=0.98
    )

    # (1) 股價走勢
    ax1 = axes[0]
    ax1.plot(df_records.index, df_records[f'{sym1}價'], label=f'{sym1} {name1}', color='#2196F3', linewidth=1)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_records.index, df_records[f'{sym2}價'], label=f'{sym2} {name2}', color='#FF9800', linewidth=1)
    ax1.set_ylabel(f'{sym1} 價格', color='#2196F3')
    ax1_twin.set_ylabel(f'{sym2} 價格', color='#FF9800')
    ax1.set_title(f'股價走勢', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # (2) Z-Score + 訊號
    ax2 = axes[1]
    ax2.plot(df_records.index, df_records['Z-Score'], color='#9C27B0', linewidth=0.8, label='Z-Score')
    ax2.axhline(Z_ENTRY, color='red', linestyle='--', alpha=0.7, label=f'+{Z_ENTRY}')
    ax2.axhline(-Z_ENTRY, color='green', linestyle='--', alpha=0.7, label=f'-{Z_ENTRY}')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.fill_between(df_records.index, Z_ENTRY, df_records['Z-Score'],
                     where=df_records['Z-Score'] > Z_ENTRY, alpha=0.3, color='red')
    ax2.fill_between(df_records.index, -Z_ENTRY, df_records['Z-Score'],
                     where=df_records['Z-Score'] < -Z_ENTRY, alpha=0.3, color='green')

    if not df_trades.empty:
        for _, trade in df_trades.iterrows():
            td = pd.Timestamp(trade['日期'])
            if td in df_records.index:
                z_val = df_records.loc[td, 'Z-Score']
                if '開倉' in trade['動作']:
                    ax2.scatter(td, z_val, marker='o', s=50, color='blue', zorder=5, edgecolors='black')
                elif '平倉' in trade['動作']:
                    color = 'green' if trade.get('損益', 0) > 0 else 'red'
                    ax2.scatter(td, z_val, marker='x', s=50, color=color, zorder=5, linewidths=2)

    ax2.set_ylabel('Z-Score')
    ax2.set_title('Z-Score 與進出場訊號 (●=開倉, ✕=平倉)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # (3) 持倉狀態
    ax3 = axes[2]
    ax3.fill_between(df_records.index, 0, df_records['持倉'],
                     where=df_records['持倉'] > 0, alpha=0.5, color='green', label='多Spread')
    ax3.fill_between(df_records.index, 0, df_records['持倉'],
                     where=df_records['持倉'] < 0, alpha=0.5, color='red', label='空Spread')
    ax3.set_ylabel('持倉方向')
    ax3.set_title('持倉狀態', fontsize=12, fontweight='bold')
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['空Spread', '空手', '多Spread'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) 損益追蹤
    ax4 = axes[3]
    ax4.fill_between(df_records.index, 0, df_records['未實現損益'],
                     where=df_records['未實現損益'] >= 0, alpha=0.4, color='green')
    ax4.fill_between(df_records.index, 0, df_records['未實現損益'],
                     where=df_records['未實現損益'] < 0, alpha=0.4, color='red')
    ax4.plot(df_records.index, df_records['未實現損益'], color='gray', linewidth=0.5, label='未實現損益')
    ax4.plot(df_records.index, df_records['累計已實現損益'], color='blue', linewidth=1.5, label='累計已實現損益')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_ylabel('損益 (TWD)')
    ax4.set_title('損益追蹤 (未實現 + 已實現)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # (5) 帳戶淨值 + 回撤
    ax5 = axes[4]
    ax5.plot(df_records.index, df_records['帳戶淨值'], color='#4CAF50', linewidth=1.5, label='帳戶淨值')
    peak = df_records['帳戶淨值'].cummax()
    ax5.plot(df_records.index, peak, color='blue', linewidth=0.8, linestyle='--', alpha=0.5, label='淨值高水位')
    ax5.fill_between(df_records.index, df_records['帳戶淨值'], peak, alpha=0.2, color='red', label='回撤區間')
    ax5.axhline(stats['initial_capital'], color='gray', linestyle='--', alpha=0.5,
                label=f"初始資金 {stats['initial_capital']:,}")
    ax5.set_ylabel('金額 (TWD)')
    ax5.set_xlabel('日期')
    ax5.set_title(f"帳戶淨值 (最終: {stats['final_equity']:,.0f})", fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f"pair_{pair_id:02d}_{sym1.replace('.', '_')}_{sym2.replace('.', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================
# 圖表繪製 — 組合總覽
# ============================================================
def plot_portfolio_summary(all_results, portfolio_equity, save_dir):
    """繪製 1000 萬組合總覽圖表"""

    fig = plt.figure(figsize=(20, 28))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

    # === (1) 組合淨值曲線 ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(portfolio_equity.index, portfolio_equity.values, color='#4CAF50', linewidth=2, label='組合淨值')
    peak = portfolio_equity.cummax()
    ax1.plot(portfolio_equity.index, peak.values, color='blue', linewidth=0.8, linestyle='--', alpha=0.5, label='高水位')
    ax1.fill_between(portfolio_equity.index, portfolio_equity.values, peak.values, alpha=0.15, color='red')
    ax1.axhline(TOTAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label=f'初始資金 {TOTAL_CAPITAL:,.0f}')

    portfolio_sharpe = calc_sharpe_ratio(portfolio_equity)
    mdd_amt, mdd_pct, _, _ = calc_mdd(portfolio_equity)
    final_eq = portfolio_equity.iloc[-1]
    total_ret = final_eq - TOTAL_CAPITAL
    total_ret_pct = (final_eq / TOTAL_CAPITAL - 1) * 100

    ax1.set_title(
        f'1,000萬組合淨值曲線\n'
        f'總報酬: {total_ret:+,.0f} ({total_ret_pct:+.1f}%) | '
        f'夏普比率: {portfolio_sharpe:.2f} | '
        f'MDD: {mdd_amt:,.0f} ({mdd_pct:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    ax1.set_ylabel('淨值 (TWD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # === (2) 各組每日淨值走勢 ===
    ax2 = fig.add_subplot(gs[1, :])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, result in enumerate(all_results):
        config = result['config']
        records = result['records']
        label = f"#{config['id']} {config['name1']}/{config['name2']}"
        ax2.plot(records.index, records['帳戶淨值'], linewidth=1, color=colors[i], label=label, alpha=0.8)
    ax2.axhline(CAPITAL_PER_PAIR, color='gray', linestyle='--', alpha=0.3)
    ax2.set_title('各組配對淨值走勢 (每組 100 萬)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('淨值 (TWD)')
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # === (3) 各組報酬率柱狀圖 ===
    ax3 = fig.add_subplot(gs[2, 0])
    pair_names = [f"#{r['config']['id']}\n{r['config']['name1']}/{r['config']['name2']}" for r in all_results]
    returns = [r['stats']['total_return'] for r in all_results]
    bar_colors = ['#4CAF50' if r >= 0 else '#F44336' for r in returns]
    bars = ax3.bar(pair_names, returns, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_title('各組累計報酬 (TWD)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('報酬 (TWD)')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, returns):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:+,.0f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)
    ax3.tick_params(axis='x', labelsize=7)

    # === (4) 各組夏普比率 ===
    ax4 = fig.add_subplot(gs[2, 1])
    sharpes = [r['stats']['sharpe_ratio'] for r in all_results]
    sharpe_colors = ['#4CAF50' if s >= 0 else '#F44336' for s in sharpes]
    bars2 = ax4.bar(pair_names, sharpes, color=sharpe_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_title('各組夏普比率', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, sharpes):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=7)
    ax4.tick_params(axis='x', labelsize=7)

    # === (5) 各組 MDD 柱狀圖 ===
    ax5 = fig.add_subplot(gs[3, 0])
    mdds = [r['stats']['max_drawdown'] for r in all_results]
    ax5.bar(pair_names, mdds, color='#FF5722', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.set_title('各組最大回撤 (TWD)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('MDD (TWD)')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.tick_params(axis='x', labelsize=7)

    # === (6) 各組勝率 ===
    ax6 = fig.add_subplot(gs[3, 1])
    win_rates = [r['stats']['win_rate'] for r in all_results]
    wr_colors = ['#4CAF50' if wr >= 50 else '#FF9800' for wr in win_rates]
    bars3 = ax6.bar(pair_names, win_rates, color=wr_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax6.set_title('各組勝率 (%)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('勝率 (%)')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, win_rates):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}%', ha='center', va='bottom', fontsize=7)
    ax6.tick_params(axis='x', labelsize=7)

    # === (7) 組合回撤曲線 ===
    ax7 = fig.add_subplot(gs[4, :])
    drawdown = peak - portfolio_equity
    drawdown_pct = drawdown / peak * 100
    ax7.fill_between(drawdown_pct.index, 0, -drawdown_pct.values, color='#F44336', alpha=0.4)
    ax7.plot(drawdown_pct.index, -drawdown_pct.values, color='#F44336', linewidth=0.8)
    ax7.set_title(f'組合回撤曲線 (最大回撤: {mdd_pct:.1f}%)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('回撤 (%)')
    ax7.set_xlabel('日期')
    ax7.grid(True, alpha=0.3)

    filepath = os.path.join(save_dir, 'portfolio_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================
# 主程式
# ============================================================
def main():
    print("=" * 80)
    print("  台灣股票期貨 10 組配對交易批次回測")
    print(f"  總資金: {TOTAL_CAPITAL:,.0f} TWD | 每組: {CAPITAL_PER_PAIR:,.0f} TWD")
    print(f"  Z-Score 參數: entry={Z_ENTRY}, exit={Z_EXIT}, window={Z_WINDOW}")
    print("=" * 80)

    # 建立輸出目錄
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 載入資料
    if not os.path.exists(DATA_FILE):
        print("[ERROR] 找不到股價資料檔 stock_prices.csv")
        return

    prices = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    prices = prices.dropna(axis=1, thresh=len(prices) * 0.8)
    prices = prices.ffill().bfill()
    print(f"\n載入資料: {len(prices)} 日 × {len(prices.columns)} 檔標的")
    print(f"期間: {prices.index[0].strftime('%Y-%m-%d')} ~ {prices.index[-1].strftime('%Y-%m-%d')}")

    # 載入公司名稱
    name_map = {}
    if os.path.exists(TAIFEX_FILE):
        try:
            taifex_df = pd.read_csv(TAIFEX_FILE)
            codes = taifex_df.iloc[:, 2].dropna().astype(str).tolist()
            names = taifex_df.iloc[:, 3].dropna().astype(str).tolist()
            for code, name in zip(codes, names):
                name_map[code.strip()] = name.strip()
        except:
            pass

    # ============================================================
    # 逐組回測
    # ============================================================
    all_results = []
    summary_rows = []

    for config in PAIR_CONFIGS:
        pair_id = config['id']
        sym1, sym2 = config['sym1'], config['sym2']
        name1, name2 = config['name1'], config['name2']

        print(f"\n{'─' * 70}")
        print(f"  配對 #{pair_id}: {sym1}({name1}) vs {sym2}({name2}) — {config['industry']}")
        print(f"{'─' * 70}")

        # 檢查資料可用性
        if sym1 not in prices.columns:
            print(f"  [WARN] {sym1} 不在資料中，跳過此組")
            continue
        if sym2 not in prices.columns:
            print(f"  [WARN] {sym2} 不在資料中，跳過此組")
            continue

        S1 = prices[sym1].dropna()
        S2 = prices[sym2].dropna()

        # 執行回測
        df_records, df_trades, stats = run_single_backtest(
            S1, S2, sym1, sym2, CAPITAL_PER_PAIR
        )

        print(f"  Hedge Ratio: {stats['hedge_ratio']:.4f}")
        print(f"  配對口數: {sym1}×{stats['contracts_s1']} / {sym2}×{stats['contracts_s2']}")
        print(f"  開倉保證金需求: {stats['total_margin_needed']:,.0f}")
        print(f"  回測期間: {stats['start_date'].strftime('%Y-%m-%d')} ~ {stats['end_date'].strftime('%Y-%m-%d')}")
        print(f"  最終淨值: {stats['final_equity']:,.0f}")
        print(f"  累計報酬: {stats['total_return']:+,.0f} ({stats['total_return_pct']:+.1f}%)")
        print(f"  夏普比率: {stats['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {stats['max_drawdown']:,.0f} ({stats['max_drawdown_pct']:.1f}%)")
        print(f"  交易次數: {stats['total_trades']} (勝率: {stats['win_rate']:.1f}%)")
        print(f"  平均獲利: {stats['avg_win']:+,.0f} | 平均虧損: {stats['avg_lose']:+,.0f}")
        print(f"  結算轉倉: {stats['settlement_rolls']} 次, 轉倉成本: {stats['total_roll_cost']:,.0f}")
        if stats['coint_pvalue'] is not None:
            print(f"  共整合 p-value: {stats['coint_pvalue']:.6f}")

        # 儲存 CSV
        csv_prefix = f"pair_{pair_id:02d}_{sym1.replace('.', '_')}_{sym2.replace('.', '_')}"
        df_records.to_csv(
            os.path.join(RESULTS_DIR, f"{csv_prefix}_daily.csv"),
            encoding='utf-8-sig'
        )
        if not df_trades.empty:
            df_trades.to_csv(
                os.path.join(RESULTS_DIR, f"{csv_prefix}_trades.csv"),
                index=False, encoding='utf-8-sig'
            )
        print(f"  [OK] CSV 已存檔")

        # 繪製圖表
        chart_path = plot_single_pair(df_records, df_trades, stats, config, RESULTS_DIR)
        print(f"  [OK] 圖表已存檔: {os.path.basename(chart_path)}")

        all_results.append({
            'config': config,
            'records': df_records,
            'trades': df_trades,
            'stats': stats
        })

        # 摘要行
        summary_rows.append({
            '配對#': pair_id,
            '標的A': f"{sym1}({name1})",
            '標的B': f"{sym2}({name2})",
            '產業': config['industry'],
            'Hedge Ratio': round(stats['hedge_ratio'], 4),
            f'{sym1}口數': stats['contracts_s1'],
            f'{sym2}口數': stats['contracts_s2'],
            '初始資金': CAPITAL_PER_PAIR,
            '最終淨值': round(stats['final_equity'], 0),
            '累計報酬': round(stats['total_return'], 0),
            '報酬率%': round(stats['total_return_pct'], 2),
            '年化報酬率%': round(stats['annualized_return'], 2),
            '夏普比率': round(stats['sharpe_ratio'], 2),
            'MDD金額': round(stats['max_drawdown'], 0),
            'MDD%': round(stats['max_drawdown_pct'], 2),
            '保證金佔用': round(stats['total_margin_needed'], 0),
            '保證金利用率%': round(stats['margin_utilization'], 1),
            '口數倍數': stats.get('multiplier', 1),
            '交易次數': stats['total_trades'],
            '勝率%': round(stats['win_rate'], 1),
            '平均獲利': round(stats['avg_win'], 0),
            '平均虧損': round(stats['avg_lose'], 0),
            '獲利因子': round(stats['profit_factor'], 2) if stats['profit_factor'] != float('inf') else 'inf',
            '共整合p值': round(stats['coint_pvalue'], 6) if stats['coint_pvalue'] else 'N/A'
        })

    if not all_results:
        print("\n[ERROR] 無任何有效的回測結果")
        return

    # ============================================================
    # 組合整合
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"  1,000 萬組合整合")
    print(f"{'=' * 80}")

    # 對齊所有組的每日淨值，加總為組合淨值
    equity_dfs = []
    for result in all_results:
        eq = result['records'][['帳戶淨值']].copy()
        eq.columns = [f"pair_{result['config']['id']}"]
        equity_dfs.append(eq)

    # 合併（forward fill 補缺值）
    combined = pd.concat(equity_dfs, axis=1)
    combined = combined.ffill().bfill()

    # 組合淨值 = 各組淨值加總
    portfolio_equity = combined.sum(axis=1)

    # 組合績效指標
    portfolio_sharpe = calc_sharpe_ratio(portfolio_equity)
    portfolio_mdd_amt, portfolio_mdd_pct, portfolio_mdd_peak, portfolio_mdd_trough = calc_mdd(portfolio_equity)
    portfolio_final = portfolio_equity.iloc[-1]
    portfolio_return = portfolio_final - TOTAL_CAPITAL
    portfolio_return_pct = (portfolio_final / TOTAL_CAPITAL - 1) * 100

    # 組合年化報酬率
    days = (portfolio_equity.index[-1] - portfolio_equity.index[0]).days
    annualized_return = ((portfolio_final / TOTAL_CAPITAL) ** (365.25 / days) - 1) * 100 if days > 0 else 0

    # 總交易次數
    total_trades_all = sum(r['stats']['total_trades'] for r in all_results)
    total_wins_all = sum(r['stats']['win_trades'] for r in all_results)
    overall_win_rate = total_wins_all / total_trades_all * 100 if total_trades_all > 0 else 0

    print(f"\n  初始總資金:     {TOTAL_CAPITAL:>14,.0f} TWD")
    print(f"  最終組合淨值:   {portfolio_final:>14,.0f} TWD")
    print(f"  累計總報酬:     {portfolio_return:>+14,.0f} TWD ({portfolio_return_pct:+.2f}%)")
    print(f"  年化報酬率:     {annualized_return:>+14.2f}%")
    print(f"  組合夏普比率:   {portfolio_sharpe:>14.2f}")
    print(f"  組合最大回撤:   {portfolio_mdd_amt:>14,.0f} TWD ({portfolio_mdd_pct:.1f}%)")
    print(f"    MDD 高點:     {portfolio_mdd_peak.strftime('%Y-%m-%d')}")
    print(f"    MDD 低點:     {portfolio_mdd_trough.strftime('%Y-%m-%d')}")
    print(f"  組合交易總次數: {total_trades_all:>14d}")
    print(f"  組合總勝率:     {overall_win_rate:>14.1f}%")

    # 儲存組合每日淨值
    portfolio_df = combined.copy()
    portfolio_df['組合淨值'] = portfolio_equity
    portfolio_df['組合累計報酬'] = portfolio_equity - TOTAL_CAPITAL
    portfolio_df['組合報酬率%'] = (portfolio_equity / TOTAL_CAPITAL - 1) * 100
    peak_series = portfolio_equity.cummax()
    portfolio_df['高水位'] = peak_series
    portfolio_df['回撤金額'] = peak_series - portfolio_equity
    portfolio_df['回撤%'] = (peak_series - portfolio_equity) / peak_series * 100
    portfolio_df.to_csv(
        os.path.join(RESULTS_DIR, 'portfolio_daily.csv'),
        encoding='utf-8-sig'
    )
    print(f"\n  [OK] 組合每日紀錄已存: portfolio_daily.csv")

    # 儲存摘要表
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        os.path.join(RESULTS_DIR, 'portfolio_summary.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"  [OK] 各組摘要表已存: portfolio_summary.csv")

    # 繪製組合圖表
    chart_path = plot_portfolio_summary(all_results, portfolio_equity, RESULTS_DIR)
    print(f"  [OK] 組合總覽圖表已存: {os.path.basename(chart_path)}")

    # ============================================================
    # 最終總結表
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"  各組配對績效總覽")
    print(f"{'=' * 80}")
    print(f"{'#':>3} {'配對':>20} {'產業':>14} {'報酬':>12} {'報酬率':>8} {'年化':>7} {'夏普':>6} {'MDD%':>6} {'口數':>8} {'保證金':>10} {'交易':>4} {'勝率':>6}")
    print(f"{'─'*3} {'─'*20} {'─'*14} {'─'*12} {'─'*8} {'─'*7} {'─'*6} {'─'*6} {'─'*8} {'─'*10} {'─'*4} {'─'*6}")

    for result in all_results:
        c = result['config']
        s = result['stats']
        pair_str = f"{c['name1']}/{c['name2']}"
        contracts_str = f"{s['contracts_s1']}/{s['contracts_s2']}"
        print(f"{c['id']:>3} {pair_str:>20} {c['industry']:>14} "
              f"{s['total_return']:>+12,.0f} {s['total_return_pct']:>+7.1f}% "
              f"{s['annualized_return']:>+6.1f}% "
              f"{s['sharpe_ratio']:>6.2f} {s['max_drawdown_pct']:>5.1f}% "
              f"{contracts_str:>8} {s['total_margin_needed']:>10,.0f} "
              f"{s['total_trades']:>4} {s['win_rate']:>5.1f}%")

    print(f"{'─'*3} {'─'*20} {'─'*14} {'─'*12} {'─'*8} {'─'*7} {'─'*6} {'─'*6} {'─'*8} {'─'*10} {'─'*4} {'─'*6}")
    print(f"{'':<3} {'組合合計':>20} {'':>14} "
          f"{portfolio_return:>+12,.0f} {portfolio_return_pct:>+7.1f}% "
          f"{annualized_return:>+6.1f}% "
          f"{portfolio_sharpe:>6.2f} {portfolio_mdd_pct:>5.1f}% "
          f"{'':>8} {'':>10} "
          f"{total_trades_all:>4} {overall_win_rate:>5.1f}%")

    print(f"\n{'=' * 80}")
    print(f"  所有結果已存至: {RESULTS_DIR}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
