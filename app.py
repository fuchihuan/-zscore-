"""
台灣股票期貨配對交易回測應用
============================
功能：
1. 從 200+ 檔有發行股票期貨的標的中任選兩檔配對
2. 自訂初始保證金
3. 自動計算最少配對口數 & 保證金需求
4. 完整模擬開倉/平倉/未實現/已實現損益
5. 自動剔除漲跌停日（無法成交）
6. 每月第三個禮拜三結算日強制平倉，次日重新開倉（含轉倉成本）
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import os
import yfinance as yf
import datetime
import calendar

# ============================================================
# 頁面設定
# ============================================================
st.set_page_config(
    page_title="台股股期配對交易回測系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
    
    html, body, .stApp {
        font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
        font-size: 0.95rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8892b0;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ccd6f6;
    }
    .metric-card .value.positive { color: #64ffda; }
    .metric-card .value.negative { color: #ff6b6b; }
    
    .trade-win { background-color: rgba(100, 255, 218, 0.1); }
    .trade-lose { background-color: rgba(255, 107, 107, 0.1); }
    
    .info-box {
        background: linear-gradient(135deg, #0a192f, #112240);
        border-left: 4px solid #64ffda;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #ccd6f6;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #2d1f0e, #3d2b14);
        border-left: 4px solid #fda085;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #fda085;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a192f, #112240);
    }
    div[data-testid="stSidebar"] * {
        color: #ccd6f6 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #f093fb, #f5576c) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 常數
# ============================================================
SHARES_PER_CONTRACT = 2000
MARGIN_RATE = 0.135
TRADING_FEE_RATE = 0.00002
COMMISSION_PER_CONTRACT = 30
LIMIT_PCT = 0.10  # 台股漲跌停 10%

DATA_FILE = os.path.join(os.path.dirname(__file__), 'stock_prices.csv')
TAIFEX_FILE = os.path.join(os.path.dirname(__file__), 'taifex_stocks.csv')


# ============================================================
# 工具函式
# ============================================================
@st.cache_data(ttl=3600)
def load_price_data():
    """載入已下載的股價資料 (更新快取)"""
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    df = df.dropna(axis=1, thresh=len(df) * 0.8)
    df = df.ffill().bfill()
    return df


def get_ticker_names():
    """嘗試取得代號與公司名對照表"""
    mapping = {}
    if os.path.exists(TAIFEX_FILE):
        try:
            taifex_df = pd.read_csv(TAIFEX_FILE)
            codes = taifex_df.iloc[:, 2].dropna().astype(str).tolist()
            names = taifex_df.iloc[:, 3].dropna().astype(str).tolist()
            for code, name in zip(codes, names):
                mapping[code.strip()] = name.strip()
        except:
            pass
    return mapping


def detect_limit_days(series):
    """
    偵測漲跌停日（以 10% 為限制）
    回傳一個 boolean Series，True 表示當日觸及漲跌停
    """
    pct = series.pct_change()
    limit_up = pct >= (LIMIT_PCT - 0.001)    # 接近或等於 +10%
    limit_down = pct <= -(LIMIT_PCT - 0.001)  # 接近或等於 -10%
    return limit_up | limit_down


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
    """找最少配對口數（以實質股價等值平衡為目標，加入口數懲罰）"""
    target_ratio = price_s2 / price_s1
    best_score = float('inf')
    best_s1, best_s2 = 1, 1
    for s2 in range(1, 21):
        for s1 in range(1, 21):
            ratio = s1 / s2
            error = abs(ratio - target_ratio)
            # 加入口數懲罰，避免為了極微小的精確率而讓合約口數暴增
            score = error + (s1 + s2) * 0.015
            if score < best_score:
                best_score = score
                best_s1, best_s2 = s1, s2
    return best_s1, best_s2


def calc_max_contracts(price_s1, price_s2, capital, margin_rate=MARGIN_RATE):
    """根據保證金自動計算最大口數"""
    c1_base, c2_base = find_min_contracts(price_s1, price_s2)
    margin_per_unit = (
        price_s1 * SHARES_PER_CONTRACT * margin_rate * c1_base +
        price_s2 * SHARES_PER_CONTRACT * margin_rate * c2_base
    )
    if margin_per_unit <= 0:
        return c1_base, c2_base, 1
    multiplier = int(capital / margin_per_unit)
    if multiplier == 0:
        return c1_base, c2_base, 0
    return c1_base * multiplier, c2_base * multiplier, multiplier


def get_third_wednesdays(start_year, end_year):
    """
    產生指定年份範圍內所有「每月第三個禮拜三」的日期集合。
    台灣股票期貨在每月第三個禮拜三收盤後結算。
    """
    settlement_dates = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            cal = calendar.monthcalendar(year, month)
            # Wednesday = index 2 in monthcalendar (Mon=0)
            wednesdays = [week[2] for week in cal if week[2] != 0]
            if len(wednesdays) >= 3:
                third_wed = datetime.date(year, month, wednesdays[2])
                settlement_dates.add(third_wed)
    return settlement_dates


def run_backtest(S1, S2, sym1, sym2, name1, name2, initial_capital, 
                 z_entry, z_exit, z_window, 
                 size_mode='依保證金上限最大化（預設）', margin_usage_pct=1.0,
                 run_start_date=None, run_end_date=None):
    """
    執行完整的配對交易保證金回測
    """
    # 對齊
    common_idx = S1.index.intersection(S2.index)
    S1 = S1[common_idx].copy()
    S2 = S2[common_idx].copy()

    # 偵測漲跌停
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

    # 在產生完指標後，再進行時間切片，確保 2020/1/2 也能立刻有指標數據
    if run_start_date and run_end_date:
        zscore = zscore.loc[run_start_date:run_end_date]
    
    if zscore.empty:
        # 回傳空結構，由外部處理
        return pd.DataFrame(), pd.DataFrame(), {'insufficient_margin_error': True, 'msg': '所選區間內無可用資料或資料長度不足滾動窗口計算！'}

    # 回測期起始保證金估算
    start_date = zscore.index[0]
    p1_start = S1[start_date]
    p2_start = S2[start_date]

    usable_capital = initial_capital * margin_usage_pct

    if size_mode == '依保證金上限最大化（預設）':
        contracts_s1, contracts_s2, multiplier = calc_max_contracts(
            p1_start, p2_start, usable_capital
        )
    else:
        contracts_s1, contracts_s2 = find_min_contracts(p1_start, p2_start)
        multiplier = 1

    margin_s1 = calc_margin_per_contract(p1_start) * contracts_s1
    margin_s2 = calc_margin_per_contract(p2_start) * contracts_s2
    total_margin_needed_start = margin_s1 + margin_s2

    base_s1, base_s2 = find_min_contracts(p1_start, p2_start)
    base_margin_s1 = calc_margin_per_contract(p1_start) * base_s1
    base_margin_s2 = calc_margin_per_contract(p2_start) * base_s2
    base_margin = base_margin_s1 + base_margin_s2

    # 檢查保證金是否連一組基本口數都下不了
    insufficient_margin_error = False
    if base_margin > initial_capital:
        insufficient_margin_error = True
        contracts_s1, contracts_s2, multiplier = 0, 0, 0

    # 產生結算日集合
    start_year = zscore.index[0].year
    end_year = zscore.index[-1].year
    settlement_dates = get_third_wednesdays(start_year, end_year)

    # --- 逐日模擬 ---
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
    settlement_rolls = 0       # 結算轉倉次數
    total_roll_cost = 0.0      # 累計轉倉成本
    pending_reopen = None      # 結算後需要在次日重新開倉的方向

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
            if size_mode == '依保證金上限最大化（預設）':
                c1, c2, _ = calc_max_contracts(p1, p2, cash * margin_usage_pct)
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
                return f'做空{sym1}({name1}) {contracts_s1}口 / 做多{sym2}({name2}) {contracts_s2}口'
            else:  # 空Spread = 做多S1 + 做空S2
                return f'做多{sym1}({name1}) {contracts_s1}口 / 做空{sym2}({name2}) {contracts_s2}口'

        # =====================================================
        # (A) 結算日後次日重新開倉 (轉倉)
        # =====================================================
        if pending_reopen is not None and position == 0:
            dyn_c1, dyn_c2, dyn_req = size_trade()
            
            if is_limit:
                action = '漲跌停，轉倉開倉延後'
                skipped_limits += 1
                # 保持 pending_reopen 到下一日
            elif dyn_c1 > 0 and cash >= dyn_req and not insufficient_margin_error:
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
                direction = '做多配對' if position == 1 else '做空配對'
                trade_log.append({
                    '日期': date.strftime('%Y-%m-%d'), '動作': f'轉倉開倉({direction})',
                    '部位描述': desc,
                    f'{sym1}({name1})多空': '做空' if position == 1 else '做多',
                    f'{sym2}({name2})多空': '做多' if position == 1 else '做空',
                    f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                    f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                    '保證金佔用': round(margin_required_today, 0),
                    'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0
                })
                pending_reopen = None
            else:
                action = f'轉倉失敗: 保證金不足'
                pending_reopen = None  # 放棄轉倉
        # =====================================================
        # (B) 結算日強制平倉
        # =====================================================
        elif is_settlement and position != 0:
            # 結算日收盤強制平倉
            cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
            net_pnl = unrealized_pnl - cost
            cash += net_pnl
            realized_pnl_total += net_pnl
            total_roll_cost += cost
            settlement_rolls += 1
            direction = '做多配對' if position == 1 else '做空配對'
            desc = make_desc(position)
            action = f'結算平倉: {direction}, 損益={net_pnl:+,.0f}'
            trade_log.append({
                '日期': date.strftime('%Y-%m-%d'), '動作': f'結算平倉({direction})',
                '部位描述': f"平倉: {desc}",
                f'{sym1}({name1})多空': '平倉(買回)' if position == 1 else '平倉(賣出)',
                f'{sym2}({name2})多空': '平倉(賣出)' if position == 1 else '平倉(買回)',
                f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                '保證金佔用': round(margin_required_today, 0),
                'Z-Score': round(z, 2),  # 統一名稱，供網頁正常顯示
                '開倉Z': round(entry_zscore, 2),
                '交易成本': round(cost, 0),
                '損益': round(net_pnl, 0)
            })
            if (position == 1 and z < z_exit) or (position == -1 and z > z_exit):
                pending_reopen = position  # 記住方向，次日重新開
            else:
                pending_reopen = None  # Z-Score 已回歸，不需轉倉

            position = 0
            unrealized_pnl = 0.0
            entry_price_s1 = 0.0
            entry_price_s2 = 0.0
            entry_date = None
            entry_zscore = 0.0
        # =====================================================
        # (C) 正常交易邏輯
        # =====================================================
        # Z > +entry 且空手 → 做空 Spread
        elif z > z_entry and position == 0 and pending_reopen is None:
            dyn_c1, dyn_c2, dyn_req = size_trade()
            if is_limit:
                action = '漲跌停，跳過開倉'
                skipped_limits += 1
            elif dyn_c1 > 0 and cash >= dyn_req and not insufficient_margin_error:
                contracts_s1, contracts_s2 = dyn_c1, dyn_c2
                margin_required_today = dyn_req
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
                trade_log.append({
                    '日期': date.strftime('%Y-%m-%d'), '動作': '開倉(做空配對)',
                    '部位描述': desc,
                    f'{sym1}({name1})多空': '做多',
                    f'{sym2}({name2})多空': '做空',
                    f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                    f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                    '保證金佔用': round(margin_required_today, 0),
                    'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0
                })
            else:
                action = f'保證金不足'

        # Z < -entry 且空手 → 做多 Spread
        elif z < -z_entry and position == 0 and pending_reopen is None:
            dyn_c1, dyn_c2, dyn_req = size_trade()
            if is_limit:
                action = '漲跌停，跳過開倉'
                skipped_limits += 1
            elif dyn_c1 > 0 and cash >= dyn_req and not insufficient_margin_error:
                contracts_s1, contracts_s2 = dyn_c1, dyn_c2
                margin_required_today = dyn_req
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
                trade_log.append({
                    '日期': date.strftime('%Y-%m-%d'), '動作': '開倉(做多配對)',
                    '部位描述': desc,
                    f'{sym1}({name1})多空': '做空',
                    f'{sym2}({name2})多空': '做多',
                    f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                    f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                    '保證金佔用': round(margin_required_today, 0),
                    'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0
                })
            else:
                action = f'保證金不足'

        # Z 穿越 exit 且有持倉 → 平倉
        elif position != 0 and (
            (position == 1 and z >= z_exit) or (position == -1 and z <= z_exit)
        ):
            if is_limit:
                action = '漲跌停，無法平倉，持續持倉'
                skipped_limits += 1
            else:
                cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
                net_pnl = unrealized_pnl - cost
                cash += net_pnl
                realized_pnl_total += net_pnl
                direction = '做多配對' if position == 1 else '做空配對'
                desc = make_desc(position)
                action = f'平倉: {direction}, 損益={net_pnl:+,.0f}'
                trade_log.append({
                    '日期': date.strftime('%Y-%m-%d'), '動作': f'平倉({direction})',
                    '部位描述': f"平倉: {desc}",
                    f'{sym1}({name1})多空': '平倉(買回)' if position == 1 else '平倉(賣出)',
                    f'{sym2}({name2})多空': '平倉(賣出)' if position == 1 else '平倉(買回)',
                    f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                    f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                    '保證金佔用': round(margin_required_today, 0),
                    'Z-Score': round(z, 2),  # 統一名稱，供網頁正常顯示
                    '開倉Z': round(entry_zscore, 2),
                    '交易成本': round(cost, 0),
                    '損益': round(net_pnl, 0)
                })
                position = 0
                unrealized_pnl = 0.0
                entry_price_s1 = 0.0
                entry_price_s2 = 0.0
                entry_date = None
                entry_zscore = 0.0

        # 淨值
        if position != 0:
            # 重算未實現(可能剛開倉)
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
            '帳戶淨值': equity,
            '動作': action
        })

    df_records = pd.DataFrame(records)
    df_records.set_index('日期', inplace=True)
    df_trades = pd.DataFrame(trade_log)

    # 統計
    close_trades = df_trades[df_trades['動作'].str.contains('平倉')] if not df_trades.empty else pd.DataFrame()
    stats = {}
    stats['insufficient_margin_error'] = insufficient_margin_error
    stats['base_s1'] = base_s1
    stats['base_s2'] = base_s2
    stats['base_margin'] = base_margin
    stats['hedge_ratio'] = hedge_ratio
    stats['contracts_s1'] = contracts_s1
    stats['contracts_s2'] = contracts_s2
    stats['multiplier'] = multiplier
    stats['total_margin_needed'] = base_margin
    stats['margin_s1'] = margin_s1
    stats['margin_s2'] = margin_s2
    stats['margin_utilization'] = (total_margin_needed_start / initial_capital * 100) if initial_capital > 0 else 0
    stats['p1_start'] = p1_start
    stats['p2_start'] = p2_start
    stats['start_date'] = valid_dates[0]
    stats['end_date'] = valid_dates[-1]
    stats['initial_capital'] = initial_capital
    stats['final_equity'] = df_records['帳戶淨值'].iloc[-1]
    stats['total_return'] = df_records['帳戶淨值'].iloc[-1] - initial_capital
    stats['total_return_pct'] = (df_records['帳戶淨值'].iloc[-1] / initial_capital - 1) * 100
    
    # MDD
    peak = df_records['帳戶淨值'].cummax()
    drawdown = peak - df_records['帳戶淨值']
    drawdown_pct = drawdown / peak * 100
    stats['max_drawdown'] = drawdown.max()
    stats['max_drawdown_pct'] = drawdown_pct.max()

    # 年化報酬率
    trading_days = len(df_records)
    years = trading_days / 252
    if years > 0 and stats['final_equity'] > 0:
        stats['annualized_return'] = ((stats['final_equity'] / initial_capital) ** (1 / years) - 1) * 100
    else:
        stats['annualized_return'] = 0.0

    if not close_trades.empty:
        win = close_trades[close_trades['損益'] > 0]
        lose = close_trades[close_trades['損益'] <= 0]
        stats['total_trades'] = len(close_trades)
        stats['win_trades'] = len(win)
        stats['lose_trades'] = len(lose)
        stats['win_rate'] = len(win) / len(close_trades) * 100
        stats['avg_win'] = win['損益'].mean() if len(win) > 0 else 0
        stats['avg_lose'] = lose['損益'].mean() if len(lose) > 0 else 0
        stats['profit_factor'] = abs(win['損益'].sum() / lose['損益'].sum()) if lose['損益'].sum() != 0 else float('inf')
    else:
        stats['total_trades'] = 0
        stats['win_trades'] = 0
        stats['lose_trades'] = 0
        stats['win_rate'] = 0
        stats['avg_win'] = 0
        stats['avg_lose'] = 0
        stats['profit_factor'] = 0

    stats['skipped_limits'] = skipped_limits
    stats['settlement_rolls'] = settlement_rolls
    stats['total_roll_cost'] = total_roll_cost

    # 持倉期間保證金
    margin_when_holding = df_records[df_records['持倉'] != 0]['凍結保證金']
    if not margin_when_holding.empty:
        stats['min_margin'] = margin_when_holding.min()
        stats['max_margin'] = margin_when_holding.max()
        stats['avg_margin'] = margin_when_holding.mean()
    else:
        stats['min_margin'] = 0
        stats['max_margin'] = 0
        stats['avg_margin'] = 0

    # 共整合 p-value
    try:
        _, pvalue, _ = coint(S1[common_idx], S2[common_idx])
        stats['coint_pvalue'] = pvalue
    except:
        stats['coint_pvalue'] = None

    return df_records, df_trades, stats


# ============================================================
# 主介面
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 台灣股票期貨 配對交易回測系統</h1>
        <p>Z-Score 均值回歸策略 ｜ 完整保證金模擬 ｜ 漲跌停過濾 ｜ 每月結算轉倉 ｜ 200+ 檔股期標的</p>
    </div>
    """, unsafe_allow_html=True)

    # 載入資料
    prices = load_price_data()
    if prices is None:
        st.error("找不到股價資料檔 (stock_prices.csv)。請先執行 `data_fetcher.py` 下載資料。")
        return

    tickers = sorted(prices.columns.tolist())
    name_map = get_ticker_names()

    # 建立選項列表 (代號 + 名稱)
    def fmt_ticker(t):
        code = t.replace('.TW', '').replace('.TWO', '')
        name = name_map.get(code, '')
        return f"{t} ({name})" if name else t

    ticker_options = tickers

    # ============================================================
    # 側邊欄: 參數設定
    # ============================================================
    with st.sidebar:
        st.markdown("## ⚙️ 回測參數設定")
        st.markdown("---")

        st.markdown("### 📌 選擇配對標的")
        col_a, col_b = st.columns(2)
        
        # 預設選擇 1513 和 6414
        default_s1 = tickers.index('1513.TW') if '1513.TW' in tickers else 0
        default_s2 = tickers.index('6414.TW') if '6414.TW' in tickers else 1

        sym1 = st.selectbox(
            "標的 A (Leg 1)",
            ticker_options,
            index=default_s1,
            format_func=fmt_ticker,
            key='sym1'
        )
        sym2 = st.selectbox(
            "標的 B (Leg 2)",
            ticker_options,
            index=default_s2,
            format_func=fmt_ticker,
            key='sym2'
        )

        if sym1 == sym2:
            st.warning("請選擇兩檔不同的標的！")
            return

        st.markdown("---")
        st.markdown("### 📅 回測期間")
        import datetime
        min_date = prices.index.min().date()
        latest_date = prices.index.max().date()
        today_date = datetime.date.today()
        
        # 預設起點設為 2020-01-02
        default_start = datetime.date(2020, 1, 2) if datetime.date(2020, 1, 2) >= min_date else min_date
        
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("開始日期", min_value=min_date, max_value=today_date, value=default_start)
        with col_end:
            # 預設終點設為今天
            end_date = st.date_input("結束日期", min_value=min_date, max_value=today_date, value=today_date)

        st.markdown("---")
        st.markdown("### 💰 保證金與資金管理")
        initial_capital = st.number_input(
            "初始總保證金 (TWD)",
            min_value=10000,
            max_value=100000000,
            value=1000000,
            step=50000,
            format="%d"
        )
        
        st.markdown("### 💰 資金滿載設定")
        size_mode = st.radio("資金佈局模式", 
                             options=['依保證金上限最大化（預設）', '最少配對口數 (僅基本單位)'],
                             index=0,
                             help="選擇是否要讓系統自動把資金運用到極限")
        
        margin_usage_pct = 1.0
        if size_mode == '依保證金上限最大化（預設）':
            margin_usage_pct = st.slider("最高保證金利用率 (%)", 0, 100, 100, 5, help="限制這筆資金最高只能被利用的比例") / 100.0

        st.markdown("---")
        st.markdown("### 📐 Z-Score 參數")
        z_entry = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", -3.0, 3.0, 2.0, 0.1)
        z_exit = st.slider("平倉閾值 (Z 回歸)", -3.0, 3.0, 0.0, 0.1)
        z_window = st.slider("滾動窗口 (天)", 20, 120, 60, 5)

        st.markdown("---")
        st.markdown("### 📋 股期規格")
        st.info(f"""
        - **1 口 = {SHARES_PER_CONTRACT:,} 股 (2 張)**
        - **保證金比率: {MARGIN_RATE*100:.1f}%**
        - **期交稅: {TRADING_FEE_RATE*100000:.0f}/100,000 (單邊)**
        - **手續費: {COMMISSION_PER_CONTRACT} 元/口 (單邊)**
        - **漲跌停: ±{LIMIT_PCT*100:.0f}% 不可交易**
        - **結算日: 每月第3個禮拜三**
        - **轉倉: 結算平倉→次日開倉**
        """)

        st.markdown("---")
        run_btn = st.button("🚀 執行回測", type="primary", use_container_width=True)

    # ============================================================
    # 主面板: 執行結果
    # ============================================================
    if run_btn:
        S1 = prices[sym1].dropna()
        S2 = prices[sym2].dropna()

        # 確保有共同日期以供模型訓練 (需要全局數據)
        common_idx = S1.index.intersection(S2.index)
        if len(common_idx) < 120:  # 至少需要兩倍滾動窗口的資料量
            st.error(f"這兩檔標的全局共同有效的交易日過少 ({len(common_idx)} 天)，無法建立計算模型！")
            return
            
        S1 = S1[common_idx]
        S2 = S2[common_idx]

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        with st.spinner("正在執行回測分析..."):
            name1 = name_map.get(sym1.replace('.TW', '').replace('.TWO', ''), '')
            name2 = name_map.get(sym2.replace('.TW', '').replace('.TWO', ''), '')
            df_records, df_trades, stats = run_backtest(
                S1, S2, sym1, sym2, name1, name2, initial_capital, 
                z_entry, z_exit, z_window, 
                size_mode=size_mode, margin_usage_pct=margin_usage_pct,
                run_start_date=start_str, run_end_date=end_str
            )

        if stats.get('insufficient_margin_error', False):
            if 'msg' in stats:
                st.error(f"❌ **回測執行攔截：**\n\n{stats['msg']}")
            else:
                st.error(f"❌ **保證金不足無法下單！**\n\n您設定的可用保證金 ({initial_capital * margin_usage_pct:,.0f} 元) 不足以下單即使是最少的 {stats.get('base_s1',0)}口/{stats.get('base_s2',0)}口 (需 {stats.get('base_margin',0):,.0f} 元)。請增加資金或選擇較低價標的。")
            return

        # --- 保證金計算摘要 ---
        st.markdown("## 📋 配對口數與保證金計算")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Hedge Ratio (β)</div>
                <div class="value">{stats['hedge_ratio']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">最小等張口數需求</div>
                <div class="value">{sym1}×{stats['base_s1']} : {sym2}×{stats['base_s2']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">最低一組基本保證金</div>
                <div class="value">{stats['base_margin']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            pct_used = stats['margin_utilization']
            color_cls = 'positive' if pct_used < 95 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">保證金利用率</div>
                <div class="value {color_cls}">{pct_used:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 詳細計算
        with st.expander("📐 詳細保證金計算", expanded=False):
            calc_col1, calc_col2 = st.columns(2)
            with calc_col1:
                st.markdown(f"""
                **{sym1}** (起始價: {stats['p1_start']:.2f})
                - 合約價值 = {stats['p1_start']:.2f} × {SHARES_PER_CONTRACT:,} = **{calc_contract_value(stats['p1_start']):,.0f}**
                - {stats['base_s1']} 口基本保證金 = {calc_contract_value(stats['p1_start']):,.0f} × {MARGIN_RATE*100:.1f}% × {stats['base_s1']} = **{stats['margin_s1']/max(1, stats['multiplier']):,.0f}**
                """)
            with calc_col2:
                st.markdown(f"""
                **{sym2}** (起始價: {stats['p2_start']:.2f})
                - 合約價值 = {stats['p2_start']:.2f} × {SHARES_PER_CONTRACT:,} = **{calc_contract_value(stats['p2_start']):,.0f}**
                - {stats['base_s2']} 口基本保證金 = {calc_contract_value(stats['p2_start']):,.0f} × {MARGIN_RATE*100:.1f}% × {stats['base_s2']} = **{stats['margin_s2']/max(1, stats['multiplier']):,.0f}**
                """)
            st.markdown(f"*(註：上述為回測期初起算時之一倍基本組合。實單會隨著帳戶總資金動態擴大倍數)*")

        if stats['base_margin'] > initial_capital:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ 初始資金 {initial_capital:,} 元不足以開出一組基本配對（需 {stats['base_margin']:,.0f} 元）！
                請提高初始保證金或選擇較低價的標的。
            </div>
            """, unsafe_allow_html=True)
        elif stats.get('multiplier', 0) < 1 and size_mode == '依保證金上限最大化（預設）':
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ 您設定的保證金利用率極限較低，分配到的起步資金不足以負荷一組配對（需 {stats['base_margin']:,.0f} 元）。
                回測在資金或股價達標前可能會保持空手。
            </div>
            """, unsafe_allow_html=True)

        # --- 績效指標 ---
        st.markdown("## 📊 回測績效摘要")

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        with m1:
            ret_cls = 'positive' if stats['total_return'] >= 0 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">累計報酬</div>
                <div class="value {ret_cls}">{stats['total_return']:+,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">報酬率</div>
                <div class="value {ret_cls}">{stats['total_return_pct']:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">年化報酬率</div>
                <div class="value {'positive' if stats['annualized_return'] >= 0 else 'negative'}">{stats['annualized_return']:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            wr_cls = 'positive' if stats['win_rate'] >= 50 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">勝率</div>
                <div class="value {wr_cls}">{stats['win_rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">交易次數</div>
                <div class="value">{stats['total_trades']}</div>
            </div>
            """, unsafe_allow_html=True)
        with m6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">最大回撤 MDD</div>
                <div class="value negative">{stats['max_drawdown_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 更多指標
        st.markdown("")
        more1, more2, more3, more4 = st.columns(4)
        with more1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">平均獲利</div>
                <div class="value positive">{stats['avg_win']:+,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with more2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">平均虧損</div>
                <div class="value negative">{stats['avg_lose']:+,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with more3:
            pf = stats['profit_factor']
            pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">獲利因子</div>
                <div class="value">{pf_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with more4:
            coint_str = f"{stats['coint_pvalue']:.6f}" if stats['coint_pvalue'] is not None else "N/A"
            coint_cls = 'positive' if stats['coint_pvalue'] is not None and stats['coint_pvalue'] < 0.05 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">共整合 P-value</div>
                <div class="value {coint_cls}">{coint_str}</div>
            </div>
            """, unsafe_allow_html=True)

        # 結算轉倉 & 漲跌停統計
        info_parts = []
        if stats['settlement_rolls'] > 0:
            info_parts.append(
                f"每月結算轉倉 <b>{stats['settlement_rolls']}</b> 次，"
                f"累計轉倉成本 <b>{stats['total_roll_cost']:,.0f}</b> 元"
            )
        if stats['skipped_limits'] > 0:
            info_parts.append(
                f"漲跌停跳過 <b>{stats['skipped_limits']}</b> 次"
            )
        if info_parts:
            st.markdown(f"""
            <div class="info-box">
                📌 {'　｜　'.join(info_parts)}
            </div>
            """, unsafe_allow_html=True)

        # --- 圖表 ---
        st.markdown("## 📈 回測圖表")

        # (1) 股價走勢
        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price.add_trace(
            go.Scatter(x=df_records.index, y=df_records[f'{sym1}價'],
                       name=sym1, line=dict(color='#2196F3', width=1.5)),
            secondary_y=False
        )
        fig_price.add_trace(
            go.Scatter(x=df_records.index, y=df_records[f'{sym2}價'],
                       name=sym2, line=dict(color='#FF9800', width=1.5)),
            secondary_y=True
        )
        fig_price.update_layout(
            title=f'{sym1} vs {sym2} 股價走勢',
            template='plotly_dark', height=350,
            margin=dict(l=60, r=60, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_price.update_yaxes(title_text=sym1, secondary_y=False)
        fig_price.update_yaxes(title_text=sym2, secondary_y=True)
        st.plotly_chart(fig_price, use_container_width=True)

        # (2) Z-Score + 訊號
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Z-Score'],
            name='Z-Score', line=dict(color='#BB86FC', width=1),
            fill='tozeroy', fillcolor='rgba(187,134,252,0.1)'
        ))
        fig_z.add_hline(y=z_entry, line_dash="dash", line_color="red",
                        annotation_text=f"+{z_entry}")
        fig_z.add_hline(y=-z_entry, line_dash="dash", line_color="green",
                        annotation_text=f"-{z_entry}")
        fig_z.add_hline(y=z_exit, line_dash="dot", line_color="gray",
                        annotation_text=f"Exit={z_exit}")

        if not df_trades.empty:
            opens = df_trades[df_trades['動作'].str.contains('開倉')]
            closes = df_trades[df_trades['動作'].str.contains('平倉')]
            if not opens.empty:
                fig_z.add_trace(go.Scatter(
                    x=pd.to_datetime(opens['日期']), y=opens['Z-Score'],
                    mode='markers', name='開倉',
                    marker=dict(symbol='circle', size=10, color='#00BCD4',
                                line=dict(width=1, color='white'))
                ))
            if not closes.empty:
                fig_z.add_trace(go.Scatter(
                    x=pd.to_datetime(closes['日期']), y=closes['Z-Score'],
                    mode='markers', name='平倉',
                    marker=dict(symbol='x', size=10, color='#FFD700',
                                line=dict(width=2, color='white'))
                ))

        fig_z.update_layout(
            title='Z-Score 與進出場訊號',
            template='plotly_dark', height=350,
            margin=dict(l=60, r=60, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_z, use_container_width=True)

        # (3) 損益追蹤
        fig_pnl = go.Figure()
        unrealized_colors = ['rgba(100,255,218,0.3)' if v >= 0 else 'rgba(255,107,107,0.3)'
                            for v in df_records['未實現損益']]
        fig_pnl.add_trace(go.Bar(
            x=df_records.index, y=df_records['未實現損益'],
            name='未實現損益', marker_color=unrealized_colors,
            opacity=0.5
        ))
        fig_pnl.add_trace(go.Scatter(
            x=df_records.index, y=df_records['累計已實現損益'],
            name='累計已實現損益', line=dict(color='#2196F3', width=2)
        ))
        fig_pnl.add_hline(y=0, line_color="gray", line_width=0.5)
        fig_pnl.update_layout(
            title='損益追蹤 (未實現 + 已實現)',
            template='plotly_dark', height=350,
            margin=dict(l=60, r=60, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        # (4) 帳戶淨值 + 保證金
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=df_records.index, y=df_records['帳戶淨值'],
            name='帳戶淨值', line=dict(color='#64FFDA', width=2),
            fill='tonexty' if False else None
        ))
        fig_eq.add_trace(go.Scatter(
            x=df_records.index, y=df_records['凍結保證金'],
            name='凍結保證金', fill='tozeroy',
            fillcolor='rgba(255,152,0,0.2)',
            line=dict(color='#FF9800', width=1)
        ))
        fig_eq.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                         annotation_text=f"初始資金 {initial_capital:,}")
        fig_eq.update_layout(
            title='帳戶淨值 & 凍結保證金',
            template='plotly_dark', height=350,
            margin=dict(l=60, r=60, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # --- 交易明細表 ---
        st.markdown("## 📝 交易明細")

        if not df_trades.empty:
            def style_trades(row):
                if '平倉' in str(row.get('動作', '')):
                    if row.get('損益', 0) > 0:
                        return ['background-color: rgba(100, 255, 218, 0.15)'] * len(row)
                    else:
                        return ['background-color: rgba(255, 107, 107, 0.15)'] * len(row)
                return [''] * len(row)

            styled = df_trades.style.apply(style_trades, axis=1).format({
                '交易成本': '{:,.0f}',
                '損益': '{:+,.0f}',
                'Z-Score': '{:.2f}'
            })
            st.dataframe(styled, use_container_width=True, height=400)
        else:
            st.info("回測期間無任何交易發生。")

        # --- 下載 ---
        st.markdown("## 💾 下載結果")
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_trades = df_trades.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載交易明細 CSV", csv_trades,
                             f"trade_log_{sym1}_{sym2}.csv", "text/csv")
        with dl2:
            csv_records = df_records.to_csv().encode('utf-8-sig')
            st.download_button("📥 下載每日紀錄 CSV", csv_records,
                             f"daily_records_{sym1}_{sym2}.csv", "text/csv")

    else:
        # 尚未執行回測，顯示使用說明
        st.markdown("""
        <div class="info-box">
            <b>🎯 使用方式</b><br>
            1. 從左側選單中選擇兩檔想配對的股期標的<br>
            2. 設定初始保證金金額<br>
            3. 調整 Z-Score 開倉/平倉閾值與滾動窗口<br>
            4. 點擊「🚀 執行回測」查看完整結果<br>
            <br>
            <b>📌 交易規則</b><br>
            • Z-Score > 閾值 → 做空 Spread（空標的B + 多標的A）<br>
            • Z-Score < -閾值 → 做多 Spread（多標的B + 空標的A）<br>
            • Z-Score 回歸平倉線 → 平倉<br>
            • 漲跌停日（±10%）自動跳過，不會產生無法成交的虛假訊號<br>
            • 每月第 3 個禮拜三結算日自動平倉，次日以新價格重新建倉（含轉倉成本）
        </div>
        """, unsafe_allow_html=True)

        # 顯示可用標的列表
        st.markdown("### 📋 可用標的清單")
        st.markdown(f"共 **{len(tickers)}** 檔標的（取自期交所股票期貨掛牌名單，已過濾有完整 5 年資料者）")

        ticker_display = []
        for t in tickers:
            code = t.replace('.TW', '').replace('.TWO', '')
            name = name_map.get(code, '')
            ticker_display.append({'代號': t, '簡稱': name, '市場': 'TWSE' if '.TW' in t else 'TPEx'})
        st.dataframe(pd.DataFrame(ticker_display), use_container_width=True, height=300)


if __name__ == '__main__':
    main()
