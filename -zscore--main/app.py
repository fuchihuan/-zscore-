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


@st.cache_data(ttl=3600)
def get_industry_data():
    """嘗試取得產業與主要業務資料"""
    industry_file = os.path.join(os.path.dirname(__file__), 'industry_data.csv')
    if os.path.exists(industry_file):
        try:
            return pd.read_csv(industry_file).set_index('Ticker')
        except:
            return None
    return None


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


def calculate_half_life(spread):
    """計算 Ornstein-Uhlenbeck (OU) 模型的收斂半衰期"""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    df = pd.DataFrame({'diff': spread_diff, 'lag': spread_lag}).dropna()
    if df.empty:
        return float('inf')
        
    X = sm.add_constant(df['lag'])
    Y = df['diff']
    
    try:
        model = sm.OLS(Y, X).fit()
        b = model.params['lag']
        
        # 判斷是否均值回歸 (b 必須小於 0)
        if b >= 0:
            return float('inf')
            
        half_life = -np.log(2) / b
        return half_life
    except:
        return float('inf')


def run_kalman_filter(Y, X, trans_cov):
    """
    簡單的一維量測、二維狀態 Kalman Filter 實作
    狀態變數：[alpha, beta]
    量測方程式：Y_t = alpha_t + beta_t * X_t + v_t
    狀態方程式：[alpha_t, beta_t] = [alpha_{t-1}, beta_{t-1}] + w_t
    """
    n = len(Y)
    state_mean = np.zeros((n, 2))
    state_cov = np.zeros((n, 2, 2))
    
    # 初始化
    state_mean[0] = [0, 1] # 假設初始 alpha=0, beta=1
    state_cov[0] = np.eye(2)
    
    # 轉移矩陣 (Identity)
    F = np.eye(2)
    # 狀態雜訊共變異數 (Q)
    Q = np.eye(2) * trans_cov
    # 觀測雜訊變異數 (R)
    R = 1e-3
    
    for t in range(1, n):
        # 1. 預測 (Predict)
        pred_mean = state_mean[t-1]
        pred_cov = state_cov[t-1] + Q
        
        # 2. 更新 (Update)
        H = np.array([[1, X.iloc[t]]]) # 觀測矩陣
        residual = Y.iloc[t] - (H @ pred_mean)[0]
        S = H @ pred_cov @ H.T + R
        K = pred_cov @ H.T @ np.linalg.inv(S)
        
        state_mean[t] = pred_mean + (K * residual).flatten()
        state_cov[t] = (np.eye(2) - K @ H) @ pred_cov
        
    return pd.DataFrame(state_mean, index=Y.index, columns=['alpha', 'beta'])


def run_backtest(S1, S2, sym1, sym2, name1, name2, initial_capital, 
                 z_entry, z_exit, z_window, 
                 size_mode='依保證金上限最大化（預設）', margin_usage_pct=1.0,
                 run_start_date=None, run_end_date=None, advanced_params=None):
    """
    執行完整的配對交易保證金回測
    """
    if advanced_params is None:
        advanced_params = {}

    # 對齊
    common_idx = S1.index.intersection(S2.index)
    S1 = S1[common_idx].copy()
    S2 = S2[common_idx].copy()

    # 偵測漲跌停
    limit_s1 = detect_limit_days(S1)
    limit_s2 = detect_limit_days(S2)
    any_limit = limit_s1 | limit_s2

    # 價格轉換
    use_log = advanced_params.get('use_log_price', False)
    if use_log:
        M1 = np.log(S1)
        M2 = np.log(S2)
    else:
        M1 = S1.copy()
        M2 = S2.copy()

    # 計算動態或靜態避險比率
    use_kalman = advanced_params.get('use_kalman', False)
    if use_kalman:
        trans_cov = advanced_params.get('kalman_trans_cov', 1e-5)
        kalman_states = run_kalman_filter(M2, M1, trans_cov)
        hedge_ratio_series = kalman_states['beta']
        spread = M2 - (kalman_states['alpha'] + hedge_ratio_series * M1)
        # 用於 stats 的靜態紀錄 (取最後一天)
        static_hedge_ratio = hedge_ratio_series.iloc[-1]
    else:
        X = sm.add_constant(M1)
        model = sm.OLS(M2, X).fit()
        static_hedge_ratio = model.params[sym1]
        hedge_ratio_series = pd.Series(static_hedge_ratio, index=M1.index)
        spread = M2 - static_hedge_ratio * M1

    # Spread & Z-Score
    spread_mean = spread.rolling(window=z_window).mean()
    use_ewma = advanced_params.get('use_ewma_z', False)
    if use_ewma:
        span = advanced_params.get('ewma_span', 20)
        spread_std = spread.ewm(span=span).std()
    else:
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

    # 預先計算進階濾波 Mask
    ou_pass_mask = pd.Series(True, index=zscore.index)
    beta_trans_mask = pd.Series(False, index=zscore.index)
    roll_hl_series = pd.Series(np.inf, index=zscore.index)
    roll_r2_series = pd.Series(0.0, index=zscore.index)
    
    _spread = spread.loc[zscore.index]
    _hr = hedge_ratio_series.loc[zscore.index]

    if advanced_params.get('use_ou_filter', False):
        ou_window = advanced_params.get('ou_window', 60)
        ou_min_r2 = advanced_params.get('ou_min_r2', 0.2)
        ou_min_hl = advanced_params.get('ou_min_hl', 1.0)
        ou_max_hl = advanced_params.get('ou_max_hl', 30.0)
        
        diff_spread = _spread.diff()
        lag_spread = _spread.shift(1)
        
        roll_cov = lag_spread.rolling(ou_window).cov(diff_spread)
        roll_var = lag_spread.rolling(ou_window).var()
        roll_b = roll_cov / roll_var
        
        roll_corr = lag_spread.rolling(ou_window).corr(diff_spread)
        roll_r2 = roll_corr ** 2
        roll_r2_series = roll_r2
        
        valid_b = roll_b < 0
        roll_hl_series[valid_b] = -np.log(2) / roll_b[valid_b]
        
        ou_pass_mask = valid_b & (roll_r2 >= ou_min_r2) & (roll_hl_series >= ou_min_hl) & (roll_hl_series <= ou_max_hl)
        
    if advanced_params.get('use_beta_transition', False):
        beta_short = advanced_params.get('beta_short', 20)
        beta_mid = advanced_params.get('beta_mid', 100)
        
        slope_short = _hr.diff(beta_short) / beta_short
        slope_mid = _hr.diff(beta_mid) / beta_mid
        
        same_dir = (slope_short * slope_mid) > 0
        amp = slope_short.abs() > (slope_mid.abs() * 2)
        beta_trans_mask = same_dir & amp

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
    current_scale_level = 0    # 當前建倉級別 (0 表示空手)

    for i, date in enumerate(dates_list):
        z = zscore[date]
        p1 = S1[date]
        p2 = S2[date]
        is_limit = any_limit.get(date, False)
        is_settlement = date.date() in settlement_dates
        ou_pass = ou_pass_mask[date]
        beta_trans = beta_trans_mask[date]

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
        def get_trade_size(mode=None, custom_pairs=1):
            if mode == "固定組數":
                c1, c2 = find_min_contracts(p1, p2)
                c1 *= custom_pairs
                c2 *= custom_pairs
            elif mode == "加到滿倉" or size_mode == '依保證金上限最大化（預設）':
                # 此處的 cash 已扣除之前的交易成本，所以直接用 cash 算剩餘最大可建倉量
                # 但如果是初始進場，cash 就是全部可用資金
                available = cash * margin_usage_pct if position == 0 else cash
                c1, c2, _ = calc_max_contracts(p1, p2, available)
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

        # (A) 結算日後次日重新開倉 (轉倉)
        # =====================================================
        if pending_reopen is not None and position == 0:
            dyn_c1, dyn_c2, dyn_req = get_trade_size(mode=None)
            
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
                # 若轉倉成功，設定為已加碼(若原先就是分批建倉模式，避免轉倉後重複加碼)
                is_scaled_in = True
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
                
                # --- 智能轉倉防呆 ---
                use_smart_roll = advanced_params.get('use_smart_roll', False)
                smart_roll_z = advanced_params.get('smart_roll_z', 1.0)
                if use_smart_roll and net_pnl > 0 and abs(z) <= smart_roll_z:
                    pending_reopen = None
                    trade_log[-1]['動作'] += ' [防呆不轉倉]'
                    action += " (智能防呆：有獲利且已收斂，次日不再建倉)"
            else:
                pending_reopen = None  # Z-Score 已回歸，不需轉倉

            position = 0
            unrealized_pnl = 0.0
            entry_price_s1 = 0.0
            entry_price_s2 = 0.0
            entry_date = None
            entry_zscore = 0.0
            current_scale_level = 0

        # =====================================================
        # (C) 正常交易邏輯 (含 N 段式分批建倉)
        # =====================================================
        use_scale_in = advanced_params.get('use_scale_in', False)
        scale_in_levels = advanced_params.get('scale_in_levels', [])

        def execute_entry(dir_sign, dyn_c1, dyn_c2, dyn_req, act_name):
            nonlocal position, entry_price_s1, entry_price_s2, entry_date, entry_zscore
            nonlocal cash, realized_pnl_total, margin_required_today, contracts_s1, contracts_s2, action
            
            contracts_s1, contracts_s2 = dyn_c1, dyn_c2
            margin_required_today = dyn_req
            position = dir_sign
            entry_price_s1 = p1
            entry_price_s2 = p2
            entry_date = date
            entry_zscore = z
            cost = calc_trading_cost(p1, contracts_s1) + calc_trading_cost(p2, contracts_s2)
            cash -= cost
            realized_pnl_total -= cost
            desc = make_desc(dir_sign)
            action = f'{act_name}: {desc}'
            trade_log.append({
                '日期': date.strftime('%Y-%m-%d'), '動作': f'{act_name}({"做多" if dir_sign==1 else "做空"}配對)',
                '部位描述': desc,
                f'{sym1}({name1})多空': '做空' if dir_sign == 1 else '做多',
                f'{sym2}({name2})多空': '做多' if dir_sign == 1 else '做空',
                f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                f'{sym1}口數': contracts_s1, f'{sym2}口數': contracts_s2,
                '保證金佔用': round(margin_required_today, 0),
                'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0
            })

        # --- N段式進場與加碼判定 ---
        if use_scale_in and len(scale_in_levels) > 0 and pending_reopen is None:
            # 尋找當前 Z 分數越過了哪幾級
            triggered_level = 0
            for i, level in enumerate(scale_in_levels):
                if abs(z) >= level['z']:
                    triggered_level = i + 1
                else:
                    break

            # 如果越過的級別大於當前已建倉級別，則需要進場或加碼
            if triggered_level > current_scale_level:
                # 確定方向 (如果尚未持倉則依照當前 z 決定)
                dir_sign = position
                if dir_sign == 0:
                    dir_sign = -1 if z > 0 else 1
                
                # 計算要增加的口數
                total_pairs_to_add = 0
                mode = "固定組數"
                for i in range(current_scale_level, triggered_level):
                    lvl_pairs = scale_in_levels[i]['pairs']
                    if lvl_pairs == -1:
                        mode = "加到滿倉"
                        break
                    else:
                        total_pairs_to_add += lvl_pairs

                dyn_c1, dyn_c2, dyn_req = get_trade_size(mode, total_pairs_to_add)
                
                if is_limit:
                    action = '漲跌停，跳過建倉/加碼'
                    skipped_limits += 1
                elif position == 0 and not ou_pass:
                    action = 'OU 結構未達標，過濾不開倉'
                elif position == 0 and beta_trans:
                    action = 'Beta 結構轉換風險，過濾不開倉'
                elif dyn_c1 > 0 and cash >= dyn_req and (position != 0 or not insufficient_margin_error):
                    if position == 0:
                        # 首次開倉
                        act_name = f'第1~{triggered_level}批開倉' if triggered_level > 1 else '第1批開倉'
                        execute_entry(dir_sign, dyn_c1, dyn_c2, dyn_req, act_name)
                    else:
                        # 加碼
                        new_c1 = contracts_s1 + dyn_c1
                        new_c2 = contracts_s2 + dyn_c2
                        entry_price_s1 = ((entry_price_s1 * contracts_s1) + (p1 * dyn_c1)) / new_c1
                        entry_price_s2 = ((entry_price_s2 * contracts_s2) + (p2 * dyn_c2)) / new_c2
                        contracts_s1 = new_c1
                        contracts_s2 = new_c2
                        margin_required_today += dyn_req
                        
                        cost = calc_trading_cost(p1, dyn_c1) + calc_trading_cost(p2, dyn_c2)
                        cash -= cost
                        realized_pnl_total -= cost
                        
                        desc = f'加碼新增 {dyn_c1} / {dyn_c2} 口'
                        act_name = f'第{current_scale_level+1}~{triggered_level}批加碼' if triggered_level > current_scale_level + 1 else f'第{triggered_level}批加碼'
                        action = f'{act_name}: {desc}'
                        trade_log.append({
                            '日期': date.strftime('%Y-%m-%d'), '動作': act_name,
                            '部位描述': desc,
                            f'{sym1}({name1})多空': '做空' if position == 1 else '做多',
                            f'{sym2}({name2})多空': '做多' if position == 1 else '做空',
                            f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                            f'{sym1}口數': dyn_c1, f'{sym2}口數': dyn_c2,
                            '保證金佔用': round(margin_required_today, 0),
                            'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0
                        })

                    current_scale_level = triggered_level
                else:
                    action = '建倉/加碼失敗: 保證金不足'
        
        # --- 原單一進場邏輯 (若未啟用分批建倉) ---
        elif not use_scale_in and position == 0 and pending_reopen is None:
            if z > z_entry or z < -z_entry:
                dir_sign = -1 if z > z_entry else 1
                dyn_c1, dyn_c2, dyn_req = get_trade_size(None, 1)
                if is_limit:
                    action = '漲跌停，跳過開倉'
                    skipped_limits += 1
                elif not ou_pass:
                    action = 'OU 結構未達標，過濾不開倉'
                elif beta_trans:
                    action = 'Beta 結構轉換風險，過濾不開倉'
                elif dyn_c1 > 0 and cash >= dyn_req and not insufficient_margin_error:
                    execute_entry(dir_sign, dyn_c1, dyn_c2, dyn_req, '開倉')
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
                current_scale_level = 0

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
            '動作': action,
            'Beta': _hr[date],
            'OU_HalfLife': roll_hl_series[date] if advanced_params.get('use_ou_filter', False) else np.nan,
            'OU_R2': roll_r2_series[date] if advanced_params.get('use_ou_filter', False) else np.nan
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
    stats['hedge_ratio'] = static_hedge_ratio
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
    
    # 計算收斂半衰期
    stats['half_life'] = calculate_half_life(spread)
    
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
            margin_usage_pct = st.slider("最高保證金利用率 (%)", 0, 100, 30, 5, help="限制這筆資金最高只能被利用的比例") / 100.0

        st.markdown("---")
        st.markdown("### 📐 Z-Score 參數")
        z_entry = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", -3.0, 3.0, 2.0, 0.1)
        z_exit = st.slider("平倉閾值 (Z 回歸)", -3.0, 3.0, 0.0, 0.1)
        z_window = st.slider("滾動窗口 (天)", 20, 120, 60, 5)

        st.markdown("---")
        st.markdown("### 🔬 進階濾波與統計模型參數")
        use_log_price = st.toggle("啟用對數價格 (Log Price)", value=False, help="將兩檔股價取自然對數(ln)。\n👉 影響：會將資產間的比例關係轉為加法關係，使價差(Spread)分佈更接近常態，增加 Z-score 的穩定度。")
        
        use_kalman = st.toggle("使用 Kalman Filter 動態估計 Beta", value=False, help="使用狀態空間模型逐日動態更新避險比率(Beta)。\n👉 影響：相比於傳統固定 OLS，能無延遲地平滑追蹤市場結構改變，避免視窗大小選擇錯誤造成的延遲。")
        kalman_trans_cov = 1e-5
        if use_kalman:
            kalman_trans_cov = st.number_input("Kalman 狀態轉移共變異數 (漂移速度)", value=1e-5, format="%e", step=1e-6, help="👉 影響：數值越大，代表允許 Beta 的漂移速度越快（反應越靈敏但也容易有雜訊）；數值越小，Beta 越平滑。")
            
        use_ewma_z = st.toggle("使用 EWMA Variance 計算 Z-score", value=False, help="以指數加權移動平均(EWMA)來計算近期波動率。\n👉 影響：能賦予近期波動更大的權重，當市場突發劇烈震盪時，Z-score 的分母會立刻放大而使數值縮小，避免在假突破時過早進場。")
        ewma_span = 20
        if use_ewma_z:
            ewma_span = st.slider("EWMA Span (天)", 5, 120, 20, 5, help="👉 影響：天數越短，對近期波動的反應越劇烈。")
            
        use_ou_filter = st.toggle("啟用 OU Regime Filter", value=False, help="強制要求配對關係在統計上必須具有『均值回歸』特性才允許開倉。\n👉 影響：會大幅過濾掉交易次數，但能避開那些陷入單邊發散趨勢的標的，提升勝率。")
        ou_window = 60
        ou_min_r2 = 0.2
        ou_min_hl = 1.0
        ou_max_hl = 30.0
        if use_ou_filter:
            ou_window = st.slider("OU Rolling Window", 20, 120, 60, 5, help="計算半衰期時的回看天數")
            ou_min_r2 = st.slider("最小 R² 門檻", 0.0, 1.0, 0.2, 0.05, help="👉 影響：要求 OU 模型對價差必須有基本的解釋力，R² 越高代表回歸特徵越明顯。")
            ou_hl_range = st.slider("合理 Half-life 區間 (天)", 1.0, 100.0, (1.0, 30.0), 1.0, help="👉 影響：太短(<1)可能是雜訊；太長(>30)代表收斂太慢資金容易卡住。")
            ou_min_hl, ou_max_hl = ou_hl_range
            
        use_beta_transition = st.toggle("啟用 Beta Transition Filter", value=False, help="監控動態 Beta 的短期與中期斜率。\n👉 影響：若斜率同向且短期急遽放大，代表兩檔股票的基本結構可能正在發生永久性破裂，系統會強制關閉新倉權限，攔截極端回撤風險。")
        beta_short = 20
        beta_mid = 100
        if use_beta_transition:
            beta_short = st.slider("Beta 短期斜率視窗 (天)", 5, 60, 20, 5)
            beta_mid = st.slider("Beta 中期斜率視窗 (天)", 20, 200, 100, 10)

        st.markdown("---")
        st.markdown("### 🛡️ 智能轉倉防呆機制")
        use_smart_roll = st.toggle("啟用轉倉防呆", value=False, help="若結算時帳上有獲利，且 Z-score 已經收斂至設定門檻內，則結算平倉後次日不再自動開新倉（等同提早獲利了結，避免微薄利潤被轉倉成本吃掉）。")
        smart_roll_z = 1.0
        if use_smart_roll:
            smart_roll_z = st.slider("收斂過濾門檻 (Z-score)", 0.1, 2.9, 1.0, 0.1)

        st.markdown("---")
        st.markdown("### 📈 分批建倉與加碼機制")
        use_scale_in = st.toggle("啟用分批建倉", value=False, help="取代單一進場點。達到初始門檻時先建立部分口數，達到加碼門檻時再建滿剩餘口數。")
        scale_in_levels = []
        if use_scale_in:
            st.info("⚠️ 啟用此功能後，將優先採用以下進場門檻，忽略上方的『Z-Score 開倉閾值』。")
            num_levels = st.number_input("總共分幾批建倉?", value=2, min_value=1, max_value=10, step=1)
            
            for i in range(num_levels):
                st.markdown(f"**第 {i+1} 批設定**")
                col1, col2 = st.columns(2)
                with col1:
                    z_val = st.number_input(f"Z 分數門檻", value=1.5 + i*1.0, step=0.1, key=f"z_{i}")
                with col2:
                    if i == num_levels - 1:
                        mode = st.selectbox(f"加碼模式", ["增加固定組數", "加到滿倉"], key=f"mode_{i}")
                        if mode == "加到滿倉":
                            pairs_val = -1
                        else:
                            pairs_val = st.number_input(f"口數 (幾組配對)", value=1, min_value=1, step=1, key=f"pairs_{i}")
                    else:
                        pairs_val = st.number_input(f"口數 (幾組配對)", value=1, min_value=1, step=1, key=f"pairs_{i}")
                scale_in_levels.append({'z': z_val, 'pairs': pairs_val})
            
            # 確保按照 Z-Score 大小排序 (防呆)
            scale_in_levels.sort(key=lambda x: x['z'])

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
            
            # 建立 advanced_params 字典
            advanced_params = {
                'use_log_price': use_log_price,
                'use_kalman': use_kalman,
                'kalman_trans_cov': kalman_trans_cov,
                'use_ewma_z': use_ewma_z,
                'ewma_span': ewma_span,
                'use_ou_filter': use_ou_filter,
                'ou_window': ou_window,
                'ou_min_r2': ou_min_r2,
                'ou_min_hl': ou_min_hl,
                'ou_max_hl': ou_max_hl,
                'use_beta_transition': use_beta_transition,
                'beta_short': beta_short,
                'beta_mid': beta_mid,
                'use_smart_roll': use_smart_roll,
                'smart_roll_z': smart_roll_z,
                'use_scale_in': use_scale_in,
                'scale_in_levels': scale_in_levels
            }
            
            df_records, df_trades, stats = run_backtest(
                S1, S2, sym1, sym2, name1, name2, initial_capital, 
                z_entry, z_exit, z_window, 
                size_mode=size_mode, margin_usage_pct=margin_usage_pct,
                run_start_date=start_str, run_end_date=end_str,
                advanced_params=advanced_params
            )

        # --- 產業資訊摘要 ---
        industry_df = get_industry_data()
        if industry_df is not None:
            ind1 = industry_df.loc[sym1] if sym1 in industry_df.index else pd.Series({'Industry': '未知', 'Business': '未知'})
            ind2 = industry_df.loc[sym2] if sym2 in industry_df.index else pd.Series({'Industry': '未知', 'Business': '未知'})
            
            st.markdown("## 🏢 標的產業與營收來源")
            ind_col1, ind_col2 = st.columns(2)
            with ind_col1:
                st.info(f"**{sym1} ({name1})**\n\n**產業類別**: {ind1['Industry']}\n\n**主要業務**: {ind1['Business']}")
            with ind_col2:
                st.info(f"**{sym2} ({name2})**\n\n**產業類別**: {ind2['Industry']}\n\n**主要業務**: {ind2['Business']}")

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
        more1, more2, more3, more4, more5, more6 = st.columns(6)
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
            # 加上夏普比率
            sharpe = 0.0
            if len(df_records) > 1:
                daily_returns = df_records['帳戶淨值'].pct_change().dropna()
                if len(daily_returns) > 1 and daily_returns.std() != 0:
                    import numpy as np
                    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                    
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">夏普比率</div>
                <div class="value {'positive' if sharpe >= 1.0 else 'negative'}">{sharpe:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with more5:
            coint_str = f"{stats['coint_pvalue']:.6f}" if stats['coint_pvalue'] is not None else "N/A"
            coint_cls = 'positive' if stats['coint_pvalue'] is not None and stats['coint_pvalue'] < 0.05 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">共整合 P-value</div>
                <div class="value {coint_cls}">{coint_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with more6:
            hl = stats['half_life']
            hl_str = f"{hl:.1f} 天" if hl != float('inf') else "發散 (未收斂)"
            hl_cls = 'positive' if hl < 30 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">平均收斂半衰期</div>
                <div class="value {hl_cls}">{hl_str}</div>
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

        # (5) 進階濾波狀態追蹤 (Beta, OU)
        st.markdown("### 🔬 進階濾波狀態追蹤")
        fig_adv = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, 
                                subplot_titles=("動態 Beta 走勢 (避險比率)", "OU 模型狀態 (Half-life & R²)"))
                                
        # Beta
        fig_adv.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Beta'],
            name='Beta', line=dict(color='#FF5722', width=2)
        ), row=1, col=1)
        
        # OU
        if 'OU_HalfLife' in df_records.columns and not df_records['OU_HalfLife'].isna().all():
            # 將 np.inf 或過大數值限制在圖表可視範圍
            hl_plot = df_records['OU_HalfLife'].copy()
            hl_plot[hl_plot > 100] = 100
            
            fig_adv.add_trace(go.Scatter(
                x=df_records.index, y=hl_plot,
                name='Half-life (天)', line=dict(color='#03A9F4', width=1.5)
            ), row=2, col=1)
            
            fig_adv.add_trace(go.Scatter(
                x=df_records.index, y=df_records['OU_R2'],
                name='R²', line=dict(color='#8BC34A', width=1.5, dash='dot'),
                yaxis="y3" # 需要在 layout 設定 secondary y 供子圖使用
            ), row=2, col=1)
            
        fig_adv.update_layout(
            template='plotly_dark', height=500,
            margin=dict(l=60, r=60, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Fix secondary Y for R2 in subplot 2
        fig_adv.update_layout(yaxis3=dict(overlaying='y2', side='right', range=[0, 1], title='R²'))
        st.plotly_chart(fig_adv, use_container_width=True)

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
