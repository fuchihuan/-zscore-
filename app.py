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
    """嘗試取得代號與公司名對照表，並附加產業別"""
    mapping = {}
    if os.path.exists(TAIFEX_FILE):
        try:
            taifex_df = pd.read_csv(TAIFEX_FILE, encoding='utf-8-sig')
            for _, row in taifex_df.iterrows():
                code = str(row.iloc[2]).strip()
                name = str(row.iloc[3]).strip()
                if code != 'nan' and name != 'nan' and code:
                    mapping[code] = name
        except:
            pass
            
    # 附加產業別
    industry_df = get_industry_data()
    if industry_df is not None:
        try:
            for tk, row in industry_df.iterrows():
                ind = row.get('Industry')
                if pd.isna(ind) or not str(ind).strip():
                    continue
                code = str(tk).replace('.TWO', '').replace('.TW', '')
                if code in mapping:
                    mapping[code] = f"{mapping[code]} | {ind}"
                else:
                    mapping[code] = str(ind)
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



def run_backtest(S1, S2, sym1, sym2, name1, name2, initial_capital,
                   selected_models, model_params,
                   size_mode='依保證金上限最大化（預設）', margin_usage_pct=1.0,
                 run_start_date=None, run_end_date=None, advanced_params=None):
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

    from advanced_models import get_zscore_signals, get_ou_signals, get_garch_signals, get_kalman_filter_signals, get_copula_signals, get_jump_diffusion_signals, get_sdde_signals, get_np_cusum_signals, get_gsadf_signals, get_dcc_garch_vecm_signals, get_markov_regime_signals

    trade_mode = advanced_params.get('trade_mode', '收斂 (均值回歸)')
    ensemble_logic = advanced_params.get('ensemble_logic', '單一模型')
    
    # If selected_models is passed as a single string (from legacy or 收斂 mode), wrap it in a list
    if isinstance(selected_models, str):
        selected_models = [selected_models]
        
    all_signals = []
    for m_type in selected_models:
        m_params = model_params.get(m_type, {})
        if m_type == 'Z-Score (標準)':
            sdf = get_zscore_signals(S1, S2, m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0), m_params.get('z_window', 20))
        elif m_type == 'OU 過程 (動態邊界)':
            sdf = get_ou_signals(S1, S2, m_params.get('z_window', 20))
        elif m_type == '共整合 + GARCH':
            sdf = get_garch_signals(S1, S2, m_params.get('z_window', 20), m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0))
        elif m_type == '卡爾曼濾波 (動態對沖比例)':
            sdf = get_kalman_filter_signals(S1, S2, m_params.get('z_window', 20), m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0))
        elif m_type == 'Copula (CMPI 機率)':
            sdf = get_copula_signals(S1, S2, m_params.get('z_window', 20), m_params.get('prob_threshold', 0.95))
        elif m_type == 'Merton 跳躍擴散模型 (過濾結構破裂)':
            sdf = get_jump_diffusion_signals(S1, S2, m_params.get('z_window', 20), m_params.get('jump_threshold', 3.0), m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0))
        elif m_type == 'SDDE 隨機延遲方程式 (過濾動能慣性)':
            sdf = get_sdde_signals(S1, S2, m_params.get('z_window', 20), m_params.get('delay_tau', 5), m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0))
        elif m_type == '非參數 CUSUM (多變量幾何破裂)':
            sdf = get_np_cusum_signals(S1, S2, m_params.get('cusum_window', 20), m_params.get('k_shift', 1.0), m_params.get('tau_threshold', 5.0), 0.0)
        elif m_type == 'GSADF (爆炸性泡沫檢定)':
            sdf = get_gsadf_signals(S1, S2, m_params.get('gsadf_window', 30), m_params.get('adf_threshold', 1.5), 0.0)
        elif m_type == 'MRS (馬爾可夫區制轉換)':
            sdf = get_markov_regime_signals(S1, S2, m_params.get('mrs_window', 120), m_params.get('prob_threshold', 0.8), 0.0)
        elif m_type == 'DCC-GARCH-VECM (特異性漂移爆發)':
            sdf = get_dcc_garch_vecm_signals(S1, S2, m_params.get('garch_window', 20), m_params.get('t_threshold', 3.0), 0.0)
        else:
            sdf = get_zscore_signals(S1, S2, 2.0, 0.0, 20)
        all_signals.append(sdf)
        
    if not all_signals:
        return pd.DataFrame(), pd.DataFrame(), {'insufficient_margin_error': True, 'msg': '沒有選擇任何模型！'}
        
    signal_df = all_signals[0].copy()
    
    if len(all_signals) > 1:
        signal_df['Upper_Bound'] = 1.0
        signal_df['Lower_Bound'] = -1.0
        signal_df['Exit_Upper'] = 0.0
        signal_df['Exit_Lower'] = 0.0
        
        artificial_indicator = pd.Series(0.0, index=signal_df.index)
        
        current_state = 0
        for date in signal_df.index:
            entry_up = []
            entry_dn = []
            exit_up = []
            exit_dn = []
            
            for sdf in all_signals:
                if date not in sdf.index:
                    continue
                ind = sdf.at[date, 'Indicator']
                ub = sdf.at[date, 'Upper_Bound']
                lb = sdf.at[date, 'Lower_Bound']
                e_ub = sdf.at[date, 'Exit_Upper']
                e_lb = sdf.at[date, 'Exit_Lower']
                
                entry_up.append(ind > ub)
                entry_dn.append(ind < lb)
                exit_up.append(ind <= e_ub)
                exit_dn.append(ind >= e_lb)
                
            if current_state == 0:
                if 'AND' in ensemble_logic:
                    if all(entry_up) and len(entry_up) > 0:
                        current_state = 1
                    elif all(entry_dn) and len(entry_dn) > 0:
                        current_state = -1
                else: # OR
                    if any(entry_up):
                        current_state = 1
                    elif any(entry_dn):
                        current_state = -1
            else:
                if current_state == 1:
                    # 無論 AND 或 OR，只要有任何一個模型說該平倉了 (回到均值)，就先落袋為安
                    if any(exit_up):
                        current_state = 0
                elif current_state == -1:
                    if any(exit_dn):
                        current_state = 0
                        
            if current_state == 1:
                artificial_indicator.at[date] = 2.0
            elif current_state == -1:
                artificial_indicator.at[date] = -2.0
            else:
                artificial_indicator.at[date] = 0.0
                
        signal_df['Indicator'] = artificial_indicator
        
    if signal_df is None or signal_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {'insufficient_margin_error': True, 'msg': '資料不足以計算模型！'}

    # 預先計算進階濾波 Mask
    zscore = signal_df['Indicator']
    spread = signal_df['Spread']
    import numpy as np
    hedge_ratio_series = pd.Series(signal_df['Hedge_Ratio'], index=zscore.index) if isinstance(signal_df.get('Hedge_Ratio'), (int, float, np.float64, np.float32)) else signal_df.get('Hedge_Ratio', pd.Series(1.0, index=zscore.index))
    
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


    # 時間切片
    if run_start_date and run_end_date:
        signal_df = signal_df.loc[run_start_date:run_end_date]
        zscore = zscore.loc[run_start_date:run_end_date]
        ou_pass_mask = ou_pass_mask.loc[run_start_date:run_end_date]
        beta_trans_mask = beta_trans_mask.loc[run_start_date:run_end_date]
        roll_hl_series = roll_hl_series.loc[run_start_date:run_end_date]
        roll_r2_series = roll_r2_series.loc[run_start_date:run_end_date]
    
    if signal_df.empty:
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

    insufficient_margin_error = False
    if base_margin > initial_capital:
        insufficient_margin_error = True
        contracts_s1, contracts_s2, multiplier = 0, 0, 0

    start_year = zscore.index[0].year
    end_year = zscore.index[-1].year
    settlement_dates = get_third_wednesdays(start_year, end_year)

    valid_dates = zscore.index
    dates_list = list(valid_dates)
    
    use_grid = advanced_params.get('use_grid', False)
    grid_levels = advanced_params.get('grid_levels', [])
    
    if not use_grid:
        grid_levels = [{
            'id': 1, 'pairs': 1
        }]

    grid_states = []
    for g in grid_levels:
        grid_states.append({
            'params': g,
            'is_active': False,
            'is_stopped_out': False,
            'direction': 0, # 1 for Long, -1 for Short
            'c1': 0,
            'c2': 0,
            'entry_p1': 0.0,
            'entry_p2': 0.0,
            'entry_zscore': 0.0,
            'entry_date': None,
            'pending_reopen_dir': 0
        })

    cash = initial_capital
    realized_pnl_total = 0.0
    records = []
    trade_log = []
    skipped_limits = 0
    settlement_rolls = 0
    total_roll_cost = 0.0
    
    def get_trade_size(p1, p2, pairs, available_cash):
        if pairs == -1 or size_mode == '依保證金上限最大化（預設）':
            c1, c2, _ = calc_max_contracts(p1, p2, available_cash)
        else:
            c1, c2 = find_min_contracts(p1, p2)
            c1 *= pairs
            c2 *= pairs
        req = calc_margin_per_contract(p1) * c1 + calc_margin_per_contract(p2) * c2
        return c1, c2, req

    def make_desc(dir_sign, c1, c2):
        if dir_sign == 1:
            return f'做空{sym1}({name1}) {c1}口 / 做多{sym2}({name2}) {c2}口'
        else:
            return f'做多{sym1}({name1}) {c1}口 / 做空{sym2}({name2}) {c2}口'

    for i, date in enumerate(dates_list):
        z = zscore[date]
        p1 = S1[date]
        p2 = S2[date]
        is_limit = any_limit.get(date, False)
        is_settlement = date.date() in settlement_dates
        ou_pass = ou_pass_mask[date]
        beta_trans = beta_trans_mask[date]
        
        # 讀取當天動態邊界
        dyn_upper = signal_df['Upper_Bound'].loc[date]
        dyn_lower = signal_df['Lower_Bound'].loc[date]
        dyn_exit_upper = signal_df['Exit_Upper'].loc[date]
        dyn_exit_lower = signal_df['Exit_Lower'].loc[date]

        unrealized_pnl = 0.0
        margin_required_today = 0.0
        
        total_position = 0
        total_c1 = 0
        total_c2 = 0
        for state in grid_states:
            if state['is_active']:
                pos = state['direction']
                total_position = pos
                total_c1 += state['c1']
                total_c2 += state['c2']
                
                margin_required_today += calc_margin_per_contract(p1) * state['c1'] + calc_margin_per_contract(p2) * state['c2']
                
                if pos == 1:
                    pnl_s2 = (p2 - state['entry_p2']) * SHARES_PER_CONTRACT * state['c2']
                    pnl_s1 = (state['entry_p1'] - p1) * SHARES_PER_CONTRACT * state['c1']
                else:
                    pnl_s2 = (state['entry_p2'] - p2) * SHARES_PER_CONTRACT * state['c2']
                    pnl_s1 = (p1 - state['entry_p1']) * SHARES_PER_CONTRACT * state['c1']
                
                state['unrealized_pnl'] = pnl_s1 + pnl_s2
                unrealized_pnl += (pnl_s1 + pnl_s2)

        action = ''
        
        # 2. 結算日後次日重新開倉 (轉倉)
        for state in grid_states:
            if not state['is_active'] and state['pending_reopen_dir'] != 0:
                pos = state['pending_reopen_dir']
                params = state['params']
                available = cash * margin_usage_pct if total_position == 0 else cash
                dyn_c1, dyn_c2, dyn_req = get_trade_size(p1, p2, params['pairs'], available)
                
                if is_limit:
                    action += f'[網格{params["id"]}] 漲跌停，轉倉延後 '
                    skipped_limits += 1
                elif dyn_c1 > 0 and cash >= dyn_req and not insufficient_margin_error:
                    state['is_active'] = True
                    state['direction'] = pos
                    state['c1'], state['c2'] = dyn_c1, dyn_c2
                    state['entry_p1'], state['entry_p2'] = p1, p2
                    state['entry_zscore'] = z
                    state['entry_date'] = date
                    
                    cost = calc_trading_cost(p1, dyn_c1) + calc_trading_cost(p2, dyn_c2)
                    cash -= cost
                    realized_pnl_total -= cost
                    total_roll_cost += cost
                    
                    direction_str = '做多配對' if pos == 1 else '做空配對'
                    desc = make_desc(pos, dyn_c1, dyn_c2)
                    action += f'[網格{params["id"]}] 轉倉開倉 '
                    trade_log.append({
                        '日期': date.strftime('%Y-%m-%d'), '動作': f'[網格{params["id"]}] 轉倉開倉({direction_str})',
                        '部位描述': desc,
                        f'{sym1}({name1})多空': '做空' if pos == 1 else '做多',
                        f'{sym2}({name2})多空': '做多' if pos == 1 else '做空',
                        f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                        f'{sym1}口數': dyn_c1, f'{sym2}口數': dyn_c2,
                        '保證金佔用': round(margin_required_today + dyn_req, 0),
                        'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0
                    })
                    state['pending_reopen_dir'] = 0
                else:
                    action += f'[網格{params["id"]}] 轉倉失敗 '
                    state['pending_reopen_dir'] = 0

        # 3. 結算日強制平倉
        if is_settlement and any(s['is_active'] for s in grid_states):
            for state in grid_states:
                if state['is_active']:
                    pos = state['direction']
                    params = state['params']
                    cost = calc_trading_cost(p1, state['c1']) + calc_trading_cost(p2, state['c2'])
                    net_pnl = state['unrealized_pnl'] - cost
                    cash += net_pnl
                    realized_pnl_total += net_pnl
                    total_roll_cost += cost
                    settlement_rolls += 1
                    
                    direction_str = '做多配對' if pos == 1 else '做空配對'
                    desc = make_desc(pos, state['c1'], state['c2'])
                    action += f'[網格{params["id"]}] 結算平倉, 損益={net_pnl:+,.0f} '
                    trade_log.append({
                        '日期': date.strftime('%Y-%m-%d'), '動作': f'[網格{params["id"]}] 結算平倉({direction_str})',
                        '部位描述': desc,
                        f'{sym1}({name1})多空': '平倉(買回)' if pos == 1 else '平倉(賣出)',
                        f'{sym2}({name2})多空': '平倉(賣出)' if pos == 1 else '平倉(買回)',
                        f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                        f'{sym1}口數': state['c1'], f'{sym2}口數': state['c2'],
                        '保證金佔用': 0,
                        'Z-Score': round(z, 2),
                        '開倉Z': round(state['entry_zscore'], 2),
                        '交易成本': round(cost, 0),
                        '損益': round(net_pnl, 0)
                    })
                    
                    use_smart_roll = advanced_params.get('use_smart_roll', False)
                    smart_roll_z = advanced_params.get('smart_roll_z', 1.0)
                    
                    trade_mode = advanced_params.get('trade_mode', '收斂 (均值回歸)')
                    if use_grid:
                        tp_z = params['tp_z']
                        if trade_mode == '收斂 (均值回歸)':
                            is_converged = (pos == 1 and z >= tp_z) or (pos == -1 and z <= -tp_z)
                        else:
                            is_converged = (pos == 1 and z <= tp_z) or (pos == -1 and z >= -tp_z)
                    else:
                        if trade_mode == '收斂 (均值回歸)':
                            is_converged = (pos == 1 and z >= dyn_exit_upper) or (pos == -1 and z <= dyn_exit_lower)
                        else:
                            is_converged = (pos == 1 and z <= dyn_exit_upper) or (pos == -1 and z >= dyn_exit_lower)
                        
                    if use_smart_roll and net_pnl > 0 and abs(z) <= smart_roll_z:
                        state['pending_reopen_dir'] = 0
                        action += f" (防呆不再建倉)"
                    elif is_converged:
                        state['pending_reopen_dir'] = 0
                    else:
                        state['pending_reopen_dir'] = pos
                        
                    state['is_active'] = False
                    state['is_stopped_out'] = False
        else:
            # 4. 正常平倉 (TP) / 停損平倉 (SL)
            for state in grid_states:
                if state['is_active']:
                    pos = state['direction']
                    params = state['params']
                    
                    trade_mode = advanced_params.get('trade_mode', '收斂 (均值回歸)')
                    if use_grid:
                        if trade_mode == '收斂 (均值回歸)':
                            hit_tp = (pos == 1 and z >= params['tp_z']) or (pos == -1 and z <= -params['tp_z'])
                            hit_sl = (pos == 1 and z <= -params['sl_z']) or (pos == -1 and z >= params['sl_z'])
                        else:
                            hit_tp = (pos == 1 and z <= params['tp_z']) or (pos == -1 and z >= -params['tp_z'])
                            hit_sl = (pos == 1 and z >= params['sl_z']) or (pos == -1 and z <= -params['sl_z'])
                    else:
                        if trade_mode == '收斂 (均值回歸)':
                            hit_tp = (pos == 1 and z >= dyn_exit_upper) or (pos == -1 and z <= dyn_exit_lower)
                        else:
                            hit_tp = (pos == 1 and z <= dyn_exit_upper) or (pos == -1 and z >= dyn_exit_lower)
                        hit_sl = False 
                    
                    if hit_tp or hit_sl:
                        if is_limit:
                            action += f'[網格{params["id"]}] 漲跌停無法平倉 '
                            skipped_limits += 1
                        else:
                            cost = calc_trading_cost(p1, state['c1']) + calc_trading_cost(p2, state['c2'])
                            net_pnl = state['unrealized_pnl'] - cost
                            cash += net_pnl
                            realized_pnl_total += net_pnl
                            
                            direction_str = '做多配對' if pos == 1 else '做空配對'
                            desc = make_desc(pos, state['c1'], state['c2'])
                            
                            if hit_tp:
                                action_name = '停利平倉'
                            else:
                                action_name = '停損平倉'
                                state['is_stopped_out'] = True
                                
                            action += f'[網格{params["id"]}] {action_name}: {direction_str}, 損益={net_pnl:+,.0f} '
                            
                            trade_log.append({
                                '日期': date.strftime('%Y-%m-%d'), '動作': f'[網格{params["id"]}] {action_name}({direction_str})',
                                '部位描述': f"{action_name}: {desc}",
                                f'{sym1}({name1})多空': '平倉(買回)' if pos == 1 else '平倉(賣出)',
                                f'{sym2}({name2})多空': '平倉(賣出)' if pos == 1 else '平倉(買回)',
                                f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                                f'{sym1}口數': state['c1'], f'{sym2}口數': state['c2'],
                                '保證金佔用': round(margin_required_today, 0),
                                'Z-Score': round(z, 2),
                                '開倉Z': round(state['entry_zscore'], 2),
                                '交易成本': round(cost, 0),
                                '損益': round(net_pnl, 0)
                            })
                            state['is_active'] = False
            
            # 5. 解除停損冷卻 (Re-entry)
            for state in grid_states:
                if state['is_stopped_out']:
                    params = state['params']
                    if use_grid:
                        if abs(z) <= params['reentry_z']:
                            state['is_stopped_out'] = False
                            action += f" [網格{params['id']}] Z分數回落解除冷卻 "
                    else:
                        state['is_stopped_out'] = False

            # 6. 正常開倉 (Entry)
            for state in grid_states:
                if not state['is_active'] and not state['is_stopped_out'] and state['pending_reopen_dir'] == 0:
                    params = state['params']
                    enter_dir = 0
                    
                    # Check trade_mode
                    trade_mode = advanced_params.get('trade_mode', '收斂 (均值回歸)')
                    
                    if use_grid:
                        if z >= params['entry_z']:
                            enter_dir = -1 if trade_mode == '收斂 (均值回歸)' else 1
                        elif z <= -params['entry_z']:
                            enter_dir = 1 if trade_mode == '收斂 (均值回歸)' else -1
                    else:
                        if z > dyn_upper:
                            enter_dir = -1 if trade_mode == '收斂 (均值回歸)' else 1
                        elif z < dyn_lower:
                            enter_dir = 1 if trade_mode == '收斂 (均值回歸)' else -1
                        
                    if enter_dir != 0:
                        available = cash * margin_usage_pct if total_position == 0 else cash
                        dyn_c1, dyn_c2, dyn_req = get_trade_size(p1, p2, params['pairs'], available)
                        
                        if is_limit:
                            action += f'[網格{params["id"]}] 漲跌停跳過開倉 '
                            skipped_limits += 1
                        elif not ou_pass:
                            action += f'[網格{params["id"]}] OU未達標過濾 '
                        elif beta_trans:
                            action += f'[網格{params["id"]}] Beta風險過濾 '
                        elif dyn_c1 > 0 and cash >= dyn_req and not insufficient_margin_error:
                            state['is_active'] = True
                            state['direction'] = enter_dir
                            state['c1'], state['c2'] = dyn_c1, dyn_c2
                            state['entry_p1'], state['entry_p2'] = p1, p2
                            state['entry_zscore'] = z
                            state['entry_date'] = date
                            
                            cost = calc_trading_cost(p1, dyn_c1) + calc_trading_cost(p2, dyn_c2)
                            cash -= cost
                            realized_pnl_total -= cost
                            
                            direction_str = '做多配對' if enter_dir == 1 else '做空配對'
                            desc = make_desc(enter_dir, dyn_c1, dyn_c2)
                            action += f'[網格{params["id"]}] 開倉: {desc} '
                            
                            trade_log.append({
                                '日期': date.strftime('%Y-%m-%d'), '動作': f'[網格{params["id"]}] 開倉({direction_str})',
                                '部位描述': desc,
                                f'{sym1}({name1})多空': '做空' if enter_dir == 1 else '做多',
                                f'{sym2}({name2})多空': '做多' if enter_dir == 1 else '做空',
                                f'{sym1}價': round(p1, 2), f'{sym2}價': round(p2, 2),
                                f'{sym1}口數': dyn_c1, f'{sym2}口數': dyn_c2,
                                '保證金佔用': round(margin_required_today + dyn_req, 0),
                                'Z-Score': round(z, 2), '交易成本': round(cost, 0), '損益': 0
                            })
                            total_position = enter_dir
                        else:
                            action += f'[網格{params["id"]}] 資金不足 '

        margin_required_today = sum(calc_margin_per_contract(p1)*s['c1'] + calc_margin_per_contract(p2)*s['c2'] for s in grid_states if s['is_active'])
        unrealized_pnl = sum(s.get('unrealized_pnl', 0.0) for s in grid_states if s['is_active'])
        
        frozen_margin = margin_required_today
        equity = cash + unrealized_pnl

        records.append({
            '日期': date,
            f'{sym1}價': p1, f'{sym2}價': p2,
            'Z-Score': z, '持倉': total_position,
            'Upper_Bound': dyn_upper if not use_grid else np.nan,
            'Lower_Bound': dyn_lower if not use_grid else np.nan,
            'Exit_Upper': dyn_exit_upper if not use_grid else np.nan,
            'Exit_Lower': dyn_exit_lower if not use_grid else np.nan,
            f'{sym1}口數': sum(s['c1'] for s in grid_states if s['is_active']),
            f'{sym2}口數': sum(s['c2'] for s in grid_states if s['is_active']),
            '未實現損益': unrealized_pnl,
            '累計已實現損益': realized_pnl_total,
            '凍結保證金': frozen_margin,
            '帳戶淨值': equity,
            '動作': action,
            'Beta': _hr[date] if advanced_params.get('use_beta_transition', False) else np.nan,
            'OU_HalfLife': roll_hl_series[date] if advanced_params.get('use_ou_filter', False) else np.nan,
            'OU_R2': roll_r2_series[date] if advanced_params.get('use_ou_filter', False) else np.nan
        })

    df_records = pd.DataFrame(records)
    df_records.set_index('日期', inplace=True)
    df_trades = pd.DataFrame(trade_log)

    close_trades = df_trades[df_trades['動作'].str.contains('平倉')] if not df_trades.empty else pd.DataFrame()
    stats = {}
    stats['insufficient_margin_error'] = insufficient_margin_error
    stats['base_s1'] = base_s1
    stats['base_s2'] = base_s2
    stats['base_margin'] = base_margin
    stats['hedge_ratio'] = signal_df['Hedge_Ratio'].iloc[-1] if not signal_df.empty else 0
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
    
    peak = df_records['帳戶淨值'].cummax()
    drawdown = peak - df_records['帳戶淨值']
    drawdown_pct = drawdown / peak * 100
    stats['max_drawdown'] = drawdown.max()
    stats['max_drawdown_pct'] = drawdown_pct.max()

    trading_days = len(df_records)
    years = trading_days / 252
    
    daily_returns = df_records['帳戶淨值'].pct_change().dropna()
    
    if years > 0 and stats['final_equity'] > 0:
        stats['annualized_return'] = ((stats['final_equity'] / initial_capital) ** (1 / years) - 1) * 100
        ann_vol = daily_returns.std() * np.sqrt(252)
        if ann_vol > 0:
            stats['sharpe_ratio'] = (stats['annualized_return'] / 100 - 0.015) / ann_vol
        else:
            stats['sharpe_ratio'] = 0.0
    else:
        stats['annualized_return'] = 0.0
        stats['sharpe_ratio'] = 0.0

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

    margin_when_holding = df_records[df_records['持倉'] != 0]['凍結保證金']
    if not margin_when_holding.empty:
        stats['min_margin'] = margin_when_holding.min()
        stats['max_margin'] = margin_when_holding.max()
        stats['avg_margin'] = margin_when_holding.mean()
    else:
        stats['min_margin'] = 0
        stats['max_margin'] = 0
        stats['avg_margin'] = 0
        
    valid_hl = df_records['OU_HalfLife'].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid_hl.empty:
        stats['avg_ou_half_life'] = valid_hl.mean()
    else:
        stats['avg_ou_half_life'] = None

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
        code = t.replace('.TWO', '').replace('.TW', '')
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
        st.markdown("### 🧭 交易邏輯與方向")
        trade_mode = st.radio("配對策略邏輯", ['收斂 (均值回歸)', '發散 (趨勢跟蹤)'], index=0, help="收斂：突破上界做空，跌破下界做多；發散：突破上界做多，跌破下界做空。")
        
        st.markdown("---")
        st.markdown("### 📐 數學模型與參數設定")
        
        ensemble_logic = '單一模型'
        selected_models = []
        if trade_mode == '收斂 (均值回歸)':
            model_options = ['Z-Score (標準)', 'OU 過程 (動態邊界)', '共整合 + GARCH', '卡爾曼濾波 (動態對沖比例)', 'Copula (CMPI 機率)', 'Merton 跳躍擴散模型 (過濾結構破裂)', 'SDDE 隨機延遲方程式 (過濾動能慣性)']
            selected_models = st.multiselect("收斂訊號組合 (可多選)", model_options, default=[model_options[0]])
            if not selected_models:
                st.warning("請至少選擇一個收斂模型！")
            if len(selected_models) > 1:
                ensemble_logic = st.radio("組合判定邏輯", ['交集 (AND) - 嚴格過濾', '聯集 (OR) - 寬鬆捕捉'], index=0)
            else:
                ensemble_logic = '單一模型'
        else:
            model_options = ['GSADF (爆炸性泡沫檢定)', '非參數 CUSUM (多變量幾何破裂)', 'DCC-GARCH-VECM (特異性漂移爆發)', 'MRS (馬爾可夫區制轉換)']
            selected_models = st.multiselect("發散訊號組合 (可多選)", model_options, default=[model_options[0]])
            if not selected_models:
                st.warning("請至少選擇一個發散模型！")
            if len(selected_models) > 1:
                ensemble_logic = st.radio("組合判定邏輯", ['交集 (AND) - 嚴格過濾', '聯集 (OR) - 寬鬆捕捉'], index=0)
            else:
                ensemble_logic = '單一模型'
        
        model_params = {}
        for idx, model_type in enumerate(selected_models):
            st.markdown(f"#### {idx+1}. {model_type} 參數")
            m_params = {}
            if model_type == 'Z-Score (標準)' or model_type == '共整合 + GARCH' or model_type == '卡爾曼濾波 (動態對沖比例)':
                m_params['z_entry'] = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", 0.1, 5.0, 2.0, 0.1, help="調整進場的敏銳度，數值越小交易越頻繁", key=f"z_entry_{idx}_{model_type}")
                m_params['z_exit'] = st.slider("平倉閾值 (Z 回歸)", -2.0, 2.0, 0.0, 0.1, help="當指標回歸到此數值時平倉", key=f"z_exit_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, help="用於計算標準差或卡爾曼標準化", key=f"z_window_{idx}_{model_type}")
            elif model_type == 'OU 過程 (動態邊界)':
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, help="用於擬合 OU 過程參數 (Theta, Mu, Sigma)", key=f"ou_window_{idx}_{model_type}")
                st.info("OU 模型會自動根據均值回歸速度(Theta)與波動率計算動態上下界，無須手動設定固定閾值。")
            elif model_type == 'Copula (CMPI 機率)':
                m_params['prob_threshold'] = st.slider("條件機率閾值 (CMPI)", 0.500, 0.999, 0.950, 0.001, format="%.3f", help="達到多少極端機率才開倉 (0.5以上)", key=f"cp_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, help="用於擬合 Copula 相關性與累積分配函數", key=f"cw_{idx}_{model_type}")
            elif model_type == 'Merton 跳躍擴散模型 (過濾結構破裂)':
                m_params['z_entry'] = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", 0.1, 5.0, 2.0, 0.1, key=f"je_{idx}_{model_type}")
                m_params['z_exit'] = st.slider("平倉閾值 (Z 回歸)", -2.0, 2.0, 0.0, 0.1, key=f"jx_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, key=f"jw_{idx}_{model_type}")
                m_params['jump_threshold'] = st.slider("跳躍過濾敏感度 (標準差倍數)", 1.0, 10.0, 3.0, 0.1, help="當日波動超過此倍數即判定為異常跳空，強制暫停該日交易。數值越低過濾越嚴格。", key=f"jt_{idx}_{model_type}")
                st.info(" Jump-Diffusion 會在偵測到極端異常跳空時，動態將上下界設為無窮大以封鎖開倉。")
            elif model_type == 'SDDE 隨機延遲方程式 (過濾動能慣性)':
                m_params['z_entry'] = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", 0.1, 5.0, 2.0, 0.1, key=f"se_{idx}_{model_type}")
                m_params['z_exit'] = st.slider("平倉閾值 (Z 回歸)", -2.0, 2.0, 0.0, 0.1, key=f"sx_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, key=f"sw_{idx}_{model_type}")
                m_params['delay_tau'] = st.slider("歷史記憶天數 (τ)", 1, 20, 5, 1, help="計算過去幾天的發散動能。動能越強，模型會主動延後進場時機。", key=f"st_{idx}_{model_type}")
                st.info(" SDDE 模型會計算時間延遲積分，自動將帶有強大動能的『假突破』Z分數縮小，避免過早進場。")
            elif model_type == '非參數 CUSUM (多變量幾何破裂)':
                st.markdown("##### 非參數 CUSUM 參數")
                m_params['cusum_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, key=f"cw2_{idx}_{model_type}")
                m_params['k_shift'] = st.slider("中位數漂移閾值 (MAD 倍數)", 0.1, 3.0, 1.0, 0.1, key=f"ck_{idx}_{model_type}")
                m_params['tau_threshold'] = st.slider("停止時間閾值 (τ)", 1.0, 20.0, 5.0, 0.5, key=f"ct_{idx}_{model_type}")
            elif model_type == 'GSADF (爆炸性泡沫檢定)':
                st.markdown("##### GSADF 參數")
                m_params['gsadf_window'] = st.slider("滾動窗口 (天)", 15, 500, 30, 1, key=f"gw_{idx}_{model_type}")
                m_params['adf_threshold'] = st.slider("ADF t-統計量 閾值", 0.5, 4.0, 1.5, 0.1, key=f"gt_{idx}_{model_type}")
            elif model_type == 'MRS (馬爾可夫區制轉換)':
                st.markdown("##### MRS 參數")
                m_params['mrs_window'] = st.slider("滾動窗口 (天)", 30, 500, 120, 5, key=f"mw_{idx}_{model_type}")
                m_params['prob_threshold'] = st.slider("發散區制機率閾值", 0.5, 0.99, 0.8, 0.05, key=f"mp_{idx}_{model_type}")
                st.info("⚠️ MRS 模型內部使用 MLE 估計轉移矩陣，計算極度耗時。")
            elif model_type == 'DCC-GARCH-VECM (特異性漂移爆發)':
                st.markdown("##### DCC-GARCH 參數")
                m_params['garch_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, key=f"dc_w_{idx}_{model_type}")
                m_params['t_threshold'] = st.slider("爆發 t-統計量閾值", 1.0, 10.0, 3.0, 0.5, key=f"dc_t_{idx}_{model_type}")
            model_params[model_type] = m_params

        st.markdown("---")

        st.markdown("---")
        st.markdown("### 🧪 進階過濾機制 (OU/Beta)")
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
        st.markdown("### 🕸️ 網格配對交易設定")
        use_grid = st.toggle("啟用獨立網格模式", value=True, help="每一層網格獨立運作，擁有專屬的進場、停利、停損與重新建倉門檻。若關閉，則退回單一進出場模式，並自動套用上方高階模型的動態邊界。")
        
        grid_levels = []
        if use_grid:
            st.info("⚠️ 啟用此功能後，將優先採用以下網格設定，忽略高階模型的浮動邊界。")
            num_grids = st.number_input("總共分幾層網格?", value=2, min_value=1, max_value=10, step=1)
            for i in range(int(num_grids)):
                st.markdown(f"**第 {i+1} 層網格**")
                col1, col2 = st.columns(2)
                with col1:
                    entry_z = st.number_input(f"進場門檻 (Z)", value=1.0 + i*1.0, step=0.1, key=f"g_ent_{i}")
                    tp_z = st.number_input(f"停利門檻 (Z)", value=0.0 + i*1.0, step=0.1, key=f"g_tp_{i}")
                    reentry_z = st.number_input(f"停損後重啟門檻 (Z)", value=entry_z, step=0.1, key=f"g_re_{i}")
                with col2:
                    pairs = st.number_input(f"口數 (組)", value=1, min_value=1, step=1, key=f"g_pairs_{i}")
                    sl_z = st.number_input(f"停損門檻 (Z)", value=3.0, step=0.1, key=f"g_sl_{i}")
                
                grid_levels.append({
                    'id': i+1,
                    'entry_z': entry_z,
                    'tp_z': tp_z,
                    'sl_z': sl_z,
                    'reentry_z': reentry_z,
                    'pairs': pairs
                })

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

        with st.spinner("正在執行高階回測分析..."):
            name1 = name_map.get(sym1.replace('.TWO', '').replace('.TW', ''), '')
            name2 = name_map.get(sym2.replace('.TWO', '').replace('.TW', ''), '')
            advanced_params = {
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
                'use_grid': use_grid,
                'grid_levels': grid_levels
            }
            
            advanced_params['trade_mode'] = trade_mode  # Pass trade_mode via advanced_params
            
            df_records, df_trades, stats = run_backtest(
        S1, S2, sym1, sym2, name1, name2, initial_capital, 
        selected_models, model_params,
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
                st.info(f"**{sym1} ({name1})**\n\n**產業類別**: {ind1.get('Industry', '未知')}\n\n**主要業務**: {ind1.get('Business', '未知')}")
            with ind_col2:
                st.info(f"**{sym2} ({name2})**\n\n**產業類別**: {ind2.get('Industry', '未知')}\n\n**主要業務**: {ind2.get('Business', '未知')}")

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
                <div class="value">{sym1}({name1})×{stats['base_s1']} : {sym2}({name2})×{stats['base_s2']}</div>
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
            sharpe_str = f"{stats['sharpe_ratio']:.2f}"
            sharpe_cls = 'positive' if stats['sharpe_ratio'] > 1.0 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">夏普比率</div>
                <div class="value {sharpe_cls}">{sharpe_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with more5:
            hl_str = f"{stats['avg_ou_half_life']:.1f}天" if stats['avg_ou_half_life'] is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">OU半衰期</div>
                <div class="value">{hl_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with more6:
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

        # (2) 指標 + 訊號
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Z-Score'],
            name='指標 (Indicator)', line=dict(color='#BB86FC', width=1),
            fill='tozeroy', fillcolor='rgba(187,134,252,0.1)'
        ))
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Upper_Bound'],
            name='上界 (做空訊號)', line=dict(color='red', width=1, dash='dash')
        ))
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Lower_Bound'],
            name='下界 (做多訊號)', line=dict(color='green', width=1, dash='dash')
        ))
        # 繪製平倉線
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Exit_Upper'],
            name='平倉線上界', line=dict(color='gray', width=1, dash='dot')
        ))
        if not (df_records['Exit_Upper'] == df_records['Exit_Lower']).all():
            fig_z.add_trace(go.Scatter(
                x=df_records.index, y=df_records['Exit_Lower'],
                name='平倉線下界', line=dict(color='gray', width=1, dash='dot')
            ))

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
            • 指標 > 上界閾值 → 做空 Spread（空標的B + 多標的A）<br>
            • 指標 < 下界閾值 → 做多 Spread（多標的B + 空標的A）<br>
            • 指標回歸平倉線 → 平倉<br>
            • 漲跌停日（±10%）自動跳過，不會產生無法成交的虛假訊號<br>
            • 每月第 3 個禮拜三結算日自動平倉，次日以新價格重新建倉（含轉倉成本）
        </div>
        """, unsafe_allow_html=True)

        # 顯示可用標的列表
        st.markdown("### 📋 可用標的清單")
        st.markdown(f"共 **{len(tickers)}** 檔標的（取自期交所股票期貨掛牌名單，已過濾有完整 5 年資料者）")

        ticker_display = []
        for t in tickers:
            code = t.replace('.TWO', '').replace('.TW', '')
            name = name_map.get(code, '')
            ticker_display.append({'代號': t, '簡稱': name, '市場': 'TWSE' if '.TW' in t else 'TPEx'})
        st.dataframe(pd.DataFrame(ticker_display), use_container_width=True, height=300)


if __name__ == '__main__':
    main()
