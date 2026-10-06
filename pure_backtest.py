import datetime
import calendar
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from advanced_models import (
    get_zscore_signals, get_ou_signals, get_garch_signals,
    get_kalman_filter_signals, get_copula_signals, get_jump_diffusion_signals,
    get_sdde_signals, get_np_cusum_signals, get_gsadf_signals,
    get_markov_regime_signals, get_dcc_garch_vecm_signals
)

SHARES_PER_CONTRACT = 2000
MARGIN_RATE = 0.135
TRADING_FEE_RATE = 0.00002
COMMISSION_PER_CONTRACT = 30
LIMIT_PCT = 0.10

def detect_limit_days(series):
    pct = series.pct_change()
    limit_up = pct >= (LIMIT_PCT - 0.001)
    limit_down = pct <= -(LIMIT_PCT - 0.001)
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
    settlement_dates = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            cal = calendar.monthcalendar(year, month)
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

    common_idx = S1.index.intersection(S2.index)
    S1 = S1[common_idx].copy()
    S2 = S2[common_idx].copy()

    use_log_price = advanced_params.get('use_log_price', False)
    if use_log_price:
        model_S1 = np.log(S1)
        model_S2 = np.log(S2)
    else:
        model_S1 = S1.copy()
        model_S2 = S2.copy()

    trade_mode = advanced_params.get('trade_mode', '收斂 (均值回歸)')
    ensemble_logic = advanced_params.get('ensemble_logic', '單一模型')
    
    if isinstance(selected_models, str):
        selected_models = [selected_models]
        
    all_signals = []
    for m_type in selected_models:
        m_params = model_params.get(m_type, {})
        if m_type == 'Z-Score (標準)':
            sdf = get_zscore_signals(model_S1, model_S2, m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0), m_params.get('z_window', 20))
        elif m_type == 'OU 過程 (動態邊界)':
            sdf = get_ou_signals(model_S1, model_S2, m_params.get('z_window', 20), risk_free_rate=m_params.get('risk_free_rate', 0.015), trading_fee=m_params.get('trading_fee', 0.0004))
        elif m_type == '共整合 + GARCH':
            sdf = get_garch_signals(model_S1, model_S2, m_params.get('z_window', 20), m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0), garch_p=m_params.get('garch_p', 1), garch_q=m_params.get('garch_q', 1), garch_dist=m_params.get('garch_dist', 'Normal'))
        elif m_type == '卡爾曼濾波 (動態對沖比例)':
            sdf = get_kalman_filter_signals(
                model_S1, model_S2, 
                m_params.get('z_window', 20), 
                m_params.get('z_entry', 2.0), 
                m_params.get('z_exit', 0.0),
                kf_q=m_params.get('kf_q', 1e-4),
                kf_r=m_params.get('kf_r', 1e-3),
                kf_p0=m_params.get('kf_p0', 1.0)
            )
        elif m_type == 'Copula (CMPI 機率)':
            sdf = get_copula_signals(model_S1, model_S2, m_params.get('z_window', 20), m_params.get('prob_threshold', 0.95), copula_clip_bounds=m_params.get('copula_clip_bounds', 0.001))
        elif m_type == 'Merton 跳躍擴散模型 (過濾結構破裂)':
            sdf = get_jump_diffusion_signals(model_S1, model_S2, m_params.get('z_window', 20), m_params.get('jump_threshold', 3.0), m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0))
        elif m_type == 'SDDE 隨機延遲方程式 (過濾動能慣性)':
            sdf = get_sdde_signals(model_S1, model_S2, m_params.get('z_window', 20), m_params.get('delay_tau', 5), m_params.get('z_entry', 2.0), m_params.get('z_exit', 0.0))
        elif m_type == '非參數 CUSUM (多變量幾何破裂)':
            sdf = get_np_cusum_signals(model_S1, model_S2, m_params.get('cusum_window', 20), m_params.get('k_shift', 1.0), m_params.get('tau_threshold', 5.0), 0.0)
        elif m_type == 'GSADF (爆炸性泡沫檢定)':
            sdf = get_gsadf_signals(model_S1, model_S2, m_params.get('gsadf_window', 30), m_params.get('adf_threshold', 1.5), 0.0, min_window_pct=m_params.get('min_window_pct', 0.2))
        elif m_type == 'MRS (馬爾可夫區制轉換)':
            sdf = get_markov_regime_signals(model_S1, model_S2, m_params.get('mrs_window', 120), m_params.get('prob_threshold', 0.8), 0.0, switching_variance=m_params.get('switching_variance', True))
        elif m_type == 'DCC-GARCH-VECM (特異性漂移爆發)':
            sdf = get_dcc_garch_vecm_signals(model_S1, model_S2, m_params.get('garch_window', 20), m_params.get('t_threshold', 3.0), 0.0, dcc_span_vol=m_params.get('dcc_span_vol', None), dcc_span_drift=m_params.get('dcc_span_drift', None))
        else:
            sdf = get_zscore_signals(model_S1, model_S2, 2.0, 0.0, 20)
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
                else:
                    if any(entry_up):
                        current_state = 1
                    elif any(entry_dn):
                        current_state = -1
            else:
                if current_state == 1:
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

    zscore = signal_df['Indicator']
    spread = signal_df['Spread']
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

    if run_start_date and run_end_date:
        signal_df = signal_df.loc[run_start_date:run_end_date]
        zscore = zscore.loc[run_start_date:run_end_date]
        ou_pass_mask = ou_pass_mask.loc[run_start_date:run_end_date]
        beta_trans_mask = beta_trans_mask.loc[run_start_date:run_end_date]
        roll_hl_series = roll_hl_series.loc[run_start_date:run_end_date]
        roll_r2_series = roll_r2_series.loc[run_start_date:run_end_date]
    
    if signal_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {'insufficient_margin_error': True, 'msg': '所選區間內無可用資料或資料長度不足滾動窗口計算！'}

    start_date = zscore.index[0]
    p1_start = S1[start_date]
    p2_start = S2[start_date]
    usable_capital = initial_capital * margin_usage_pct

    if size_mode == '依保證金上限最大化（預設）':
        contracts_s1, contracts_s2, multiplier = calc_max_contracts(p1_start, p2_start, usable_capital)
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
    use_grid = advanced_params.get('use_grid', False)
    grid_levels = advanced_params.get('grid_levels', [])
    
    if not use_grid:
        grid_levels = [{
            'id': 1, 'pairs': advanced_params.get('fixed_pair_multiplier', 1)
        }]

    grid_states = []
    for g in grid_levels:
        grid_states.append({
            'params': g,
            'is_active': False,
            'is_stopped_out': False,
            'direction': 0,
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

    limit_s1 = detect_limit_days(S1[valid_dates])
    limit_s2 = detect_limit_days(S2[valid_dates])

    for date in valid_dates:
        p1 = S1[date]
        p2 = S2[date]
        z = zscore[date]
        is_limit = limit_s1[date] or limit_s2[date]
        is_settlement = (date.date() in settlement_dates)
        ou_pass = ou_pass_mask[date]
        beta_trans = beta_trans_mask[date]
        
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
                    state['pending_reopen_dir'] = 0
                else:
                    state['pending_reopen_dir'] = 0

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
                    
                    trade_log.append({
                        '日期': date.strftime('%Y-%m-%d'),
                        '損益': round(net_pnl, 0)
                    })
                    
                    trade_mode = advanced_params.get('trade_mode', '收斂 (均值回歸)')
                    if use_grid:
                        tp_z = params['tp_z']
                        is_converged = (pos == 1 and z >= tp_z) or (pos == -1 and z <= -tp_z) if trade_mode == '收斂 (均值回歸)' else (pos == 1 and z <= tp_z) or (pos == -1 and z >= -tp_z)
                    else:
                        is_converged = (pos == 1 and z >= dyn_exit_upper) or (pos == -1 and z <= dyn_exit_lower) if trade_mode == '收斂 (均值回歸)' else (pos == 1 and z <= dyn_exit_upper) or (pos == -1 and z >= dyn_exit_lower)
                    
                    if not is_converged and not is_limit:
                        state['pending_reopen_dir'] = pos
                    else:
                        state['pending_reopen_dir'] = 0
                        
                    state['is_active'] = False
                    state['unrealized_pnl'] = 0.0

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
                        skipped_limits += 1
                    else:
                        cost = calc_trading_cost(p1, state['c1']) + calc_trading_cost(p2, state['c2'])
                        net_pnl = state['unrealized_pnl'] - cost
                        cash += net_pnl
                        realized_pnl_total += net_pnl
                        trade_log.append({
                            '日期': date.strftime('%Y-%m-%d'),
                            '損益': round(net_pnl, 0)
                        })
                        state['is_active'] = False

        for state in grid_states:
            if state['is_stopped_out']:
                params = state['params']
                if use_grid:
                    if abs(z) <= params['reentry_z']:
                        state['is_stopped_out'] = False
                else:
                    state['is_stopped_out'] = False

        for state in grid_states:
            if not state['is_active'] and not state['is_stopped_out'] and state['pending_reopen_dir'] == 0:
                params = state['params']
                enter_dir = 0
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
                        skipped_limits += 1
                    elif not ou_pass:
                        pass
                    elif beta_trans:
                        pass
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
                        total_position = enter_dir

        margin_required_today = sum(calc_margin_per_contract(p1)*s['c1'] + calc_margin_per_contract(p2)*s['c2'] for s in grid_states if s['is_active'])
        unrealized_pnl = sum(s.get('unrealized_pnl', 0.0) for s in grid_states if s['is_active'])
        equity = cash + unrealized_pnl
        records.append({
            '日期': date,
            '帳戶淨值': equity
        })

    df_records = pd.DataFrame(records).set_index('日期')
    df_trades = pd.DataFrame(trade_log)

    stats = {}
    stats['final_equity'] = df_records['帳戶淨值'].iloc[-1]
    stats['total_return'] = df_records['帳戶淨值'].iloc[-1] - initial_capital
    stats['total_return_pct'] = (df_records['帳戶淨值'].iloc[-1] / initial_capital - 1) * 100
    
    peak = df_records['帳戶淨值'].cummax()
    drawdown = peak - df_records['帳戶淨值']
    drawdown_pct = drawdown / peak * 100
    stats['max_drawdown_pct'] = drawdown_pct.max()

    trading_days = len(df_records)
    years = trading_days / 252
    daily_returns = df_records['帳戶淨值'].pct_change().dropna()
    
    if years > 0 and stats['final_equity'] > 0:
        stats['annualized_return'] = ((stats['final_equity'] / initial_capital) ** (1 / years) - 1) * 100
        ann_vol = daily_returns.std() * np.sqrt(252)
        stats['sharpe_ratio'] = (stats['annualized_return'] / 100 - 0.015) / ann_vol if ann_vol > 0 else 0.0
    else:
        stats['annualized_return'] = 0.0
        stats['sharpe_ratio'] = 0.0

    if not df_trades.empty:
        win = df_trades[df_trades['損益'] > 0]
        lose = df_trades[df_trades['損益'] <= 0]
        stats['total_trades'] = len(df_trades)
        stats['win_trades'] = len(win)
        stats['lose_trades'] = len(lose)
        stats['win_rate'] = len(win) / len(df_trades) * 100
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

    try:
        _, pvalue, _ = coint(S1[common_idx], S2[common_idx])
        stats['coint_pvalue'] = pvalue
    except:
        stats['coint_pvalue'] = 1.0

    stats['base_s1'] = base_s1
    stats['base_s2'] = base_s2
    stats['base_margin'] = base_margin

    return df_records, df_trades, stats
