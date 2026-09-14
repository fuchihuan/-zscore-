import os

app_path = "app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

start_marker = "    # --- 逐日模擬 ---"
end_marker = "    df_records = pd.DataFrame(records)"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Markers not found!")
    exit(1)

new_logic = """    # --- 逐日模擬 ---
    valid_dates = zscore.index
    dates_list = list(valid_dates)
    
    use_grid = advanced_params.get('use_grid', False)
    grid_levels = advanced_params.get('grid_levels', [])
    
    # 支援向下相容（如果關閉網格模式）
    if not use_grid:
        z_entry = advanced_params.get('z_entry', 2.0)
        z_exit = advanced_params.get('z_exit', 0.0)
        grid_levels = [{
            'id': 1, 'entry_z': z_entry, 'tp_z': z_exit, 'sl_z': 999.0, 'reentry_z': 999.0, 'pairs': 1
        }]

    # 初始化每一層網格的狀態
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
    
    # 動態決定口數的內部函數
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

        unrealized_pnl = 0.0
        margin_required_today = 0.0
        
        # 1. 總結當前持有狀態 (計算總未實現損益與保證金)
        total_position = 0
        total_c1 = 0
        total_c2 = 0
        for state in grid_states:
            if state['is_active']:
                pos = state['direction']
                total_position = pos # 簡單標記方向
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
                    if use_smart_roll and net_pnl > 0 and abs(z) <= smart_roll_z:
                        state['pending_reopen_dir'] = 0
                        action += f" (防呆不再建倉)"
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
                    hit_tp = (pos == 1 and z >= params['tp_z']) or (pos == -1 and z <= -params['tp_z'])
                    hit_sl = (pos == 1 and z <= -params['sl_z']) or (pos == -1 and z >= params['sl_z'])
                    
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
                    if abs(z) <= params['reentry_z']:
                        state['is_stopped_out'] = False
                        action += f" [網格{params['id']}] Z分數回落解除冷卻 "

            # 6. 正常開倉 (Entry)
            for state in grid_states:
                if not state['is_active'] and not state['is_stopped_out'] and state['pending_reopen_dir'] == 0:
                    params = state['params']
                    enter_dir = 0
                    if z >= params['entry_z']:
                        enter_dir = -1
                    elif z <= -params['entry_z']:
                        enter_dir = 1
                        
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
            f'{sym1}口數': sum(s['c1'] for s in grid_states if s['is_active']),
            f'{sym2}口數': sum(s['c2'] for s in grid_states if s['is_active']),
            '未實現損益': unrealized_pnl,
            '累計已實現損益': realized_pnl_total,
            '凍結保證金': frozen_margin,
            '帳戶淨值': equity,
            '動作': action,
            'Beta': _hr[date],
            'OU_HalfLife': roll_hl_series[date] if advanced_params.get('use_ou_filter', False) else np.nan,
            'OU_R2': roll_r2_series[date] if advanced_params.get('use_ou_filter', False) else np.nan
        })

"""

new_content = content[:start_idx] + new_logic + content[end_idx:]

with open(app_path, "w", encoding="utf-8") as f:
    f.write(new_content)
print("Success")
