import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
from advanced_models import get_zscore_signals, get_ou_signals, get_garch_signals, get_kalman_filter_signals, get_copula_signals

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_FILE = 'stock_prices.csv'
RESULTS_DIR = 'Advanced_Models_3045_2412'

def run_simulation(S1, S2, signal_df, initial_capital=1000000):
    cash = initial_capital
    position = 0
    records = []
    
    shares_per_contract = 2000
    fee_rate = 0.00002
    comm = 30
    
    entry_s1 = 0
    entry_s2 = 0
    
    for date in signal_df.index:
        p1 = S1.loc[date]
        p2 = S2.loc[date]
        ind = signal_df['Indicator'].loc[date]
        up = signal_df['Upper_Bound'].loc[date]
        dn = signal_df['Lower_Bound'].loc[date]
        ex_up = signal_df['Exit_Upper'].loc[date]
        ex_dn = signal_df['Exit_Lower'].loc[date]
        
        c1, c2 = 1, 1 
        
        unrealized = 0
        if position == 1:
            unrealized = (p2 - entry_s2)*shares_per_contract*c2 + (entry_s1 - p1)*shares_per_contract*c1
        elif position == -1:
            unrealized = (entry_s2 - p2)*shares_per_contract*c2 + (p1 - entry_s1)*shares_per_contract*c1
            
        action = ''
        
        if position == 0:
            if ind > up:
                position = -1
                entry_s1, entry_s2 = p1, p2
                cost = (p1*shares_per_contract*c1 + p2*shares_per_contract*c2) * fee_rate + comm * 2
                cash -= cost
                action = 'Short Spread'
            elif ind < dn:
                position = 1
                entry_s1, entry_s2 = p1, p2
                cost = (p1*shares_per_contract*c1 + p2*shares_per_contract*c2) * fee_rate + comm * 2
                cash -= cost
                action = 'Long Spread'
        else:
            if (position == 1 and ind >= ex_up) or (position == -1 and ind <= ex_dn):
                cost = (p1*shares_per_contract*c1 + p2*shares_per_contract*c2) * fee_rate + comm * 2
                cash += unrealized - cost
                position = 0
                unrealized = 0
                action = 'Close'
                
        equity = cash + unrealized
        
        records.append({
            'Date': date,
            'Price1': p1, 'Price2': p2,
            'Indicator': ind, 'Position': position,
            'Upper': up, 'Lower': dn,
            'Equity': equity, 'Action': action
        })
        
    df = pd.DataFrame(records).set_index('Date')
    
    total_ret = df['Equity'].iloc[-1] / initial_capital - 1
    peak = df['Equity'].cummax()
    mdd = ((peak - df['Equity']) / peak).max()
    
    returns = df['Equity'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    return df, total_ret, mdd, sharpe

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    sym1 = '3045.TW'
    sym2 = '2412.TW'
    
    S1 = df[sym1].dropna()
    S2 = df[sym2].dropna()
    
    common = S1.index.intersection(S2.index)
    S1, S2 = S1.loc[common], S2.loc[common]
    
    configs = [
        {'name': '1_ZScore_Standard', 'model': 'Z-Score', 'params': {'z_entry': 2.0, 'z_exit': 0.0, 'z_window': 60}},
        {'name': '2_ZScore_Aggressive', 'model': 'Z-Score', 'params': {'z_entry': 1.5, 'z_exit': 0.0, 'z_window': 40}},
        {'name': '3_OU_Process_Standard', 'model': 'OU', 'params': {'z_window': 60}},
        {'name': '4_OU_Process_Fast', 'model': 'OU', 'params': {'z_window': 30}},
        {'name': '5_GARCH_Standard', 'model': 'GARCH', 'params': {'z_entry': 2.0, 'z_exit': 0.0, 'z_window': 60}},
        {'name': '6_GARCH_Conservative', 'model': 'GARCH', 'params': {'z_entry': 2.5, 'z_exit': 0.0, 'z_window': 60}},
        {'name': '7_Kalman_Standard', 'model': 'Kalman', 'params': {'z_entry': 2.0, 'z_exit': 0.0}},
        {'name': '8_Kalman_Aggressive', 'model': 'Kalman', 'params': {'z_entry': 1.5, 'z_exit': 0.0}},
        {'name': '9_Copula_Standard', 'model': 'Copula', 'params': {'prob_threshold': 0.95, 'z_window': 60}},
        {'name': '10_Copula_Extreme', 'model': 'Copula', 'params': {'prob_threshold': 0.99, 'z_window': 60}}
    ]
    
    results_summary = []
    
    for conf in configs:
        print(f"Running {conf['name']}...")
        model = conf['model']
        p = conf['params']
        
        if model == 'Z-Score':
            sig = get_zscore_signals(S1, S2, p['z_entry'], p['z_exit'], p['z_window'])
        elif model == 'OU':
            sig = get_ou_signals(S1, S2, p['z_window'])
        elif model == 'GARCH':
            sig = get_garch_signals(S1, S2, p['z_window'], p['z_entry'], p['z_exit'])
        elif model == 'Kalman':
            sig = get_kalman_filter_signals(S1, S2, p['z_entry'], p['z_exit'])
        elif model == 'Copula':
            sig = get_copula_signals(S1, S2, p['z_window'], p['prob_threshold'])
            
        res_df, ret, mdd, sharpe = run_simulation(S1, S2, sig)
        
        results_summary.append({
            'Config': conf['name'],
            'Return(%)': ret * 100,
            'MDD(%)': mdd * 100,
            'Sharpe': sharpe
        })
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        axes[0].plot(res_df.index, res_df['Price1'], label=sym1)
        axes[0].plot(res_df.index, res_df['Price2'], label=sym2)
        axes[0].set_title(f"{conf['name']} - Prices")
        axes[0].legend()
        
        axes[1].plot(res_df.index, res_df['Indicator'], label='Indicator')
        axes[1].plot(res_df.index, res_df['Upper'], 'r--', label='Upper Bound')
        axes[1].plot(res_df.index, res_df['Lower'], 'g--', label='Lower Bound')
        axes[1].set_title("Indicator & Bounds")
        axes[1].legend()
        
        axes[2].plot(res_df.index, res_df['Equity'], label='Equity')
        axes[2].set_title(f"Equity Curve (Ret: {ret*100:.1f}%, MDD: {mdd*100:.1f}%)")
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{conf['name']}.png"))
        plt.close()
        
        res_df.to_csv(os.path.join(RESULTS_DIR, f"{conf['name']}.csv"))
        
    pd.DataFrame(results_summary).to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print("Done. Results saved to", RESULTS_DIR)

if __name__ == '__main__':
    main()
