import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

def main():
    if not os.path.exists('stock_prices.csv') or not os.path.exists('cointegrated_pairs.csv'):
        print("Required files 'stock_prices.csv' or 'cointegrated_pairs.csv' do not exist.")
        return
        
    prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
    pairs = pd.read_csv('cointegrated_pairs.csv')
    
    if pairs.empty:
        print("No cointegrated pairs to backtest.")
        return
        
    print(f"Loaded {len(pairs)} cointegrated pairs.")
    
    # Pick the best pair (lowest p-value)
    best_pair = pairs.iloc[0]
    sym1 = best_pair['Stock1']
    sym2 = best_pair['Stock2']
    
    print(f"Backtesting best pair: {sym1} and {sym2} (p-value: {best_pair['Coint_pvalue']:.4f})")
    
    S1 = prices[sym1]
    S2 = prices[sym2]
    
    # Calculate Hedge Ratio using OLS regression
    S1_c = sm.add_constant(S1)
    model = sm.OLS(S2, S1_c).fit()
    hedge_ratio = model.params[sym1]
    
    # Spread = S2 - Hedge_Ratio * S1
    spread = S2 - hedge_ratio * S1
    
    # Z-score calculation (using rolling 60 days window for dynamic mean/std, or static)
    # Market conditions change, so rolling z-score is commonly used in pair trading.
    window = 60
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    zscore = (spread - spread_mean) / spread_std
    
    # Remove initial NaNs
    zscore = zscore.dropna()
    valid_dates = zscore.index
    
    # Simple trading logic:
    # Enter Short Spread (Short S2, Long S1): Z > 2
    # Enter Long Spread (Long S2, Short S1): Z < -2
    # Exit: Z crosses 0
    
    # Plotting
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Prices
    plt.subplot(3, 1, 1)
    S1[valid_dates].plot(label=sym1, color='blue')
    S2_scaled = S2[valid_dates] / hedge_ratio # scale for visualization
    S2_scaled.plot(label=f"{sym2} (scaled by 1/{hedge_ratio:.2f})", color='orange')
    plt.title(f"Prices of {sym1} and {sym2}")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Spread
    plt.subplot(3, 1, 2)
    spread[valid_dates].plot(color='green')
    spread_mean[valid_dates].plot(color='black', linestyle='--')
    plt.title("Spread (S2 - beta * S1)")
    plt.grid(True)
    
    # Plot 3: Z-Score
    plt.subplot(3, 1, 3)
    zscore.plot(color='purple')
    plt.axhline(0, color='black')
    plt.axhline(2.0, color='red', linestyle='--', label='Upper Threshold (+2.0)')
    plt.axhline(-2.0, color='red', linestyle='--', label='Lower Threshold (-2.0)')
    
    # Mark signals
    buy_signals = zscore[zscore < -2.0]
    sell_signals = zscore[zscore > 2.0]
    plt.scatter(buy_signals.index, buy_signals, color='green', marker='^', label='Long Spread Signal')
    plt.scatter(sell_signals.index, sell_signals, color='red', marker='v', label='Short Spread Signal')
    
    plt.title("Z-Score of Spread")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'PairTrading_{sym1}_{sym2}.png')
    print(f"Generated chart and saved to PairTrading_{sym1}_{sym2}.png")
    
if __name__ == "__main__":
    main()
