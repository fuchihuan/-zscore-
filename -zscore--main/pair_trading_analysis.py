import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import itertools
import os

def main():
    if not os.path.exists('stock_prices.csv'):
        print("Data file 'stock_prices.csv' not found. Please run data fetcher first.")
        return

    # Load data
    print("Loading stock prices...")
    df = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
    
    # Drop columns that have too many NaNs
    df = df.dropna(axis=1, thresh=len(df)*0.8)
    df = df.ffill().bfill() # fill remaining gaps
    
    symbols = df.columns.tolist()
    print(f"Loaded {len(symbols)} valid ticker data for analysis.", flush=True)
    
    # Calculate returns for correlation
    print("Calculating correlation matrix to filter candidates...", flush=True)
    returns = df.pct_change().dropna()
    corr_matrix = returns.corr()
    
    # Get upper triangle of correlation matrix to avoid duplicates
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Filter pairs with correlation > 0.6 to reduce Cointegration workload
    # We use a relatively low threshold like 0.6 because cointegration is based on price, not just returns
    # But for speed, let's just pick highly correlated returns > 0.6
    # Actually, correlation of prices is better for pairs trading pre-filter
    price_corr_matrix = df.corr()
    price_upper_tri = price_corr_matrix.where(np.triu(np.ones(price_corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs with price correlation > 0.85 efficiently
    corr_stacked = price_upper_tri.stack()
    high_corr_series = corr_stacked[corr_stacked < 0.9999].sort_values(ascending=False).head(100)
    
    high_corr_pairs = []
    for (sym1, sym2), val in high_corr_series.items():
        if sym1 != sym2:
            high_corr_pairs.append((sym1, sym2, val))
                
    print(f"Found {len(high_corr_pairs)} highly correlated pairs. Running cointegration tests...", flush=True)
    
    coint_results = []
    
    for i, (sym1, sym2, corr_val) in enumerate(high_corr_pairs):
        S1 = df[sym1]
        S2 = df[sym2]
        
        try:
            # Engle-Granger Cointegration test
            # H0: no cointegration. Low p-value means they are cointegrated.
            score, pvalue, _ = coint(S1, S2)
            coint_results.append({
                'Pair': f"{sym1}-{sym2}",
                'Stock1': sym1,
                'Stock2': sym2,
                'Price_Corr': corr_val,
                'Coint_pvalue': pvalue,
                'Coint_score': score
            })
        except Exception as e:
            pass
            
        if (i+1) % 20 == 0:
            print(f"Processed {i+1} / {len(high_corr_pairs)} pairs...", flush=True)
            
    coint_df = pd.DataFrame(coint_results)
    
    # Filter and sort
    if not coint_df.empty:
        # P-value < 0.05 indicates statistically significant cointegration
        valid_pairs = coint_df[coint_df['Coint_pvalue'] < 0.05]
        valid_pairs = valid_pairs.sort_values(by='Coint_pvalue', ascending=True)
        
        print("\n--- Top 20 Cointegrated Pairs ---")
        print(valid_pairs.head(20).to_string(index=False))
        
        valid_pairs.to_csv('cointegrated_pairs.csv', index=False)
        print("\nSaved significant pairs to 'cointegrated_pairs.csv'.")
    else:
        print("No cointegrated pairs found.")

if __name__ == "__main__":
    main()
