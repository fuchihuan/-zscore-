import pandas as pd
import yfinance as yf
import datetime

def get_taifex_symbols(csv_path='taifex_stocks.csv'):
    # Read the extracted symbols from CSV
    df = pd.read_csv(csv_path)
    symbols = df.iloc[:, 2].dropna().astype(str).tolist()
    # Remove any non-numeric symbols or symbols with weird letters if any
    symbols = [s.strip() for s in symbols if s.strip().isalnum()]
    print(f"Loaded {len(symbols)} symbols from TAIFEX list.")
    return symbols

def main():
    symbols = get_taifex_symbols()
    
    # yfinance uses .TW for TWSE and .TWO for TPEx
    # We will generate both for each symbol since we don't know the exchange
    tickers_to_try = []
    for s in symbols:
        tickers_to_try.append(f"{s}.TW")
        tickers_to_try.append(f"{s}.TWO")
        
    print(f"Total tickers to query yfinance: {len(tickers_to_try)}")
    
    # 抓取長期資料 (從 2013 至今)
    end_date = datetime.datetime.now()
    start_date = datetime.datetime(2013, 1, 1)
    
    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ...")
    
    # We use yf.download to grab them in parallel. 
    # Valid tickers will return data, invalid will return empty/NaN.
    data = yf.download(
        tickers_to_try,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        auto_adjust=False, # Use adj close for pair trading
        group_by='ticker',
        threads=True
    )
    
    # Process and filter valid downloaded data
    valid_tickers = []
    price_df_list = []
    
    # Depending on yfinance version, data might be structured differently. If multiple tickers, it's (ticker, field).
    # Since yf handles multiple tickers, we will get a MultiIndex if successfully downloaded.
    if isinstance(data.columns, pd.MultiIndex):
        # Yfinance >= 0.2 format usually has ticker as level 0 if group_by='ticker', let's check
        tickers_downloaded = data.columns.get_level_values(0).unique()
        for tk in tickers_downloaded:
            tk_data = data[tk]
            # Use Adj Close for analysis to account for dividends and splits
            if 'Adj Close' in tk_data.columns and not tk_data['Adj Close'].dropna().empty:
                adj_close = tk_data['Adj Close'].dropna()
                # 降低要求至 1250 天 (約 5 年交易日) 以確保至少有資料
                if len(adj_close) > 1250:
                    adj_close.name = tk
                    price_df_list.append(adj_close)
                    valid_tickers.append(tk)
    else:
        # If only one ticker somehow, or old format
        pass
        
    if len(price_df_list) > 0:
        # Combine all valid stocks into one DataFrame
        prices_df = pd.concat(price_df_list, axis=1)
        # Forward fill and then back fill missing data
        prices_df = prices_df.ffill().bfill()
        
        print(f"Successfully fetched valid 5-year data for {len(valid_tickers)} tickers.")
        print(prices_df.head())
        
        # Save to parquet or csv for next step
        output_file = 'stock_prices.csv'
        prices_df.to_csv(output_file)
        print(f"Saved stock prices to {output_file}")
    else:
        print("Failed to get any valid stock data.")

if __name__ == "__main__":
    main()
