"""
增量更新股價資料
================
讀取現有 stock_prices.csv，從最後一天的日期開始下載到今天的新資料，
然後合併去重並覆寫 stock_prices.csv。
"""

import pandas as pd
import yfinance as yf
import datetime
import os
import sys

# 修正 Windows console 編碼問題
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_prices.csv')


def update_prices():
    if not os.path.exists(DATA_FILE):
        print("[ERROR] 找不到 stock_prices.csv，請先執行 data_fetcher.py 下載完整資料。")
        sys.exit(1)

    print("=" * 60)
    print("  股價資料增量更新")
    print("=" * 60)

    # 1. 讀取現有資料
    print("\n[1] 讀取現有股價資料...")
    existing = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    last_date = existing.index.max()
    today = datetime.datetime.now()

    print(f"    現有資料範圍: {existing.index.min().strftime('%Y-%m-%d')} ~ {last_date.strftime('%Y-%m-%d')}")
    print(f"    現有 ticker 數: {len(existing.columns)}")
    print(f"    今天日期: {today.strftime('%Y-%m-%d')}")

    # 2. 判斷是否需要更新
    if last_date.date() >= today.date() - datetime.timedelta(days=1):
        print("\n[OK] 資料已經是最新的！無須更新。")
        return

    # 從最後一天的前一天開始抓（確保銜接），後面再去重
    start_date = (last_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    print(f"\n[2] 下載新資料: {start_date} ~ {end_date}")

    # 3. 取得所有 ticker
    tickers = list(existing.columns)
    print(f"    下載 {len(tickers)} 檔股票...")

    # 4. 分批下載 (yfinance 一次太多可能失敗)
    batch_size = 50
    new_dfs = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        print(f"    批次 {batch_num}/{total_batches}: 下載 {len(batch)} 檔...", end=" ", flush=True)

        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                group_by='ticker',
                threads=True,
                progress=False
            )

            if data.empty:
                print("[WARN] 無資料")
                continue

            # 提取 Adj Close
            batch_prices = pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        if tk in data.columns.get_level_values(0):
                            col_data = data[tk]
                            if 'Adj Close' in col_data.columns:
                                adj_close = col_data['Adj Close']
                            elif 'Close' in col_data.columns:
                                adj_close = col_data['Close']
                            else:
                                continue
                            batch_prices[tk] = adj_close
                    except Exception:
                        pass
            else:
                # 單一 ticker
                if len(batch) == 1:
                    tk = batch[0]
                    if 'Adj Close' in data.columns:
                        batch_prices[tk] = data['Adj Close']
                    elif 'Close' in data.columns:
                        batch_prices[tk] = data['Close']

            if not batch_prices.empty:
                new_dfs.append(batch_prices)
                print(f"[OK] 取得 {len(batch_prices)} 天資料")
            else:
                print("[WARN] 無有效資料")

        except Exception as e:
            print(f"[ERROR] {e}")
            continue

    if not new_dfs:
        print("\n[WARN] 無法下載任何新資料，可能是非交易日或網路問題。")
        return

    # 5. 合併新資料
    new_data = pd.concat(new_dfs, axis=1)
    new_data.index = pd.to_datetime(new_data.index)

    # 確保欄位順序一致
    new_data = new_data.reindex(columns=existing.columns)

    print(f"\n[3] 新下載資料: {new_data.index.min().strftime('%Y-%m-%d')} ~ {new_data.index.max().strftime('%Y-%m-%d')}, {len(new_data)} 筆")

    # 6. 合併舊資料與新資料
    combined = pd.concat([existing, new_data])
    combined = combined[~combined.index.duplicated(keep='last')]  # 去重，保留新資料
    combined = combined.sort_index()
    combined = combined.ffill().bfill()

    new_days = len(combined) - len(existing)
    print(f"    原有 {len(existing)} 天 -> 更新後 {len(combined)} 天 (新增 {new_days} 天)")
    print(f"    更新後範圍: {combined.index.min().strftime('%Y-%m-%d')} ~ {combined.index.max().strftime('%Y-%m-%d')}")

    # 7. 備份舊檔並儲存
    backup_file = DATA_FILE.replace('.csv', f'_backup_{last_date.strftime("%Y%m%d")}.csv')
    if not os.path.exists(backup_file):
        os.rename(DATA_FILE, backup_file)
        print(f"\n[BACKUP] 舊資料備份至: {os.path.basename(backup_file)}")
    else:
        os.remove(DATA_FILE)

    combined.to_csv(DATA_FILE)
    print(f"[OK] 已儲存更新後的股價資料至 stock_prices.csv")

    print("\n" + "=" * 60)
    print("  更新完成！")
    print("=" * 60)


if __name__ == '__main__':
    update_prices()
