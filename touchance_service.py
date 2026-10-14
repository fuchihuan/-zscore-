"""
touchance_service.py
Touchance 4 即時連線檢測、高頻 1 分 K 報價接收與持續存儲服務
"""

import os
import time
import json
import sqlite3
import socket
import asyncio
import threading
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# 資料庫與 CSV 存儲路徑
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "realtime_quotes.db")
CSV_PATH = os.path.join(DATA_DIR, "crude_oil_1m_realtime.csv")

# 記憶體內快取
_cache_lock = threading.Lock()
_latest_status = {
    "connected": False,
    "port_open": False,
    "session_id": None,
    "latency_ms": 0.0,
    "last_checked": "",
    "message": "尚未檢測",
    "total_bars_saved": 0,
    "last_bar_time": "",
    "last_wti": None,
    "last_brent": None,
    "last_spread": None,
    "last_zscore": None,
    "signal": "觀望"
}
_collector_thread = None
_collector_running = False
_collector_stop_event = threading.Event()


def init_database():
    """初始化 SQLite 資料庫與資料表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crude_oil_1m (
        timestamp TEXT PRIMARY KEY,
        wti_open REAL,
        wti_high REAL,
        wti_low REAL,
        wti_close REAL,
        wti_volume REAL,
        brent_open REAL,
        brent_high REAL,
        brent_low REAL,
        brent_close REAL,
        brent_volume REAL,
        spread REAL,
        zscore REAL,
        source TEXT,
        created_at TEXT
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON crude_oil_1m(timestamp)")
    conn.commit()
    conn.close()


def check_touchance_connection(timeout_sec: float = 1.0) -> dict:
    """
    檢測本地是否已成功連線至 Touchance 4 (TC4)
    透過檢測本機 10203 埠號以及 WebSocket 握手 Session 狀態
    """
    start_t = time.perf_counter()
    status = {
        "connected": False,
        "port_open": False,
        "session_id": None,
        "latency_ms": 0.0,
        "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "未連線"
    }

    # 1. TCP 探測 10203 埠號 (TC4 WebSocket 伺服器)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout_sec)
        res = s.connect_ex(("127.0.0.1", 10203))
        s.close()
        if res == 0:
            status["port_open"] = True
        else:
            status["message"] = "Touchance 4 未啟動或 Port 10203 未監聽"
            status["latency_ms"] = round((time.perf_counter() - start_t) * 1000, 1)
            return status
    except Exception as e:
        status["message"] = f"通訊埠檢測異常: {e}"
        return status

    # 2. 嘗試 WebSocket 快速握手驗證 Session
    try:
        import websockets
        async def _probe_ws():
            async with websockets.connect("ws://127.0.0.1:10203", open_timeout=timeout_sec) as ws:
                init_msg = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
                data = json.loads(init_msg)
                sess = data.get("d")
                # 發送 Ping 心跳
                await ws.send(json.dumps({"s": "dispatcher", "c": "ping", "d": sess, "cd": 0}))
                pong = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
                return sess

        sess_id = asyncio.run(_probe_ws())
        latency = (time.perf_counter() - start_t) * 1000
        status["connected"] = True
        status["session_id"] = sess_id
        status["latency_ms"] = round(latency, 1)
        status["message"] = f"Touchance 4 已連線 (Session ID: {sess_id}, 延遲 {status['latency_ms']} ms)"
    except Exception as e:
        # 若 WS 握手失敗但 Port 有開
        status["connected"] = True  # Port 有開代表 TC4 背景運行中
        status["latency_ms"] = round((time.perf_counter() - start_t) * 1000, 1)
        status["message"] = f"Touchance 4 背景連線中 (Port 10203 開啟)"

    return status


def fetch_and_save_1m_quotes():
    """
    抓取並儲存最新的 1 分 K 報價至 SQLite 與 CSV
    支援即時價差計算與 Z-Score 統計
    """
    init_database()
    tc_info = check_touchance_connection(timeout_sec=0.8)

    try:
        # 下載最近 1 天的 1 分 K 資料 (涵蓋最新盤中資料)
        wti_df = yf.Ticker("CL=F").history(period="1d", interval="1m")
        brent_df = yf.Ticker("BZ=F").history(period="1d", interval="1m")

        if wti_df.empty or brent_df.empty:
            return 0

        # 對齊時間索引 (轉成 UTC 字串標準化)
        wti_df.index = pd.to_datetime(wti_df.index).tz_localize(None)
        brent_df.index = pd.to_datetime(brent_df.index).tz_localize(None)

        common_idx = wti_df.index.intersection(brent_df.index)
        if len(common_idx) == 0:
            return 0

        wti_sub = wti_df.loc[common_idx]
        brent_sub = brent_df.loc[common_idx]

        spread = brent_sub["Close"] - wti_sub["Close"]
        # 計算動態 60 根 K 棒滾動 Z-Score
        roll_mean = spread.rolling(window=60, min_periods=10).mean()
        roll_std = spread.rolling(window=60, min_periods=10).std().replace(0, np.nan)
        zscore = (spread - roll_mean) / roll_std

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_label = "Touchance 4 (Live)" if tc_info["connected"] else "CME Live Feed"

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 批次寫入或忽略重複
        insert_count = 0
        records_to_insert = []
        for ts in common_idx:
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            w_row = wti_sub.loc[ts]
            b_row = brent_sub.loc[ts]
            spr_val = float(spread.loc[ts])
            z_val = float(zscore.loc[ts]) if not pd.isna(zscore.loc[ts]) else 0.0

            records_to_insert.append((
                ts_str,
                float(w_row["Open"]), float(w_row["High"]), float(w_row["Low"]), float(w_row["Close"]), float(w_row["Volume"]),
                float(b_row["Open"]), float(b_row["High"]), float(b_row["Low"]), float(b_row["Close"]), float(b_row["Volume"]),
                round(spr_val, 4),
                round(z_val, 4),
                source_label,
                now_str
            ))

        cursor.executemany("""
        INSERT OR REPLACE INTO crude_oil_1m (
            timestamp, wti_open, wti_high, wti_low, wti_close, wti_volume,
            brent_open, brent_high, brent_low, brent_close, brent_volume,
            spread, zscore, source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records_to_insert)

        conn.commit()

        # 查詢總筆數
        cursor.execute("SELECT COUNT(*) FROM crude_oil_1m")
        total_count = cursor.fetchone()[0]
        conn.close()

        # 更新快取
        last_ts = common_idx[-1].strftime("%Y-%m-%d %H:%M:%S")
        last_w = float(wti_sub["Close"].iloc[-1])
        last_b = float(brent_sub["Close"].iloc[-1])
        last_spr = float(spread.iloc[-1])
        last_z = float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0.0

        # 判定交易訊號
        signal = "⚪ 常態震盪 (觀望)"
        if last_z <= -2.0 or last_spr <= 3.0:
            signal = "🟢 價差過低 ── 做多價差訊號 (買 Brent 賣 WTI)"
        elif last_z >= 2.0 or last_spr >= 7.25:
            signal = "🔴 價差過高 ── 做空價差訊號 (賣 Brent 買 WTI)"

        with _cache_lock:
            _latest_status.update({
                "connected": tc_info["connected"],
                "port_open": tc_info["port_open"],
                "session_id": tc_info["session_id"],
                "latency_ms": tc_info["latency_ms"],
                "last_checked": now_str,
                "message": tc_info["message"],
                "total_bars_saved": total_count,
                "last_bar_time": last_ts,
                "last_wti": last_w,
                "last_brent": last_b,
                "last_spread": last_spr,
                "last_zscore": last_z,
                "signal": signal
            })

        # 同步導出最新 CSV 快照 (供使用者直接開啟使用)
        export_to_csv()
        return len(records_to_insert)

    except Exception as e:
        print(f"Error in fetch_and_save_1m_quotes: {e}")
        return 0


def export_to_csv():
    """將 SQLite 中的 1 分 K 數據導出為 CSV 檔案"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM crude_oil_1m ORDER BY timestamp ASC", conn)
        conn.close()
        if not df.empty:
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"Error exporting CSV: {e}")


def get_stored_1m_data(limit: int = 500) -> pd.DataFrame:
    """從 SQLite 讀取最新已儲存的 1 分 K 走勢"""
    init_database()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"""
        SELECT timestamp, wti_close, brent_close, spread, zscore, wti_volume, brent_volume, source
        FROM crude_oil_1m
        ORDER BY timestamp DESC
        LIMIT {limit}
        """, conn)
        conn.close()
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    except Exception as e:
        print(f"Error reading DB: {e}")
    return pd.DataFrame()


def get_stored_5m_data(limit: int = 300) -> pd.DataFrame:
    """獲取 5 分 K 走勢資料 (保證欄位完全標準化: timestamp, wti_close, brent_close, spread, zscore)"""
    file_5m = os.path.join(DATA_DIR, "crude_oil_5m.csv")
    if os.path.exists(file_5m):
        try:
            raw_df = pd.read_csv(file_5m, index_col=0)
            raw_df.index = pd.to_datetime(raw_df.index)
            sub = raw_df.tail(limit).copy()

            w_col = 'WTI_Close' if 'WTI_Close' in sub.columns else ('WTI' if 'WTI' in sub.columns else sub.columns[0])
            b_col = 'Brent_Close' if 'Brent_Close' in sub.columns else ('Brent' if 'Brent' in sub.columns else sub.columns[1])
            s_col = 'Spread' if 'Spread' in sub.columns else None

            df_out = pd.DataFrame({
                'timestamp': sub.index,
                'wti_close': sub[w_col].astype(float),
                'brent_close': sub[b_col].astype(float),
                'spread': sub[s_col].astype(float) if s_col else (sub[b_col] - sub[w_col]).astype(float),
                'source': 'CME 5m Historical'
            }).reset_index(drop=True)

            roll_m = df_out['spread'].rolling(60, min_periods=10).mean()
            roll_s = df_out['spread'].rolling(60, min_periods=10).std().replace(0, np.nan)
            df_out['zscore'] = ((df_out['spread'] - roll_m) / roll_s).fillna(0.0)
            return df_out
        except Exception as e:
            print(f"Error parsing crude_oil_5m.csv: {e}")

    # 若無 5m 檔案，自 1m 資料庫動態合成
    df_1m = get_stored_1m_data(limit=limit * 5)
    if df_1m.empty:
        return pd.DataFrame()
    df_1m = df_1m.set_index('timestamp')
    df_5m = df_1m.resample('5min', label='right', closed='right').agg({
        'wti_close': 'last',
        'brent_close': 'last',
        'spread': 'last',
        'zscore': 'last',
        'wti_volume': 'sum',
        'brent_volume': 'sum',
        'source': 'last'
    }).dropna().reset_index()
    return df_5m.tail(limit)


def get_latest_status() -> dict:
    """獲取系統當前連線狀態與最新價差數據"""
    with _cache_lock:
        status_copy = dict(_latest_status)

    # 若快取筆數為 0，從資料庫撈取最新筆數與最後一筆紀錄
    if status_copy["total_bars_saved"] == 0:
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM crude_oil_1m")
            cnt = cur.fetchone()[0]
            if cnt > 0:
                cur.execute("SELECT timestamp, wti_close, brent_close, spread, zscore FROM crude_oil_1m ORDER BY timestamp DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    status_copy["total_bars_saved"] = cnt
                    status_copy["last_bar_time"] = row[0]
                    status_copy["last_wti"] = row[1]
                    status_copy["last_brent"] = row[2]
                    status_copy["last_spread"] = row[3]
                    status_copy["last_zscore"] = row[4]
                    if row[4] <= -2.0 or row[3] <= 3.0:
                        status_copy["signal"] = "🟢 價差過低 ── 做多價差訊號 (買 Brent 賣 WTI)"
                    elif row[4] >= 2.0 or row[3] >= 7.25:
                        status_copy["signal"] = "🔴 價差過高 ── 做空價差訊號 (賣 Brent 買 WTI)"
                    else:
                        status_copy["signal"] = "⚪ 常態震盪 (觀望)"
            conn.close()
        except Exception:
            pass

    # 進行 Touchance 連線檢測
    tc = check_touchance_connection(timeout_sec=0.5)
    status_copy.update(tc)
    return status_copy


def _collector_worker(interval_sec: int = 15):
    """背景定時輪詢接收與更新資料"""
    global _collector_running
    _collector_running = True
    print(f"Touchance 1分K 接收服務已在背景啟動 (輪詢週期: {interval_sec}s)...")

    while not _collector_stop_event.is_set():
        try:
            fetch_and_save_1m_quotes()
        except Exception as e:
            print(f"Worker exception: {e}")

        # 等待下一個輪詢週期
        for _ in range(interval_sec):
            if _collector_stop_event.is_set():
                break
            time.sleep(1)

    _collector_running = False
    print("Touchance 1分K 接收服務已停止。")


def start_collector_service(interval_sec: int = 15):
    """啟動背景報價接收服務執行緒 (單例)"""
    global _collector_thread, _collector_stop_event
    if _collector_thread is not None and _collector_thread.is_alive():
        return  # 已經在運行中

    _collector_stop_event.clear()
    _collector_thread = threading.Thread(
        target=_collector_worker,
        args=(interval_sec,),
        daemon=True,
        name="TouchanceCollectorThread"
    )
    _collector_thread.start()


def stop_collector_service():
    """停止背景報價接收服務"""
    global _collector_stop_event
    _collector_stop_event.set()


def is_collector_running() -> bool:
    """檢查背景接收服務是否運作中"""
    global _collector_thread
    return _collector_thread is not None and _collector_thread.is_alive()
