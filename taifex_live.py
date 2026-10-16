"""
taifex_live.py
期交所 (TAIFEX MIS) 股票期貨即時報價串接 + 配對盤中 Z 分數計算

資料來源: https://mis.taifex.com.tw/futures/api/getQuoteList (KindID=4 = 全部股票期貨)
一次 POST 即可取得所有股期標的之:
  - 現貨列   (SymbolID 以 '-S' 結尾, 如 'DFF-S')   -> 標的股票即時成交價
  - 期貨列   (SymbolID 以 '-F' 結尾, 如 'DFFJ6-F') -> 各月份股票期貨即時成交價
並以 SpotID (股票代號) 對應配對主檔中的 Ticker。
"""

import math
import datetime
import requests
import numpy as np
import pandas as pd

TAIFEX_QUOTE_URL = "https://mis.taifex.com.tw/futures/api/getQuoteList"
_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://mis.taifex.com.tw/futures/",
}
_MONTH_CODES = "ABCDEFGHIJKL"  # A=1月 ... L=12月


def _to_float(v):
    try:
        f = float(str(v).replace(",", ""))
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _quote_price(row):
    """成交價 > 買賣中價 > 參考價 (昨收)"""
    last = _to_float(row.get("CLastPrice"))
    if last:
        return last, "成交"
    bid, ask = _to_float(row.get("CBestBidPrice") or row.get("CBidPrice1")), _to_float(row.get("CBestAskPrice") or row.get("CAskPrice1"))
    if bid and ask:
        return (bid + ask) / 2.0, "買賣中價"
    ref = _to_float(row.get("CRefPrice"))
    if ref:
        return ref, "參考價"
    return None, ""


def _expiry_key(symbol_id):
    """'DFFJ6-F' -> (6, 10)，用於挑選最近月份"""
    base = symbol_id.split("-")[0]
    if len(base) >= 2 and base[-1].isdigit() and base[-2] in _MONTH_CODES:
        return (int(base[-1]), _MONTH_CODES.index(base[-2]) + 1)
    return (99, 99)


def fetch_taifex_stock_quotes(timeout=8):
    """
    取得期交所全部股票期貨 + 標的現貨即時報價。
    回傳 dict: { '2330': {spot, spot_src, spot_ref, fut, fut_symbol, fut_ref, fut_volume, time} }
    """
    body = {"MarketType": "0", "SymbolType": "F", "KindID": "4", "CID": "", "ExpireMonth": "",
            "RowSize": "全部", "PageNo": "", "SortColumn": "", "AscDesc": "A"}
    r = requests.post(TAIFEX_QUOTE_URL, json=body, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    rows = r.json().get("RtData", {}).get("QuoteList", []) or []

    today_str = datetime.datetime.now().strftime("%Y%m%d")
    quotes = {}
    fut_candidates = {}
    for row in rows:
        sid = str(row.get("SymbolID", ""))
        spot_id = str(row.get("SpotID", "")).strip()
        if not spot_id:
            continue
        q = quotes.setdefault(spot_id, {"spot": None, "spot_src": "", "spot_ref": None, "fut": None,
                                        "fut_symbol": "", "fut_ref": None, "fut_volume": 0, "time": ""})
        is_today = str(row.get("CDate", "")) == today_str
        if sid.endswith("-S"):
            px, src = _quote_price(row)
            q["spot"], q["spot_src"] = px, (src if is_today else "非今日")
            q["spot_ref"] = _to_float(row.get("CRefPrice"))
            t = str(row.get("CTime", ""))
            if t and is_today:
                q["time"] = f"{t[:2]}:{t[2:4]}:{t[4:6]}"
        elif sid.endswith("-F"):
            fut_candidates.setdefault(spot_id, []).append(row)

    # 每檔標的挑選「最近月份」中成交量最大的期貨 (排除小型股期與遠月干擾)
    for spot_id, cands in fut_candidates.items():
        nearest = min(_expiry_key(str(c.get("SymbolID", ""))) for c in cands)
        near_rows = [c for c in cands if _expiry_key(str(c.get("SymbolID", ""))) == nearest]
        best = max(near_rows, key=lambda c: _to_float(c.get("CTotalVolume")) or 0)
        px, _ = _quote_price(best)
        q = quotes[spot_id]
        q["fut"] = px
        q["fut_symbol"] = str(best.get("DispEName", best.get("SymbolID", "")))
        q["fut_ref"] = _to_float(best.get("CRefPrice"))
        q["fut_volume"] = int(_to_float(best.get("CTotalVolume")) or 0)
    return quotes


def _code(ticker):
    return str(ticker).replace(".TWO", "").replace(".TW", "")


SHARES_PER_CONTRACT = 2000  # 標準型股票期貨 1 口 = 2,000 股


def find_min_contracts(p1, p2, tolerance=0.10, max_contracts=10, *args, **kwargs):
    """
    依「雙邊市值差距在約 10% 內皆可接受」為原則，尋找「總口數最小」之最簡整數配對：
    1. 逐層遞增搜尋總口數 T = c1 + c2 (從 2 口開始：2, 3, 4, ...)。
    2. 一旦在某個最小總口數 T 下，存在互質整數 (c1, c2) 使雙邊市值相對誤差 diff <= tolerance (10%)，
       即判定為最優解（若同口數有多組符合，取誤差最小者）。
       - 避免無謂放大手數 (例如環球晶 901 vs 中美晶 198.5，1口配5口 市值差 9.2% <= 10%，不放大至 2口:9口)。
       - 避免粗暴 1:1 造成嚴重未對沖 (例如台塑化 122.5 vs 台塑 82.1 取 2口:3口 市值差 0.5%；瑞昱 778 vs 祥碩 1490 取 2口:1口 市值差 4.2%)。
    3. 若 10 口內無任何組合能在 10% 內，則取綜合誤差最小的最優解。
    """
    p1 = float(p1) if p1 and p1 > 0 else 1.0
    p2 = float(p2) if p2 and p2 > 0 else 1.0
    
    # 優先尋找在市值差 <= tolerance (10%) 內，總口數最小的最簡整數比
    for total in range(2, max_contracts * 2 + 1):
        candidates = []
        for c1 in range(1, total):
            c2 = total - c1
            if c1 > max_contracts or c2 > max_contracts:
                continue
            if math.gcd(c1, c2) > 1:
                continue
            v1, v2 = c1 * p1, c2 * p2
            diff = abs(v1 - v2) / max(v1, v2)
            if diff <= tolerance:
                candidates.append((diff, c1, c2))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1], candidates[0][2]

    # 保底：若 10% 內無解，取綜合誤差最小之解
    best_c1, best_c2 = 1, 1
    best_score = float('inf')
    for c1 in range(1, max_contracts + 1):
        for c2 in range(1, max_contracts + 1):
            if math.gcd(c1, c2) > 1:
                continue
            v1, v2 = c1 * p1, c2 * p2
            diff = abs(v1 - v2) / max(v1, v2)
            score = diff + (c1 + c2) * 0.015
            if score < best_score:
                best_score = score
                best_c1, best_c2 = c1, c2
    return best_c1, best_c2


def _parse_contracts(s):
    """'3 : 2' -> (3, 2)；第一個數字對應 Ticker1 (A)、第二個對應 Ticker2 (B)"""
    try:
        a, b = str(s).split(":")
        return max(int(float(a)), 1), max(int(float(b)), 1)
    except Exception:
        return 1, 1


def _sim_position(z, p1, p2, dates, z_entry, z_exit):
    """
    依回測規則 (±z_entry 進場、回到 z_exit 平倉、不設停損) 重播收盤價。
    回傳截至最後一根的持倉 (1 多價差 / -1 空價差 / 0 空手) 與該筆開倉資訊。
    """
    pos, entry = 0, None
    for i, v in enumerate(z):
        if np.isnan(v):
            continue
        if pos == 0:
            if v <= -z_entry or v >= z_entry:
                pos = 1 if v <= -z_entry else -1
                entry = {"EntryDate": dates[i].strftime("%Y-%m-%d"), "EntryP1": float(p1[i]),
                         "EntryP2": float(p2[i]), "EntryZ": float(v)}
        elif (pos == 1 and v >= -z_exit) or (pos == -1 and v <= z_exit):
            pos, entry = 0, None
    return pos, entry


def prepare_pair_baselines(df_pairs, prices_df, z_entry=2.0, z_exit=0.0):
    """
    盤前基準 (只用「今天以前」的收盤價，杜絕盤中資料混入)：
    前日收盤價、前收價差、前收 Z、回測規則下的模擬持倉、計算即時 Z 所需的最近 (w-1) 筆價差。
    """
    today = pd.Timestamp(datetime.date.today())
    out = []
    for _, r in df_pairs.iterrows():
        t1, t2 = str(r["Ticker1"]), str(r["Ticker2"])
        if t1 not in prices_df.columns or t2 not in prices_df.columns:
            continue
        w = int(r.get("Window", 20))
        hr = float(r.get("HedgeRatio", 1.0))
        px = prices_df[[t1, t2]].dropna()
        px = px[px.index < today]
        if len(px) < w + 1:
            continue
        spread = px[t2] - hr * px[t1]
        mean = spread.rolling(w).mean()
        std = spread.rolling(w).std()
        z = ((spread - mean) / std).replace([np.inf, -np.inf], np.nan)
        mask = z.index >= "2020-01-01"
        pos, entry = _sim_position(z[mask].values, px[t1][mask].values, px[t2][mask].values,
                                   z.index[mask], z_entry, z_exit)
        c1, c2 = _parse_contracts(r.get("BaseContracts", "1 : 1"))
        out.append({
            "SubIndustry": r.get("SubIndustry", ""), "Ticker1": t1, "Name1": r["Name1"], "Ticker2": t2, "Name2": r["Name2"],
            "Window": w, "HedgeRatio": hr, "BaseContracts": r.get("BaseContracts", ""), "BaseMargin": r.get("BaseMargin", np.nan),
            "C1": c1, "C2": c2,
            "ProfitFactor": r.get("ProfitFactor", np.nan), "WinRate": r.get("WinRate", np.nan),
            "AvgConvergenceDays": r.get("AvgConvergenceDays", np.nan),
            "PrevDate": px.index[-1].strftime("%Y-%m-%d"),
            "PrevClose1": float(px[t1].iloc[-1]), "PrevClose2": float(px[t2].iloc[-1]),
            "PrevSpread": float(spread.iloc[-1]), "PrevMean": float(mean.iloc[-1]), "PrevStd": float(std.iloc[-1]),
            "PrevZ": float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else 0.0,
            "Position": pos,
            "EntryDate": entry["EntryDate"] if entry else None,
            "EntryP1": entry["EntryP1"] if entry else None,
            "EntryP2": entry["EntryP2"] if entry else None,
            "EntryZ": entry["EntryZ"] if entry else None,
            "SpreadTail": spread.iloc[-(w - 1):].values.astype(float),
        })
    return out


def compute_live_board(baselines, quotes, price_basis="spot", z_entry=2.0, z_exit=0.0, z_watch=1.5):
    """
    以期交所即時報價重算每組配對的價差 / Z 分數 / 開平倉訊號，
    並計算最小配對 1 組之開倉價格、平倉價格、已實現/未實現損益與報酬率，
    最後依 |即時 Z| 由大到小排序。
    """
    rows = []
    for b in baselines:
        q1 = quotes.get(_code(b["Ticker1"]), {})
        q2 = quotes.get(_code(b["Ticker2"]), {})
        key = "fut" if price_basis == "fut" else "spot"
        l1, l2 = q1.get(key), q2.get(key)
        live_ok = bool(l1 and l2)
        p1 = l1 if live_ok else b["PrevClose1"]
        p2 = l2 if live_ok else b["PrevClose2"]

        live_spread = p2 - b["HedgeRatio"] * p1
        window_vals = np.append(b["SpreadTail"], live_spread)
        mu = float(np.mean(window_vals))
        sd = float(np.std(window_vals, ddof=1))
        live_z = (live_spread - mu) / sd if sd > 1e-9 else 0.0

        pos = b["Position"]
        # 判斷訊號
        if pos == 1:
            signal = "✅ 達平倉門檻 (多價差 Z回歸停利)" if live_z >= -z_exit else "📌 持有多價差中"
        elif pos == -1:
            signal = "✅ 達平倉門檻 (空價差 Z回歸停利)" if live_z <= z_exit else "📌 持有空價差中"
        elif live_z <= -z_entry:
            signal = "🔥 達開倉門檻：做多價差 (買B賣A)"
        elif live_z >= z_entry:
            signal = "🔥 達開倉門檻：放空價差 (賣B買A)"
        elif abs(live_z) >= z_watch:
            signal = "⏳ 接近開倉門檻"
        else:
            signal = "⚪ 觀望"

        # ----------------------------------------------------
        # 開倉價格、平倉價格與損益計算 (最小配對 1 組)
        # 依當前市價動態計算等合約總值 (Dollar Neutrality) 最適口數
        # 徹底解決歷史 CSV 殘留多年前股價導致口數嚴重失衡、未等額對沖之問題
        # ----------------------------------------------------
        # 若有歷史開倉紀錄 (持倉中或平倉)
        if pos != 0 and b.get("EntryP1") is not None and b.get("EntryP2") is not None:
            entry_p1 = b["EntryP1"]
            entry_p2 = b["EntryP2"]
            entry_z = b["EntryZ"]
            entry_date = b["EntryDate"]
            entry_spread = entry_p2 - b["HedgeRatio"] * entry_p1
            is_holding = True
            
            # 以進場價計算當時開倉口數與保證金
            c1, c2 = find_min_contracts(entry_p1, entry_p2)
            val1 = p1 * c1 * SHARES_PER_CONTRACT
            val2 = p2 * c2 * SHARES_PER_CONTRACT
            val_diff_pct = abs(val1 - val2) / max(val1, val2) * 100 if max(val1, val2) > 0 else 0.0
            base_margin = (entry_p1 * c1 + entry_p2 * c2) * SHARES_PER_CONTRACT * 0.135
            
            # 多價差 (pos=1): 買 B 賣 A
            if pos == 1:
                pnl_a = (entry_p1 - p1) * c1 * SHARES_PER_CONTRACT  # 放空 A 獲利 = 進場價 - 現價
                pnl_b = (p2 - entry_p2) * c2 * SHARES_PER_CONTRACT  # 做多 B 獲利 = 現價 - 進場價
                exp_room = mu - live_spread
            # 空價差 (pos=-1): 賣 B 買 A
            else:
                pnl_a = (p1 - entry_p1) * c1 * SHARES_PER_CONTRACT  # 做多 A 獲利 = 現價 - 進場價
                pnl_b = (entry_p2 - p2) * c2 * SHARES_PER_CONTRACT  # 放空 B 獲利 = 進場價 - 現價
                exp_room = live_spread - mu
                
            total_pnl = pnl_a + pnl_b
            pnl_pct = (total_pnl / base_margin * 100) if base_margin > 0 else 0.0
            exp_pnl = exp_room * c2 * SHARES_PER_CONTRACT
            exp_roi = (exp_pnl / base_margin * 100) if base_margin > 0 else 0.0
        
        # 若無持倉但達到今日開倉門檻 (新進場機會)
        elif signal.startswith("🔥"):
            entry_p1 = p1
            entry_p2 = p2
            entry_z = live_z
            entry_date = "今日盤中觸發"
            entry_spread = live_spread
            is_holding = False
            pnl_a = 0.0
            pnl_b = 0.0
            total_pnl = 0.0
            pnl_pct = 0.0
            
            # 以當前即時市價計算最優配置口數
            c1, c2 = find_min_contracts(p1, p2)
            val1 = p1 * c1 * SHARES_PER_CONTRACT
            val2 = p2 * c2 * SHARES_PER_CONTRACT
            val_diff_pct = abs(val1 - val2) / max(val1, val2) * 100 if max(val1, val2) > 0 else 0.0
            base_margin = (val1 + val2) * 0.135
            
            if live_z <= -z_entry:
                exp_room = mu - live_spread  # 做多價差預期擴大/回升空間
            else:
                exp_room = live_spread - mu  # 放空價差預期收斂空間
            exp_pnl = exp_room * c2 * SHARES_PER_CONTRACT
            exp_roi = (exp_pnl / base_margin * 100) if base_margin > 0 else 0.0
            
        else:
            entry_p1 = None
            entry_p2 = None
            entry_z = None
            entry_date = None
            entry_spread = None
            is_holding = False
            pnl_a = None
            pnl_b = None
            total_pnl = None
            pnl_pct = None
            
            c1, c2 = find_min_contracts(p1, p2)
            val1 = p1 * c1 * SHARES_PER_CONTRACT
            val2 = p2 * c2 * SHARES_PER_CONTRACT
            val_diff_pct = abs(val1 - val2) / max(val1, val2) * 100 if max(val1, val2) > 0 else 0.0
            base_margin = (val1 + val2) * 0.135
            
            exp_room = abs(live_spread - mu)
            exp_pnl = exp_room * c2 * SHARES_PER_CONTRACT
            exp_roi = (exp_pnl / base_margin * 100) if base_margin > 0 else 0.0

        rows.append({
            **{k: v for k, v in b.items() if k != "SpreadTail"},
            "C1": c1,
            "C2": c2,
            "Val1": val1,
            "Val2": val2,
            "Val1W": val1 / 10000.0,
            "Val2W": val2 / 10000.0,
            "ValDiffPct": val_diff_pct,
            "Live1": l1,
            "Live2": l2,
            "Fut1": q1.get("fut"),
            "Fut2": q2.get("fut"),
            "FutSym1": q1.get("fut_symbol", ""),
            "FutSym2": q2.get("fut_symbol", ""),
            "Chg1": (p1 / b["PrevClose1"] - 1) * 100,
            "Chg2": (p2 / b["PrevClose2"] - 1) * 100,
            "QuoteTime": q1.get("time") or q2.get("time") or "",
            "LiveOK": live_ok,
            "LiveSpread": live_spread,
            "LiveMean": mu,
            "LiveStd": sd,
            "LiveZ": live_z,
            "DeltaZ": live_z - b["PrevZ"],
            "EntryLong": mu - z_entry * sd,
            "EntryShort": mu + z_entry * sd,
            "Signal": signal,
            "IsHolding": is_holding,
            "EntryDate": entry_date,
            "EntrySpread": entry_spread,
            "EntryP1": entry_p1,
            "EntryP2": entry_p2,
            "EntryZ": entry_z,
            "ExitP1": p1,
            "ExitP2": p2,
            "PnLA": pnl_a,
            "PnLB": pnl_b,
            "TotalPnL": total_pnl,
            "PnLPct": pnl_pct,
            "ExpRoom": exp_room,
            "ExpPnL": exp_pnl,
            "ExpROI": exp_roi,
            "RealMargin": base_margin,
        })
        
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["AbsZ"] = df["LiveZ"].abs()
    df = df.sort_values("AbsZ", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df

