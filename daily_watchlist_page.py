"""
Daily Convergence Watchlist Page (每日精選收斂配對監控看板)
=========================================================
100% 仿照回測頁面之高階機構級 UI 樣式：
1. 頂部 Main Header 與 4 大 Metric Cards (同回測主題 CSS)
2. 完整條件快篩與 PF > 2.0 排序資料總表
3. 單一配對展開深度剖析：
   - 🏢 兩檔標的產品營收結構深度對比卡片 (MOPS / CMoney 月產銷組合)
   - ⚡ 產業同質性與走勢連動診斷 Banner (diag-banner-green / yellow / red)
   - 📋 配對口數與即時保證金計算 (4 欄 Metric Cards: β, 口數, 保證金, 當前Z分數 + 📐 詳細計算 Expander)
   - 📊 歷史回測績效與收斂週期統計 (12 欄 Metric Cards: PF, 勝率, 交易次數, MDD, 年化報酬, 夏普比率, 平均/最短/最長收斂, 協整p值，2020年至今)
   - 🎯 今日進場與停利實單操作指示 Banner (零空白縮排純 HTML，杜絕程式碼方塊逃逸)
   - 📈 全頁直式展開四大圖表與完整明細 (2020年1月1日至今，全平鋪在同一頁，無需分頁點選)：
     1. 價差走勢與 ±2σ 通道 (含 Z-Score 走勢)
     2. 兩檔股價歷史走勢對比 (雙軸)
     3. 歷史回測累計報酬曲線 (Equity Curve)
     4. 歷史收斂交易明細全記錄 (2020年1月1日至今所有交易完整展示)
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import pure_backtest
import taifex_live

CSV_DEDUP_PATH = 'subindustry_pf2_convergence_dedup.csv'
CSV_MASTER_PATH = 'subindustry_pf2_convergence_master.csv'
INDUSTRY_PATH = 'industry_data.csv'

def render_metric_card(label: str, value: str, status: str = "") -> str:
    """產生標準回測樣式 Metric Card (單行字串，杜絕 Markdown 縮排解析錯誤)"""
    color_cls = f" {status}" if status else ""
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value{color_cls}">{value}</div></div>'

@st.cache_data(ttl=60)
def load_convergence_data():
    """載入掃描結果資料庫"""
    target_path = CSV_DEDUP_PATH if os.path.exists(CSV_DEDUP_PATH) else CSV_MASTER_PATH
    if not os.path.exists(target_path):
        return None
    try:
        df = pd.read_csv(target_path)
        return df
    except Exception:
        return None

@st.cache_data(ttl=300)
def load_industry_lookup():
    """載入產業與營收結構字典"""
    if not os.path.exists(INDUSTRY_PATH):
        return None
    try:
        df = pd.read_csv(INDUSTRY_PATH).set_index('Ticker')
        return df
    except Exception:
        return None

@st.cache_data(ttl=600)
def run_single_pair_backtest_2020(s1_series, s2_series, sym1, sym2, name1, name2, window):
    """自 2020-01-01 起算之單一配對精確回測"""
    try:
        comm = s1_series.index.intersection(s2_series.index)
        comm = comm[comm >= '2020-01-01']
        if len(comm) < window + 20:
            return None, None, None
            
        s1_sub = s1_series.loc[comm]
        s2_sub = s2_series.loc[comm]
        
        df_rec, df_trades, stats = pure_backtest.run_backtest(
            s1_sub, s2_sub, sym1, sym2, name1, name2,
            initial_capital=1000000,
            selected_models=['Z-Score (標準)'],
            model_params={'Z-Score (標準)': {'z_entry': 2.0, 'z_exit': 0.0, 'z_window': window}},
            size_mode='最少配對口數 (僅基本單位)',
            margin_usage_pct=1.0,
            run_start_date='2020-01-01',
            advanced_params={'fixed_pair_multiplier': 1, 'trade_mode': '收斂 (均值回歸)', 'use_grid': False, 'enable_stop_loss': False}
        )
        return df_rec, df_trades, stats
    except Exception:
        return None, None, None

def get_stock_profile(sym, industry_df):
    """取得標的的營收產品比重與業務背景 (相容 app.py)"""
    clean_code = sym.replace('.TWO', '').replace('.TW', '')
    default_prof = {
        'ticker': sym,
        'code': clean_code,
        'name': clean_code,
        'market': '上市' if sym.endswith('.TW') else '上櫃',
        'category': '一般產業',
        'sub_industry': '一般同業',
        'industry': '未分類',
        'primary_product': '主要產品',
        'primary_ratio': 100.0,
        'product_breakdown': '',
        'business': '台灣期交所掛牌之股票期貨標的。',
        'products': []
    }
    
    if industry_df is None:
        return default_prof
        
    row = None
    if sym in industry_df.index:
        row = industry_df.loc[sym]
    else:
        matches = industry_df[industry_df['Code'].astype(str) == clean_code]
        if not matches.empty:
            row = matches.iloc[0]
            
    if row is not None:
        prods = []
        pj = row.get('ProductsJSON')
        if pd.notna(pj) and str(pj).strip():
            try:
                p_list = json.loads(str(pj))
                for p in p_list:
                    if isinstance(p, dict):
                        prods.append((str(p.get('item', '')), float(p.get('ratio', 0.0))))
            except Exception:
                pass
        if not prods:
            pb = row.get('ProductBreakdown')
            if pd.notna(pb) and str(pb).strip():
                parts = str(pb).split('|')
                for part in parts:
                    part = part.strip()
                    if part:
                        sub_parts = part.rsplit(' ', 1)
                        if len(sub_parts) == 2 and sub_parts[1].endswith('%'):
                            try:
                                prods.append((sub_parts[0], float(sub_parts[1].replace('%', ''))))
                            except Exception:
                                prods.append((part, 0.0))
                        else:
                            prods.append((part, 0.0))
        p_name = str(row.get('PrimaryProduct', '主要業務')) if pd.notna(row.get('PrimaryProduct')) else '主要業務'
        p_ratio = float(row.get('PrimaryRatio', 100.0)) if pd.notna(row.get('PrimaryRatio')) else 100.0
        
        return {
            'ticker': sym,
            'code': clean_code,
            'name': str(row.get('Name', clean_code)),
            'market': str(row.get('Market', '上市')),
            'category': str(row.get('Category', '一般產業')),
            'sub_industry': str(row.get('SubIndustry', '一般同業')),
            'industry': str(row.get('Industry', '')),
            'primary_product': p_name,
            'primary_ratio': p_ratio,
            'product_breakdown': str(row.get('ProductBreakdown', '')),
            'business': str(row.get('Business', '無詳細說明')),
            'products': prods if prods else [(p_name, p_ratio)]
        }
    return default_prof

def render_stock_card_html(prof):
    """渲染與回測頁面完全一致的標的營收結構卡片"""
    prods = prof.get('products', [])
    html_rows = []
    if prods:
        for item, ratio in prods[:6]:
            bar_pct = min(100.0, max(0.0, float(ratio)))
            html_rows.append(
                f'<div class="rev-row">'
                f'<div class="rev-name" title="{item}">{item}</div>'
                f'<div class="rev-bar-bg"><div class="rev-bar-fill" style="width: {bar_pct:.1f}%;"></div></div>'
                f'<div class="rev-pct">{ratio:.1f}%</div>'
                f'</div>'
            )
        rev_html = "".join(html_rows)
    else:
        p_name = prof.get('primary_product', '未提供細項')
        p_ratio = prof.get('primary_ratio', 100.0)
        rev_html = (
            f'<div class="rev-row">'
            f'<div class="rev-name" title="{p_name}">{p_name}</div>'
            f'<div class="rev-bar-bg"><div class="rev-bar-fill" style="width: 100%;"></div></div>'
            f'<div class="rev-pct">{p_ratio:.0f}%</div>'
            f'</div>'
        )
        
    mkt_tag = f"<span class='cat-badge'>{prof['market']}</span>"
    clean_sym = prof['ticker'].replace('.TWO', '').replace('.TW', '')
    
    return (
        f'<div class="industry-panel">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem; flex-wrap:wrap; gap:0.4rem;">'
        f'<div>'
        f'<span style="font-size:1.25rem; font-weight:700; color:#fff;">{clean_sym} {prof["name"]}</span>'
        f'<span style="font-size:0.85rem; color:#94a3b8; margin-left:4px;">({prof["ticker"]})</span> '
        f'{mkt_tag}'
        f'</div>'
        f'<div>'
        f'<span class="subind-badge">{prof["sub_industry"]}</span>'
        f'<span class="cat-badge">{prof["category"]}</span>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:0.86rem; color:#94a3b8; line-height:1.55; margin-bottom:1rem; border-left:3px solid #64ffda; padding-left:0.6rem;">'
        f'{prof["business"]}'
        f'</div>'
        f'<div style="font-weight:600; font-size:0.88rem; color:#ccd6f6; margin-bottom:0.4rem;">'
        f'📊 實質產品營收佔比結構 (MOPS / CMoney 月產銷組合)'
        f'</div>'
        f'{rev_html}'
        f'</div>'
    )

def render_co_movement_banner_html(prof1, prof2, price_corr=None, ret_corr=None, coint_p=None, hl_str=None):
    """渲染與回測頁面完全一致的同質性連動診斷 Banner"""
    is_same_sub = (prof1['sub_industry'] == prof2['sub_industry']) and (prof1['sub_industry'] not in ['其他', '一般產業'])
    is_same_cat = (prof1['category'] == prof2['category']) and (prof1['category'] not in ['其他', '其他資產'])
    
    if is_same_sub:
        banner_cls = "diag-banner-green"
        tag_text = f"🔥 同細產業·高度共振配對 (同屬【{prof1['sub_industry']}】)"
        msg = f"兩檔標的同屬【{prof1['sub_industry']}】，核心業務產品（{prof1['name']}: {prof1['primary_product']} vs {prof2['name']}: {prof2['primary_product']}）及上下游成本驅動高度同質，基本面價差均值回歸可靠度極高！"
    elif is_same_cat:
        banner_cls = "diag-banner-yellow"
        tag_text = f"⚡ 同大類產業連動配對 (同屬【{prof1['category']}】)"
        msg = f"兩檔標的同屬大類【{prof1['category']}】，但在細分產品領域不同（{prof1['sub_industry']} vs {prof2['sub_industry']}）。受相同宏觀景氣循環驅動，但需留意個別次產業產品週期之分歧。"
    else:
        banner_cls = "diag-banner-red"
        tag_text = f"⚠️ 跨產業標的 (發散風險警示)"
        msg = f"警告：兩檔標的分屬不同產業（{prof1['sub_industry']} vs {prof2['sub_industry']}），缺乏共同實質營收與成本驅動因子！"

    metrics_html = []
    if price_corr is not None:
        metrics_html.append(f"<div>📈 <b>價格走勢相關度:</b> <span style='font-size:1rem; font-weight:700;'>{price_corr:+.3f}</span></div>")
    if ret_corr is not None:
        metrics_html.append(f"<div>⚡ <b>日報酬相關係數:</b> <span style='font-size:1rem; font-weight:700;'>{ret_corr:+.3f}</span></div>")
    if coint_p is not None:
        coint_display = f"{coint_p:.4f} (通過協整 ✅)" if coint_p < 0.05 else f"{coint_p:.4f} (未顯著 ⚠️)"
        metrics_html.append(f"<div>🔬 <b>共整合檢驗 (p-value):</b> <span style='font-size:1rem; font-weight:700;'>{coint_display}</span></div>")
    if hl_str:
        metrics_html.append(f"<div>⏱️ <b>平均收斂週期:</b> <span style='font-size:1rem; font-weight:700;'>{hl_str}</span></div>")

    metrics_section = ""
    if metrics_html:
        metrics_section = (
            f'<div style="display:flex; flex-wrap:wrap; gap:1.2rem; font-size:0.85rem; padding-top:0.4rem; border-top:1px dashed rgba(255,255,255,0.2);">'
            f'{"".join(metrics_html)}'
            f'</div>'
        )

    return (
        f'<div class="{banner_cls}">'
        f'<div style="font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;">{tag_text}</div>'
        f'<div style="font-size:0.9rem; line-height:1.5; margin-bottom:0.6rem; opacity:0.95;">{msg}</div>'
        f'{metrics_section}'
        f'</div>'
    )

def update_watchlist_with_live_prices(df_data, prices_df):
    """
    使用最新的價格資料庫 (prices_df) 動態更新每組配對的當前最新價格、價差與 Z-Score。
    """
    if prices_df is None or prices_df.empty or df_data is None or df_data.empty:
        return df_data, None

    updated_df = df_data.copy()
    latest_date_str = str(prices_df.index[-1])[:10] if hasattr(prices_df.index, 'strftime') else str(prices_df.index[-1])
    
    # 遍歷每組配對更新
    for idx, row in updated_df.iterrows():
        s1_tk = str(row['Ticker1'])
        s2_tk = str(row['Ticker2'])
        w = int(row.get('Window', 20))
        
        if s1_tk in prices_df.columns and s2_tk in prices_df.columns:
            s1 = prices_df[s1_tk].dropna()
            s2 = prices_df[s2_tk].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) >= w:
                s1_sub = s1.loc[common]
                s2_sub = s2.loc[common]
                p1_latest = float(s1_sub.iloc[-1])
                p2_latest = float(s2_sub.iloc[-1])
                
                hr = float(row.get('HedgeRatio', 1.0))
                if hr == 0.0 or pd.isna(hr):
                    X = sm.add_constant(s1_sub)
                    ols_fit = sm.OLS(s2_sub, X).fit()
                    hr = float(ols_fit.params.iloc[1])
                
                spread = s2_sub - hr * s1_sub
                spread_mean = spread.rolling(window=w).mean()
                spread_std = spread.rolling(window=w).std()
                
                lat_spread = float(spread.iloc[-1])
                lat_mean = float(spread_mean.iloc[-1])
                lat_std = float(spread_std.iloc[-1]) if pd.notna(spread_std.iloc[-1]) else 0.0
                
                cur_z = (lat_spread - lat_mean) / lat_std if lat_std > 1e-6 else 0.0
                
                # 更新數值
                updated_df.at[idx, 'LatestPrice1'] = p1_latest
                updated_df.at[idx, 'LatestPrice2'] = p2_latest
                updated_df.at[idx, 'CurrentSpread'] = round(lat_spread, 2)
                updated_df.at[idx, 'SpreadMean'] = round(lat_mean, 2)
                updated_df.at[idx, 'SpreadStd'] = round(lat_std, 2)
                updated_df.at[idx, 'CurrentZScore'] = round(cur_z, 2)
                
                # 操作建議
                if cur_z >= 2.0:
                    action = '🔥 放空價差收斂 (賣B買A)'
                    room = lat_spread - lat_mean
                    target_tp = f"價差回歸至 {lat_mean:.2f} (Z=0, 空間 +{room:.2f} 點)"
                elif cur_z <= -2.0:
                    action = '🔥 做多價差收斂 (買B賣A)'
                    room = lat_mean - lat_spread
                    target_tp = f"價差回歸至 {lat_mean:.2f} (Z=0, 空間 +{room:.2f} 點)"
                elif cur_z >= 1.2:
                    action = '⏳ 價差偏高 (接近進場門檻)'
                    room = lat_spread - lat_mean
                    target_tp = f"目標中樞 {lat_mean:.2f} (目前偏離 +{room:.2f} 點)"
                elif cur_z <= -1.2:
                    action = '⏳ 價差偏低 (接近進場門檻)'
                    room = lat_mean - lat_spread
                    target_tp = f"目標中樞 {lat_mean:.2f} (目前偏離 -{room:.2f} 點)"
                else:
                    action = '🎯 常態區間 (Z接近0)'
                    target_tp = f"常態中樞 {lat_mean:.2f} (Z={cur_z:.2f})"
                    
                updated_df.at[idx, 'ActionRecommendation'] = action
                updated_df.at[idx, 'TakeProfitTarget'] = target_tp

    return updated_df, latest_date_str

@st.cache_data(ttl=3, show_spinner=False)
def _cached_taifex_quotes():
    return taifex_live.fetch_taifex_stock_quotes()


def render_signal_card(r):
    """
    使用純原生 Streamlit 卡片 (st.container, st.metric, st.success, st.warning)，
    徹底杜絕 HTML 縮排被 Markdown 解析器誤判為程式碼區塊 (<pre><code>) 的問題，
    呈現完全無代碼框、字體超清晰、主題適配的高階視覺化實單指引。
    """
    sig = str(r["Signal"])
    is_open = sig.startswith("🔥")
    is_close = sig.startswith("✅")
    n1, n2 = r["Name1"], r["Name2"]
    sym1, sym2 = taifex_live._code(r["Ticker1"]), taifex_live._code(r["Ticker2"])
    c1, c2 = int(r.get("C1", 1)), int(r.get("C2", 1))
    margin_w = (r.get("RealMargin") or 0.0) / 10000.0
    val1_w = float(r.get("Val1W") or (r['ExitP1'] * c1 * 2000 / 10000.0))
    val2_w = float(r.get("Val2W") or (r['ExitP2'] * c2 * 2000 / 10000.0))
    val_diff = float(r.get("ValDiffPct") or 0.0)

    with st.container(border=True):
        st.markdown(f"#### {sig} ｜ **{n1} ({sym1}) vs {n2} ({sym2})**")
        st.caption(
            f"細產業：**{r['SubIndustry']}** ｜ 窗口：**{r['Window']}日** ｜ "
            f"最小配置：**{n1}×{c1}口 ({val1_w:.1f}萬) : {n2}×{c2}口 ({val2_w:.1f}萬)** ｜ "
            f"原始保證金：約 **{margin_w:.1f} 萬** ｜ 報價時間：{r.get('QuoteTime', '')}"
        )

        if is_close:
            pnl = r.get("TotalPnL") or 0.0
            pnl_pct = r.get("PnLPct") or 0.0
            pnl_a = r.get("PnLA") or 0.0
            pnl_b = r.get("PnLB") or 0.0
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            with c_m1:
                st.metric(
                    label=f"📌 歷史開倉價 ({r.get('EntryDate', '前日')})",
                    value=f"{r['EntryP1']:.2f} / {r['EntryP2']:.2f}",
                    delta=f"開倉價差 {r.get('EntrySpread', 0):.2f} (Z={r.get('EntryZ', 0):+.2f})"
                )
            with c_m2:
                st.metric(
                    label="🏁 今日即時平倉價",
                    value=f"{r['ExitP1']:.2f} / {r['ExitP2']:.2f}",
                    delta=f"即時價差 {r['LiveSpread']:.2f} (Z={r['LiveZ']:+.2f})"
                )
            with c_m3:
                st.metric(
                    label="💰 1組平倉總損益",
                    value=f"{pnl:+,.0f} 元",
                    delta=f"報酬率 {pnl_pct:+.2f}%"
                )
            with c_m4:
                st.metric(
                    label=f"各腿損益 ({n1} : {n2})",
                    value=f"{pnl_a:+,.0f} / {pnl_b:+,.0f}"
                )
            st.success(f"🎯 **操作指示**：價差已回歸停利中樞 (Z 回到 0)，建議實單以即時價 **{n1}: {r['ExitP1']:.2f} / {n2}: {r['ExitP2']:.2f}**，將 **{n1}×{c1}口 與 {n2}×{c2}口** 全數平倉停利出場！")

        elif is_open:
            exp_pnl = r.get("ExpPnL") or 0.0
            exp_roi = r.get("ExpROI") or 0.0
            exp_room = r.get("ExpRoom") or 0.0
            action_text = sig.split("：")[-1] if "：" in sig else sig
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            with c_m1:
                st.metric(
                    label="🎯 建議開倉價 (A / B)",
                    value=f"{r['ExitP1']:.2f} / {r['ExitP2']:.2f}",
                    delta=f"即時價差 {r['LiveSpread']:.2f}"
                )
            with c_m2:
                st.metric(
                    label="🎯 目標停利中樞 (Z=0)",
                    value=f"{r['LiveMean']:.2f}",
                    delta=f"預期收斂空間 +{exp_room:.2f} 點"
                )
            with c_m3:
                st.metric(
                    label="🚀 預估 1 組收斂獲利",
                    value=f"+{exp_pnl:,.0f} 元",
                    delta=f"預估報酬率 +{exp_roi:.1f}%"
                )
            with c_m4:
                st.metric(
                    label="即時 Z 分數偏離",
                    value=f"{r['LiveZ']:+.2f}",
                    delta=f"前收基準 Z: {r['PrevZ']:+.2f}"
                )
            st.warning(f"⚡ **實單下單動作**：**{action_text}** ｜ 下單配置：**{n1}×{c1}口 ＋ {n2}×{c2}口**（市值：**{val1_w:.1f}萬 vs {val2_w:.1f}萬**）｜ 原始保證金需求約 **{margin_w:.1f} 萬**。")

        else:
            pnl = r.get("TotalPnL") or 0.0
            pnl_pct = r.get("PnLPct") or 0.0
            pnl_a = r.get("PnLA") or 0.0
            pnl_b = r.get("PnLB") or 0.0
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            with c_m1:
                st.metric(
                    label=f"📌 歷史開倉價 ({r.get('EntryDate', '')})",
                    value=f"{r['EntryP1']:.2f} / {r['EntryP2']:.2f}"
                )
            with c_m2:
                st.metric(
                    label="🏁 今日即時價",
                    value=f"{r['ExitP1']:.2f} / {r['ExitP2']:.2f}",
                    delta=f"即時Z {r['LiveZ']:+.2f}"
                )
            with c_m3:
                st.metric(
                    label="💰 1組未實現損益",
                    value=f"{pnl:+,.0f} 元",
                    delta=f"{pnl_pct:+.2f}%"
                )
            with c_m4:
                st.metric(
                    label=f"各腿損益 ({n1} : {n2})",
                    value=f"{pnl_a:+,.0f} / {pnl_b:+,.0f}"
                )
            st.info(f"⏳ **持倉中**：目前價差 {r['LiveSpread']:.2f}，距離目標停利中樞 {r['LiveMean']:.2f} (Z=0) 尚有 {r['ExpRoom']:.2f} 點空間。")



def _style_live_board(view_df):
    """即時排行榜配色：文字深飽和高對比，避免淺色主題下模糊不清"""
    def z_color(v):
        try:
            z = float(v)
        except (TypeError, ValueError):
            return ""
        if abs(z) < 1.0:
            return ""
        # 柔和背景色搭配清晰深色文字
        if z > 0:
            return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
        else:
            return "background-color: #dcfce7; color: #166534; font-weight: 700;"

    def sig_color(v):
        s = str(v)
        if s.startswith("🔥"):
            return "background-color: #ffedd5; color: #9a3412; font-weight: 800;"
        if s.startswith("✅"):
            return "background-color: #d1fae5; color: #065f46; font-weight: 800;"
        if s.startswith("📌"):
            return "background-color: #e0f2fe; color: #0369a1; font-weight: 700;"
        if s.startswith("⏳"):
            return "background-color: #fef9c3; color: #854d0e; font-weight: 700;"
        return ""

    def pnl_color(v):
        s = str(v)
        if "預估" in s or "預期" in s:
            return "color: #0284c7; font-weight: 800;"
        if s.startswith("+"):
            return "color: #16a34a; font-weight: 800;"
        if s.startswith("-"):
            return "color: #dc2626; font-weight: 800;"
        return ""

    styler = view_df.style
    styler = styler.map(z_color, subset=["即時Z", "前收Z"]) if hasattr(styler, "map") else styler.applymap(z_color, subset=["即時Z", "前收Z"])
    styler = styler.map(sig_color, subset=["盤中訊號"]) if hasattr(styler, "map") else styler.applymap(sig_color, subset=["盤中訊號"])
    styler = styler.map(pnl_color, subset=["1組損益(元)", "損益率%"]) if hasattr(styler, "map") else styler.applymap(pnl_color, subset=["1組損益(元)", "損益率%"])
    return styler


def render_live_board(df_data, prices_df):
    """⚡ 盤中即時 Z 分數排行榜 (期交所 MIS 即時報價)"""
    st.markdown("## ⚡ 盤中即時 Z 分數排行榜（期交所即時報價）")
    if prices_df is None or prices_df.empty:
        st.warning("缺少歷史收盤價資料，無法計算即時 Z 分數。")
        return

    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.2, 1.6])
    with c1:
        basis = st.radio("Z 分數計價基準", ["現貨即時價 (與回測一致)", "近月股期即時價"], index=0, horizontal=True, key="live_basis")
    with c2:
        refresh_label = st.radio("自動更新頻率", ["5 秒", "10 秒", "30 秒", "暫停"], index=1, horizontal=True, key="live_refresh")
    with c3:
        z_entry = st.number_input("開倉門檻 |Z| ≥", value=2.0, step=0.1, min_value=0.5, key="live_z_entry")
    with c4:
        only_signal = st.checkbox("只顯示達開倉 / 平倉門檻的組合", value=False, key="live_only_signal")

    baselines = taifex_live.prepare_pair_baselines(df_data, prices_df, z_entry=z_entry, z_exit=0.0)
    if not baselines:
        st.warning("配對池內的標的在歷史價格資料中找不到，無法建立盤前基準。")
        return

    price_basis = "fut" if "期" in basis else "spot"
    run_every = {"5 秒": 5, "10 秒": 10, "30 秒": 30}.get(refresh_label)

    @st.fragment(run_every=run_every)
    def _live_fragment():
        try:
            quotes = _cached_taifex_quotes()
            err = None
        except Exception as e:
            quotes, err = {}, str(e)

        board = taifex_live.compute_live_board(baselines, quotes, price_basis=price_basis, z_entry=z_entry, z_exit=0.0)
        if board.empty:
            st.info("目前沒有可計算的配對。")
            return

        n_live = int(board["LiveOK"].sum())
        prev_date = board["PrevDate"].iloc[0]
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        if err:
            st.error(f"❌ 期交所即時報價連線失敗：{err}（以下暫以 {prev_date} 收盤價計算）")
        elif n_live == 0:
            st.warning(f"⏸️ 尚未取得今日成交價（盤前或休市），以下為 {prev_date} 收盤價計算之 Z 分數。更新時間 {now_str}")
        else:
            st.success(f"🟢 期交所即時報價已連線：{n_live}/{len(board)} 組已用即時價更新 ｜ 前收基準日 {prev_date} ｜ 更新時間 {now_str}")

        open_sig = board[board["Signal"].str.startswith("🔥")]
        close_sig = board[board["Signal"].str.startswith("✅")]
        holding = board[board["Position"] != 0]
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(render_metric_card("🔥 盤中達開倉門檻", f"{len(open_sig)} 組", "positive" if len(open_sig) else ""), unsafe_allow_html=True)
        with m2:
            st.markdown(render_metric_card("✅ 盤中達平倉門檻", f"{len(close_sig)} 組", "positive" if len(close_sig) else ""), unsafe_allow_html=True)
        with m3:
            st.markdown(render_metric_card("📌 回測規則持倉中", f"{len(holding)} 組"), unsafe_allow_html=True)
        with m4:
            top = board.iloc[0]
            st.markdown(render_metric_card("最大偏離組合", f"{top['Name1']}/{top['Name2']} {top['LiveZ']:+.2f}"), unsafe_allow_html=True)

        # 1. 達門檻通知卡片 (使用原生 Streamlit Metric 卡片，無任何代碼框與縮排問題)
        triggered_df = pd.concat([open_sig, close_sig])
        if not triggered_df.empty:
            st.markdown("### 🔔 盤中即時觸發動作指示")
            for _, r in triggered_df.iterrows():
                render_signal_card(r)

        # 2. 持倉中監控 Expander (讓投資人隨時查看目前持倉明細與未實現損益)
        if not holding.empty:
            with st.expander(f"📌 查看回測持倉中之配對明細與未實現損益 (共 {len(holding)} 組)", expanded=False):
                for _, r in holding.iterrows():
                    render_signal_card(r)

        show = board if not only_signal else board[board["Signal"].str.startswith(("🔥", "✅"))]
        if show.empty:
            st.info("目前沒有組合達到開倉或平倉門檻。")
            return

        def fmt(v, nd=2):
            return f"{v:.{nd}f}" if v is not None and pd.notna(v) else "—"

        def fmt_pnl(r):
            if r.get("IsHolding") and r.get("TotalPnL") is not None:
                return f"{r['TotalPnL']:+,.0f} 元"
            elif str(r.get("Signal", "")).startswith("🔥") and r.get("ExpPnL") is not None:
                return f"預估 +{r['ExpPnL']:,.0f} 元"
            return "——"

        def fmt_roi(r):
            if r.get("IsHolding") and r.get("PnLPct") is not None:
                return f"{r['PnLPct']:+.1f}%"
            elif str(r.get("Signal", "")).startswith("🔥") and r.get("ExpROI") is not None:
                return f"預期 +{r['ExpROI']:.1f}%"
            return "——"

        view = pd.DataFrame({
            "排名": show["Rank"],
            "盤中訊號": show["Signal"],
            "配對 (A vs B)": [f"{a}({taifex_live._code(x)}) vs {b}({taifex_live._code(y)})" for a, x, b, y in zip(show["Name1"], show["Ticker1"], show["Name2"], show["Ticker2"])],
            "細產業": show["SubIndustry"],
            "窗口": show["Window"].astype(str) + "日",
            "口數配置": [f"{r['Name1']}×{int(r['C1'])} ({r.get('Val1W', 0):.1f}萬) : {r['Name2']}×{int(r['C2'])} ({r.get('Val2W', 0):.1f}萬)" for _, r in show.iterrows()],
            "開倉價A": [fmt(r["EntryP1"]) if r.get("EntryP1") is not None else "—" for _, r in show.iterrows()],
            "開倉價B": [fmt(r["EntryP2"]) if r.get("EntryP2") is not None else "—" for _, r in show.iterrows()],
            "即時價A": [fmt(v) for v in show["ExitP1"]],
            "即時價B": [fmt(v) for v in show["ExitP2"]],
            "1組損益(元)": [fmt_pnl(r) for _, r in show.iterrows()],
            "損益率%": [fmt_roi(r) for _, r in show.iterrows()],
            "即時價差": show["LiveSpread"].round(2),
            "停利中樞(Z=0)": show["LiveMean"].round(2),
            "即時Z": show["LiveZ"].round(2),
            "前收Z": show["PrevZ"].round(2),
            "ΔZ": show["DeltaZ"].round(2),
            "保證金(萬)": (show["RealMargin"] / 10000.0).round(1),
            "獲利因子": show["ProfitFactor"].round(2),
            "勝率%": show["WinRate"].round(1),
            "報價時間": show["QuoteTime"],
        })
        
        st.dataframe(_style_live_board(view), use_container_width=True, hide_index=True, height=min(720, len(view) * 35 + 40))
        st.caption(
            "說明：\n"
            "1. **最小口數配置**：以高價股固定 1 口為基準，低價股配最接近但不超過高價股總市值之口數（向下取整，價格小的總金額不超過價格大的），確保最小資金佔用；1 口股票期貨 = 2,000 股。\n"
            "2. **開倉價格與損益**：若組合為回測規則持倉中，顯示該筆交易歷史開倉價與今日平倉/未實現累計損益；若為今日剛達開倉門檻之組合，顯示以現價建倉之建議進場價與預期收斂至中樞時之預估獲利金額。\n"
            "3. **資料來源**：期交所 MIS 即時行情 (mis.taifex.com.tw) 自動高速串接。"
        )


    _live_fragment()
    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


def render_watchlist_page(prices_df=None):
    """渲染今日精選收斂配對監控頁面 (100% 仿照回測樣式架構，2020年至今)"""
    # 1. Main Header (回測旗艦樣式)
    st.markdown(
        '<div class="main-header">'
        '<h1>🎯 今日同產業收斂配對監控中心</h1>'
        '<p>回測計算區間：2020年1月1日至今 ｜ Z-Score 20天 / 60天雙窗口 ｜ 最小 1 組口數配置 ｜ 每日精準計算當前 Z 分數、停利價差目標與歷史平均收斂時間</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # 1.5 資料庫來源切換
    c_p1, c_p2 = st.columns([3, 1])
    with c_p1:
        pool_mode = st.radio(
            "📚 配對池顯示範圍",
            ["🏆 精選去重配對 (52 組，同標的優先保留達進場線組合)", "🌐 完整雙向配對主檔 (76 組，含雙向所有組合)"],
            index=0,
            horizontal=True,
            help="【精選去重配對】：同一組標的僅保留表現最優或當前已達進場線之單一方向組合。\n【完整雙向配對主檔】：展示回測篩選出的所有 Leg1/Leg2 組合。"
        )
    with c_p2:
        if st.button("🔄 重新載入最新資料庫", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    use_master = "完整雙向" in pool_mode
    target_path = CSV_MASTER_PATH if use_master else (CSV_DEDUP_PATH if os.path.exists(CSV_DEDUP_PATH) else CSV_MASTER_PATH)
    try:
        df_data = pd.read_csv(target_path)
    except Exception:
        df_data = load_convergence_data()

    if df_data is None or df_data.empty:
        st.warning("⚠️ 目前正在計算或尚未載入收斂配對主資料庫，請稍候片刻或點擊下方按鈕重新讀取！")
        if st.button("🔄 重新載入最新資料庫", key="retry_load_data"):
            st.rerun()
        return

    # 若傳入 prices_df，自動進行動態最新報價與 Z-Score 更新
    live_date_label = None
    if prices_df is not None and not prices_df.empty:
        df_data, live_date_label = update_watchlist_with_live_prices(df_data, prices_df)
        if live_date_label:
            st.info(f"⚡ 即時監控已同步：已使用最新市場行情資料（基準日：{live_date_label}）動態更新 Z 分數、價差與進出場指示。")

    industry_df = load_industry_lookup()

    # 2. 頂部 4 大回測風格 Metric Cards
    total_pairs = len(df_data)
    actionable_today = df_data[df_data['CurrentZScore'].abs() >= 2.0]
    avg_winrate = df_data['WinRate'].mean()
    avg_conv_days = df_data['AvgConvergenceDays'].mean()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(render_metric_card("精選高獲利配對池 (PF > 2.0)", f"{total_pairs} 組"), unsafe_allow_html=True)
    with col_m2:
        st.markdown(render_metric_card("今日進場訊號觸發 (|Z| ≥ 2.0)", f"{len(actionable_today)} 組可立即做", "positive"), unsafe_allow_html=True)
    with col_m3:
        st.markdown(render_metric_card("配對池平均歷史勝率", f"{avg_winrate:.1f}%", "positive"), unsafe_allow_html=True)
    with col_m4:
        st.markdown(render_metric_card("平均收斂週期", f"{avg_conv_days:.1f} 天"), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # 2.5 盤中即時 Z 分數排行榜 (期交所 MIS 即時報價，全配對池依 |Z| 排名)
    render_live_board(df_data, prices_df)

    # 3. 條件篩選與排序控制列
    st.markdown("## 🎛️ 條件篩選與標的挑選")
    c_f1, c_f2, c_f3, c_f4 = st.columns([2.6, 1.2, 1.8, 1.6])

    with c_f1:
        status_filter = st.radio(
            "⚡ 今日狀態快篩",
            ["🔥 今日已達進場點 (|Z| ≥ 2.0)", "🚀 極端偏離大機會 (|Z| ≥ 3.0)", "📋 全部高獲利配對 (PF > 2.0)", "⏳ 偏離觀察中 (|Z| ≥ 1.2)"],
            index=0 if len(df_data[df_data['CurrentZScore'].abs() >= 2.0]) > 0 else 2,
            horizontal=True
        )

    with c_f2:
        window_choice = st.selectbox(
            "📅 回測天期窗口",
            ["全部天期", "20日窗口 (敏感型)", "60日窗口 (穩健型)"],
            index=0
        )

    with c_f3:
        all_subs = ["全部細產業族群"] + sorted(df_data['SubIndustry'].unique().tolist())
        selected_sub = st.selectbox("🏷️ 細產業族群", all_subs, index=0)

    with c_f4:
        search_kw = st.text_input("🔍 搜尋代號或名稱", placeholder="例如: 2317 或 鴻海")

    # 篩選邏輯
    filtered_df = df_data.copy()
    if status_filter == "🔥 今日已達進場點 (|Z| ≥ 2.0)":
        filtered_df = filtered_df[filtered_df['CurrentZScore'].abs() >= 2.0]
    elif status_filter == "🚀 極端偏離大機會 (|Z| ≥ 3.0)":
        filtered_df = filtered_df[filtered_df['CurrentZScore'].abs() >= 3.0]
    elif status_filter == "⏳ 偏離觀察中 (|Z| ≥ 1.2)":
        filtered_df = filtered_df[filtered_df['CurrentZScore'].abs() >= 1.2]

    if "20日" in window_choice:
        filtered_df = filtered_df[filtered_df['Window'] == 20]
    elif "60日" in window_choice:
        filtered_df = filtered_df[filtered_df['Window'] == 60]

    if selected_sub != "全部細產業族群":
        filtered_df = filtered_df[filtered_df['SubIndustry'] == selected_sub]

    if search_kw:
        kw = search_kw.strip().lower()
        filtered_df = filtered_df[
            filtered_df['Ticker1'].str.lower().str.contains(kw) |
            filtered_df['Name1'].str.lower().str.contains(kw) |
            filtered_df['Ticker2'].str.lower().str.contains(kw) |
            filtered_df['Name2'].str.lower().str.contains(kw) |
            filtered_df['SubIndustry'].str.lower().str.contains(kw)
        ]

    # 根據獲利因子由大到小排序
    filtered_df = filtered_df.sort_values(by=['ProfitFactor', 'TotalReturn', 'WinRate'], ascending=[False, False, False])

    st.markdown(f"### 📋 監控配對總表（共 **{len(filtered_df)}** 組符合條件，依獲利因子由大到小排列）")

    if filtered_df.empty:
        st.info("💡 目前在此篩選條件下沒有標的。您可以切換為【全部高獲利配對 (PF > 2.0)】檢視全市場候選池！")
        return

    # 建立適合顯示的展示 DataFrame
    display_rows = []
    for idx, (_, r) in enumerate(filtered_df.iterrows()):
        pf_val = r['ProfitFactor']
        pf_display = "∞ (100%勝)" if pf_val >= 900 else f"{pf_val:.2f}"
        
        display_rows.append({
            '排名': idx + 1,
            '細產業': r['SubIndustry'],
            '配對標的': f"{r['Name1']} ({r['Ticker1'].replace('.TW','').replace('.TWO','')}) vs {r['Name2']} ({r['Ticker2'].replace('.TW','').replace('.TWO','')})",
            '窗口': f"{r['Window']}日",
            '最小口數': r['BaseContracts'],
            '保證金(萬)': f"{r['BaseMargin']/10000:.1f}萬",
            '當前Z分數': f"{r['CurrentZScore']:+.2f}",
            '今日建議動作': r['ActionRecommendation'],
            '停利目標': r['TakeProfitTarget'],
            '獲利因子': pf_display,
            '勝率': f"{r['WinRate']:.1f}%",
            '交易次數': f"{r['TotalTrades']}次 ({r['WinTrades']}勝/{r['LoseTrades']}敗)",
            '平均收斂': f"{r['AvgConvergenceDays']:.1f}天",
            '最短收斂': f"{r['MinConvergenceDays']}天",
            '最長收斂': f"{r['MaxConvergenceDays']}天",
            '歷史總報酬': f"{r['TotalReturn']:+,.0f}元",
            '報酬率': f"{r['TotalReturnPct']:+.1f}%",
            'MDD': f"{r['MaxDrawdownPct']:.1f}%",
            '協整p值': f"{r['CointPValue']:.4f}"
        })

    view_df = pd.DataFrame(display_rows)
    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, len(view_df) * 38 + 45)
    )

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    # ============================================================
    # 4. 單一配對深度剖析 (完全仿照回測樣式架構)
    # ============================================================
    st.markdown("## 🔍 單一配對深度剖析與實單指引 (2020年1月1日至今)")
    
    pair_options = [
        f"#{r['排名']} ｜ [{r['細產業']}] {r['配對標的']} (窗口: {r['窗口']}, 口數: {r['最小口數']}) ｜ 當前Z: {r['當前Z分數']} ｜ PF: {r['獲利因子']}"
        for _, r in view_df.iterrows()
    ]
    
    selected_option = st.selectbox("🎯 選擇要深入剖析的配對：", pair_options, index=0)
    sel_idx = pair_options.index(selected_option)
    selected_raw = filtered_df.iloc[sel_idx]

    sym1 = selected_raw['Ticker1']
    sym2 = selected_raw['Ticker2']
    name1 = selected_raw['Name1']
    name2 = selected_raw['Name2']
    w = int(selected_raw['Window'])
    z_now = float(selected_raw['CurrentZScore'])
    hr = float(selected_raw.get('HedgeRatio', 1.0))
    p1_latest = float(selected_raw['LatestPrice1'])
    p2_latest = float(selected_raw['LatestPrice2'])
    contracts_str = str(selected_raw['BaseContracts'])
    sp_now = float(selected_raw['CurrentSpread'])
    sp_mean = float(selected_raw['SpreadMean'])
    sp_std = float(selected_raw['SpreadStd'])

    # 依最新市價動態計算等合約總值最優口數與保證金
    c1_dyn, c2_dyn = taifex_live.find_min_contracts(p1_latest, p2_latest)
    c1_num = str(c1_dyn)
    c2_num = str(c2_dyn)
    base_margin = (p1_latest * c1_dyn + p2_latest * c2_dyn) * 2000 * 0.135

    prof1 = get_stock_profile(sym1, industry_df)
    prof2 = get_stock_profile(sym2, industry_df)

    # 執行從 2020-01-01 起算之高精確回測
    df_rec = None
    df_trades = None
    stats = None
    if prices_df is not None and sym1 in prices_df.columns and sym2 in prices_df.columns:
        s1_full = prices_df[sym1].dropna()
        s2_full = prices_df[sym2].dropna()
        df_rec, df_trades, stats = run_single_pair_backtest_2020(s1_full, s2_full, sym1, sym2, name1, name2, w)

    # 若精準回測計算成功，以 2020 至今的統計數據為主；否則回退至掃描主檔
    if stats is not None:
        total_ret = stats.get('total_return', selected_raw['TotalReturn'])
        total_ret_pct = stats.get('total_return_pct', selected_raw['TotalReturnPct'])
        ann_ret = stats.get('annualized_return', selected_raw['AnnualizedReturn'])
        winrate = stats.get('win_rate', selected_raw['WinRate'])
        total_trades = stats.get('total_trades', selected_raw['TotalTrades'])
        mdd_pct = stats.get('max_drawdown_pct', selected_raw['MaxDrawdownPct'])
        pf_val = stats.get('profit_factor', selected_raw['ProfitFactor'])
        sharpe = stats.get('sharpe_ratio', selected_raw['SharpeRatio'])
        avg_conv = stats.get('avg_convergence_days', selected_raw['AvgConvergenceDays'])
        min_conv = stats.get('min_convergence_days', selected_raw['MinConvergenceDays'])
        max_conv = stats.get('max_convergence_days', selected_raw['MaxConvergenceDays'])
        coint_p = stats.get('coint_pvalue', selected_raw['CointPValue'])
    else:
        total_ret = selected_raw['TotalReturn']
        total_ret_pct = selected_raw['TotalReturnPct']
        ann_ret = selected_raw['AnnualizedReturn']
        winrate = selected_raw['WinRate']
        total_trades = selected_raw['TotalTrades']
        mdd_pct = selected_raw['MaxDrawdownPct']
        pf_val = selected_raw['ProfitFactor']
        sharpe = selected_raw['SharpeRatio']
        avg_conv = selected_raw['AvgConvergenceDays']
        min_conv = selected_raw['MinConvergenceDays']
        max_conv = selected_raw['MaxConvergenceDays']
        coint_p = selected_raw['CointPValue']

    # 4.1 兩檔標的並排營收結構卡片 (完全回測樣式)
    st.markdown("### 🏢 標的細產業分類與產品營收結構深度對比")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(render_stock_card_html(prof1), unsafe_allow_html=True)
    with col_c2:
        st.markdown(render_stock_card_html(prof2), unsafe_allow_html=True)

    # 4.2 產業同質性與連動診斷 Banner (完全回測樣式)
    p_corr = selected_raw.get('PriceCorr', None)
    r_corr = selected_raw.get('ReturnCorr', None)
    hl_str = f"{avg_conv:.1f} 天 (歷史最短 {min_conv} 天 / 最長 {max_conv} 天)"
    st.markdown(render_co_movement_banner_html(prof1, prof2, price_corr=p_corr, ret_corr=r_corr, coint_p=coint_p, hl_str=hl_str), unsafe_allow_html=True)

    # 4.3 實單口數與即時保證金計算 (4 欄 Metric Cards + Expander)
    st.markdown("### 📋 配對口數與即時保證金計算")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    with col_p1:
        st.markdown(render_metric_card("Hedge Ratio (β)", f"{hr:.4f}"), unsafe_allow_html=True)

    with col_p2:
        st.markdown(render_metric_card("最小等值口數配置", f"{name1}×{c1_num} : {name2}×{c2_num}"), unsafe_allow_html=True)

    with col_p3:
        st.markdown(render_metric_card("最低一組基本保證金", f"{base_margin:,.0f} 元"), unsafe_allow_html=True)

    with col_p4:
        z_cls = 'positive' if z_now <= -2.0 else ('negative' if z_now >= 2.0 else '')
        st.markdown(render_metric_card(f"當前 Z 分數 ({w}日窗口)", f"{z_now:+.2f}", z_cls), unsafe_allow_html=True)

    # 詳細保證金計算 Expander
    with st.expander("📐 詳細保證金計算與合約規格", expanded=False):
        calc_col1, calc_col2 = st.columns(2)
        m1_contract_val = p1_latest * 2000
        val1_total = m1_contract_val * int(c1_num)
        m1_margin_req = val1_total * 0.135
        m2_contract_val = p2_latest * 2000
        val2_total = m2_contract_val * int(c2_num)
        m2_margin_req = val2_total * 0.135
        diff_pct = abs(val1_total - val2_total) / max(val1_total, val2_total) * 100 if max(val1_total, val2_total) > 0 else 0
        with calc_col1:
            st.markdown(f"""
            **Leg 1: {name1} ({sym1})** (現價: {p1_latest:.2f} 元)
            - 1 口合約價值 = {p1_latest:.2f} × 2,000 股 = **{m1_contract_val:,.0f} 元**
            - {c1_num} 口總市值 = **{val1_total:,.0f} 元**
            - {c1_num} 口原始保證金 = {val1_total:,.0f} × 13.5% = **{m1_margin_req:,.0f} 元**
            """)
        with calc_col2:
            st.markdown(f"""
            **Leg 2: {name2} ({sym2})** (現價: {p2_latest:.2f} 元)
            - 1 口合約價值 = {p2_latest:.2f} × 2,000 股 = **{m2_contract_val:,.0f} 元**
            - {c2_num} 口總市值 = **{val2_total:,.0f} 元**
            - {c2_num} 口原始保證金 = {val2_total:,.0f} × 13.5% = **{m2_margin_req:,.0f} 元**
            """)
        st.caption(f"*(註：依台灣期交所標準規格：1口股票期貨 = 2,000股。雙邊配置市值為 **{val1_total/10000:.1f}萬 vs {val2_total/10000:.1f}萬**，市值差距僅 **{diff_pct:.1f}%**，達到等金額 Dollar Neutrality 市場中性對沖；法定原始保證金為總市值之 13.5%)*")

    # 4.4 歷史回測績效與收斂週期 (12 欄 Metric Cards，2020年1月1日至今)
    st.markdown("### 📊 回測績效與收斂週期摘要 (2020年1月1日至今)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    ret_cls = 'positive' if total_ret >= 0 else 'negative'
    wr_cls = 'positive' if winrate >= 50 else 'negative'
    pf_disp = f"{pf_val:.2f}" if pf_val < 900 else "∞"

    with m1:
        st.markdown(render_metric_card("累計報酬", f"{total_ret:+,.0f}", ret_cls), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card("報酬率", f"{total_ret_pct:+.1f}%", ret_cls), unsafe_allow_html=True)
    with m3:
        st.markdown(render_metric_card("年化報酬率", f"{ann_ret:+.1f}%", "positive"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_metric_card("勝率", f"{winrate:.1f}%", wr_cls), unsafe_allow_html=True)
    with m5:
        st.markdown(render_metric_card("交易次數", f"{total_trades} 次"), unsafe_allow_html=True)
    with m6:
        st.markdown(render_metric_card("最大回撤 MDD", f"{mdd_pct:.1f}%", "negative"), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)
    c_sub1, c_sub2, c_sub3, c_sub4, c_sub5, c_sub6 = st.columns(6)
    sh_cls = 'positive' if sharpe > 1.0 else ''

    with c_sub1:
        st.markdown(render_metric_card("獲利因子 (PF)", pf_disp, "positive"), unsafe_allow_html=True)
    with c_sub2:
        st.markdown(render_metric_card("夏普比率", f"{sharpe:.2f}", sh_cls), unsafe_allow_html=True)
    with c_sub3:
        st.markdown(render_metric_card("平均收斂天數", f"{avg_conv:.1f} 天", "positive"), unsafe_allow_html=True)
    with c_sub4:
        st.markdown(render_metric_card("最短收斂天數", f"{min_conv} 天"), unsafe_allow_html=True)
    with c_sub5:
        st.markdown(render_metric_card("最長收斂天數", f"{max_conv} 天"), unsafe_allow_html=True)
    with c_sub6:
        st.markdown(render_metric_card("協整檢定 p 值", f"{coint_p:.4f}", "positive"), unsafe_allow_html=True)

    # 4.5 今日進場與停利實單操作指示 Banner (零空白縮排純 HTML，保證完美呈現)
    st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)
    
    if z_now >= 3.0:
        sig_cls = "diag-banner-red"
        sig_title = f"🚀 今日極端放空價差收斂大機會已觸發 (Z = {z_now:+.2f} ≥ +3.0)"
        act_lead = f"價差高達歷史中樞 {z_now:.2f} 個標準差（極端超買偏離）！依指示<b>預設不設停損</b>，極端偏離蘊藏極高均值回歸收益空間，建議【放空價差】：賣出 Leg 2 ({name2})、買進 Leg 1 ({name1})，耐心持倉至 Z=0 全數停利。"
        room_val = sp_now - sp_mean
        tp_text = f"當價差自 <b>{sp_now:.2f}</b> 回落至中樞 <b>{sp_mean:.2f}</b> (Z 回歸 0) 時全數停利出場！"
        sl_text = "<b>平倉風控機制</b>：本策略依指示不設停損，持倉耐心等待價差完全均值回歸中樞 (Z=0) 時全數停利出場。"
    elif z_now <= -3.0:
        sig_cls = "diag-banner-green"
        sig_title = f"🚀 今日極端做多價差收斂大機會已觸發 (Z = {z_now:+.2f} ≤ -3.0)"
        act_lead = f"價差低達歷史中樞 {abs(z_now):.2f} 個標準差（極端超賣偏離）！依指示<b>預設不設停損</b>，極端偏離蘊藏極高均值回歸收益空間，建議【做多價差】：買進 Leg 2 ({name2})、賣出 Leg 1 ({name1})，耐心持倉至 Z=0 全數停利。"
        room_val = sp_mean - sp_now
        tp_text = f"當價差自 <b>{sp_now:.2f}</b> 回升至中樞 <b>{sp_mean:.2f}</b> (Z 回歸 0) 時全數停利出場！"
        sl_text = "<b>平倉風控機制</b>：本策略依指示不設停損，持倉耐心等待價差完全均值回歸中樞 (Z=0) 時全數停利出場。"
    elif z_now >= 2.0:
        sig_cls = "diag-banner-red"
        sig_title = f"🔥 今日放空價差收斂訊號已觸發 (Z = {z_now:+.2f} ≥ +2.0)"
        act_lead = f"價差高於歷史中樞 2 個標準差，統計回歸機率極高！建議【放空價差】：賣出 Leg 2 ({name2})、買進 Leg 1 ({name1})"
        room_val = sp_now - sp_mean
        tp_text = f"當價差自 <b>{sp_now:.2f}</b> 回落至中樞 <b>{sp_mean:.2f}</b> (Z 回歸 0) 時全數停利出場！"
        sl_text = "<b>平倉風控機制</b>：本策略依指示不設停損，持倉耐心等待價差完全均值回歸中樞 (Z=0) 時全數停利出場。"
    elif z_now <= -2.0:
        sig_cls = "diag-banner-green"
        sig_title = f"🔥 今日做多價差收斂訊號已觸發 (Z = {z_now:+.2f} ≤ -2.0)"
        act_lead = f"價差低於歷史中樞 2 個標準差，統計回歸機率極高！建議【做多價差】：買進 Leg 2 ({name2})、賣出 Leg 1 ({name1})"
        room_val = sp_mean - sp_now
        tp_text = f"當價差自 <b>{sp_now:.2f}</b> 回升至中樞 <b>{sp_mean:.2f}</b> (Z 回歸 0) 時全數停利出場！"
        sl_text = "<b>平倉風控機制</b>：本策略依指示不設停損，持倉耐心等待價差完全均值回歸中樞 (Z=0) 時全數停利出場。"
    else:
        sig_cls = "diag-banner-yellow"
        sig_title = f"⏳ 偏離觀察中 (當前 Z = {z_now:+.2f}，尚未觸發 ±2.0 進場線)"
        act_lead = f"目前價差處於常態偏離區間（當前價差: {sp_now:.2f} ｜ 中樞: {sp_mean:.2f}），建議耐心等待價差達到極值再行建倉。"
        room_val = abs(sp_now - sp_mean)
        tp_text = f"目標中樞價差為 <b>{sp_mean:.2f}</b>"
        sl_text = "目前未持倉，維持空手觀察。"

    instruction_banner = (
        f'<div class="{sig_cls}">'
        f'<div style="font-size:1.18rem; font-weight:800; margin-bottom:0.6rem;">{sig_title}</div>'
        f'<div style="font-size:0.95rem; line-height:1.6; margin-bottom:0.8rem; opacity:0.95;">{act_lead}</div>'
        f'<div style="font-size:0.9rem; line-height:1.75; padding-top:0.6rem; border-top:1px dashed rgba(255,255,255,0.2);">'
        f'<div>• <b>實單配置</b>：<b>{name1} ({sym1})</b> × {c1_num} 口 (現價 {p1_latest:.2f} 元) ｜ <b>{name2} ({sym2})</b> × {c2_num} 口 (現價 {p2_latest:.2f} 元)</div>'
        f'<div>• <b>最低保證金需求</b>：約 <b>{base_margin:,.0f} 元</b></div>'
        f'<div>• <b>停利出場條件</b>：{tp_text} (預期收斂空間 <b>+{room_val:.2f} 點</b>)</div>'
        f'<div>• <b>停損風控機制</b>：{sl_text}</div>'
        f'<div>• <b>歷史收斂特性 (2020至今)</b>：平均 <b>{avg_conv:.1f} 個交易日</b> 收斂，歷史最短 {min_conv} 天，最長 {max_conv} 天 (歷史勝率 <b>{winrate:.1f}%</b>)</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(instruction_banner, unsafe_allow_html=True)

    # ============================================================
    # 4.6 回測標準圖表與交易明細 (依用戶需求全平鋪在同一頁，從2020年1月1日開始)
    # ============================================================
    st.markdown("## 📈 回測圖表與通道走勢分析 (2020年1月1日至今)")

    if prices_df is not None and sym1 in prices_df.columns and sym2 in prices_df.columns:
        s1 = prices_df[sym1].dropna()
        s2 = prices_df[sym2].dropna()
        comm = s1.index.intersection(s2.index)
        # 嚴格篩選自 2020-01-01 開始之數據
        comm_2020 = comm[comm >= '2020-01-01']
        
        if len(comm_2020) >= w + 10:
            s1_sub = s1.loc[comm_2020]
            s2_sub = s2.loc[comm_2020]
            
            spread_series = s2_sub - hr * s1_sub
            mean_series = spread_series.rolling(window=w).mean()
            std_series = spread_series.rolling(window=w).std()
            upper_series = mean_series + 2.0 * std_series
            lower_series = mean_series - 2.0 * std_series
            z_series = (spread_series - mean_series) / std_series
            
            # (圖 1) 價差走勢與 ±2σ 通道 + Z-Score 波動走勢 (自 2020-01-01 起算)
            st.markdown(f"### (1) 📈 {name1} vs {name2} 價差走勢與 ±2σ 進出場通道 (窗口: {w}日，2020年1月1日至今)")
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.65, 0.35],
                subplot_titles=(f"價差走勢 (Spread = {name2} - {hr:.4f} × {name1}) 與 ±2σ 通道", "Z-Score 波動走勢與進出場警戒線 (±2.0 進場 / 0 停利平倉，不設停損)")
            )
            
            # 價差與通道 (回測配色標準)
            fig.add_trace(go.Scatter(x=comm_2020, y=upper_series, line=dict(color='rgba(239, 68, 68, 0.75)', width=1.2, dash='dash'), name='上界 (+2σ 放空線)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=comm_2020, y=mean_series, line=dict(color='#64ffda', width=2), name='停利中樞線 (Z=0)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=comm_2020, y=lower_series, line=dict(color='rgba(34, 197, 94, 0.75)', width=1.2, dash='dash'), name='下界 (-2σ 做多線)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=comm_2020, y=spread_series, line=dict(color='#FF9800', width=2.2), name='實際價差 (Spread)'), row=1, col=1)
            
            # Z-Score (回測配色標準)
            fig.add_trace(go.Scatter(
                x=comm_2020, y=z_series,
                line=dict(color='#BB86FC', width=1.5),
                name='Z-Score 指標',
                fill='tozeroy', fillcolor='rgba(187, 134, 252, 0.12)'
            ), row=2, col=1)
            fig.add_hline(y=2.0, line_dash="dash", line_color="#ef4444", row=2, col=1)
            fig.add_hline(y=0.0, line_dash="solid", line_color="#64ffda", row=2, col=1)
            fig.add_hline(y=-2.0, line_dash="dash", line_color="#22c55e", row=2, col=1)
            
            fig.update_layout(
                height=480,
                margin=dict(l=50, r=40, t=50, b=30),
                template='plotly_dark',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="價差點數", row=1, col=1)
            fig.update_yaxes(title_text="Z-Score", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            # (圖 2) 兩檔個股歷史股價走勢對比 (雙軸，自 2020-01-01 起算)
            st.markdown(f"### (2) 📊 {name1} ({sym1}) vs {name2} ({sym2}) 雙軸歷史股價走勢對比 (2020年1月1日至今)")
            fig_price = make_subplots(specs=[[{"secondary_y": True}]])
            fig_price.add_trace(
                go.Scatter(x=comm_2020, y=s1_sub,
                           name=f"{name1} ({sym1})", line=dict(color='#2196F3', width=2)),
                secondary_y=False
            )
            fig_price.add_trace(
                go.Scatter(x=comm_2020, y=s2_sub,
                           name=f"{name2} ({sym2})", line=dict(color='#FF9800', width=2)),
                secondary_y=True
            )
            fig_price.update_layout(
                template='plotly_dark',
                height=400,
                margin=dict(l=50, r=50, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_price.update_yaxes(title_text=f"{name1} 股價 (TWD)", secondary_y=False)
            fig_price.update_yaxes(title_text=f"{name2} 股價 (TWD)", secondary_y=True)
            st.plotly_chart(fig_price, use_container_width=True)

            # (圖 3) 歷史回測累計報酬曲線 (自 2020-01-01 起算)
            st.markdown(f"### (3) 💰 {name1} vs {name2} 歷史回測累計報酬曲線 (2020年1月1日至今)")
            fig_eq = go.Figure()
            has_eq_data = False
            cum_pnl = None
            if df_rec is not None and not df_rec.empty:
                if '帳戶淨值' in df_rec.columns:
                    cum_pnl = df_rec['帳戶淨值'] - 1000000
                    has_eq_data = True
                elif '累計損益' in df_rec.columns:
                    cum_pnl = df_rec['累計損益']
                    has_eq_data = True
                elif '累計已實現損益' in df_rec.columns:
                    cum_pnl = df_rec['累計已實現損益']
                    has_eq_data = True

                if has_eq_data:
                    fig_eq.add_trace(go.Scatter(
                        x=df_rec.index,
                        y=cum_pnl,
                        mode='lines',
                        line=dict(color='#64ffda', width=2),
                        name='累計損益 (TWD)',
                        fill='tozeroy',
                        fillcolor='rgba(100, 255, 218, 0.12)'
                    ))
                    if '未實現損益' in df_rec.columns and not df_rec['未實現損益'].isna().all():
                        unrealized_colors = ['rgba(100,255,218,0.25)' if v >= 0 else 'rgba(255,107,107,0.25)'
                                             for v in df_rec['未實現損益']]
                        fig_eq.add_trace(go.Bar(
                            x=df_rec.index, y=df_rec['未實現損益'],
                            name='未實現損益 (Float)', marker_color=unrealized_colors,
                            opacity=0.4
                        ))

            if not has_eq_data and df_trades is not None and not df_trades.empty:
                trades_sorted = df_trades.sort_values(by='日期')
                fig_eq.add_trace(go.Scatter(
                    x=pd.to_datetime(trades_sorted['日期']),
                    y=trades_sorted['損益'].cumsum(),
                    mode='lines+markers',
                    line=dict(color='#64ffda', width=2),
                    name='累計已實現損益 (TWD)',
                    fill='tozeroy',
                    fillcolor='rgba(100, 255, 218, 0.12)'
                ))
                has_eq_data = True

            fig_eq.add_hline(y=0, line_color="gray", line_width=0.8)
            fig_eq.update_layout(
                template='plotly_dark',
                height=400,
                margin=dict(l=50, r=40, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_eq.update_yaxes(title_text="累計損益 (元)")
            st.plotly_chart(fig_eq, use_container_width=True)

            # (區塊 4) 歷史收斂交易明細全記錄 (2020年1月1日至今，全部展示，不截斷)
            if stats and stats.get('active_position'):
                act = stats['active_position']
                pnl = act['unrealized_pnl']
                pnl_color = "#ff6b6b" if pnl < 0 else "#64ffda"
                dir_desc = f"做空 {name1} {act['c1']}口 / 做多 {name2} {act['c2']}口" if act['direction'] == 1 else f"做多 {name1} {act['c1']}口 / 做空 {name2} {act['c2']}口"
                st.markdown(
                    f'<div style="background: rgba(255, 171, 0, 0.12); border: 1px solid rgba(255, 171, 0, 0.4); border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1.2rem;">'
                    f'<div style="font-weight: 700; font-size: 1.05rem; color: #ffb74d; margin-bottom: 0.5rem;">'
                    f'⏳ 提示：目前尚有 1 筆持倉中部位尚未平倉（已即時反映於上方累計報酬曲線，平倉後將計入下方歷史明細）'
                    f'</div>'
                    f'<div style="display: flex; flex-wrap: wrap; gap: 1.5rem; font-size: 0.9rem; line-height: 1.6;">'
                    f'<div>📅 <b>進場日期：</b>{act["entry_date"]} (持有 {act["days_held"]} 個交易日)</div>'
                    f'<div>⚖️ <b>持倉方向：</b>{dir_desc}</div>'
                    f'<div>🎯 <b>開倉時 Z 分數：</b>{act["entry_zscore"]} ｜ <b>當前 Z 分數：</b>{act["current_zscore"]}</div>'
                    f'<div>💲 <b>進場價格：</b>{name1} {act["entry_p1"]:.1f} / {name2} {act["entry_p2"]:.1f}</div>'
                    f'<div>📌 <b>最新價格：</b>{name1} {act["current_p1"]:.1f} / {name2} {act["current_p2"]:.1f}</div>'
                    f'<div>📈 <b>未實現浮動損益：</b><span style="color: {pnl_color}; font-weight: 700; font-size: 1.1rem;">{pnl:+,.0f} 元</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            if df_trades is not None and not df_trades.empty:
                st.markdown(f"### (4) 📝 歷史收斂交易明細全記錄 (2020年1月1日至今，共 {len(df_trades)} 筆交易，完整展示)")
                
                # 交易明細上方快速摘要條
                win_cnt = sum(1 for _, row in df_trades.iterrows() if row.get('損益', 0) > 0)
                lose_cnt = sum(1 for _, row in df_trades.iterrows() if row.get('損益', 0) <= 0)
                tot_pnl = df_trades['損益'].sum() if '損益' in df_trades.columns else 0.0
                st.info(f"📊 **2020至今交易彙整**：共執行 **{len(df_trades)}** 筆交易 ｜ **{win_cnt}** 勝 / **{lose_cnt}** 敗 (勝率 **{win_cnt/len(df_trades)*100:.1f}%**) ｜ 累計已實現淨損益 **{tot_pnl:+,.0f} 元**")
                
                # 倒序排列：最新交易排在最上方
                sorted_trades = df_trades.copy().iloc[::-1]

                def style_trades(row):
                    p = row.get('損益', 0)
                    if p > 0:
                        return ['background-color: rgba(100, 255, 218, 0.15)'] * len(row)
                    elif p < 0:
                        return ['background-color: rgba(255, 107, 107, 0.15)'] * len(row)
                    return [''] * len(row)

                fmt_dict = {}
                if '損益' in sorted_trades.columns:
                    fmt_dict['損益'] = '{:+,.0f}'
                if '交易成本' in sorted_trades.columns:
                    fmt_dict['交易成本'] = '{:,.0f}'
                if 'Z-Score' in sorted_trades.columns:
                    fmt_dict['Z-Score'] = '{:.2f}'

                styled_trades = sorted_trades.style.apply(style_trades, axis=1)
                if fmt_dict:
                    styled_trades = styled_trades.format(fmt_dict)

                st.dataframe(
                    styled_trades,
                    use_container_width=True,
                    hide_index=True,
                    height=min(600, len(sorted_trades) * 38 + 45)
                )

                # 下載該配對交易明細 CSV
                csv_t = sorted_trades.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label=f"📥 下載 {name1} vs {name2} 2020至今完整交易明細 CSV (共 {len(sorted_trades)} 筆)",
                    data=csv_t,
                    file_name=f"trade_log_{sym1}_{sym2}_2020_to_latest.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("此配對自 2020 年至今交易次數較少。")
        else:
            st.info("自 2020 年以來共同有效交易日較少，無法繪製高階走勢通道圖。")

    # 4.7 匯出 CSV
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    csv_bytes = filtered_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label=f"📥 下載今日精選收斂配對清單 (共 {len(filtered_df)} 組 CSV)",
        data=csv_bytes,
        file_name=f"tw_stock_futures_convergence_watchlist_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
