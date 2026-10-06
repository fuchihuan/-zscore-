"""
台灣股票期貨配對交易回測應用
============================
功能：
1. 從 200+ 檔有發行股票期貨的標的中任選兩檔配對
2. 自訂初始保證金
3. 自動計算最少配對口數 & 保證金需求
4. 完整模擬開倉/平倉/未實現/已實現損益
5. 自動剔除漲跌停日（無法成交）
6. 每月第三個禮拜三結算日強制平倉，次日重新開倉（含轉倉成本）
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import os
import json
import yfinance as yf
import datetime
import calendar

# ============================================================
# 頁面設定
# ============================================================
st.set_page_config(
    page_title="台股股期配對交易回測系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
    
    html, body, .stApp {
        font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
        font-size: 0.95rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8892b0;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ccd6f6;
    }
    .metric-card .value.positive { color: #64ffda; }
    .metric-card .value.negative { color: #ff6b6b; }
    
    .trade-win { background-color: rgba(100, 255, 218, 0.1); }
    .trade-lose { background-color: rgba(255, 107, 107, 0.1); }
    
    .info-box {
        background: linear-gradient(135deg, #0a192f, #112240);
        border-left: 4px solid #64ffda;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #ccd6f6;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #2d1f0e, #3d2b14);
        border-left: 4px solid #fda085;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #fda085;
    }
    
    /* 側邊欄寬度與排版優化 */
    section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
        min-width: 360px !important;
    }
    
    /* 側邊欄專屬標的檔案卡片 */
    .stock-badge-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 9px 12px;
        margin-top: -6px;
        margin-bottom: 12px;
        color: #f1f5f9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .stock-badge-card-a {
        border-left: 4px solid #38bdf8;
    }
    .stock-badge-card-b {
        border-left: 4px solid #f472b6;
    }
    .subind-pill {
        display: inline-block;
        background: #0284c7;
        color: white !important;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .subind-pill-b {
        display: inline-block;
        background: #db2777;
        color: white !important;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .pair-summary-pill {
        background: #0f172a;
        border: 1px dashed #475569;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 14px;
        font-size: 0.82rem;
        color: #e2e8f0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #f093fb, #f5576c) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.4) !important;
    }
    
    /* 細產業與產品營收卡片樣式 */
    .industry-panel {
        background: linear-gradient(135deg, #131b2e, #0c1424);
        border: 1px solid rgba(100, 255, 218, 0.25);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    }
    .subind-badge {
        display: inline-block;
        background: linear-gradient(90deg, #64ffda, #38ef7d);
        color: #0a192f !important;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        margin-right: 0.4rem;
    }
    .cat-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        color: #ccd6f6 !important;
        font-size: 0.78rem;
        padding: 0.22rem 0.55rem;
        border-radius: 6px;
    }
    .rev-row {
        display: flex;
        align-items: center;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .rev-name {
        width: 38%;
        color: #ccd6f6;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .rev-bar-bg {
        width: 44%;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        height: 8px;
        margin: 0 0.5rem;
        overflow: hidden;
    }
    .rev-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #64ffda, #00b4d8);
        border-radius: 4px;
    }
    .rev-pct {
        width: 18%;
        text-align: right;
        font-weight: 600;
        color: #64ffda;
        font-size: 0.85rem;
    }
    .diag-banner-green {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(5, 150, 105, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin: 1rem 0;
        color: #6ee7b7;
    }
    .diag-banner-yellow {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.18), rgba(217, 119, 6, 0.08));
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin: 1rem 0;
        color: #fde68a;
    }
    .diag-banner-red {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18), rgba(185, 28, 28, 0.08));
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin: 1rem 0;
        color: #fca5a5;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 常數
# ============================================================
SHARES_PER_CONTRACT = 2000
MARGIN_RATE = 0.135
TRADING_FEE_RATE = 0.00002
COMMISSION_PER_CONTRACT = 30
LIMIT_PCT = 0.10  # 台股漲跌停 10%

DATA_FILE = os.path.join(os.path.dirname(__file__), 'stock_prices.csv')
TAIFEX_FILE = os.path.join(os.path.dirname(__file__), 'taifex_stocks.csv')


# ============================================================
# 工具函式
# ============================================================
@st.cache_data(ttl=3600)
def load_price_data():
    """載入已下載的股價資料並自動更新到今日（包含台股與全球資產）"""
    import yfinance as yf
    import datetime
    
    if not os.path.exists(DATA_FILE):
        return None
        
    # 1. 載入本地台股資料庫
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    df = df.dropna(axis=1, thresh=len(df) * 0.8)
    
    dirty = False
    
    # 2. 判斷台股最後更新日
    tw_cols = [c for c in df.columns if '.TW' in c or '.TWO' in c]
    if tw_cols:
        tw_last_date = df[tw_cols].dropna(how='all').index[-1]
    else:
        tw_last_date = df.index[-1]
        
    today = datetime.datetime.now()
    if tw_last_date.date() < today.date() - datetime.timedelta(days=1):
        try:
            start_str = (tw_last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            if tw_cols:
                new_data = yf.download(tw_cols, start=start_str, group_by='ticker', auto_adjust=False, threads=True, progress=False)
                new_df_list = []
                if isinstance(new_data.columns, pd.MultiIndex):
                    for tk in tw_cols:
                        if tk in new_data:
                            tk_data = new_data[tk]
                            if 'Adj Close' in tk_data:
                                series = tk_data['Adj Close'].dropna()
                                series.name = tk
                                new_df_list.append(series)
                if new_df_list:
                    new_df = pd.concat(new_df_list, axis=1)
                    # Use update to fill missing values and add new rows via concat
                    df = pd.concat([df, new_df])
                    # Remove duplicated index just in case
                    df = df[~df.index.duplicated(keep='last')]
                    dirty = True
        except Exception as e:
            print(f"Error updating TW data: {e}")

    # 3. 更新全球資產資料
    global_symbols = {
        'NQ=F': '小納斯達克期貨', 'NIY=F': '日經225期貨', 'ES=F': 'S&P500期貨', 'YM=F': '小道瓊期貨',
        '^VIX': 'VIX恐慌指數', 'GC=F': '黃金期貨', 'CL=F': '輕原油期貨', 'SI=F': '白銀期貨', 'HG=F': '銅期貨',
        'EURUSD=X': '歐元/美元', 'USDJPY=X': '美元/日圓', 'GBPUSD=X': '英鎊/美元', 'AUDUSD=X': '澳幣/美元',
        'BTC-USD': '比特幣', 'ETH-USD': '以太幣',
        'SPY': 'SPDR S&P 500 ETF', 'QQQ': 'Invesco QQQ', 'DIA': 'SPDR Dow Jones', 'IWM': 'iShares Russell 2000',
        'VTI': 'Vanguard Total Stock', 'TLT': 'iShares 20+ Yr Treasury', 'GLD': 'SPDR Gold Trust', 
        'USO': 'United States Oil', 'VNQ': 'Vanguard Real Estate'
    }
    try:
        existing_global = [tk for tk in global_symbols.keys() if tk in df.columns]
        if existing_global:
            global_last_date = df[existing_global].dropna(how='all').index[-1]
            g_start_str = global_last_date.strftime('%Y-%m-%d')
        else:
            g_start_str = '2020-01-01'
            
        if not existing_global or global_last_date.date() < today.date() - datetime.timedelta(days=1):
            global_data = yf.download(
                list(global_symbols.keys()),
                start=g_start_str,
                group_by='ticker',
                auto_adjust=False,
                threads=True,
                progress=False
            )
            global_df_list = []
            if isinstance(global_data.columns, pd.MultiIndex):
                for tk in global_symbols.keys():
                    if tk in global_data:
                        tk_data = global_data[tk]
                        if 'Adj Close' in tk_data:
                            series = tk_data['Adj Close'].dropna()
                            series.name = tk
                            global_df_list.append(series)
            else:
                if 'Adj Close' in global_data:
                    series = global_data['Adj Close'].dropna()
                    series.name = list(global_symbols.keys())[0]
                    global_df_list.append(series)
            
            if global_df_list:
                global_df = pd.concat(global_df_list, axis=1)
                
                # Update existing columns and append new ones
                cols_to_add = [c for c in global_df.columns if c not in df.columns]
                cols_to_update = [c for c in global_df.columns if c in df.columns]
                
                if cols_to_add:
                    df = df.join(global_df[cols_to_add], how='outer')
                    dirty = True
                
                if cols_to_update:
                    for col in cols_to_update:
                        new_series = global_df[col].dropna()
                        if not new_series.empty:
                            df.loc[new_series.index, col] = new_series
                    dirty = True
                    
    except Exception as e:
        print(f"Error fetching global data: {e}")

    df = df.ffill().bfill()
    if dirty:
        try:
            df.to_csv(DATA_FILE)
            print("Saved updated prices to CSV.")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            
    return df

@st.cache_data(ttl=3600)
def get_industry_data():
    """載入全方位細產業與產品營收比重資料庫"""
    industry_file = os.path.join(os.path.dirname(__file__), 'industry_data.csv')
    if os.path.exists(industry_file):
        try:
            return pd.read_csv(industry_file).set_index('Ticker')
        except:
            return None
    return None


@st.cache_data(ttl=3600)
def get_subindustry_pairs():
    """載入細產業內已預先計算之協整與高相關配對"""
    f = os.path.join(os.path.dirname(__file__), 'cointegrated_subindustry_pairs.json')
    if os.path.exists(f):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                return json.load(fp)
        except:
            return []
    return []


@st.cache_data(ttl=3600)
def get_high_pf_pairs():
    """載入細產業獲利因子 > 3.0 精選配對回測數據"""
    f = os.path.join(os.path.dirname(__file__), 'filtered_profit_factor_gt3_dedup.csv')
    if os.path.exists(f):
        try:
            return pd.read_csv(f)
        except:
            return None
    return None


def get_ticker_names():
    """取得代號與純淨公司名稱對照表"""
    mapping = {}
    ind_df = get_industry_data()
    if ind_df is not None:
        for tk, row in ind_df.iterrows():
            code = str(row.get('Code', '')).strip()
            name = str(row.get('Name', '')).strip()
            if code and name and name != 'nan':
                mapping[code] = name
            if tk:
                mapping[str(tk)] = name

    if os.path.exists(TAIFEX_FILE):
        try:
            taifex_df = pd.read_csv(TAIFEX_FILE, encoding='utf-8-sig')
            for _, row in taifex_df.iterrows():
                code = str(row.iloc[2]).strip()
                name = str(row.iloc[3]).strip()
                if code != 'nan' and name != 'nan' and code and code not in mapping:
                    mapping[code] = name
        except:
            pass

    return mapping


def get_stock_profile(sym, industry_df=None):
    """取得標的的完整產業與產品營收結構"""
    import json
    if industry_df is None:
        industry_df = get_industry_data()
    
    clean_code = str(sym).replace('.TWO', '').replace('.TW', '').strip()
    
    if industry_df is not None:
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
                except:
                    pass
            if not prods:
                pb = row.get('ProductBreakdown')
                if pd.notna(pb) and str(pb).strip():
                    parts = str(pb).split('|')
                    for part in parts:
                        part = part.strip()
                        if part:
                            sub_parts = part.rsplit(' ', 1)
                            if len(sub_parts) == 2 and '%' in sub_parts[1]:
                                try:
                                    r_val = float(sub_parts[1].replace('%', '').strip())
                                    prods.append((sub_parts[0].strip(), r_val))
                                except:
                                    prods.append((part, 0.0))
                            else:
                                prods.append((part, 0.0))
                                
            return {
                'ticker': sym,
                'code': clean_code,
                'name': str(row.get('Name', clean_code)),
                'market': str(row.get('Market', '上市' if '.TW' in sym else '上櫃')),
                'category': str(row.get('Category', '未分類')),
                'sub_industry': str(row.get('SubIndustry', '一般產業')),
                'industry': str(row.get('Industry', '一般產業')),
                'primary_product': str(row.get('PrimaryProduct', '主要產品')),
                'primary_ratio': float(row.get('PrimaryRatio', 0.0)) if pd.notna(row.get('PrimaryRatio')) else 0.0,
                'product_breakdown': str(row.get('ProductBreakdown', '')),
                'business': str(row.get('Business', '無主要業務說明')),
                'products': prods
            }
            
    return {
        'ticker': sym,
        'code': clean_code,
        'name': clean_code,
        'market': '全球' if '=' in sym or '-' in sym or '^' in sym else ('上櫃' if '.TWO' in sym else '上市'),
        'category': '其他資產',
        'sub_industry': '其他',
        'industry': '其他',
        'primary_product': '綜合資產',
        'primary_ratio': 100.0,
        'product_breakdown': '',
        'business': '無業務說明',
        'products': []
    }


def render_stock_card(prof):
    prods = prof.get('products', [])
    html_rows = []
    if prods:
        for item, ratio in prods[:6]:
            bar_pct = min(100.0, max(0.0, float(ratio)))
            html_rows.append(f"""
            <div class="rev-row">
                <div class="rev-name" title="{item}">{item}</div>
                <div class="rev-bar-bg">
                    <div class="rev-bar-fill" style="width: {bar_pct:.1f}%;"></div>
                </div>
                <div class="rev-pct">{ratio:.1f}%</div>
            </div>
            """)
        rev_html = "".join(html_rows)
    else:
        p_name = prof.get('primary_product', '未提供細項')
        p_ratio = prof.get('primary_ratio', 100.0)
        rev_html = f"""
        <div class="rev-row">
            <div class="rev-name" title="{p_name}">{p_name}</div>
            <div class="rev-bar-bg"><div class="rev-bar-fill" style="width: 100%;"></div></div>
            <div class="rev-pct">{p_ratio:.0f}%</div>
        </div>
        """
        
    mkt_tag = f"<span class='cat-badge'>{prof['market']}</span>"
    clean_sym = prof['ticker'].replace('.TWO', '').replace('.TW', '')
    card_html = f"""
    <div class="industry-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem; flex-wrap:wrap; gap:0.4rem;">
            <div>
                <span style="font-size:1.25rem; font-weight:700; color:#fff;">{clean_sym} {prof['name']}</span>
                <span style="font-size:0.85rem; color:#94a3b8; margin-left:4px;">({prof['ticker']})</span>
                {mkt_tag}
            </div>
            <div>
                <span class="subind-badge">{prof['sub_industry']}</span>
                <span class="cat-badge">{prof['category']}</span>
            </div>
        </div>
        <div style="font-size:0.86rem; color:#94a3b8; line-height:1.55; margin-bottom:1rem; border-left:3px solid #64ffda; padding-left:0.6rem;">
            {prof['business']}
        </div>
        <div style="font-weight:600; font-size:0.88rem; color:#ccd6f6; margin-bottom:0.4rem;">
            📊 實質產品營收佔比結構 (MOPS / CMoney 月產銷組合)
        </div>
        {rev_html}
    </div>
    """
    return card_html


def render_sidebar_stock_badge(prof, sym, leg_label, color_theme="sky"):
    clean_sym = sym.replace('.TWO', '').replace('.TW', '')
    p_name = prof.get('primary_product', '未提供細項')
    p_ratio = prof.get('primary_ratio', 0.0)
    sub = prof.get('sub_industry', '未分類')
    mkt = prof.get('market', '')
    cat = prof.get('category', '')
    ratio_str = f"{p_ratio:.1f}%" if p_ratio > 0 else "-"
    
    border_class = "stock-badge-card-a" if color_theme == "sky" else "stock-badge-card-b"
    title_color = "#38bdf8" if color_theme == "sky" else "#f472b6"
    badge_class = "subind-pill" if color_theme == "sky" else "subind-pill-b"
    
    card_html = f"""
    <div class="stock-badge-card {border_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-weight: 700; color: {title_color}; font-size: 0.95rem;">
                🏢 {clean_sym} {prof['name']}
            </span>
            <span class="{badge_class}">
                {sub}
            </span>
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 2px;">
            🏷️ <b>市場大類</b>：{mkt} ｜ {cat}
        </div>
        <div style="font-size: 0.8rem; color: #e2e8f0;">
            📦 <b>主力營收</b>：{p_name} <b style="color: {title_color};">({ratio_str})</b>
        </div>
    </div>
    """
    return card_html


def render_sidebar_pair_diagnostic(prof1, prof2, sym1, sym2, sub_pairs):
    is_same_sub = (prof1['sub_industry'] == prof2['sub_industry']) and (prof1['sub_industry'] not in ['其他', '一般產業'])
    
    # 尋找預計算協整資料庫
    m_pairs = [
        p for p in sub_pairs 
        if (p['Ticker1'] == sym1 and p['Ticker2'] == sym2) or (p['Ticker1'] == sym2 and p['Ticker2'] == sym1)
    ]
    pair_info = m_pairs[0] if m_pairs else None
    
    if is_same_sub:
        status_html = f"<span style='color: #4ade80; font-weight: 700;'>🟢 同細產業族群 ({prof1['sub_industry']})</span>"
        desc_text = "核心業務產品高度同質，具備天然協整基礎"
    else:
        status_html = f"<span style='color: #fbbf24; font-weight: 700;'>🟡 跨細產業配對</span>"
        desc_text = f"【{prof1['sub_industry']}】vs【{prof2['sub_industry']}】，留意走勢發散風險"
        
    if pair_info:
        corr_val = pair_info.get('PriceCorr', 0.0)
        coint_rat = pair_info.get('Rating', '無評級')
        metric_text = f"走勢相關: <b style='color:#38bdf8;'>{corr_val:+.2f}</b> ｜ 協整: <b style='color:#fbbf24;'>{coint_rat}</b>"
    else:
        metric_text = "即時自選標的 ｜ 點擊「🚀 執行回測」開始運算"
        
    return f"""
    <div class="pair-summary-pill">
        <div style="margin-bottom: 3px;">{status_html}</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">{desc_text}</div>
        <div style="font-size: 0.75rem; color: #cbd5e1;">📈 {metric_text}</div>
    </div>
    """


def render_co_movement_banner(prof1, prof2, price_corr=None, ret_corr=None, coint_p=None, hl_str=None):
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
        tag_text = f"⚠️ 跨產業異質標的 (發散風險警示)"
        msg = f"警告：兩檔標的分屬不同產業（{prof1['sub_industry']} vs {prof2['sub_industry']}），缺乏共同實質營收與成本驅動因子！走勢脫鉤時容易引發單邊發散，強烈建議嚴格設置停損或採用發散模式！"

    metrics_html = []
    if price_corr is not None:
        metrics_html.append(f"<div>📈 <b>價格走勢相關度:</b> <span style='font-size:1rem; font-weight:700;'>{price_corr:+.3f}</span></div>")
    if ret_corr is not None:
        metrics_html.append(f"<div>⚡ <b>日報酬相關係數:</b> <span style='font-size:1rem; font-weight:700;'>{ret_corr:+.3f}</span></div>")
    if coint_p is not None:
        coint_display = f"{coint_p:.4f} (通過協整 ✅)" if coint_p < 0.05 else f"{coint_p:.4f} (未顯著 ⚠️)"
        metrics_html.append(f"<div>🔬 <b>共整合檢驗 (p-value):</b> <span style='font-size:1rem; font-weight:700;'>{coint_display}</span></div>")
    if hl_str:
        metrics_html.append(f"<div>⏱️ <b>OU 均值回歸半衰期:</b> <span style='font-size:1rem; font-weight:700;'>{hl_str}</span></div>")

    metrics_section = ""
    if metrics_html:
        metrics_section = f"""
        <div style="display:flex; flex-wrap:wrap; gap:1.2rem; font-size:0.85rem; padding-top:0.4rem; border-top:1px dashed rgba(255,255,255,0.2);">
            {''.join(metrics_html)}
        </div>
        """

    banner_html = f"""
    <div class="{banner_cls}">
        <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.4rem;">{tag_text}</div>
        <div style="font-size:0.9rem; line-height:1.5; margin-bottom:0.6rem; opacity:0.95;">{msg}</div>
        {metrics_section}
    </div>
    """
    return banner_html


def detect_limit_days(series):
    """
    偵測漲跌停日（以 10% 為限制）
    回傳一個 boolean Series，True 表示當日觸及漲跌停
    """
    pct = series.pct_change()
    limit_up = pct >= (LIMIT_PCT - 0.001)    # 接近或等於 +10%
    limit_down = pct <= -(LIMIT_PCT - 0.001)  # 接近或等於 -10%
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
    """找最少配對口數（以實質股價等值平衡為目標，加入口數懲罰）"""
    target_ratio = price_s2 / price_s1
    best_score = float('inf')
    best_s1, best_s2 = 1, 1
    for s2 in range(1, 21):
        for s1 in range(1, 21):
            ratio = s1 / s2
            error = abs(ratio - target_ratio)
            # 加入口數懲罰，避免為了極微小的精確率而讓合約口數暴增
            score = error + (s1 + s2) * 0.015
            if score < best_score:
                best_score = score
                best_s1, best_s2 = s1, s2
    return best_s1, best_s2


def calc_max_contracts(price_s1, price_s2, capital, margin_rate=MARGIN_RATE):
    """根據保證金自動計算最大口數"""
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
    """
    產生指定年份範圍內所有「每月第三個禮拜三」的日期集合。
    台灣股票期貨在每月第三個禮拜三收盤後結算。
    """
    settlement_dates = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            cal = calendar.monthcalendar(year, month)
            # Wednesday = index 2 in monthcalendar (Mon=0)
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

    # 對齊
    common_idx = S1.index.intersection(S2.index)
    S1 = S1[common_idx].copy()
    S2 = S2[common_idx].copy()

    # Log Price 轉換
    use_log_price = advanced_params.get('use_log_price', False)
    if use_log_price:
        model_S1 = np.log(S1)
        model_S2 = np.log(S2)
    else:
        model_S1 = S1.copy()
        model_S2 = S2.copy()

    # 偵測漲跌停
    limit_s1 = detect_limit_days(S1)
    limit_s2 = detect_limit_days(S2)
    any_limit = limit_s1 | limit_s2

    from advanced_models import get_zscore_signals, get_ou_signals, get_garch_signals, get_kalman_filter_signals, get_copula_signals, get_jump_diffusion_signals, get_sdde_signals, get_np_cusum_signals, get_gsadf_signals, get_dcc_garch_vecm_signals, get_markov_regime_signals

    trade_mode = advanced_params.get('trade_mode', '收斂 (均值回歸)')
    ensemble_logic = advanced_params.get('ensemble_logic', '單一模型')
    
    # If selected_models is passed as a single string (from legacy or 收斂 mode), wrap it in a list
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
                else: # OR
                    if any(entry_up):
                        current_state = 1
                    elif any(entry_dn):
                        current_state = -1
            else:
                if current_state == 1:
                    # 無論 AND 或 OR，只要有任何一個模型說該平倉了 (回到均值)，就先落袋為安
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

    # 預先計算進階濾波 Mask
    zscore = signal_df['Indicator']
    spread = signal_df['Spread']
    import numpy as np
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


    # 時間切片
    if run_start_date and run_end_date:
        signal_df = signal_df.loc[run_start_date:run_end_date]
        zscore = zscore.loc[run_start_date:run_end_date]
        ou_pass_mask = ou_pass_mask.loc[run_start_date:run_end_date]
        beta_trans_mask = beta_trans_mask.loc[run_start_date:run_end_date]
        roll_hl_series = roll_hl_series.loc[run_start_date:run_end_date]
        roll_r2_series = roll_r2_series.loc[run_start_date:run_end_date]
    
    if signal_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {'insufficient_margin_error': True, 'msg': '所選區間內無可用資料或資料長度不足滾動窗口計算！'}

    # 回測期起始保證金估算
    start_date = zscore.index[0]
    p1_start = S1[start_date]
    p2_start = S2[start_date]

    usable_capital = initial_capital * margin_usage_pct

    if size_mode == '依保證金上限最大化（預設）':
        contracts_s1, contracts_s2, multiplier = calc_max_contracts(
            p1_start, p2_start, usable_capital
        )
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
    dates_list = list(valid_dates)
    
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
        
        # 讀取當天動態邊界
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
                    
                    trade_mode = advanced_params.get('trade_mode', '收斂 (均值回歸)')
                    if use_grid:
                        tp_z = params['tp_z']
                        if trade_mode == '收斂 (均值回歸)':
                            is_converged = (pos == 1 and z >= tp_z) or (pos == -1 and z <= -tp_z)
                        else:
                            is_converged = (pos == 1 and z <= tp_z) or (pos == -1 and z >= -tp_z)
                    else:
                        if trade_mode == '收斂 (均值回歸)':
                            is_converged = (pos == 1 and z >= dyn_exit_upper) or (pos == -1 and z <= dyn_exit_lower)
                        else:
                            is_converged = (pos == 1 and z <= dyn_exit_upper) or (pos == -1 and z >= dyn_exit_lower)
                        
                    if use_smart_roll and net_pnl > 0 and abs(z) <= smart_roll_z:
                        state['pending_reopen_dir'] = 0
                        action += f" (防呆不再建倉)"
                    elif is_converged:
                        state['pending_reopen_dir'] = 0
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
                    if use_grid:
                        if abs(z) <= params['reentry_z']:
                            state['is_stopped_out'] = False
                            action += f" [網格{params['id']}] Z分數回落解除冷卻 "
                    else:
                        state['is_stopped_out'] = False

            # 6. 正常開倉 (Entry)
            for state in grid_states:
                if not state['is_active'] and not state['is_stopped_out'] and state['pending_reopen_dir'] == 0:
                    params = state['params']
                    enter_dir = 0
                    
                    # Check trade_mode
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
            'Upper_Bound': dyn_upper if not use_grid else np.nan,
            'Lower_Bound': dyn_lower if not use_grid else np.nan,
            'Exit_Upper': dyn_exit_upper if not use_grid else np.nan,
            'Exit_Lower': dyn_exit_lower if not use_grid else np.nan,
            f'{sym1}口數': sum(s['c1'] for s in grid_states if s['is_active']),
            f'{sym2}口數': sum(s['c2'] for s in grid_states if s['is_active']),
            '未實現損益': unrealized_pnl,
            '累計已實現損益': realized_pnl_total,
            '凍結保證金': frozen_margin,
            '帳戶淨值': equity,
            '動作': action,
            'Beta': _hr[date] if advanced_params.get('use_beta_transition', False) else np.nan,
            'OU_HalfLife': roll_hl_series[date] if advanced_params.get('use_ou_filter', False) else np.nan,
            'OU_R2': roll_r2_series[date] if advanced_params.get('use_ou_filter', False) else np.nan
        })

    df_records = pd.DataFrame(records)
    df_records.set_index('日期', inplace=True)
    df_trades = pd.DataFrame(trade_log)

    close_trades = df_trades[df_trades['動作'].str.contains('平倉')] if not df_trades.empty else pd.DataFrame()
    stats = {}
    stats['insufficient_margin_error'] = insufficient_margin_error
    stats['base_s1'] = base_s1
    stats['base_s2'] = base_s2
    stats['base_margin'] = base_margin
    stats['hedge_ratio'] = signal_df['Hedge_Ratio'].iloc[-1] if not signal_df.empty else 0
    stats['contracts_s1'] = contracts_s1
    stats['contracts_s2'] = contracts_s2
    stats['multiplier'] = multiplier
    stats['total_margin_needed'] = base_margin
    stats['margin_s1'] = margin_s1
    stats['margin_s2'] = margin_s2
    stats['margin_utilization'] = (total_margin_needed_start / initial_capital * 100) if initial_capital > 0 else 0
    stats['p1_start'] = p1_start
    stats['p2_start'] = p2_start
    stats['start_date'] = valid_dates[0]
    stats['end_date'] = valid_dates[-1]
    stats['initial_capital'] = initial_capital
    stats['final_equity'] = df_records['帳戶淨值'].iloc[-1]
    stats['total_return'] = df_records['帳戶淨值'].iloc[-1] - initial_capital
    stats['total_return_pct'] = (df_records['帳戶淨值'].iloc[-1] / initial_capital - 1) * 100
    
    peak = df_records['帳戶淨值'].cummax()
    drawdown = peak - df_records['帳戶淨值']
    drawdown_pct = drawdown / peak * 100
    stats['max_drawdown'] = drawdown.max()
    stats['max_drawdown_pct'] = drawdown_pct.max()

    trading_days = len(df_records)
    years = trading_days / 252
    
    daily_returns = df_records['帳戶淨值'].pct_change().dropna()
    
    if years > 0 and stats['final_equity'] > 0:
        stats['annualized_return'] = ((stats['final_equity'] / initial_capital) ** (1 / years) - 1) * 100
        ann_vol = daily_returns.std() * np.sqrt(252)
        if ann_vol > 0:
            stats['sharpe_ratio'] = (stats['annualized_return'] / 100 - 0.015) / ann_vol
        else:
            stats['sharpe_ratio'] = 0.0
    else:
        stats['annualized_return'] = 0.0
        stats['sharpe_ratio'] = 0.0

    if not close_trades.empty:
        win = close_trades[close_trades['損益'] > 0]
        lose = close_trades[close_trades['損益'] <= 0]
        stats['total_trades'] = len(close_trades)
        stats['win_trades'] = len(win)
        stats['lose_trades'] = len(lose)
        stats['win_rate'] = len(win) / len(close_trades) * 100
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

    stats['skipped_limits'] = skipped_limits
    stats['settlement_rolls'] = settlement_rolls
    stats['total_roll_cost'] = total_roll_cost

    margin_when_holding = df_records[df_records['持倉'] != 0]['凍結保證金']
    if not margin_when_holding.empty:
        stats['min_margin'] = margin_when_holding.min()
        stats['max_margin'] = margin_when_holding.max()
        stats['avg_margin'] = margin_when_holding.mean()
    else:
        stats['min_margin'] = 0
        stats['max_margin'] = 0
        stats['avg_margin'] = 0
        
    valid_hl = df_records['OU_HalfLife'].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid_hl.empty:
        stats['avg_ou_half_life'] = valid_hl.mean()
    else:
        stats['avg_ou_half_life'] = None

    try:
        _, pvalue, _ = coint(S1[common_idx], S2[common_idx])
        stats['coint_pvalue'] = pvalue
    except:
        stats['coint_pvalue'] = None

    return df_records, df_trades, stats


# ============================================================
# 主介面
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 台灣股票期貨 配對交易回測系統</h1>
        <p>Z-Score 均值回歸策略 ｜ 完整保證金模擬 ｜ 漲跌停過濾 ｜ 每月結算轉倉 ｜ 200+ 檔股期標的</p>
    </div>
    """, unsafe_allow_html=True)

    # 載入資料
    prices = load_price_data()
    if prices is None:
        st.error("找不到股價資料檔 (stock_prices.csv)。請先執行 `data_fetcher.py` 下載資料。")
        return

    tickers = sorted(prices.columns.tolist())
    name_map = get_ticker_names()
    industry_df = get_industry_data()
    sub_pairs = get_subindustry_pairs()
    high_pf_df = get_high_pf_pairs()

    # 建立所有標的的個人產業與產品營收結構快取
    stock_profiles = {}
    for t in tickers:
        stock_profiles[t] = get_stock_profile(t, industry_df)

    # 依細產業族群進行歸類
    sub_industry_dict = {}
    for t in tickers:
        sub = stock_profiles[t]['sub_industry']
        if sub not in sub_industry_dict:
            sub_industry_dict[sub] = []
        sub_industry_dict[sub].append(t)

    # 排序可選細產業（至少有 2 檔標的者優先，且依標的數量降冪排列）
    sorted_subs = sorted(
        [s for s, t_list in sub_industry_dict.items() if len(t_list) >= 2],
        key=lambda s: len(sub_industry_dict[s]),
        reverse=True
    )
    other_subs = sorted([s for s, t_list in sub_industry_dict.items() if len(t_list) < 2])
    all_available_subs = sorted_subs + other_subs

    # 格式化顯示函式 (以代號簡稱為首，確保絕不被截斷)
    def fmt_ticker_simple(t):
        prof = stock_profiles.get(t) or get_stock_profile(t, industry_df)
        clean_t = t.replace('.TWO', '').replace('.TW', '')
        name = prof['name']
        p1 = prof['primary_product']
        r1 = prof['primary_ratio']
        if r1 > 0 and p1 and p1 != '主要產品':
            return f"{clean_t} {name} ｜ {p1} {r1:.0f}%"
        else:
            return f"{clean_t} {name}"

    def fmt_ticker_grouped(t):
        prof = stock_profiles.get(t) or get_stock_profile(t, industry_df)
        clean_t = t.replace('.TWO', '').replace('.TW', '')
        name = prof['name']
        sub = prof['sub_industry']
        p1 = prof['primary_product']
        r1 = prof['primary_ratio']
        if r1 > 0 and p1 and p1 != '主要產品':
            return f"{clean_t} {name} ｜ {sub} ({r1:.0f}%)"
        else:
            return f"{clean_t} {name} ｜ {sub}"

    # ============================================================
    # 側邊欄: 參數設定
    # ============================================================
    with st.sidebar:
        st.markdown("## ⚙️ 回測參數設定")
        st.markdown("---")

        st.markdown("### 📌 選擇配對標的")
        pair_mode = st.radio(
            "標的挑選模式",
            ["🎯 依細產業族群挑選 (推薦·高連動)", "🌐 全市場自選 (依細產業排序)"],
            index=0,
            help="【依細產業族群挑選】：僅篩選同一細產業之同業標的，具備高度產品營收連動性與協整基礎，大幅降低走勢發散風險！"
        )

        if pair_mode == "🎯 依細產業族群挑選 (推薦·高連動)":
            default_sub = "銅箔基板 (CCL)" if "銅箔基板 (CCL)" in all_available_subs else (sorted_subs[0] if sorted_subs else all_available_subs[0])
            sel_sub_idx = all_available_subs.index(default_sub) if default_sub in all_available_subs else 0
            selected_sub = st.selectbox(
                "🏷️ 選擇細產業族群",
                all_available_subs,
                index=sel_sub_idx,
                help="選擇特定的細產業族群，系統將自動篩選該族群內的所有標的"
            )
            sub_tickers = sub_industry_dict.get(selected_sub, tickers)
            
            # 尋找該細產業內的獲利因子 > 3.0 推薦配對
            sub_high_pf = high_pf_df[high_pf_df['SubIndustry'] == selected_sub] if high_pf_df is not None and not high_pf_df.empty else pd.DataFrame()
            chosen_hp = None
            if not sub_high_pf.empty:
                hp_labels = ["-- 手動挑選下方標的 --"] + [
                    f"🔥 {r['Name1']} vs {r['Name2']} ｜ PF {r['ProfitFactor']:.2f} ｜ 勝率 {r['WinRate']:.1f}% ｜ 報酬 {r['TotalReturnPct']:+.1f}%"
                    for _, r in sub_high_pf.iterrows()
                ]
                chosen_hp = st.selectbox("🔥 該族群獲利因子 > 3.0 精選配對 (一鍵帶入)", hp_labels, index=0)
                if chosen_hp != hp_labels[0]:
                    h_idx = hp_labels.index(chosen_hp) - 1
                    target_row = sub_high_pf.iloc[h_idx]
                    default_s1 = sub_tickers.index(target_row['Ticker1']) if target_row['Ticker1'] in sub_tickers else 0
                    default_s2 = sub_tickers.index(target_row['Ticker2']) if target_row['Ticker2'] in sub_tickers else (1 if len(sub_tickers) > 1 else 0)

            # 尋找該細產業內的推薦配對 (來自 pre-computed 協整資料)
            if not chosen_hp or chosen_hp == hp_labels[0]:
                rec_pairs_for_sub = [
                    p for p in sub_pairs 
                    if p.get('SubIndustry') == selected_sub and p.get('Ticker1') in sub_tickers and p.get('Ticker2') in sub_tickers
                ]
                
                if rec_pairs_for_sub:
                    rec_pair_labels = ["-- 手動挑選下方標的 --"] + [
                        f"{p['Pair']} ({p['Name1']} vs {p['Name2']}) ｜ r={p['PriceCorr']:+.2f} ｜ {p['Rating']}"
                        for p in rec_pairs_for_sub[:15]
                    ]
                    chosen_rec = st.selectbox("💡 該族群高協整/高相關配對 (一鍵帶入)", rec_pair_labels, index=0)
                    if chosen_rec != "-- 手動挑選下方標的 --":
                        rec_idx = rec_pair_labels.index(chosen_rec) - 1
                        target_pair = rec_pairs_for_sub[rec_idx]
                        default_s1 = sub_tickers.index(target_pair['Ticker1']) if target_pair['Ticker1'] in sub_tickers else 0
                        default_s2 = sub_tickers.index(target_pair['Ticker2']) if target_pair['Ticker2'] in sub_tickers else (1 if len(sub_tickers) > 1 else 0)
                    else:
                        default_s1 = 0
                        default_s2 = 1 if len(sub_tickers) > 1 else 0
                else:
                    default_s1 = 0
                    default_s2 = 1 if len(sub_tickers) > 1 else 0

            st.markdown("#### 🅰️ 標的 A (Leg 1)")
            sym1 = st.selectbox("標的 A (Leg 1)", sub_tickers, index=default_s1, format_func=fmt_ticker_simple, key='sym1', label_visibility="collapsed")
            st.markdown(render_sidebar_stock_badge(stock_profiles[sym1], sym1, "A", "sky"), unsafe_allow_html=True)

            st.markdown("#### 🅱️ 標的 B (Leg 2)")
            sym2 = st.selectbox("標的 B (Leg 2)", sub_tickers, index=default_s2, format_func=fmt_ticker_simple, key='sym2', label_visibility="collapsed")
            st.markdown(render_sidebar_stock_badge(stock_profiles[sym2], sym2, "B", "pink"), unsafe_allow_html=True)

        else:
            # 全市場自選: 依細產業分組排序
            sorted_all_tickers = sorted(
                tickers,
                key=lambda t: (stock_profiles[t]['sub_industry'], t)
            )
            default_s1 = sorted_all_tickers.index('2383.TW') if '2383.TW' in sorted_all_tickers else 0
            default_s2 = sorted_all_tickers.index('6213.TW') if '6213.TW' in sorted_all_tickers else (sorted_all_tickers.index('6274.TWO') if '6274.TWO' in sorted_all_tickers else 1)

            # 全市場獲利因子 > 3.0 推薦一鍵帶入
            chosen_pf = None
            if high_pf_df is not None and not high_pf_df.empty:
                pf_labels = ["-- 手動挑選下方全市場標的 --"] + [
                    f"🔥 [{r['SubIndustry']}] {r['Name1']} vs {r['Name2']} ｜ PF {r['ProfitFactor']:.2f} ｜ 勝率 {r['WinRate']:.1f}% ｜ 報酬 {r['TotalReturnPct']:+.1f}%"
                    for _, r in high_pf_df.iterrows()
                ]
                chosen_pf = st.selectbox("🔥 獲利因子 > 3.0 同業精選 (一鍵帶入)", pf_labels, index=0)
                if chosen_pf != pf_labels[0]:
                    p_idx = pf_labels.index(chosen_pf) - 1
                    target_row = high_pf_df.iloc[p_idx]
                    t1, t2 = target_row['Ticker1'], target_row['Ticker2']
                    if t1 in sorted_all_tickers:
                        default_s1 = sorted_all_tickers.index(t1)
                    if t2 in sorted_all_tickers:
                        default_s2 = sorted_all_tickers.index(t2)

            # 全市場高協整推薦一鍵帶入
            if not chosen_pf or chosen_pf == pf_labels[0]:
                coint_top = [p for p in sub_pairs if p.get('IsCointegrated') or p.get('CointPValue', 1.0) < 0.05][:20]
                if coint_top:
                    coint_labels = ["-- 手動挑選下方全市場標的 --"] + [
                        f"{p['Pair']} ({p['Name1']} vs {p['Name2']}) · {p['SubIndustry']} ｜ {p['Rating']}"
                        for p in coint_top
                    ]
                    chosen_coint = st.selectbox("💡 全市場精選高協整配對 (一鍵帶入)", coint_labels, index=0)
                    if chosen_coint != coint_labels[0]:
                        c_idx = coint_labels.index(chosen_coint) - 1
                        tp = coint_top[c_idx]
                        if tp['Ticker1'] in sorted_all_tickers:
                            default_s1 = sorted_all_tickers.index(tp['Ticker1'])
                        if tp['Ticker2'] in sorted_all_tickers:
                            default_s2 = sorted_all_tickers.index(tp['Ticker2'])

            st.markdown("#### 🅰️ 標的 A (Leg 1)")
            sym1 = st.selectbox("標的 A (Leg 1)", sorted_all_tickers, index=default_s1, format_func=fmt_ticker_grouped, key='sym1', label_visibility="collapsed")
            st.markdown(render_sidebar_stock_badge(stock_profiles[sym1], sym1, "A", "sky"), unsafe_allow_html=True)

            st.markdown("#### 🅱️ 標的 B (Leg 2)")
            sym2 = st.selectbox("標的 B (Leg 2)", sorted_all_tickers, index=default_s2, format_func=fmt_ticker_grouped, key='sym2', label_visibility="collapsed")
            st.markdown(render_sidebar_stock_badge(stock_profiles[sym2], sym2, "B", "pink"), unsafe_allow_html=True)

        # 側邊欄配對關係總結 Pill
        st.markdown(render_sidebar_pair_diagnostic(stock_profiles[sym1], stock_profiles[sym2], sym1, sym2, sub_pairs), unsafe_allow_html=True)

        if sym1 == sym2:
            st.warning("請選擇兩檔不同的標的！")
            return

        st.markdown("---")
        st.markdown("### 📅 回測期間")
        import datetime
        min_date = prices.index.min().date()
        latest_date = prices.index.max().date()
        today_date = datetime.date.today()
        
        # 預設起點設為 2022-01-01
        default_start = datetime.date(2022, 1, 1) if datetime.date(2022, 1, 1) >= min_date else min_date
        
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("開始日期", min_value=min_date, max_value=today_date, value=default_start)
        with col_end:
            # 預設終點設為今天
            end_date = st.date_input("結束日期", min_value=min_date, max_value=today_date, value=today_date)

        st.markdown("---")
        st.markdown("### 💰 保證金與資金管理")
        initial_capital = st.number_input(
            "初始總保證金 (TWD)",
            min_value=10000,
            max_value=100000000,
            value=1000000,
            step=50000,
            format="%d"
        )
        
        st.markdown("### 💰 資金滿載設定")
        size_mode = st.radio("資金佈局模式", 
                             options=['依保證金上限最大化（預設）', '最少配對口數 (僅基本單位)'],
                             index=0,
                             help="選擇是否要讓系統自動把資金運用到極限")
        
        margin_usage_pct = 1.0
        fixed_pair_multiplier = 1
        if size_mode == '依保證金上限最大化（預設）':
            margin_usage_pct = st.slider("最高保證金利用率 (%)", 0, 100, 100, 5, help="限制這筆資金最高只能被利用的比例") / 100.0
        else:
            fixed_pair_multiplier = st.slider("固定配對口數倍數", 1, 10, 1, 1, help="調整基本單位的倍數 (例如設定2代表每次進場2組基本配對)")

        st.markdown("---")
        st.markdown("### 🧭 交易邏輯與方向")
        trade_mode = st.radio("配對策略邏輯", ['收斂 (均值回歸)', '發散 (趨勢跟蹤)'], index=0, help="收斂：突破上界做空，跌破下界做多；發散：突破上界做多，跌破下界做空。")
        use_log_price = st.toggle("對數價格轉換 (Log Price)", value=False, help="將價格取自然對數後再計算價差。能將比例關係轉為加法關係，適合長期配對或價格落差大的標的。")
        
        st.markdown("---")
        st.markdown("### 📐 數學模型與參數設定")
        
        ensemble_logic = '單一模型'
        selected_models = []
        if trade_mode == '收斂 (均值回歸)':
            model_options = ['Z-Score (標準)', 'OU 過程 (動態邊界)', '共整合 + GARCH', '卡爾曼濾波 (動態對沖比例)', 'Copula (CMPI 機率)', 'Merton 跳躍擴散模型 (過濾結構破裂)', 'SDDE 隨機延遲方程式 (過濾動能慣性)']
            selected_models = st.multiselect("收斂訊號組合 (可多選)", model_options, default=[model_options[0]])
            if not selected_models:
                st.warning("請至少選擇一個收斂模型！")
            if len(selected_models) > 1:
                ensemble_logic = st.radio("組合判定邏輯", ['交集 (AND) - 嚴格過濾', '聯集 (OR) - 寬鬆捕捉'], index=0)
            else:
                ensemble_logic = '單一模型'
        else:
            model_options = ['GSADF (爆炸性泡沫檢定)', '非參數 CUSUM (多變量幾何破裂)', 'DCC-GARCH-VECM (特異性漂移爆發)', 'MRS (馬爾可夫區制轉換)']
            selected_models = st.multiselect("發散訊號組合 (可多選)", model_options, default=[model_options[0]])
            if not selected_models:
                st.warning("請至少選擇一個發散模型！")
            if len(selected_models) > 1:
                ensemble_logic = st.radio("組合判定邏輯", ['交集 (AND) - 嚴格過濾', '聯集 (OR) - 寬鬆捕捉'], index=0)
            else:
                ensemble_logic = '單一模型'
        
        model_params = {}
        for idx, model_type in enumerate(selected_models):
            st.markdown(f"#### {idx+1}. {model_type} 參數")
            m_params = {}
            if model_type == 'Z-Score (標準)':
                m_params['z_entry'] = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", 0.1, 5.0, 2.0, 0.1, help="調整進場的敏銳度，數值越小交易越頻繁", key=f"z_entry_{idx}_{model_type}")
                m_params['z_exit'] = st.slider("平倉閾值 (Z 回歸)", -2.0, 2.0, 0.0, 0.1, help="當指標回歸到此數值時平倉", key=f"z_exit_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, help="用於計算標準差", key=f"z_window_{idx}_{model_type}")
            elif model_type == '共整合 + GARCH':
                m_params['z_entry'] = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", 0.1, 5.0, 2.0, 0.1, help="調整進場的敏銳度，數值越小交易越頻繁", key=f"z_entry_{idx}_{model_type}")
                m_params['z_exit'] = st.slider("平倉閾值 (Z 回歸)", -2.0, 2.0, 0.0, 0.1, help="當指標回歸到此數值時平倉", key=f"z_exit_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, help="用於計算標準差", key=f"z_window_{idx}_{model_type}")
                st.markdown("##### GARCH 模型進階參數")
                m_params['garch_p'] = st.slider("GARCH(p) 階數", 1, 3, 1, 1, help="控制過去變異數的影響力", key=f"gp_{idx}_{model_type}")
                m_params['garch_q'] = st.slider("GARCH(q) 階數", 1, 3, 1, 1, help="控制過去殘差的影響力", key=f"gq_{idx}_{model_type}")
                m_params['garch_dist'] = st.selectbox("殘差分配假設 (Distribution)", ['Normal', 't', 'skewt'], index=0, help="選擇厚尾分佈可更快適應極端行情", key=f"gd_{idx}_{model_type}")
            elif model_type == '卡爾曼濾波 (動態對沖比例)':
                m_params['z_entry'] = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", 0.1, 5.0, 2.0, 0.1, help="調整進場的敏銳度，數值越小交易越頻繁", key=f"z_entry_{idx}_{model_type}")
                m_params['z_exit'] = st.slider("平倉閾值 (Z 回歸)", -2.0, 2.0, 0.0, 0.1, help="當指標回歸到此數值時平倉", key=f"z_exit_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, help="用於計算卡爾曼標準化", key=f"z_window_{idx}_{model_type}")
                st.markdown("##### 卡爾曼濾波進階參數")
                q_options = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
                q_format = {1e-2: "1e-2 (極高靈敏)", 1e-3: "1e-3 (高靈敏)", 1e-4: "1e-4 (標準/常用)", 1e-5: "1e-5 (平滑)", 1e-6: "1e-6 (極度平滑)"}
                m_params['kf_q'] = st.selectbox("過程雜訊協方差 (Q)", q_options, index=2, format_func=lambda x: q_format[x], help="決定對沖比率隨時間變動的速度。調大會對新數據更敏感，但也更容易受短期雜訊干擾。", key=f"kf_q_{idx}_{model_type}")
                m_params['kf_r'] = st.selectbox("量測雜訊協方差 (R)", [1e-1, 1e-2, 1e-3, 1e-4], index=2, format_func=lambda x: f"{x:g}", help="代表價格本身的隨機波動大小。通常固定為 1e-3，與 Q 的比例決定濾波特徵。", key=f"kf_r_{idx}_{model_type}")
                m_params['kf_p0'] = st.number_input("初始狀態協方差 (P0)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="影響剛開始運行時的收斂速度。數值大代表初期修正快，但隨時間推移影響遞減。", key=f"kf_p0_{idx}_{model_type}")
            elif model_type == 'OU 過程 (動態邊界)':
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, help="用於擬合 OU 過程參數 (Theta, Mu, Sigma)", key=f"ou_window_{idx}_{model_type}")
                st.markdown("##### OU 進階參數")
                m_params['risk_free_rate'] = st.number_input("無風險利率", min_value=0.000, max_value=0.050, value=0.015, step=0.001, format="%.3f", help="影響進場邊界計算", key=f"rf_{idx}_{model_type}")
                m_params['trading_fee'] = st.number_input("單邊手續費率", min_value=0.0000, max_value=0.0100, value=0.0004, step=0.0001, format="%.4f", help="影響平倉邊界寬度", key=f"tf_{idx}_{model_type}")
                st.info("OU 模型會自動根據均值回歸速度(Theta)、波動率及上方設定的資金成本計算動態上下界，無須手動設定固定閾值。")
            elif model_type == 'Copula (CMPI 機率)':
                m_params['prob_threshold'] = st.slider("條件機率閾值 (CMPI)", 0.500, 0.999, 0.950, 0.001, format="%.3f", help="達到多少極端機率才開倉 (0.5以上)", key=f"cp_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, help="用於擬合 Copula 相關性與累積分配函數", key=f"cw_{idx}_{model_type}")
                st.markdown("##### Copula 進階參數")
                m_params['copula_clip_bounds'] = st.number_input("極端值截斷閾值", min_value=0.0001, max_value=0.0500, value=0.0010, step=0.0001, format="%.4f", help="避免 Infinity 的百分位數極限", key=f"cb_{idx}_{model_type}")
            elif model_type == 'Merton 跳躍擴散模型 (過濾結構破裂)':
                m_params['z_entry'] = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", 0.1, 5.0, 2.0, 0.1, key=f"je_{idx}_{model_type}")
                m_params['z_exit'] = st.slider("平倉閾值 (Z 回歸)", -2.0, 2.0, 0.0, 0.1, key=f"jx_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, key=f"jw_{idx}_{model_type}")
                m_params['jump_threshold'] = st.slider("跳躍過濾敏感度 (標準差倍數)", 1.0, 10.0, 3.0, 0.1, help="當日波動超過此倍數即判定為異常跳空，強制暫停該日交易。數值越低過濾越嚴格。", key=f"jt_{idx}_{model_type}")
                st.info(" Jump-Diffusion 會在偵測到極端異常跳空時，動態將上下界設為無窮大以封鎖開倉。")
            elif model_type == 'SDDE 隨機延遲方程式 (過濾動能慣性)':
                m_params['z_entry'] = st.slider("開倉閾值 (Z絕對值, 進場 ±Z)", 0.1, 5.0, 2.0, 0.1, key=f"se_{idx}_{model_type}")
                m_params['z_exit'] = st.slider("平倉閾值 (Z 回歸)", -2.0, 2.0, 0.0, 0.1, key=f"sx_{idx}_{model_type}")
                m_params['z_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, key=f"sw_{idx}_{model_type}")
                m_params['delay_tau'] = st.slider("歷史記憶天數 (τ)", 1, 20, 5, 1, help="計算過去幾天的發散動能。動能越強，模型會主動延後進場時機。", key=f"st_{idx}_{model_type}")
                st.info(" SDDE 模型會計算時間延遲積分，自動將帶有強大動能的『假突破』Z分數縮小，避免過早進場。")
            elif model_type == '非參數 CUSUM (多變量幾何破裂)':
                st.markdown("##### 非參數 CUSUM 參數")
                m_params['cusum_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, key=f"cw2_{idx}_{model_type}")
                m_params['k_shift'] = st.slider("中位數漂移閾值 (MAD 倍數)", 0.1, 3.0, 1.0, 0.1, key=f"ck_{idx}_{model_type}")
                m_params['tau_threshold'] = st.slider("停止時間閾值 (τ)", 1.0, 20.0, 5.0, 0.5, key=f"ct_{idx}_{model_type}")
            elif model_type == 'GSADF (爆炸性泡沫檢定)':
                st.markdown("##### GSADF 參數")
                m_params['gsadf_window'] = st.slider("滾動窗口 (天)", 15, 500, 30, 1, key=f"gw_{idx}_{model_type}")
                m_params['adf_threshold'] = st.slider("ADF t-統計量 閾值", 0.5, 4.0, 1.5, 0.1, key=f"gt_{idx}_{model_type}")
                st.markdown("##### GSADF 進階參數")
                m_params['min_window_pct'] = st.slider("初始樣本涵蓋比例", 0.1, 0.5, 0.2, 0.05, help="決定 SADF 內部擴展檢定的最小 K 棒數量比例", key=f"gm_{idx}_{model_type}")
            elif model_type == 'MRS (馬爾可夫區制轉換)':
                st.markdown("##### MRS 參數")
                m_params['mrs_window'] = st.slider("滾動窗口 (天)", 30, 500, 120, 5, key=f"mw_{idx}_{model_type}")
                m_params['prob_threshold'] = st.slider("發散區制機率閾值", 0.5, 0.99, 0.8, 0.05, key=f"mp_{idx}_{model_type}")
                st.markdown("##### MRS 進階參數")
                m_params['switching_variance'] = st.toggle("切換變異數 (Switching Variance)", value=True, help="決定兩個市場狀態 (Regimes) 的波動率是否允許不同", key=f"sv_{idx}_{model_type}")
                st.info("⚠️ MRS 模型內部使用 MLE 估計轉移矩陣，計算極度耗時。")
            elif model_type == 'DCC-GARCH-VECM (特異性漂移爆發)':
                st.markdown("##### DCC-GARCH 參數")
                m_params['garch_window'] = st.slider("滾動窗口 (天)", 5, 500, 20, 1, key=f"dc_w_{idx}_{model_type}")
                m_params['t_threshold'] = st.slider("爆發 t-統計量閾值", 1.0, 10.0, 3.0, 0.5, key=f"dc_t_{idx}_{model_type}")
                st.markdown("##### DCC-GARCH 進階參數")
                _vol_span = st.number_input("條件波動率半衰期 (0=自動)", min_value=0, max_value=100, value=0, step=1, help="GARCH(1,1) 的 EMA 逼近參數。設 0 為自動 (Window/2)", key=f"dv_{idx}_{model_type}")
                _drift_span = st.number_input("局部漂移半衰期 (0=自動)", min_value=0, max_value=50, value=0, step=1, help="漂移項計算。設 0 為自動 (Window/5)", key=f"dd_{idx}_{model_type}")
                m_params['dcc_span_vol'] = _vol_span if _vol_span > 0 else None
                m_params['dcc_span_drift'] = _drift_span if _drift_span > 0 else None
            model_params[model_type] = m_params

        st.markdown("---")

        st.markdown("---")
        st.markdown("### 🧪 進階過濾機制 (OU/Beta)")
        use_ou_filter = st.toggle("啟用 OU Regime Filter", value=False, help="強制要求配對關係在統計上必須具有『均值回歸』特性才允許開倉。\n👉 影響：會大幅過濾掉交易次數，但能避開那些陷入單邊發散趨勢的標的，提升勝率。")
        ou_window = 60
        ou_min_r2 = 0.2
        ou_min_hl = 1.0
        ou_max_hl = 30.0
        if use_ou_filter:
            ou_window = st.slider("OU Rolling Window", 20, 120, 60, 5, help="計算半衰期時的回看天數")
            ou_min_r2 = st.slider("最小 R² 門檻", 0.0, 1.0, 0.2, 0.05, help="👉 影響：要求 OU 模型對價差必須有基本的解釋力，R² 越高代表回歸特徵越明顯。")
            ou_hl_range = st.slider("合理 Half-life 區間 (天)", 1.0, 100.0, (1.0, 30.0), 1.0, help="👉 影響：太短(<1)可能是雜訊；太長(>30)代表收斂太慢資金容易卡住。")
            ou_min_hl, ou_max_hl = ou_hl_range
            
        use_beta_transition = st.toggle("啟用 Beta Transition Filter", value=False, help="監控動態 Beta 的短期與中期斜率。\n👉 影響：若斜率同向且短期急遽放大，代表兩檔股票的基本結構可能正在發生永久性破裂，系統會強制關閉新倉權限，攔截極端回撤風險。")
        beta_short = 20
        beta_mid = 100
        if use_beta_transition:
            beta_short = st.slider("Beta 短期斜率視窗 (天)", 5, 60, 20, 5)
            beta_mid = st.slider("Beta 中期斜率視窗 (天)", 20, 200, 100, 10)

        st.markdown("---")
        st.markdown("### 🛡️ 智能轉倉防呆機制")
        use_smart_roll = st.toggle("啟用轉倉防呆", value=False, help="若結算時帳上有獲利，且 Z-score 已經收斂至設定門檻內，則結算平倉後次日不再自動開新倉（等同提早獲利了結，避免微薄利潤被轉倉成本吃掉）。")
        smart_roll_z = 1.0
        if use_smart_roll:
            smart_roll_z = st.slider("收斂過濾門檻 (Z-score)", 0.1, 2.9, 1.0, 0.1)

        st.markdown("---")
        st.markdown("### 🕸️ 網格配對交易設定")
        use_grid = st.toggle("啟用獨立網格模式", value=True, help="每一層網格獨立運作，擁有專屬的進場、停利、停損與重新建倉門檻。若關閉，則退回單一進出場模式，並自動套用上方高階模型的動態邊界。")
        
        grid_levels = []
        if use_grid:
            st.info("⚠️ 啟用此功能後，將優先採用以下網格設定，忽略高階模型的浮動邊界。")
            num_grids = st.number_input("總共分幾層網格?", value=2, min_value=1, max_value=10, step=1)
            for i in range(int(num_grids)):
                st.markdown(f"**第 {i+1} 層網格**")
                col1, col2 = st.columns(2)
                with col1:
                    entry_z = st.number_input(f"進場門檻 (Z)", value=1.0 + i*1.0, step=0.1, key=f"g_ent_{i}")
                    tp_z = st.number_input(f"停利門檻 (Z)", value=0.0 + i*1.0, step=0.1, key=f"g_tp_{i}")
                    reentry_z = st.number_input(f"停損後重啟門檻 (Z)", value=entry_z, step=0.1, key=f"g_re_{i}")
                with col2:
                    pairs = st.number_input(f"口數 (組)", value=1, min_value=1, step=1, key=f"g_pairs_{i}")
                    sl_z = st.number_input(f"停損門檻 (Z)", value=3.0, step=0.1, key=f"g_sl_{i}")
                
                grid_levels.append({
                    'id': i+1,
                    'entry_z': entry_z,
                    'tp_z': tp_z,
                    'sl_z': sl_z,
                    'reentry_z': reentry_z,
                    'pairs': pairs
                })

        st.markdown("### 📋 股期規格")
        st.info(f"""
        - **1 口 = {SHARES_PER_CONTRACT:,} 股 (2 張)**
        - **保證金比率: {MARGIN_RATE*100:.1f}%**
        - **期交稅: {TRADING_FEE_RATE*100000:.0f}/100,000 (單邊)**
        - **手續費: {COMMISSION_PER_CONTRACT} 元/口 (單邊)**
        - **漲跌停: ±{LIMIT_PCT*100:.0f}% 不可交易**
        - **結算日: 每月第3個禮拜三**
        - **轉倉: 結算平倉→次日開倉**
        """)

        st.markdown("---")
        run_btn = st.button("🚀 執行回測", type="primary", use_container_width=True)

    # ============================================================
    # 主面板: 執行結果
    # ============================================================
    if run_btn:
        S1 = prices[sym1].dropna()
        S2 = prices[sym2].dropna()

        # 確保有共同日期以供模型訓練 (需要全局數據)
        common_idx = S1.index.intersection(S2.index)
        if len(common_idx) < 120:  # 至少需要兩倍滾動窗口的資料量
            st.error(f"這兩檔標的全局共同有效的交易日過少 ({len(common_idx)} 天)，無法建立計算模型！")
            return
            
        S1 = S1[common_idx]
        S2 = S2[common_idx]

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        with st.spinner("正在執行高階回測分析..."):
            name1 = name_map.get(sym1.replace('.TWO', '').replace('.TW', ''), '')
            name2 = name_map.get(sym2.replace('.TWO', '').replace('.TW', ''), '')
            advanced_params = {
                'use_ou_filter': use_ou_filter,
                'ou_window': ou_window,
                'ou_min_r2': ou_min_r2,
                'ou_min_hl': ou_min_hl,
                'ou_max_hl': ou_max_hl,
                'use_beta_transition': use_beta_transition,
                'beta_short': beta_short,
                'beta_mid': beta_mid,
                'use_smart_roll': use_smart_roll,
                'smart_roll_z': smart_roll_z,
                'use_grid': use_grid,
                'grid_levels': grid_levels
            }
            
            advanced_params['trade_mode'] = trade_mode  # Pass trade_mode via advanced_params
            advanced_params['fixed_pair_multiplier'] = fixed_pair_multiplier
            
            df_records, df_trades, stats = run_backtest(
        S1, S2, sym1, sym2, name1, name2, initial_capital, 
        selected_models, model_params,
        size_mode=size_mode, margin_usage_pct=margin_usage_pct,
                run_start_date=start_str, run_end_date=end_str,
                advanced_params=advanced_params
            )

        # --- 🏢 標的細產業分類與產品營收結構深度對比 ---
        industry_df = get_industry_data()
        prof1 = get_stock_profile(sym1, industry_df)
        prof2 = get_stock_profile(sym2, industry_df)
        
        st.markdown("## 🏢 標的細產業分類與產品營收結構深度對比")
        
        # 1. 兩檔標的並排卡片
        col_card1, col_card2 = st.columns(2)
        with col_card1:
            st.markdown(render_stock_card(prof1), unsafe_allow_html=True)
        with col_card2:
            st.markdown(render_stock_card(prof2), unsafe_allow_html=True)

        # 2. 產業同質性與走勢連動診斷 Banner
        p1_common = S1.loc[S1.index.intersection(S2.index)]
        p2_common = S2.loc[p1_common.index]
        price_corr = p1_common.corr(p2_common)
        ret_corr = p1_common.pct_change().dropna().corr(p2_common.pct_change().dropna())
        coint_p = stats.get('coint_pvalue')
        hl_str = f"{stats.get('avg_ou_half_life'):.1f} 天" if stats.get('avg_ou_half_life') else "無顯著回歸"

        st.markdown(render_co_movement_banner(prof1, prof2, price_corr=price_corr, ret_corr=ret_corr, coint_p=coint_p, hl_str=hl_str), unsafe_allow_html=True)

        if stats.get('insufficient_margin_error', False):
            if 'msg' in stats:
                st.error(f"""❌ **回測執行攔截：**

{stats['msg']}""")
            else:
                st.error(f"""❌ **保證金不足無法下單！**

您設定的可用保證金 ({initial_capital * margin_usage_pct:,.0f} 元) 不足以下單即使是最少的 {stats.get('base_s1',0)}口/{stats.get('base_s2',0)}口 (需 {stats.get('base_margin',0):,.0f} 元)。請增加資金或選擇較低價標的。""")
            return

        # --- 保證金計算摘要 ---
        st.markdown("## 📋 配對口數與保證金計算")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Hedge Ratio (β)</div>
                <div class="value">{stats['hedge_ratio']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">最小等張口數需求</div>
                <div class="value">{sym1}({name1})×{stats['base_s1']} : {sym2}({name2})×{stats['base_s2']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">最低一組基本保證金</div>
                <div class="value">{stats['base_margin']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            pct_used = stats['margin_utilization']
            color_cls = 'positive' if pct_used < 95 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">保證金利用率</div>
                <div class="value {color_cls}">{pct_used:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 詳細計算
        with st.expander("📐 詳細保證金計算", expanded=False):
            calc_col1, calc_col2 = st.columns(2)
            with calc_col1:
                st.markdown(f"""
                **{sym1}** (起始價: {stats['p1_start']:.2f})
                - 合約價值 = {stats['p1_start']:.2f} × {SHARES_PER_CONTRACT:,} = **{calc_contract_value(stats['p1_start']):,.0f}**
                - {stats['base_s1']} 口基本保證金 = {calc_contract_value(stats['p1_start']):,.0f} × {MARGIN_RATE*100:.1f}% × {stats['base_s1']} = **{stats['margin_s1']/max(1, stats['multiplier']):,.0f}**
                """)
            with calc_col2:
                st.markdown(f"""
                **{sym2}** (起始價: {stats['p2_start']:.2f})
                - 合約價值 = {stats['p2_start']:.2f} × {SHARES_PER_CONTRACT:,} = **{calc_contract_value(stats['p2_start']):,.0f}**
                - {stats['base_s2']} 口基本保證金 = {calc_contract_value(stats['p2_start']):,.0f} × {MARGIN_RATE*100:.1f}% × {stats['base_s2']} = **{stats['margin_s2']/max(1, stats['multiplier']):,.0f}**
                """)
            st.markdown(f"*(註：上述為回測期初起算時之一倍基本組合。實單會隨著帳戶總資金動態擴大倍數)*")

        if stats['base_margin'] > initial_capital:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ 初始資金 {initial_capital:,} 元不足以開出一組基本配對（需 {stats['base_margin']:,.0f} 元）！
                請提高初始保證金或選擇較低價的標的。
            </div>
            """, unsafe_allow_html=True)
        elif stats.get('multiplier', 0) < 1 and size_mode == '依保證金上限最大化（預設）':
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ 您設定的保證金利用率極限較低，分配到的起步資金不足以負荷一組配對（需 {stats['base_margin']:,.0f} 元）。
                回測在資金或股價達標前可能會保持空手。
            </div>
            """, unsafe_allow_html=True)

        # --- 績效指標 ---
        st.markdown("## 📊 回測績效摘要")

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        with m1:
            ret_cls = 'positive' if stats['total_return'] >= 0 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">累計報酬</div>
                <div class="value {ret_cls}">{stats['total_return']:+,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">報酬率</div>
                <div class="value {ret_cls}">{stats['total_return_pct']:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">年化報酬率</div>
                <div class="value {'positive' if stats['annualized_return'] >= 0 else 'negative'}">{stats['annualized_return']:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            wr_cls = 'positive' if stats['win_rate'] >= 50 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">勝率</div>
                <div class="value {wr_cls}">{stats['win_rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">交易次數</div>
                <div class="value">{stats['total_trades']}</div>
            </div>
            """, unsafe_allow_html=True)
        with m6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">最大回撤 MDD</div>
                <div class="value negative">{stats['max_drawdown_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 更多指標
        st.markdown("")
        more1, more2, more3, more4, more5, more6 = st.columns(6)
        with more1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">平均獲利</div>
                <div class="value positive">{stats['avg_win']:+,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with more2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">平均虧損</div>
                <div class="value negative">{stats['avg_lose']:+,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with more3:
            pf = stats['profit_factor']
            pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">獲利因子</div>
                <div class="value">{pf_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with more4:
            sharpe_str = f"{stats['sharpe_ratio']:.2f}"
            sharpe_cls = 'positive' if stats['sharpe_ratio'] > 1.0 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">夏普比率</div>
                <div class="value {sharpe_cls}">{sharpe_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with more5:
            hl_str = f"{stats['avg_ou_half_life']:.1f}天" if stats['avg_ou_half_life'] is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">OU半衰期</div>
                <div class="value">{hl_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with more6:
            coint_str = f"{stats['coint_pvalue']:.6f}" if stats['coint_pvalue'] is not None else "N/A"
            coint_cls = 'positive' if stats['coint_pvalue'] is not None and stats['coint_pvalue'] < 0.05 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">共整合 P-value</div>
                <div class="value {coint_cls}">{coint_str}</div>
            </div>
            """, unsafe_allow_html=True)

        # 結算轉倉 & 漲跌停統計
        info_parts = []
        if stats['settlement_rolls'] > 0:
            info_parts.append(
                f"每月結算轉倉 <b>{stats['settlement_rolls']}</b> 次，"
                f"累計轉倉成本 <b>{stats['total_roll_cost']:,.0f}</b> 元"
            )
        if stats['skipped_limits'] > 0:
            info_parts.append(
                f"漲跌停跳過 <b>{stats['skipped_limits']}</b> 次"
            )
        if info_parts:
            st.markdown(f"""
            <div class="info-box">
                📌 {'　｜　'.join(info_parts)}
            </div>
            """, unsafe_allow_html=True)

        # --- 圖表 ---
        st.markdown("## 📈 回測圖表")

        # (1) 股價走勢
        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price.add_trace(
            go.Scatter(x=df_records.index, y=df_records[f'{sym1}價'],
                       name=sym1, line=dict(color='#2196F3', width=1.5)),
            secondary_y=False
        )
        fig_price.add_trace(
            go.Scatter(x=df_records.index, y=df_records[f'{sym2}價'],
                       name=sym2, line=dict(color='#FF9800', width=1.5)),
            secondary_y=True
        )
        fig_price.update_layout(
            title=f'{sym1} vs {sym2} 股價走勢',
            template='plotly_dark', height=350,
            margin=dict(l=60, r=60, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_price.update_yaxes(title_text=sym1, secondary_y=False)
        fig_price.update_yaxes(title_text=sym2, secondary_y=True)
        st.plotly_chart(fig_price, use_container_width=True)

        # (2) 指標 + 訊號
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Z-Score'],
            name='指標 (Indicator)', line=dict(color='#BB86FC', width=1),
            fill='tozeroy', fillcolor='rgba(187,134,252,0.1)'
        ))
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Upper_Bound'],
            name='上界 (做空訊號)', line=dict(color='red', width=1, dash='dash')
        ))
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Lower_Bound'],
            name='下界 (做多訊號)', line=dict(color='green', width=1, dash='dash')
        ))
        # 繪製平倉線
        fig_z.add_trace(go.Scatter(
            x=df_records.index, y=df_records['Exit_Upper'],
            name='平倉線上界', line=dict(color='gray', width=1, dash='dot')
        ))
        if not (df_records['Exit_Upper'] == df_records['Exit_Lower']).all():
            fig_z.add_trace(go.Scatter(
                x=df_records.index, y=df_records['Exit_Lower'],
                name='平倉線下界', line=dict(color='gray', width=1, dash='dot')
            ))

        if not df_trades.empty:
            opens = df_trades[df_trades['動作'].str.contains('開倉')]
            closes = df_trades[df_trades['動作'].str.contains('平倉')]
            if not opens.empty:
                fig_z.add_trace(go.Scatter(
                    x=pd.to_datetime(opens['日期']), y=opens['Z-Score'],
                    mode='markers', name='開倉',
                    marker=dict(symbol='circle', size=10, color='#00BCD4',
                                line=dict(width=1, color='white'))
                ))
            if not closes.empty:
                fig_z.add_trace(go.Scatter(
                    x=pd.to_datetime(closes['日期']), y=closes['Z-Score'],
                    mode='markers', name='平倉',
                    marker=dict(symbol='x', size=10, color='#FFD700',
                                line=dict(width=2, color='white'))
                ))

        fig_z.update_layout(
            title='Z-Score 與進出場訊號',
            template='plotly_dark', height=350,
            margin=dict(l=60, r=60, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_z, use_container_width=True)

        # (3) 損益追蹤
        fig_pnl = go.Figure()
        unrealized_colors = ['rgba(100,255,218,0.3)' if v >= 0 else 'rgba(255,107,107,0.3)'
                            for v in df_records['未實現損益']]
        fig_pnl.add_trace(go.Bar(
            x=df_records.index, y=df_records['未實現損益'],
            name='未實現損益', marker_color=unrealized_colors,
            opacity=0.5
        ))
        fig_pnl.add_trace(go.Scatter(
            x=df_records.index, y=df_records['累計已實現損益'],
            name='累計已實現損益', line=dict(color='#2196F3', width=2)
        ))
        fig_pnl.add_hline(y=0, line_color="gray", line_width=0.5)
        fig_pnl.update_layout(
            title='損益追蹤 (未實現 + 已實現)',
            template='plotly_dark', height=350,
            margin=dict(l=60, r=60, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        # (4) 帳戶淨值 + 保證金
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=df_records.index, y=df_records['帳戶淨值'],
            name='帳戶淨值', line=dict(color='#64FFDA', width=2),
            fill='tonexty' if False else None
        ))
        fig_eq.add_trace(go.Scatter(
            x=df_records.index, y=df_records['凍結保證金'],
            name='凍結保證金', fill='tozeroy',
            fillcolor='rgba(255,152,0,0.2)',
            line=dict(color='#FF9800', width=1)
        ))
        fig_eq.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                         annotation_text=f"初始資金 {initial_capital:,}")
        fig_eq.update_layout(
            title='帳戶淨值 & 凍結保證金',
            template='plotly_dark', height=350,
            margin=dict(l=60, r=60, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # --- 交易明細表 ---
        st.markdown("## 📝 交易明細")

        if not df_trades.empty:
            def style_trades(row):
                if '平倉' in str(row.get('動作', '')):
                    if row.get('損益', 0) > 0:
                        return ['background-color: rgba(100, 255, 218, 0.15)'] * len(row)
                    else:
                        return ['background-color: rgba(255, 107, 107, 0.15)'] * len(row)
                return [''] * len(row)

            styled = df_trades.style.apply(style_trades, axis=1).format({
                '交易成本': '{:,.0f}',
                '損益': '{:+,.0f}',
                'Z-Score': '{:.2f}'
            })
            st.dataframe(styled, use_container_width=True, height=400)
        else:
            st.info("回測期間無任何交易發生。")

        # --- 下載 ---
        st.markdown("## 💾 下載結果")
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_trades = df_trades.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載交易明細 CSV", csv_trades,
                             f"trade_log_{sym1}_{sym2}.csv", "text/csv")
        with dl2:
            csv_records = df_records.to_csv().encode('utf-8-sig')
            st.download_button("📥 下載每日紀錄 CSV", csv_records,
                             f"daily_records_{sym1}_{sym2}.csv", "text/csv")

    else:
        # 尚未執行回測，即時展示當前選定標的之細產業與產品營收結構深度對比
        prof1 = stock_profiles.get(sym1) or get_stock_profile(sym1, industry_df)
        prof2 = stock_profiles.get(sym2) or get_stock_profile(sym2, industry_df)

        st.markdown(f"## 🎯 當前選定配對：{prof1['code']} {prof1['name']}  ⚡  {prof2['code']} {prof2['name']}")
        st.caption("👈 透過左側側邊欄自由挑選或一鍵帶入，中央即時預覽兩檔標的之實質產品營收佔比、細產業所屬與基本面連動關係：")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(render_stock_card(prof1), unsafe_allow_html=True)
        with col_c2:
            st.markdown(render_stock_card(prof2), unsafe_allow_html=True)

        # 快速取得或計算兩者之走勢連動指標
        p_corr = None
        r_corr = None
        c_pval = None
        m_pairs = [p for p in sub_pairs if (p['Ticker1'] == sym1 and p['Ticker2'] == sym2) or (p['Ticker1'] == sym2 and p['Ticker2'] == sym1)]
        if m_pairs:
            p_corr = m_pairs[0].get('PriceCorr')
            r_corr = m_pairs[0].get('ReturnCorr')
            c_pval = m_pairs[0].get('CointPValue')
        elif prices is not None and sym1 in prices.columns and sym2 in prices.columns:
            s1_sub = prices[sym1].dropna()
            s2_sub = prices[sym2].dropna()
            comm = s1_sub.index.intersection(s2_sub.index)[-250:]
            if len(comm) > 30:
                p_corr = float(s1_sub[comm].corr(s2_sub[comm]))
                r_corr = float(s1_sub[comm].pct_change().corr(s2_sub[comm].pct_change()))
            if len(comm) > 60:
                try:
                    _, c_pval, _ = coint(s1_sub[comm], s2_sub[comm])
                except:
                    pass

        st.markdown(render_co_movement_banner(prof1, prof2, price_corr=p_corr, ret_corr=r_corr, coint_p=c_pval), unsafe_allow_html=True)

        st.info("👈 **標的已就緒！** 可在左側側邊欄調整交易策略、模型與資金管理設定，並點擊「🚀 執行回測」開始完整運算。")

        with st.expander("📌 配對交易規則、保證金機制與操作指引", expanded=False):
            st.markdown("""
            <b>🎯 使用方式</b><br>
            1. 從左側選單中選擇兩檔想配對的股期標的（系統預設提供同細產業高協整與高相關配對）<br>
            2. 設定初始保證金金額與資金佈局模式<br>
            3. 選擇數學模型（收斂模型或發散模型）並調整敏感度參數<br>
            4. 點擊「🚀 執行回測」查看累積損益、夏普值、回撤與逐日交易明細<br>
            <br>
            <b>📌 交易規則</b><br>
            • 指標 > 上界閾值 → 做空 Spread（空標的B + 多標的A）<br>
            • 指標 < 下界閾值 → 做多 Spread（多標的B + 空標的A）<br>
            • 指標回歸平倉線 → 平倉<br>
            • 漲跌停日（±10%）自動跳過，不會產生無法成交的虛假訊號<br>
            • 每月第 3 個禮拜三結算日自動平倉，次日以新價格重新建倉（含轉倉成本）
            """, unsafe_allow_html=True)

        # 0. 🔥 獲利因子大於 3.0 同細產業高勝率精選配對總表
        if high_pf_df is not None and not high_pf_df.empty:
            st.markdown("### 🔥 獲利因子大於 3.0 同細產業精選配對排行榜")
            st.markdown("""
            全市場 50 個同細產業族群、**1,094 組同業配對**全面嚴格回測（**基礎 Z-Score 模型 · 固定 100 萬保證金 · 最少配對 1 組基本單位**），
            篩選出 **獲利因子 (Profit Factor) > 3.0** 之高勝率、低回撤同業配對：
            """)
            display_pf_rows = []
            for _, r in high_pf_df.iterrows():
                display_pf_rows.append({
                    '細產業族群': r['SubIndustry'],
                    '配對組合': f"{r['Name1']} ({r['Ticker1']}) vs {r['Name2']} ({r['Ticker2']})",
                    '獲利因子': f"🔥 {r['ProfitFactor']:.2f}",
                    '總報酬 (元)': f"+{r['TotalReturn']:,.0f}",
                    '總報酬率': f"+{r['TotalReturnPct']:.1f}%",
                    '年化報酬': f"+{r['AnnualizedReturn']:.1f}%",
                    '勝率': f"{r['WinRate']:.1f}%",
                    '交易次數': f"{r['TotalTrades']} 次 ({r['WinTrades']}勝/{r['LoseTrades']}敗)",
                    '最大回撤 (MDD)': f"{r['MaxDrawdownPct']:.1f}%",
                    '夏普值': f"{r['SharpeRatio']:.2f}",
                    '基本口數': r['BaseContracts'],
                    '基本保證金': f"{r['BaseMargin']:,.0f} 元",
                    'A 主力產品': r['Prod1'],
                    'B 主力產品': r['Prod2'],
                    '走勢相關度': f"{r['PriceCorr']:+.2f}",
                    '協整 p值': f"{r['CointPValue']:.4f}"
                })
            st.dataframe(pd.DataFrame(display_pf_rows), use_container_width=True, height=360)
            st.markdown("---")

        # 1. 🏆 推薦同細產業高協整配對排行榜
        st.markdown("### 🏆 推薦同細產業高協整配對排行榜 (Pairs Universe)")
        st.markdown("透過 MOPS/CMoney 實質產品營收細分驗證，並經嚴格 **Engle-Granger 協整檢驗 (p < 0.05)** 與走勢相關度排序之優質配對池：")
        
        if sub_pairs:
            top_coint_pairs = [p for p in sub_pairs if p.get('IsCointegrated') or p.get('CointPValue', 1.0) < 0.05][:20]
            if top_coint_pairs:
                df_top_coint = pd.DataFrame([
                    {
                        '細產業族群': p['SubIndustry'],
                        '配對組合': p['Pair'],
                        '標的 A': p['Name1'],
                        '標的 B': p['Name2'],
                        'A 主力產品': p['Prod1'],
                        'B 主力產品': p['Prod2'],
                        '價格走勢相關度': f"{p['PriceCorr']:+.3f}",
                        '日報酬相關係數': f"{p['ReturnCorr']:+.3f}",
                        '協整 p-value': f"{p['CointPValue']:.4f}",
                        '連動評級': p['Rating']
                    }
                    for p in top_coint_pairs
                ])
                st.dataframe(df_top_coint, use_container_width=True, height=330)
            else:
                st.info("尚無預先計算之協整配對資料。")

        st.markdown("---")

        # 2. 🏢 股期細產業地圖與實質產品營收總表
        st.markdown("### 🏢 股期細產業地圖與實質產品營收總表")
        st.markdown(f"共收錄 **{len(tickers)}** 檔標的之最新產品營收結構與業務明細（資料源：MOPS / CMoney 月產銷組合）：")

        # 細產業篩選器
        filter_sub_opts = ["🌐 全部細產業 (顯示全部)"] + all_available_subs
        chosen_filter_sub = st.selectbox("🔎 依細產業過濾標的：", filter_sub_opts, index=0)

        ticker_display = []
        for t in tickers:
            prof = stock_profiles.get(t) or get_stock_profile(t, industry_df)
            sub = prof['sub_industry']
            if chosen_filter_sub != "🌐 全部細產業 (顯示全部)" and sub != chosen_filter_sub:
                continue
            ticker_display.append({
                '代號': t,
                '簡稱': prof['name'],
                '市場': prof['market'],
                '細產業類別': sub,
                '大產業大類': prof['category'],
                '主力營收產品': prof['primary_product'],
                '主力比重 (%)': f"{prof['primary_ratio']:.1f}%" if prof['primary_ratio'] > 0 else "-",
                '產品營收明細 (Top 1~5)': prof['product_breakdown'],
                '公司主要業務說明': prof['business']
            })

        st.dataframe(pd.DataFrame(ticker_display), use_container_width=True, height=450)


if __name__ == '__main__':
    main()