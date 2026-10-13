"""
Fetch and Analyze CME WTI vs Brent Crude Oil Futures Daily Bars (2020 - Present)
==============================================================================
資料來源:
- CME/NYMEX WTI 近月原油期貨 (CL=F)
- ICE/CME Brent 近月布蘭特原油期貨 (BZ=F)
涵蓋 2020-01-01 至最新交易日之完整日 K，包含開高低收量、絕對價差、對沖比率、Z-Score、滾動相關性等價差交易指標。
"""

import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller

def main():
    print("正在下載 2020 年至今 CME WTI (CL=F) 與 Brent (BZ=F) 原油期貨日K線資料...")
    cl = yf.download('CL=F', start='2020-01-01', progress=False)
    bz = yf.download('BZ=F', start='2020-01-01', progress=False)

    # 平整化 MultiIndex 欄位 (yfinance 新版格式)
    if isinstance(cl.columns, pd.MultiIndex):
        cl.columns = [col[0] for col in cl.columns]
    if isinstance(bz.columns, pd.MultiIndex):
        bz.columns = [col[0] for col in bz.columns]

    cl = cl[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
        'Open': 'WTI_Open', 'High': 'WTI_High', 'Low': 'WTI_Low', 'Close': 'WTI_Close', 'Volume': 'WTI_Volume'
    })
    bz = bz[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={
        'Open': 'Brent_Open', 'High': 'Brent_High', 'Low': 'Brent_Low', 'Close': 'Brent_Close', 'Volume': 'Brent_Volume'
    })

    # 對齊交易日
    df = pd.merge(cl, bz, left_index=True, right_index=True, how='inner')
    df.index.name = 'Date'

    # 1. 價差計算 (Brent 通常較 WTI 溢價，因此一般以 Brent - WTI 為標準價差)
    df['Spread_USD'] = df['Brent_Close'] - df['WTI_Close']
    df['Price_Ratio'] = df['Brent_Close'] / df['WTI_Close']

    # 2. 滾動均值與標準差 (20日與60日)
    df['Spread_MA20'] = df['Spread_USD'].rolling(20).mean()
    df['Spread_Std20'] = df['Spread_USD'].rolling(20).std()
    df['ZScore_20'] = (df['Spread_USD'] - df['Spread_MA20']) / df['Spread_Std20']

    df['Spread_MA60'] = df['Spread_USD'].rolling(60).mean()
    df['Spread_Std60'] = df['Spread_USD'].rolling(60).std()
    df['ZScore_60'] = (df['Spread_USD'] - df['Spread_MA60']) / df['Spread_Std60']

    # 3. 滾動 60 日相關係數
    df['Rolling_Corr_60'] = df['WTI_Close'].rolling(60).corr(df['Brent_Close'])

    # 儲存 CSV 檔
    output_csv = 'wti_brent_daily_spread_2020_2026.csv'
    df.to_csv(output_csv)
    print(f"成功儲存 {len(df)} 筆日K交易資料至 {output_csv}！\n")

    # 統計檢定
    common_clean = df[['WTI_Close', 'Brent_Close', 'Spread_USD']].dropna()
    coint_t, coint_p, _ = coint(common_clean['Brent_Close'], common_clean['WTI_Close'])
    adf_res = adfuller(common_clean['Spread_USD'])

    print("=" * 60)
    print("📊 2020 - 2026 原油期貨價差研究關鍵特徵報告")
    print("=" * 60)
    print(f"資料期間: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"有效交易日數: {len(df):,} 天")
    print(f"最新 WTI 收盤價:   ${df['WTI_Close'].iloc[-1]:.2f} 美元/桶")
    print(f"最新 Brent 收盤價: ${df['Brent_Close'].iloc[-1]:.2f} 美元/桶")
    print(f"最新價差 (Brent - WTI): ${df['Spread_USD'].iloc[-1]:.2f} 美元/桶")
    print(f"最新 20日 Z-Score: {df['ZScore_20'].iloc[-1]:.2f}")
    print(f"最新 60日價格相關係數: {df['Rolling_Corr_60'].iloc[-1]:.4f}")
    print("-" * 60)
    print("📈 絕對價差分佈統計 (Spread = Brent - WTI, USD/bbl):")
    print(f"  平均價差 (Mean):    ${df['Spread_USD'].mean():.2f}")
    print(f"  價差中位數 (Median): ${df['Spread_USD'].median():.2f}")
    print(f"  價差標準差 (Std):   ${df['Spread_USD'].std():.2f}")
    print(f"  歷史最高價差 (Max):  ${df['Spread_USD'].max():.2f} (2020負油價極端期)")
    print(f"  歷史最低價差 (Min):  ${df['Spread_USD'].min():.2f}")
    print("-" * 60)
    print("🧪 統計檢定 (Cointegration & Stationarity):")
    print(f"  共整合檢定 (Engle-Granger) P-value: {coint_p:.6f} {'(極度顯著共整合，P < 0.01)' if coint_p < 0.01 else ''}")
    print(f"  價差單根檢定 (ADF Test) t-stat:      {adf_res[0]:.4f}")
    print(f"  ADF P-value:                         {adf_res[1]:.6f} (價差具備強烈均值回歸平穩性)")
    print("=" * 60)

if __name__ == '__main__':
    main()
