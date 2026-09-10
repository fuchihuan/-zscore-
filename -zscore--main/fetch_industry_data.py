import pandas as pd
import requests
import io
import os
import sys

# 修正 Windows console 編碼問題
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def fetch_mops_data(url):
    """從公開資訊觀測站下載並解析 CSV"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    # 忽略 SSL 錯誤，因為政府網站憑證有時會有問題
    res = requests.get(url, headers=headers, verify=False, timeout=15)
    res.raise_for_status()
    # MOPS 檔案通常是 UTF-8
    text = res.content.decode('utf-8-sig', errors='replace')
    return pd.read_csv(io.StringIO(text))

# 台灣股市產業代碼對照表
IND_MAP = {
    '01': '水泥工業', '02': '食品工業', '03': '塑膠工業', '04': '紡織纖維', '05': '電機機械',
    '06': '電器電纜', '07': '玻璃陶瓷', '08': '造紙工業', '09': '鋼鐵工業', '10': '橡膠工業',
    '11': '汽車工業', '12': '建材營造', '14': '航運業', '15': '觀光餐旅', '16': '金融保險',
    '17': '貿易百貨', '18': '綜合', '20': '其他', '21': '化學工業', '22': '生技醫療業',
    '23': '油電燃氣業', '24': '半導體業', '25': '電腦及週邊設備業', '26': '光電業',
    '27': '通信網路業', '28': '電子零組件業', '29': '電子通路業', '30': '資訊服務業',
    '31': '其他電子業', '32': '文化創意業', '33': '農業科技業', '34': '電子商務',
    '35': '綠能環保', '36': '數位雲端', '37': '運動休閒', '38': '居家生活'
}

def main():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    print("=" * 60)
    print("  爬取台指期標的：產業類別與主要業務 (MOPS OpenData)")
    print("=" * 60)
    
    stock_prices_file = os.path.join(SCRIPT_DIR, 'stock_prices.csv')
    if not os.path.exists(stock_prices_file):
        print("[ERROR] 找不到 stock_prices.csv。")
        sys.exit(1)
        
    df_prices = pd.read_csv(stock_prices_file, index_col=0, nrows=1)
    tickers = list(df_prices.columns)
    print(f"共有 {len(tickers)} 檔期貨標的需要查詢...")
    
    print("\n[1/3] 正在下載上市與上櫃公司基本資料...")
    try:
        url_twse = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"
        url_tpex = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"
        
        df_twse = fetch_mops_data(url_twse)
        df_tpex = fetch_mops_data(url_tpex)
        
        # 合併上市與上櫃資料
        df_all = pd.concat([df_twse, df_tpex], ignore_index=True)
        # 轉代碼為字串以確保一致性並去除空白
        df_all['公司代號'] = df_all['公司代號'].astype(str).str.strip()
        print(f"-> 成功下載共 {len(df_all)} 筆上市櫃資料。")
    except Exception as e:
        print(f"[ERROR] 下載或解析政府開放資料失敗: {e}")
        sys.exit(1)
        
    print("\n[2/3] 進行比對並產出 industry_data.csv...")
    results = []
    
    # 建立一個 mapping 字典加速查詢
    mops_map = df_all.set_index('公司代號').to_dict('index')
    
    # 用於 markdown 報告的分群字典
    industry_groups = {}
    
    for tk in tickers:
        # 去除後綴如 '.TW' 或 '.TWO'
        short_ticker = str(tk).split('.')[0].strip()
        info = mops_map.get(short_ticker)
        if info:
            raw_ind = str(info.get('產業別', '未知')).strip()
            # 若為單個數字，補零
            if raw_ind.isdigit() and len(raw_ind) == 1:
                raw_ind = '0' + raw_ind
            ind = IND_MAP.get(raw_ind, f"未知產業({raw_ind})")
            bus = '-' # MOPS CSV 不含主要業務詳細敘述
            name = str(info.get('公司名稱', '')).strip()
            name_abbr = str(info.get('公司簡稱', '')).strip()
        else:
            ind = "未知"
            bus = "未知"
            name = tk
            name_abbr = tk
            
        results.append({
            'Ticker': tk,
            'Industry': ind,
            'Business': bus
        })
        
        # 加入分群字典
        if ind not in industry_groups:
            industry_groups[ind] = []
        industry_groups[ind].append({
            'ticker': tk,
            'name': name_abbr or name,
            'business': bus
        })

    # 輸出 CSV
    out_csv = os.path.join(SCRIPT_DIR, 'industry_data.csv')
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"-> [OK] 產業資料已儲存至 {out_csv}")
    
    print("\n[3/3] 產生產業分類報表 (industry_grouping.md)...")
    out_md = os.path.join(SCRIPT_DIR, 'industry_grouping.md')
    
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# 📊 股票期貨標的產業分類總表\n\n")
        f.write("此表格由政府開放資料 (公開資訊觀測站) 自動生成，方便您挑選同產業的公司進行配對交易回測。\n\n")
        
        # 依產業名稱排序
        for ind in sorted(industry_groups.keys()):
            f.write(f"## 🔹 {ind} (共 {len(industry_groups[ind])} 檔)\n")
            f.write("| 股票代號 | 公司簡稱 | 主要業務 |\n")
            f.write("|---|---|---|\n")
            for item in industry_groups[ind]:
                # 清除業務描述中的換行符號以符合 Markdown 表格格式
                clean_bus = str(item['business']).replace('\n', '').replace('\r', '')
                f.write(f"| `{item['ticker']}` | **{item['name']}** | {clean_bus} |\n")
            f.write("\n")
            
    print(f"-> [OK] 產業分類報表已儲存至 {out_md}")
    print("\n全部完成！網頁已可讀取最新分類資料。")

if __name__ == '__main__':
    main()
