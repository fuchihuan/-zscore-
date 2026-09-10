import pandas as pd
import yfinance as yf
import os
import sys
import time

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 簡單的翻譯字典，用來將常見英文單字轉換成中文，提升閱讀體驗
KEYWORD_MAP = {
    "manufactures": "製造", "sells": "銷售", "provides": "提供", "develops": "開發",
    "semiconductor": "半導體", "integrated circuits": "積體電路", "wafers": "晶圓",
    "electronic": "電子", "components": "零組件", "equipment": "設備",
    "telecommunications": "電信", "services": "服務", "products": "產品",
    "materials": "材料", "chemicals": "化學品", "technology": "科技",
    "network": "網路", "communications": "通訊", "computer": "電腦",
    "software": "軟體", "hardware": "硬體", "display": "顯示器",
    "optical": "光學", "automotive": "車用", "medical": "醫療",
    "financial": "金融", "insurance": "保險", "banking": "銀行",
    "real estate": "房地產", "construction": "營造", "shipping": "航運",
    "logistics": "物流", "retail": "零售", "wholesale": "批發",
    "energy": "能源", "power": "電力", "solar": "太陽能",
    "machinery": "機械", "industrial": "工業", "consumer": "消費性",
    "Taiwan": "台灣", "Company": "公司", "Limited": "有限公司",
    "subsidiaries": "子公司", "together with its": "及其",
    "engages in": "從事", "research and development": "研發",
    "design": "設計", "testing": "測試", "packaging": "封裝",
    "memory": "記憶體", "servers": "伺服器", "smartphones": "智慧型手機",
    "displays": "顯示器", "panels": "面板", "lenses": "鏡頭"
}

def translate_summary(text):
    if not text or not isinstance(text, str):
        return "-"
    # 取前 300 個字元，避免太長
    text = text[:300] + ("..." if len(text) > 300 else "")
    # 簡單的關鍵字替換
    for eng, zhtw in KEYWORD_MAP.items():
        text = text.replace(eng, zhtw)
        text = text.replace(eng.capitalize(), zhtw)
        text = text.replace(eng.title(), zhtw)
    return text

def main():
    print("=" * 60)
    print("  使用 yfinance 補充主要營收產品與細產業資料 (方案 B)")
    print("=" * 60)
    
    csv_file = os.path.join(SCRIPT_DIR, 'industry_data.csv')
    if not os.path.exists(csv_file):
        print("[ERROR] 找不到 industry_data.csv，請先執行 fetch_industry_data.py。")
        sys.exit(1)
        
    df = pd.read_csv(csv_file)
    
    print(f"開始為 {len(df)} 檔標的擷取詳細營收業務 (從 Yahoo Finance 抓取英文年報摘要)...\n")
    
    new_businesses = []
    
    for i, row in df.iterrows():
        tk = row['Ticker']
        print(f"[{i+1}/{len(df)}] 查詢 {tk} ...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(tk)
            info = ticker.info
            summary = info.get('longBusinessSummary', '')
            
            if summary:
                # 簡單英轉中
                translated = translate_summary(summary)
                new_businesses.append(translated)
                print("OK")
            else:
                new_businesses.append("-")
                print("無資料")
                
        except Exception as e:
            print(f"錯誤")
            new_businesses.append("-")
            
    df['Business'] = new_businesses
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 詳細業務資料已更新至 {csv_file}")
    
    print("\n重新產生產業分類報表 (industry_grouping.md)...")
    out_md = os.path.join(SCRIPT_DIR, 'industry_grouping.md')
    
    # 建立分群字典
    industry_groups = {}
    for i, row in df.iterrows():
        ind = row['Industry']
        tk = row['Ticker']
        bus = row['Business']
        if ind not in industry_groups:
            industry_groups[ind] = []
        industry_groups[ind].append({
            'ticker': tk,
            'business': bus
        })
        
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# 📊 股票期貨標的細產業與產品營收總表 (加強版)\n\n")
        f.write("此表格的大產業分類來自政府開放資料 (MOPS)，主要產品與營收來源由 `yfinance` 自動擷取最新年報摘要並進行關鍵字翻譯。\n\n")
        
        for ind in sorted(industry_groups.keys()):
            f.write(f"## 🔹 {ind} (共 {len(industry_groups[ind])} 檔)\n")
            f.write("| 股票代號 | 主要產品與營收來源 (年報摘要) |\n")
            f.write("|---|---|\n")
            for item in industry_groups[ind]:
                clean_bus = str(item['business']).replace('\n', '<br>').replace('\r', '')
                f.write(f"| `{item['ticker']}` | {clean_bus} |\n")
            f.write("\n")
            
    print(f"-> [OK] 細產業報表已更新至 {out_md}")
    print("\n全部完成！網頁已可讀取最新分類資料。")

if __name__ == '__main__':
    main()
