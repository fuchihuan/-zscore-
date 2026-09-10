import requests
from bs4 import BeautifulSoup

def fetch_profile(ticker):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    short_ticker = ticker.split('.')[0]
    url = f"https://tw.stock.yahoo.com/quote/{short_ticker}/profile"
    
    industry = "未知"
    business = "未知"
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return industry, business
            
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 尋找包含文字的 div
        for div in soup.find_all('div'):
            text = div.get_text(strip=True)
            if text == "產業類別":
                # 通常它的下一個兄弟節點或父節點的兄弟節點包含值
                parent = div.parent
                if parent:
                    # 抓取父節點下的所有文字，看能不能分離出值
                    all_text = parent.get_text(separator='|')
                    parts = all_text.split('|')
                    if len(parts) > 1:
                        industry = parts[-1].strip()
            elif text == "主要業務":
                parent = div.parent
                if parent:
                    all_text = parent.get_text(separator='|')
                    parts = all_text.split('|')
                    if len(parts) > 1:
                        business = parts[-1].strip()
                        
        return industry, business
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return industry, business

print(fetch_profile("2330"))
print(fetch_profile("3105"))
print(fetch_profile("2603")) # 長榮
