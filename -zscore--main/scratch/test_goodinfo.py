import requests
from bs4 import BeautifulSoup

def fetch_goodinfo(ticker):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://goodinfo.tw/tw/index.asp',
    }
    short_ticker = str(ticker).split('.')[0]
    url = f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={short_ticker}"
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')
    
    industry = "未知"
    business = "未知"
    
    try:
        # Find 產業別
        ind_td = soup.find('td', string='產業別')
        if ind_td:
            industry = ind_td.find_next_sibling('td').text.strip()
            
        # Find 主要業務
        bus_td = soup.find('td', string='主要業務')
        if bus_td:
            business = bus_td.find_next_sibling('td').text.strip()
    except:
        pass
        
    return industry, business

ind, bus = fetch_goodinfo('2330')
with open('result.txt', 'w', encoding='utf-8') as f:
    f.write(f"Industry: {ind}\nBusiness: {bus}\n")
