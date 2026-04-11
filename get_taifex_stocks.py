import pandas as pd
import requests
from io import StringIO

url = 'https://www.taifex.com.tw/cht/2/stockLists'

try:
    response = requests.get(url)
    response.encoding = 'utf-8' # Try utf-8 first
    html_content = response.text
    
    tables = pd.read_html(StringIO(html_content))
    table = tables[1] # The second table should contain the list
    
    print("Columns:")
    print(list(table.columns))
    
    print("First 5 rows:")
    print(table.head())
    
    # Let's save it to csv to inspect
    table.to_csv('taifex_stocks.csv', index=False, encoding='utf-8-sig')
    print("Saved to taifex_stocks.csv")
    
except Exception as e:
    print(f"Error fetching data: {e}")
