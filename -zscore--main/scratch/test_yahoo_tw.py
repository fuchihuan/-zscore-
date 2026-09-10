import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://tw.stock.yahoo.com/quote/2330.TW/profile"
res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, 'html.parser')

# Find the profile table
# Yahoo TW profile has "產業類別" and "主要業務"
print("Yahoo TW:")
for div in soup.find_all('div', class_='Py(8px)'):
    print(div.text)

print("\n---")
