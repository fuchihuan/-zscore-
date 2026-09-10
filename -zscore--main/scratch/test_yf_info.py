import yfinance as yf
info = yf.Ticker('2330.TW').info
print(f"Sector: {info.get('sector')}")
print(f"Industry: {info.get('industry')}")
print(f"LongBusinessSummary: {info.get('longBusinessSummary', '')[:200]}")
