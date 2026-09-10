import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
S1 = df['1513.TW']
S2 = df['6414.TW']
X = sm.add_constant(S1)
m = sm.OLS(S2, X).fit()
hr = m.params['1513.TW']
ic = m.params['const']
print(f"Hedge Ratio (beta): {hr:.4f}")
print(f"Intercept: {ic:.4f}")
print(f"R-squared: {m.rsquared:.4f}")
print(f"Latest 1513.TW price: {S1.iloc[-1]:.2f}")
print(f"Latest 6414.TW price: {S2.iloc[-1]:.2f}")
