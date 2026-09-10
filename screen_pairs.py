"""
大規模配對篩選 - 修正版
只禁止曾用過的「組合」，個別標的可以重複出現在不同配對中
從所有股價資料的 ticker 中，找 corr > 0.8 + coint p < 0.05 的配對
然後跑快速回測篩選 Sharpe > 1, Return > 100%, MDD < 40%
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
import os, calendar, datetime, warnings
warnings.filterwarnings('ignore')

SHARES_PER_CONTRACT = 2000
MARGIN_RATE = 0.135
TRADING_FEE_RATE = 0.00002
COMMISSION_PER_CONTRACT = 30
RISK_FREE_RATE = 0.015
CAPITAL = 1_000_000
Z_ENTRY = 2.0; Z_EXIT = 0.0; Z_WINDOW = 60

# 禁止的是「組合」不是個別標的
BANNED_PAIRS = {
    ('2408.TW','2344.TW'), ('3017.TW','6257.TW'), ('3260.TWO','8299.TWO'),
    ('2059.TW','2404.TW'), ('1718.TW','1909.TW'), ('2449.TW','3653.TW'),
    ('2421.TW','3231.TW'), ('2376.TW','2356.TW'), ('2886.TW','4904.TW'),
    ('3044.TW','6139.TW'), ('3264.TWO','2368.TW'), ('6414.TW','6188.TWO'),
    ('2392.TW','6176.TW'), ('6488.TWO','3152.TWO'), ('8039.TW','2327.TW'),
    ('3711.TW','6669.TW'), ('4958.TW','6223.TWO'), ('1802.TW','1717.TW'),
    ('2881.TW','3227.TWO'), ('3211.TWO','2328.TW'), ('1609.TW','4938.TW'),
    ('2515.TW','6239.TW'), ('2474.TW','1216.TW'), ('2439.TW','9939.TW'),
    ('2603.TW','2454.TW'), ('3005.TW','6121.TWO'), ('2485.TW','2337.TW'),
    ('3711.TW','6257.TW'),
}

def is_banned(s1, s2):
    return (s1,s2) in BANNED_PAIRS or (s2,s1) in BANNED_PAIRS

# Exclude ETFs, bonds, steel, petrochemical tickers
EXCLUDE_TICKERS = set()
EXCLUDE_PREFIXES = ['0050','0056','006','00','IX']  # ETFs
STEEL_PETRO = {'2002.TW','2006.TW','2014.TW','1326.TW','1314.TW','6505.TW',
               '2010.TW','2031.TW','2027.TW','9958.TW'}  # steel/petrochemical

def find_min_contracts(hedge_ratio):
    """找最少配對口數（加入口數懲罰，避免盲目放大）"""
    abs_hedge = abs(hedge_ratio)
    best_score = float('inf')
    best_s1, best_s2 = 1, 1
    for s2 in range(1, 21):
        for s1 in range(1, 21):
            ratio = s1 / s2
            error = abs(ratio - abs_hedge)
            score = error + (s1 + s2) * 0.015
            if score < best_score:
                best_score = score
                best_s1, best_s2 = s1, s2
    return best_s1, best_s2

def get_third_weds(sy, ey):
    dates = set()
    for y in range(sy, ey+1):
        for m in range(1,13):
            cal = calendar.monthcalendar(y, m)
            ws = [w[2] for w in cal if w[2]!=0]
            if len(ws)>=3: dates.add(datetime.date(y, m, ws[2]))
    return dates

def quick_backtest(S1, S2, sym1, sym2):
    common = S1.index.intersection(S2.index)
    S1, S2 = S1[common].copy(), S2[common].copy()
    if len(common) < 200: return None
    limit = (S1.pct_change().abs()>=0.099) | (S2.pct_change().abs()>=0.099)
    X = sm.add_constant(S1); model = sm.OLS(S2, X).fit(); hr = model.params[sym1]
    spread = S2 - hr * S1
    z = ((spread - spread.rolling(Z_WINDOW).mean()) / spread.rolling(Z_WINDOW).std()).dropna()
    if len(z) < 100: return None
    c1, c2 = find_min_contracts(S1[z.index[0]], S2[z.index[0]])
    settle = get_third_weds(z.index[0].year, z.index[-1].year)
    pos=0; ep1=ep2=0.0; cash=CAPITAL; pending=None; equities=[]; tc=0; wins=0; edate=None
    for date in z.index:
        zv=z[date]; p1=S1[date]; p2=S2[date]; il=limit.get(date,False); iset=date.date() in settle
        mreq=p1*SHARES_PER_CONTRACT*MARGIN_RATE*c1+p2*SHARES_PER_CONTRACT*MARGIN_RATE*c2; upnl=0.0
        if pos!=0:
            if pos==1: upnl=(p2-ep2)*SHARES_PER_CONTRACT*c2+(ep1-p1)*SHARES_PER_CONTRACT*c1
            else: upnl=(ep2-p2)*SHARES_PER_CONTRACT*c2+(p1-ep1)*SHARES_PER_CONTRACT*c1
        cost=(p1*SHARES_PER_CONTRACT*c1+p2*SHARES_PER_CONTRACT*c2)*TRADING_FEE_RATE+COMMISSION_PER_CONTRACT*(c1+c2)
        if pending is not None and pos==0:
            if not il and cash>=mreq: pos=pending; ep1=p1; ep2=p2; edate=date; cash-=cost
            pending=None
        elif iset and pos!=0:
            npnl=upnl-cost; cash+=npnl; tc+=1
            if npnl>0: wins+=1
            if (pos==1 and zv<Z_EXIT) or (pos==-1 and zv>Z_EXIT): pending=pos
            pos=0; upnl=0; ep1=ep2=0; edate=None
        elif zv>Z_ENTRY and pos==0 and pending is None:
            if not il and cash>=mreq: pos=-1; ep1=p1; ep2=p2; edate=date; cash-=cost
        elif zv<-Z_ENTRY and pos==0 and pending is None:
            if not il and cash>=mreq: pos=1; ep1=p1; ep2=p2; edate=date; cash-=cost
        elif pos!=0 and ((pos==1 and zv>=Z_EXIT) or (pos==-1 and zv<=Z_EXIT)):
            if not il:
                if pos==1: upnl=(p2-ep2)*SHARES_PER_CONTRACT*c2+(ep1-p1)*SHARES_PER_CONTRACT*c1
                else: upnl=(ep2-p2)*SHARES_PER_CONTRACT*c2+(p1-ep1)*SHARES_PER_CONTRACT*c1
                npnl=upnl-cost; cash+=npnl; tc+=1
                if npnl>0: wins+=1
                pos=0; upnl=0; ep1=ep2=0; edate=None
        if pos!=0:
            if pos==1: upnl=(p2-ep2)*SHARES_PER_CONTRACT*c2+(ep1-p1)*SHARES_PER_CONTRACT*c1
            else: upnl=(ep2-p2)*SHARES_PER_CONTRACT*c2+(p1-ep1)*SHARES_PER_CONTRACT*c1
            equities.append(cash+upnl)
        else: equities.append(cash)
    eq_s = pd.Series(equities, index=z.index)
    dr = eq_s.pct_change().dropna()
    if len(dr)<2 or dr.std()==0: return None
    sharpe = (dr.mean()-RISK_FREE_RATE/252)/dr.std()*np.sqrt(252)
    ret_pct = (eq_s.iloc[-1]/CAPITAL-1)*100
    mdd_pct = ((eq_s.cummax()-eq_s)/eq_s.cummax()).max()*100
    wr = wins/tc*100 if tc>0 else 0
    return (sharpe, ret_pct, mdd_pct, tc, wr, hr, c1, c2)

# ========== Main ==========
print("=" * 70)
print("  Large-scale pair screening (banned = combinations only)")
print("=" * 70)

prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True)
prices = prices.dropna(axis=1, thresh=len(prices)*0.8).ffill().bfill()

# Filter out ETFs, bonds, steel, petrochemical
tickers = [t for t in prices.columns 
           if t not in STEEL_PETRO 
           and not any(t.startswith(p) for p in EXCLUDE_PREFIXES)]
print(f"Valid tickers: {len(tickers)}")

# Step 1: Correlation filter
print("Step 1: Computing correlations...")
corr_matrix = prices[tickers].corr()
candidate_pairs = []
for i, t1 in enumerate(tickers):
    for j, t2 in enumerate(tickers):
        if j <= i: continue
        if is_banned(t1, t2): continue
        c = corr_matrix.loc[t1, t2]
        if abs(c) > 0.8:
            candidate_pairs.append((t1, t2, c))

print(f"Pairs with |corr| > 0.8: {len(candidate_pairs)}")

# Step 2: Cointegration test
print("Step 2: Cointegration testing...")
coint_pairs = []
for t1, t2, corr_val in candidate_pairs:
    try:
        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 200: continue
        _, pval, _ = coint(s1[common], s2[common])
        if pval < 0.05:
            coint_pairs.append((t1, t2, corr_val, pval))
    except:
        pass

print(f"Cointegrated pairs (p < 0.05): {len(coint_pairs)}")

# Step 3: Quick backtest
print("Step 3: Running backtests...")
results = []
for idx, (t1, t2, corr_val, pval) in enumerate(coint_pairs):
    if idx % 50 == 0:
        print(f"  Progress: {idx}/{len(coint_pairs)}")
    try:
        r = quick_backtest(prices[t1].dropna(), prices[t2].dropna(), t1, t2)
        if r:
            sharpe, ret, mdd, trades, wr, hr, c1, c2 = r
            results.append({
                'sym1':t1,'sym2':t2,'corr':corr_val,'pvalue':pval,
                'sharpe':sharpe,'return_pct':ret,'mdd_pct':mdd,
                'trades':trades,'win_rate':wr,'hr':hr,'c1':c1,'c2':c2
            })
    except:
        pass

df = pd.DataFrame(results)
print(f"\nTotal backtested: {len(df)}")

# Filter: Sharpe > 1, Return > 100%, MDD < 40%
perfect = df[(df['sharpe']>1.0) & (df['return_pct']>100) & (df['mdd_pct']<40)].sort_values('sharpe', ascending=False)
print(f"PERFECT (Sharpe>1, Ret>100%, MDD<40%): {len(perfect)}")
print()
print(f"{'#':>3} {'Pair':>28} {'Sharpe':>7} {'Return%':>9} {'MDD%':>7} {'Trades':>6} {'WinR%':>6} {'p-val':>10} {'Corr':>6}")
print("-" * 90)
for i, (_, r) in enumerate(perfect.head(40).iterrows(), 1):
    print(f"{i:>3} {r['sym1']+'/'+r['sym2']:>28} {r['sharpe']:>7.2f} {r['return_pct']:>+8.1f}% {r['mdd_pct']:>6.1f}% {r['trades']:>6.0f} {r['win_rate']:>5.1f}% {r['pvalue']:>10.6f} {r['corr']:>5.2f}")

# Also show near-perfect
print(f"\n--- Near-perfect (Sharpe>0.9, Ret>80%, MDD<40%) ---")
near = df[(df['sharpe']>0.9) & (df['return_pct']>80) & (df['mdd_pct']<40)].sort_values('sharpe', ascending=False)
for i, (_, r) in enumerate(near.head(30).iterrows(), 1):
    tag = " ***" if r['sharpe']>1 and r['return_pct']>100 else ""
    print(f"{i:>3} {r['sym1']+'/'+r['sym2']:>28} {r['sharpe']:>7.2f} {r['return_pct']:>+8.1f}% {r['mdd_pct']:>6.1f}% {r['trades']:>6.0f} {r['win_rate']:>5.1f}%{tag}")

# Save
df.sort_values('sharpe', ascending=False).to_csv('results/full_screening_results.csv', index=False, encoding='utf-8-sig')
print(f"\nFull results ({len(df)} pairs) saved to results/full_screening_results.csv")
