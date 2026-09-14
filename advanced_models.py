import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t
from arch import arch_model

def get_zscore_signals(S1, S2, z_entry, z_exit, window):
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    zscore = (spread - spread_mean) / spread_std
    
    # 返回 DataFrame
    df = pd.DataFrame({
        'Indicator': zscore,
        'Upper_Bound': z_entry,
        'Lower_Bound': -z_entry,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,  # 修正平倉線
        'Spread': spread,
        'Spread_Mean': spread_mean,
        'Hedge_Ratio': hedge_ratio
    })
    return df.dropna()

def fit_ou_process(spread):
    """
    Fits an Ornstein-Uhlenbeck process dX_t = theta * (mu - X_t)dt + sigma * dW_t
    Using Maximum Likelihood Estimation via OLS
    """
    if len(spread) < 2:
        return np.nan, np.nan, np.nan
    X = spread.values[:-1]
    Y = spread.values[1:]
    
    X_mat = sm.add_constant(X)
    try:
        ols_model = sm.OLS(Y, X_mat).fit()
        a = ols_model.params[1]
        b = ols_model.params[0]
        sigma_e = np.sqrt(ols_model.mse_resid)
        
        if a <= 0 or a >= 1:
            return np.nan, np.nan, np.nan
            
        theta = -np.log(a)
        mu = b / (1 - a)
        sigma = sigma_e * np.sqrt(-2 * np.log(a) / (1 - a**2))
        return theta, mu, sigma
    except:
        return np.nan, np.nan, np.nan

def get_ou_signals(S1, S2, window, risk_free_rate=0.015, trading_fee=0.0004):
    """
    Ornstein-Uhlenbeck Process based boundaries
    Approximated optimal boundaries considering transaction cost and theta.
    """
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    
    upper_bounds = pd.Series(index=spread.index, dtype=float)
    lower_bounds = pd.Series(index=spread.index, dtype=float)
    exit_uppers = pd.Series(index=spread.index, dtype=float)
    exit_lowers = pd.Series(index=spread.index, dtype=float)
    spread_means = pd.Series(index=spread.index, dtype=float)
    
    for i in range(window, len(spread)):
        current_spread = spread.iloc[i-window:i]
        theta, mu, sigma = fit_ou_process(current_spread)
        
        if np.isnan(theta):
            # Fallback to simple mean/std
            mu = current_spread.mean()
            sigma = current_spread.std()
            b = 0.5 * sigma
            d = 2.0 * sigma
        else:
            # Analytical approximation for optimal threshold d (entry) and b (exit)
            theta_eff = max(theta, 0.001)
            b = (0.5 * sigma) / np.sqrt(theta_eff) # exit (spread mean + b)
            d = b + (sigma / np.sqrt(theta_eff)) * np.sqrt(max(0.1, 1 - risk_free_rate)) # entry
        
        spread_means.iloc[i] = mu
        upper_bounds.iloc[i] = mu + d
        lower_bounds.iloc[i] = mu - d
        exit_uppers.iloc[i] = mu + b
        exit_lowers.iloc[i] = mu - b
        
    df = pd.DataFrame({
        'Indicator': spread,
        'Upper_Bound': upper_bounds,
        'Lower_Bound': lower_bounds,
        'Exit_Upper': exit_uppers,
        'Exit_Lower': exit_lowers,
        'Spread': spread,
        'Spread_Mean': spread_means,
        'Hedge_Ratio': hedge_ratio
    })
    return df.dropna()

def get_garch_signals(S1, S2, window, z_entry=2.0, z_exit=0.0):
    """
    Cointegration + GARCH(1,1) dynamic volatility
    """
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=window).mean()
    
    garch_vol = pd.Series(index=spread.index, dtype=float)
    
    for i in range(window, len(spread), 5):  # Step by 5 for speed
        resids = spread.iloc[i-window:i] - spread_mean.iloc[i-window:i]
        resids = resids.dropna()
        if len(resids) > 20:
            try:
                scale = 100.0 / resids.std()
                am = arch_model(resids * scale, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
                res = am.fit(disp='off', show_warning=False)
                forecasts = res.forecast(horizon=5)
                pred_var = forecasts.variance.iloc[-1].values
                for j in range(min(5, len(spread) - i)):
                    garch_vol.iloc[i+j] = np.sqrt(pred_var[j]) / scale
            except:
                for j in range(min(5, len(spread) - i)):
                    garch_vol.iloc[i+j] = resids.std()
        else:
            for j in range(min(5, len(spread) - i)):
                garch_vol.iloc[i+j] = np.nan
             
    garch_vol = garch_vol.ffill()
    zscore_garch = (spread - spread_mean) / garch_vol
    
    df = pd.DataFrame({
        'Indicator': zscore_garch,
        'Upper_Bound': z_entry,
        'Lower_Bound': -z_entry,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,
        'Spread': spread,
        'Spread_Mean': spread_mean,
        'Hedge_Ratio': hedge_ratio,
        'Dynamic_Vol': garch_vol
    })
    return df.dropna()

def get_kalman_filter_signals(S1, S2, window=60, z_entry=2.0, z_exit=0.0):
    """
    Kalman Filter (State Space Model)
    Dynamically tracks hedge ratio and spread mean
    """
    n = len(S1)
    
    state_mean = np.zeros((n, 2))
    state_cov = np.zeros((n, 2, 2))
    
    state_mean[0] = [0, S2.iloc[0]/S1.iloc[0] if S1.iloc[0]!=0 else 1]
    state_cov[0] = np.eye(2) * 1.0
    
    V_w = np.eye(2) * 1e-4  # State covariance
    V_v = 1e-3              # Observation covariance
    
    predictions = np.zeros(n)
    variances = np.zeros(n)
    
    for t in range(1, n):
        prior_mean = state_mean[t-1]
        prior_cov = state_cov[t-1] + V_w
        
        x_t = np.array([1, S1.iloc[t]])
        y_t = S2.iloc[t]
        
        y_pred = np.dot(x_t, prior_mean)
        predictions[t] = y_pred
        
        F = np.dot(np.dot(x_t, prior_cov), x_t.T) + V_v
        variances[t] = F
        
        K = np.dot(prior_cov, x_t.T) / F
        
        state_mean[t] = prior_mean + K * (y_t - y_pred)
        state_cov[t] = prior_cov - np.outer(K, x_t).dot(prior_cov)
        
    alpha_series = pd.Series(state_mean[:, 0], index=S1.index)
    beta_series = pd.Series(state_mean[:, 1], index=S1.index)
    
    spread = S2 - (alpha_series + beta_series * S1)
    
    # 修正卡爾曼濾波變異數過大問題：改用滾動標準差進行標準化
    spread_std = spread.rolling(window=window).std()
    indicator = spread / spread_std
    indicator.iloc[0:window] = np.nan 
    
    df = pd.DataFrame({
        'Indicator': indicator,
        'Upper_Bound': z_entry,
        'Lower_Bound': -z_entry,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,
        'Spread': spread,
        'Spread_Mean': 0, 
        'Hedge_Ratio': beta_series
    })
    return df.dropna()

def get_copula_signals(S1, S2, window, prob_threshold=0.95):
    """
    Copula based Cumulative Mispricing Index (CMPI)
    """
    # 改良版 Copula：平穩價差隨機過程 (Stationary Spread Processes)
    # 不使用日報酬率，改用去均值後的相對價格變化，具備均值回歸特性
    returns_S1 = S1 - S1.rolling(window).mean()
    returns_S2 = S2 - S2.rolling(window).mean()
    
    returns_S1 = returns_S1.dropna()
    returns_S2 = returns_S2.dropna()
    
    common_idx = returns_S1.index.intersection(returns_S2.index)
    returns_S1 = returns_S1.loc[common_idx]
    returns_S2 = returns_S2.loc[common_idx]
    
    indicator = pd.Series(index=common_idx, dtype=float)
    hedge_ratios = pd.Series(index=common_idx, dtype=float)
    
    for i in range(window, len(common_idx)):
        u1 = returns_S1.iloc[i-window:i]
        u2 = returns_S2.iloc[i-window:i]
        
        # 修正 Infinity Bug：將百分位數限制在 0.001 ~ 0.999 之間
        rank1 = np.clip(u1.rank(pct=True).iloc[-1], 0.001, 0.999)
        rank2 = np.clip(u2.rank(pct=True).iloc[-1], 0.001, 0.999)
        
        rho = u1.corr(u2)
        
        try:
            norm_u1 = norm.ppf(rank1)
            norm_u2 = norm.ppf(rank2)
            cond_prob_2_given_1 = norm.cdf((norm_u2 - rho * norm_u1) / np.sqrt(max(1 - rho**2, 0.0001)))
            indicator.iloc[i] = cond_prob_2_given_1
        except:
            indicator.iloc[i] = 0.5
            
        X = sm.add_constant(S1.loc[u1.index])
        model = sm.OLS(S2.loc[u2.index], X).fit()
        hedge_ratios.iloc[i] = model.params.iloc[1]

    indicator = indicator.reindex(S1.index).ffill()
    hedge_ratios = hedge_ratios.reindex(S1.index).ffill()
    
    df = pd.DataFrame({
        'Indicator': indicator,
        'Upper_Bound': prob_threshold,
        'Lower_Bound': 1.0 - prob_threshold,
        'Exit_Upper': 0.5,
        'Exit_Lower': 0.5,
        'Spread': S2 - hedge_ratios * S1,
        'Spread_Mean': (S2 - hedge_ratios * S1).rolling(window=window).mean(),
        'Hedge_Ratio': hedge_ratios
    })
    return df.dropna()


def get_jump_diffusion_signals(S1, S2, window, jump_threshold=3.0, z_entry=2.0, z_exit=0.0):
    """
    Merton Jump-Diffusion Model for Pairs Trading
    Filters out structural breaks by detecting Poisson jumps in the spread process.
    """
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=window).mean()
    
    # Calculate daily returns (differences) of the spread
    spread_diff = spread.diff()
    
    # Robust standard deviation estimation using Median Absolute Deviation (MAD)
    # MAD is resilient to outliers (jumps), giving us the "true" diffusion volatility
    mad = spread_diff.rolling(window=window).apply(lambda x: np.median(np.abs(x - np.median(x[~np.isnan(x)]))) if len(x[~np.isnan(x)])>0 else np.nan, raw=True)
    robust_std = mad * 1.4826  # Scale to match normal std
    
    # Jump Intensity: how many standard deviations the current move is
    jump_intensity = np.abs(spread_diff) / robust_std
    
    # Z-score based on standard calculation
    spread_std = spread.rolling(window=window).std()
    zscore = (spread - spread_mean) / spread_std
    
    # Dynamic Bounds: If a jump is detected, block trading by setting bounds to infinity
    is_jump = jump_intensity > jump_threshold
    
    upper_bounds = pd.Series(z_entry, index=spread.index)
    lower_bounds = pd.Series(-z_entry, index=spread.index)
    
    # Set bounds to infinity where jumps are detected to prevent entry
    upper_bounds[is_jump] = np.inf
    lower_bounds[is_jump] = -np.inf
    
    df = pd.DataFrame({
        'Indicator': zscore,
        'Upper_Bound': upper_bounds,
        'Lower_Bound': lower_bounds,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,
        'Spread': spread,
        'Spread_Mean': spread_mean,
        'Hedge_Ratio': hedge_ratio,
        'Jump_Intensity': jump_intensity
    })
    return df.dropna()

def get_sdde_signals(S1, S2, window, delay_tau=5, z_entry=2.0, z_exit=0.0):
    """
    Stochastic Delay Differential Equation (SDDE) model.
    Incorporates market memory and delayed mean-reversion by evaluating the 
    integral of past deviations (momentum).
    """
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    zscore = (spread - spread_mean) / spread_std
    
    # Calculate the delayed integral term (momentum of the spread deviation)
    # \int_{t-\tau}^{t} (X_s - \mu) ds
    deviation = spread - spread_mean
    delayed_integral = deviation.rolling(window=delay_tau).sum()
    
    # Normalize the integral term to make it comparable to Z-score
    integral_std = delayed_integral.rolling(window=window).std()
    normalized_momentum = delayed_integral / integral_std
    
    # Dynamic Z-score: 
    # If momentum is strong in the direction of the deviation, it delays entry.
    # We want the indicator to be "smaller" (less likely to hit bounds) if momentum is strong.
    # If Z > 0 and momentum > 0, the deviation is still expanding.
    alpha = 0.5  # Weight of the memory effect
    adjusted_indicator = zscore - alpha * normalized_momentum
    
    df = pd.DataFrame({
        'Indicator': adjusted_indicator,
        'Upper_Bound': z_entry,
        'Lower_Bound': -z_entry,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,
        'Spread': spread,
        'Spread_Mean': spread_mean,
        'Hedge_Ratio': hedge_ratio,
        'Raw_Zscore': zscore,
        'Momentum': normalized_momentum
    })
    return df.dropna()

# =============================================================================
# 發散模型 (Divergence / Trend Following)
# =============================================================================

def get_np_cusum_signals(S1, S2, window, k_shift=1.0, tau_threshold=5.0, z_exit=0.0):
    """
    非參數 CUSUM (Multivariate Non-Parametric CUSUM).
    不依賴常態分佈假設，使用局部中位數與中位數絕對偏差 (MAD) 計算結構破裂。
    """
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    
    # Non-parametric local median and MAD
    spread_median = spread.rolling(window=window).median()
    # MAD calculation
    spread_mad = spread.rolling(window=window).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    # Prevent division by zero
    spread_mad = spread_mad.replace(0, 1e-5)
    
    # Robust Z-score
    robust_z = (spread - spread_median) / spread_mad
    
    cusum_pos = pd.Series(0.0, index=spread.index)
    cusum_neg = pd.Series(0.0, index=spread.index)
    
    for i in range(1, len(robust_z)):
        if pd.isna(robust_z.iloc[i]):
            continue
        z = robust_z.iloc[i]
        # Empirical LLR proxy
        cusum_pos.iloc[i] = max(0, cusum_pos.iloc[i-1] + z - k_shift/2)
        cusum_neg.iloc[i] = max(0, cusum_neg.iloc[i-1] - z - k_shift/2)
        
    indicator = cusum_pos.where(cusum_pos > cusum_neg, -cusum_neg)
    
    df = pd.DataFrame({
        'Indicator': indicator,
        'Upper_Bound': tau_threshold,
        'Lower_Bound': -tau_threshold,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,
        'Spread': spread,
        'Spread_Mean': spread.rolling(window=window).mean(),
        'Hedge_Ratio': hedge_ratio
    })
    return df.dropna()

def get_gsadf_signals(S1, S2, window, adf_threshold=1.5, z_exit=0.0):
    """
    PSY GSADF (Generalized Sup ADF) 檢定逼近法。
    在滾動窗口內計算多個起點的 ADF 檢定，找出最大的 t-statistic (爆發根)。
    """
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=window).mean()
    
    gsadf_stats = pd.Series(0.0, index=spread.index)
    
    from statsmodels.tsa.stattools import adfuller
    import warnings
    warnings.filterwarnings("ignore")
    
    # To save computation, instead of full GSADF O(N^2), 
    # we compute SADF on the last `window` data points with min_length = window/2
    min_len = max(int(window / 2), 15)
    
    for i in range(window, len(spread)):
        current_spread = spread.iloc[i-window:i].dropna()
        if len(current_spread) >= min_len:
            max_adf = -np.inf
            # Expand from end backwards to find supremum ADF
            for start_idx in range(0, len(current_spread) - min_len + 1, max(1, int(window/5))):
                sub_spread = current_spread.iloc[start_idx:]
                try:
                    res = adfuller(sub_spread, maxlag=1, regression='c', autolag=None)
                    if res[0] > max_adf:
                        max_adf = res[0]
                except:
                    pass
            if max_adf != -np.inf:
                gsadf_stats.iloc[i] = max_adf
            else:
                gsadf_stats.iloc[i] = 0.0
                
    # Determine direction of explosion (positive or negative bubble)
    is_positive = spread > spread_mean
    indicator = gsadf_stats.copy()
    indicator[~is_positive] = -indicator[~is_positive] 
    
    df = pd.DataFrame({
        'Indicator': indicator,
        'Upper_Bound': adf_threshold,
        'Lower_Bound': -adf_threshold,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,
        'Spread': spread,
        'Spread_Mean': spread_mean,
        'Hedge_Ratio': hedge_ratio
    })
    return df.dropna()

def get_dcc_garch_vecm_signals(S1, S2, window, t_threshold=3.0, z_exit=0.0):
    """
    DCC-GARCH-VECM 特異性漂移爆發檢定 (Idiosyncratic Drift-Burst)。
    剔除 Beta 噪音，使用 GARCH 估計條件波動率，針對殘差局部漂移進行 t-檢定。
    """
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=window).mean()
    
    # 1. 取得 VECM 殘差 (此處以 OLS 殘差之一階差分近似 VECM 均衡誤差調整項)
    residuals = spread - spread.shift(1).rolling(window).mean() 
    residuals = residuals.fillna(0)
    
    # 2. GARCH 條件異方差估計 (使用 EMA 近似 GARCH(1,1) 的動態波動率以追求即時性與穩定性)
    span_vol = int(window / 2)
    conditional_var = (residuals ** 2).ewm(span=span_vol).mean()
    conditional_vol = np.sqrt(conditional_var)
    
    # 3. 殘差漂移項 (Local Drift of Residuals)
    span_drift = max(int(window / 5), 3)
    local_drift = residuals.ewm(span=span_drift).mean()
    
    # 4. Drift-Burst t-statistic
    t_stat = (local_drift / (conditional_vol + 1e-8)) * np.sqrt(span_drift)
    
    df = pd.DataFrame({
        'Indicator': t_stat,
        'Upper_Bound': t_threshold,
        'Lower_Bound': -t_threshold,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,
        'Spread': spread,
        'Spread_Mean': spread_mean,
        'Hedge_Ratio': hedge_ratio
    })
    return df.dropna()

def get_markov_regime_signals(S1, S2, window, prob_threshold=0.8, z_exit=0.5):
    """
    Markov Regime-Switching (MRS) Model.
    Fits a 2-regime model on rolling windows.
    Returns the probability of being in the High-Volatility / Trending regime.
    """
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    spread = S2 - hedge_ratio * S1
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    import statsmodels.api as sm_api
    import warnings
    warnings.filterwarnings("ignore")
    
    prob_regime_1 = pd.Series(0.0, index=spread.index)
    update_freq = 30  # Update every 30 days for speed
    
    for i in range(window, len(spread), update_freq):
        current_spread = spread.iloc[i-window:i].dropna()
        if len(current_spread) > 30:
            try:
                mod = sm_api.tsa.MarkovRegression(current_spread, k_regimes=2, trend='c', switching_variance=True)
                res = mod.fit(disp=False, search_reps=3, maxiter=20)
                var_0 = res.params.get('sigma2[0]', 1)
                var_1 = res.params.get('sigma2[1]', 1)
                
                high_vol_regime = 1 if var_1 > var_0 else 0
                probs = res.smoothed_marginal_probabilities[high_vol_regime]
                prob_regime_1.iloc[i] = probs.iloc[-1]
            except:
                prob_regime_1.iloc[i] = prob_regime_1.iloc[i-update_freq] if i >= update_freq else 0.0
                
    prob_regime_1 = prob_regime_1.replace(0.0, np.nan).ffill().fillna(0.0)
    
    zscore = (spread - spread_mean) / spread_std
    direction = np.sign(zscore)
    indicator = prob_regime_1 * direction
    
    df = pd.DataFrame({
        'Indicator': indicator,
        'Upper_Bound': prob_threshold,
        'Lower_Bound': -prob_threshold,
        'Exit_Upper': z_exit,
        'Exit_Lower': -z_exit,
        'Spread': spread,
        'Spread_Mean': spread_mean,
        'Hedge_Ratio': hedge_ratio
    })
    return df.dropna()
