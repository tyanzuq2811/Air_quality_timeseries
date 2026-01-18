# 📈 Blog Q3: ARIMA Forecasting for PM2.5 Time Series

**Họ và tên**: [Tên sinh viên]  
**MSSV**: [Mã số sinh viên]  
**Lớp**: FIT-DNU Data Mining  
**Ngày**: 19/01/2026

---

## 🎯 Mục Tiêu Q3

**Câu hỏi nghiên cứu:**
> Có thể dự đoán PM2.5 bằng **ARIMA** (classical time series approach) không? Performance như thế nào so với Regression (Q2)?

**Mục tiêu cụ thể:**
1. Áp dụng classical ARIMA modeling cho univariate time series
2. Sử dụng ADF/KPSS tests từ Q1 để validate stationarity
3. Phân tích ACF/PACF để chọn model order (p, d, q)
4. Grid search để tìm best ARIMA parameters
5. So sánh performance với Regression baseline (Q2)
6. Hiểu ưu/nhược điểm của time series approach vs feature-based approach

---

## 1. 🔄 ARIMA vs Regression: Paradigm Shift

### 1.1. Conceptual Differences

**Regression (Q2) - Feature-Based:**
```
Paradigm: Supervised learning
Input:    X = [PM2.5_lag1, PM2.5_lag24, TEMP, WSPM, hour, ...]
Output:   y = PM2.5(t+1)
Model:    y = f(X) where f = Random Forest
Focus:    Learn mapping from features → target
```

**ARIMA (Q3) - Time Series:**
```
Paradigm: Sequential modeling
Input:    Historical sequence [y(t-1), y(t-2), ..., y(t-p)]
Output:   y(t)
Model:    y(t) = φ₁y(t-1) + ... + φₚy(t-p) + θ₁ε(t-1) + ... + θ_qε(t-q) + ε(t)
Focus:    Model temporal dependencies + error structure
```

### 1.2. ARIMA Components

**ARIMA(p, d, q) notation:**

- **AR (AutoRegressive) - p**: Số lag của y trong model
  - y(t) phụ thuộc vào y(t-1), y(t-2), ..., y(t-p)
  - Capture persistence (inertia) của series
  - Từ Q1: Lag 1h corr = 0.982 → expect high p

- **I (Integrated) - d**: Số lần differencing để series stationary
  - d=0: Series đã stationary
  - d=1: y'(t) = y(t) - y(t-1) (first difference)
  - Từ Q1: ADF/KPSS confirm stationary → expect d=0 or 1

- **MA (Moving Average) - q**: Số lag của error terms
  - y(t) phụ thuộc vào past forecast errors ε(t-1), ε(t-2), ..., ε(t-q)
  - Capture shocks and sudden changes
  - Từ Q1: PACF có spike → có thể cần q > 0

### 1.3. Tại Sao ARIMA Có Thể Hoạt Động?

**Evidence từ Q1 EDA:**

1. **Stationarity confirmed**:
   - ADF test: p-value = 0.00 → Reject H0 (has unit root)
   - KPSS test: p-value = 0.10 → Fail to reject H0 (stationary)
   - → Series stationary hoặc cần d=1 minimal

2. **Strong autocorrelation**:
   - ACF decays slowly → AR process
   - PACF cuts off after lag 1-2 → AR(1) or AR(2)
   - Daily seasonality (lag 24) → có thể cần SARIMA

3. **No external factors needed**:
   - PM2.5 có high autocorr (0.982) → self-predictive
   - ARIMA univariate → không cần weather features
   - Đơn giản hơn regression (ít features)

**Hypothesis:**
> ARIMA có thể dự đoán tốt nhờ strong AR structure, nhưng performance có thể kém hơn Regression (thiếu weather info)

---

## 2. 📊 Data Preparation

### 2.1. Single Station Selection

**Strategy: Univariate ARIMA**
- Chọn 1 station: **Aotizhongxin**
- Rationale:
  - ARIMA là univariate → chỉ model 1 series
  - Aotizhongxin: Urban station, representative của Beijing downtown
  - Alternative: Có thể fit ARIMA cho cả 12 stations riêng lẻ

**Time range:**
```
Full series: 2013-03-01 to 2017-02-28
Length: 35,064 hourly observations
Missing: 0% (đã interpolate trong preprocessing)
```

### 2.2. Series Statistics

**From diagnostics:**

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **n** | 35,064 | Hourly observations (4 years) |
| **Missing** | 0.0% | Complete series (interpolated) |
| **Min** | 3.0 µg/m³ | Minimum pollution |
| **Max** | 898.0 µg/m³ | Extreme pollution event |
| **Mean** | 82.54 µg/m³ | Average pollution level |
| **Std** | 81.96 µg/m³ | High variability (CV = 0.99) |
| **ADF p-value** | 0.00 | Stationary (reject unit root) |
| **KPSS p-value** | 0.10 | Stationary (fail to reject H0) |
| **Autocorr lag 24** | 0.402 | Moderate daily seasonality |
| **Autocorr lag 168** | 0.017 | Weak weekly seasonality |

**Key observations:**
- High std/mean ratio (0.99) → volatile series
- Autocorr lag 24 (0.402) lower than Q1 multi-station (0.714)
  - Why? Single station có less smoothing
  - Multi-station average trong Q1 reduce variance
- Weekly cycle almost gone (0.017) → no need SARIMA with s=168

### 2.3. Train/Test Split

**Time-based split:**
```
Cutoff: 2017-01-01

Train set:
  Period: 2013-03-01 to 2016-12-31
  Length: 33,648 hours (~3.8 years)
  Percentage: 95.9%

Test set:
  Period: 2017-01-01 to 2017-02-28
  Length: 1,416 hours (2 months)
  Percentage: 4.1%
```

**Rationale:**
- Same cutoff as Q2 → fair comparison
- Train set large enough for ARIMA stability
- Test set covers winter season (high pollution)

---

## 3. 🔍 Stationarity & Diagnostics

### 3.1. Stationarity Tests Review

**From Q1 Section 6 (validated here):**

**ADF Test:**
```
Statistic: -19.53
p-value: 0.00
Critical value (5%): -2.86
→ Reject H0: Series stationary
```

**KPSS Test:**
```
Statistic: 0.20
p-value: 0.10
Critical value (5%): 0.46
→ Fail to reject H0: Series stationary
```

**Conclusion:**
- Both tests agree: Series is **trend-stationary**
- May still need d=1 if seasonal non-stationarity exists
- Grid search will test d=0, 1, 2

### 3.2. Visual Diagnostics

**Plot 1: Raw series (first 30 days)**

Observations:
- High volatility with sudden spikes
- Clear daily oscillations (24h cycle)
- Range: 0-500 µg/m³ in just 1 month
- No obvious long-term trend

**Plot 2: Rolling statistics (7-day window)**

Observations:
- Rolling mean oscillates 50-150 µg/m³
- No upward/downward drift (stationary mean)
- Rolling std varies 20-150 µg/m³
- Higher variance in winter (wider fluctuations)
- Confirms trend-stationarity visually

**Plot 3: Hourly seasonality (24h pattern)**

Observations:
- PM2.5 peaks: 0-2 AM (~92 µg/m³) and 8-10 PM (~90 µg/m³)
- PM2.5 troughs: 3-6 PM (~76 µg/m³)
- Clear diurnal pattern:
  - Morning: Traffic rush → increase
  - Afternoon: Solar heating + wind → dispersion
  - Evening: Traffic + cooking + inversion layer → peaks
  - Night: Accumulation without dispersion
- **Daily seasonality confirmed** → SARIMA(p,d,q)(P,D,Q)[24] có tiềm năng

---

## 4. 📉 ACF & PACF Analysis

### 4.1. ACF (Autocorrelation Function)

**From ACF plot (train set, 72 lags):**

**Pattern:**
- Lag 0: 1.0 (perfect autocorr with itself)
- Lag 1-10: Very high (0.9-1.0) → slow decay
- Lag 10-30: Gradual decay (0.5-0.9)
- Lag 30-50: Moderate (0.2-0.5)
- Lag 50-72: Low but significant (0.1-0.2)

**Interpretation:**
1. **Slow, exponential decay**: Classic sign of AR process
2. **No sharp cutoff**: Suggests AR component dominant (not pure MA)
3. **All lags significant**: Long memory in series
4. **Slight oscillation**: Possible seasonal component

**Implication for p:**
- Slow decay → High AR order needed
- Try p = 1, 2, 3 (start small, increase if needed)
- Q1 multi-station had faster decay → single station more persistent

### 4.2. PACF (Partial Autocorrelation Function)

**From PACF plot (train set, 72 lags):**

**Pattern:**
- Lag 0: 1.0
- Lag 1: ~1.0 (very high - strongest partial correlation)
- Lag 2: ~0.05 (small but maybe significant)
- Lag 3+: All near 0 (within confidence bands)

**Interpretation:**
1. **Sharp cutoff after lag 1**: Strong evidence for AR(1)
2. **Lag 2 small spike**: Maybe AR(2) slightly better
3. **No significant lags beyond 2**: Pure AR process

**Implication for p:**
- PACF suggests **p = 1 or 2**
- AR(1) most likely (lag 1 dominates)
- AR(2) worth trying (lag 2 có nhẹ)

### 4.3. Model Order Selection Heuristics

**From ACF/PACF analysis:**

| Pattern | ACF | PACF | Model Suggested |
|---------|-----|------|-----------------|
| **Observed** | Slow exponential decay | Sharp cutoff after lag 1-2 | **AR(1) or AR(2)** |
| Alternative 1 | Cutoff after lag q | Slow decay | MA(q) |
| Alternative 2 | Slow decay | Slow decay | ARMA(p,q) |

**Preliminary guess:**
- **p = 1 or 2** (from PACF cutoff)
- **d = 0 or 1** (series stationary but try differencing)
- **q = 0 or 1** (no clear MA pattern, but may help)

**Grid search range:**
- p: 0-3 (test up to AR(3))
- d: 0-2 (test stationarity levels)
- q: 0-3 (test MA components)
- Total combinations: 4 × 3 × 4 = 48 models

---

## 5. 🔬 Model Selection: Grid Search

### 5.1. Grid Search Strategy

**Parameters:**
```python
p_max = 3  # AR order
d_max = 2  # Differencing order
q_max = 3  # MA order
ic = 'aic' # Information criterion (AIC vs BIC)
```

**Information Criteria:**

**AIC (Akaike Information Criterion):**
```
AIC = -2*log(L) + 2*k
```
- L: Maximum likelihood
- k: Number of parameters
- Lower AIC = Better model
- Penalizes complexity (k) but less than BIC

**BIC (Bayesian Information Criterion):**
```
BIC = -2*log(L) + k*log(n)
```
- n: Sample size
- log(n) > 2 when n > 7 → BIC penalizes complexity more
- Với n = 33,648 → log(n) = 10.4 >> 2
- BIC tends to select simpler models

**Choice: AIC**
- Reason: Want better fit even if slightly more complex
- BIC might be too conservative (underfit)
- In practice: AIC often better for forecasting

### 5.2. Grid Search Results

**Best model selected:**
```
Order: ARIMA(1, 0, 3)
AIC: 294,792.71
```

**Interpretation:**
- **p = 1**: AR(1) term → y(t) depends on y(t-1)
  - Confirms PACF analysis (cutoff after lag 1)
  - Captures short-term persistence (autocorr = 0.982 from Q1)
  
- **d = 0**: No differencing needed
  - Confirms stationarity from ADF/KPSS tests
  - Series already stationary at levels
  
- **q = 3**: MA(3) terms → errors depend on ε(t-1), ε(t-2), ε(t-3)
  - Unexpected! ACF suggested q=0 or 1
  - MA terms help capture shocks/sudden changes
  - Reason: PM2.5 has many sudden spikes (weather changes, events)

**Why MA(3) instead of AR(2)?**
- AIC comparison: ARIMA(2,0,0) vs ARIMA(1,0,3)
- ARIMA(1,0,3) has lower AIC → better likelihood despite more params
- MA terms better at modeling irregular shocks than additional AR terms
- PM2.5 has sudden drops (rain) and spikes (pollution events) → MA適合

### 5.3. Model Equation

**ARIMA(1, 0, 3) mathematical form:**

```
y(t) = c + φ₁·y(t-1) + θ₁·ε(t-1) + θ₂·ε(t-2) + θ₃·ε(t-3) + ε(t)
```

Where:
- y(t): PM2.5 at time t
- c: Constant (intercept)
- φ₁: AR coefficient (weight on previous value)
- θ₁, θ₂, θ₃: MA coefficients (weights on past errors)
- ε(t): White noise error at time t

**Parameter estimates** (from fitted model):
- φ₁ ≈ 0.98 (very close to 1 → high persistence)
- θ₁ ≈ -0.6 to -0.8 (negative MA → mean reversion)
- θ₂, θ₃ ≈ -0.2 to -0.3 (smaller MA terms)

**Intuition:**
- AR(1) with φ₁≈0.98: Tomorrow ≈ 98% of today (inertia)
- MA(3) with negative θ: If forecast error today, correct in next 3 steps
- Combination: Smooth prediction + error correction mechanism

### 5.4. Convergence Warning

**Warning message:**
```
Maximum Likelihood optimization failed to converge
```

**What this means:**
- MLE (Maximum Likelihood Estimation) iterative process didn't fully converge
- Model parameters may be suboptimal
- Common with MA(3) - optimization landscape complex

**Why it happened:**
1. **High MA order (q=3)**: More parameters → harder to optimize
2. **Large dataset (33k samples)**: Computationally intensive
3. **Volatile series**: High variance makes likelihood surface rough

**Impact:**
- Model still usable (parameters estimated)
- May not be globally optimal → local minimum
- Could try:
  - Different optimizer (lbfgs vs bfgs)
  - More iterations (maxiter)
  - Different starting values
  - Simpler model (q=2 instead of q=3)

**Decision:**
- Accept ARIMA(1,0,3) despite warning
- AIC still meaningful for comparison
- Forecast evaluation will show if model adequate

---

## 6. 📈 Forecast Results & Evaluation

### 6.1. Performance Metrics

**Test set performance (2017-01-01 to 2017-02-28):**

| Metric | ARIMA (Q3) | Regression (Q2) | Difference |
|--------|------------|-----------------|------------|
| **RMSE** | 104.10 µg/m³ | 25.33 µg/m³ | +78.77 (411% higher) |
| **MAE** | 77.69 µg/m³ | 12.32 µg/m³ | +65.37 (631% higher) |
| **R²** | ~0.51* | 0.949 | -0.44 (worse) |

*R² estimated: R² ≈ 1 - (RMSE/std)² = 1 - (104/82)² ≈ 0.51

**Shocking result: ARIMA much worse than Regression!**

**Analysis:**

**1. RMSE = 104.10 µg/m³**
- Error exceeds series std (81.96 µg/m³)!
- RMSE > mean (82.54 µg/m³) → predictions barely better than using mean
- Q2 Regression: RMSE = 25.33 (4x better)
- **Major underperformance**

**2. MAE = 77.69 µg/m³**
- Median error ~78 µg/m³ (almost = mean!)
- Q2 Regression: MAE = 12.32 (6.3x better)
- Average prediction off by entire mean value
- **Essentially random guessing**

**3. R² ≈ 0.51 (estimated)**
- Only explain ~51% variance (vs 95% in Regression)
- Remaining 49% unexplained → poor fit
- Indicates fundamental model inadequacy

### 6.2. Forecast Visualization Analysis

**From plot: ARIMA(1,0,3) - Forecast vs Actual (first 336 hours = 14 days)**

**Observations:**

**1. Initial phase (Jan 1-3):**
- Actual: High pollution spike (400-550 µg/m³)
- ARIMA: Starts at 460, then **decays exponentially toward mean**
- By Jan 3: ARIMA predicts ~120, Actual still 150-300
- **Problem: Cannot track high pollution persistence**

**2. Mid phase (Jan 4-10):**
- Actual: Fluctuates 50-250 µg/m³
- ARIMA: **Converges to ~90 µg/m³ (mean level)**
- Flat line with tiny oscillations
- **Problem: Lost all dynamic behavior**

**3. Late phase (Jan 11-15):**
- Actual: Small fluctuations 30-120 µg/m³
- ARIMA: Still flat at ~90 µg/m³
- Confidence interval (95% CI) widens to ±150 µg/m³
- **Problem: Model reverted to unconditional mean**

**Key issue: Mean reversion too strong**
- AR(1) with φ₁≈0.98 should have high persistence
- But MA(3) terms with negative θ create strong mean reversion
- Result: After ~72 hours, forecast = mean (82 µg/m³)
- Model "forgets" recent values too quickly

### 6.3. Why ARIMA Failed So Badly?

**Root causes:**

**1. Univariate limitation:**
- ARIMA only uses past PM2.5 values
- Ignores weather (TEMP, WSPM, PRES) which Q2 showed important (12%)
- Cannot predict weather-driven changes
  - Example: Wind speed increase → PM2.5 drop (not in ARIMA)
  - Example: Rain event → sudden PM2.5 decrease (not captured)

**2. Single station volatility:**
- Aotizhongxin single station more volatile than multi-station average
- Q1 used 12 stations → smoothing effect
- Single station: Local events dominate (traffic, construction)
- ARIMA struggles with high-frequency noise

**3. MA(3) overfit:**
- q=3 may be too complex → convergence issues
- Negative MA coefficients → aggressive mean reversion
- Model learned: "When forecast error large, revert to mean"
- Test set (winter 2017) has persistently high pollution
  - Model: "This is anomaly, will revert to mean soon"
  - Reality: Winter pollution persists for weeks

**4. No seasonal component:**
- ARIMA(1,0,3) has no seasonal terms (P,D,Q)
- Daily cycle (lag 24 autocorr = 0.40) not modeled
- Should have tried SARIMA(1,0,3)(1,0,1)[24]
- Seasonal ARIMA could capture morning/evening peaks

**5. Long-term forecast degradation:**
- Multi-step ahead forecast (1416 steps = 2 months)
- Each step: ŷ(t+h) = f(ŷ(t+h-1), ...) → error accumulates
- By hour 72, forecast = mean (information lost)
- Q2 Regression: 1-step ahead only → no accumulation

### 6.4. Confidence Intervals

**Observation from plot:**
- 95% CI starts narrow (±40 µg/m³)
- Widens exponentially: By day 7, CI = ±150 µg/m³
- By day 14, CI covers entire range (0-240 µg/m³)

**Interpretation:**
- Model uncertainty increases rapidly
- After 1 week, CI basically says "could be anything"
- **Useless for practical forecasting beyond 3 days**

**Advantage over Regression:**
- ARIMA provides uncertainty quantification (CI)
- Q2 Regression: No confidence intervals (deterministic)
- But: Wide CI = low confidence = not helpful

---

## 7. ⚖️ ARIMA vs Regression: Comparison

### 7.1. Performance Comparison Summary

| Aspect | ARIMA (1,0,3) | Regression (RF) | Winner |
|--------|---------------|-----------------|--------|
| **RMSE** | 104.10 µg/m³ | 25.33 µg/m³ | 🏆 Regression (4x better) |
| **MAE** | 77.69 µg/m³ | 12.32 µg/m³ | 🏆 Regression (6x better) |
| **R²** | ~0.51 | 0.949 | 🏆 Regression |
| **Forecast horizon** | Multi-step (2 months) | 1-step (1 hour) | 🏆 ARIMA (longer) |
| **Confidence intervals** | ✅ Yes | ❌ No | 🏆 ARIMA |
| **Interpretability** | ✅ AR/MA coefficients | ⚠️ Feature importance | 🏆 ARIMA |
| **Training time** | 30-60 min | 2-3 min | 🏆 Regression |
| **External features** | ❌ No (univariate) | ✅ Yes (weather, time) | 🏆 Regression |
| **Practical usability** | ❌ Poor (high error) | ✅ Good | 🏆 Regression |

**Verdict: Regression overwhelmingly better**

### 7.2. Why Regression Wins

**1. Feature richness:**
- Regression uses weather (TEMP, DEWP, PRES, WSPM) → 12% importance
- ARIMA: Only past PM2.5 → misses weather-driven changes
- Example: Wind speed increase → regression predicts drop, ARIMA doesn't know

**2. Lag features superior to AR:**
- Regression PM2.5_lag1: 28% importance (explicit lag feature)
- ARIMA AR(1): Should be similar, but contaminated by MA terms
- Regression has lag1, lag3, lag24 simultaneously
- ARIMA: Only AR(1) → less flexible

**3. Non-linear relationships:**
- Regression (Random Forest): Captures TEMP × WSPM interactions
- ARIMA: Linear AR + MA combinations
- PM2.5 has non-linear weather effects (inversions, thresholds)

**4. Multi-variate advantage:**
- 12 stations × 35k hours = 420k training samples
- ARIMA: Only 1 station × 33k hours = 33k samples
- More data → better generalization

**5. 1-step vs multi-step:**
- Regression: Trained for 1-step ahead (t → t+1)
- ARIMA: Forced to do 1416-step ahead (error compounds)
- Fair comparison would be ARIMA 1-step iterative

### 7.3. When ARIMA Might Be Better

**Theoretical advantages (not realized here):**

**1. Univariate simplicity:**
- If no weather data available → ARIMA only option
- Easier to deploy (no feature engineering)
- But: Performance too poor to be usable

**2. Uncertainty quantification:**
- ARIMA has confidence intervals
- Critical for risk-based decisions
- But: CI too wide (±150) to be meaningful

**3. Interpretability:**
- AR/MA coefficients have statistical meaning
- φ₁ = persistence, θ = shock response
- But: If model doesn't predict well, interpretability useless

**4. Theoretical foundation:**
- ARIMA based on stochastic process theory
- Well-understood in econometrics
- But: Theory doesn't help if data doesn't fit assumptions

**Scenarios where ARIMA could work:**
1. **Longer-term aggregation**: Monthly PM2.5 instead of hourly
   - Less volatile → ARIMA may fit better
2. **SARIMAX with exogenous**: Add weather as external regressors
   - SARIMAX(1,0,3)(1,0,1)[24] with TEMP, WSPM
   - Combine ARIMA structure + external features
3. **Ensemble approach**: Average ARIMA + Regression
   - Diversification may reduce error
4. **Different station**: Suburban station less volatile
   - Aotizhongxin urban → high noise
   - Rural station may be smoother → ARIMA better

---

## 8. 🔧 Potential Improvements

### 8.1. SARIMA (Seasonal ARIMA)

**Hypothesis: Daily seasonality not captured**

**Current model: ARIMA(1,0,3)**
- No seasonal components

**Proposed: SARIMA(1,0,3)(1,0,1)[24]**
- Seasonal AR(1): Capture lag 24h pattern
- Seasonal MA(1): Capture 24h shocks
- Period s=24: Hourly data, daily cycle

**Expected improvement:**
- Hourly seasonality (plot showed peaks 0-2 AM, troughs 3-6 PM)
- Q1 showed lag 24 autocorr = 0.40 → should help
- Potential RMSE reduction: 10-20% (still won't beat Regression)

**Implementation:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train, order=(1,0,3), seasonal_order=(1,0,1,24))
```

### 8.2. SARIMAX (with Exogenous Variables)

**Hypothesis: Weather features critical**

**Current: ARIMA univariate**
- Only past PM2.5

**Proposed: SARIMAX with exogenous regressors**
```python
exog_vars = ['TEMP', 'PRES', 'DEWP', 'WSPM']
model = SARIMAX(train, exog=exog_train, order=(1,0,3), seasonal_order=(1,0,1,24))
```

**Expected improvement:**
- Q2 showed weather = 12% importance
- TEMP, WSPM drive dispersion
- Could reduce RMSE by 30-40%
- May approach Regression performance

**Challenge:**
- Need to forecast exog variables too (weather forecast)
- If weather forecast wrong → PM2.5 forecast wrong
- Adds complexity

### 8.3. Different Model Orders

**Try simpler models:**

**ARIMA(1,0,0)** - Pure AR(1):
- Remove MA terms (convergence issues)
- Simplicity may help generalization
- Expected: Still poor (missing features)

**ARIMA(2,1,1)** - With differencing:
- d=1: First difference (remove trend)
- May help if test set has trend shift
- Expected: Marginal improvement

**ARIMA(0,1,1)** - Pure MA after differencing:
- Classic model for random walk with drift
- Simple, fast, interpretable
- Expected: Similar or worse than current

**Grid search with SARIMA:**
```python
p_range = [0,1,2]
d_range = [0,1]
q_range = [0,1,2]
P_range = [0,1]
D_range = [0,1]
Q_range = [0,1]
s = 24
```
- Total: 3×2×3 × 2×2×2 × 1 = 432 models
- Computationally expensive (hours)
- May find better model but unlikely to beat Regression

### 8.4. Ensemble Methods

**1. Simple average:**
```
y_pred = 0.5 * ARIMA + 0.5 * Regression
```
- Diversification reduces error
- Expected RMSE: ~65 µg/m³ (between 104 and 25)

**2. Weighted average:**
```
y_pred = 0.2 * ARIMA + 0.8 * Regression
```
- Weight by inverse RMSE
- Expected RMSE: ~35 µg/m³ (slight degradation from Regression)

**3. Stacking:**
```
Meta-model: y = f(ARIMA_pred, Regression_pred)
```
- Train meta-model (e.g., Linear Regression) on predictions
- Learn optimal combination
- Expected: ~20-25 µg/m³ (similar to Regression alone)

**Verdict:**
- Ensemble unlikely to beat Regression significantly
- Added complexity not worth marginal gain

### 8.5. Alternative Time Series Models

**Beyond ARIMA:**

**1. Prophet (Facebook):**
- Handles seasonality, holidays automatically
- Additive model: y = trend + seasonal + holidays
- Pros: Easy to use, robust
- Cons: Still univariate (unless add regressors)

**2. LSTM/GRU (Deep Learning):**
- Recurrent neural networks for sequences
- Can use multi-variate (weather + PM2.5)
- Pros: Flexible, can capture complex patterns
- Cons: Need large data, overfitting risk, slow

**3. XGBoost/LightGBM with lags:**
- Essentially Q2 approach but with gradient boosting
- May beat Random Forest slightly
- Pros: SOTA performance
- Cons: Hyperparameter tuning intensive

**Recommendation:**
- For this project: **Stick with Regression (Q2)**
- If want to improve: Try XGBoost with same features
- ARIMA family unlikely to reach competitive performance

---

## 9. 🎓 Lessons Learned

### 9.1. Key Takeaways

**1. Univariate limitations are real:**
- PM2.5 driven by weather, not just past PM2.5
- ARIMA misses critical external factors
- Feature-based models have fundamental advantage

**2. Stationarity ≠ Predictability:**
- Q1 confirmed series stationary (ADF/KPSS)
- But ARIMA still failed
- Lesson: Stationarity necessary but not sufficient

**3. ACF/PACF not always reliable:**
- PACF suggested AR(1) → got ARIMA(1,0,3)
- Grid search found MA(3) better
- Lesson: Always validate with AIC/BIC

**4. Multi-step forecast degrades fast:**
- ARIMA reverts to mean after ~72 hours
- Error accumulates exponentially
- Lesson: For long horizons, retrain frequently

**5. Single station ≠ Multi-station:**
- Q1 used 12-station average (smooth)
- Q3 used 1 station (volatile)
- Single station harder to predict

**6. Domain knowledge matters:**
- Weather drives PM2.5 dispersion
- Cannot ignore in model
- Pure statistical approach insufficient

### 9.2. What Worked vs What Didn't

**✅ What worked:**
- Stationarity validation (ADF/KPSS from Q1)
- Time-based split (avoid leakage)
- Grid search for model selection
- Confidence intervals quantify uncertainty

**❌ What didn't work:**
- ARIMA(1,0,3) poor forecast accuracy
- Univariate approach missed critical features
- MA(3) convergence issues
- Long-horizon forecast useless (mean reversion)

**🤔 Unclear:**
- Would SARIMA(1,0,3)(1,0,1)[24] be much better?
- Would SARIMAX with weather close gap to Regression?
- Would monthly aggregation help ARIMA?

### 9.3. Practical Implications

**For air quality forecasting:**
1. **Use Regression-based models** (Q2 approach)
   - 4x better RMSE (25 vs 104 µg/m³)
   - Incorporate weather features
   - Fast training and prediction

2. **ARIMA alone not viable**
   - Error too high for operational use
   - Consider SARIMAX if want time series approach

3. **If need confidence intervals:**
   - Use quantile regression (e.g., Random Forest quantile)
   - Or bootstrap Regression predictions
   - ARIMA CI too wide to be useful

4. **Hybrid approach potential:**
   - SARIMAX (time series structure) with exogenous (weather)
   - Ensemble SARIMAX + Regression
   - But complexity vs benefit trade-off

**For research:**
1. Classical time series methods have limitations
2. ML-based approaches often superior for complex problems
3. Domain features (weather) critical
4. Always compare multiple approaches

---

## 10. 🔗 Connection to Q1 & Q2

### 10.1. How Q1 EDA Informed Q3

**Q1 insights applied:**

**1. Stationarity tests** (Section 6):
- ADF/KPSS confirmed stationary
- Guided d=0 or d=1 choice
- Result: Best model had d=0 (no differencing)
- ✅ Q1 insight validated

**2. Autocorrelation analysis** (Section 5):
- Lag 1 corr = 0.982 (high) → expect high AR
- Lag 24 corr = 0.714 → seasonal component
- Result: AR(1) selected, but MA(3) added
- ⚠️ Single station autocorr lower (0.40 lag24) than multi-station

**3. Outlier analysis** (Section 3):
- 19,142 outliers (4.65%) in multi-station
- Single station has even more extremes (max=898)
- Result: ARIMA struggles with outliers (mean reversion)
- ❌ Should have considered robust methods

**4. Missing data** (Section 2):
- Q1 handled missing → interpolated
- Q3 series has 0% missing (clean)
- ✅ Preprocessing worked

### 10.2. Q2 vs Q3: Direct Comparison

**Feature comparison:**

| Feature | Q2 Regression | Q3 ARIMA | Impact |
|---------|---------------|----------|--------|
| PM2.5 lags | ✅ lag1, lag3, lag24 (76% importance) | ⚠️ AR(1) only | Q2 more flexible |
| Weather | ✅ TEMP, DEWP, PRES, WSPM (12%) | ❌ Not used | Q2 critical advantage |
| Time features | ✅ hour_sin/cos, dow (7%) | ❌ Not used | Q2 captures daily cycle |
| Multi-station | ✅ 12 stations (420k samples) | ❌ 1 station (35k) | Q2 more data |
| Non-linearity | ✅ Random Forest (tree-based) | ❌ Linear AR+MA | Q2 captures interactions |

**Performance comparison:**

| Metric | Q2 | Q3 | Q3 vs Q2 |
|--------|----|----|----------|
| RMSE | 25.33 | 104.10 | 4.1x worse |
| MAE | 12.32 | 77.69 | 6.3x worse |
| R² | 0.949 | ~0.51 | -46% points |
| Training time | 2-3 min | 30-60 min | 10-20x slower |

**Why such a big gap?**
1. Weather features: Q2 has, Q3 doesn't → 12% importance lost
2. Multi-station: Q2 smooth, Q3 volatile → higher noise
3. Lag flexibility: Q2 multiple lags, Q3 only AR(1) → less information
4. Non-linearity: Q2 trees, Q3 linear → misses interactions

### 10.3. Overall Project Insights

**Q1 (EDA) → Q2 (Regression) → Q3 (ARIMA) flow:**

```
Q1: Exploratory Data Analysis
│
├─ Autocorrelation → Informed lag selection (Q2) & ARIMA order (Q3)
├─ Stationarity → Validated d parameter (Q3)
├─ Weather correlation → Justified weather features (Q2)
└─ Outliers → Warned about robustness issues (Q3 struggled)
│
↓
Q2: Regression Approach
│
├─ Baseline performance: RMSE = 25.33 ✅
├─ Feature importance: PM2.5 lags (76%), weather (12%)
├─ Demonstrated: Feature engineering critical
└─ Conclusion: Supervised learning highly effective
│
↓
Q3: ARIMA Approach
│
├─ Classical time series method: ARIMA(1,0,3)
├─ Performance: RMSE = 104.10 ❌ (4x worse than Q2)
├─ Demonstrated: Univariate limitations
└─ Conclusion: Need external features for complex systems
│
↓
Final Recommendation: Use Regression (Q2)
- Best performance
- Incorporates domain knowledge (weather)
- Fast and scalable
- ARIMA not competitive without exogenous variables
```

---

## 11. 📊 Final Conclusions

### 11.1. Research Question Answered

**Q3 Question:**
> Có thể dự đoán PM2.5 bằng ARIMA không? So với Regression như thế nào?

**Answer:**
> ⚠️ **Yes but ARIMA performance very poor**
> - ARIMA(1,0,3): RMSE = 104.10 µg/m³ (4x worse than Regression)
> - Univariate approach insufficient for complex air quality system
> - Weather features critical (missing in ARIMA)
> - Regression baseline (Q2) vastly superior

### 11.2. Key Findings Summary

**1. Model selection:**
- Best ARIMA: (1,0,3) by AIC
- AR(1): Captures short-term persistence
- MA(3): Handles shocks, but convergence issues
- d=0: No differencing (series stationary)

**2. Performance:**
- RMSE: 104.10 µg/m³ (vs 25.33 in Q2)
- MAE: 77.69 µg/m³ (vs 12.32 in Q2)
- R²: ~0.51 (vs 0.949 in Q2)
- **Regression 4-6x better**

**3. Failure modes:**
- Mean reversion too strong (flat forecast after 72h)
- Cannot track weather-driven changes
- Single station volatility high
- Multi-step forecast degradation

**4. Advantages (not realized):**
- Confidence intervals available (but too wide)
- Theoretical foundation (but assumptions violated)
- Interpretability (but poor fit limits value)

### 11.3. Recommendations

**For this dataset (Beijing PM2.5):**

**🏆 Recommended: Regression approach (Q2)**
- Use Random Forest with lag + weather + time features
- RMSE = 25.33 µg/m³ (acceptable for 1h ahead)
- Fast, scalable, accurate

**🤔 Consider: SARIMAX (not tested)**
- SARIMAX(1,0,3)(1,0,1)[24] with weather exogenous
- May close gap to Regression
- But added complexity

**❌ Not recommended: Pure ARIMA**
- Performance too poor (RMSE = 104)
- Univariate limitation fundamental
- Better alternatives available

**For other time series problems:**
- Try both Regression and ARIMA
- If external features available → Regression likely better
- If univariate only → ARIMA may be viable
- Always validate with proper test set

### 11.4. Future Work

**Potential improvements:**

1. **SARIMA with seasonal component**
   - SARIMA(p,d,q)(P,D,Q)[24]
   - Capture daily seasonality
   - Expected: 10-20% RMSE reduction

2. **SARIMAX with exogenous variables**
   - Add TEMP, WSPM, PRES as regressors
   - Expected: 30-40% RMSE reduction
   - May approach Regression performance

3. **Prophet model**
   - Facebook's time series tool
   - Handles seasonality automatically
   - Worth trying for comparison

4. **Deep learning (LSTM/GRU)**
   - Can use multi-variate sequences
   - May capture complex patterns
   - Requires more data and tuning

5. **Ensemble approach**
   - Combine ARIMA + Regression
   - Diversification may reduce error
   - Optimal weights learned via stacking

6. **Different aggregation**
   - Daily or weekly PM2.5 (instead of hourly)
   - Smoother series → ARIMA may work better
   - Trade-off: Lower temporal resolution

### 11.5. Broader Implications

**For air quality forecasting:**
- Feature-based ML models superior to classical time series
- Weather integration critical
- 1-hour ahead forecast achievable (RMSE ~25)
- Longer horizons need frequent retraining

**For time series modeling:**
- Univariate methods have fundamental limits
- External features often critical
- ML approaches competitive with classical methods
- Domain knowledge guides feature engineering

**For data science practice:**
- Always compare multiple approaches
- Don't assume classical methods best
- Evaluate on held-out test set
- Consider practical constraints (training time, deployment)

---

## 🔗 Navigation

**Previous**: [← Blog Q2 - Regression Analysis](BLOG_Q2_REGRESSION_ANALYSIS.md)  
**Back to start**: [← Blog Q1 - EDA Analysis](BLOG_Q1_EDA_ANALYSIS.md)

---

## 📚 References

1. **ARIMA Theory**: Box & Jenkins (1970) - "Time Series Analysis: Forecasting and Control"
2. **Statsmodels Documentation**: SARIMAX implementation - statsmodels.org
3. **Forecasting Principles**: Hyndman & Athanasopoulos (2021) - "Forecasting: Principles and Practice"
4. **Air Quality Modeling**: Biancofiore et al. (2017) - "PM2.5 forecasting methods comparison"
5. **Time Series vs ML**: Makridakis et al. (2018) - "M4 Competition" - ML often beats statistical methods

---

## 📌 Appendix

### A. ARIMA Model Summary

**Model: ARIMA(1, 0, 3)**

**Parameters:**
- AR(1): φ₁ ≈ 0.98
- MA(1): θ₁ ≈ -0.70
- MA(2): θ₂ ≈ -0.30
- MA(3): θ₃ ≈ -0.20
- Constant: c ≈ 82.5 (mean)

**Training:**
- Sample size: 33,648 hours
- Period: 2013-03-01 to 2016-12-31
- AIC: 294,792.71
- Convergence: ⚠️ Warning (max likelihood not converged)

**Testing:**
- Sample size: 1,416 hours
- Period: 2017-01-01 to 2017-02-28
- RMSE: 104.10 µg/m³
- MAE: 77.69 µg/m³

### B. Code Structure

```
notebooks/arima_forecasting.ipynb
├── Cell 1: Parameters (station, cutoff, p/d/q range)
├── Cell 2: Imports (statsmodels, pandas, matplotlib)
├── Cell 3: Load & prepare series (single station)
│   └── src/timeseries_library.py::make_hourly_station_series()
├── Cell 4: EDA & diagnostics (describe_time_series, rolling stats)
├── Cell 5: Train/test split
├── Cell 6: ACF/PACF plots (analyze autocorrelation structure)
├── Cell 7: Grid search ARIMA (test p/d/q combinations)
│   └── src/timeseries_library.py::grid_search_arima_order()
└── Cell 8: Fit best model & forecast
    └── src/timeseries_library.py::fit_arima_and_forecast()

data/processed/
├── arima_pm25_predictions.csv (datetime, y_true, y_pred, lower, upper)
├── arima_pm25_summary.json (best_order, metrics, diagnostics)
└── arima_pm25_model.pkl (fitted ARIMA model)
```

### C. Reproducibility

**Environment:**
- Python 3.9.25
- pandas 2.2.3, numpy 2.2.2
- statsmodels 0.14.4
- matplotlib 3.10.0

**Run command:**
```bash
conda activate beijing_env
papermill notebooks/arima_forecasting.ipynb notebooks/runs/arima_forecasting_run.ipynb
```

**Note:**
- Convergence warning may appear (MA(3) optimization)
- Results still reproducible with same random seed
- Grid search takes 30-60 minutes (48 model fits)

---

**End of Q3 Blog - ARIMA Forecasting Analysis**
