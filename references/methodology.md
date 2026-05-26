# Statistical Methodology Deep-Dive

## Overview

This document provides a rigorous explanation of the statistical methods used in the Lotto Max ML Predictor. All calculations are updated for the current pool of **1 to 52** numbers (as of May 2026).

## Updated Probability Baseline

For a uniformly random lottery with k=7 numbers drawn from N=52:

```
P(number i appears in a draw) = k/N = 7/52 ≈ 0.13462
Expected sum of 7 numbers = 7 × (52+1)/2 = 7 × 26.5 = 185.5
Total combinations = C(52,7) = 133,784,560
```

## 1. Frequency Analysis

### Mathematical Foundation

Over n draws, the expected count for each number follows a Binomial distribution:

```
E[Xi] = n × (7/52)
Var[Xi] = n × (7/52) × (45/52)
```

### Z-Score Computation

```
Zi = (Xi - E[Xi]) / sqrt(Var[Xi])
```

Under H0, Zi ~ approximately N(0,1). Numbers with |Zi| > 1.96 are statistically significant at the 5% level.

### Chi-Squared Goodness-of-Fit

```
chi_sq = Σ (Oi - Ei)^2 / Ei  for i = 1 to 52
```

Under H0, chi_sq ~ chi-squared(51).

### Limitations
- Multiple testing: with 52 numbers, some will appear significant by chance
- Frequency analysis assumes stationarity
- Short windows have high variance

## 2. Hot/Cold Number Analysis

### Temperature Score

```
Ti = (Oi - E[Oi]) / E[Oi]
```

where E[Oi] = w × 7/52 is the expected count over window w.

### Mean Reversion Probability

For extreme temperatures:

```
P(reversion | T > threshold) ≈ 0.65 + |T| × 0.2
```

### Limitations
- "Hot" and "cold" are descriptive, not predictive
- Mean reversion is a tendency, not a law

## 3. Gap Analysis

### Gap Distribution

Under uniformity, the gap distribution is geometric:

```
P(gap = g) = (45/52)^g × (7/52)
```

### Next Appearance Probability

```
P(appears next | current gap = g) = 1 - exp(-g / avg_gap)
```

### Overdue Score

```
overdue_score = (current_gap - avg_gap) / std_gap
```

Numbers with overdue_score > 1.5 are flagged as "overdue".

## 4. Markov Chain Analysis

### First-Order

```
P(number j appears in draw t | number i appeared in draw t-1)
Tij = count(i→j) / Σk count(i→k)
```

### Second-Order

```
P(number k appears | numbers i,j appeared in previous draw)
```

Captures pairwise sequential dependencies but requires more data.

### Stationary Distribution

```
pi = pi × T  (computed via power iteration)
```

Should equal the uniform distribution (7/52 per number) if draws are truly random.

## 5. Bayesian Inference

### Conjugate Prior

```
Prior: P(theta_i) = Beta(1, 1)
Likelihood: Xi | theta_i ~ Binomial(n, theta_i)
Posterior: theta_i | Xi ~ Beta(1 + Xi, 1 + n - Xi)
```

### Posterior Mean

```
E[theta_i | data] = (1 + Xi) / (2 + n)
```

### James-Stein Shrinkage

```
theta_hat_JS = lambda × theta_hat_MLE + (1 - lambda) × (7/52)
```

## 6. Pattern Detection

### Range Distribution (Updated for 52)

Numbers binned into: 1-10, 11-20, 21-30, 31-40, 41-52

Under uniformity, each bin has expected proportion:
- 1-10: 10/52 ≈ 19.2%
- 11-20: 10/52 ≈ 19.2%
- 21-30: 10/52 ≈ 19.2%
- 31-40: 10/52 ≈ 19.2%
- 41-52: 12/52 ≈ 23.1%

### Even/Odd Ratio

Under uniformity: 26 even (2,4,...,52), 26 odd (1,3,...,51)

Expected even count per draw: 7 × 26/52 = 3.5

### Runs Test

Wald-Wolfowitz runs test for randomness of binary sequences:

```
Z = (R - E[R]) / sqrt(Var[R])
```

## 7. Entropy & Randomness

### Shannon Entropy

```
H = -Σ p_i × log2(p_i)
```

For 52 numbers with p_appear = 7/52:

```
H_max = 52 × [-(7/52)×log2(7/52) - (45/52)×log2(45/52)]
     ≈ 52 × 0.3971 ≈ 20.65 bits
```

### Autocorrelation

Lag-1 autocorrelation per number's binary sequence. Under randomness, rho(1) ≈ 0.

## 8. Monte Carlo Simulation

### Simulation Strategies
1. **Uniform**: Equal probability 1/52 for each number
2. **Frequency-weighted**: Biased by historical frequency
3. **ML-ensemble**: Weighted by combined model scores

### Convergence Check

```
max_dev = max_i |count_i - expected_i| / expected_i
```

Convergence at max_dev < 0.05.

## 9. Ensemble Methodology

### Default Weights

| Module | Weight | Rationale |
|--------|--------|-----------|
| Frequency | 15% | Long-term tendency, stable |
| Hot/Cold | 20% | Recent signal, actionable |
| Gap | 15% | Mean reversion, complementary |
| Markov | 15% | Sequential dependency |
| Bayesian | 15% | Principled updating |
| Pattern | 10% | Structural features |
| Entropy | 5% | Regime detection |
| Simulation | 5% | Validation |

### Score Normalization

Each module's scores normalized to [0,1] before combining.

## Critical Disclaimer

All methods above are applied to a system designed to be random. The pool expansion from 50 to 52 further reduces any marginal predictive advantage from statistical analysis. The expected return on any lottery ticket remains negative regardless of the analytical approach used.

## Data Source

The analysis uses approximately 6 months of real Lotto Max historical draw data sourced from ca.lottonumbers.com. Users are welcome to:
- Replace the data in `assets/historical_draws.json` with data from any other source
- Modify the `data_fetcher.py` module to scrape from alternative websites
- Add draws manually using the validation functions
- The only requirement is that each draw has: `date`, `numbers` (7 integers, 1-52), and `bonus` (1 integer, 1-52)
