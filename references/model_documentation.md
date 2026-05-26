# ML Model Technical Documentation

## Overview

This document provides technical specifications for all machine learning models implemented in the Lotto Max ML Predictor. Updated for the current pool of **1 to 52** numbers (May 2026).

## Model Architecture

### 1. MarkovChainModel

**File**: `scripts/ml_models.py`
**Class**: `MarkovChainModel`

#### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| order | int | 1 | Markov chain order (1 or 2) |

#### Methods
- `fit(draws)`: Train model on historical draws
- `predict(last_draw, top_k=15)`: Predict likely next numbers
- `get_stationary_distribution()`: Compute stationary distribution via power iteration

#### Implementation Details
- **First-order**: Models P(next_num | prev_num) for each number in previous draw
- **Second-order**: Models P(next_num | prev_pair) for all pairs in previous draw
- **Stationary distribution**: Computed via 100 iterations of power iteration
- **State space**: 52 numbers, binary appearance per draw

#### Complexity
- Training: O(n × k²) for first-order, O(n × C(k,2) × k) for second-order
- Prediction: O(k × k) for first-order

### 2. BayesianModel

**File**: `scripts/ml_models.py`
**Class**: `BayesianModel`

#### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| alpha_prior | float | 1.0 | Beta distribution alpha parameter |
| beta_prior | float | 1.0 | Beta distribution beta parameter |

#### Methods
- `fit(draws)`: Update priors with observed data
- `predict(top_k=15)`: Get top numbers by posterior mean
- `credible_interval(num, level=0.95)`: Compute credible interval
- `bayesian_surprise(draw)`: Compute KL divergence from prior
- `james_stein_estimate()`: Apply James-Stein shrinkage

#### Implementation Details
- **Prior**: Beta(1, 1) for each of 52 numbers
- **Posterior**: Beta(1 + appearances, 1 + non_appearances)
- **Global mean**: 7/52 ≈ 0.13462

### 3. GapProbabilityModel

**File**: `scripts/ml_models.py`
**Class**: `GapProbabilityModel`

#### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| decay_rate | float | 1.0 | Exponential decay rate for gap probability |

#### Methods
- `fit(draws)`: Compute gap statistics from historical data
- `predict(top_k=15)`: Predict numbers based on gap probability

#### Implementation Details
- Average gap under uniformity: 52/7 ≈ 7.43 draws

### 4. EnsembleModel

**File**: `scripts/ml_models.py`
**Class**: `EnsembleModel`

#### Default Weights
```python
{
    'frequency': 0.15,
    'hotcold': 0.20,
    'gap': 0.15,
    'markov': 0.15,
    'bayesian': 0.15,
    'pattern': 0.10,
    'entropy': 0.05,
    'simulation': 0.05
}
```

#### Methods
- `set_weights(weights)`: Customize model weights
- `add_model(name, model, scores)`: Add a trained model
- `predict(top_k=7)`: Generate ensemble prediction

## Monte Carlo Simulator

**File**: `scripts/monte_carlo_sim.py`
**Class**: `MonteCarloSimulator`

### Supported Strategies
1. **uniform**: Equal probability (1/52 per number)
2. **frequency**: Weighted by historical frequency
3. **hot-cold**: Weighted by temperature scores
4. **gap-based**: Weighted by gap probability
5. **ml-ensemble**: Weighted by combined ML scores

## Data Format

### Historical Draws JSON Schema
```json
{
  "last_updated": "2026-05-20",
  "source": "https://ca.lottonumbers.com/lotto-max/past-numbers",
  "total_draws": 52,
  "draws": [
    {
      "date": "2026-05-19",
      "numbers": [5, 15, 22, 24, 34, 37, 49],
      "bonus": 2
    }
  ]
}
```

## Updated Probability Reference (Pool 1-52)

| Match | Probability | Odds (1 in) |
|-------|-------------|-------------|
| 7/7 (Jackpot) | 7.5×10⁻⁹ | 133,784,560 |
| 6/7 + Bonus | 4.6×10⁻⁸ | 21,897,368 |
| 6/7 | 3.2×10⁻⁷ | 3,128,195 |
| 5/7 | 9.2×10⁻⁶ | 108,891 |
| 4/7 | 1.4×10⁻⁴ | 7,188 |
| 3/7 + Bonus | 1.7×10⁻⁴ | 5,915 |
| 3/7 | 1.8×10⁻³ | 554 |

## Dependencies

**Python Standard Library Only** (no external packages required):
- `random`, `math`, `statistics`, `json`, `collections`, `itertools`, `datetime`, `argparse`, `os`, `sys`

**Minimum Python Version**: 3.8+

## Performance Benchmarks

With 52 real draws from ca.lottonumbers.com:

| Module | ~Time |
|--------|-------|
| Frequency | <0.01s |
| Hot/Cold | <0.01s |
| Gap | <0.01s |
| Markov (1st) | <0.02s |
| Bayesian | <0.01s |
| Pattern | <0.01s |
| Entropy | <0.01s |
| Simulation (10K) | <1.0s |
| Full Pipeline | <2.0s |
