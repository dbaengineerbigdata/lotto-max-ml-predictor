---
name: lotto-max-ml-predictor
description: Advanced Lotto Max lottery analysis and prediction engine using machine learning (Markov chains, frequency analysis, hot/cold tracking, gap analysis, entropy scoring), Monte Carlo simulation for next draw, pattern detection, and statistical modeling. Use this skill whenever users mention Lotto Max, Canadian lottery, lottery prediction, number analysis, lottery simulation, lottery ML, lottery statistics, hot cold numbers, number frequency, jackpot analysis, or want intelligent number recommendations for any lottery draw. Also trigger when users ask about lottery odds, probability calculations, or pattern recognition in number sequences.


Created by: Reza Azizi – May 2026
If you are interested in developing or enhancing an AI automation system—whether for workflows,
data processing, monitoring, or custom integrations—please feel free to reach out to
flowaiautomationsupport@gmail.com. I would be happy to discuss your requirements, propose a tailored solution,
and explore how automation can improve efficiency, accuracy, and scalability for your environment.

---

# Lotto Max ML Predictor

Advanced machine learning-powered Lotto Max lottery analysis, prediction, and simulation engine. This skill provides deep statistical analysis of historical draw data, applies multiple ML models for pattern recognition, runs Monte Carlo simulations for next-draw probabilities, and generates intelligent number recommendations with confidence scoring.

## When to Use This Skill

Use this skill when users:
- Ask for Lotto Max number predictions or recommendations
- Want to analyze lottery number patterns, trends, or frequencies
- Request hot/cold number analysis for any lottery
- Need Monte Carlo simulation for next draw outcomes
- Ask about lottery odds, probabilities, or expected values
- Want to understand number clustering, streaks, or gap patterns
- Request a wheeling system or balanced number selection
- Ask for comparison between random vs. ML-informed number picks
- Want to simulate multiple draw scenarios
- Ask about jackpot size trends or prize tier distributions
- Need historical draw statistical summaries
- Want entropy-based randomness scoring of recent draws

## Lotto Max Game Rules

Lotto Max is a Canadian national lottery with the following structure:
- **Main Numbers**: Select 7 numbers from a pool of **1 to 52** (updated May 2026, previously 1-50)
- **Draws**: Twice weekly (Tuesday and Friday)
- **Jackpot**: Starts at $10 million, can grow up to $80 million (capped)
- **MaxMillions**: Additional $1 million prizes when jackpot exceeds $50 million
- **MaxPlus**: Additional guaranteed prize draws at higher jackpot levels
- **Odds of winning jackpot**: 1 in 133,784,560 (per single line)
- **Cost**: $6 per play (3 sets of numbers)

Understanding these rules is important because the ML models and simulations are calibrated to this exact number space (7 from 52). The pool expansion from 50 to 52 in May 2026 changes all probability calculations.

## Architecture Overview

The skill operates in a multi-layered analysis pipeline:

```
Historical Data → Statistical Layer → ML Layer → Simulation Layer → Output
     │                  │                │              │            │
     ├─ Frequency       ├─ Markov Chain  ├─ Monte Carlo ├─ Ranked
     ├─ Hot/Cold        ├─ Bayesian Inf  ├─ Bootstrap   │   Picks
     ├─ Gap Analysis    ├─ LSTM Pattern  ├─ Confidence  ├─ Visual
     ├─ Entropy         ├─ Clustering    │   Intervals  │   Report
     └─ Streaks         └─ Ensemble      └─ Scenarios   └─ Export
```

## Core Analysis Modules

### Module 1: Frequency Analysis Engine

Computes comprehensive frequency statistics across configurable time windows.

When triggered, execute the frequency analysis by running `scripts/lotto_max_engine.py` with the `--mode frequency` flag. This module calculates:
- Overall frequency of each number (1-52) across all historical draws
- Rolling window frequencies (last 10, 25, 50, 100 draws)
- Frequency deviation from expected uniform distribution
- Chi-squared goodness-of-fit test for uniformity
- Z-score for each number indicating statistical significance of deviation
- Trend direction (rising, falling, stable) per number

### Module 2: Hot/Cold Number Tracker

Identifies numbers that are statistically "hot" (appearing more than expected) or "cold" (appearing less than expected) based on recent draws.

Run with `--mode hotcold`. This module provides:
- Hot numbers: Top N numbers with highest recent frequency above expected
- Cold numbers: Top N numbers with lowest recent frequency below expected
- Warm numbers: Numbers near expected frequency
- Temperature score for each number (continuous scale from -1.0 to +1.0)
- Cross-validation between short-term and long-term temperature
- Mean reversion probability estimate for each extreme number

### Module 3: Gap Analysis

Tracks how many draws since each number last appeared, and analyzes gap distributions.

Run with `--mode gap`. This module delivers:
- Current gap (draws since last appearance) for each number
- Average historical gap per number
- Gap standard deviation and coefficient of variation
- Numbers currently at historically large gaps (overdue indicators)
- Gap probability model: P(number appears in next draw | current gap = g)
- Overdue score combining current gap against historical distribution

### Module 4: Markov Chain Predictor

Models state transitions between draws using first-order and second-order Markov chains.

Run with `--mode markov`. This module builds:
- First-order transition matrix: P(number j appears | number i appeared last draw)
- Second-order transition matrix: P(number k appears | numbers i,j appeared in previous draw)
- State space: each number's appearance/non-appearance as binary state
- Stationary distribution comparison with observed distribution
- Transition entropy as unpredictability measure
- Top transition pairs with highest conditional probabilities

### Module 5: Bayesian Inference Engine

Applies Bayesian updating to refine number probability estimates as new draws occur.

Run with `--mode bayesian`. This module computes:
- Prior distribution: uniform Beta(1,1) for each number
- Posterior distribution updated with observed frequencies
- Credible intervals for each number's true appearance probability
- Bayesian surprise score: how surprising was the last draw?
- Predictive posterior for next draw
- Shrinkage estimation toward global mean (James-Stein estimator)

### Module 6: Pattern Detection & Clustering

Detects non-random patterns and clusters in draw sequences.

Run with `--mode pattern`. This module identifies:
- Consecutive number pairs appearing together
- Number clustering by range (1-10, 11-20, 21-30, 31-40, 41-52)
- Even/odd ratio distribution and anomaly detection
- Sum range analysis with percentile bands
- Repeating number sequences across draws
- Temporal patterns (day-of-week effects, monthly trends)
- Runs test for randomness of each number's appearance sequence

### Module 7: Entropy & Randomness Scoring

Measures the information entropy of draws to detect deviations from randomness.

Run with `--mode entropy`. This module provides:
- Shannon entropy of each draw
- Rolling entropy average to detect entropy regime changes
- Kolmogorov-Smirnov test for uniformity of number selection
- Autocorrelation analysis at lags 1-10
- Randomness quality score (0-100) for recent draw window
- Comparison to theoretical maximum entropy for 7-from-52 selection

### Module 8: Monte Carlo Simulation Engine

Runs thousands of simulated draws using learned probability distributions to estimate next-draw likelihoods.

Run with `--mode simulate`. This module executes:
- 10,000+ simulated draws using weighted probability distributions
- Three simulation strategies:
  - **Uniform**: Pure random baseline (7 from 52, uniform)
  - **Frequency-weighted**: Biased by historical frequency analysis
  - **ML-ensemble**: Weighted by combined ML model outputs
- Convergence diagnostics for simulation stability
- Number occurrence count across all simulated draws
- Probability ranking of each number for the next draw
- Confidence intervals for each number's next-draw probability
- Expected value calculations for each number selection strategy
- Comparison table: simulated hit rate vs. theoretical random baseline

### Module 9: Ensemble Prediction & Final Recommendation

Combines all module outputs into a unified prediction with confidence scoring.

Run with `--mode predict`. This module produces:
- Ensemble score for each number (weighted average of all module outputs)
- Confidence level (Low / Medium / High / Very High) per number
- Final recommended pick: 7 numbers with highest ensemble scores
- Alternative picks: Conservative (high confidence), Aggressive (high potential), Balanced
- Expected hit rate improvement over random selection
- Full rationale for each recommended number
- Data source information (6 months of real draws from ca.lottonumbers.com)



## How to Use This Skill

### Quick Prediction (Default)

When a user simply asks for Lotto Max numbers, run the full pipeline:

```bash
python scripts/lotto_max_engine.py --mode predict
```

This executes all modules in sequence and produces the final recommendation. The predict mode automatically checks for the latest draw data from ca.lottonumbers.com before generating predictions. If the server is unreachable, it falls back to the bundled historical data seamlessly.

### Specific Analysis

Users can request specific analysis types:

```bash
# Frequency analysis only
python scripts/lotto_max_engine.py --mode frequency

# Hot/cold numbers
python scripts/lotto_max_engine.py --mode hotcold

# Gap analysis (overdue numbers)
python scripts/lotto_max_engine.py --mode gap

# Markov chain predictions
python scripts/lotto_max_engine.py --mode markov

# Bayesian inference
python scripts/lotto_max_engine.py --mode bayesian

# Pattern detection
python scripts/lotto_max_engine.py --mode pattern

# Entropy scoring
python scripts/lotto_max_engine.py --mode entropy

# Monte Carlo simulation
python scripts/lotto_max_engine.py --mode simulate --iterations 10000
```

### Full Report

For comprehensive analysis, generate a full report:

```bash
python scripts/lotto_max_engine.py --mode full-report
```

This runs all modules and produces a consolidated report with all findings.

### Custom Parameters

```bash
# Analyze last N draws only
python scripts/lotto_max_engine.py --mode predict --window 50

# Adjust simulation iterations
python scripts/lotto_max_engine.py --mode simulate --iterations 50000

# Generate multiple recommendation sets
python scripts/lotto_max_engine.py --mode predict --sets 5

# Export results as JSON
python scripts/lotto_max_engine.py --mode predict --format json

# Compare strategies
python scripts/lotto_max_engine.py --mode compare-strategies
```

## Output Format

### Prediction Output

Present predictions using this exact format:

```
================================================================
           LOTTO MAX ML PREDICTOR - NEXT DRAW
================================================================

  RECOMMENDED NUMBERS  07  14  23  31  38  42  47  72.3%

  Strategy: ML-Ensemble (Weighted)
  Pool: 1 to 52 (Updated May 2026)
  Expected Improvement vs Random: +8.7%

----------------------------------------------------------------
  DATA SOURCE & METHODOLOGY:
  - Historical data: 51 draws from 2025-11-28 to 2026-05-22
  - Source: ca.lottonumbers.com/lotto-max/past-numbers
  - Users are welcome to use other data sources
  - Modify assets/historical_draws.json or --mode update-data

----------------------------------------------------------------
  ALTERNATIVE SETS

  Conservative:  03  12  22  29  36  41  48  81.2%
  Aggressive:    05  11  19  27  34  39  46  58.4%
  Balanced:      04  15  24  30  37  43  49  70.1%

----------------------------------------------------------------
  TOP NUMBER RATIONALE:

  07 - Hot streak (4 appearances in last 10 draws)
  14 - Overdue (gap: 12 draws, 95th percentile)
  23 - Markov transition favorite from last draw numbers
  31 - Bayesian posterior above prior (p=0.162 vs 0.14)
  38 - Frequency leader in 50-draw window
  42 - Cluster analysis: pairs well with 38 and 47
  47 - High ensemble score, consistent across all models

----------------------------------------------------------------
  SIMULATION SUMMARY (10,000 draws):
  - Most simulated number: 38 (appeared in 1,687 sims)
  - Least simulated number: 02 (appeared in 1,324 sims)
  - Convergence: YES (variance < threshold)
  - Strategy edge: +8.7% more 3+ matches vs random

----------------------------------------------------------------
  DISCLOSURE: Lottery draws are random events. ML analysis
  identifies statistical tendencies but cannot guarantee
  outcomes. Play responsibly. For entertainment only.
================================================================
```

### Frequency Analysis Output

```
📊 FREQUENCY ANALYSIS - Lotto Max (Last 100 Draws)

Number | Total | Last10 | Last25 | Last50 | Z-Score | Trend
-------|-------|--------|--------|--------|---------|------
  01   |  14   |   2    |   4    |   7    |  -0.31  |  →
  02   |  11   |   0    |   2    |   5    |  -1.24  |  ↓
  ...
  38   |  22   |   4    |   6    |  11   |  +2.18  |  ↑
  ...

Chi-squared: 48.7 (df=49, p=0.483) - Distribution is NOT significantly non-uniform
```

### Simulation Output

```
🎲 MONTE CARLO SIMULATION - 10,000 Draws

Rank | Number | Occurrences | Probability | vs Uniform
-----|--------|-------------|-------------|------------
  1  |   38   |    1687     |   16.87%    |  +2.47%
  2  |   07   |    1654     |   16.54%    |  +2.14%
  ...
 50  |   02   |    1324     |   13.24%    |  -1.16%

Simulation convergence: ✓ (variance < 0.001 across last 2000 iterations)
```

## Advanced Features

### Strategy Comparison Mode

When users want to compare different selection strategies:

```bash
python scripts/lotto_max_engine.py --mode compare-strategies
```

This runs all strategies in parallel and produces a side-by-side comparison:
- Pure Random (baseline)
- Frequency-Weighted
- Hot/Cold Hybrid
- Markov Chain
- Bayesian Posterior
- Gap-Based (Overdue)
- ML-Ensemble (recommended)

Each strategy is evaluated on: average numbers matched per simulated draw, 3+ match rate, 4+ match rate, and consistency score.

### Wheeling System Generator

For users who want systematic coverage:

```bash
python scripts/lotto_max_engine.py --mode wheeling --numbers 10
```

Generates an abbreviated wheeling system from the top N recommended numbers, providing balanced coverage across all 7-number combinations while minimizing the number of lines played.

### Trend Visualization Data

Export data suitable for charting:

```bash
python scripts/lotto_max_engine.py --mode export-charts
```

Produces JSON with frequency histograms, gap distributions, rolling averages, and trend lines that can be rendered as charts.

### Data Management

The skill ships with pre-loaded historical data (~6 months of real draws, sourced from ca.lottonumbers.com). No manual data entry is needed. Users are welcome to use other data sources by modifying `assets/historical_draws.json` or implementing their own data fetcher. Users can verify data integrity and update to get the latest draws:

```bash
# Check data info (source, date range, pool compatibility, top numbers)
python scripts/lotto_max_engine.py --mode data-info

# Validate all historical draws (checks number range, format, duplicates)
python scripts/lotto_max_engine.py --mode validate-data

# Update historical data (fetches latest draws from ca.lottonumbers.com)
python scripts/lotto_max_engine.py --mode update-data
```

The `validate-data` mode confirms all draws have valid numbers (1-52), correct format (7 unique numbers + bonus), and detects if the data is using the legacy 1-50 pool instead of the current 1-52 pool.

The `update-data` mode requires the `requests` library (`pip install requests`) and an internet connection. If the fetch fails, the skill continues using the bundled historical data.

## Probability Reference

Key probabilities for Lotto Max that the skill uses (updated for pool 1-52):

| Match | Probability | Odds (1 in) |
|-------|-------------|-------------|
| 7/7 (Jackpot) | 0.00000075% | 133,784,560 |
| 6/7 + Bonus | 0.0000046% | 21,897,368 |
| 6/7 | 0.000032% | 3,128,195 |
| 5/7 | 0.00092% | 108,891 |
| 4/7 | 0.014% | 7,188 |
| 3/7 + Bonus | 0.017% | 5,915 |
| 3/7 | 0.18% | 554 |
| Any Prize | ~0.60% | ~167 |

These baseline probabilities are used to contextualize all ML predictions and simulation results.

## Model Weights (Ensemble)

The ML-Ensemble prediction combines module outputs using these default weights:

| Module | Weight | Rationale |
|--------|--------|-----------|
| Frequency Analysis | 15% | Long-term tendency indicator |
| Hot/Cold Tracker | 20% | Recent momentum signal |
| Gap Analysis | 15% | Mean reversion probability |
| Markov Chain | 15% | Sequential dependency capture |
| Bayesian Inference | 15% | Principled probability updating |
| Pattern Detection | 10% | Structural pattern exploitation |
| Entropy Scoring | 5% | Randomness regime adjustment |
| Monte Carlo | 5% | Simulation-based validation |

Weights sum to 100% and can be customized via `--weights` flag.

## Important Disclaimers

This skill must always include these disclaimers in its output:

1. **Randomness**: Lottery draws are designed to be random. No statistical method can reliably predict lottery numbers.
2. **Entertainment Only**: This tool is for educational and entertainment purposes. It should not be used as the basis for financial decisions.
3. **No Guarantee**: Past patterns do not guarantee future results. The ML models identify statistical tendencies, not certainties.
4. **Responsible Play**: Always play within your means. The expected value of lottery tickets is negative.
5. **Equal Odds**: Every combination has an equal probability of being drawn. The ML models may suggest numbers with interesting statistical profiles, but the probability of any specific combination being drawn remains the same.

## File Structure

```
lotto-max-ml-predictor/
├── SKILL.md                          # This file - full skill instructions
├── scripts/
│   ├── lotto_max_engine.py           # Main analysis engine
│   ├── monte_carlo_sim.py            # Monte Carlo simulation module
│   ├── ml_models.py                  # ML model implementations
│   ├── data_fetcher.py               # Historical data management
│   └── utils.py                      # Utility functions
├── references/
│   ├── lotto_max_rules.md            # Detailed game rules and prize structure
│   ├── methodology.md                # Statistical methodology deep-dive
│   └── model_documentation.md        # ML model technical documentation
└── assets/
    └── historical_draws.json         # Built-in historical draw dataset
```

## Implementation Notes

### Execution Environment

The Python scripts require Python 3.8+ with the following standard-library-only approach:
- `random`, `math`, `statistics`, `json`, `collections`, `datetime`, `itertools`, `argparse`
- No external ML libraries required (all models implemented from scratch)
- This ensures maximum compatibility across all agent environments

### Data Persistence

Historical draw data is stored in `assets/historical_draws.json` with this structure. Data covers approximately 6 months of real draws sourced from [ca.lottonumbers.com](https://ca.lottonumbers.com/lotto-max/past-numbers). Users are welcome to use other data sources by modifying the JSON file directly or implementing a custom data fetcher. Use `--mode data-info` to see current data status. Use `--mode validate-data` to verify data integrity. Use `--mode update-data` to fetch the latest draws (requires `requests` library and internet):
```json
{
  "last_updated": "2026-05-26",
  "source": "https://ca.lottonumbers.com/lotto-max/past-numbers",
  "total_draws": 51,
  "draws": [
    {
      "date": "2026-05-22",
      "numbers": [11, 22, 31, 36, 42, 47, 48],
      "bonus": 44
    }
  ]
}
```

### Performance Considerations

- Frequency analysis and gap analysis: O(n) where n = number of draws
- Markov chain construction: O(n * k) where k = numbers per draw
- Monte Carlo simulation: O(iterations * 7) - 10K iterations complete in < 2 seconds
- Full pipeline execution: typically under 5 seconds for 500+ historical draws

## Examples

**Example 1:**
Input: "Give me Lotto Max numbers for the next draw"
Output: Runs `--mode predict` and presents the full prediction box with recommended numbers, alternatives, rationale, and simulation summary.

**Example 2:**
Input: "What are the hot and cold numbers in Lotto Max?"
Output: Runs `--mode hotcold` and shows temperature scores, top 5 hot and cold numbers, and mean reversion probabilities.

**Example 3:**
Input: "Run a simulation for the next Lotto Max draw with 50,000 iterations"
Output: Runs `--mode simulate --iterations 50000` and presents occurrence rankings, convergence check, and strategy comparison.

**Example 4:**
Input: "Which Lotto Max numbers are overdue?"
Output: Runs `--mode gap` and shows numbers with the largest current gaps, their historical gap distributions, and overdue probability scores.

**Example 5:**
Input: "Compare all prediction strategies for Lotto Max"
Output: Runs `--mode compare-strategies` and presents a table comparing 7 strategies on multiple metrics.

**Example 6:**
Input: "Is there any pattern in the last 20 Lotto Max draws?"
Output: Runs `--mode pattern --window 20` and shows detected patterns: consecutive pairs, cluster distributions, even/odd anomalies, and randomness test results.
