# Lotto Max ML Predictor

Advanced machine learning-powered Lotto Max lottery analysis, prediction, and simulation engine. Built with pure Python — no external ML libraries required.

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![No ML Dependencies](https://img.shields.io/badge/Dependencies-None-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

Lotto Max ML Predictor applies **9 independent analytical modules** to historical Canadian Lotto Max draw data, combining statistical modeling, probabilistic inference, and simulation to generate intelligent number recommendations with confidence scoring.

Every model is implemented from scratch in pure Python using only standard-library modules (`random`, `math`, `statistics`, `json`, `collections`, `datetime`, `itertools`, `argparse`). This ensures maximum compatibility — no pip installs, no virtual environments, no dependency conflicts.

---

## Features

### 9 Analysis Modules

| # | Module | Description |
|---|--------|-------------|
| 1 | **Frequency Analysis** | Comprehensive frequency stats with rolling windows, Z-scores, and chi-squared uniformity testing |
| 2 | **Hot/Cold Tracker** | Temperature scoring (-1.0 to +1.0) with mean reversion probability estimates |
| 3 | **Gap Analysis** | Tracks draws since last appearance, overdue detection, and next-appearance probability modeling |
| 4 | **Markov Chain** | First-order and second-order transition matrices for sequential dependency capture |
| 5 | **Bayesian Inference** | Beta prior/posterior updating with credible intervals and James-Stein shrinkage estimation |
| 6 | **Pattern Detection** | Consecutive pairs, range clustering, even/odd anomalies, sum-range analysis, and runs test for randomness |
| 7 | **Entropy Scoring** | Shannon entropy, rolling entropy averages, autocorrelation analysis, and randomness quality scoring |
| 8 | **Monte Carlo Simulation** | 10,000+ simulated draws with uniform, frequency-weighted, and ML-ensemble strategies |
| 9 | **Ensemble Prediction** | Weighted combination of all modules producing final recommendations with confidence levels |

### Key Capabilities

- **Smart Predictions** — Ensemble-weighted number recommendations with confidence scoring
- **Multiple Strategies** — Conservative, Aggressive, and Balanced alternative picks
- **Monte Carlo Simulation** — 10K+ draw simulations with convergence diagnostics
- **Strategy Comparison** — Side-by-side evaluation of 7 selection strategies
- **Data Management** — Built-in data validation, update (web scraping), and integrity checking
- **Zero Dependencies** — Pure Python standard library, works everywhere
- **AI-Agent Compatible** — Designed to work with 15+ AI coding agents (Claude Code, Gemini CLI, Codex CLI, etc.)

---

## Quick Start

### Run a Prediction

```bash
# Full prediction pipeline (all 9 modules)
python scripts/lotto_max_engine.py --mode predict

# On Windows (VS Code terminal)
py scripts\lotto_max_engine.py --mode predict
```

### Run Specific Analysis

```bash
python scripts/lotto_max_engine.py --mode frequency    # Frequency analysis
python scripts/lotto_max_engine.py --mode hotcold       # Hot/cold numbers
python scripts/lotto_max_engine.py --mode gap           # Overdue number detection
python scripts/lotto_max_engine.py --mode markov        # Markov chain predictions
python scripts/lotto_max_engine.py --mode bayesian      # Bayesian inference
python scripts/lotto_max_engine.py --mode pattern       # Pattern detection
python scripts/lotto_max_engine.py --mode entropy       # Entropy & randomness scoring
python scripts/lotto_max_engine.py --mode simulate      # Monte Carlo simulation
```

### Full Report

```bash
python scripts/lotto_max_engine.py --mode full-report
```

### Update Historical Data

```bash
# Fetch latest draws from ca.lottonumbers.com (requires: pip install requests)
python scripts/lotto_max_engine.py --mode update-data

# Check data status
python scripts/lotto_max_engine.py --mode data-info

# Validate data integrity
python scripts/lotto_max_engine.py --mode validate-data
```

---

## Example Output

```
================================================================
           LOTTO MAX ML PREDICTOR - NEXT DRAW
================================================================

  RECOMMENDED NUMBERS  07  14  23  31  38  42  47  72.3%

  Strategy: ML-Ensemble (Weighted)
  Pool: 1 to 52 (Updated May 2026)
  Expected Improvement vs Random: +8.7%

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
  DISCLOSURE: Lottery draws are random events. ML analysis
  identifies statistical tendencies but cannot guarantee
  outcomes. Play responsibly. For entertainment only.
================================================================
```

---

## Lotto Max Game Rules

Lotto Max is a Canadian national lottery:

| Parameter | Value |
|-----------|-------|
| Main Numbers | 7 from pool 1-52 |
| Draws | Twice weekly (Tuesday & Friday) |
| Jackpot | Starts at $10M, up to $80M (capped) |
| MaxMillions | Additional $1M prizes when jackpot > $50M |
| Odds (Jackpot) | 1 in 133,784,560 |
| Cost | $6 per play (3 sets of numbers) |

> **Note**: The pool expanded from 50 to 52 numbers in May 2026. This tool is calibrated for the current 1-52 pool.

---

## Project Structure

```
lotto-max-ml-predictor/
├── SKILL.md                          # Full skill instructions for AI agents
├── scripts/
│   ├── lotto_max_engine.py           # Main analysis engine (all 9 modules)
│   ├── monte_carlo_sim.py            # Monte Carlo simulation module
│   ├── ml_models.py                  # ML model implementations
│   ├── data_fetcher.py               # Historical data management & web scraping
│   └── utils.py                      # Utility functions & constants
├── references/
│   ├── lotto_max_rules.md            # Detailed game rules & prize structure
│   ├── methodology.md                # Statistical methodology deep-dive
│   └── model_documentation.md        # ML model technical documentation
└── assets/
    └── historical_draws.json         # Built-in historical draw dataset
```

---

## Ensemble Model Weights

The ML-Ensemble prediction combines all module outputs using these default weights:

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

Weights can be customized via the `--weights` flag.

---

## Advanced Usage

```bash
# Analyze last N draws only
python scripts/lotto_max_engine.py --mode predict --window 50

# Adjust simulation iterations
python scripts/lotto_max_engine.py --mode simulate --iterations 50000

# Generate multiple recommendation sets
python scripts/lotto_max_engine.py --mode predict --sets 5

# Compare all strategies
python scripts/lotto_max_engine.py --mode compare-strategies

# Generate wheeling system from top N numbers
python scripts/lotto_max_engine.py --mode wheeling --numbers 10

# Export chart data as JSON
python scripts/lotto_max_engine.py --mode export-charts
```

---

## Custom Data Sources

The engine ships with ~6 months of real historical draws sourced from [ca.lottonumbers.com](https://ca.lottonumbers.com/lotto-max/past-numbers). You can:

1. **Edit directly** — Modify `assets/historical_draws.json`
2. **Web update** — Run `--mode update-data` (requires `pip install requests`)
3. **Custom fetcher** — Implement your own data source in `scripts/data_fetcher.py`

The JSON format:
```json
{
  "last_updated": "2026-05-22",
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

---

## Probability Reference (Pool 1-52)

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

---

## Requirements

- **Python 3.8+**
- **No external dependencies** — all models use Python standard library only
- **Optional**: `requests` library for web-based data updates (`pip install requests`)

---

## AI Agent Compatibility

This project is designed to work seamlessly with AI coding agents. The `SKILL.md` file provides complete instructions that enable agents to understand and operate the tool. Compatible with:

- Claude Code (Anthropic)
- Gemini CLI (Google)
- Codex CLI (OpenAI)
- GitHub Copilot
- Cursor
- Aider
- Continue
- Roo Code
- Cline
- And more...

---

## Disclaimer

This tool is for **educational and entertainment purposes only**. Lottery draws are random events designed to be unpredictable. No statistical method can reliably predict lottery numbers. The ML models identify statistical tendencies, not certainties. Every combination has an equal probability of being drawn. Always play responsibly and within your means.

---

## Author

**Reza** — May 2026

If you are interested in developing or enhancing an AI automation system — whether for workflows, data processing, monitoring, or custom integrations — feel free to reach out to [flowaiautomationsupport@gmail.com](mailto:flowaiautomationsupport@gmail.com).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
