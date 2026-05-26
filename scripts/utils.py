#!/usr/bin/env python3
"""
Utility Functions for Lotto Max ML Predictor
Common helpers, constants, and formatting utilities.

Created by: Reza Azizi - May 2026
If you are interested in developing or enhancing an AI automation system—whether for workflows,
data processing, monitoring, or custom integrations—please feel free to reach out to
flowaiautomationsupport@gmail.com. I would be happy to discuss your requirements, propose a tailored solution,
and explore how automation can improve efficiency, accuracy, and scalability for your environment.

"""

import math
from collections import Counter

# ─── Lotto Max Constants ─────────────────────────────────────────────────────
NUM_POOL = 52
NUMS_PER_DRAW = 7
TOTAL_COMBINATIONS = math.comb(52, 7)  # 99,884,400

# Prize tiers and their probabilities
PRIZE_TIERS = {
    '7/7 (Jackpot)': {'match': 7, 'probability': 1 / 133784560, 'odds': 133784560},
    '6/7 + Bonus': {'match': '6+b', 'probability': 1 / 21897368, 'odds': 21897368},
    '6/7': {'match': 6, 'probability': 1 / 3128195, 'odds': 3128195},
    '5/7': {'match': 5, 'probability': 1 / 108891, 'odds': 108891},
    '4/7': {'match': 4, 'probability': 1 / 7188, 'odds': 7188},
    '3/7 + Bonus': {'match': '3+b', 'probability': 1 / 5915, 'odds': 5915},
    '3/7': {'match': 3, 'probability': 1 / 554, 'odds': 554},
}

# Agent compatibility paths for skillhub.club
AGENT_PATHS = {
    'Claude Code': '~/.claude/skills/',
    'Codex CLI': '~/.codex/skills/',
    'Gemini CLI': '~/.gemini/skills/',
    'OpenCode': '~/.opencode/skills/',
    'OpenClaw': '~/.openclaw/skills/',
    'GitHub Copilot': '~/.copilot/skills/',
    'Cursor': '~/.cursor/skills/',
    'Windsurf': '~/.codeium/windsurf/skills/',
    'Cline': '~/.cline/skills/',
    'Roo Code': '~/.roo/skills/',
    'Kiro': '~/.kiro/skills/',
    'Junie': '~/.junie/skills/',
    'Augment Code': '~/.augment/skills/',
    'Warp': '~/.warp/skills/',
    'Goose': '~/.config/goose/skills/',
}


def format_number(num, width=2):
    """Format a number with leading zeros."""
    return f'{num:0{width}d}'


def format_percentage(value, decimals=2):
    """Format a value as percentage string."""
    return f'{value * 100:.{decimals}f}%'


def format_odds(numerator, denominator=1):
    """Format odds in a readable format."""
    if denominator == 0:
        return "N/A"
    ratio = numerator / denominator
    if ratio >= 1_000_000:
        return f"1 in {ratio:,.0f}"
    elif ratio >= 1000:
        return f"1 in {ratio:,.0f}"
    else:
        return f"1 in {ratio:.1f}"


def calculate_match_probability(matches, total_numbers=7, pool_size=52):
    """Calculate the probability of matching exactly N numbers."""
    from math import comb
    # Hypergeometric distribution
    # P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
    # where K = total_numbers (successes in population), N = pool_size, n = numbers drawn
    K = total_numbers
    N = pool_size
    n = total_numbers
    k = matches

    if k > K or k > n:
        return 0

    return comb(K, k) * comb(N - K, n - k) / comb(N, n)


def calculate_expected_value(ticket_cost=6, jackpot=10000000):
    """Calculate expected value of a Lotto Max ticket."""
    expected_prize = 0

    prize_amounts = {
        '7/7 (Jackpot)': jackpot,
        '6/7 + Bonus': min(jackpot * 0.05, 250000),
        '6/7': min(jackpot * 0.02, 100000),
        '5/7': 500,
        '4/7': 50,
        '3/7 + Bonus': 20,
        '3/7': 6,
    }

    for tier, info in PRIZE_TIERS.items():
        if tier in prize_amounts:
            expected_prize += info['probability'] * prize_amounts[tier]

    return {
        'ticket_cost': ticket_cost,
        'expected_prize': round(expected_prize, 4),
        'expected_value': round(expected_prize - ticket_cost, 4),
        'ev_ratio': round(expected_prize / ticket_cost, 4),
        'jackpot_breakpoint': round(ticket_cost / PRIZE_TIERS['7/7 (Jackpot)']['probability']),
    }


def compute_entropy(numbers_list, pool_size=52):
    """Compute Shannon entropy of a number selection."""
    if not numbers_list:
        return 0

    p_appear = len(numbers_list) / pool_size
    p_not = 1 - p_appear

    entropy = 0
    if p_appear > 0:
        entropy -= p_appear * math.log2(p_appear)
    if p_not > 0:
        entropy -= p_not * math.log2(p_not)

    return entropy * pool_size


def normalize_scores(scores):
    """Normalize a dictionary of scores to [0, 1] range."""
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    range_score = max_score - min_score

    if range_score == 0:
        return {k: 0.5 for k in scores}

    return {k: (v - min_score) / range_score for k, v in scores.items()}


def weighted_choice(weights, available=None):
    """Make a weighted random choice."""
    import random

    if available is None:
        available = list(range(1, len(weights) + 1))

    total = sum(weights[i] for i in range(min(len(weights), len(available))))
    if total == 0:
        return random.choice(available)

    r = random.random() * total
    cumulative = 0
    for i, item in enumerate(available):
        if i < len(weights):
            cumulative += weights[i]
        if r <= cumulative:
            return item

    return available[-1]


def create_number_report(number, freq_data, hotcold_data, gap_data, bayesian_data):
    """Create a detailed report for a single number."""
    report = {
        'number': number,
        'frequency': freq_data.get('numbers', {}).get(number, {}),
        'temperature': hotcold_data.get('all_temperatures', {}).get(number, {}),
        'gap': gap_data.get(number, {}),
        'bayesian': bayesian_data.get('numbers', {}).get(number, {}),
    }

    # Overall assessment
    signals = []
    if report['temperature'].get('temperature', 0) > 0.2:
        signals.append('HOT')
    elif report['temperature'].get('temperature', 0) < -0.2:
        signals.append('COLD')

    if report['gap'].get('is_overdue', False):
        signals.append('OVERDUE')

    if report['bayesian'].get('above_prior', False):
        signals.append('ABOVE_PRIOR')

    freq_trend = report['frequency'].get('trend', 'stable')
    if freq_trend == 'rising':
        signals.append('RISING')
    elif freq_trend == 'falling':
        signals.append('FALLING')

    report['signals'] = signals
    report['assessment'] = 'FAVORABLE' if len(signals) >= 2 else 'NEUTRAL' if len(signals) == 1 else 'UNFAVORABLE'

    return report


def box_print(text, width=64, style='='):
    """Print text in a box format."""
    lines = text.split('\n')
    print(style * width)
    for line in lines:
        print(line)
    print(style * width)


def progress_bar(current, total, width=40):
    """Display a progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = '#' * filled + '-' * (width - filled)
    return f'[{bar}] {pct * 100:.1f}%'


if __name__ == '__main__':
    print("Utility Functions - Lotto Max ML Predictor")
    print("=" * 50)

    # Test probability calculations
    print("\nPrize Tier Probabilities:")
    for tier, info in PRIZE_TIERS.items():
        print(f"  {tier}: {format_odds(info['odds'])} ({info['probability']*100:.6f}%)")

    # Test expected value
    ev = calculate_expected_value(6, 50_000_000)
    print(f"\nExpected Value ($50M jackpot):")
    print(f"  Ticket cost: ${ev['ticket_cost']}")
    print(f"  Expected prize: ${ev['expected_prize']:.4f}")
    print(f"  Expected value: ${ev['expected_value']:.4f}")
    print(f"  EV ratio: {ev['ev_ratio']:.4f}")

    # Test agent paths
    print(f"\nCompatible Agents ({len(AGENT_PATHS)}):")
    for agent, path in AGENT_PATHS.items():
        print(f"  {agent}: {path}")
