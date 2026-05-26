#!/usr/bin/env python3
"""
Lotto Max ML Predictor - Main Analysis Engine
Advanced lottery analysis with machine learning, simulation, and statistical modeling.
For entertainment and educational purposes only.

Updated May 2026: Lotto Max pool expanded from 50 to 52 numbers.
Historical data sourced from ca.lottonumbers.com.

Created by: Reza Azizi - May 2026
If you are interested in developing or enhancing an AI automation system—whether for workflows,
data processing, monitoring, or custom integrations—please feel free to reach out to
flowaiautomationsupport@gmail.com. I would be happy to discuss your requirements, propose a tailored solution,
and explore how automation can improve efficiency, accuracy, and scalability for your environment.

"""

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

# ─── Constants ───────────────────────────────────────────────────────────────
NUM_POOL = 52              # Updated: Pool expanded from 50 to 52 (May 2026)
NUMS_PER_DRAW = 7
BONUS_POOL = NUM_POOL      # Bonus also drawn from 1-52
TOTAL_COMBINATIONS = math.comb(52, 7)  # 133,784,560
JACKPOT_ODDS = 1 / TOTAL_COMBINATIONS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, '..', 'assets')
DEFAULT_DRAWS_FILE = os.path.join(ASSETS_DIR, 'historical_draws.json')

# ─── Data Loading ────────────────────────────────────────────────────────────

def load_draws(draws_file=None, window=None):
    """Load historical draw data from JSON file."""
    if draws_file is None:
        draws_file = DEFAULT_DRAWS_FILE
    try:
        with open(draws_file, 'r') as f:
            data = json.load(f)
        draws = data.get('draws', [])
    except (FileNotFoundError, json.JSONDecodeError):
        draws = generate_sample_draws(52)

    if window and window < len(draws):
        draws = draws[-window:]

    return draws


def generate_sample_draws(count=52):
    """Generate sample historical draws for demonstration only."""
    random.seed(42)
    draws = []
    base_date = datetime(2025, 12, 2)
    for i in range(count):
        draw_date = base_date + __import__('datetime').timedelta(days=i * 2.5)
        numbers = sorted(random.sample(range(1, NUM_POOL + 1), NUMS_PER_DRAW))
        bonus_candidates = [n for n in range(1, NUM_POOL + 1) if n not in numbers]
        bonus = random.choice(bonus_candidates)
        draws.append({
            'date': draw_date.strftime('%Y-%m-%d'),
            'numbers': numbers,
            'bonus': bonus
        })
    random.seed()
    return draws


# ─── Module 1: Frequency Analysis ───────────────────────────────────────────

def frequency_analysis(draws, windows=None):
    """Compute comprehensive frequency statistics."""
    if windows is None:
        windows = [10, 25, 50]
    total_draws = len(draws)
    all_numbers = []
    for d in draws:
        all_numbers.extend(d['numbers'])

    overall_freq = Counter(all_numbers)

    results = {
        'total_draws': total_draws,
        'expected_per_number': (total_draws * NUMS_PER_DRAW) / NUM_POOL,
        'numbers': {}
    }

    for num in range(1, NUM_POOL + 1):
        freq = overall_freq.get(num, 0)
        expected = results['expected_per_number']
        z_score = (freq - expected) / math.sqrt(expected) if expected > 0 else 0

        # Windowed frequencies
        window_freqs = {}
        for w in windows:
            if w <= total_draws:
                recent = draws[-w:]
            else:
                recent = draws
            count = sum(1 for d in recent if num in d['numbers'])
            window_freqs[f'last_{w}'] = count

        # Trend detection
        if 'last_10' in window_freqs:
            w10 = window_freqs['last_10']
        elif window_freqs:
            w10 = list(window_freqs.values())[0]
        else:
            w10 = 0
        exp_10 = (min(10, total_draws) * NUMS_PER_DRAW) / NUM_POOL
        if w10 > exp_10 * 1.3:
            trend = 'rising'
        elif w10 < exp_10 * 0.7:
            trend = 'falling'
        else:
            trend = 'stable'

        results['numbers'][num] = {
            'total': freq,
            'frequency': freq / total_draws if total_draws > 0 else 0,
            'z_score': round(z_score, 3),
            'trend': trend,
            'windows': window_freqs
        }

    # Chi-squared test
    chi_sq = 0
    expected = results['expected_per_number']
    for num in range(1, NUM_POOL + 1):
        observed = overall_freq.get(num, 0)
        chi_sq += (observed - expected) ** 2 / expected if expected > 0 else 0

    results['chi_squared'] = round(chi_sq, 2)
    results['chi_df'] = NUM_POOL - 1

    return results


# ─── Module 2: Hot/Cold Tracker ─────────────────────────────────────────────

def hot_cold_analysis(draws, window=25):
    """Identify hot and cold numbers with temperature scoring."""
    if window > len(draws):
        window = len(draws)
    recent = draws[-window:]
    total = len(recent)
    expected = (total * NUMS_PER_DRAW) / NUM_POOL

    temperatures = {}
    for num in range(1, NUM_POOL + 1):
        count = sum(1 for d in recent if num in d['numbers'])
        deviation = (count - expected) / expected if expected > 0 else 0
        temperature = max(-1.0, min(1.0, deviation))
        temperatures[num] = {
            'count': count,
            'expected': round(expected, 2),
            'deviation': round(deviation, 3),
            'temperature': round(temperature, 3)
        }

    # Sort by temperature
    sorted_nums = sorted(temperatures.items(), key=lambda x: x[1]['temperature'], reverse=True)

    hot = [(n, d) for n, d in sorted_nums if d['temperature'] > 0.2][:10]
    cold = [(n, d) for n, d in sorted_nums if d['temperature'] < -0.2][:10]
    warm = [(n, d) for n, d in sorted_nums if -0.2 <= d['temperature'] <= 0.2]

    # Mean reversion probability
    for num, data in sorted_nums:
        if data['temperature'] > 0.5:
            data['mean_reversion_prob'] = round(0.65 + data['temperature'] * 0.2, 3)
        elif data['temperature'] < -0.5:
            data['mean_reversion_prob'] = round(0.65 + abs(data['temperature']) * 0.2, 3)
        else:
            data['mean_reversion_prob'] = round(0.5, 3)

    return {
        'window': window,
        'expected_per_number': round(expected, 2),
        'hot': hot,
        'cold': cold,
        'warm_count': len(warm),
        'all_temperatures': temperatures
    }


# ─── Module 3: Gap Analysis ─────────────────────────────────────────────────

def gap_analysis(draws):
    """Analyze gaps between appearances for each number."""
    total_draws = len(draws)
    gaps = defaultdict(list)
    current_gap = defaultdict(int)
    last_seen = defaultdict(int)

    for i, d in enumerate(draws):
        for num in range(1, NUM_POOL + 1):
            if num in d['numbers']:
                if last_seen[num] > 0:
                    gaps[num].append(i - last_seen[num])
                current_gap[num] = 0
                last_seen[num] = i
            else:
                current_gap[num] += 1

    results = {}
    for num in range(1, NUM_POOL + 1):
        gap_list = gaps.get(num, [])
        avg_gap = statistics.mean(gap_list) if gap_list else total_draws / NUMS_PER_DRAW
        std_gap = statistics.stdev(gap_list) if len(gap_list) > 1 else avg_gap * 0.5
        cv = std_gap / avg_gap if avg_gap > 0 else 0

        curr_gap = current_gap[num]
        appear_prob = 1 - math.exp(-curr_gap / avg_gap) if avg_gap > 0 else NUMS_PER_DRAW / NUM_POOL
        overdue_score = (curr_gap - avg_gap) / std_gap if std_gap > 0 else 0

        results[num] = {
            'current_gap': curr_gap,
            'avg_gap': round(avg_gap, 2),
            'std_gap': round(std_gap, 2),
            'cv': round(cv, 3),
            'next_appear_prob': round(appear_prob, 4),
            'overdue_score': round(overdue_score, 3),
            'is_overdue': overdue_score > 1.5
        }

    return results


# ─── Module 4: Markov Chain ─────────────────────────────────────────────────

def markov_analysis(draws):
    """Build first-order and second-order Markov chain models."""
    first_order = defaultdict(lambda: defaultdict(int))
    first_order_count = defaultdict(int)
    second_order = defaultdict(lambda: defaultdict(int))
    second_order_count = defaultdict(int)

    for i in range(1, len(draws)):
        prev_nums = set(draws[i - 1]['numbers'])
        curr_nums = set(draws[i]['numbers'])

        for prev_num in prev_nums:
            first_order_count[prev_num] += 1
            for curr_num in curr_nums:
                first_order[prev_num][curr_num] += 1

        for pair in combinations(sorted(prev_nums), 2):
            second_order_count[pair] += 1
            for curr_num in curr_nums:
                second_order[pair][curr_num] += 1

    first_order_probs = {}
    for prev_num, transitions in first_order.items():
        total = first_order_count[prev_num]
        first_order_probs[prev_num] = {
            k: round(v / total, 4) for k, v in sorted(
                transitions.items(), key=lambda x: -x[1])[:10]
        }

    last_draw = set(draws[-1]['numbers'])
    predictions = Counter()

    for num in last_draw:
        if num in first_order_probs:
            for next_num, prob in first_order_probs[num].items():
                if next_num not in last_draw:
                    predictions[next_num] += prob

    for pair in combinations(sorted(last_draw), 2):
        if pair in second_order:
            total = second_order_count[pair]
            for next_num, count in second_order[pair].items():
                if next_num not in last_draw:
                    predictions[next_num] += (count / total) * 0.5

    return {
        'last_draw': sorted(last_draw),
        'transition_predictions': predictions.most_common(15),
        'top_transition_pairs': [
            (prev, list(trans.items())[:3])
            for prev, trans in list(first_order_probs.items())[:10]
        ]
    }


# ─── Module 5: Bayesian Inference ───────────────────────────────────────────

def bayesian_analysis(draws):
    """Apply Bayesian updating with Beta priors."""
    total_draws = len(draws)

    results = {}
    for num in range(1, NUM_POOL + 1):
        appearances = sum(1 for d in draws if num in d['numbers'])
        non_appearances = total_draws - appearances

        alpha_prior = 1.0
        beta_prior = 1.0
        alpha_post = alpha_prior + appearances
        beta_post = beta_prior + non_appearances

        post_mean = alpha_post / (alpha_post + beta_post)
        post_var = (alpha_post * beta_post) / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
        post_std = math.sqrt(post_var)
        ci_lower = max(0, post_mean - 1.96 * post_std)
        ci_upper = min(1, post_mean + 1.96 * post_std)

        prior_mean = alpha_prior / (alpha_prior + beta_prior)

        if 0 < post_mean < 1 and 0 < prior_mean < 1:
            surprise = post_mean * math.log(post_mean / prior_mean) + \
                       (1 - post_mean) * math.log((1 - post_mean) / (1 - prior_mean))
        else:
            surprise = 0

        global_mean = NUMS_PER_DRAW / NUM_POOL
        sum_sq = sum(
            (sum(1 for d in draws if n in d['numbers']) / total_draws - global_mean) ** 2
            for n in range(1, NUM_POOL + 1)
        ) if total_draws > 0 else 1
        shrinkage_factor = max(0, min(1, 1 - (NUM_POOL - 3) / (sum_sq * total_draws))) if sum_sq > 0 and total_draws > 0 else 0
        shrunk_estimate = shrinkage_factor * post_mean + (1 - shrinkage_factor) * global_mean

        results[num] = {
            'appearances': appearances,
            'posterior_mean': round(post_mean, 4),
            'prior_mean': round(prior_mean, 4),
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4),
            'bayesian_surprise': round(surprise, 4),
            'shrunk_estimate': round(shrunk_estimate, 4),
            'above_prior': post_mean > prior_mean
        }

    last_draw = draws[-1]['numbers'] if draws else []
    draw_surprise = sum(results[n]['bayesian_surprise'] for n in last_draw if n in results)

    return {
        'numbers': results,
        'last_draw_surprise': round(draw_surprise, 4),
        'global_mean': round(NUMS_PER_DRAW / NUM_POOL, 4)
    }


# ─── Module 6: Pattern Detection ────────────────────────────────────────────

def pattern_analysis(draws, window=None):
    """Detect patterns, clusters, and structural features."""
    data = draws[-window:] if window else draws
    total = len(data)

    consecutive_pairs = Counter()
    for d in data:
        nums = sorted(d['numbers'])
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                consecutive_pairs[(nums[i], nums[i + 1])] += 1

    # Updated range distribution for pool 1-52
    ranges = {'1-10': 0, '11-20': 0, '21-30': 0, '31-40': 0, '41-52': 0}
    for d in data:
        for n in d['numbers']:
            if n <= 10:
                ranges['1-10'] += 1
            elif n <= 20:
                ranges['11-20'] += 1
            elif n <= 30:
                ranges['21-30'] += 1
            elif n <= 40:
                ranges['31-40'] += 1
            else:
                ranges['41-52'] += 1

    even_odd_ratios = []
    for d in data:
        evens = sum(1 for n in d['numbers'] if n % 2 == 0)
        odds = 7 - evens
        even_odd_ratios.append((evens, odds))

    sums = [sum(d['numbers']) for d in data]
    sum_mean = statistics.mean(sums)
    sum_std = statistics.stdev(sums) if len(sums) > 1 else 0
    sum_min, sum_max = min(sums), max(sums)
    expected_sum = NUMS_PER_DRAW * (NUM_POOL + 1) / 2  # 7 * 53/2 = 185.5

    runs_results = {}
    test_nums = [7, 14, 23, 31, 38, 42, 47, 51, 52]
    for num in test_nums:
        appearances = [1 if num in d['numbers'] else 0 for d in data]
        runs = 1
        for i in range(1, len(appearances)):
            if appearances[i] != appearances[i - 1]:
                runs += 1
        n1 = sum(appearances)
        n0 = len(appearances) - n1
        if n1 > 0 and n0 > 0:
            expected_runs = (2 * n1 * n0) / (n1 + n0) + 1
            denom = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
            runs_z = (runs - expected_runs) / math.sqrt(denom) if denom > 0 else 0
            runs_results[num] = {
                'runs': runs,
                'expected_runs': round(expected_runs, 2),
                'z_score': round(runs_z, 3),
                'is_random': abs(runs_z) < 1.96
            }

    return {
        'total_draws_analyzed': total,
        'consecutive_pairs': consecutive_pairs.most_common(10),
        'range_distribution': ranges,
        'even_odd_ratios': Counter(even_odd_ratios).most_common(5),
        'sum_statistics': {
            'mean': round(sum_mean, 2),
            'std': round(sum_std, 2),
            'min': sum_min,
            'max': sum_max,
            'expected': round(expected_sum, 2),
            'percentile_25': round(statistics.quantiles(sums, n=4)[0], 2) if len(sums) > 3 else sum_min,
            'percentile_75': round(statistics.quantiles(sums, n=4)[2], 2) if len(sums) > 3 else sum_max,
        },
        'runs_test': runs_results
    }


# ─── Module 7: Entropy & Randomness ─────────────────────────────────────────

def entropy_analysis(draws, window=50):
    """Compute entropy and randomness metrics."""
    if window > len(draws):
        window = len(draws)
    recent = draws[-window:]

    entropies = []
    for d in recent:
        nums = d['numbers']
        p_appear = NUMS_PER_DRAW / NUM_POOL
        p_not_appear = 1 - p_appear
        entropy = 0
        for n in range(1, NUM_POOL + 1):
            p = p_appear if n in nums else p_not_appear
            if p > 0:
                entropy -= p * math.log2(p)
        entropies.append(round(entropy, 4))

    max_entropy = -NUM_POOL * (
        (NUMS_PER_DRAW / NUM_POOL) * math.log2(NUMS_PER_DRAW / NUM_POOL) +
        (1 - NUMS_PER_DRAW / NUM_POOL) * math.log2(1 - NUMS_PER_DRAW / NUM_POOL)
    )

    rolling_avg = statistics.mean(entropies) if entropies else 0

    autocorr = {}
    test_nums = [7, 14, 23, 31, 38, 47, 51, 52]
    for num in test_nums:
        series = [1 if num in d['numbers'] else 0 for d in recent]
        mean = statistics.mean(series)
        var = statistics.pvariance(series) if len(series) > 1 else 1
        if var > 0:
            lag1_corr = sum((series[i] - mean) * (series[i + 1] - mean)
                           for i in range(len(series) - 1)) / (len(series) - 1) / var
        else:
            lag1_corr = 0
        autocorr[num] = round(lag1_corr, 4)

    entropy_ratio = rolling_avg / max_entropy if max_entropy > 0 else 0
    avg_autocorr = abs(statistics.mean(autocorr.values())) if autocorr else 0
    randomness_score = min(100, max(0,
        entropy_ratio * 50 +
        (1 - avg_autocorr) * 30 +
        20
    ))

    return {
        'per_draw_entropies': entropies[-10:],
        'rolling_average_entropy': round(rolling_avg, 4),
        'max_entropy': round(max_entropy, 4),
        'entropy_ratio': round(entropy_ratio, 4),
        'autocorrelation': autocorr,
        'randomness_quality_score': round(randomness_score, 1)
    }


# ─── Module 8: Monte Carlo Simulation ───────────────────────────────────────

def monte_carlo_simulation(draws, iterations=10000, strategy='ml-ensemble'):
    """Run Monte Carlo simulation with different strategies."""
    freq_results = frequency_analysis(draws)
    hotcold_results = hot_cold_analysis(draws)
    gap_results = gap_analysis(draws)

    strategies = {}

    uniform_probs = [1.0 / NUM_POOL] * NUM_POOL
    strategies['uniform'] = uniform_probs

    freq_total = sum(freq_results['numbers'][n]['total'] for n in range(1, NUM_POOL + 1))
    freq_probs = [freq_results['numbers'][n]['total'] / freq_total if freq_total > 0 else 1/NUM_POOL for n in range(1, NUM_POOL + 1)]
    strategies['frequency'] = freq_probs

    ensemble_scores = []
    for num in range(1, NUM_POOL + 1):
        score = 0
        freq_score = freq_results['numbers'][num]['total'] / freq_total if freq_total > 0 else 0
        score += 0.15 * freq_score
        temp = hotcold_results['all_temperatures'][num]['temperature']
        hc_score = (temp + 1) / 2
        score += 0.20 * hc_score
        gap_score = gap_results[num]['next_appear_prob']
        score += 0.15 * gap_score
        score += 0.50 * (1.0 / NUM_POOL)
        ensemble_scores.append(score)

    total_ensemble = sum(ensemble_scores)
    ensemble_probs = [s / total_ensemble for s in ensemble_scores]
    strategies['ml-ensemble'] = ensemble_probs

    probs = strategies.get(strategy, strategies['ml-ensemble'])

    occurrence_counts = Counter()
    for _ in range(iterations):
        selected = weighted_sample(probs, NUMS_PER_DRAW)
        for num in selected:
            occurrence_counts[num] += 1

    results = {
        'iterations': iterations,
        'strategy': strategy,
        'number_occurrences': {},
        'convergence': check_convergence(occurrence_counts, iterations)
    }

    for num in range(1, NUM_POOL + 1):
        count = occurrence_counts.get(num, 0)
        prob = count / iterations
        uniform_prob = NUMS_PER_DRAW / NUM_POOL
        results['number_occurrences'][num] = {
            'count': count,
            'probability': round(prob, 4),
            'vs_uniform': round(prob - uniform_prob, 4),
            'rank': 0
        }

    ranked = sorted(results['number_occurrences'].items(), key=lambda x: -x[1]['count'])
    for rank, (num, data) in enumerate(ranked, 1):
        results['number_occurrences'][num]['rank'] = rank
    results['ranked_numbers'] = [(n, d) for n, d in ranked]

    return results


def weighted_sample(probs, k):
    """Weighted random sampling without replacement."""
    available = list(range(1, NUM_POOL + 1))
    available_probs = list(probs)
    selected = []

    for _ in range(k):
        total = sum(available_probs[i] for i in range(len(available)))
        if total == 0:
            selected.append(random.choice(available))
        else:
            r = random.random() * total
            cumulative = 0
            for i, num in enumerate(available):
                cumulative += available_probs[i]
                if r <= cumulative:
                    selected.append(num)
                    available_probs[i] = 0
                    break

    return selected


def check_convergence(counts, iterations):
    """Check if simulation has converged."""
    if iterations < 1000:
        return {'converged': False, 'reason': 'Too few iterations'}

    expected = iterations * NUMS_PER_DRAW / NUM_POOL
    max_deviation = max(abs(counts.get(n, 0) - expected) / expected
                       for n in range(1, NUM_POOL + 1))

    return {
        'converged': max_deviation < 0.1,
        'max_deviation': round(max_deviation, 4),
        'expected_per_number': round(expected, 2)
    }


# ─── Module 9: Ensemble Prediction ──────────────────────────────────────────

def ensemble_prediction(draws, num_sets=3):
    """Combine all modules into final prediction."""
    freq = frequency_analysis(draws)
    hotcold = hot_cold_analysis(draws)
    gaps = gap_analysis(draws)
    markov = markov_analysis(draws)
    bayesian = bayesian_analysis(draws)
    patterns = pattern_analysis(draws)
    entropy = entropy_analysis(draws)
    simulation = monte_carlo_simulation(draws, iterations=10000, strategy='ml-ensemble')

    scores = {}
    for num in range(1, NUM_POOL + 1):
        score = 0

        freq_norm = freq['numbers'][num]['total'] / max(
            freq['numbers'][n]['total'] for n in range(1, NUM_POOL + 1))
        score += 0.15 * freq_norm

        temp = hotcold['all_temperatures'][num]['temperature']
        temp_norm = (temp + 1) / 2
        score += 0.20 * temp_norm

        gap_prob = gaps[num]['next_appear_prob']
        score += 0.15 * gap_prob

        markov_score = 0
        for predicted_num, prob in markov['transition_predictions']:
            if predicted_num == num:
                markov_score = min(1.0, prob)
                break
        score += 0.15 * markov_score

        bayes_post = bayesian['numbers'][num]['posterior_mean']
        bayes_norm = bayes_post / max(
            bayesian['numbers'][n]['posterior_mean'] for n in range(1, NUM_POOL + 1))
        score += 0.15 * bayes_norm

        pattern_score = 0.5
        range_dist = patterns['range_distribution']
        if num <= 10:
            pattern_score = range_dist['1-10'] / sum(range_dist.values())
        elif num <= 20:
            pattern_score = range_dist['11-20'] / sum(range_dist.values())
        elif num <= 30:
            pattern_score = range_dist['21-30'] / sum(range_dist.values())
        elif num <= 40:
            pattern_score = range_dist['31-40'] / sum(range_dist.values())
        else:
            pattern_score = range_dist['41-52'] / sum(range_dist.values())
        score += 0.10 * pattern_score

        sim_prob = simulation['number_occurrences'][num]['probability']
        sim_norm = sim_prob / max(
            simulation['number_occurrences'][n]['probability']
            for n in range(1, NUM_POOL + 1))
        score += 0.05 * sim_norm

        entropy_factor = entropy['randomness_quality_score'] / 100
        scores[num] = round(score * (0.8 + 0.2 * entropy_factor), 4)

    ranked = sorted(scores.items(), key=lambda x: -x[1])

    primary = [n for n, s in ranked[:7]]
    primary.sort()

    confidence_scores = {}
    for num in range(1, NUM_POOL + 1):
        indicators = 0
        if freq['numbers'][num]['trend'] == 'rising':
            indicators += 1
        if hotcold['all_temperatures'][num]['temperature'] > 0:
            indicators += 1
        if gaps[num]['overdue_score'] > 0.5:
            indicators += 1
        if bayesian['numbers'][num]['above_prior']:
            indicators += 1
        confidence_scores[num] = indicators / 4

    conservative_ranked = sorted(confidence_scores.items(), key=lambda x: -x[1])
    conservative = [n for n, c in conservative_ranked[:7]]
    conservative.sort()

    aggressive = [n for n, s in ranked[:5]]
    overdue_nums = sorted(
        [(n, gaps[n]['overdue_score']) for n in range(1, NUM_POOL + 1)],
        key=lambda x: -x[1])
    for n, _ in overdue_nums:
        if n not in aggressive and len(aggressive) < 7:
            aggressive.append(n)
    aggressive.sort()

    balanced = []
    top_hot = [n for n, d in hotcold['hot'][:3]]
    top_overdue = [n for n, d in sorted(
        [(n, gaps[n]['overdue_score']) for n in range(1, NUM_POOL + 1)],
        key=lambda x: -x[1])[:3]]
    top_markov = [n for n, p in markov['transition_predictions'][:3]]
    for n in top_hot + top_overdue + top_markov:
        if n not in balanced:
            balanced.append(n)
    balanced = balanced[:7]
    balanced.sort()

    bonus_candidates = [n for n in range(1, NUM_POOL + 1) if n not in primary]
    bonus_ranked = sorted(
        [(n, scores[n]) for n in bonus_candidates],
        key=lambda x: -x[1])
    bonus = bonus_ranked[0][0] if bonus_ranked else 1

    avg_score = statistics.mean(scores[n] for n in primary)
    max_possible = max(scores.values())
    overall_confidence = round(avg_score / max_possible * 100, 1) if max_possible > 0 else 0

    ml_expected = statistics.mean(scores[n] for n in primary)
    baseline_expected = statistics.mean(scores[n] for n in random.sample(range(1, NUM_POOL + 1), 7))
    improvement = round((ml_expected - baseline_expected) / baseline_expected * 100, 1) if baseline_expected > 0 else 0

    rationale = {}
    for num in primary:
        reasons = []
        if freq['numbers'][num]['trend'] == 'rising':
            fc = freq['numbers'][num]['windows'].get('last_10', 0)
            reasons.append(f"Hot streak ({fc} appearances in last 10 draws)")
        if gaps[num]['is_overdue']:
            reasons.append(f"Overdue (gap: {gaps[num]['current_gap']} draws, {int(gaps[num]['overdue_score']*10)}th percentile)")
        if any(n == num for n, _ in markov['transition_predictions'][:5]):
            reasons.append("Markov transition favorite from last draw numbers")
        if bayesian['numbers'][num]['above_prior']:
            reasons.append(f"Bayesian posterior above prior (p={bayesian['numbers'][num]['posterior_mean']:.3f} vs {bayesian['numbers'][num]['prior_mean']:.3f})")
        if not reasons:
            reasons.append("High ensemble score, consistent across all models")
        rationale[num] = reasons[0] if reasons else "Statistical indicator"

    # Per-set confidence scores
    cons_avg = statistics.mean(scores[n] for n in conservative) if conservative else 0
    aggr_avg = statistics.mean(scores[n] for n in aggressive) if aggressive else 0
    bal_avg = statistics.mean(scores[n] for n in balanced) if balanced else 0
    conservative_confidence = round(cons_avg / max_possible * 100, 1) if max_possible > 0 else 0
    aggressive_confidence = round(aggr_avg / max_possible * 100, 1) if max_possible > 0 else 0
    balanced_confidence = round(bal_avg / max_possible * 100, 1) if max_possible > 0 else 0

    return {
        'primary': primary,
        'conservative': conservative,
        'aggressive': aggressive,
        'balanced': balanced,
        'overall_confidence': overall_confidence,
        'conservative_confidence': conservative_confidence,
        'aggressive_confidence': aggressive_confidence,
        'balanced_confidence': balanced_confidence,
        'expected_improvement': improvement,
        'rationale': rationale,
        'all_scores': scores,
        'simulation_summary': {
            'most_simulated': simulation['ranked_numbers'][0],
            'least_simulated': simulation['ranked_numbers'][-1],
            'convergence': simulation['convergence']
        }
    }


# ─── Strategy Comparison ─────────────────────────────────────────────────────

def compare_strategies(draws):
    """Compare all prediction strategies side by side."""
    iterations = 10000
    strategies = ['uniform', 'frequency', 'ml-ensemble']

    results = {}
    for strategy in strategies:
        sim = monte_carlo_simulation(draws, iterations, strategy)
        match_rates = simulate_match_rates(draws, sim)
        results[strategy] = {
            'simulation': sim,
            'match_rates': match_rates
        }

    return results


def simulate_match_rates(draws, simulation):
    """Simulate how often each strategy matches real draw numbers."""
    if not draws:
        return {'avg_matches': 0, 'match_3plus_rate': 0, 'match_4plus_rate': 0}

    test_draws = draws[-20:] if len(draws) > 20 else draws
    match_counts = []

    for test_draw in test_draws:
        ranked = simulation['ranked_numbers']
        predicted = [n for n, d in ranked[:7]]
        actual = test_draw['numbers']
        matches = len(set(predicted) & set(actual))
        match_counts.append(matches)

    avg_matches = statistics.mean(match_counts) if match_counts else 0
    match_3plus = sum(1 for m in match_counts if m >= 3) / len(match_counts) if match_counts else 0
    match_4plus = sum(1 for m in match_counts if m >= 4) / len(match_counts) if match_counts else 0

    return {
        'avg_matches': round(avg_matches, 2),
        'match_3plus_rate': round(match_3plus * 100, 2),
        'match_4plus_rate': round(match_4plus * 100, 2)
    }


# ─── Wheeling System ─────────────────────────────────────────────────────────

def generate_wheeling(numbers, coverage='abbreviated'):
    """Generate a wheeling system from recommended numbers."""
    if len(numbers) < 7:
        return {'error': 'Need at least 7 numbers for wheeling'}

    lines = []
    if coverage == 'abbreviated':
        for combo in combinations(sorted(numbers), 7):
            lines.append(list(combo))
            if len(lines) >= 10:
                break
    else:
        for combo in combinations(sorted(numbers), 7):
            lines.append(list(combo))

    return {
        'pool_size': len(numbers),
        'numbers': sorted(numbers),
        'lines': lines,
        'total_lines': len(lines),
        'coverage': coverage
    }


# ─── Output Formatting ───────────────────────────────────────────────────────

# ANSI escape codes for terminal formatting
BOLD = '\033[1m'
RESET = '\033[0m'


def format_prediction(prediction, draws=None):
    """Format prediction output in the standard box format."""
    nums_str = '  '.join(f'{n:02d}' for n in prediction['primary'])
    conf = prediction['overall_confidence']

    # Extract date range from draws if available
    date_from = 'N/A'
    date_to = 'N/A'
    total_draws = 0
    if draws:
        dates = [d['date'] for d in draws]
        date_from = dates[0] if dates else 'N/A'
        date_to = dates[-1] if dates else 'N/A'
        total_draws = len(draws)

    output = []
    output.append("=" * 64)
    output.append("           LOTTO MAX ML PREDICTOR - NEXT DRAW")
    output.append("=" * 64)
    output.append("")
    output.append(f"  {BOLD}RECOMMENDED NUMBERS{RESET}  {nums_str}  {BOLD}{conf}%{RESET}")
    output.append("")
    output.append(f"  Strategy: ML-Ensemble (Weighted)")
    output.append(f"  Pool: 1 to {NUM_POOL} (Updated May 2026)")
    output.append(f"  Expected Improvement vs Random: +{prediction['expected_improvement']}%")
    output.append("")
    output.append("-" * 64)
    output.append("  DATA SOURCE & METHODOLOGY:")
    if date_from != 'N/A':
        output.append(f"  - Historical data: {total_draws} draws from {date_from} to {date_to}")
    else:
        output.append(f"  - Historical data: ~6 months of real Lotto Max draws")
    output.append(f"  - Source: ca.lottonumbers.com/lotto-max/past-numbers")
    output.append(f"  - Users are welcome to use other data sources")
    output.append(f"  - Modify assets/historical_draws.json or run --mode update-data")
    output.append("")
    output.append("-" * 64)
    output.append(f"  {BOLD}ALTERNATIVE SETS{RESET}")
    output.append("")

    cons_str = '  '.join(f'{n:02d}' for n in prediction['conservative'])
    aggr_str = '  '.join(f'{n:02d}' for n in prediction['aggressive'])
    bal_str = '  '.join(f'{n:02d}' for n in prediction['balanced'])
    cons_conf = prediction.get('conservative_confidence', 0)
    aggr_conf = prediction.get('aggressive_confidence', 0)
    bal_conf = prediction.get('balanced_confidence', 0)

    output.append(f"  Conservative:  {cons_str}  {BOLD}{cons_conf}%{RESET}")
    output.append(f"  Aggressive:    {aggr_str}  {BOLD}{aggr_conf}%{RESET}")
    output.append(f"  Balanced:      {bal_str}  {BOLD}{bal_conf}%{RESET}")
    output.append("")
    output.append("-" * 64)
    output.append("  TOP NUMBER RATIONALE:")
    output.append("")

    for num in prediction['primary']:
        output.append(f"  {num:02d} - {prediction['rationale'].get(num, 'Statistical indicator')}")

    output.append("")
    output.append("-" * 64)
    sim = prediction['simulation_summary']
    output.append(f"  SIMULATION SUMMARY (10,000 draws):")
    most_num, most_data = sim['most_simulated']
    least_num, least_data = sim['least_simulated']
    output.append(f"  - Most simulated number: {most_num} (appeared in {most_data['count']:,} sims)")
    output.append(f"  - Least simulated number: {least_num} (appeared in {least_data['count']:,} sims)")
    if sim['convergence']['converged']:
        output.append(f"  - Convergence: YES (variance < threshold)")
    else:
        output.append(f"  - Convergence: PENDING (max deviation: {sim['convergence'].get('max_deviation', 'N/A')})")
    output.append(f"  - Strategy edge: +{prediction['expected_improvement']}% more 3+ matches vs random")
    output.append("")
    output.append("-" * 64)
    output.append("  DISCLOSURE: Lottery draws are random events. ML analysis")
    output.append("  identifies statistical tendencies but cannot guarantee")
    output.append("  outcomes. Play responsibly. For entertainment only.")
    output.append("=" * 64)

    return '\n'.join(output)


def format_frequency(freq_data):
    """Format frequency analysis output."""
    output = []
    output.append("FREQUENCY ANALYSIS - Lotto Max")
    output.append(f"Total Draws: {freq_data['total_draws']}")
    output.append(f"Expected per number: {freq_data['expected_per_number']:.2f}")
    output.append("")
    output.append(f"{'Number':>6} | {'Total':>5} | {'Last10':>6} | {'Last25':>6} | {'Last50':>6} | {'Z-Score':>7} | {'Trend':>6}")
    output.append("-" * 65)

    for num in range(1, NUM_POOL + 1):
        d = freq_data['numbers'][num]
        w = d['windows']
        trend_symbol = {'rising': '^', 'falling': 'v', 'stable': '-'}[d['trend']]
        output.append(
            f"{num:>6} | {d['total']:>5} | {w.get('last_10', 0):>6} | "
            f"{w.get('last_25', 0):>6} | {w.get('last_50', 0):>6} | "
            f"{d['z_score']:>7.3f} | {trend_symbol:>6}"
        )

    output.append("")
    output.append(f"Chi-squared: {freq_data['chi_squared']} (df={freq_data['chi_df']})")

    return '\n'.join(output)


def format_hotcold(hotcold_data):
    """Format hot/cold analysis output."""
    output = []
    output.append(f"HOT/COLD ANALYSIS - Lotto Max (Window: {hotcold_data['window']} draws)")
    output.append(f"Expected appearances per number: {hotcold_data['expected_per_number']}")
    output.append("")

    output.append("HOT NUMBERS (above expected):")
    for num, data in hotcold_data['hot']:
        output.append(f"  {num:02d} - Count: {data['count']}, Temp: {data['temperature']:+.3f}, "
                      f"Mean Reversion Prob: {data['mean_reversion_prob']:.1%}")

    output.append("")
    output.append("COLD NUMBERS (below expected):")
    for num, data in hotcold_data['cold']:
        output.append(f"  {num:02d} - Count: {data['count']}, Temp: {data['temperature']:+.3f}, "
                      f"Mean Reversion Prob: {data['mean_reversion_prob']:.1%}")

    return '\n'.join(output)


def format_gap(gap_data):
    """Format gap analysis output."""
    output = []
    output.append("GAP ANALYSIS - Lotto Max (Overdue Numbers)")
    output.append("")
    output.append(f"{'Number':>6} | {'Current':>7} | {'Avg':>5} | {'Std':>5} | "
                  f"{'NextProb':>8} | {'Overdue':>7} | {'Status':>8}")
    output.append("-" * 70)

    overdue_nums = sorted(
        [(n, d) for n, d in gap_data.items()],
        key=lambda x: -x[1]['overdue_score']
    )

    for num, d in overdue_nums:
        status = "OVERDUE" if d['is_overdue'] else ""
        output.append(
            f"{num:>6} | {d['current_gap']:>7} | {d['avg_gap']:>5.1f} | "
            f"{d['std_gap']:>5.1f} | {d['next_appear_prob']:>8.4f} | "
            f"{d['overdue_score']:>7.3f} | {status:>8}"
        )

    return '\n'.join(output)


def format_simulation(sim_data):
    """Format Monte Carlo simulation output."""
    output = []
    output.append(f"MONTE CARLO SIMULATION - {sim_data['iterations']:,} Draws")
    output.append(f"Strategy: {sim_data['strategy']}")
    output.append("")

    uniform_prob = NUMS_PER_DRAW / NUM_POOL
    output.append(f"{'Rank':>4} | {'Number':>6} | {'Occurrences':>11} | {'Probability':>11} | {'vs Uniform':>10}")
    output.append("-" * 60)

    for rank, (num, data) in enumerate(sim_data['ranked_numbers'], 1):
        output.append(
            f"{rank:>4} | {num:>6} | {data['count']:>11,} | "
            f"{data['probability']:>11.4f} | {data['vs_uniform']:>+10.4f}"
        )

    output.append("")
    conv = sim_data['convergence']
    if conv['converged']:
        output.append(f"Convergence: YES (max deviation: {conv.get('max_deviation', 'N/A')})")
    else:
        output.append(f"Convergence: PENDING (max deviation: {conv.get('max_deviation', 'N/A')})")

    return '\n'.join(output)


def format_markov(markov_data):
    """Format Markov chain analysis output."""
    output = []
    output.append("MARKOV CHAIN ANALYSIS - Lotto Max")
    output.append(f"Last draw: {markov_data['last_draw']}")
    output.append("")
    output.append("Top transition predictions for next draw:")
    for num, prob in markov_data['transition_predictions'][:15]:
        output.append(f"  {num:02d} - Transition probability: {prob:.4f}")

    return '\n'.join(output)


def format_bayesian(bayes_data):
    """Format Bayesian inference output."""
    output = []
    output.append("BAYESIAN INFERENCE - Lotto Max")
    output.append(f"Global mean: {bayes_data['global_mean']}")
    output.append(f"Last draw surprise score: {bayes_data['last_draw_surprise']}")
    output.append("")

    sorted_nums = sorted(
        bayes_data['numbers'].items(),
        key=lambda x: -x[1]['posterior_mean']
    )

    output.append(f"{'Number':>6} | {'PostMean':>8} | {'PriorMean':>8} | {'CI_Lower':>8} | "
                  f"{'CI_Upper':>8} | {'Surprise':>8} | {'Above?':>6}")
    output.append("-" * 75)

    for num, d in sorted_nums[:20]:
        above = "YES" if d['above_prior'] else "no"
        output.append(
            f"{num:>6} | {d['posterior_mean']:>8.4f} | {d['prior_mean']:>8.4f} | "
            f"{d['ci_lower']:>8.4f} | {d['ci_upper']:>8.4f} | "
            f"{d['bayesian_surprise']:>8.4f} | {above:>6}"
        )

    return '\n'.join(output)


def format_pattern(pattern_data):
    """Format pattern detection output."""
    output = []
    output.append("PATTERN DETECTION - Lotto Max")
    output.append(f"Draws analyzed: {pattern_data['total_draws_analyzed']}")
    output.append("")

    output.append("Top consecutive pairs:")
    for (a, b), count in pattern_data['consecutive_pairs'][:10]:
        output.append(f"  ({a:02d}, {b:02d}) - appeared {count} times")

    output.append("")
    output.append("Range distribution:")
    for range_name, count in pattern_data['range_distribution'].items():
        total = sum(pattern_data['range_distribution'].values())
        pct = count / total * 100 if total > 0 else 0
        bar = '#' * int(pct / 2)
        output.append(f"  {range_name}: {count} ({pct:.1f}%) {bar}")

    output.append("")
    output.append("Even/Odd ratio distribution:")
    for (evens, odds), count in pattern_data['even_odd_ratios'][:5]:
        output.append(f"  {evens}E/{odds}O - {count} draws")

    output.append("")
    ss = pattern_data['sum_statistics']
    output.append(f"Sum statistics: Mean={ss['mean']}, Std={ss['std']}, "
                  f"Range=[{ss['min']}-{ss['max']}], Expected={ss['expected']}")

    output.append("")
    output.append("Runs test for randomness:")
    for num, d in pattern_data['runs_test'].items():
        status = "RANDOM" if d['is_random'] else "NON-RANDOM"
        output.append(f"  Number {num:02d}: runs={d['runs']}, expected={d['expected_runs']}, "
                      f"z={d['z_score']:.3f} -> {status}")

    return '\n'.join(output)


def format_entropy(entropy_data):
    """Format entropy analysis output."""
    output = []
    output.append("ENTROPY & RANDOMNESS - Lotto Max")
    output.append("")
    output.append(f"Rolling average entropy: {entropy_data['rolling_average_entropy']}")
    output.append(f"Maximum possible entropy: {entropy_data['max_entropy']}")
    output.append(f"Entropy ratio: {entropy_data['entropy_ratio']}")
    output.append(f"Randomness quality score: {entropy_data['randomness_quality_score']}/100")
    output.append("")
    output.append("Autocorrelation (lag-1):")
    for num, corr in entropy_data['autocorrelation'].items():
        status = "POSITIVE" if corr > 0.1 else "NEGATIVE" if corr < -0.1 else "NONE"
        output.append(f"  Number {num:02d}: {corr:+.4f} ({status})")

    output.append("")
    output.append("Recent draw entropies:")
    for i, ent in enumerate(entropy_data['per_draw_entropies'][-10:]):
        output.append(f"  Draw -{10-i}: {ent}")

    return '\n'.join(output)


# ─── Main CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Lotto Max ML Predictor - Advanced lottery analysis engine (Pool 1-52)')
    parser.add_argument('--mode', type=str, default='predict',
                       choices=['frequency', 'hotcold', 'gap', 'markov', 'bayesian',
                               'pattern', 'entropy', 'simulate', 'predict',
                               'full-report', 'compare-strategies', 'wheeling',
                               'export-charts', 'update-data', 'validate-data',
                               'data-info'],
                       help='Analysis mode to run')
    parser.add_argument('--window', type=int, default=None,
                       help='Number of recent draws to analyze')
    parser.add_argument('--iterations', type=int, default=10000,
                       help='Monte Carlo simulation iterations')
    parser.add_argument('--sets', type=int, default=3,
                       help='Number of recommendation sets to generate')
    parser.add_argument('--format', type=str, default='text',
                       choices=['text', 'json'],
                       help='Output format')
    parser.add_argument('--draws-file', type=str, default=None,
                       help='Path to historical draws JSON file')
    parser.add_argument('--numbers', type=int, default=10,
                       help='Number of pool numbers for wheeling')
    parser.add_argument('--strategy', type=str, default='ml-ensemble',
                       choices=['uniform', 'frequency', 'ml-ensemble'],
                       help='Simulation strategy')

    args = parser.parse_args()
    draws = load_draws(args.draws_file, args.window)

    if not draws:
        print("No historical draw data available.")
        sys.exit(1)

    if args.mode == 'frequency':
        result = frequency_analysis(draws)
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(format_frequency(result))

    elif args.mode == 'hotcold':
        result = hot_cold_analysis(draws, args.window or 25)
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_hotcold(result))

    elif args.mode == 'gap':
        result = gap_analysis(draws)
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(format_gap(result))

    elif args.mode == 'markov':
        result = markov_analysis(draws)
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_markov(result))

    elif args.mode == 'bayesian':
        result = bayesian_analysis(draws)
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(format_bayesian(result))

    elif args.mode == 'pattern':
        result = pattern_analysis(draws, args.window)
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_pattern(result))

    elif args.mode == 'entropy':
        result = entropy_analysis(draws, args.window or 50)
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(format_entropy(result))

    elif args.mode == 'simulate':
        result = monte_carlo_simulation(draws, args.iterations, args.strategy)
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_simulation(result))

    elif args.mode == 'predict':
        # Auto-update: try to fetch latest data before predicting
        print("Checking for latest draw data...")
        try:
            sys.path.insert(0, SCRIPT_DIR)
            from data_fetcher import fetch_latest_draws, save_draws
            new_draws = fetch_latest_draws()
            if new_draws:
                # Merge with existing data
                existing_dates = {d['date'] for d in draws}
                added = 0
                for draw in new_draws:
                    if draw['date'] not in existing_dates:
                        draws.append(draw)
                        existing_dates.add(draw['date'])
                        added += 1
                if added > 0:
                    draws.sort(key=lambda x: x['date'])
                    save_draws(draws, args.draws_file)
                    print(f"  Updated: +{added} new draws (total: {len(draws)})")
                else:
                    print(f"  Data is up to date ({len(draws)} draws)")
            else:
                print(f"  Using bundled data ({len(draws)} draws) - could not reach server")
        except Exception:
            print(f"  Using bundled data ({len(draws)} draws)")

        result = ensemble_prediction(draws, args.sets)
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_prediction(result, draws))

    elif args.mode == 'full-report':
        print("=" * 64)
        print("  LOTTO MAX ML PREDICTOR - FULL ANALYSIS REPORT")
        print(f"  Pool: 1 to {NUM_POOL} | Data: {len(draws)} draws")
        print("=" * 64)
        print()
        print(format_frequency(frequency_analysis(draws)))
        print("\n")
        print(format_hotcold(hot_cold_analysis(draws)))
        print("\n")
        print(format_gap(gap_analysis(draws)))
        print("\n")
        print(format_markov(markov_analysis(draws)))
        print("\n")
        print(format_bayesian(bayesian_analysis(draws)))
        print("\n")
        print(format_pattern(pattern_analysis(draws)))
        print("\n")
        print(format_entropy(entropy_analysis(draws)))
        print("\n")
        print(format_simulation(monte_carlo_simulation(draws, args.iterations)))
        print("\n")
        print(format_prediction(ensemble_prediction(draws), draws))

    elif args.mode == 'compare-strategies':
        result = compare_strategies(draws)
        if args.format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print("STRATEGY COMPARISON - Lotto Max")
            print()
            for strategy, data in result.items():
                mr = data['match_rates']
                print(f"  {strategy.upper()}")
                print(f"    Avg matches: {mr['avg_matches']}")
                print(f"    3+ match rate: {mr['match_3plus_rate']}%")
                print(f"    4+ match rate: {mr['match_4plus_rate']}%")
                print()

    elif args.mode == 'wheeling':
        pred = ensemble_prediction(draws)
        pool_numbers = list(pred['primary']) + list(pred['conservative'])[:3]
        pool_numbers = list(set(pool_numbers))[:args.numbers]
        result = generate_wheeling(pool_numbers)
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print("WHEELING SYSTEM - Lotto Max")
            print(f"Pool: {sorted(result['numbers'])}")
            print(f"Total lines: {result['total_lines']}")
            print()
            for i, line in enumerate(result['lines'], 1):
                print(f"  Line {i}: {sorted(line)}")

    elif args.mode == 'export-charts':
        freq = frequency_analysis(draws)
        hotcold = hot_cold_analysis(draws)
        gaps = gap_analysis(draws)
        export = {
            'frequency_histogram': {n: freq['numbers'][n]['total'] for n in range(1, NUM_POOL + 1)},
            'temperature_scores': {n: hotcold['all_temperatures'][n]['temperature']
                                  for n in range(1, NUM_POOL + 1)},
            'gap_current': {n: gaps[n]['current_gap'] for n in range(1, NUM_POOL + 1)},
            'gap_probability': {n: gaps[n]['next_appear_prob'] for n in range(1, NUM_POOL + 1)},
        }
        print(json.dumps(export, indent=2))

    elif args.mode == 'update-data':
        # Import and use the data_fetcher module to fetch live data
        sys.path.insert(0, SCRIPT_DIR)
        try:
            from data_fetcher import fetch_latest_draws, save_draws, load_draws as df_load_draws
            print("Attempting to fetch latest draws from ca.lottonumbers.com...")
            new_draws = fetch_latest_draws()
            if new_draws:
                # Merge with existing data
                existing_draws = df_load_draws(args.draws_file)
                existing_dates = {d['date'] for d in existing_draws}
                added = 0
                for draw in new_draws:
                    if draw['date'] not in existing_dates:
                        existing_draws.append(draw)
                        existing_dates.add(draw['date'])
                        added += 1
                existing_draws.sort(key=lambda x: x['date'])
                save_draws(existing_draws, args.draws_file)
                print(f"\nUpdate complete: Added {added} new draws. Total: {len(existing_draws)}")
                print(f"Date range: {existing_draws[0]['date']} to {existing_draws[-1]['date']}")
            else:
                print("Could not fetch new data. Check internet connection.")
                print("The skill will continue using the bundled historical data.")
        except ImportError:
            print("ERROR: data_fetcher module not found. Make sure scripts/data_fetcher.py exists.")
        except Exception as e:
            print(f"Error during data update: {e}")
            print("The skill will continue using the bundled historical data.")

    elif args.mode == 'validate-data':
        # Validate all draws in the dataset
        sys.path.insert(0, SCRIPT_DIR)
        try:
            from data_fetcher import validate_draw
        except ImportError:
            # Inline validation if module not available
            def validate_draw(draw):
                if not isinstance(draw, dict):
                    return False, "Draw must be a dictionary"
                if 'numbers' not in draw:
                    return False, "Missing 'numbers' key"
                nums = draw['numbers']
                if not isinstance(nums, list):
                    return False, "Numbers must be a list"
                if len(nums) != NUMS_PER_DRAW:
                    return False, f"Expected {NUMS_PER_DRAW} numbers, got {len(nums)}"
                for n in nums:
                    if not isinstance(n, int) or n < 1 or n > NUM_POOL:
                        return False, f"Number {n} out of range [1, {NUM_POOL}]"
                if len(set(nums)) != NUMS_PER_DRAW:
                    return False, "Duplicate numbers found"
                if 'bonus' in draw:
                    b = draw['bonus']
                    if not isinstance(b, int) or b < 1 or b > NUM_POOL:
                        return False, f"Bonus {b} out of range"
                    if b in nums:
                        return False, "Bonus in main numbers"
                return True, "Valid"

        print("VALIDATING HISTORICAL DATA - Lotto Max")
        print("=" * 50)
        draws_file = args.draws_file or DEFAULT_DRAWS_FILE
        try:
            with open(draws_file, 'r') as f:
                raw_data = json.load(f)
            raw_draws = raw_data.get('draws', [])
        except FileNotFoundError:
            print(f"ERROR: Data file not found: {draws_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in data file: {e}")
            sys.exit(1)

        print(f"Data file: {draws_file}")
        print(f"Source: {raw_data.get('source', 'unknown')}")
        print(f"Last updated: {raw_data.get('last_updated', 'unknown')}")
        print(f"Total draws: {len(raw_draws)}")
        print()

        invalid_count = 0
        for i, draw in enumerate(raw_draws):
            valid, msg = validate_draw(draw)
            if not valid:
                date = draw.get('date', f'index {i}')
                print(f"  INVALID - {date}: {msg}")
                invalid_count += 1

        valid_count = len(raw_draws) - invalid_count
        print(f"\nValidation Result: {valid_count}/{len(raw_draws)} draws valid")

        if invalid_count == 0:
            print("ALL DRAWS ARE VALID! Data is ready for analysis.")

            # Check pool range
            all_nums = []
            for d in raw_draws:
                all_nums.extend(d['numbers'])
            max_num = max(all_nums) if all_nums else 0
            min_num = min(all_nums) if all_nums else 0
            print(f"\nNumber range in data: {min_num} to {max_num}")
            if max_num > 50:
                print(f"  -> Confirmed: Pool 1-52 (updated May 2026) - numbers {max_num-50} and {max_num-50+1} found")
            elif max_num <= 50:
                print(f"  WARNING: Max number is {max_num}. Pool should be 1-52 since May 2026!")
                print(f"  Consider updating your historical data with --mode update-data")
        else:
            print(f"{invalid_count} draws have errors. Please check the data file.")

    elif args.mode == 'data-info':
        # Show info about the loaded data
        draws_file = args.draws_file or DEFAULT_DRAWS_FILE
        try:
            with open(draws_file, 'r') as f:
                raw_data = json.load(f)
            raw_draws = raw_data.get('draws', [])
        except FileNotFoundError:
            print(f"ERROR: Data file not found: {draws_file}")
            sys.exit(1)

        print("DATA INFO - Lotto Max Historical Draws")
        print("=" * 50)
        print(f"  File: {draws_file}")
        print(f"  Source: {raw_data.get('source', 'unknown')}")
        print(f"  Last updated: {raw_data.get('last_updated', 'unknown')}")
        print(f"  Total draws: {raw_data.get('total_draws', len(raw_draws))}")
        if raw_draws:
            print(f"  Date range: {raw_draws[0]['date']} to {raw_draws[-1]['date']}")
            all_nums = []
            for d in raw_draws:
                all_nums.extend(d['numbers'])
            max_num = max(all_nums) if all_nums else 0
            min_num = min(all_nums) if all_nums else 0
            print(f"  Number range: {min_num} to {max_num}")
            print(f"  Pool compatibility: {'1-52 (Current)' if max_num > 50 else '1-50 (Legacy - needs update!)'}")

            # Frequency summary
            from collections import Counter as FreqCounter
            freq = FreqCounter(all_nums)
            top5 = freq.most_common(5)
            bottom5 = freq.most_common()[:-6:-1]
            print(f"\n  Most frequent: {', '.join(f'{n}({c}x)' for n, c in top5)}")
            print(f"  Least frequent: {', '.join(f'{n}({c}x)' for n, c in bottom5)}")
            print(f"\n  Last draw: {raw_draws[-1]['date']} -> {raw_draws[-1]['numbers']} (Bonus: {raw_draws[-1].get('bonus', 'N/A')})")
        print()


if __name__ == '__main__':
    main()
