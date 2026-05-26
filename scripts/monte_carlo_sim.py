#!/usr/bin/env python3
"""
Monte Carlo Simulation Module for Lotto Max ML Predictor
Standalone simulation engine with multiple strategies and convergence tracking.

Created by: Reza Azizi - May 2026
If you are interested in developing or enhancing an AI automation system—whether for workflows,
data processing, monitoring, or custom integrations—please feel free to reach out to
flowaiautomationsupport@gmail.com. I would be happy to discuss your requirements, propose a tailored solution,
and explore how automation can improve efficiency, accuracy, and scalability for your environment.


"""

import math
import random
from collections import Counter, defaultdict

NUM_POOL = 52
NUMS_PER_DRAW = 7


class MonteCarloSimulator:
    """Configurable Monte Carlo simulation engine for lottery analysis."""

    def __init__(self, probabilities=None, strategy='ml-ensemble'):
        self.probabilities = probabilities or [1.0 / NUM_POOL] * NUM_POOL
        self.strategy = strategy
        self.results = []
        self.convergence_data = []

    def set_probabilities(self, probs):
        """Set custom probability distribution."""
        total = sum(probs)
        self.probabilities = [p / total for p in probs]

    def run(self, iterations=10000, track_convergence=True, convergence_interval=1000):
        """Execute Monte Carlo simulation."""
        occurrence_counts = Counter()
        self.results = []
        self.convergence_data = []

        for i in range(iterations):
            selected = self._weighted_sample(NUMS_PER_DRAW)
            for num in selected:
                occurrence_counts[num] += 1

            # Track convergence
            if track_convergence and (i + 1) % convergence_interval == 0:
                convergence_point = self._compute_convergence(occurrence_counts, i + 1)
                self.convergence_data.append({
                    'iteration': i + 1,
                    'max_deviation': convergence_point,
                    'converged': convergence_point < 0.05
                })

        self.results = self._format_results(occurrence_counts, iterations)
        return self.results

    def _weighted_sample(self, k):
        """Weighted random sampling without replacement."""
        available = list(range(1, NUM_POOL + 1))
        available_probs = list(self.probabilities)
        selected = []

        for _ in range(k):
            total = sum(available_probs)
            if total == 0:
                selected.append(random.choice(available))
                continue

            r = random.random() * total
            cumulative = 0
            chosen_idx = 0
            for idx in range(len(available)):
                cumulative += available_probs[idx]
                if r <= cumulative:
                    chosen_idx = idx
                    break

            selected.append(available[chosen_idx])
            available_probs[chosen_idx] = 0

        return selected

    def _compute_convergence(self, counts, iterations):
        """Compute maximum deviation from expected."""
        expected = iterations * NUMS_PER_DRAW / NUM_POOL
        if expected == 0:
            return float('inf')
        max_deviation = max(
            abs(counts.get(n, 0) - expected) / expected
            for n in range(1, NUM_POOL + 1)
        )
        return max_deviation

    def _format_results(self, counts, iterations):
        """Format simulation results."""
        results = {
            'iterations': iterations,
            'strategy': self.strategy,
            'numbers': {},
            'ranked': [],
            'convergence': {}
        }

        uniform_prob = NUMS_PER_DRAW / NUM_POOL

        for num in range(1, NUM_POOL + 1):
            count = counts.get(num, 0)
            prob = count / iterations
            results['numbers'][num] = {
                'count': count,
                'probability': round(prob, 4),
                'vs_uniform': round(prob - uniform_prob, 4)
            }

        ranked = sorted(results['numbers'].items(),
                       key=lambda x: -x[1]['count'])
        for rank, (num, data) in enumerate(ranked, 1):
            results['numbers'][num]['rank'] = rank

        results['ranked'] = [(n, d) for n, d in ranked]

        # Final convergence
        if self.convergence_data:
            final = self.convergence_data[-1]
            results['convergence'] = {
                'converged': final['converged'],
                'max_deviation': round(final['max_deviation'], 4),
                'checkpoints': len(self.convergence_data)
            }

        return results

    def simulate_match_rates(self, test_draws, top_k=7):
        """Simulate match rates against historical draws."""
        if not self.results or not self.results.get('ranked'):
            return {'avg_matches': 0, 'match_3plus_rate': 0}

        predicted = [n for n, d in self.results['ranked'][:top_k]]
        match_counts = []

        for draw in test_draws:
            actual = draw['numbers']
            matches = len(set(predicted) & set(actual))
            match_counts.append(matches)

        avg = sum(match_counts) / len(match_counts) if match_counts else 0
        match_3plus = sum(1 for m in match_counts if m >= 3) / len(match_counts) if match_counts else 0

        return {
            'avg_matches': round(avg, 2),
            'match_3plus_rate': round(match_3plus * 100, 2),
            'match_distribution': Counter(match_counts)
        }


def build_strategy_probabilities(strategy, freq_data=None, hotcold_data=None,
                                  gap_data=None, bayesian_data=None):
    """Build probability distribution for a given strategy."""
    if strategy == 'uniform':
        return [1.0 / NUM_POOL] * NUM_POOL

    elif strategy == 'frequency':
        if not freq_data:
            return [1.0 / NUM_POOL] * NUM_POOL
        total = sum(freq_data['numbers'][n]['total'] for n in range(1, NUM_POOL + 1))
        if total == 0:
            return [1.0 / NUM_POOL] * NUM_POOL
        return [freq_data['numbers'][n]['total'] / total for n in range(1, NUM_POOL + 1)]

    elif strategy == 'hot-cold':
        if not hotcold_data:
            return [1.0 / NUM_POOL] * NUM_POOL
        temps = [hotcold_data['all_temperatures'][n]['temperature']
                 for n in range(1, NUM_POOL + 1)]
        # Convert temperature (-1 to 1) to probability weight
        weights = [(t + 1.5) for t in temps]  # shift to positive
        total = sum(weights)
        return [w / total for w in weights]

    elif strategy == 'gap-based':
        if not gap_data:
            return [1.0 / NUM_POOL] * NUM_POOL
        probs = [gap_data[n]['next_appear_prob'] for n in range(1, NUM_POOL + 1)]
        total = sum(probs)
        if total == 0:
            return [1.0 / NUM_POOL] * NUM_POOL
        return [p / total for p in probs]

    elif strategy == 'ml-ensemble':
        # Combine all signals with weighted ensemble
        scores = []
        for num in range(1, NUM_POOL + 1):
            score = 1.0 / NUM_POOL  # baseline

            if freq_data:
                freq_norm = freq_data['numbers'][num]['total'] / max(
                    freq_data['numbers'][n]['total'] for n in range(1, NUM_POOL + 1))
                score += 0.15 * freq_norm

            if hotcold_data:
                temp = hotcold_data['all_temperatures'][num]['temperature']
                score += 0.20 * (temp + 1) / 2

            if gap_data:
                score += 0.15 * gap_data[num]['next_appear_prob']

            if bayesian_data:
                bayes_norm = bayesian_data['numbers'][num]['posterior_mean'] / max(
                    bayesian_data['numbers'][n]['posterior_mean']
                    for n in range(1, NUM_POOL + 1))
                score += 0.15 * bayes_norm

            scores.append(score)

        total = sum(scores)
        return [s / total for s in scores]

    return [1.0 / NUM_POOL] * NUM_POOL


if __name__ == '__main__':
    # Demo: Run a simple simulation
    print("Monte Carlo Simulation Module - Lotto Max ML Predictor")
    print("=" * 50)

    sim = MonteCarloSimulator(strategy='uniform')
    result = sim.run(iterations=10000)

    print(f"\nSimulation complete: {result['iterations']:,} iterations")
    print(f"Strategy: {result['strategy']}")

    if result.get('convergence'):
        print(f"Converged: {result['convergence'].get('converged', 'N/A')}")

    print("\nTop 10 numbers by occurrence:")
    for rank, (num, data) in enumerate(result['ranked'][:10], 1):
        print(f"  {rank}. Number {num:02d}: {data['count']:,} occurrences "
              f"({data['probability']:.4f}, vs uniform: {data['vs_uniform']:+.4f})")
