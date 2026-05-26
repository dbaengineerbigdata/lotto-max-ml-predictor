#!/usr/bin/env python3
"""
ML Models Module for Lotto Max ML Predictor
Implements Markov chains, Bayesian inference, and ensemble methods from scratch.

Created by: Reza Azizi - May 2026
If you are interested in developing or enhancing an AI automation system—whether for workflows,
data processing, monitoring, or custom integrations—please feel free to reach out to
flowaiautomationsupport@gmail.com. I would be happy to discuss your requirements, propose a tailored solution,
and explore how automation can improve efficiency, accuracy, and scalability for your environment.

"""

import math
import random
from collections import Counter, defaultdict
from itertools import combinations

NUM_POOL = 52
NUMS_PER_DRAW = 7


class MarkovChainModel:
    """First-order and second-order Markov chain models for lottery draws."""

    def __init__(self, order=1):
        self.order = order
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.state_counts = defaultdict(int)
        self.transition_probs = {}
        self.trained = False

    def fit(self, draws):
        """Train Markov chain on historical draws."""
        for i in range(1, len(draws)):
            prev = set(draws[i - 1]['numbers'])
            curr = set(draws[i]['numbers'])

            if self.order == 1:
                for prev_num in prev:
                    self.state_counts[prev_num] += 1
                    for curr_num in curr:
                        self.transition_counts[prev_num][curr_num] += 1

            elif self.order == 2:
                prev_sorted = sorted(prev)
                for pair in combinations(prev_sorted, 2):
                    self.state_counts[pair] += 1
                    for curr_num in curr:
                        self.transition_counts[pair][curr_num] += 1

        # Convert to probabilities
        for state, transitions in self.transition_counts.items():
            total = self.state_counts[state]
            self.transition_probs[state] = {
                k: v / total for k, v in transitions.items()
            }

        self.trained = True
        return self

    def predict(self, last_draw, top_k=15):
        """Predict likely numbers for the next draw."""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")

        predictions = Counter()
        last_nums = set(last_draw)

        if self.order == 1:
            for num in last_nums:
                if num in self.transition_probs:
                    for next_num, prob in self.transition_probs[num].items():
                        if next_num not in last_nums:
                            predictions[next_num] += prob

        elif self.order == 2:
            for pair in combinations(sorted(last_nums), 2):
                if pair in self.transition_probs:
                    for next_num, prob in self.transition_probs[pair].items():
                        if next_num not in last_nums:
                            predictions[next_num] += prob

        return predictions.most_common(top_k)

    def get_stationary_distribution(self):
        """Compute stationary distribution via power iteration."""
        if not self.trained or self.order != 1:
            return {}

        # Initialize uniform distribution
        dist = {n: 1.0 / NUM_POOL for n in range(1, NUM_POOL + 1)}

        # Power iteration
        for _ in range(100):
            new_dist = defaultdict(float)
            for state, transitions in self.transition_probs.items():
                for next_state, prob in transitions.items():
                    new_dist[next_state] += dist.get(state, 0) * prob
            total = sum(new_dist.values())
            if total > 0:
                dist = {k: v / total for k, v in new_dist.items()}
            # Add missing numbers back with small probability
            for n in range(1, NUM_POOL + 1):
                if n not in dist:
                    dist[n] = 1e-6
            total = sum(dist.values())
            dist = {k: v / total for k, v in dist.items()}

        return dict(sorted(dist.items()))


class BayesianModel:
    """Bayesian inference with conjugate Beta priors for lottery number prediction."""

    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.posteriors = {}
        self.trained = False

    def fit(self, draws):
        """Update priors with observed data to get posteriors."""
        total_draws = len(draws)

        for num in range(1, NUM_POOL + 1):
            appearances = sum(1 for d in draws if num in d['numbers'])
            non_appearances = total_draws - appearances

            alpha_post = self.alpha_prior + appearances
            beta_post = self.beta_prior + non_appearances

            self.posteriors[num] = {
                'alpha': alpha_post,
                'beta': beta_post,
                'mean': alpha_post / (alpha_post + beta_post),
                'variance': (alpha_post * beta_post) /
                           ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)),
                'mode': (alpha_post - 1) / (alpha_post + beta_post - 2)
                        if alpha_post > 1 and beta_post > 1 else 0
            }

        self.trained = True
        return self

    def predict(self, top_k=15):
        """Get top numbers by posterior mean."""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")

        ranked = sorted(
            self.posteriors.items(),
            key=lambda x: -x[1]['mean']
        )
        return [(num, data['mean']) for num, data in ranked[:top_k]]

    def credible_interval(self, num, level=0.95):
        """Approximate credible interval using normal approximation."""
        if num not in self.posteriors:
            return None

        post = self.posteriors[num]
        std = math.sqrt(post['variance'])
        z = 1.96 if level == 0.95 else 2.576  # 95% or 99%

        return {
            'lower': max(0, post['mean'] - z * std),
            'upper': min(1, post['mean'] + z * std),
            'level': level
        }

    def bayesian_surprise(self, draw):
        """Compute Bayesian surprise (KL divergence) for a draw."""
        if not self.trained:
            return 0

        prior_mean = self.alpha_prior / (self.alpha_prior + self.beta_prior)
        total_surprise = 0

        for num in draw:
            if num in self.posteriors:
                post_mean = self.posteriors[num]['mean']
                if 0 < post_mean < 1 and 0 < prior_mean < 1:
                    kl = post_mean * math.log(post_mean / prior_mean) + \
                         (1 - post_mean) * math.log((1 - post_mean) / (1 - prior_mean))
                    total_surprise += kl

        return total_surprise

    def james_stein_estimate(self):
        """Apply James-Stein shrinkage to posterior means."""
        if not self.trained:
            return {}

        global_mean = NUMS_PER_DRAW / NUM_POOL
        observations = {n: p['mean'] for n, p in self.posteriors.items()}

        # Shrinkage factor
        sum_sq = sum((m - global_mean) ** 2 for m in observations.values())
        if sum_sq == 0:
            return observations

        shrinkage = max(0, min(1, 1 - (NUM_POOL - 3) / (sum_sq * len(observations))))

        return {
            n: shrinkage * m + (1 - shrinkage) * global_mean
            for n, m in observations.items()
        }


class EnsembleModel:
    """Ensemble of all ML models with configurable weights."""

    def __init__(self, weights=None):
        self.default_weights = {
            'frequency': 0.15,
            'hotcold': 0.20,
            'gap': 0.15,
            'markov': 0.15,
            'bayesian': 0.15,
            'pattern': 0.10,
            'entropy': 0.05,
            'simulation': 0.05
        }
        self.weights = weights or self.default_weights
        self.models = {}
        self.trained = False

    def set_weights(self, weights):
        """Customize model weights."""
        for key, value in weights.items():
            if key in self.default_weights:
                self.weights[key] = value
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def add_model(self, name, model, scores):
        """Add a trained model with pre-computed scores."""
        self.models[name] = {
            'model': model,
            'scores': scores  # dict: {number: score}
        }

    def predict(self, top_k=7):
        """Generate ensemble prediction combining all models."""
        if not self.models:
            raise ValueError("No models added to ensemble")

        ensemble_scores = defaultdict(float)

        for model_name, model_data in self.models.items():
            weight = self.weights.get(model_name, 0)
            scores = model_data['scores']

            # Normalize model scores
            max_score = max(scores.values()) if scores else 1
            if max_score == 0:
                max_score = 1

            for num, score in scores.items():
                ensemble_scores[num] += weight * (score / max_score)

        # Rank
        ranked = sorted(ensemble_scores.items(), key=lambda x: -x[1])

        # Select top_k
        selected = [num for num, score in ranked[:top_k]]
        selected.sort()

        return {
            'selected': selected,
            'ensemble_scores': dict(ensemble_scores),
            'ranked': ranked,
            'confidence': self._compute_confidence(ranked, top_k)
        }

    def _compute_confidence(self, ranked, top_k):
        """Compute confidence level for the prediction."""
        top_scores = [score for _, score in ranked[:top_k]]
        if not top_scores:
            return {'level': 'Low', 'percentage': 0}

        avg_score = sum(top_scores) / len(top_scores)
        max_possible = ranked[0][1] if ranked else 1
        confidence_pct = min(100, (avg_score / max_possible * 100) if max_possible > 0 else 0)

        if confidence_pct >= 80:
            level = 'Very High'
        elif confidence_pct >= 65:
            level = 'High'
        elif confidence_pct >= 45:
            level = 'Medium'
        else:
            level = 'Low'

        return {
            'level': level,
            'percentage': round(confidence_pct, 1)
        }


class GapProbabilityModel:
    """Probability model based on gap analysis."""

    def __init__(self, decay_rate=1.0):
        self.decay_rate = decay_rate
        self.gap_stats = {}
        self.trained = False

    def fit(self, draws):
        """Compute gap statistics from historical draws."""
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

        for num in range(1, NUM_POOL + 1):
            gap_list = gaps.get(num, [])
            avg_gap = sum(gap_list) / len(gap_list) if gap_list else len(draws) / NUMS_PER_DRAW
            self.gap_stats[num] = {
                'current_gap': current_gap[num],
                'avg_gap': avg_gap,
                'gap_history': gap_list
            }

        self.trained = True
        return self

    def predict(self, top_k=15):
        """Predict numbers based on gap probability."""
        if not self.trained:
            raise ValueError("Model must be trained")

        scores = {}
        for num in range(1, NUM_POOL + 1):
            stats = self.gap_stats[num]
            gap = stats['current_gap']
            avg = stats['avg_gap']

            # Exponential model: probability increases with gap
            prob = 1 - math.exp(-self.decay_rate * gap / avg) if avg > 0 else 0
            scores[num] = prob

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]


if __name__ == '__main__':
    # Demo usage
    print("ML Models Module - Lotto Max ML Predictor")
    print("=" * 50)

    # Generate sample data
    random.seed(42)
    sample_draws = []
    for _ in range(100):
        nums = sorted(random.sample(range(1, 53), 7))
        bonus_candidates = [n for n in range(1, 53) if n not in nums]
        sample_draws.append({
            'numbers': nums,
            'bonus': random.choice(bonus_candidates)
        })
    random.seed()

    # Test Markov Chain
    print("\n--- Markov Chain Model ---")
    mc = MarkovChainModel(order=1)
    mc.fit(sample_draws)
    predictions = mc.predict(sample_draws[-1]['numbers'])
    print(f"Top predictions: {predictions[:5]}")

    # Test Bayesian
    print("\n--- Bayesian Model ---")
    bayes = BayesianModel()
    bayes.fit(sample_draws)
    top_bayes = bayes.predict(5)
    print(f"Top posterior means: {top_bayes}")

    # Test Gap Model
    print("\n--- Gap Probability Model ---")
    gap_model = GapProbabilityModel()
    gap_model.fit(sample_draws)
    top_gaps = gap_model.predict(5)
    print(f"Top gap probabilities: {top_gaps}")
