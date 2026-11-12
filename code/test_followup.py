"""Checks of weighting identities, selection separation and resampled quotas."""
import numpy as np
from sklearn.linear_model import Ridge

from experiment import fit, predict
from followup import sampling_probabilities, sample_pool, partition_dev, choose_method, evaluate
from followup_stats import bootstrap_gain, simultaneous_intervals


def test_exact_weighted_risk_on_finite_pool():
    D = np.array([-1, 0, 0, 0, 1])
    loss = np.array([9., 2., 5., 1., 3.])
    for regime in ["natural", "rescue_rich", "harm_rich", "discordant_rich"]:
        p = sampling_probabilities(D, regime)
        w = 1/(len(D)*p)
        np.testing.assert_allclose(np.sum(p*w*loss), loss.mean())
        np.testing.assert_allclose(np.sum(p*w), 1)
    p = sampling_probabilities(D, "harm_rich")
    assert abs(p@loss-loss.mean()) > 1  # Uncorrected risk really changes.


def test_nested_draws_and_exact_weights():
    p = sampling_probabilities(np.array([-1, 0, 1, 0]), "harm_rich")
    small, sw = sample_pool(np.arange(4), p, 7, 20)
    large, lw = sample_pool(np.arange(4), p, 7, 100)
    np.testing.assert_array_equal(small, large[:20])
    np.testing.assert_array_equal(sw, lw[:20])
    np.testing.assert_allclose(sw, 1/(4*p[small]))


def test_tuning_and_selector_are_disjoint():
    for seed in range(30):
        for n in [200, 500, 1000]:
            tune, select = partition_dev(np.arange(1000), seed, n)
            assert len(tune) == n//2 and len(select) == 500
            assert not set(tune) & set(select)


def test_weighted_ridge_matches_normal_equations():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(40, 3))
    W, S = rng.integers(0, 2, size=(2, 40))
    w = rng.uniform(.2, 3, 40)
    strength = .03
    A = np.column_stack([X, np.ones(len(X))])
    penalty = np.diag([len(X)*strength]*3+[0])
    theta = np.linalg.solve(A.T@(w[:, None]*A)+penalty, A.T@(w*(S-W)))
    model = fit("direct", X, W, S, strength, w)
    np.testing.assert_allclose(predict(model, X), A@theta, atol=1e-12)


def test_paired_ridge_difference_identity():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 4))
    W, S = rng.integers(0, 2, size=(2, 40))
    direct = Ridge(alpha=3).fit(X, S-W).predict(X)
    marginal = Ridge(alpha=3).fit(X, S).predict(X)-Ridge(alpha=3).fit(X, W).predict(X)
    np.testing.assert_allclose(direct, marginal, atol=1e-12)


def test_bootstrap_matches_expanded_batch_with_ties():
    rng = np.random.default_rng(15)
    score = np.array([1., 1., 2., 0., 0.])
    D = np.array([1, -1, 0, 1, -1])
    tie = rng.random(5)
    counts = rng.multinomial(5, np.full(5, .2), size=100)
    for fraction in [0., .2, .4, 1.]:
        actual = bootstrap_gain(score, D, tie, counts, fraction)
        expected = []
        for c in counts:
            ids = np.repeat(np.arange(5), c)
            picked = np.lexsort((tie[ids], -score[ids]))[:int(5*fraction)]
            expected.append(D[ids][picked].sum()/5)
        np.testing.assert_allclose(actual, expected)


def test_selector_uses_validation_accuracy_and_simple_tie_break():
    W, S = np.array([0, 1, 1, 0]), np.array([1, 0, 1, 0])
    scores = {m: np.zeros(4) for m in ["confidence", "direct", "weak_error", "events"]}
    tie = np.arange(4)
    assert choose_method(scores, W, S, tie, .25) == "confidence"
    scores["confidence"] = np.array([0., 2., 0., 0.])
    assert choose_method(scores, W, S, tie, .25) == "direct"
    assert evaluate(scores["direct"], W, S, tie, .25)["calls"] == 1


def test_simultaneous_intervals_cover_zero_for_null_draws():
    draws = np.random.default_rng(18).normal(size=(2000, 14))
    result = simultaneous_intervals(np.zeros(14), draws)
    assert (result["low"] < 0).all() and (result["high"] > 0).all()
    np.testing.assert_array_equal(result["adjusted_p"], np.ones(14))


def test_equal_discordant_enrichment_can_reverse_ranking():
    p = np.array([[.2, 0., .8], [.4, .15, .45]])  # rescue, harm, neutral
    q = p * np.array([4., 4., 1.])
    q /= q.sum(axis=1, keepdims=True)
    assert p[0, 0]-p[0, 1] < p[1, 0]-p[1, 1]
    assert q[0, 0]-q[0, 1] > q[1, 0]-q[1, 1]
    np.testing.assert_allclose(q[:, 0]-q[:, 1], [.5, 1/2.65])


def test_topk_bound_against_exhaustive_oracle():
    from itertools import combinations
    rng = np.random.default_rng(19)
    for _ in range(30):
        gain = rng.uniform(-1, 1, 8)
        estimate = gain+rng.uniform(-.2, .2, 8)
        eps = np.max(np.abs(estimate-gain))
        for k in range(1, 8):
            oracle = max(gain[list(c)].sum() for c in combinations(range(8), k))
            selected = gain[np.argsort(-estimate)[:k]].sum()
            assert -1e-12 <= oracle-selected <= 2*k*eps+1e-12
