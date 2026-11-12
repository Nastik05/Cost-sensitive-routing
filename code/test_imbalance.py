"""Mechanistic and numerical checks of all estimators and imbalance metrics."""
import numpy as np
import pytest
from scipy.optimize import check_grad
from imbalance_estimators import METHODS, Classifier, fit_estimator, soft_objective
from imbalance_study import REGIMES, probabilities, draw_ids, metrics, choose


@pytest.mark.parametrize("method", METHODS)
def test_unit_weights_preserve_each_estimator(method):
    rng = np.random.default_rng(51)
    X = rng.normal(size=(80, 5))
    W, S = rng.integers(0, 2, size=(2, len(X)))
    raw = fit_estimator(method, X, W, S, .01)
    weighted = fit_estimator(method, X, W, S, .01, np.ones(len(X)), 1.)
    np.testing.assert_allclose(raw.predict(X), weighted.predict(X), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("outcome", [(0, 0), (1, 1), (0, 1), (1, 0)])
def test_all_constant_joint_outcomes(method, outcome):
    X = np.arange(60).reshape(20, 3)/60
    W, S = [np.full(20, x) for x in outcome]
    model = fit_estimator(method, X, W, S, .1, np.linspace(.2, 2, 20), 1.)
    np.testing.assert_allclose(model.predict(X), outcome[1]-outcome[0], atol=1e-8)


def test_soft_weighted_gradient():
    rng = np.random.default_rng(54)
    X, theta = rng.normal(size=(35, 4)), rng.normal(size=5)
    target = rng.choice([0., .5, 1.], size=35)
    weights = rng.uniform(.1, 4, size=35)
    f = lambda t: soft_objective(t, X, target, .03, weights)[0]
    g = lambda t: soft_objective(t, X, target, .03, weights)[1]
    assert check_grad(f, g, theta) < 1e-6


def test_full_and_conditional_probability_correction():
    D = np.array([-1, -1, 0, 0, 0, 0, 1])
    loss = np.array([2., 1., 5., 6., 2., 1., 8.])
    nonneutral = D != 0
    for regime in REGIMES:
        p = probabilities(D, regime)
        w = 1/(len(D)*p)
        np.testing.assert_allclose(p@(w*loss), loss.mean())
        scale = p[nonneutral].sum()/nonneutral.mean()
        conditional_p = p[nonneutral]/p[nonneutral].sum()
        np.testing.assert_allclose(conditional_p@(w[nonneutral]*scale*loss[nonneutral]), loss[nonneutral].mean())
        small, sw, ss = draw_ids(np.arange(len(D)), D, regime, 2, 20)
        large, lw, ls = draw_ids(np.arange(len(D)), D, regime, 2, 100)
        np.testing.assert_array_equal(small, large[:20])
        np.testing.assert_array_equal(sw, lw[:20])
        np.testing.assert_allclose(ss, scale)


def test_equal_accuracy_can_have_different_rescue_precision_and_recall():
    W = np.array([0, 1, 0, 1, 1, 1])
    S = np.array([1, 0, 1, 0, 1, 1])
    a = metrics(np.array([6, 5, 4, 3, 2, 1]), W, S, np.arange(6), 2/6)
    b = metrics(np.array([1, 2, 3, 4, 6, 5]), W, S, np.arange(6), 2/6)
    assert a["accuracy"] == b["accuracy"]
    assert a["rescue_precision"] == .5 and b["rescue_precision"] == 0
    assert a["rescue_recall"] == .5 and b["rescue_recall"] == 0
    assert a["harm_rate"] == .5 and b["harm_rate"] == 0


def test_metrics_identities_and_undefined_recall():
    W = np.array([0, 1, 0, 1, 1, 1])
    S = np.array([1, 0, 1, 0, 1, 1])
    r = metrics(np.arange(6), W, S, np.arange(6), .5)
    np.testing.assert_allclose(r["accuracy"]-W.mean(), (r["rescues"]-r["harms"])/6)
    np.testing.assert_allclose(r["rescue_f1"], 2*r["rescue_precision"]*r["rescue_recall"]/(r["rescue_precision"]+r["rescue_recall"]) if r["rescues"] else 0)
    assert r["missed_rescues"]+r["rescues"] == 2
    assert r["avoided_harms"]+r["harms"] == 2
    no_event = metrics(np.arange(6), W, W, np.arange(6), .5)
    assert np.isnan(no_event["rescue_recall"]) and np.isnan(no_event["harm_recall"])


def test_missing_classes_map_to_correct_columns():
    X = np.arange(30).reshape(10, 3)/30
    model = Classifier(X, np.array([0, 3]*5), .1, 4, np.ones(10))
    p = model.predict(X)
    np.testing.assert_array_equal(p[:, [1, 2]], np.zeros((10, 2)))
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_bootstrap_profiles_match_literal_repeated_batches():
    from imbalance_statistics import bootstrap_profile
    rng = np.random.default_rng(55)
    score, D, tie = np.array([2., 2., 1., 0., 0.]), np.array([1, -1, 1, 0, -1]), rng.random(5)
    counts = rng.multinomial(5, np.full(5, .2), size=80)
    for fraction in [0., .2, .4, 1.]:
        actual = bootstrap_profile(score, D, tie, counts, fraction)
        expected = []
        for c in counts:
            ids = np.repeat(np.arange(5), c)
            picked = np.lexsort((tie[ids], -score[ids]))[:int(5*fraction)]
            expected.append([(D[ids][picked] == 1).sum(), (D[ids][picked] == -1).sum()])
        np.testing.assert_array_equal(actual, expected)


def test_bootstrap_prefix_fallback_is_exact():
    from imbalance_statistics import bootstrap_profile
    n = 1000
    score = np.arange(n, dtype=float)[::-1]
    D = np.ones(n, dtype=int)
    D[-1] = -1
    counts = np.zeros((2, n), dtype=int)
    counts[0, -1] = n  # Every copy lies outside the initial top-k prefix.
    counts[1, 0] = n
    np.testing.assert_array_equal(bootstrap_profile(score, D, np.arange(n), counts), [[0, 200], [200, 0]])


def test_interaction_is_not_automatically_a_supported_crossover():
    import pandas as pd
    from analyze_imbalance import interpret
    rows = []
    for pair, natural_low in [(0, -.2), (1, .2)]:
        common = {"pair_id": pair, "method": "direct", "reference": "events", "gain_only": True, "adjusted_p": .01}
        for kind, arm, metric, point, low, high in [
            ("simple", "natural_raw", "accuracy", .5, natural_low, .8),
            ("simple", "harm8_raw", "accuracy", -1., -1.5, -.5),
            ("interaction", "harm8_raw", "accuracy", -1.5, -2., -1.),
            ("profile", "natural_raw", "rescue_precision", 10., 5., 15.),
            ("profile", "natural_raw", "harm_rate", 9., 4., 14.)]:
            rows.append({**common, "kind": kind, "arm": arm, "metric": metric, "difference_pp": point, "low": low, "high": high})
    _, interactions, profiles = interpret(pd.DataFrame(rows))
    assert interactions.significant.all() and interactions.point_ranking_flip.all()
    assert interactions.supported_crossover.tolist() == [False, True]
    assert profiles.profile_without_accuracy_detection.tolist() == [True, False]


def test_soft_loss_is_exactly_corrected_on_full_pool():
    rng = np.random.default_rng(57)
    X = rng.normal(size=(12, 3))
    D = np.array([-1, 0, 1]*4)
    theta = rng.normal(size=4)
    p = probabilities(D, "harm8")
    w = 1/(len(D)*p)
    # Deterministic enumeration of the expected weighted per-draw loss/gradient.
    values, gradients = [], []
    for i in range(len(D)):
        value, gradient = soft_objective(theta, X[i:i+1], (D[i:i+1]+1)/2, 0., w[i:i+1])
        values.append(value)
        gradients.append(gradient)
    natural_value, natural_gradient = soft_objective(theta, X, (D+1)/2, 0., np.ones(len(D)))
    np.testing.assert_allclose(p@values, natural_value)
    np.testing.assert_allclose(p@np.stack(gradients), natural_gradient)
