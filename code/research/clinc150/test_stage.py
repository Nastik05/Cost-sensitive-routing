import numpy as np
from common import OLD_CODE
from run_routers import select_lambda
from imbalance_study import metrics
from imbalance_statistics import bootstrap_profile


def test_criteria_use_the_same_grid_and_fixed_tie_rule():
    grid = [{"strength": .001, "tune_mse": .1, "tune_accuracy_0.2": .8},
            {"strength": .01, "tune_mse": .12, "tune_accuracy_0.2": .85},
            {"strength": .1, "tune_mse": .2, "tune_accuracy_0.2": .85},
            {"strength": 1., "tune_mse": .1, "tune_accuracy_0.2": .7}]
    assert select_lambda(grid,"mse",.2)["strength"] == 1.
    assert select_lambda(grid,"accuracy",.2)["strength"] == .1


def test_smaller_mse_need_not_give_better_quota_accuracy():
    W, S = np.array([0,1,1,1]), np.array([1,0,1,1])
    a = metrics(np.array([0.,.1,0.,0.]),W,S,np.arange(4),.25)
    b = metrics(np.array([1.,.9,.9,.9]),W,S,np.arange(4),.25)
    assert a["gain_mse"] < b["gain_mse"] and a["accuracy"] < b["accuracy"]


def test_positive_affine_scores_preserve_actions_not_mse():
    score = np.array([.2,-.3,.4,.1])
    W, S = np.array([0,1,1,0]), np.array([1,0,1,0])
    a = metrics(score,W,S,np.arange(4),.5)
    b = metrics(2*score+.7,W,S,np.arange(4),.5)
    assert a["accuracy"] == b["accuracy"] and a["rescues"] == b["rescues"]
    assert a["gain_mse"] != b["gain_mse"]


def test_bootstrap_works_for_different_test_size():
    rng = np.random.default_rng(112)
    n = 37
    score, D, tie = rng.normal(size=n), rng.choice([-1,0,1],size=n), rng.random(n)
    counts = rng.multinomial(n, np.full(n,1/n),size=40)
    computed = bootstrap_profile(score,D,tie,counts,.2)
    literal = []
    for c in counts:
        ids = np.repeat(np.arange(n),c)
        selected = ids[np.lexsort((tie[ids],-score[ids]))[:int(.2*n)]]
        literal.append([(D[selected]==1).sum(),(D[selected]==-1).sum()])
    np.testing.assert_array_equal(computed,literal)


def test_prespecified_family_and_difference_of_differences():
    from infer import specs
    from collections import Counter
    rows = specs()
    assert Counter(r["kind"] for r in rows) == {"simple_vs_weak":42,"interaction_vs_weak":14,"gain_simple":126,
        "gain_interaction":42,"correction":16,"selector":12,"criterion_effect":24,"criterion_interaction":7}
    def value(arm,c,m):
        if m == "direct":
            return {("natural_raw","mse"):.9,("harm8_raw","mse"):.8,
                    ("natural_raw","accuracy"):.91,("harm8_raw","accuracy"):.85}[arm,c]
        return .88
    row = next(r for r in rows if r["kind"]=="criterion_interaction" and r["method"]=="direct")
    actual = sum(w*value(a,c,m) for a,c,m,w in row["terms"])
    np.testing.assert_allclose(actual,.04,atol=1e-12)
