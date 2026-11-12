"""All eight original router estimators, with explicit loss-weight controls."""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge

METHODS = ["weak_error", "direct", "marginal", "events", "ternary", "joint", "hurdle", "soft_gain"]
STRENGTHS = [.001, .01, .1, 1.]


class Classifier:
    def __init__(self, X, y, strength, n_classes, sample_weight=None):
        self.n_classes = n_classes
        self.classes = np.unique(y).astype(int)
        assert len(y) and self.classes.min() >= 0 and self.classes.max() < n_classes
        self.model = None
        if len(self.classes) > 1:
            self.model = LogisticRegression(C=1/(len(y)*strength), solver="lbfgs", max_iter=1000, tol=1e-7)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        p = np.zeros((len(X), self.n_classes))
        if self.model is None:
            p[:, self.classes[0]] = 1
        else:
            p[:, self.model.classes_.astype(int)] = self.model.predict_proba(X)
        return p

    @property
    def parameter_count(self):
        return 0 if self.model is None else self.model.coef_.size+self.model.intercept_.size


def soft_objective(theta, X, target, strength, sample_weight):
    logits = X@theta[:-1]+theta[-1]
    residual = sample_weight*(expit(logits)-target)
    value = np.mean(sample_weight*(np.logaddexp(0, logits)-target*logits))+strength/2*(theta[:-1]@theta[:-1])
    gradient = np.r_[X.T@residual/len(X)+strength*theta[:-1], residual.mean()]
    return value, gradient


class SoftGain:
    def __init__(self, X, D, strength, sample_weight=None):
        self.constant = float(D[0]) if np.all(D == D[0]) else None
        self.weights = None
        if self.constant is not None:
            return
        weights = np.ones(len(X)) if sample_weight is None else sample_weight
        result = minimize(soft_objective, np.zeros(X.shape[1]+1), args=(X, (D+1)/2, strength, weights),
                          jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7})
        if not result.success:
            raise RuntimeError(f"Soft gain fit failed: {result.message}")
        self.weights = result.x

    def predict(self, X):
        if self.constant is not None:
            return np.full(len(X), self.constant)
        return 2*expit(X@self.weights[:-1]+self.weights[-1])-1

    @property
    def parameter_count(self):
        return 0 if self.weights is None else len(self.weights)


@dataclass
class GainEstimator:
    method: str
    models: list
    offset: float = 0.

    def predict(self, X):
        if self.method in ["direct", "soft_gain"]:
            return self.models[0].predict(X)
        if self.method == "weak_error":
            return self.models[0].predict(X)[:, 1]-self.offset
        if self.method in ["marginal", "events"]:
            return self.models[0].predict(X)[:, 1]-self.models[1].predict(X)[:, 1]
        if self.method == "ternary":
            p = self.models[0].predict(X)
            return p[:, 2]-p[:, 0]
        if self.method == "joint":
            p = self.models[0].predict(X)
            return p[:, 1]-p[:, 2]
        if self.method == "hurdle":
            if self.models[1] is None:
                return np.zeros(len(X))
            occurrence = self.models[0].predict(X)[:, 1]
            direction = self.models[1].predict(X)[:, 1]
            return occurrence*(2*direction-1)
        raise ValueError(self.method)

    @property
    def parameter_count(self):
        if self.method == "direct":
            return self.models[0].coef_.size+1
        return sum(m.parameter_count for m in self.models if m is not None)+(self.method == "weak_error")


def fit_estimator(method, X, W, S, strength, sample_weight=None, conditional_scale=1.):
    X = np.asarray(X, dtype=float)
    W, S = np.asarray(W, dtype=int), np.asarray(S, dtype=int)
    assert len(X) == len(W) == len(S) and len(X) > 0
    D = S-W
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        assert sample_weight.shape == W.shape and (sample_weight > 0).all() and np.isfinite(sample_weight).all()
    clf = lambda y, k=2: Classifier(X, y.astype(int), strength, k, sample_weight)
    if method == "direct":
        return GainEstimator(method, [Ridge(alpha=len(X)*strength).fit(X, D, sample_weight=sample_weight)])
    if method == "weak_error":
        return GainEstimator(method, [clf(1-W)], float(np.average(1-S, weights=sample_weight)))
    if method == "marginal":
        return GainEstimator(method, [clf(S), clf(W)])
    if method == "events":
        return GainEstimator(method, [clf(D == 1), clf(D == -1)])
    if method == "ternary":
        return GainEstimator(method, [clf(D+1, 3)])
    if method == "joint":
        return GainEstimator(method, [clf(2*W+S, 4)])
    if method == "hurdle":
        discordant = D != 0
        weights = None if sample_weight is None else sample_weight[discordant]*conditional_scale
        direction = Classifier(X[discordant], (D[discordant] == 1).astype(int), strength, 2, weights) if discordant.any() else None
        return GainEstimator(method, [clf(discordant), direction])
    if method == "soft_gain":
        return GainEstimator(method, [SoftGain(X, D, strength, sample_weight)])
    raise ValueError(method)
