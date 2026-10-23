"""Checks for the routing experiment and counterfactual outcome conventions."""
import json
import numpy as np
import pandas as pd
import pytest
from experiment import ROOT, Binary, digest, fit, predict


def test_cache_and_reference_hashes():
    m = json.loads((ROOT / "data/manifest.json").read_text())
    assert digest(ROOT / "data/banking77.npz") == m["cache_sha256"]
    assert digest(ROOT / "data/reference.csv.gz") == m["reference_sha256"]
    with np.load(ROOT / "data/banking77.npz") as z:
        assert len(z["row_id"]) == len(set(z["row_id"])) == 6080
        assert not (z["split"] == "expert_train").any()
        for split,n in [("router_fit",2000),("policy_dev",1000),("test",3080)]:
            assert (z["split"] == split).sum() == n
        assert z["clean_test"].sum() == 3072


@pytest.mark.parametrize("method",["weak_error","direct","events"])
def test_intercept_only_recovers_mean_gain(method):
    X = np.zeros((100,3))
    W = np.r_[np.zeros(30),np.ones(10),np.ones(40),np.zeros(20)].astype(int)
    S = np.r_[np.ones(30),np.zeros(10),np.ones(40),np.zeros(20)].astype(int)
    np.testing.assert_allclose(predict(fit(method,X,W,S,.1),X),.2,atol=1e-5)


@pytest.mark.parametrize("outcome",[0,1])
def test_binary_missing_class_is_exact_constant(outcome):
    X = np.random.default_rng(0).normal(size=(40,4))
    model = Binary(X,np.full(40,outcome),.1)
    np.testing.assert_array_equal(model.predict(X),np.full(40,outcome))


def test_alternative_decision_sign():
    W,S = np.array([0,1,0,1]),np.array([1,0,0,1])
    r = np.array([True,False,True,False])
    chosen,alternative = np.where(r,S,W),np.where(r,W,S)
    np.testing.assert_array_equal(chosen-alternative,[1,1,0,0])
    np.testing.assert_array_equal(chosen-alternative,np.where(r,S-W,W-S))


def test_generated_profile_totals_and_sign():
    q = pd.read_csv(ROOT.parent / "paper/generated/outcomes.csv")
    for method,group in q.groupby("method"):
        np.testing.assert_allclose(group.escalate_yes.sum(),616)
        np.testing.assert_allclose(group.escalate_no.sum(),2464)
    c = q[q.method == "confidence"].set_index("outcome")
    assert c.loc[1,"escalate_yes"] == 123
    assert c.loc[-1,"escalate_yes"] == 102
    assert c.loc[1,"escalate_no"] == 138
    assert c.loc[-1,"escalate_no"] == 60
