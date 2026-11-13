import numpy as np
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from common import from_predictions


def test_matches_sklearn_including_missing_classes():
    truth=np.array([0,0,0,1,1,2,2,2,2])
    pred=np.array([0,0,1,1,0,0,0,3,3])
    result,_,_,_=from_predictions(truth,pred,5)
    for average in ["micro","macro","weighted"]:
        p,r,f,_=precision_recall_fscore_support(truth,pred,labels=np.arange(5),average=average,zero_division=0)
        np.testing.assert_allclose([result[average+"_precision"],result[average+"_recall"],result[average+"_f1"]],[p,r,f])
    assert result["accuracy"]==accuracy_score(truth,pred)


def test_general_micro_and_weighted_recall_identity():
    rng=np.random.default_rng(2026111301)
    for _ in range(100):
        truth,pred=rng.integers(0,7,size=(2,173))
        r,_,_,_=from_predictions(truth,pred,7)
        np.testing.assert_allclose([r["micro_precision"],r["micro_recall"],r["micro_f1"],r["weighted_recall"]],r["accuracy"])


def test_balanced_test_additional_identities():
    truth=np.repeat(np.arange(4),10)
    pred=np.random.default_rng(2).integers(0,4,size=40)
    r,_,_,_=from_predictions(truth,pred,4)
    np.testing.assert_allclose(r["macro_recall"],r["accuracy"])
    np.testing.assert_allclose(r["macro_precision"],r["weighted_precision"])
    np.testing.assert_allclose(r["macro_f1"],r["weighted_f1"])


def test_equal_accuracy_different_macro_f1():
    truth=np.array([0,0,1,1,2,2])
    a,_,_,_=from_predictions(truth,np.array([0,0,1,1,0,0]),3)
    b,_,_,_=from_predictions(truth,np.array([0,0,1,0,2,1]),3)
    assert a["accuracy"]==b["accuracy"]
    assert abs(a["macro_f1"]-b["macro_f1"])>.09
    assert a["macro_recall"]==b["macro_recall"]


def test_wrong_to_wrong_changes_f1_without_changing_correctness():
    truth=np.array([0,0,1,1,2,2])
    a=np.array([0,0,1,0,2,1])
    b=np.array([0,0,1,2,2,1])
    np.testing.assert_array_equal(a==truth,b==truth)
    ra,_,_,_=from_predictions(truth,a,3)
    rb,_,_,_=from_predictions(truth,b,3)
    assert ra["accuracy"]==rb["accuracy"] and ra["macro_f1"]!=rb["macro_f1"]


def test_stratified_bootstrap_and_partial_duplicate_boundary():
    from bootstrap import multiclass_bootstrap,stratified_counts
    rng=np.random.default_rng(3)
    truth=np.repeat(np.arange(3),6)
    first=rng.integers(0,3,size=len(truth))
    second=rng.integers(0,3,size=len(truth))
    scores=rng.choice([0.,1.,2.],size=len(truth))
    tie=rng.random(len(truth))
    counts=stratified_counts(truth,rng,70)
    for label in range(3):
        np.testing.assert_array_equal(counts[:,truth==label].sum(1),np.full(70,6))
    for fraction in [0.,.2,.5,1.]:
        actual=multiclass_bootstrap(scores,truth,first,second,tie,counts,3,fraction=fraction)
        expected=[]
        for c in counts:
            ids=np.repeat(np.arange(len(truth)),c)
            selected=np.lexsort((tie[ids],-scores[ids]))[:int(fraction*len(truth))]
            pred=first[ids].copy()
            pred[selected]=second[ids][selected]
            values,_,_,_=from_predictions(truth[ids],pred,3)
            expected.append([values[m] for m in ["accuracy","macro_precision","macro_f1"]])
        np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_bootstrap_full_fallback_is_exact():
    from bootstrap import multiclass_bootstrap
    n=1000
    truth=np.arange(n)%2
    first=truth.copy()
    second=1-truth
    scores=np.arange(n,dtype=float)[::-1]
    counts=np.zeros((1,n),dtype=int)
    counts[0,-2:]=500
    actual=multiclass_bootstrap(scores,truth,first,second,np.arange(n),counts,2,.2)
    ids=np.repeat(np.arange(n),counts[0])
    pred=first[ids].copy()
    chosen=np.lexsort((ids,-scores[ids]))[:200]
    pred[chosen]=second[ids][chosen]
    values,_,_,_=from_predictions(truth[ids],pred,2)
    np.testing.assert_allclose(actual[0],[values[m] for m in ["accuracy","macro_precision","macro_f1"]])
