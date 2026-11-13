"""Class-stratified paired bootstrap with exact rerouting of duplicated requests."""
import numpy as np


def batch_count(labels,weights,k):
    labels=np.asarray(labels,dtype=np.int64)
    weights=np.asarray(weights)
    index=labels[None,:]+np.arange(len(weights),dtype=np.int64)[:,None]*k
    return np.bincount(index.ravel(),weights=weights.ravel(),minlength=len(weights)*k).reshape(len(weights),k)


def prepare_base(truth,first,counts,k):
    assert np.all(counts.sum(axis=1)==len(truth))
    return {"support":batch_count(truth,counts,k),"predicted":batch_count(first,counts,k),
            "tp":batch_count(truth,counts*(first==truth)[None,:],k)}


def multiclass_bootstrap(score,truth,first,second,tie,counts,k,fraction=.2,base=None,chunk=128):
    n=len(truth)
    calls=int(fraction*n)
    if base is None:
        base=prepare_base(truth,first,counts,k)
    order=np.lexsort((tie,-score))
    prefix=min(n,max(calls+64,calls+int(8*np.sqrt(calls+1))))
    output=np.empty((len(counts),3))
    for begin in range(0,len(counts),chunk):
        end=min(begin+chunk,len(counts))
        ids=order[:prefix]
        c=counts[begin:end,ids].astype(np.int32)
        if np.any(c.sum(1)<calls):
            ids=order
            c=counts[begin:end,ids].astype(np.int32)
        total=np.cumsum(c,axis=1,dtype=np.int32)
        chosen=c-np.clip(total-calls,0,c)
        assert np.all(chosen.sum(1)==calls)
        y,w,s=truth[ids],first[ids],second[ids]
        changed=w!=s
        pred=base["predicted"][begin:end].copy()
        pred-=batch_count(w[changed],chosen[:,changed],k)
        pred+=batch_count(s[changed],chosen[:,changed],k)
        rescue=(w!=y)&(s==y)
        harm=(w==y)&(s!=y)
        tp=base["tp"][begin:end].copy()
        tp+=batch_count(y[rescue],chosen[:,rescue],k)
        tp-=batch_count(y[harm],chosen[:,harm],k)
        support=base["support"][begin:end]
        assert (tp>=0).all() and (pred>=0).all() and (tp<=support).all() and (tp<=pred).all()
        np.testing.assert_array_equal(pred.sum(1),np.full(len(pred),n))
        p=np.divide(tp,pred,out=np.zeros_like(tp),where=pred>0)
        f=np.divide(2*tp,pred+support,out=np.zeros_like(tp),where=pred+support>0)
        output[begin:end,0]=tp.sum(1)/n
        output[begin:end,1]=p.mean(1)
        output[begin:end,2]=f.mean(1)
    return output


def stratified_counts(truth,rng,repeats):
    counts=np.zeros((repeats,len(truth)),dtype=np.int16)
    for label in np.unique(truth):
        ids=np.flatnonzero(truth==label)
        counts[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=repeats)
    return counts
