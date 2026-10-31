"""Pure executed AUC identities and conditional label-scenario extrema."""
import numpy as np
from scipy.stats import rankdata

def auc_counts(y,f,counts):
    order=np.argsort(f,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(f[order]))+1]
    yy=y[order];cc=counts[:,order]
    pos=np.add.reduceat(cc*yy,starts,axis=1);neg=np.add.reduceat(cc*(1-yy),starts,axis=1)
    den=pos.sum(axis=1)*neg.sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):return (pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(axis=1)/den


def exact_auc_label_bounds(score,lo,hi):
    n=len(score);ranks=rankdata(score);base=int(lo.sum());known=ranks[lo==1].sum();amb=np.sort(ranks[(lo==0)&(hi==1)]);m=len(amb);vals=[]
    for k in range(m+1):
        pos=base+k
        if pos in [0,n]:continue
        den=pos*(n-pos);shift=pos*(pos+1)/2
        vals.append(((known+amb[:k].sum()-shift)/den,(known+(amb[-k:].sum() if k else 0)-shift)/den))
    return min(v[0] for v in vals),max(v[1] for v in vals)
