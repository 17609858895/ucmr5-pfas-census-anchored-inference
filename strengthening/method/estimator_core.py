"""Pure executed variable-completion estimator identities."""
import itertools
import numpy as np

def dual_score(u,P):
    return .5*(.25*u/u.mean()+.75*P*u/(P*u).mean())+.5*(.25+.75*P/P.mean())


def estimate(y,yt,f,a,pi,I,J,q,mode='tuned'):
    idx=np.flatnonzero(I);j=J[idx];w=1/pi[idx];aa=a[idx];ff=f[idx];tt=yt[idx]
    if mode=='tuned':
        yb=np.sum(w*aa*tt)/np.sum(w*aa);fb=np.sum(w*aa*ff)/np.sum(w*aa)
        lam=np.clip(np.sum(w*aa**2*(tt-yb)*(ff-fb))/max(np.sum(w*aa**2*(ff-fb)**2),1e-12),0,2)
    elif mode=='one':lam=1.
    else:lam=0.
    e=aa*(tt-lam*ff);d=aa*(y[idx]-tt);A=a.sum()
    correction=np.sum(w[j]*d[j]/q[idx[j]])
    th=(lam*np.sum(a*f)+np.sum(w*e)+correction)/A
    v1=np.sum(w*(w-1)*e**2);wj=w[j];ej=e[j];dj=d[j];qj=q[idx[j]]
    v2=np.sum(wj/qj*((wj-1)*(2*ej*dj+dj**2)+wj*(1/qj-1)*dj**2))
    v=max(v1+v2,0)/A**2
    # A separate nonnegative unbiased fixed-coefficient decomposition.
    v_first=np.sum((1-pi[idx[j]])/(pi[idx[j]]**2*qj)*(ej+dj)**2)/A**2
    v_second=np.sum((1-qj)/(pi[idx[j]]**2*qj**2)*dj**2)/A**2
    miss=np.sum((d>0)&j)
    return (th,np.sqrt(v),max(len(idx)-2,1),len(idx),lam,int(j.sum()),int(miss),correction/A,v1/A**2,v2/A**2,v_first+v_second)


def variance_test():
    y=np.array([1.,1.]);yt=np.array([0.,1.]);f=np.array([.2,.7]);a=np.array([1.,3.]);pi=np.array([.3,.7]);q=np.array([.2,.6]);vals=[]
    for states in itertools.product(range(3),repeat=2):
        I=np.array(states)>0;J=np.array(states)==2
        prob=np.prod([1-pi[i] if s==0 else pi[i]*(1-q[i]) if s==1 else pi[i]*q[i] for i,s in enumerate(states)])
        if not I.any():
            # fixed lambda=1 remains defined even when the phase-one sample is empty
            th=np.sum(a*f)/a.sum();v=0
        else:
            z=estimate(y,yt,f,a,pi,I,J,q,'one');th=z[0];v=z[1]**2
        vals.append((prob,th,v))
    p,t,v=np.array(vals).T;truth=np.sum(a*y)/a.sum();assert np.isclose(np.sum(p*t),truth)
    var=np.sum(p*(t-truth)**2);meanvar=np.sum(p*v);assert np.isclose(var,meanvar,rtol=1e-12)
    return {'enumerated_fixed_lambda_unbiased':True,'variable_q_mean_variance':float(meanvar),'exact_sampling_variance':float(var)}


def hajek(y,yt,f,a,pi,I,J,q):
    D=np.sum(a[I]/pi[I]);A=a.sum()
    num=np.sum(a[I]*yt[I]/pi[I]);j=I&J
    num+=np.sum(a[j]*(y[j]-yt[j])/(pi[j]*q[j]))
    th=num/D
    v=estimate(y,yt,np.full(len(y),th),a,pi,I,J,q,'one')
    scale=A/D
    # The point/SE already use the random Hájek denominator. Apply the same
    # scale to diagnostic correction/variance components, not only the SE.
    return (th,v[1]*scale,max(I.sum()-1,1),v[3],0.,v[5],v[6],v[7]*scale,v[8]*scale**2,v[9]*scale**2,v[10]*scale**2)
