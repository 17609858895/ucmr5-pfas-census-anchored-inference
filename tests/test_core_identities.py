"""Deterministic scientific identity checks; no real data or model fitting."""
from pathlib import Path
import itertools,sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
import anchor_core as ac
import s24_two_level as tp

class CoreIdentities(unittest.TestCase):
 def test_probability_caps_preserve_expected_sample_size(self):
  p=ac.cap_probs(np.array([100.,3.,2.,1.]),2)
  self.assertTrue(np.all((p>0)&(p<=1)))
  self.assertAlmostEqual(float(p.sum()),2,places=12)
  self.assertEqual(p[0],1.)
 def test_budget_calibration_preserves_expected_cost_after_caps(self):
  c=np.array([4.,2.,1.,3.]);p=tp.budget_probs(np.array([50.,2.,3.,1.]),c,6.)
  self.assertTrue(np.all((p>0)&(p<=1)))
  self.assertAlmostEqual(float(p@c),6.,places=10)
 def test_full_observation_reduces_to_exact_weighted_mean(self):
  y=np.array([1.,0.,1.]);yt=np.array([0.,0.,1.]);a=np.array([1.,2.,4.]);f=np.array([.6,.2,.5])
  hit=np.ones(3,dtype=bool)
  th,se,_,_=tp.estimate(y,yt,f,a,np.ones(3),hit,hit,1.,a.sum(),float(a@f),True)
  self.assertAlmostEqual(th,float(a@y/a.sum()),places=14)
  self.assertAlmostEqual(se,0.,places=14)
 def test_exact_enumeration_two_phase_design_mean(self):
  y=np.array([1.,0.,1.]);yt=np.array([0.,0.,1.]);a=np.array([1.,2.,4.]);f=np.array([.6,.2,.5]);pi=np.array([.25,.6,.8]);q=.3
  expected=0.;mass=0.
  for bits in itertools.product([False,True],repeat=6):
   i=np.array(bits[:3]);j=np.array(bits[3:]);prob=np.prod(np.where(i,pi,1-pi))*np.prod(np.where(j,q,1-q))
   th,*_=tp.estimate(y,yt,f,a,pi,i,j,q,a.sum(),float(a@f),False)
   expected+=prob*th;mass+=prob
  self.assertAlmostEqual(mass,1.,places=14)
  self.assertAlmostEqual(expected,float(a@y/a.sum()),places=13)
 def test_scalar_vector_interval_agreement(self):
  th=np.array([0.,.2,.8,1.]);se=np.array([.03,.08,.07,.02]);df=np.array([5.,12.,20.,7.]);n=np.array([15.,30.,40.,20.])
  vec=ac.intervals_vec(th,se,df,n)
  for k in range(4):
   scalar=ac.intervals(th[k],se[k],df[k],n[k])
   for name in ac.INTERVALS:
    np.testing.assert_allclose(scalar[name],[vec[name][0][k],vec[name][1][k]],rtol=1e-12,atol=1e-12)

if __name__=='__main__':unittest.main()
