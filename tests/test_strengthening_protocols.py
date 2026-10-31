"""No-data scientific boundary tests for the executed strengthening helpers."""
import itertools,sys,unittest
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from strengthening.external.protocol_core import (washington_pwsid,source_event_count,
 exclude_known_ids,first_count_window,first_complete_calendar,retained_mean_outcome,effective_observation_end)
from strengthening.external.metric_core import auc_counts,exact_auc_label_bounds
from strengthening.method.estimator_core import estimate,variance_test,hajek
from strengthening.spatial import spatial_transfer as st
from strengthening.spatial.summarize_spatial import weighted_u

class ExternalProtocolTests(unittest.TestCase):
 def test_followup_opportunity_uses_actual_state_end(self):
  mi=effective_observation_end(pd.to_datetime(['2023-01-01','2025-12-24']),'2026-06-30')
  wa=effective_observation_end(pd.to_datetime(['2023-01-01','2026-10-13']),'2026-06-30')
  self.assertEqual(mi-pd.Timedelta(days=366),pd.Timestamp('2024-12-23'))
  self.assertEqual(wa-pd.Timedelta(days=366),pd.Timestamp('2025-06-29'))
  self.assertGreater(pd.Timestamp('2025-01-01'),mi-pd.Timedelta(days=366))
 def test_exact_ID_and_all_UCMR_membership_exclusion(self):
  self.assertEqual(washington_pwsid('="60850"'),'WA5360850')
  self.assertEqual(washington_pwsid('AE500'),'WA53AE500')
  with self.assertRaises(ValueError):washington_pwsid('abcdefg')
  f=pd.DataFrame({'PWSID':['WA5360850','WA53AE500','MI0000001'],'complete_label':[0,0,1]})
  # Known membership, including incomplete outcomes, controls exclusion.
  q=exclude_known_ids(f,['WA5360850','MI0000001'])
  self.assertEqual(q.PWSID.tolist(),['WA53AE500'])
 def test_verified_source_class_controls_event_count(self):
  self.assertEqual(source_event_count('Ground Water'),2)
  self.assertEqual(source_event_count('Ground Water Well'),2)
  self.assertEqual(source_event_count('Surface'),4)
  for x in ['Intertie','SeaWater','RIGallery','Other',None,np.nan]:self.assertEqual(source_event_count(x),0)
 def test_first_count_span_boundary_and_no_later_search(self):
  d=pd.DataFrame({'date':pd.to_datetime(['2023-01-01','2024-01-03','2024-01-04']),'value':[0,1,1]})
  self.assertIsNone(first_count_window(d,2))
  d.loc[1,'date']=pd.Timestamp('2024-01-01')
  q=first_count_window(d,2);self.assertEqual((q.date.max()-q.date.min()).days,365)
  d.loc[1,'date']=pd.Timestamp('2024-01-02')
  self.assertIsNotNone(first_count_window(d,2))
  leap=pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2025-01-01'])})
  self.assertIsNotNone(first_count_window(leap,2))
  leap.loc[1,'date']=pd.Timestamp('2025-01-02')
  self.assertIsNone(first_count_window(leap,2))
 def test_censor_before_mean_and_either_analyte(self):
  self.assertEqual(retained_mean_outcome([[3.9,0],[4.2,0]]),0)
  self.assertEqual(retained_mean_outcome([[0,0],[8.1,0]]),1)
  self.assertEqual(retained_mean_outcome([[0,0],[0,8.0]]),0)
  self.assertEqual(retained_mean_outcome([[0,0],[0,8.0]],4.0),1)
 def test_calendar_first_complete_year_first_record_no_outcome_selection(self):
  d=pd.DataFrame({'date':pd.to_datetime(['2023-01-01','2024-01-01','2024-02-01','2024-07-01','2025-01-01','2025-07-01']),'need_actual':[2]*6,'y':[1,0,1,0,1,1]})
  q=first_complete_calendar(d.sample(frac=1,random_state=3))
  self.assertEqual(q.date.dt.year.unique().tolist(),[2024])
  self.assertEqual(q.y.tolist(),[0,0])
  s=pd.DataFrame({'date':pd.to_datetime(['2023-01-01','2023-04-02','2023-07-02','2023-10-02']),'need_actual':[4]*4})
  self.assertEqual(len(first_complete_calendar(s)),4)
  self.assertIsNone(first_complete_calendar(s.iloc[:3]))

class MetricIdentityTests(unittest.TestCase):
 def test_weighted_AUC_with_ties(self):
  y=np.array([0,1,0,1,1]);f=np.array([.1,.1,.4,.7,.7]);counts=np.array([[1,1,1,1,1],[3,2,1,4,1]],float)
  actual=auc_counts(y,f,counts)
  for i,w in enumerate(counts):self.assertAlmostEqual(actual[i],roc_auc_score(y,f,sample_weight=w),places=14)
 def test_exact_label_scenario_bounds_match_all_assignments(self):
  score=np.array([.1,.3,.3,.7,.9]);lo=np.array([0,0,1,0,0]);hi=np.array([0,1,1,1,1]);a=np.flatnonzero(lo!=hi)
  values=[]
  for z in itertools.product([0,1],repeat=len(a)):
   y=lo.copy();y[a]=z;values.append(roc_auc_score(y,score))
  lower,upper=exact_auc_label_bounds(score,lo,hi)
  self.assertAlmostEqual(lower,min(values),places=14);self.assertAlmostEqual(upper,max(values),places=14)
 def test_variable_completion_exact_sampling_variance(self):
  r=variance_test();self.assertTrue(r['enumerated_fixed_lambda_unbiased'])
  self.assertAlmostEqual(r['variable_q_mean_variance'],r['exact_sampling_variance'],places=13)
 def test_nonnegative_fixed_coefficient_variance_identity(self):
  y=np.array([1.,0.]);yt=np.array([0.,0.]);f=np.array([.2,.3]);a=np.array([1.,4.]);pi=np.array([.3,.7]);q=np.array([.2,.6]);vals=[]
  for state in itertools.product(range(3),repeat=2):
   I=np.array(state)>0;J=np.array(state)==2
   p=np.prod([1-pi[i] if s==0 else pi[i]*(1-q[i]) if s==1 else pi[i]*q[i] for i,s in enumerate(state)])
   r=estimate(y,yt,f,a,pi,I,J,q,'one');vals.append((p,r[0],r[10]))
  p,t,v=np.array(vals).T;truth=np.average(y,weights=a)
  self.assertAlmostEqual(np.sum(p*t),truth,places=14)
  self.assertTrue((v>=0).all());self.assertAlmostEqual(np.sum(p*v),np.sum(p*(t-truth)**2),places=13)
 def test_q1_hajek_point_and_SE_match_classic(self):
  sys.path.insert(0,str(REPO/'analysis'));import anchor_core as ac
  rng=np.random.default_rng(17171);n=200;y=rng.binomial(1,.2,n).astype(float);a=rng.lognormal(4,2,n);f=rng.uniform(.01,.4,n);pi=rng.uniform(.1,.8,n);I=rng.random(n)<pi
  v=hajek(y,y,f,a,pi,I,np.ones(n,bool),np.ones(n));classic=ac.est_hajek(y[I],a[I],1/pi[I],pi[I],a.sum(),('poisson',))
  np.testing.assert_allclose(v[:2],classic[:2],rtol=1e-13,atol=1e-15)

class SpatialInformationTests(unittest.TestCase):
 def setUp(self):
  self.ref=pd.DataFrame({'PWSID':['a','b'],'lat':[0.,.01],'lon':[0.,0.],'y_mcl':[0.,1.],'y_det':[0.,1.],'block':[1,2]})
 def test_target_labels_do_not_enter_neighbours(self):
  t=pd.DataFrame({'PWSID':['t'],'lat':[.001],'lon':[0.],'y_mcl':[0.],'y_det':[0.],'block':[3]})
  pd.testing.assert_frame_equal(st.nb_features(t,self.ref),st.nb_features(t.assign(y_mcl=1.,y_det=1.),self.ref))
 def test_own_ID_and_own_region_excluded(self):
  own=st.nb_features(self.ref.iloc[[0]],self.ref);self.assertEqual(own.nb_n50.iloc[0],1);self.assertEqual(own.nb_mcl50.iloc[0],1)
  a=st.training_nb(self.ref);r2=self.ref.copy();r2.loc[r2.block==1,['y_mcl','y_det']]=1;b=st.training_nb(r2)
  pd.testing.assert_frame_equal(a.loc[self.ref.block==1],b.loc[self.ref.block==1])
 def test_unseen_target_categories_not_fitted(self):
  tr=pd.DataFrame({'owner':['P','R'],'state':['AA','BB'],'region':['1','2']});te=pd.DataFrame({'owner':['P'],'state':['ZZ'],'region':['9']})
  _,mapped,_=st.category_maps(tr,te);self.assertTrue(mapped.state.isna().all());self.assertTrue(mapped.region.isna().all())
 def test_region_weighted_AUC_matrix_identity(self):
  rng=np.random.default_rng(20261031);y=rng.integers(0,2,100);p=rng.random(100);w=rng.random(100)+.1;g=rng.integers(0,10,100)
  u,pos,neg=weighted_u(y,p,w,g)
  for _ in range(20):
   c=rng.multinomial(10,np.full(10,.1));self.assertAlmostEqual((c@u@c)/((c@pos)*(c@neg)),roc_auc_score(y,p,sample_weight=w*c[g]),places=12)

if __name__=='__main__':unittest.main()
