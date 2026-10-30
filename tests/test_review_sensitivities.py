"""Synthetic software checks for the conditional sensitivity analyses."""
from pathlib import Path
import subprocess, sys, tempfile, unittest
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'analysis'))
from s25_weight_sensitivity import estimate, weights_for, rake
from s27_direct_budget_check import summarize
import s24_two_level as tl


class ReviewSensitivityChecks(unittest.TestCase):
    @staticmethod
    def frame():
        rows=[]
        for k,(band,pop) in enumerate([('<=100',80),('101-500',250),('501-1000',800),('1001-3300',2000)]):
            for src in ['GW','SW']:
                for j in range(8):
                    observed=j<4
                    y=float(j%3==0) if observed else np.nan
                    rows.append(dict(PWSID=f'X{len(rows):05d}',pop=pop,P=pop,
                        y_mcl=y,f=.1+.07*j+.01*k,src=src,scls=band,
                        cell=src+'|'+band,EPA_REGION='R1',PRIMACY_AGENCY_CODE='XX'))
        return pd.DataFrame(rows)

    def test_raking_preserves_known_margins(self):
        d=self.frame();base=weights_for(d,'cell')
        w,info=rake(d,['cell','EPA_REGION'],base)
        self.assertLess(info['maximum_relative_margin_error'],1e-10)
        self.assertTrue(np.allclose(w,base))
        sample=d[d.y_mcl.notna()]
        got=pd.Series(w,index=sample.index).groupby(sample.cell).sum()
        self.assertTrue(np.allclose(got,d.groupby('cell').size()))

    def test_outcome_odds_one_reproduces_poststratification(self):
        d=self.frame();c=d.groupby('cell').agg(N=('PWSID','size'),n=('y_mcl','count'),positive=('y_mcl','sum'))
        q=c.positive/c.n
        scenario=(c.positive.sum()+((c.N-c.n)*q).sum())/len(d)
        direct=estimate(d,weights_for(d,'cell'))['systems_hajek']
        self.assertAlmostEqual(scenario,direct,14)

    def test_shared_expected_cost_and_summary(self):
        score=np.array([.1,.5,2,4.]);full=np.array([2,4,2,4.]);reduced=np.ones(4)
        expected_cost=.2*full+.8*reduced
        pi=tl.budget_probs(score,expected_cost,3.)
        self.assertAlmostEqual(float(np.sum(pi*expected_cost)),3.,12)
        rows=[(.3,.02,8,10,20),(.4,.03,8,10,22)]
        out=summarize(rows,.35)
        self.assertAlmostEqual(out['mse'],.0025,14)
        self.assertEqual(out['draws'],2)

    def test_additive_bootstrap_and_calendar_boundary(self):
        d=self.frame()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);frame=p/'frame.parquet';d.to_parquet(frame,index=False)
            pd.DataFrame({'PWSID':d.PWSID,'Size':'S'}).to_parquet(p/'occurrence.parquet',index=False)
            subprocess.run([sys.executable,str(REPO/'analysis/s25_weight_sensitivity.py'),
                '--frame',str(frame),'--occurrence',str(p/'occurrence.parquet'),
                '--output',str(p/'weight'),'--replicates','40'],check=True,capture_output=True)
            dom=pd.read_csv(p/'weight/additive_domain_burden_canonical.csv')
            z=estimate(d,weights_for(d,'cell'))
            for target in ['systems','people']:
                x=dom[dom.estimand==target]
                self.assertAlmostEqual(x.affected_share.sum(),1.,12)
                denominator=len(d) if target=='systems' else d.P.sum()
                self.assertAlmostEqual(x.affected_total.sum(),z[target+'_difference']*denominator,8)
            curve=pd.read_csv(p/'weight/within_cell_selection_odds_scenarios.csv')
            self.assertAlmostEqual(curve.loc[curve.gamma.eq(1),'system_prevalence'].iloc[0],z['systems_hajek'],12)
            events=[]
            for i,row in d[d.y_mcl.notna()].iterrows():
                n=2 if row.src=='GW' else 4
                span=367 if i==0 else 366
                for k in range(n):
                    events.append(dict(PWSID=row.PWSID,FacilityID='F',SamplePointID='P',
                        SampleEventCode=f'SE{k+1}',WaterType=row.src,
                        date=pd.Timestamp('2024-01-01')+pd.Timedelta(days=span*k/(n-1)),
                        v_PFOA=5. if row.y_mcl else np.nan,v_PFOS=np.nan))
            pd.DataFrame(events).to_parquet(p/'events.parquet',index=False)
            d[['PWSID']].to_parquet(p/'inventory.parquet',index=False)
            subprocess.run([sys.executable,str(REPO/'analysis/s26_timing_sensitivity.py'),
                '--events',str(p/'events.parquet'),'--frame',str(frame),
                '--inventory',str(p/'inventory.parquet'),'--output',str(p/'timing')],check=True,capture_output=True)
            timing=pd.read_csv(p/'timing/timing_burden_sensitivity.csv')
            self.assertEqual(timing.n_observed.tolist(),[32,31])
            self.assertEqual(pd.read_csv(p/'timing/locations_span_over366days.csv').shape[0],1)


if __name__=='__main__':
    unittest.main()
