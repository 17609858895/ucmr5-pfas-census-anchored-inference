from pathlib import Path
import json
import numpy as np
import pandas as pd
import argparse
from project_paths import R

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,default=R/'review_sensitivity')
    args=ap.parse_args();P=args.directory
    d=pd.read_csv(P/'direct_budget_by_seed.csv')
    dual=d[d.strategy.str.startswith('Shared')].copy();ref=d[d.strategy.str.startswith('Reference')].copy()
    agg=dual.groupby(['scenario','budget','estimand'],sort=False).agg(mean_mse_ratio=('mse_ratio_to_reference','mean'),min_mse_ratio=('mse_ratio_to_reference','min'),max_mse_ratio=('mse_ratio_to_reference','max'),mean_coverage=('cov_kg','mean'),min_coverage=('cov_kg','min'),max_coverage=('cov_kg','max'),mean_interval_width=('mean_interval_width','mean'),mean_realized_cost=('mean_cost','mean'),mean_selected_systems=('mean_n','mean')).reset_index()
    agg['reference_expected_cost']=[ref.loc[(ref.scenario==s)&(ref.estimand==e),'expected_cost'].mean() for s,e in zip(agg.scenario,agg.estimand)]
    agg['expected_sample_reduction_pct']=100*(1-agg.budget/agg.reference_expected_cost)
    agg.to_csv(P/'direct_budget_figure_data.csv',index=False)
    wide=[];display=[]
    for (sc,b),g in agg.groupby(['scenario','budget'],sort=False):
        row={'scenario':sc,'budget':float(b),'reference_expected_cost':float(g.reference_expected_cost.iloc[0]),'expected_sample_reduction_pct':float(g.expected_sample_reduction_pct.iloc[0])}
        for _,r in g.iterrows():
            for k in ['mean_mse_ratio','min_mse_ratio','max_mse_ratio','mean_coverage','min_coverage','max_coverage','mean_interval_width','mean_realized_cost','mean_selected_systems']:
                row[r.estimand+'_'+k]=r[k]
        wide.append(row)
        display.append({'Scenario':sc,'Expected samples / reference (reduction)':f"{b:,.0f} / {row['reference_expected_cost']:,.0f} ({row['expected_sample_reduction_pct']:.1f}%)",'System MSE ratio':f"{row['systems_mean_mse_ratio']:.2f}",'Population MSE ratio':f"{row['people_mean_mse_ratio']:.2f}",'System coverage':f"{row['systems_mean_coverage']:.3f}",'Population coverage':f"{row['people_mean_coverage']:.3f}"})
    pd.DataFrame(wide).to_csv(P/'tableS11_direct_budget_canonical.csv',index=False)
    pd.DataFrame(display).to_csv(P/'tableS11_direct_budget_display.csv',index=False)
    refagg=ref.groupby(['scenario','estimand']).agg(mean_expected_cost=('expected_cost','mean'),mean_coverage=('cov_kg','mean'),min_coverage=('cov_kg','min'),max_coverage=('cov_kg','max'),mean_mse=('mse','mean'),mean_interval_width=('mean_interval_width','mean')).reset_index()
    refagg.to_csv(P/'direct_reference_summary.csv',index=False)
    print(pd.DataFrame(display).to_string(index=False))

if __name__=='__main__':
    main()
