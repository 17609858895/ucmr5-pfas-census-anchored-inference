"""Current three-row figures; read frozen quantitative outputs without fitting."""
from pathlib import Path
import argparse,json,os,sys,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from project_paths import PROJECT_ROOT, R

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only',nargs='*');ap.add_argument('--draft',action='store_true')
    args=ap.parse_args()
    (PROJECT_ROOT/'checks').mkdir(parents=True,exist_ok=True)
    (PROJECT_ROOT/'figure_data').mkdir(parents=True,exist_ok=True)
    for name in ['weight_calibration_sensitivity.csv','fig5c_additive_domain_shares.csv',
        'additive_domain_burden_canonical.csv','within_cell_selection_odds_curve.csv',
        'fig5c_full_inventory_shares.csv','timing_burden_sensitivity.csv','direct_budget_figure_data.csv']:
        source=R/'review_sensitivity'/name
        if source.exists():shutil.copy2(source,PROJECT_ROOT/'figure_data'/name)
    import generate_refined_figures as source
    import publication_layout as layout
    import jhm_burden_panels as burden
    import jhm_supporting_panels as supporting
    import jhm_direct_budget_panels as direct
    from f_common import C,OUTPUT
    from audit_palette_usage import validate_series_colors

    validate_series_colors('nature_new',C)
    for key in ['1','3']:
        builders,plan,size,subsizes,folder=source.TASKS[key]
        source.TASKS[key]=(builders,plan,(190,202),subsizes,folder)
    builders,plan,size,subsizes,folder=source.TASKS['4']
    source.TASKS['4']=(builders,plan,(190,200),[(145,105)]*4+subsizes[4:],folder)
    source.TASKS['5']=([burden.national,burden.weight_sensitivity,burden.selection_odds,burden.composition],burden.layout5,(190,212),[(185,80),(105,90),(105,90),(185,95)],'Fig5_small_burden')
    source.TASKS['S3']=([burden.state_map,burden.state_population,burden.size_prevalence,source.rp3.projected_cost3],burden.layout_s3,(190,210),[(125,95),(110,110),(185,85),(185,90)],'FigS3_geographic_scenarios')
    source.TASKS['S1']=([source.shared_panel(supporting.interval_coverage),source.shared_panel(supporting.interval_width),source.shared_panel(supporting.fraction()),source.shared_panel(supporting.fraction(True))],supporting.layout_s1,(190,155),[(105,88)]*4,'FigS1_interval_completion')
    source.TASKS['S2']=([supporting.scenario_forest('systems'),supporting.scenario_forest('people'),source.shared_panel(source.feature_dumbbell('systems')),source.shared_panel(source.feature_dumbbell('people')),source.rp3.ablation_dots3],supporting.layout_s2,(190,198),[(115,105),(115,105),(110,85),(110,85),(180,85)],'FigS2_design_sensitivity')
    source.TASKS['7']=([source.tr4b.rules_scatter4,source.depth_eff,direct.mse,source.rp3.depth_coverage3,direct.coverage],source.tr4b.layout7_three_rows,(190,202),[(105,95),(120,95),(185,85),(115,95),(115,95)],'Fig7_next_round_design')
    (OUTPUT/'checks/figure_registry.json').write_text(json.dumps([{'number':number,'folder':values[4],'panels':len(values[0]),'canvas_mm':values[2]} for number,values in source.TASKS.items()],indent=2),encoding='utf-8')
    if args.draft:
        layout.FIG=OUTPUT/'checks/draft_figures'
        source.fs.QA=layout.FIG/'qa'
        os.environ['FIG_COMBINED_ONLY']='1';os.environ['FIG_DPI']='180'
    for number in args.only or source.TASKS:
        builders,plan,size,subsizes,folder=source.TASKS[number]
        source.save(number,builders,plan,size,subsizes,folder)


if __name__=='__main__':
    main()
