"""Current figure dispatch; never rebuilds superseded Fig.2/6/7 by default."""
from pathlib import Path
import argparse,os,subprocess,sys,shutil
REPO=Path(__file__).resolve().parent
BASE={'1','3','4','5','S1','S2','S3'}
NEW={'2','6','7','S4','S5','S6'}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--only',nargs='+',choices=sorted(BASE|NEW),default=['2','6','7','S4','S5','S6'])
 ap.add_argument('--workdir',type=Path,default=REPO)
 ap.add_argument('--legacy-data',type=Path,help='Complete historical figure_data dependency directory, if not generated in workdir')
 a=ap.parse_args();work=a.workdir.resolve();env=os.environ.copy();env['ANCHOR_WORKDIR']=str(work);env['PYTHONIOENCODING']='utf-8'
 def run(stage,*args):subprocess.run([sys.executable,str(REPO/'run_strengthening.py'),stage,'--workdir',str(work),*args],cwd=REPO,env=env,check=True)
 chosen=set(a.only)
 for group,stage in [({'2','S4'},'figures-spatial'),({'6','S5'},'figures-external')]:
  use=chosen&group
  if not use:continue
  if stage=='figures-spatial':run(stage,'--only',*sorted(use))
  elif use==group:run(stage)
  else:run(stage,'--only',next(iter(use)))
 if chosen&{'7','S6'}:run('figures-method','--only','all' if {'7','S6'}<=chosen else 'main' if '7' in chosen else 'supplement')
 if chosen&BASE:
  legacy=REPO/'legacy_support' if (REPO/'legacy_support').is_dir() else REPO
  if a.legacy_data:
   target=work/'figure_data';target.mkdir(parents=True,exist_ok=True)
   for f in a.legacy_data.resolve().glob('*.csv'):shutil.copy2(f,target/f.name)
  required=['fig1_map_points.csv','fig3_inclusion_probabilities.csv','fig4_grid_summary.csv']
  if not all((work/'figure_data'/f).exists() for f in required):raise FileNotFoundError('Legacy imports require complete figure_data; regenerate using original figure-data stage or pass --legacy-data. Large system-level figure inputs are not redistributed.')
  subprocess.run([sys.executable,str(legacy/'plotting/refine_jhm_figures.py'),'--only',*sorted(chosen&BASE)],cwd=legacy,env=env,check=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
