"""Refined pictorial vector graphical abstract, 190 x 76 mm, regular typography.

Usage: python draw_graphical_abstract.py --outdir <new output folder>
No model fitting. Numerical labels are read from the current frozen CSV tables.
This is one continuous summary with no independently interpretable subpanels.
All pictorial elements are constructed from original geometric paths, not images.
"""
from pathlib import Path
import math
import argparse,csv,hashlib,json,sys

HERE=Path(__file__).resolve().parent
def source_tables():
    """Frozen inputs: the submission package's figure source data, or the code release's reference results."""
    fsd=HERE.parent/'Figure_Source_Data'          # submission package: Source_Code sits beside Figure_Source_Data
    tables=(fsd/'inventory_burden_and_census_designs/fig5_burden.csv',fsd/'monitoring_designs/external_primary_R12_B300.csv')
    if all(p.is_file() for p in tables):
        return (fsd,*tables)
    ref=HERE/'strengthening/reference_results'
    return ref,ref/'figure_data/fig5_burden.csv',ref/'method/external_primary_R12_B300.csv'
SOURCE_ROOT,BURDEN,EXTERNAL=source_tables()
HELPERS=HERE/'strengthening/method/figure_helpers/figkit/publication_helpers'
sys.path.insert(0,str(HELPERS))
from scientific_palettes import get_palette
from ml_publication_style import configure_publication_style
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager

configure_publication_style()
plt.rcParams.update({'font.family':'Arial','font.weight':'normal','axes.labelweight':'normal',
                     'axes.titleweight':'normal','pdf.fonttype':42,'svg.fonttype':'none'})

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as s:return list(csv.DictReader(s))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def evidence():
    burden,external=BURDEN,EXTERNAL
    data=rows(burden)
    a=next(r for r in data if r['group']=='all' and r['estimand']=='systems' and r['estimator']=='ANCHOR PPI++')
    b=next(r for r in data if r['group']=='all' and r['estimand']=='people' and r['estimator']=='ANCHOR PPI++')
    tests=[r for r in rows(external) if r['estimand']=='people' and r['policy'] in ('uniform20','pps_two_phase_hajek') and r['coefficient']!='one']
    # Headline range follows the shared-sample designs summarized in the paper;
    # the comparison is full-schedule Neyman, never full-schedule SRS.
    gains=[100*(1-float(r['mse_ratio_to_Neyman_full'])) for r in tests]
    assert int(a['n'])==970 and int(a['N'])==56591
    assert 4440<float(a['count'])<4460 and 1.89e6<float(b['count'])<1.90e6
    assert len(tests)==4 and 69<min(gains)<72 and 82<max(gains)<85,gains
    return {'observed_systems':int(a['n']),'inventory_systems':int(a['N']),
            'estimated_exceeding_systems':float(a['count']),'associated_service_population':float(b['count']),
            'system_prevalence':float(a['theta']),'MSE_gain_percent_vs_full_Neyman':gains,
            'sources':[{'path':p.relative_to(SOURCE_ROOT).as_posix(),'sha256':sha(p)} for p in (burden,external)], 'map_outline_sha256':sha(HERE/'graphical_abstract_us_outline.json')}


# Geometry, glyphs and label positions are shared with Fig. 3a (pictorial_scene.py).
sys.path.insert(0,str(HERE))
from pictorial_scene import draw_scene,TEXT,W,H,GA_LABELS

def draw(ax):
    draw_scene(ax,GA_LABELS)


def render(outdir,overwrite=False):
    out=Path(outdir).resolve()
    if out.exists() and any(out.iterdir()) and not overwrite:raise FileExistsError('Use a fresh output folder.')
    out.mkdir(parents=True,exist_ok=True)
    evidence_data=evidence()
    TEXT.clear()
    fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white');ax=fig.add_axes([0,0,1,1]);draw(ax)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bounds=[]
    for t in TEXT:
        box=t.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:raise ValueError('Clipped text: '+t.get_text())
        bounds.append({'text':t.get_text(),'font':t.get_fontfamily(),'weight':t.get_fontweight(),'point_size':t.get_fontsize(),'bbox_figure_fraction':[box.x0/fig.bbox.width,box.y0/fig.bbox.height,box.width/fig.bbox.width,box.height/fig.bbox.height]})
    meta={'Author':'CHONG LIU','Creator':'CHONG LIU','Producer':'CHONG LIU','Title':'Graphical abstract'}
    fig.savefig(out/'Graphical_Abstract.png',dpi=600,facecolor='white',metadata=meta)
    fig.savefig(out/'Graphical_Abstract.pdf',facecolor='white',metadata=meta)
    # One uninterrupted graphical abstract: component crops would lose its scope.
    (out/'subfigures').mkdir(exist_ok=True)
    (out/'subfigures/README.md').write_text('This graphical abstract is a single continuous composition with no panel letters or separately interpretable subfigures. No cropped or duplicate panel exports are supplied.\n',encoding='utf-8')
    (out/'render_record.json').write_text(json.dumps({'canvas_mm':[W,H],'palette':'nature_new','all_text_regular_weight':True,'font_points_min':min(t.get_fontsize() for t in TEXT),'vector_PDF':True,'sources':evidence_data,'text_items':bounds},ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig);print('Created',out/'Graphical_Abstract.pdf')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--overwrite',action='store_true')
    a=ap.parse_args();render(a.outdir,a.overwrite)
if __name__=='__main__':main()
