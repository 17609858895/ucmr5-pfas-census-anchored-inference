"""Native-size semantic figure modules, with independently redrawn panels.



Layouts group axes by scientific question. All sizes are specified in millimetres;

numeric results are read from retained analysis outputs without model fitting.

"""
from pathlib import Path as _ProjectPath

import os

import gc

import json

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.axes import Axes

from matplotlib.text import Text

import figstyle as fs

from figstyle import *  # shared palette, projection and source paths



# Current submission exports; retained data and schematic sources remain in OUTPUT.

FIG = OUTPUT / 'figures'   # this project's figure exports



fs.MIN_PT = 8.5

fs.LETTER_PT = 11.0

fs.COMPACT = 1.0





def setup_native():

    configure_publication_style()

    plt.rcParams.update({

        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],

        'font.size': 8.5, 'axes.labelsize': 9.5, 'axes.labelweight': 'normal',

        'axes.linewidth': .8, 'axes.labelpad': 4, 'xtick.labelsize': 8.5,

        'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'legend.title_fontsize': 8.5,

        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,

        'xtick.major.width': .8, 'ytick.major.width': .8,

        'xtick.major.pad': 3, 'ytick.major.pad': 3,

        'xtick.minor.size': 2, 'ytick.minor.size': 2,

        'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 600})





setup_native()





def style_axis(ax, xlabel=None, ylabel=None, tick_size=8.5, label_size=9.5):

    fs.style_axis(ax, xlabel=xlabel, ylabel=ylabel, tick_size=tick_size, label_size=label_size)

    for spine in ['left', 'bottom']:

        ax.spines[spine].set_linewidth(.8)

    ax.tick_params(which='major', length=3.5, width=.8, pad=3, bottom=True, left=True)

    ax.xaxis.labelpad=4

    ax.yaxis.labelpad=4

    return ax





def hcolorbar(ax, cmap, norm, label, ticks=None, extend='neither',

              pos=(.60,.06,.34,.05), fmt=None):

    cb = fs.hcolorbar(ax,cmap,norm,label,ticks=ticks,extend=extend,pos=pos,fmt=fmt)

    cb.ax.tick_params(labelsize=8.5, width=.7, length=2.5)

    cb.set_label(label, fontsize=8.5, labelpad=3)

    cb.outline.set_linewidth(.7)

    return cb





def draw_module(fig, ss, key, builder):

    """builder(fig, SubplotSpec, context) can create nested axes/legend strips."""

    before = set(fig.axes)

    builder(fig, ss, {'standalone': getattr(fig, '_standalone', False)})

    axes = [a for a in fig.axes if a not in before]

    if not axes:

        raise ValueError(f'Panel {key} created no axes')

    if not hasattr(fig, '_panel_records'):

        fig._panel_records = []

    fig._panel_records.append(dict(key=key,axes=axes,target=ss.get_position(fig).frozen()))





def simple(draw):

    """Adapt an ordinary draw(ax) function to a semantic module."""

    def builder(fig, ss, context):

        draw(fig.add_subplot(ss))

    return builder





def save_modules(builders, folder, stem, layout, figsize_mm=(180,160),

                 sizes_mm=None, letters='abcdefghijkl'):

    out = (OUTPUT / 'figures' if folder.startswith('qa/') else FIG) / folder

    sub = out / 'subfigures'

    sub.mkdir(parents=True,exist_ok=True)

    # Remove only surplus current-stem panels after a panel-count change.

    # The untouched preceding revision retains the previous figure system.

    valid={f'{stem}_{k}' for k in letters[:len(builders)]}

    for old in [*sub.glob(f'{stem}_*.png'),*sub.glob(f'{stem}_*.pdf')]:

        if old.stem not in valid: old.unlink()

    fig = plt.figure(figsize=np.asarray(figsize_mm)/25.4)

    cells = layout(fig)

    assert len(cells)==len(builders), (stem,len(cells),len(builders))

    for b,ss,k in zip(builders,cells,letters): draw_module(fig,ss,k,b)

    # Export calls the shared outer-boundary refinement and palette/tick audits.

    report = fs._export(fig,out,stem,True)

    print(stem, 'colours',report['n_unexpected'],'overlaps',len(report['label_overlaps']),flush=True)

    if os.environ.get('FIG_COMBINED_ONLY')=='1': return

    if sizes_mm is None: sizes_mm=[(110,85)]*len(builders)

    for b,size,k in zip(builders,sizes_mm,letters):

        fg=plt.figure(figsize=np.asarray(size)/25.4)

        fg._standalone=True

        gs=fg.add_gridspec(1,1,left=.06,right=.985,bottom=.06,top=.97)

        draw_module(fg,gs[0],k,b)

        fs._export(fg,sub,f'{stem}_{k}',False)

        gc.collect()

    (sub/'README.md').write_text('Each semantic panel is independently redrawn by the same source builder, including its own reading-essential legend. These files are not crops.\n',encoding='utf-8')



