"""Publication layout, semantic palette and outer-boundary audits."""
from pathlib import Path as _ProjectPath
import json, pathlib, re, sys, gc, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba_array
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, ScaledTranslation
from matplotlib.text import Text
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

SK = pathlib.Path(__file__).resolve().parent / "publication_helpers"
sys.path.insert(0, str(SK))
from ml_publication_style import configure_publication_style, style_axis, numeric_ticks, export_subfigures, save_figure_with_subfigures  # noqa: E402,F401
from scientific_palettes import get_palette, STYLE_COLORS  # noqa: E402
from audit_palette_usage import validate_series_colors, audit_series_artists, _series_colours

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"analysis"))
from project_paths import PROJECT_ROOT, STATE_KML
OUTPUT = PROJECT_ROOT
RES = OUTPUT / "results"
FIG = OUTPUT / "figures"
QA = FIG / "qa"
PAL = get_palette("nature_new")   # DF9E9B 99BADF D8E7CA 99CDCE 999ACD FFD0E9 F6D58B C7A3CC
INK, MUTED, LIGHT = STYLE_COLORS["ink"], STYLE_COLORS["muted"], STYLE_COLORS["light"]
LAND, BORDER = "#EEF0F2", "#C3C8CE"   # structural map neutrals
MIN_PT, LETTER_PT, COMPACT = 14.0, 16, 0.88

ROLE = {"CLF": PAL[6], "SUB": PAL[0], "TOBIT": PAL[2], "AFT": PAL[1], "CURE": PAL[4], "STACK": PAL[7],
        "ANCHOR": PAL[3], "AIPW": PAL[1], "FE": PAL[0], "UCMR3": PAL[1], "UCMR5": PAL[0],
        "ND": PAL[1], "c4_10": PAL[6], "c10_20": PAL[0], "c20": PAL[4],
        "carboxylate": PAL[0], "sulfonate": PAL[1], "GAC": PAL[3], "IEX": PAL[4], "untreated": PAL[6],
        "compliant": PAL[1], "exceed": PAL[0], "sampling": PAL[6]}
SEQ = LinearSegmentedColormap.from_list("seq_purple", ["#FFFFFF", PAL[4]])
SEQ_RED = LinearSegmentedColormap.from_list("seq_red", ["#FFFFFF", PAL[0]])
DIV = LinearSegmentedColormap.from_list("div_blue_red", [PAL[1], "#FFFFFF", PAL[0]])


def setup():
    configure_publication_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "axes.labelpad": 7, "xtick.major.pad": 5, "ytick.major.pad": 5, "legend.frameon": False,
                         "font.size": 13, "axes.labelsize": 15.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
                         "legend.fontsize": 11.5, "axes.labelweight": "normal", "font.weight": "normal",
                         "pdf.fonttype": 42})


setup()


def outside_legend(ax, above=False, pad_pt=10, **kw):
    """Legend outside the axes with a constant physical gap."""
    off = (pad_pt if above else -pad_pt) / 72
    tr = ax.transAxes + ScaledTranslation(0, off, ax.figure.dpi_scale_trans)
    return ax.legend(loc="lower center" if above else "upper center", bbox_to_anchor=(.5, 1 if above else 0),
                     bbox_transform=tr, borderaxespad=0, **kw)


# ------------------------------------------------------------------ panel registration and layout
def draw_panel(fig, ss, key, draw):
    before = set(fig.axes)
    draw(fig.add_subplot(ss))
    axes = [a for a in fig.axes if a not in before]
    if not hasattr(fig, "_panel_records"):
        fig._panel_records = []
    fig._panel_records.append(dict(key=key, axes=axes, target=ss.get_position(fig).frozen()))


def _panel_bbox(fig, axes, renderer):
    complete = []
    def visit(ax):
        if ax not in complete:
            complete.append(ax)
            for child in ax.child_axes: visit(child)
    for ax in axes: visit(ax)
    boxes = [a.get_tightbbox(renderer) for a in complete if a.get_visible()]
    return Bbox.union([b for b in boxes if b is not None]).transformed(fig.transFigure.inverted())


def _normalise_text(fig):
    fig.draw_without_rendering()
    for t in fig.findobj(Text):
        if t.get_text():
            t.set_fontsize(max(MIN_PT, t.get_fontsize()))
            t.set_fontweight("normal")
            if t.get_gid() != "keep_colour":
                t.set_color(INK)
    for ax in fig.findobj(Axes):
        for label in ax.get_xticklabels()+ax.get_yticklabels():
            label.set_fontweight('bold');label.set_fontsize(max(12.5,label.get_fontsize()))
        for label in [ax.xaxis.label,ax.yaxis.label]:
            label.set_fontweight('bold');label.set_fontsize(max(13.5,label.get_fontsize()))
        if ax.axison:
            # Numeric/category axes only; maps have no tick labels.
            ax.tick_params(which='major',length=4.5,width=1.0,bottom=True,left=True)
            for side in ['bottom','left']:
                if ax.spines[side].get_visible():ax.spines[side].set_linewidth(1.0)


def refine_layout(fig):
    _normalise_text(fig)
    for a in fig.findobj(Axes):
        for label in a.get_xticklabels()+a.get_yticklabels():
            label.set_fontweight('bold')
    recs = getattr(fig, "_panel_records", [])
    for r in recs:
        for a in r["axes"]:
            a.set_axes_locator(None)
            for label in a.get_xticklabels() + a.get_yticklabels():
                label.set_fontweight("bold")
    pt = np.tile(fig.get_size_inches(), 2) * 72
    # maps: width fills the cell, height follows the aspect; same-row panels adopt that outer height
    for r in recs:
        if len(r["axes"]) != 1 or r["axes"][0].get_gid() != "physical_basemap":
            continue
        ax, alloc = r["axes"][0], r["target"]
        for _ in range(6):
            fig.draw_without_rendering(); outer = _panel_bbox(fig, r["axes"], fig.canvas.get_renderer())
            sx = alloc.width / outer.width; p = ax.get_position(original=True)
            ax.set_position([alloc.x0 + (p.x0 - outer.x0) * sx, p.y0, p.width * sx, p.height * sx], which="both")
        fig.draw_without_rendering(); outer = _panel_bbox(fig, r["axes"], fig.canvas.get_renderer())
        h = min(outer.height, alloc.height)
        for o in recs:
            pv = o["target"]
            if abs(pv.y0 - alloc.y0) < 1e-8 and abs(pv.y1 - alloc.y1) < 1e-8:
                o["target"] = Bbox.from_extents(pv.x0, alloc.y1 - h, pv.x1, alloc.y1)
    report = []
    for r in recs:
        tgt, axes = r["target"], r["axes"]
        if len(axes) == 1 and axes[0].get_gid() == "schematic":
            fig.draw_without_rendering()
            outer = _panel_bbox(fig, axes, fig.canvas.get_renderer())
            report.append(dict(panel=r["key"], target=list(outer.extents), outer=list(outer.extents),
                               residual_pt=[0.,0.,0.,0.], max_residual_pt=0.,
                               excluded_from_alignment="Schematic; original aspect preserved"))
            continue
        physical_map = len(axes) == 1 and axes[0].get_gid() == "physical_basemap"
        data_axes = [a for a in axes if a.axison and not hasattr(a, '_colorbar')]
        fixed_square = (len(data_axes) == 1 and data_axes[0].get_aspect() != 'auto'
                        and not physical_map)
        fixed = physical_map or fixed_square
        allocation = tgt.frozen()
        for _ in range(20):
            fig.draw_without_rendering(); outer = _panel_bbox(fig, axes, fig.canvas.get_renderer())
            err = (np.array(outer.extents) - np.array(tgt.extents)) * pt
            if np.max(np.abs(err)) < 0.35:
                break
            sx, sy = tgt.width / outer.width, tgt.height / outer.height
            if fixed:
                sx = sy = min(sx, sy)
            for a in axes:
                p = a.get_position(original=True)
                a.set_position([tgt.x0 + (p.x0 - outer.x0) * sx, tgt.y0 + (p.y0 - outer.y0) * sy,
                                p.width * sx, p.height * sy], which="both")
        fig.draw_without_rendering(); outer = _panel_bbox(fig, axes, fig.canvas.get_renderer())
        if physical_map and outer.width < tgt.width - 1e-4:
            # height-limited map: keep top/bottom alignment and centre it horizontally in its cell
            dx = (tgt.x0 + tgt.x1) / 2 - (outer.x0 + outer.x1) / 2
            p = axes[0].get_position(original=True)
            axes[0].set_position([p.x0 + dx, p.y0, p.width, p.height], which="both")
            fig.draw_without_rendering(); outer = _panel_bbox(fig, axes, fig.canvas.get_renderer())
            tgt = Bbox.from_extents(outer.x0, tgt.y0, outer.x1, tgt.y1)
        if fixed_square:
            # Equal-aspect data rectangles cannot fill arbitrary outer cells in
            # both directions. Fit inside the allocation, centre horizontally
            # and align the complete visual top. Record the natural footprint
            # explicitly, including the unused allocation, rather than stretch
            # a reliability/scatter plot away from its 1:1 geometry.
            dx = (allocation.x0+allocation.x1-outer.x0-outer.x1)/2
            dy = allocation.y1-outer.y1
            for a in axes:
                p=a.get_position(original=True)
                a.set_position([p.x0+dx,p.y0+dy,p.width,p.height],which='both')
            fig.draw_without_rendering(); outer=_panel_bbox(fig,axes,fig.canvas.get_renderer())
            natural_height=outer.height
            natural_width=outer.width
            centre=(allocation.x0+allocation.x1)/2
            tgt=Bbox.from_extents(centre-natural_width/2,allocation.y1-natural_height,
                                  centre+natural_width/2,allocation.y1)
        res = (np.array(outer.extents) - np.array(tgt.extents)) * pt
        report.append(dict(panel=r["key"], target=list(tgt.extents), outer=list(outer.extents),
                           residual_pt=res.tolist(), max_residual_pt=float(np.max(np.abs(res))),
                           allocation=list(allocation.extents),
                           aspect_constraint='equal data aspect; natural outer footprint' if fixed_square else None,
                           allocation_overflow_pt=float(max(0,(allocation.x0-outer.x0)*pt[0],
                                                             (outer.x1-allocation.x1)*pt[0],
                                                             (allocation.y0-outer.y0)*pt[1],
                                                             (outer.y1-allocation.y1)*pt[1]))))
    return report


def add_letters(fig, report):
    W = fig.get_size_inches()[0] * 72
    arts = []
    for p in report:
        x0, y0, x1, y1 = p["outer"]
        arts.append(fig.text(x0 - 6 / W, y1, p["panel"], ha="right", va="top", fontsize=LETTER_PT,
                             fontweight="normal", color=INK, zorder=100))
    fig.draw_without_rendering(); rd = fig.canvas.get_renderer(); out = []
    for t in arts:
        b = t.get_window_extent(rd).transformed(fig.transFigure.inverted())
        hits = [p["panel"] for p in report if b.overlaps(Bbox.from_extents(*p["outer"]))]
        out.append(dict(letter=t.get_text(), bbox=list(b.extents), overlaps=hits))
    return out


# ------------------------------------------------------------------ audits
def audit_style(fig):
    ok = [to_rgb(c) for c in PAL] + [to_rgb(c) for c in [INK, MUTED, LIGHT, LAND, BORDER, "white", "black"]]
    bad = []
    for art in fig.findobj(Artist):
        if not art.get_visible() or isinstance(art, Text):
            continue
        if art.get_gid() == "choropleth":
            continue
        if hasattr(art, "get_array") and art.get_array() is not None:
            continue   # numeric colour map (white -> palette or palette -> white -> palette)
        getters = ["get_color", "get_facecolor", "get_edgecolor"]
        if isinstance(art, Line2D):
            if art.get_linestyle() in [None, "None", "", "none"]:
                getters.remove("get_color")
            getters += ["get_markerfacecolor", "get_markeredgecolor"]
        for g in getters:
            if not hasattr(art, g):
                continue
            try:
                cols = to_rgba_array(getattr(art, g)())
            except (TypeError, ValueError):
                continue
            for c in cols:
                if c[3] > 0 and not any(np.allclose(c[:3], o, atol=2e-3) for o in ok):
                    bad.append(dict(artist=type(art).__name__, getter=g, rgb=[round(v, 3) for v in c[:3]]))
    txt = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    return dict(unexpected_colours=bad[:20], n_unexpected=len(bad),
                min_font_pt=min(t.get_fontsize() for t in txt) if txt else None,
                bold_tick_labels=all(t.get_fontweight() in ("bold", 700) for a in fig.findobj(Axes) for t in a.get_xticklabels()+a.get_yticklabels() if t.get_visible()),
                tick_axes_including_children=len(fig.findobj(Axes)),
                regular_text=all(t.get_fontweight() in ("normal", 400) for t in txt),
                nonregular_text=[t.get_text() for t in txt if t.get_fontweight() not in ("normal",400)])


def audit_categorical_artists(fig):
    """Audit actual coloured data marks and legend proxies, with neutral outlines separate.

    Numeric maps/images interpolate the documented palette endpoints. Black/grey
    intervals, joins, axes and hollow-marker outlines are structural. A coloured
    data mark is never granted a baseline/control exemption.
    """
    from matplotlib.collections import Collection
    from matplotlib.patches import Patch
    palette=np.array([to_rgb(c) for c in PAL]);declared={};skipped=[]
    for i,art in enumerate(fig.findobj(Artist)):
        if not art.get_visible() or isinstance(art,Text):continue
        if art.get_gid()=='choropleth':continue
        if not isinstance(art,(Line2D,Patch,Collection)):continue
        try:cols=_series_colours(art)
        except (TypeError,ValueError):continue
        if not cols:continue
        visible=[np.asarray(matplotlib.colors.to_rgba(c)) for c in cols if matplotlib.colors.to_rgba(c)[3]>0]
        if not visible:continue
        is_palette=[bool(np.any(np.all(np.isclose(palette,c[:3],atol=1e-8,rtol=0),axis=1))) for c in visible]
        if all(is_palette):declared[f'{type(art).__name__}_{i}']=art
        elif any(is_palette):
            # E.g. an ANCHOR palette curve with a hollow, ink-outlined diamond.
            # Its coloured components were already checked by audit_style.
            skipped.append({'artist':type(art).__name__,'reason':'palette curve plus structural hollow outline'})
    result=audit_series_artists('nature_new',declared) if declared else {'passed':True,'checks':[],'scope':'schematic or numerical colour map; no categorical marks'}
    result['structural_outline_combinations']=skipped
    return result


def audit_overlaps(fig):
    from matplotlib.axes import Axes
    rd = fig.canvas.get_renderer(); axes = fig.findobj(Axes); obst, issues = [], []
    for i, a in enumerate(axes):
        lg = a.get_legend()
        if lg is not None and lg.get_visible():
            obst.append(("legend", i, lg.get_window_extent(rd)))
        if hasattr(a, "_colorbar"):
            obst.append(("colorbar", i, a.get_window_extent(rd)))
    for kind, owner, box in obst:
        for i, a in enumerate(axes):
            if not a.axison:
                continue
            if kind == "colorbar" and owner == i:
                continue
            for lab in [a.xaxis.label, a.yaxis.label] + a.get_xticklabels() + a.get_yticklabels():
                if not lab.get_visible() or not lab.get_text():
                    continue
                b = lab.get_window_extent(rd)
                w = min(box.x1, b.x1) - max(box.x0, b.x0); h = min(box.y1, b.y1) - max(box.y0, b.y0)
                if w > 1 and h > 1:
                    issues.append(dict(kind=kind, text=lab.get_text()))
            if kind=='legend' and not axes[owner].axison and owner!=i:
                plotbox=a.get_window_extent(rd)
                w=min(box.x1,plotbox.x1)-max(box.x0,plotbox.x0)
                h=min(box.y1,plotbox.y1)-max(box.y0,plotbox.y0)
                if w>1 and h>1:issues.append(dict(kind='legend-strip/data-rectangle',axis=i,overlap_px=[w,h]))
    return issues


def schematic(ax, filename):
    """Place the complete original schematic asset without cropping."""
    asset = FIG / "schematic_assets" / filename
    if not asset.exists(): raise FileNotFoundError(asset)
    ax.imshow(plt.imread(asset), aspect="equal")
    ax.set_gid("schematic")
    ax.axis("off")


def _export(fig, folder, stem, letters_on):
    report = refine_layout(fig)
    letters = add_letters(fig, report) if letters_on else []
    style = audit_style(fig)
    style['actual_categorical_artists']=audit_categorical_artists(fig)
    style.update(alignment_max_residual_pt=max((r["max_residual_pt"] for r in report), default=0),
                 panel_letters=letters, label_overlaps=audit_overlaps(fig))
    style['canvas_mm'] = (fig.get_size_inches()*25.4).tolist()
    style['data_rectangles_mm'] = [dict(gid=a.get_gid(), width=a.get_position().width*fig.get_figwidth()*25.4,
                                      height=a.get_position().height*fig.get_figheight()*25.4)
                                    for a in fig.axes if a.axison]
    QA.mkdir(parents=True, exist_ok=True)
    json.dump(dict(alignment=report, style=style), open(QA / f"{stem}.json", "w", encoding="utf-8"), indent=1)
    meta = {'Author':'CHONG LIU', 'Creator':'CHONG LIU', 'Producer':'CHONG LIU'}
    # Axes tight bounds already include the ticks actually drawn, labels, legends
    # and child insets. Including every Text separately also includes unused
    # locator ticks outside the axis limits, producing large invisible margins.
    extras=[t for t in fig.texts if t.get_visible() and t.get_text()]
    extras += [a for a in fig.findobj(Axes) if a.get_visible()]
    # Explicit labels remain necessary when bbox_extra_artists is supplied:
    # otherwise Matplotlib can omit the far edge of a long axis label.
    for ax in fig.findobj(Axes):
        if ax.get_visible():
            extras += [t for t in [ax.xaxis.label, ax.yaxis.label, ax.title] + ax.texts
                       if t.get_visible() and t.get_text()]
    fig.savefig(folder / f"{stem}.png", dpi=int(os.environ.get('FIG_DPI','600')), bbox_inches="tight", bbox_extra_artists=extras, pad_inches=0.04, facecolor="white", metadata=meta)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", bbox_extra_artists=extras, pad_inches=0.04, facecolor="white", metadata=meta)
    plt.close(fig)
    gc.collect()
    return style


def save(builders, folder, stem, layout, figsize, sizes, letters="abcdefghij"):
    """builders: list of f(ax); layout(fig) -> list of SubplotSpec (one per builder)."""
    out = FIG / folder
    sub = out / "subfigures"
    sub.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.png")) + list(out.glob("*.pdf")) + list(sub.glob("*.png")) + list(sub.glob("*.pdf")):
        old.unlink()
    fig = plt.figure(figsize=np.asarray(figsize) * COMPACT)
    for f, ss, l in zip(builders, layout(fig), letters):
        draw_panel(fig, ss, l, f)
    style = _export(fig, out, stem, True)
    del fig
    gc.collect()
    print(f"{stem}: max alignment residual {style['alignment_max_residual_pt']:.2f} pt; "
          f"unexpected colours {style['n_unexpected']}; label overlaps {len(style['label_overlaps'])}; "
          f"letter overlaps {sum(bool(l['overlaps']) for l in style['panel_letters'])}")
    for f, s, l in zip(builders, sizes, letters):
        fg = plt.figure(figsize=np.asarray(s) * (COMPACT if s[0] >= 14 else .90))
        fg._standalone = True
        gs = fg.add_gridspec(1, 1, left=.03, right=.97, bottom=.04, top=.96)
        draw_panel(fg, gs[0], l, f)
        _export(fg, sub, f"{stem}_{l}", False)
        del fg
        gc.collect()
    (sub / "README.md").write_text(
        "# Subfigures\n\nEach panel is re-drawn on its own canvas by the same draw function used for the combined "
        "figure (not cropped). Panel letters appear only in the combined figure. Layout/style audits: "
        "`figures/qa/`.\n", encoding="utf-8")
    print("saved", out)


# ------------------------------------------------------------------ Albers equal-area (spherical)
def albers(lon, lat, lon0=-96.0, lat0=23.0, lat1=29.5, lat2=45.5):
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    l0, p0, p1, p2 = map(np.radians, (lon0, lat0, lat1, lat2))
    n = (np.sin(p1) + np.sin(p2)) / 2
    C = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = np.sqrt(C - 2 * n * np.sin(lat)) / n
    rho0 = np.sqrt(C - 2 * n * np.sin(p0)) / n
    th = n * (lon - l0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


INSETS = {"AK": (dict(lon0=-154, lat0=50, lat1=55, lat2=65), 0.35, -0.33, -0.03),
          "HI": (dict(lon0=-157, lat0=3, lat1=8, lat2=18), 1.0, -0.06, -0.27)}
SKIP = {"PR", "VI", "GU", "AS", "MP"}
MAP_XLIM, MAP_YLIM = (-0.44, 0.40), (-0.085, 0.51)
ZOOMS = {"Northeast corridor": (-78.6, 38.3, -69.9, 43.0),
         "Carolinas–Georgia": (-85.6, 32.4, -77.5, 36.8),
         "Southern California": (-120.2, 32.5, -115.0, 35.2)}


def project_points(lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    x, y = albers(lon, lat)
    ak = (lat > 50) & (lon < -129)
    hi = (lon < -150) & (lat < 25) & (lat > 15)
    for m, st in [(ak, "AK"), (hi, "HI")]:
        kw, s, dx, dy = INSETS[st]
        xx, yy = albers(lon[m], lat[m], **kw)
        x[m], y[m] = xx * s + dx, yy * s + dy
    keep = (lon < -60) & (lon > -180) & (lat > 15) & np.isfinite(lat) & ~((lat < 25) & (lon > -80))
    return x, y, keep


_STATES = None


def load_states():
    global _STATES
    if _STATES is not None:
        return _STATES
    kml = STATE_KML.read_text(encoding="utf-8")
    out = []
    for pm in re.findall(r"<Placemark.*?</Placemark>", kml, flags=re.S):
        st = re.search(r'<SimpleData name="STUSPS">([^<]+)', pm).group(1)
        if st in SKIP:
            continue
        rings = []
        for coords in re.findall(r"<outerBoundaryIs>.*?<coordinates>(.*?)</coordinates>", pm, flags=re.S):
            xy = np.array([[float(v) for v in p.split(",")[:2]] for p in coords.split()])
            if (st == "AK" and (xy[:, 0] > 0).any()) or len(xy) < 3:
                continue
            if st in INSETS:
                kw, s, dx, dy = INSETS[st]
                x, y = albers(xy[:, 0], xy[:, 1], **kw)
                x, y = x * s + dx, y * s + dy
            else:
                x, y = albers(xy[:, 0], xy[:, 1])
            rings.append(np.c_[x, y])
        out.append((st, rings))
    _STATES = out
    return out


def basemap(ax, values=None, cmap=None, norm=None, nan_hatch=False, lw=0.4, extent=True):
    """Land + state borders, optional state choropleth {STUSPS: value}. Main maps get gid physical_basemap."""
    for st, rings in load_states():
        fc, hatch = LAND, None
        if values is not None:
            v = values.get(st, np.nan)
            if np.isfinite(v):
                fc = cmap(norm(v))
            elif nan_hatch:
                hatch = "////"
        for r in rings:
            pg = ax.fill(r[:, 0], r[:, 1], facecolor=fc, edgecolor=BORDER, lw=lw, zorder=1, hatch=hatch)
            if fc is not LAND:
                pg[0].set_gid("choropleth")   # colour-map interpolation, audited as a numeric map
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if extent:
        ax.set_xlim(*MAP_XLIM); ax.set_ylim(*MAP_YLIM)
        ax.set_gid("physical_basemap")


def zoom_bbox(name):
    lo0, la0, lo1, la1 = ZOOMS[name]
    lo = np.r_[np.linspace(lo0, lo1, 20), np.full(20, lo1), np.linspace(lo1, lo0, 20), np.full(20, lo0)]
    la = np.r_[np.full(20, la0), np.linspace(la0, la1, 20), np.full(20, la1), np.linspace(la1, la0, 20)]
    x, y = albers(lo, la)
    return x.min(), x.max(), y.min(), y.max()


def add_zooms(ax, draw_fn, names=tuple(ZOOMS), width=0.27):
    n = len(names); h = 0.92 / n
    for i, name in enumerate(names):
        x0, x1, y0, y1 = zoom_bbox(name)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=MUTED, lw=1.0, zorder=6))
        ia = ax.inset_axes([1.03, 1 - (i + 1) * h - 0.02 * i, width, h * 0.84])
        basemap(ia, extent=False)
        draw_fn(ia)
        ia.set_xlim(x0, x1); ia.set_ylim(y0, y1); ia.set_aspect("equal", adjustable="datalim")
        for s in ia.spines.values():
            s.set_visible(True); s.set_edgecolor(MUTED); s.set_linewidth(1.0)
        ia.set_title(name, fontsize=MIN_PT, loc="left", pad=3)


def hcolorbar(ax, cmap, norm, label, ticks=None, extend="neither", pos=(0.60, 0.06, 0.34, 0.05), fmt=None):
    cax = ax.inset_axes(list(pos))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend=extend)
    cax._colorbar = cb
    cb.outline.set_linewidth(1.2)
    if ticks is not None:
        cb.set_ticks(ticks)
        if fmt:
            cb.set_ticklabels([fmt(t) for t in ticks])
    cb.ax.minorticks_off()
    cb.ax.tick_params(width=1.2, length=4)
    cb.set_label(label, labelpad=3, fontsize=13)
    return cb


def map_points(ax, lon, lat, **kw):
    x, y, keep = project_points(lon, lat)
    kw.setdefault("rasterized", True)
    return ax.scatter(x[keep], y[keep], **{k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
                                          for k, v in kw.items()})
