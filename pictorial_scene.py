"""Shared vector scene for the graphical abstract and Fig. 3a (190 x 76 data units = mm at full size).

Every element is an original geometric path; no raster or generated artwork. The scene does not change rcParams:
callers set typography. Two label sets share one geometry: GA_LABELS carries the headline results for the graphical
abstract, FIGURE_LABELS names the framework stages for the manuscript figure.
"""
from pathlib import Path
import json, math, sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'strengthening/method/figure_helpers/figkit/publication_helpers'))
from scientific_palettes import get_palette  # noqa: E402
from matplotlib.patches import Rectangle, Polygon, Circle, Ellipse, FancyArrowPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

PAL = get_palette('nature_new'); CORAL, BLUE, GREEN, TEAL, PURPLE, PINK, GOLD, MAUVE = PAL
INK = '#30343B'; MUTED = '#6F7782'
W, H = 190, 76
TEXT = []

# (x, y, text, point size); identical geometry for both label sets
GA_LABELS = {
    'title': (32, 69.3, 'PFOA / PFOS', 13.0), 'subtitle': (32, 62.7, 'U.S. small water systems', 10.4),
    'data': (31.8, 13.4, '970 measured / 56,591 systems', 10.2),
    'target1': (96, 68.6, '≈4,450', 19.0), 'target1_note': (96, 61.5, 'exceeding systems', 10.5),
    'target2': (94.7, 12.7, '≈1.9 M', 18.5), 'target2_note': (94.7, 5.7, 'people served', 10.7),
    'full': (143.6, 68.4, 'Full', 10.3), 'reduced': (173.1, 68.4, 'Reduced', 10.3),
    'completion': (160.4, 32.0, '20% random completion', 10.2),
    'design': (160.1, 23.3, 'Population MSE ↓70–83%', 12.5),
    'design_note': (160.1, 13.4, 'vs full-schedule Neyman', 9.3),
    'design_scope': (160.1, 6.8, 'MI/WA simulations · equal budgets', 8.7),
}
FIGURE_LABELS = dict(GA_LABELS)
FIGURE_LABELS.update({
    'data': (31.8, 13.4, '970 measured / 56,591 systems', 10.2),
    'data_note': (31.8, 6.6, 'larger-system risk and error models', 9.0),
    'target1': (96, 68.6, 'Exceeding systems', 12.5), 'target1_note': (96, 61.8, 'utility target', 10.0),
    'target2': (94.7, 12.7, 'People served', 12.5), 'target2_note': (94.7, 5.9, 'population target', 10.0),
    'design': (160.1, 23.3, 'Allocation × sampling depth', 11.5),
    'design_note': (160.1, 13.4, 'equal analytical budgets', 9.3),
    'design_scope': (160.1, 6.8, 'finite-population and MI/WA tests', 8.7),
})


def line(ax, xs, ys, c=INK, lw=.8, **kw):
    return ax.plot(xs, ys, color=c, lw=lw, solid_capstyle='round', solid_joinstyle='round', **kw)[0]
def rect(ax, x, y, w, h, fc='white', ec=INK, lw=.8, **kw):
    p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, **kw); ax.add_patch(p); return p
def poly(ax, points, fc='white', ec=INK, lw=.8, **kw):
    p = Polygon(points, closed=True, fc=fc, ec=ec, lw=lw, joinstyle='round', **kw); ax.add_patch(p); return p
def text(ax, x, y, t, size=10.5, ha='center', color=INK, **kw):
    q = ax.text(x, y, t, ha=ha, va='center', fontsize=size, fontweight='normal', fontstyle='normal', color=color, linespacing=1.12, **kw)
    TEXT.append(q); return q
def arrow(ax, x0, y0, x1, y1, color=INK, lw=.85, scale=7, **kw):
    p = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=scale, color=color, lw=lw, **kw); ax.add_patch(p); return p
def curve(ax, points, color=TEAL, lw=1.5, arrowhead=False, **kw):
    verts = points; codes = [MPath.MOVETO] + [MPath.CURVE4] * 3
    ax.add_patch(PathPatch(MPath(verts, codes), fill=False, ec=color, lw=lw, capstyle='round', **kw))
    if arrowhead:
        x, y = verts[-1]; a, b = verts[-2]; angle = math.atan2(y - b, x - a); u = 1.3
        poly(ax, [(x, y), (x - u * math.cos(angle) + .45 * math.sin(angle), y - u * math.sin(angle) - .45 * math.cos(angle)),
                  (x - u * math.cos(angle) - .45 * math.sin(angle), y - u * math.sin(angle) + .45 * math.cos(angle))], fc=color, ec=color, lw=.4)
def ellipse(ax, x, y, w, h, c, edge=INK, lw=.8, **kw):
    p = Ellipse((x, y), w, h, fc=c, ec=edge, lw=lw, **kw); ax.add_patch(p); return p
def house(ax, x, y, s=1, c=BLUE):
    # A restrained axonometric house: depth comes from geometry, not lighting.
    poly(ax, [(x, y), (x + 7 * s, y), (x + 7 * s, y + 5.6 * s), (x, y + 5.6 * s)], fc=c, lw=.65)
    poly(ax, [(x + 7 * s, y), (x + 9.5 * s, y + 1.8 * s), (x + 9.5 * s, y + 7.4 * s), (x + 7 * s, y + 5.6 * s)], fc=c, lw=.65, alpha=.7)
    poly(ax, [(x - .5 * s, y + 5.6 * s), (x + 3.5 * s, y + 9 * s), (x + 7.5 * s, y + 5.6 * s)], fc='white', lw=.7)
    poly(ax, [(x + 3.5 * s, y + 9 * s), (x + 6 * s, y + 10.8 * s), (x + 10 * s, y + 7.4 * s), (x + 7.5 * s, y + 5.6 * s)], fc=BLUE, lw=.7, alpha=.75)
    rect(ax, x + 2.8 * s, y, 1.6 * s, 3 * s, fc='white', lw=.5)
    for dx in [1.0, 5.2]: rect(ax, x + dx * s, y + 3.3 * s, 1.15 * s, 1.3 * s, lw=.45)
def tank(ax, x, y, s=1, c=TEAL):
    for dx in [-3.8, 3.8]: line(ax, [x + dx * s, x + dx * .65 * s], [y, y + 13 * s], lw=.8)
    line(ax, [x - 3.1 * s, x + 3.1 * s], [y + 5.2 * s, y + 5.2 * s], lw=.65)
    line(ax, [x - 2.7 * s, x + 2.7 * s], [y + 8.6 * s, y + 8.6 * s], lw=.65)
    rect(ax, x - 5.3 * s, y + 12.7 * s, 10.6 * s, 6.5 * s, fc=c, lw=.8)
    ellipse(ax, x, y + 12.7 * s, 10.6 * s, 2.5 * s, c, lw=.8)
    ellipse(ax, x, y + 19.2 * s, 10.6 * s, 3 * s, c, lw=.8)
    line(ax, [x, x], [y - 1 * s, y + 12.7 * s], lw=.65)
    line(ax, [x, x], [y + 20.8 * s, y + 22 * s], lw=.7)
def building(ax, x, y, s=1, c=BLUE):
    rect(ax, x, y, 11 * s, 8 * s, fc=c, lw=.7)
    poly(ax, [(x + 11 * s, y), (x + 14 * s, y + 2 * s), (x + 14 * s, y + 10 * s), (x + 11 * s, y + 8 * s)], fc=c, lw=.7, alpha=.7)
    poly(ax, [(x, y + 8 * s), (x + 3 * s, y + 10 * s), (x + 14 * s, y + 10 * s), (x + 11 * s, y + 8 * s)], fc='white', lw=.7)
    for dx in [1.2, 4.1, 7.0]: rect(ax, x + dx * s, y + 4.6 * s, 1.8 * s, 1.8 * s, lw=.45)
    rect(ax, x + 7.1 * s, y, 2 * s, 3.3 * s, lw=.45)
def person(ax, x, y, s=1, c=PURPLE):
    ellipse(ax, x, y + 7 * s, 2.6 * s, 2.6 * s, c, lw=.65)
    verts = [(x - 1.9 * s, y), (x - 1.9 * s, y + 4 * s), (x - 1.9 * s, y + 5 * s), (x, y + 5 * s), (x + 1.9 * s, y + 5 * s), (x + 1.9 * s, y + 4 * s), (x + 1.9 * s, y), (x - 1.9 * s, y)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=c, ec=INK, lw=.65, joinstyle='round'))
def vial(ax, x, y, s=1, c=TEAL):
    rect(ax, x, y, 3.4 * s, 10.0 * s, lw=.65)
    rect(ax, x + .5 * s, y + .6 * s, 2.4 * s, 4.2 * s, fc=c, ec='none')
    rect(ax, x - .2 * s, y + 10.0 * s, 3.8 * s, 1.45 * s, fc=c, lw=.6)
    for yy in [6.3, 7.8]: line(ax, [x + 2.3 * s, x + 3.2 * s], [y + yy * s, y + yy * s], lw=.45)
def database(ax, x, y, s=1):
    for yy, c in [(0, BLUE), (3.0, TEAL), (6.0, BLUE)]:
        rect(ax, x - 6 * s, y + yy * s, 12 * s, 3.3 * s, fc=c, lw=.65, alpha=.9)
        ellipse(ax, x, y + yy * s, 12 * s, 3.4 * s, c, lw=.65)
    ellipse(ax, x, y + 9.3 * s, 12 * s, 3.4 * s, BLUE, lw=.7)
    for yy in [1.2, 4.2, 7.2]: ellipse(ax, x + 3.3 * s, y + yy * s, .65 * s, .65 * s, 'white', edge='none')
def map_context(ax):
    d = json.loads((HERE / 'graphical_abstract_us_outline.json').read_text(encoding='utf-8'))
    # This small context map has no outcome locations or concentration encoding.
    x0, y0, w, h = 2.4, 26.5, 59.2, 32.5
    for state in d['states']:
        for ring in state['rings']:
            pts = [(x0 + (lon + 124.8) / 58.0 * w, y0 + (lat - 24.4) / 25.1 * h) for lon, lat in ring]
            poly(ax, pts, fc=BLUE, ec='white', lw=.32, alpha=.35, zorder=1)
def calendar(ax, x, y, s=1, reduced=False):
    rect(ax, x, y, 20 * s, 20.4 * s, lw=.8)
    rect(ax, x, y + 15.7 * s, 20 * s, 4.7 * s, fc=PURPLE if not reduced else TEAL, lw=.7)
    for dx in [4.2, 15.8]: line(ax, [x + dx * s, x + dx * s], [y + 18.5 * s, y + 22 * s], lw=.75)
    for row in range(3):
        for col in range(4):
            xx = x + (3 + 4.65 * col) * s; yy = y + (3.1 + 4.7 * row) * s
            if not reduced or col == 0:
                ellipse(ax, xx, yy, 1.25 * s, 1.25 * s, PURPLE if not reduced else TEAL, edge='none')
            else: line(ax, [xx - .6 * s, xx + .6 * s], [yy, yy], c=MUTED, lw=.45)
    if reduced:
        # Random-completion loop is pictorial; this single slot has no rate scale.
        ax.add_patch(Circle((x + 15.0 * s, y + 3.1 * s), 2 * s, fc=GOLD, ec=INK, lw=.55))
        line(ax, [x + 13.9 * s, x + 14.7 * s, x + 16.2 * s], [y + 3.0 * s, y + 2.3 * s, y + 4 * s], lw=.6)


def draw_scene(ax, labels=GA_LABELS):
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    lab = lambda k: text(ax, labels[k][0], labels[k][1], labels[k][2], labels[k][3])
    # Flat, low-opacity colour fields unify the scene without cards or gradients.
    ellipse(ax, 31.5, 39, 63, 57, BLUE, edge='none', alpha=.09, zorder=-4)
    ellipse(ax, 95.7, 39, 59, 63, TEAL, edge='none', alpha=.10, zorder=-4)
    ellipse(ax, 160.5, 39, 59, 58, PURPLE, edge='none', alpha=.08, zorder=-4)
    # A single visual current, rather than disconnected columns.
    curve(ax, [(54, 39), (64, 39), (67, 44), (73, 44)], color=TEAL, lw=5.5, alpha=.24, zorder=-2)
    curve(ax, [(116, 42), (126, 42), (130, 50), (138, 50)], color=PURPLE, lw=5.5, alpha=.24, zorder=-2)
    arrow(ax, 61, 39.5, 68.5, 42.3, color=MUTED, lw=.9, scale=7)
    arrow(ax, 124.8, 45.6, 132, 49.1, color=MUTED, lw=.9, scale=7)

    # Source layer: geography + physical observations + a system registry.
    lab('title'); lab('subtitle')
    map_context(ax)
    for x, y, s, c in [(13, 23, 1.03, TEAL), (19, 21.8, 1.18, BLUE), (26.1, 23, .99, TEAL)]: vial(ax, x, y, s, c)
    database(ax, 46.2, 24, 1.1)
    # Merge glyph: measured records are linked to the full system registry.
    curve(ax, [(28, 21), (31, 17), (37, 17), (40, 20)], color=TEAL, lw=1.0, arrowhead=True)
    lab('data')
    if 'data_note' in labels: lab('data_note')

    # Main scene: estimated affected facilities physically serve the community.
    lab('target1'); lab('target1_note')
    poly(ax, [(72, 35), (104, 35), (116, 42), (84, 42)], fc=TEAL, ec='none', alpha=.20)
    building(ax, 75.7, 39.5, .90, c=BLUE)
    tank(ax, 99, 36.8, .90, c=TEAL)
    # Supply main, valves and service branches make the relationship visible.
    line(ax, [99, 99, 111.5, 111.5], [35.8, 32.2, 32.2, 27], c=TEAL, lw=1.8)
    line(ax, [81.5, 81.5, 111.5], [30.0, 32.2, 32.2], c=TEAL, lw=1.8)
    line(ax, [95, 95], [29.5, 32.2], c=TEAL, lw=1.5)
    ellipse(ax, 103.6, 32.2, 2, 2, 'white', lw=.65)
    line(ax, [103.6, 103.6], [30.9, 33.6], lw=.65)
    house(ax, 75.0, 21.9, .82, c=GREEN)
    house(ax, 88.7, 21.6, .88, c=BLUE)
    house(ax, 103.0, 24.1, .70, c=TEAL)
    for x, y, s, c in [(107.1, 17.5, .78, PURPLE), (112, 18.8, .69, BLUE), (116, 17.6, .78, TEAL)]: person(ax, x, y, s, c)
    lab('target2'); lab('target2_note')

    # Targeted shortening + explicit random full-schedule completion.
    lab('full'); lab('reduced')
    calendar(ax, 134.0, 42.0, .94, False)
    calendar(ax, 163.7, 42.0, .94, True)
    arrow(ax, 154.6, 52.0, 160.8, 52.0, color=INK, lw=.85, scale=7)
    # Curved return to a full schedule, signalled by the gold completion mark.
    ax.add_patch(FancyArrowPatch((178.3, 40.5), (146.2, 40.5), connectionstyle='arc3,rad=-.25', arrowstyle='-|>',
                                 mutation_scale=7, color=GOLD, lw=1.5))
    lab('completion'); lab('design'); lab('design_note'); lab('design_scope')
