"""Author-drawn vector workflow schematic (Fig. 3a and graphical abstract).

Four stages in fixed columns of a unit axes: monitored larger systems -> risk and error models ->
allocation and sampling depth -> systems and people served. Drawn entirely with matplotlib patches in the
nature_new palette; no raster or generated artwork. Column spans match refine_mechanism_round3.workflow3
(stage centres 0.128, 0.392, 0.660, 0.894; arrows between columns are drawn by the caller).
Sizes are given in axes-height units; `kx` (axes height / axes width, physical) converts them to x units so
that glyphs keep their proportions at any panel size.
"""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse


class Glyphs:
    def __init__(self, ax, kx, ink):
        self.ax, self.kx, self.ink = ax, kx, ink

    def rbox(self, x, y, w, h, fc, r=.012, lw=.6, z=3, ec=None):
        """Rounded box; x, y, w in x units, h in y units; r is the corner radius in y units."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r * self.kx}',
                                         mutation_aspect=1 / self.kx, fc=fc, ec=ec or self.ink, lw=lw, zorder=z))

    def tower(self, x, y, s, fc, lw=.6, z=3):
        """Water tower of height s with ground point (x, y)."""
        k, ink = self.kx, self.ink; tw = .62 * s * k; base = .30 * s * k
        for dx0, dx1 in [(-base, -.36 * tw), (base, .36 * tw), (0, 0)]:
            self.ax.plot([x + dx0, x + dx1], [y, y + .58 * s], color=ink, lw=lw, zorder=z, solid_capstyle='round')
        self.ax.plot([x - .62 * base, x + .62 * base], [y + .24 * s, y + .24 * s], color=ink, lw=lw * .8, zorder=z)
        self.rbox(x - tw / 2, y + .58 * s, tw, .30 * s, fc, r=.08 * s, lw=lw, z=z + 1)
        self.ax.add_patch(Ellipse((x, y + .88 * s), tw * .78, .09 * s, fc=fc, ec=ink, lw=lw, zorder=z + 2))
        self.ax.plot([x, x], [y + .92 * s, y + s], color=ink, lw=lw, zorder=z + 2)

    def person(self, x, y, s, fc, lw=.55, z=3):
        k = self.kx
        self.rbox(x - .23 * s * k, y, .46 * s * k, .50 * s, fc, r=.17 * s, lw=lw, z=z)
        self.ax.add_patch(Ellipse((x, y + .70 * s), .30 * s * k, .30 * s, fc=fc, ec=self.ink, lw=lw, zorder=z + 1))

    def ring(self, x, y, d, ec, lw=1.1, z=2):
        self.ax.add_patch(Ellipse((x, y), d * self.kx, d, fc='none', ec=ec, lw=lw, zorder=z))

    def vial(self, x, y, h, fc, full=True, lw=.55, z=3):
        k, w = self.kx, .26 * h * self.kx
        self.rbox(x - w / 2, y, w, h, 'white', r=.08 * h, lw=lw, z=z)
        if full:
            self.ax.add_patch(Rectangle((x - .36 * w, y + .07 * h), .72 * w, .52 * h, fc=fc, ec='none', zorder=z + 1))
        self.ax.add_patch(Rectangle((x - .56 * w, y + h), 1.12 * w, .14 * h, fc=self.ink, ec=self.ink, lw=lw, zorder=z + 1))

    def table(self, x, y, w, h, fc, rows=3, cols=3, lw=.55, z=3):
        ax, ink = self.ax, self.ink
        ax.add_patch(Rectangle((x, y), w, h, fc='white', ec=ink, lw=lw, zorder=z))
        ax.add_patch(Rectangle((x, y + h * (rows - 1) / rows), w, h / rows, fc=fc, ec=ink, lw=lw, zorder=z + 1))
        for r in range(1, rows):
            ax.plot([x, x + w], [y + h * r / rows] * 2, color=ink, lw=lw * .8, zorder=z + 2)
        for c in range(1, cols):
            ax.plot([x + w * c / cols] * 2, [y, y + h], color=ink, lw=lw * .8, zorder=z + 2)


def draw_workflow(ax, pal, ink, kx, y0=.17, y1=.985, fs=10):
    """Draw the workflow in unit-axes coordinates; artwork spans y0..y1. kx = axes height / axes width (physical)."""
    g = Glyphs(ax, kx, ink); H = y1 - y0; Y = lambda f: y0 + f * H
    exceed, clean, model, people, gold = pal[0], pal[1], pal[4], pal[3], pal[6]
    # 1. monitored larger systems: two rows of towers (three exceeding) above a measured-outcome table
    for j in range(2):
        for i in range(4):
            k = j * 4 + i
            g.tower(.045 + i * .055, Y(.52 + .25 * j), .22 * H, exceed if k in (1, 6) else clean)
    g.table(.03, Y(.06), .19, .30 * H, clean, rows=3, cols=4)
    # 2. risk and error models: compact tree with two outputs (risk f, error u)
    nodes = {'r': (.392, .80, model), 'a': (.338, .56, clean), 'b': (.446, .56, people),
             'c': (.310, .32, exceed), 'd': (.366, .32, model), 'e': (.418, .32, people), 'f': (.474, .32, exceed)}
    bw, bh = .044, .11
    for p, ch in [('r', 'a'), ('r', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'e'), ('b', 'f')]:
        (xa, fa, _), (xb, fb, _) = nodes[p], nodes[ch]; ym = Y((fa + fb) / 2)
        ax.plot([xa, xa, xb, xb], [Y(fa) - bh * H / 2, ym, ym, Y(fb) + bh * H / 2], color=ink, lw=.7, zorder=2)
    for x, f, c in nodes.values():
        g.rbox(x - bw / 2, Y(f) - bh * H / 2, bw, bh * H, c, r=.02 * H)
    for k, (lab, c, frac) in enumerate([('f', exceed, .70), ('u', gold, .42)]):
        yb = Y(.10 - .09 * k) if False else Y(.13 - .11 * k)
        ax.add_patch(Rectangle((.31, yb), .17, .07 * H, fc='white', ec=ink, lw=.55, zorder=3))
        ax.add_patch(Rectangle((.31, yb), .17 * frac, .07 * H, fc=c, ec=ink, lw=.55, zorder=4))
        ax.text(.302, yb + .035 * H, lab, ha='right', va='center', fontsize=fs, style='italic', zorder=5)
    # 3. allocation and sampling depth: small systems (selected ones ringed and filled), full vs shortened schedules
    sel = {(1, 2): exceed, (4, 2): people, (2, 1): people, (5, 1): exceed, (0, 0): exceed, (3, 0): people}
    for j in range(3):
        for i in range(6):
            x, y = .575 + i * .033, Y(.42 + .19 * j)
            if (i, j) in sel:
                g.ring(x, y + .07 * H, .17 * H, sel[(i, j)])
            g.tower(x, y, .14 * H, clean if (i, j) in sel else 'white', lw=.5)
    for r, (n, c, lab) in enumerate([(4, clean, 'Full'), (2, people, 'R12')]):
        yv = Y(.20 - .18 * r)
        ax.text(.585, yv + .06 * H, lab, ha='right', va='center', fontsize=fs - .5)
        for v in range(4):
            g.vial(.605 + v * .035, yv, .12 * H, c, full=v < n)
    # 4. systems and people served: towers, a service-population group and a two-target summary
    g.tower(.84, Y(.47), .38 * H, exceed)
    g.tower(.93, Y(.47), .27 * H, clean)
    for i, c in enumerate([clean, exceed, people, model, gold, clean]):
        g.person(.808 + i * .031, Y(.24), .20 * H, c)
    g.table(.80, Y(.02), .19, .17 * H, people, rows=2, cols=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
