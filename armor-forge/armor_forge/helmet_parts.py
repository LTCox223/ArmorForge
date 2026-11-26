"""Close helm as separate parts, found from the concept's gold trim: a wedge visor in front of a skull.

Depth comes from matching heights between orthographic views: the front view gives a part's width
at each height, the side view where it starts and ends front-to-back at the same height.

1. Gold pixels (the trim colour) are picked out of the concept in the front and side views and mapped
   into the metric artboard frame (registration from concept.py).
2. Straight edges of the visor frame are fitted through the gold, which tolerates gaps, then moved
   from the middle of each gold band to its outer edge:
     front view: the outer side lines (|x| = w(z)) and the top "V" (z = z0 + k |x|);
     side view:  the rear edge (y = y_r(z)) and the top edge (z = z0 + k (y - y_front)).
3. Visor: in this concept the slit panel is visible from the front AND the side, so its faces are
   steeply angled: plan section is a wedge from the centre ridge (the side outline's front) back to
   the rear edge at x = +-w. Top and bottom of each column come from both views (averaged).
4. Skull: the helmet outlines with the visor's region removed in the front and side views, lofted by
   helmet.build (so its front sits behind the visor and its face opening is where the visor is).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from . import helmet as hm


# ----------------------------------------------------------------------------- gold
def gold_score(rgb):
    """0..1 'gold trim' score per pixel: warm (red well above blue), not dark, green not too low."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    warm = np.clip((r - b - 30.0) / 30.0, 0, 1)
    lit = np.clip((r - 90.0) / 40.0, 0, 1)
    yellow = np.clip((g - 0.55 * r) / (0.1 * r + 1.0), 0, 1)
    return warm * lit * yellow


@dataclass
class ViewData:
    name: str
    sil: np.ndarray        # bool, artboard frame (piece silhouette)
    gold: np.ndarray       # bool, artboard frame (gold inside the silhouette)
    c0: float
    r0: float
    s: float = 1600.0

    def to_metric(self, rows, cols):
        return (cols + 0.5 - self.c0) / self.s, (self.r0 - rows - 0.5) / self.s

    def gold_points(self):
        r, c = np.nonzero(self.gold)
        return self.to_metric(r, c)

    def region(self, fn):
        """Boolean artboard mask of silhouette pixels where fn(u, z) is True."""
        r, c = np.nonzero(self.sil)
        u, z = self.to_metric(r, c)
        out = np.zeros_like(self.sil)
        keep = fn(u, z)
        out[r[keep], c[keep]] = True
        return out


def view_data(name, art_rgb, reg, cam, sil, s=1600.0, pad=20):
    """Gold of one view resampled into the artboard frame (only around the silhouette)."""
    rows, cols = np.nonzero(sil)
    r0, r1 = max(0, rows.min() - pad), min(sil.shape[0], rows.max() + pad)
    c0, c1 = max(0, cols.min() - pad), min(sil.shape[1], cols.max() + pad)
    rr, cc = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    ax, ay = reg.to_art(cc + 0.5, rr + 0.5)
    g = ndi.map_coordinates(gold_score(art_rgb), [ay - 0.5, ax - 0.5], order=1) > 0.5
    gold = np.zeros_like(sil)
    gold[r0:r1, c0:c1] = g
    return ViewData(name, sil, gold & sil, cam[0], cam[1], s)


# ----------------------------------------------------------------------------- line fitting
def fit_line(a, b, a0_range, slopes, ref, window, tol=0.004, step=0.002, pick="max", frac=0.5):
    """Best line a = a0 + slope * (b - ref) through points (a, b), counting points within tol (m)
    whose b lies in window. pick 'max' = most votes; 'outermost' = largest a0 among lines with at
    least frac of the best vote. -> (a0, slope, votes)."""
    inw = (b >= window[0]) & (b <= window[1])
    a, b = a[inw], b[inw]
    best = []
    for a0 in np.arange(a0_range[0], a0_range[1], step):
        for k in slopes:
            n = int((np.abs(a - (a0 + k * (b - ref))) < tol).sum())
            best.append((n, a0, k))
    best.sort(key=lambda t: -t[0])
    if pick == "outermost":
        top = best[0][0]
        cands = [t for t in best if t[0] >= frac * top]
        n, a0, k = max(cands, key=lambda t: t[1])
    else:
        n, a0, k = best[0]
    return float(a0), float(k), n


def band_edge(a, b, a0, k, ref, window, outward=1.0, reach=0.015, q=90):
    """Shift of a fitted line (a = a0 + k (b - ref)) from the middle of its gold band to the band's
    outer edge: the q-th percentile of the gold's offset on the outward side, within reach."""
    inw = (b >= window[0]) & (b <= window[1])
    r = (a[inw] - (a0 + k * (b[inw] - ref))) * outward
    r = r[np.abs(r) < reach]
    if len(r) < 10:
        return 0.0
    return float(np.clip(np.percentile(r, q), 0.0, reach)) * outward


@dataclass
class VisorLines:
    rear: tuple           # side view: y = y0 + t (z - zm)
    top_side: tuple       # side view: z = z0 + k (y - y_ref)
    sides: tuple          # front view: |x| = x0 + t (z - zm)
    top_front: tuple      # front view: z = z0 + k |x|

    def y_rear(self, z):
        y0, t, zm = self.rear
        return y0 + t * (np.asarray(z) - zm)

    def z_top_side(self, y):
        z0, k, yr = self.top_side
        return z0 + k * (np.asarray(y) - yr)

    def w(self, z):
        x0, t, zm = self.sides
        return x0 + t * (np.asarray(z) - zm)

    def z_top_front(self, x):
        z0, k = self.top_front
        return z0 + k * np.abs(np.asarray(x))

    def as_dict(self):
        return dict(rear=self.rear, top_side=self.top_side, sides=self.sides, top_front=self.top_front)


def detect_visor(front: ViewData, side: ViewData):
    """Fit the visor frame's straight edges through the gold (search windows are fractions of the
    helmet's height and depth, so they carry over to other close helms)."""
    _, sz = side.to_metric(*np.nonzero(side.sil))
    su, _ = side.to_metric(*np.nonzero(side.sil))
    zmin, zmax = float(sz.min()), float(sz.max())
    H = zmax - zmin
    y_front = float(su.min())
    gu, gz = side.gold_points()
    # rear edge: near-vertical, 6..26 cm behind the front, in the lower 70 %
    zm = zmin + 0.35 * H
    y0, t, _ = fit_line(gu, gz, (y_front + 0.06, y_front + 0.26), np.linspace(-0.6, 0.6, 25), zm,
                        (zmin, zmin + 0.7 * H))
    y0 += band_edge(gu, gz, y0, t, zm, (zmin, zmin + 0.7 * H), outward=1.0)
    rear = (y0, t, zm)
    # top edge: near-horizontal, between the front and the rear edge
    ymax = y0 + t * (zmin + 0.6 * H - zm)
    keep = gu < ymax
    z0, k, _ = fit_line(gz[keep], gu[keep], (zmin + 0.3 * H, zmin + 0.85 * H), np.linspace(-0.2, 0.8, 26),
                        y_front, (y_front, ymax))
    z0 += band_edge(gz[keep], gu[keep], z0, k, y_front, (y_front, ymax), outward=1.0)
    top_side = (z0, k, y_front)
    # front view (folded to |x|)
    fu, fz = front.gold_points()
    fa = np.abs(fu)
    _, fsz = front.to_metric(*np.nonzero(front.sil))
    fzmin, fzmax = float(fsz.min()), float(fsz.max())
    FH = fzmax - fzmin
    zmf = fzmin + 0.3 * FH
    x0, tx, _ = fit_line(fa, fz, (0.03, 0.14), np.linspace(-0.3, 0.3, 13), zmf, (fzmin, fzmin + 0.6 * FH),
                         pick="outermost")
    x0 += band_edge(fa, fz, x0, tx, zmf, (fzmin, fzmin + 0.6 * FH), outward=1.0)
    sides = (x0, tx, zmf)
    keep = fa < x0
    z0f, kf, _ = fit_line(fz[keep], fa[keep], (fzmin + 0.3 * FH, fzmin + 0.85 * FH), np.linspace(-1, 1, 41),
                          0.0, (0.0, x0))
    z0f += band_edge(fz[keep], fa[keep], z0f, kf, 0.0, (0.0, x0), outward=1.0)
    return VisorLines(rear, top_side, sides, (z0f, kf))


def visor_regions(lines: VisorLines, front: ViewData, side: ViewData):
    """The visor's area in the front and side views (artboard masks)."""
    rf = front.region(lambda u, z: (np.abs(u) < lines.w(z)) & (z < lines.z_top_front(u)))
    rs = side.region(lambda u, z: (u < lines.y_rear(z)) & (z < lines.z_top_side(u)))
    return rf, rs


# ----------------------------------------------------------------------------- visor geometry
@dataclass
class VisorParams:
    cols: int = 8            # quads from the ridge to each rear edge
    rows: int = 10           # quads from bottom to top
    bulge: float = 0.0       # outward bow of each face (m); 0 = flat planes


def build_visor(lines: VisorLines, side_front: hm.Profile, bottom_front, bottom_side, p=VisorParams()):
    """-> verts (N, 3), quad faces, crease edges (ridge). Plan at height z: ridge (0, y_f(z)),
    rear corners (+-w(z), y_r(z)), straight faces (optionally bowed)."""
    zf, yf_ = side_front.z, side_front.lo

    def plan(sig, z):
        yf = np.interp(z, zf, yf_)
        yr = lines.y_rear(z)
        w = lines.w(z)
        x = sig * w
        y = yf + np.abs(sig) * (yr - yf)
        if p.bulge:
            # outward normal of the face (pointing away from the inside of the wedge)
            nx, ny = (yr - yf), -w
            n = np.hypot(nx, ny)
            off = p.bulge * np.sin(np.pi * np.abs(sig))
            x = x + np.sign(sig) * off * nx / n
            y = y + off * ny / n
        return x, y

    def edge_z(sig, f_front, f_side, z=1.9):
        for _ in range(8):
            x, y = plan(sig, z)
            z = 0.5 * (f_front(x) + f_side(y))
        return z

    top_f = lines.z_top_front
    top_s = lines.z_top_side
    bot_f = lambda x: np.interp(x, *bottom_front)        # noqa: E731
    bot_s = lambda y: np.interp(y, *bottom_side)          # noqa: E731
    sigs = np.linspace(-1, 1, 2 * p.cols + 1)
    verts = []
    for sig in sigs:
        zt = edge_z(sig, top_f, top_s)
        zb = edge_z(sig, bot_f, bot_s, z=1.8)
        for t in np.linspace(0, 1, p.rows + 1):
            z = zb + t * (zt - zb)
            x, y = plan(sig, z)
            verts.append((float(x), float(y), float(z)))
    verts = np.array(verts)
    R = p.rows + 1
    idx = np.arange(len(sigs) * R).reshape(len(sigs), R)
    faces = []
    for i in range(len(sigs) - 1):
        for j in range(p.rows):
            faces.append([int(idx[i, j]), int(idx[i + 1, j]), int(idx[i + 1, j + 1]), int(idx[i, j + 1])])
    # outward: front of the wedge faces -y
    f0 = verts[faces[len(faces) // 4]]
    nrm = np.cross(f0[1] - f0[0], f0[2] - f0[0])
    if nrm[1] > 0:
        faces = [f[::-1] for f in faces]
    ridge = p.cols
    crease = [(int(idx[ridge, j]), int(idx[ridge, j + 1])) for j in range(p.rows)]
    return verts, faces, crease


# ----------------------------------------------------------------------------- skull
def skull_params():
    """Skull loft settings: boxy behind the visor (the beak comes from skull_beak), crisp hem steps."""
    return hm.HelmetParams(n_visor=6.0, min_visor=0.0, hem_sigma=0.7)


def neck_floor(side_sil, cam, s=1600.0):
    """Height of the neck-guard rim: the median bottom edge over the back half of the side view."""
    u, z = hm.bottom_edge(side_sil, *cam, s)
    return float(np.median(z[u > np.median(u)]))


def skull_hem(lines: VisorLines):
    """Hem of the skull, before smoothing: where the visor covers the face (between its rear
    corners) the skull is open up to the visor's top; elsewhere the drawn bottom edge."""
    def fn(o_, x, y):
        h = hm.hem_heights(o_, x, y)
        z_face = lines.z_top_front(x)
        hidden = (np.abs(x) < lines.w(z_face)) & (y < lines.y_rear(z_face) + 0.01)
        return np.where(hidden, z_face, h)
    return fn


def skull_beak(lines: VisorLines, gap=0.004, fade=0.03):
    """Per-height beak that makes the skull's plan pass just behind the visor's rear corners, so the
    skull is the same wedge as the visor where the visor covers it, fading to round above the visor's
    top."""
    z_hi = float(lines.z_top_side(lines.y_rear(1.95)))

    def fn(z, a, yf, yb, n):
        ym, depth = 0.5 * (yf + yb), 0.5 * (yb - yf)
        rx = np.clip((lines.w(z) - gap) / max(a, 1e-6), 0.05, 0.98)
        sy = (1 - rx ** n) ** (1 / n)
        beak = (1 - (ym - (lines.y_rear(z) + gap)) / max(depth * sy, 1e-6)) / rx
        fall = np.clip((z_hi - z) / fade + 1.0, 0.0, 1.0)
        return float(np.clip(beak, 0.0, 1.5) * fall)
    return fn


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def behind_visor(verts, lines: VisorLines, side_front: hm.Profile, gap=0.004):
    """Push skull points that sit in front of the visor's wedge (between its rear corners, up to its
    top edge) back to just behind it."""
    v = verts.copy()
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    w = lines.w(z)
    yf = np.interp(z, side_front.z, side_front.lo)
    yr = lines.y_rear(z)
    a = np.clip(np.abs(x) / w, 0, 1)
    y_w = yf + a * (yr - yf)
    z_top = 0.5 * (lines.z_top_front(x) + lines.z_top_side(y_w))
    hit = (np.abs(x) < w) & (z <= z_top + 0.002) & (y < y_w + gap)
    v[hit, 1] = y_w[hit] + gap
    return v


# ----------------------------------------------------------------------------- whole helmet
@dataclass
class Parts:
    lines: VisorLines
    visor: tuple             # verts, faces, crease
    skull: tuple             # verts, faces, info (helmet.build)
    regions: tuple           # visor masks (front, side)


def build_parts(front_sil, back_sil, side_sil, art_rgb, regs, cams, vp=VisorParams(), sp=None, vp_gap=0.004):
    """Everything from the traced silhouettes, the concept and the registrations."""
    front = view_data("front", art_rgb, regs["front"], cams["front"], front_sil)
    side = view_data("side", art_rgb, regs["side"], cams["side"], side_sil)
    lines = detect_visor(front, side)
    rf, rs = visor_regions(lines, front, side)
    s = front.s
    side_prof = hm.profile(side_sil, *cams["side"], s)
    bf = hm.bottom_edge(front_sil, *cams["front"], s)
    bs = hm.bottom_edge(side_sil, *cams["side"], s)
    visor = build_visor(lines, side_prof, bf, bs, vp)
    # skull: the full outline with the visor's area cut from the front view (its bottom edge there
    # becomes the face opening), the side view cropped at the neck-guard rim (masks often count the
    # neck behind the visor as helmet), slivers opened away
    open_r = int(0.01 * s)
    sk_f = ndi.binary_opening(front_sil & ~rf, structure=_disk(open_r))
    floor = neck_floor(side_sil, cams["side"], s)
    sk_s = side_sil & side.region(lambda u, z: z >= floor)
    skull_o = hm.outlines_from_masks(sk_f, back_sil, sk_s, cams, s)
    # the skull's front comes from the side view without the visor; below the visor's top the skull
    # is hidden, so its brow line is carried straight down (the wedge from skull_beak does the rest)
    front_only = hm.profile(sk_s & ~rs, *cams["side"], s)
    z_cut = lines.top_side[0] + 0.01
    lo = np.interp(skull_o.side.z, front_only.z, front_only.lo)
    lo[skull_o.side.z < z_cut] = np.interp(z_cut, front_only.z, front_only.lo)
    skull_o.side = hm.Profile(skull_o.side.z, lo, skull_o.side.hi)
    sp = sp or skull_params()
    if sp.brow <= 0:
        sp = hm.HelmetParams(**{**vars(sp), "brow": z_cut})
    sv, sf, sinfo = hm.build(skull_o, sp, hem_fn=skull_hem(lines), beak_fn=skull_beak(lines, vp_gap))
    sv = behind_visor(sv, lines, side_prof, vp_gap)
    sinfo["crease"] = []          # the skull is smooth; its front is open behind the visor
    skull = (sv, sf, sinfo)
    return Parts(lines, visor, skull, (rf, rs))


# ----------------------------------------------------------------------------- checking
def project_mask(meshes, view, cam, shape, s=1600.0):
    """Rasterize meshes [(verts, faces), ...] into a view's artboard frame (orthographic)."""
    from PIL import Image, ImageDraw
    c0, r0 = cam
    img = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for verts, faces in meshes:
        u = {"front": verts[:, 0], "back": -verts[:, 0], "side": verts[:, 1]}[view]
        cols, rows = c0 + u * s, r0 - verts[:, 2] * s
        for f in faces:
            d.polygon([(cols[i], rows[i]) for i in f], fill=255)
    return np.asarray(img) > 0


def silhouette_iou(parts: Parts, masks, cams):
    """{view: IoU} of the built parts against the given silhouettes."""
    meshes = [(parts.visor[0], parts.visor[1]), (parts.skull[0], parts.skull[1])]
    out = {}
    for v, m in masks.items():
        b = project_mask(meshes, v, cams[v], m.shape)
        out[v] = float((b & m).sum() / max((b | m).sum(), 1))
    return out
