"""Trimmed plates: a four-sided armor plate fitted to the concept, built as flat panels with a rim.

Armor plates in these concepts have a trim band around their edge (smoother than the plate) and
their curvature shows as a few creases between flat panels. So a plate is:

  * four edges (each a quadratic curve: two end corners shared with the neighbours plus one middle
    control point) and a smooth surface stretched between them (Coons patch);
  * fitted to the concept: its projection in front, back and side should cover the piece's colour
    (silhouette overlap) and its edge loop should land on the trim (distance to the gold), with no
    folds and fair edges;
  * built as flat panels: the surface is sampled only on the crease lines (rows `creases_u`, columns
    `creases_v`) and panels are straight between them, a raised rim runs round the edge, and an
    optional lip turns up along one edge (the neck edge of a pauldron).

Edges and parameters: u runs from edge "u0" (e.g. the neck edge) to "u1" (the arm edge), v from
"v0" (front) to "v1" (back). Corners: c00 = (u0, v0), c10 = (u1, v0), c01 = (u0, v1), c11 = (u1, v1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

S = 1600.0


# ----------------------------------------------------------------------------- views
@dataclass
class PlateView:
    name: str
    mask: np.ndarray        # bool, downsampled artboard (the piece's colour, one side only)
    c0: float               # downsampled origin column (u = 0)
    r0: float               # downsampled origin row (z = 0)
    ds: int
    gold: np.ndarray        # (N, 2) trim points in (u, z) metres, u as in project()
    hidden_ok: np.ndarray = None   # bool, downsampled: pieces in front of this one (covering them is free)
    turn: float = 0.0              # side view only: drawn turned by this angle (radians), see project()

    @property
    def ppm(self):
        return S / self.ds


def project(P, view, turn=0.0, centre=None):
    """u (across the view) and z of points: front u = x, back u = -x, side u = y.
    turn: the side view drawn turned by this angle (radians) about a vertical axis through `centre`
    (artists turn pieces a little toward the viewer to show depth); u = y cos - x sin about it."""
    if view == "side" and turn:
        c = np.zeros(3) if centre is None else centre
        u = (P[..., 1] - c[1]) * np.cos(turn) - (P[..., 0] - c[0]) * np.sin(turn) + c[1]
        return u, P[..., 2]
    u = {"front": P[..., 0], "back": -P[..., 0], "side": P[..., 1]}[view]
    return u, P[..., 2]


def make_view(name, full_mask, cam, gold_full=None, side=+1, ds=4, n_gold=600, seed=0, in_front=None,
              edge_band=None):
    """full_mask: the piece's colour in the artboard frame (4096 px, 1600 px/m). side=+1 keeps the
    wearer's left (+x) part in the front and back views. gold_full: trim pixels (same frame).
    in_front: colours of pieces that overlap this one (e.g. the cap over a lame); the plate may
    reach under them without penalty. side=0 keeps both halves (pieces on the centre line).
    edge_band: keep only trim pixels within this distance (m) of the piece's outline (so decorative
    gold inside a plate does not count as its edge)."""
    m = full_mask.copy()
    c0, r0 = cam
    c = int(c0)
    if side and ((name == "front" and side > 0) or (name == "back" and side < 0)):
        m[:, :c] = False                 # keep u > 0
    elif side and name in ("front", "back"):
        m[:, c:] = False                 # keep u < 0
    H, W = m.shape
    small = m[:H - H % ds, :W - W % ds].reshape(H // ds, ds, W // ds, ds).mean((1, 3)) > 0.5
    hid = None
    if in_front is not None:
        f = in_front[:H - H % ds, :W - W % ds]
        hid = f.reshape(H // ds, ds, W // ds, ds).mean((1, 3)) > 0.5
    gold = np.zeros((0, 2))
    if gold_full is not None:
        g = gold_full & ndi.binary_dilation(m, iterations=8)
        if edge_band:
            r_ = int(edge_band * S)
            g &= ~ndi.binary_erosion(m, iterations=r_)
        gy, gx = np.nonzero(g)
        if len(gx):
            sel = np.random.default_rng(seed).choice(len(gx), min(n_gold, len(gx)), replace=False)
            gold = np.stack([(gx[sel] + .5 - c0) / S, (r0 - gy[sel] - .5) / S], 1)
    return PlateView(name, small, c0 / ds, r0 / ds, ds, gold, hid)


# ----------------------------------------------------------------------------- surface
def _bez(a, m, b, t):
    t = np.asarray(t, float)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * m + t ** 2 * b


@dataclass
class Plate:
    """Corners and edge middle controls (all (3,) arrays)."""
    c00: np.ndarray
    c10: np.ndarray
    c01: np.ndarray
    c11: np.ndarray
    m_u0: np.ndarray   # middle control of edge u = 0 (c00 -> c01)
    m_u1: np.ndarray   # edge u = 1 (c10 -> c11)
    m_v0: np.ndarray   # edge v = 0 (c00 -> c10)
    m_v1: np.ndarray   # edge v = 1 (c01 -> c11)

    def vector(self):
        return np.concatenate([self.c00, self.c10, self.c01, self.c11, self.m_u0, self.m_u1, self.m_v0, self.m_v1])

    @staticmethod
    def from_vector(x):
        return Plate(*np.asarray(x, float).reshape(8, 3))

    def centre(self):
        return 0.25 * (self.c00 + self.c10 + self.c01 + self.c11)

    def symmetric_vector(self, axis=1):
        """Parameters of the plate made mirror-symmetric (v0 <-> v1) about a plane normal to `axis`
        (1: front/back, y = const): c00, c10, the u-edge middles without that axis, m_v0, plane."""
        k = [i for i in range(3) if i != axis]
        plane = 0.25 * (self.c00[axis] + self.c01[axis] + self.c10[axis] + self.c11[axis])
        return np.concatenate([self.c00, self.c10, self.m_u0[k], self.m_u1[k], self.m_v0, [plane]])

    @staticmethod
    def from_symmetric(x, axis=1):
        x = np.asarray(x, float)
        c00, c10, mv0, plane = x[0:3], x[3:6], x[10:13], x[13]
        k = [i for i in range(3) if i != axis]

        def mir(p):
            q = p.copy()
            q[axis] = 2 * plane - p[axis]
            return q

        def mid(xy):
            q = np.zeros(3)
            q[k] = xy
            q[axis] = plane
            return q
        return Plate(c00, c10, mir(c00), mir(c10), mid(x[6:8]), mid(x[8:10]), mv0, mir(mv0))

    def surface(self, u, v):
        """Coons patch at the grid u x v -> (len(u), len(v), 3)."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        C0 = _bez(self.c00, self.m_u0, self.c01, v)
        C1 = _bez(self.c10, self.m_u1, self.c11, v)
        D0 = _bez(self.c00, self.m_v0, self.c10, u)
        D1 = _bez(self.c01, self.m_v1, self.c11, u)
        U, V = u[:, None, None], v[None, :, None]
        return ((1 - U) * C0[None] + U * C1[None] + (1 - V) * D0[:, None] + V * D1[:, None]
                - ((1 - U) * (1 - V) * self.c00 + U * (1 - V) * self.c10 + (1 - U) * V * self.c01
                   + U * V * self.c11))


def _loop(Sm):
    return np.concatenate([Sm[0, :], Sm[1:, -1], Sm[-1, -2::-1], Sm[-2:0:-1, 0]])


def _raster(Sm, view: PlateView, centre=None):
    img = Image.new("L", (view.mask.shape[1], view.mask.shape[0]), 0)
    d = ImageDraw.Draw(img)
    u, z = project(Sm, view.name, view.turn, centre)
    cx, cy = view.c0 + u * view.ppm, view.r0 - z * view.ppm
    for i in range(Sm.shape[0] - 1):
        for j in range(Sm.shape[1] - 1):
            d.polygon([(cx[i, j], cy[i, j]), (cx[i + 1, j], cy[i + 1, j]), (cx[i + 1, j + 1], cy[i + 1, j + 1]),
                       (cx[i, j + 1], cy[i, j + 1])], fill=255)
    return np.asarray(img) > 0


def _seg_dist(P, A, B):
    AB = B - A
    t = np.clip(((P[:, None] - A[None]) * AB[None]).sum(-1) / np.maximum((AB ** 2).sum(-1), 1e-12), 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    return np.sqrt(((P[:, None] - Q) ** 2).sum(-1)).min(1)


@dataclass
class BodyProfile:
    """The body under a plate as a stack of elliptical cross-sections (e.g. the mannequin's torso):
    half-width a(z), centre y yc(z), half-depth b(z)."""
    z: np.ndarray
    a: np.ndarray
    yc: np.ndarray
    b: np.ndarray
    gap: float = 0.02        # the plate stays at least this far off the body
    max_off: float = 0.035   # ...and no more than this beyond the gap (it wraps the body)

    def inside(self, P):
        """How far points are inside the body + gap (0 outside; ~1 at the centre)."""
        z = P[..., 2]
        a = np.interp(z, self.z, self.a) + self.gap
        b = np.interp(z, self.z, self.b) + self.gap
        yc = np.interp(z, self.z, self.yc)
        r2 = (P[..., 0] / a) ** 2 + ((P[..., 1] - yc) / b) ** 2
        inz = (z >= self.z.min()) & (z <= self.z.max())
        return np.where(inz, np.clip(1 - np.sqrt(r2), 0, None), 0.0)

    def too_far(self, P):
        """How far (m) points stand off beyond gap + max_off (0 if closer)."""
        z = P[..., 2]
        a = np.interp(z, self.z, self.a) + self.gap
        b = np.interp(z, self.z, self.b) + self.gap
        yc = np.interp(z, self.z, self.yc)
        dx, dy = P[..., 0], P[..., 1] - yc
        r = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
        d = np.hypot(dx, dy) * (1 - 1 / np.maximum(r, 1e-9))
        inz = (z >= self.z.min()) & (z <= self.z.max())
        return np.where(inz, np.clip(d - self.max_off, 0, None), 0.0)


def body_from_mesh(V, T, z_range, step=0.02):
    """Elliptical cross-sections of a mesh (its edges cut every `step` metres)."""
    E = {tuple(sorted(e)) for t in T for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    E = np.array(sorted(E))
    zs, a, yc, b = [], [], [], []
    for z in np.arange(z_range[0], z_range[1] + 1e-9, step):
        A, B = V[E[:, 0]], V[E[:, 1]]
        m = (A[:, 2] - z) * (B[:, 2] - z) <= 0
        if not m.any():
            continue
        A, B = A[m], B[m]
        t = (z - A[:, 2]) / np.where(B[:, 2] != A[:, 2], B[:, 2] - A[:, 2], 1)
        P = A + (B - A) * t[:, None]
        zs.append(z)
        a.append(np.abs(P[:, 0]).max())
        yc.append(0.5 * (P[:, 1].min() + P[:, 1].max()))
        b.append(0.5 * (P[:, 1].max() - P[:, 1].min()))
    return BodyProfile(np.array(zs), np.array(a), np.array(yc), np.array(b))


@dataclass
class FitWeights:
    trim: float = 8.0       # per metre of mean distance from trim pixels to the edge loop
    fold: float = 5.0
    length: float = 0.3     # per metre of edge loop
    trim_cap: float = 0.03  # distances clipped here (trim drawn on other pieces nearby)
    body: float = 20.0      # per unit of mean penetration into the body + gap
    hug: float = 40.0       # per metre of mean stand-off beyond the body's max_off


def score(plate: Plate, views, w=FitWeights(), detail=False, body: BodyProfile = None):
    Sm = plate.surface(np.linspace(0, 1, 9), np.linspace(0, 1, 13))
    loop = _loop(Sm)
    centre = plate.centre()
    tot, out = 0.0, {}
    for vw in views:
        b = _raster(Sm, vw, centre)
        extra = b & ~vw.mask
        if vw.hidden_ok is not None:
            extra &= ~vw.hidden_ok
        iou = (b & vw.mask).sum() / max(vw.mask.sum() + extra.sum(), 1)
        trim = 0.0
        if len(vw.gold):
            u, z = project(loop, vw.name, vw.turn, centre)
            poly = np.stack([u, z], 1)
            trim = float(np.minimum(_seg_dist(vw.gold, poly[:-1], poly[1:]), w.trim_cap).mean())
        tot += (1 - iou) + w.trim * trim
        out[vw.name] = dict(iou=float(iou), trim_mm=trim * 1000)
    a = Sm[1:, :-1] - Sm[:-1, :-1]
    b_ = Sm[:-1, 1:] - Sm[:-1, :-1]
    n = np.cross(a, b_)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    mean = n.reshape(-1, 3).mean(0)
    mean /= max(np.linalg.norm(mean), 1e-12)
    fold = float(np.clip(0.3 - (n @ mean), 0, None).mean())
    length = float(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum())
    tot += w.fold * fold + w.length * length
    out.update(fold=fold, length=length)
    if body is not None:
        pen = float(body.inside(Sm).mean())
        far = float(body.too_far(Sm).mean())
        tot += w.body * pen + w.hug * far
        out["body"] = pen
        out["standoff_mm"] = far * 1000
    return (tot, out) if detail else tot


def fit(views, init: Plate, w=FitWeights(), maxiter=6000, symmetric=True, sym_axis=1, fit_turn=False,
        turn_max=np.radians(25), body: BodyProfile = None, extra=None):
    """Fit the plate. symmetric: mirror-symmetric about a plane normal to sym_axis (pauldron front /
    back: 1), which also keeps the side view's drawing cheats from bending it. fit_turn: also fit the
    side view's turn (the views' `turn` is the start). body: keep off / close to the body.
    extra(plate) -> penalty: piece-specific rules (e.g. where its edges must meet another plate).
    -> (plate, turn)."""
    from scipy.optimize import minimize
    side = [vw for vw in views if vw.name == "side"]
    turn0 = side[0].turn if side else 0.0
    if symmetric:
        x0 = init.symmetric_vector(sym_axis)
        make = lambda x: Plate.from_symmetric(x, sym_axis)  # noqa: E731
    else:
        x0 = init.vector()
        make = Plate.from_vector
    if fit_turn and side:
        x0 = np.concatenate([x0, [turn0]])

    def f(x):
        if fit_turn and side:
            for vw in side:
                vw.turn = x[-1]
            p_ = make(x[:-1])
            return (score(p_, views, w, body=body) + 5.0 * max(0.0, abs(x[-1]) - turn_max)
                    + (extra(p_) if extra else 0.0))
        p_ = make(x)
        return score(p_, views, w, body=body) + (extra(p_) if extra else 0.0)
    res = minimize(f, x0, method="Powell", options=dict(maxiter=maxiter, xtol=1e-4, ftol=1e-5))
    if fit_turn and side:
        turn = float(res.x[-1])
        for vw in side:
            vw.turn = turn
        return make(res.x[:-1]), turn
    return make(res.x), turn0


def init_from_views(front_mask, back_mask, side_mask, cams, side=+1):
    """Starting plate from the outline extremes (full-res artboard masks of the piece). Front view:
    the innermost point is the u0/v0 corner, the lowest the u1/v0 corner, the highest the top of
    edge u0, the outermost the top of edge u1; the back view gives the v1 corners; depth (y) comes
    from the side view at the same height."""
    def pts(m, cam, sgn):
        r, c = np.nonzero(m)
        return (c + .5 - cam[0]) / S * sgn, (cam[1] - r - .5) / S
    fx, fz = pts(front_mask, cams["front"], 1)
    bx, bz = pts(back_mask, cams["back"], -1)
    k = fx * side > 0
    fx, fz = fx[k], fz[k]
    k = bx * side > 0
    bx, bz = bx[k], bz[k]
    sy, sz = pts(side_mask, cams["side"], 1)

    def y_at(z, front):
        z = np.clip(z, sz.min() + 0.003, sz.max() - 0.003)
        kk = np.abs(sz - z) < 0.004
        return sy[kk].min() if front else sy[kk].max()
    ax = np.abs
    i = np.argmin(ax(fx))
    c00 = np.array([fx[i], y_at(fz[i], True), fz[i]])
    i = np.argmin(fz)
    c10 = np.array([fx[i], y_at(fz[i], True), fz[i]])
    i = np.argmin(ax(bx))
    c01 = np.array([bx[i], y_at(bz[i], False), bz[i]])
    i = np.argmin(bz)
    c11 = np.array([bx[i], y_at(bz[i], False), bz[i]])
    i = np.argmax(fz)
    top0 = np.array([fx[i], 0.5 * (c00[1] + c01[1]), fz[i]])
    i = np.argmax(ax(fx))
    top1 = np.array([fx[i], 0.5 * (c10[1] + c11[1]), fz[i]])
    return Plate(c00, c10, c01, c11,
                 m_u0=2 * top0 - 0.5 * (c00 + c01), m_u1=2 * top1 - 0.5 * (c10 + c11),
                 m_v0=0.5 * (c00 + c10) + [0, -0.02, 0], m_v1=0.5 * (c01 + c11) + [0, 0.02, 0])


def seam_rule(y_seam, side=-1, overlap=0.01, weight=2000.0):
    """Penalty keeping a wrapped plate's side edges (v = 0 and v = 1) at the depth where it meets the
    opposite plate: y_seam, reaching `overlap` past it (side -1: a breastplate, in front; +1: a
    backplate)."""
    target = y_seam - side * overlap

    def rule(plate: Plate):
        E = plate.surface(np.linspace(0, 1, 9), [0.0, 1.0])
        return weight * float(np.mean((E[..., 1] - target) ** 2))
    return rule


def init_wrap(face_mask, side_mask, cams, view="front", y_side=None):
    """Starting plate for a piece wrapped round the torso and centred on x = 0 (a breastplate:
    view "front"; a backplate: "back"). u runs top -> bottom, v from the wearer's right (-x in the
    front view) to left; depth from the matching half of the side view (front-most / back-most y at
    each height for the centre line, the side view's middle for the side edges)."""
    sgn = 1 if view == "front" else -1
    r, c = np.nonzero(face_mask)
    x = (c + .5 - cams[view][0]) / S * sgn
    z = (cams[view][1] - r - .5) / S
    sr, sc = np.nonzero(side_mask)
    sy = (sc + .5 - cams["side"][0]) / S
    sz = (cams["side"][1] - sr - .5) / S
    z_top, z_bot = np.percentile(z, 99), np.percentile(z, 1)
    z_mid = 0.5 * (z_top + z_bot)

    def half_width(zz):
        k = np.abs(z - zz) < 0.01
        return float(np.percentile(np.abs(x[k]), 95)) if k.any() else 0.1

    def y_centre(zz):
        zz = np.clip(zz, sz.min() + 0.005, sz.max() - 0.005)
        k = np.abs(sz - zz) < 0.01
        return float(sy[k].min() if view == "front" else sy[k].max())
    y_side = float(np.median(sy)) if y_side is None else y_side
    corners = {}
    for key, zz in (("t", z_top - 0.01), ("b", z_bot + 0.01)):
        corners[key] = np.array([-half_width(zz), y_side, zz])

    def mirror(p):
        return p * [-1, 1, 1]

    def through(P, a, b):
        return 2 * P - 0.5 * (a + b)
    c00, c10 = corners["t"], corners["b"]
    c01, c11 = mirror(c00), mirror(c10)
    top_c = np.array([0.0, y_centre(z_top - 0.01), z_top - 0.01])
    bot_c = np.array([0.0, y_centre(z_bot + 0.01), z_bot + 0.01])
    side_l = np.array([-half_width(z_mid), y_side, z_mid])
    return Plate(c00, c10, c01, c11, m_u0=through(top_c, c00, c01), m_u1=through(bot_c, c10, c11),
                 m_v0=through(side_l, c00, c10), m_v1=through(mirror(side_l), c01, c11))


def continue_from(above: Plate, u0=0.85, u1=1.35):
    """Starting plate for the next plate down (a lame): the piece above, continued past its u = 1
    edge (the surface is extrapolated), starting a little under it at u0."""
    S_ = above.surface([u0, 0.5 * (u0 + u1), u1], [0.0, 0.5, 1.0])
    c00, c10, c01, c11 = S_[0, 0], S_[2, 0], S_[0, 2], S_[2, 2]
    return Plate(c00, c10, c01, c11,
                 m_u0=2 * S_[0, 1] - 0.5 * (c00 + c01), m_u1=2 * S_[2, 1] - 0.5 * (c10 + c11),
                 m_v0=2 * S_[1, 0] - 0.5 * (c00 + c10), m_v1=2 * S_[1, 2] - 0.5 * (c01 + c11))


# ----------------------------------------------------------------------------- mesh
@dataclass
class PanelLayout:
    creases_u: tuple = (0.5,)                 # crease rows (0..1 along u)
    creases_v: tuple = (0.18, 0.5, 0.82)      # crease columns (0..1 along v)
    rim: float = 0.012                        # rim width (m)
    rim_height: float = 0.003                 # rim stands this far proud of the panels (m)
    lip: float = 0.0                          # height of the lip on edge u = 0 (m), 0 = none


@dataclass
class PlateMesh:
    verts: np.ndarray
    faces: list
    groups: list = field(default_factory=list)    # per face: "panel" | "rim" | "lip"
    sharp: list = field(default_factory=list)     # edges (vertex pairs) to mark sharp


def _normals_grid(P):
    du = np.gradient(P, axis=0)
    dv = np.gradient(P, axis=1)
    n = np.cross(du, dv)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def build_mesh(plate: Plate, layout=PanelLayout(), outward=None):
    """Flat panels between creases, a raised rim round the edge, an optional lip on edge u = 0.
    outward: a point inside the piece (normals point away from it); default: the plate's centroid
    pushed back along its mean normal."""
    ku = np.array(sorted({0.0, 1.0, *layout.creases_u}))
    kv = np.array(sorted({0.0, 1.0, *layout.creases_v}))
    K = plate.surface(ku, kv)                          # crease-line intersections (panel corners)
    # rim insets in parameter space, from the real edge lengths
    lu = np.linalg.norm(np.diff(plate.surface(np.linspace(0, 1, 21), [0.5])[:, 0], axis=0), axis=1).sum()
    lv = np.linalg.norm(np.diff(plate.surface([0.5], np.linspace(0, 1, 21))[0], axis=0), axis=1).sum()
    du, dv = layout.rim / max(lu, 1e-6), layout.rim / max(lv, 1e-6)
    us = np.array(sorted({*ku, du, 1 - du}))
    vs = np.array(sorted({*kv, dv, 1 - dv}))

    def flat(u, v):
        """Bilinear inside the panel that contains (u, v): panels stay flat-ish (exactly planar when
        their four corners are)."""
        i = min(np.searchsorted(ku, u, side="right") - 1, len(ku) - 2)
        j = min(np.searchsorted(kv, v, side="right") - 1, len(kv) - 2)
        a = (u - ku[i]) / (ku[i + 1] - ku[i])
        b = (v - kv[j]) / (kv[j + 1] - kv[j])
        return ((1 - a) * (1 - b) * K[i, j] + a * (1 - b) * K[i + 1, j] + (1 - a) * b * K[i, j + 1]
                + a * b * K[i + 1, j + 1])
    G = np.array([[flat(u, v) for v in vs] for u in us])          # (nu, nv, 3)
    N = _normals_grid(plate.surface(us, vs))
    c = G.reshape(-1, 3).mean(0)
    if outward is None:
        outward = c - 0.1 * N.reshape(-1, 3).mean(0)
    if (N.reshape(-1, 3) * (G.reshape(-1, 3) - outward)).sum() < 0:
        N = -N
    nu, nv = len(us), len(vs)
    on_rim = np.zeros((nu, nv), bool)
    on_rim[[0, -1], :] = True
    on_rim[:, [0, -1]] = True
    inset = np.zeros((nu, nv), bool)                              # the rim's inner ring
    iu = [1, nu - 2]
    jv = [1, nv - 2]
    inset[iu[0]:iu[1] + 1, jv] = True
    inset[iu, jv[0]:jv[1] + 1] = True
    verts = []
    vid = -np.ones((nu, nv), int)
    vid_low = -np.ones((nu, nv), int)                              # inner ring at panel level
    for i in range(nu):
        for j in range(nv):
            raised = on_rim[i, j] or inset[i, j]
            p = G[i, j] + (N[i, j] * layout.rim_height if raised else 0)
            vid[i, j] = len(verts)
            verts.append(p)
            if inset[i, j]:
                vid_low[i, j] = len(verts)
                verts.append(G[i, j])
    faces, groups = [], []
    for i in range(nu - 1):
        for j in range(nv - 1):
            quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            rim_face = any(on_rim[q] for q in quad)
            ids = [vid[q] if (rim_face or not inset[q]) else vid_low[q] for q in quad]
            faces.append(ids)
            groups.append("rim" if rim_face else "panel")
    # step faces between the raised rim and the panels (along the inner ring)
    ring = ([(iu[0], j) for j in range(jv[0], jv[1] + 1)] + [(i, jv[1]) for i in range(iu[0] + 1, iu[1] + 1)]
            + [(iu[1], j) for j in range(jv[1] - 1, jv[0] - 1, -1)] + [(i, jv[0]) for i in range(iu[1] - 1, iu[0], -1)])
    for a, b in zip(ring, ring[1:] + ring[:1]):
        faces.append([vid[a], vid[b], vid_low[b], vid_low[a]])
        groups.append("rim")
    # lip: the u = 0 edge turned up along the outward normal
    if layout.lip > 0:
        top = []
        for j in range(nv):
            top.append(len(verts))
            verts.append(G[0, j] + N[0, j] * (layout.rim_height + layout.lip))
        for j in range(nv - 1):
            faces.append([vid[0, j + 1], vid[0, j], top[j], top[j + 1]])
            groups.append("lip")
    verts = np.array(verts)
    # consistent winding: rim/panel faces outward
    f0 = verts[faces[len(faces) // 3]]
    n0 = np.cross(f0[1] - f0[0], f0[2] - f0[0])
    if n0 @ (f0.mean(0) - outward) < 0:
        faces = [f[::-1] for f in faces]
    # sharp edges: crease lines (panel corners' rows / columns) and the rim's inner ring
    sharp = []
    crease_i = [i for i, u in enumerate(us) if any(np.isclose(u, ku[1:-1]))]
    crease_j = [j for j, v in enumerate(vs) if any(np.isclose(v, kv[1:-1]))]
    for i in crease_i:
        for j in range(nv - 1):
            sharp.append((vid_low[i, j] if inset[i, j] else vid[i, j], vid_low[i, j + 1] if inset[i, j + 1] else vid[i, j + 1]))
    for j in crease_j:
        for i in range(nu - 1):
            sharp.append((vid_low[i, j] if inset[i, j] else vid[i, j], vid_low[i + 1, j] if inset[i + 1, j] else vid[i + 1, j]))
    for a, b in zip(ring, ring[1:] + ring[:1]):
        sharp.append((vid[a], vid[b]))
        sharp.append((vid_low[a], vid_low[b]))
    return PlateMesh(verts, faces, groups, sharp)


def mirror(mesh: PlateMesh):
    v = mesh.verts * [-1, 1, 1]
    return PlateMesh(v, [f[::-1] for f in mesh.faces], list(mesh.groups), list(mesh.sharp))
