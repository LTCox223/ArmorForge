"""Pauldrons: a cap plus lames, each a shell wrapped around the upper arm's axis.

Frame: the arm axis (from the mannequin's arm, A-pose) with t along it (outward), and an angle phi
around it (0 = straight up, perpendicular to the axis; +90 = toward the back). Each piece is a grid
of rows along t and columns across phi, so the topology is clean by construction.

Shape from the concept, per piece colour (cap = pauldron colour, lames = lame colour):
  1. visual hull: a voxel is in the piece if it projects into the piece's colour in the front, back
     and side views (the side view sees the left arm, +x);
  2. from the axis, a ray in each (t, phi) direction: the hull's exit distance is the outer surface;
     the mannequin's exit distance is the body under it;
  3. the piece covers (t, phi) where the hull reaches clearly beyond the body; per row the covered
     run of phi around the top gives the plate's edges; edges and radius are smoothed;
  4. lames are tucked under the piece above them (their top edge extends `overlap` under it).

Built for the wearer's left (+x) and mirrored for the right.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

S = 1600.0


# ----------------------------------------------------------------------------- mannequin
def load_mannequin(path, name="GunHound Fitting Body"):
    """-> (verts (N,3), triangles (M,3)) with the Mirror modifier applied."""
    d = json.load(open(path))[name]
    V = np.array(d["verts"], float)
    tris = [(f[0], f[i], f[i + 1]) for f in d["faces"] for i in range(1, len(f) - 1)]
    T = np.array(tris, int)
    if d.get("mirror"):
        n = len(V)
        V = np.vstack([V, V * [-1, 1, 1]])
        T = np.vstack([T, T[:, ::-1] + n])
    return V, T


def sample_surface(V, T, n=20000, seed=0):
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T), n, p=area / area.sum())
    r1 = np.sqrt(rng.random(n))[:, None]
    r2 = rng.random(n)[:, None]
    return (1 - r1) * A[idx] + r1 * (1 - r2) * B[idx] + r1 * r2 * C[idx]


@dataclass
class ArmAxis:
    origin: np.ndarray      # a point on the axis
    d: np.ndarray           # unit direction, outward along the arm
    up: np.ndarray          # unit, perpendicular to d, toward world +z (phi = 0)
    back: np.ndarray        # unit, d x up direction chosen toward +y (phi = +90 deg)

    def point(self, t):
        return self.origin + np.multiply.outer(t, self.d)

    def direction(self, phi):
        return np.multiply.outer(np.cos(phi), self.up) + np.multiply.outer(np.sin(phi), self.back)

    def to_local(self, p):
        q = np.asarray(p) - self.origin
        t = q @ self.d
        return t, q @ self.up, q @ self.back


def arm_axis(V, T, x_range=(0.24, 0.60), z_min=1.2):
    """Principal axis of the left upper arm + forearm surface (A-pose)."""
    P = sample_surface(V, T)
    s = P[(P[:, 0] > x_range[0]) & (P[:, 0] < x_range[1]) & (P[:, 2] > z_min)]
    c = s.mean(0)
    _, _, vt = np.linalg.svd(s - c)
    d = vt[0] * np.sign(vt[0][0])
    up = np.array([0, 0, 1.0]) - d[2] * d
    up /= np.linalg.norm(up)
    back = np.cross(d, up)
    if back[1] < 0:
        back = -back
    return ArmAxis(c, d, up, back)


def _pixels(mask, cam, sign=1.0):
    r, c = np.nonzero(mask)
    return (c + 0.5 - cam[0]) / S * sign, (cam[1] - r - 0.5) / S


def axis_from_views(front_arm, side_arm, cams, x_min=0.12):
    """Arm axis from the concept's own arm pieces (e.g. sleeves + vambraces): the line through them
    in the front view (x, z) and their tilt in the side view (y against z). The arm as drawn may not
    match the mannequin's pose, and the pauldron must follow the drawn arm."""
    x, z = _pixels(front_arm, cams["front"])
    keep = x > x_min
    x, z = x[keep], z[keep]
    c = np.array([x.mean(), z.mean()])
    _, _, vt = np.linalg.svd(np.stack([x, z], 1) - c, full_matrices=False)
    dx, dz = vt[0] * np.sign(vt[0][0])
    y, zs = _pixels(side_arm, cams["side"])
    inz = (zs >= z.min()) & (zs <= z.max())
    b, a = np.polyfit(zs[inz], y[inz], 1)          # y = a + b z
    origin = np.array([c[0], a + b * c[1], c[1]])
    d = np.array([dx, b * dz, dz])
    d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) - d[2] * d
    up /= np.linalg.norm(up)
    back = np.cross(d, up)
    if back[1] < 0:
        back = -back
    return ArmAxis(origin, d, up, back)


def arm_radius(front_sleeve, cams, axis: ArmAxis, x_min=0.12, q=90):
    """Arm radius from the drawn sleeve: the q-th percentile distance of its front-view pixels from
    the axis line (measured in the front view's x-z plane)."""
    x, z = _pixels(front_sleeve, cams["front"])
    keep = x > x_min
    P = np.stack([x[keep], z[keep]], 1)
    o = axis.origin[[0, 2]]
    d2 = axis.d[[0, 2]] / np.linalg.norm(axis.d[[0, 2]])
    q_ = P - o
    dist = np.abs(q_[:, 0] * d2[1] - q_[:, 1] * d2[0])
    return float(np.percentile(dist, q))


def ray_exit(V, T, origins, dirs, max_r=0.5):
    """Largest hit distance of each ray against the mesh (0 where it hits nothing)."""
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = B - A, C - A
    out = np.zeros(len(origins))
    for i, (o, dv) in enumerate(zip(origins, dirs)):
        p = np.cross(dv, e2)
        det = (e1 * p).sum(1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        tv = o - A
        u = (tv * p).sum(1) * inv
        q = np.cross(tv, e1)
        v = (dv * q).sum(1) * inv
        t = (e2 * q).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < max_r)
        out[i] = t[hit].max() if hit.any() else 0.0
    return out


# ----------------------------------------------------------------------------- visual hull
@dataclass
class Views:
    """Piece silhouettes (bool, artboard frame) and their origins (cams.json)."""
    front: np.ndarray
    back: np.ndarray
    side: np.ndarray
    cams: dict

    def inside(self, P):
        """Points (N,3) inside all three silhouettes."""
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        ok = np.ones(len(P), bool)
        for m, u, cam in ((self.front, x, self.cams["front"]), (self.back, -x, self.cams["back"]),
                          (self.side, y, self.cams["side"])):
            c = np.floor(cam[0] + u * S).astype(int)
            r = np.floor(cam[1] - z * S).astype(int)
            good = (r >= 0) & (r < m.shape[0]) & (c >= 0) & (c < m.shape[1])
            hit = np.zeros(len(P), bool)
            hit[good] = m[r[good], c[good]]
            ok &= hit
        return ok


def hull_exit(views: Views, origins, dirs, max_r=0.3, step=0.002):
    """Distance along each ray to the last point inside the hull (0 if never inside)."""
    rs = np.arange(step, max_r, step)
    out = np.zeros(len(origins))
    for i in range(0, len(origins), 256):
        o, d = origins[i:i + 256], dirs[i:i + 256]
        P = o[:, None, :] + rs[None, :, None] * d[:, None, :]
        ins = views.inside(P.reshape(-1, 3)).reshape(len(o), len(rs))
        last = np.where(ins.any(1), len(rs) - 1 - np.argmax(ins[:, ::-1], axis=1), -1)
        out[i:i + 256] = np.where(last >= 0, rs[np.maximum(last, 0)], 0.0)
    return out


# ----------------------------------------------------------------------------- shaping
@dataclass
class PauldronParams:
    dome_step_deg: float = 2.0     # dome sampling (both angles)
    t_step: float = 0.004          # tube sampling along the arm (m)
    phi_step_deg: float = 3.0      # tube sampling around the arm
    min_gap: float = 0.008         # the hull must reach this far beyond the body to count as covered
    gap: float = 0.010             # the plate stays at least this far off the body
    shoulder_scale: float = 1.25   # shoulder "ball" radius = this x the arm radius
    cap_rows: int = 12             # mesh rows over the cap (neck side -> down the arm)
    lame_rows: int = 4
    cols: int = 16                 # mesh columns across a plate (front -> back)
    smooth_deg: float = 8.0        # edge / radius smoothing on the dome
    smooth_t: float = 0.012        # ... and along the arm for lames (m)
    overlap: float = 0.015         # a lame's top edge reaches this far up under the piece above
    tuck: float = 0.006            # ...and sits this much inside it
    thickness: float = 0.005


def _gauss1(v, sigma):
    if sigma <= 0:
        return v
    k = int(3 * sigma) + 1
    x = np.arange(-k, k + 1)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    return np.convolve(np.pad(v, k, mode="edge"), w, mode="valid")


class TubeFrame:
    """Rays from the arm axis: rows = t along the arm, cols = angle around it (0 = up)."""
    circular = True

    def __init__(self, axis: ArmAxis):
        self.axis = axis

    def rays(self, rows, cols):
        rr, cc_ = np.meshgrid(rows, cols, indexing="ij")
        return self.axis.point(rr.ravel()), self.axis.direction(cc_.ravel())


class DomeFrame:
    """Rays from the shoulder joint: rows = angle in the frontal plane (0 = up, + = out toward the
    arm, - = toward the neck), cols = angle toward the back (+) / front (-)."""
    circular = False

    def __init__(self, joint, up=(0, 0, 1.0), out=(1.0, 0, 0), back=(0, 1.0, 0)):
        self.joint = np.asarray(joint, float)
        self.up, self.out, self.back = (np.asarray(v, float) for v in (up, out, back))

    def rays(self, rows, cols):
        rr, cc_ = np.meshgrid(rows, cols, indexing="ij")
        r, c = rr.ravel(), cc_.ravel()
        D = (np.cos(c)[:, None] * (np.cos(r)[:, None] * self.up + np.sin(r)[:, None] * self.out)
             + np.sin(c)[:, None] * self.back)
        return np.broadcast_to(self.joint, D.shape).copy(), D


def sample(views: Views, frame, rows, cols, body_r, p: PauldronParams):
    """Hull exit and coverage on a (row, col) grid. body_r: the body's distance along every ray
    (a number: a sphere / cylinder of that radius around the frame)."""
    O, D = frame.rays(rows, cols)
    h = hull_exit(views, O, D).reshape(len(rows), len(cols))
    b = np.full_like(h, float(body_r))
    return dict(rows=np.asarray(rows), cols=np.asarray(cols), hull=h, body=b, cov=h > b + p.min_gap)


def _runs(row, circular):
    """Runs of True -> list of (start, length); circular runs may wrap."""
    n = len(row)
    if row.all():
        return [(0, n)]
    if not row.any():
        return []
    start = int(np.argmin(row)) if circular else 0
    runs, i = [], 0
    while i < n:
        j = (start + i) % n
        if row[j]:
            k = 0
            while i + k < n and row[(j + k) % n]:
                k += 1
            runs.append((j, k))
            i += k
        else:
            i += 1
    return runs


def plate_edges(cg, circular, min_frac=0.08):
    """Per row: the covered run of cols around col = 0 (the top / the frontal plane) as (lo, hi);
    then the longest contiguous span of rows that have one. -> lo, hi (radians), span mask."""
    cols = cg["cols"]
    n = len(cols)
    step = cols[1] - cols[0]
    mid0 = int(np.argmin(np.abs(cols)))
    lo, hi, keep = [], [], []
    for row in cg["cov"]:
        best = None
        for s0, ln in _runs(row, circular):
            mid = s0 + ln / 2
            dist = abs(mid - mid0) if not circular else min(abs(mid % n - mid0), n - abs(mid % n - mid0))
            score = ln - 0.5 * dist
            if best is None or score > best[0]:
                best = (score, s0, ln)
        if best is None or best[2] < min_frac * n:
            lo.append(np.nan)
            hi.append(np.nan)
            keep.append(False)
            continue
        _, s0, ln = best
        a = cols[s0]
        if circular and s0 + ln > n and s0 > mid0:      # wraps through -180: express around 0
            a -= 2 * np.pi
        lo.append(a)
        hi.append(a + (ln - 1) * step)
        keep.append(True)
    keep = np.array(keep)
    best, cur = (0, 0, -1), None
    for i, k in enumerate(keep):
        if k:
            cur = (cur[0], i) if cur else (i, i)
            if cur[1] - cur[0] + 1 > best[0]:
                best = (cur[1] - cur[0] + 1, cur[0], cur[1])
        else:
            cur = None
    span = np.zeros_like(keep)
    span[best[1]:best[2] + 1] = True
    return np.array(lo), np.array(hi), span


def build_piece(cg, frame, p: PauldronParams, n_rows, sigma_rows, sigma_cols, tuck_mesh=None,
                extend=0.0):
    """Quad grid over the covered part of the (row, col) grid. tuck_mesh: (V, T) of the piece
    above; the first row is extended by `extend` (row units) and kept inside that piece.
    -> verts, faces."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    lo, hi, span = plate_edges(cg, frame.circular)
    rows, cols = cg["rows"], cg["cols"]
    rs = rows[span]
    lo_s = _gauss1(lo[span], sigma_rows)
    hi_s = _gauss1(hi[span], sigma_rows)
    Rf = np.where(cg["cov"], cg["hull"], np.nan)
    Rf = np.where(np.isnan(Rf), np.nanmedian(Rf), Rf)
    mode = ("nearest", "wrap" if frame.circular else "nearest")
    Rs = gaussian_filter(Rf, (sigma_rows, sigma_cols), mode=mode)
    Rs = np.maximum(Rs, cg["body"] + p.gap + p.thickness)

    def radius(r, c):
        i = np.interp(r, rows, np.arange(len(rows)))
        if frame.circular:
            c = (np.asarray(c) - cols[0]) % (2 * np.pi) + cols[0]
        j = np.interp(c, cols, np.arange(len(cols)))
        return map_coordinates(Rs, [np.atleast_1d(i), np.atleast_1d(j)], order=1, mode="nearest")

    rr = np.linspace(rs[0], rs[-1], n_rows + 1)
    lo_r, hi_r = np.interp(rr, rs, lo_s), np.interp(rr, rs, hi_s)
    if tuck_mesh is not None and extend:
        rr = np.concatenate([[rr[0] - extend], rr])
        lo_r = np.concatenate([[lo_r[0]], lo_r])
        hi_r = np.concatenate([[hi_r[0]], hi_r])
    verts = []
    for i, (r, a, b) in enumerate(zip(rr, lo_r, hi_r)):
        c = np.linspace(a, b, p.cols + 1)
        O, D = frame.rays(np.array([r]), c)
        rad = radius(np.full_like(c, r), c)
        if tuck_mesh is not None and i == 0:
            above = ray_exit(*tuck_mesh, O, D, max_r=1.0)
            rad = np.where(above > 0, np.minimum(rad, above - p.tuck), rad)
        verts.append(O + rad[:, None] * D)
    verts = np.vstack(verts)
    C = p.cols + 1
    faces = [[i * C + j, i * C + j + 1, (i + 1) * C + j + 1, (i + 1) * C + j]
             for i in range(len(rr) - 1) for j in range(p.cols)]
    # outward: the middle face's normal should point along its ray
    k = len(faces) // 2
    f0 = verts[faces[k]]
    n = np.cross(f0[1] - f0[0], f0[2] - f0[0])
    i0 = k // p.cols
    O, D = frame.rays(np.array([rr[i0]]), np.array([0.5 * (lo_r[i0] + hi_r[i0])]))
    if n @ D[0] < 0:
        faces = [f[::-1] for f in faces]
    return verts, faces


def triangles(faces):
    return np.array([(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)], int)


@dataclass
class Pauldron:
    pieces: list = field(default_factory=list)      # [(name, verts, faces)]
    info: dict = field(default_factory=dict)


def shoulder_joint(views_cap: Views, axis: ArmAxis):
    """The point on the arm axis under the cap's centre (front view)."""
    r, c = np.nonzero(views_cap.front)
    x = (c + 0.5 - views_cap.cams["front"][0]) / S
    z = (views_cap.cams["front"][1] - r - 0.5) / S
    keep = x > 0
    cx, cz = x[keep].mean(), z[keep].mean()
    d2 = axis.d[[0, 2]] / np.linalg.norm(axis.d[[0, 2]])
    t = (np.array([cx, cz]) - axis.origin[[0, 2]]) @ d2 / np.linalg.norm(axis.d[[0, 2]])
    return axis.point(t)


def build(views_cap: Views, views_lames: Views, r_arm: float, axis: ArmAxis, p=PauldronParams()):
    """Cap (dome over the shoulder joint) + lame (tube around the arm, tucked under the cap), for
    the wearer's left. -> Pauldron."""
    out = Pauldron()
    J = shoulder_joint(views_cap, axis)
    dome = DomeFrame(J)
    st = np.radians(p.dome_step_deg)
    rows = np.arange(np.radians(-110), np.radians(175), st)
    cols = np.arange(np.radians(-80), np.radians(80.01), st)
    cg = sample(views_cap, dome, rows, cols, p.shoulder_scale * r_arm, p)
    sg = p.smooth_deg / p.dome_step_deg
    cv, cf = build_piece(cg, dome, p, p.cap_rows, sg, sg)
    out.pieces.append(("Pauldron cap", cv, cf))
    out.info["joint"] = J.tolist()
    if views_lames is not None:
        tube = TubeFrame(axis)
        ts = np.arange(-0.35, 0.35, p.t_step)
        phis = np.radians(np.arange(-180, 180, p.phi_step_deg))
        lg = sample(views_lames, tube, ts, phis, r_arm, p)
        lv, lf = build_piece(lg, tube, p, p.lame_rows, p.smooth_t / p.t_step, p.smooth_deg / p.phi_step_deg,
                             tuck_mesh=(cv, triangles(cf)), extend=p.overlap)
        out.pieces.append(("Pauldron lame 1", lv, lf))
    return out


def mirror(verts, faces):
    """The right-side copy: x -> -x, winding flipped."""
    return verts * [-1, 1, 1], [f[::-1] for f in faces]
