import collections

import numpy as np

from armor_forge import molds
from armor_forge import shellfit as sf


def views_with_dome(size=256, f=16.0, r=0.12, zc=1.85, top=2.0, bottom=1.72):
    """A helmet-like piece (id 1): round in front and side, between bottom and top."""
    def paint(name, lab):
        rows = np.arange(size)
        z = (3712 - (rows + 0.5) * f) / 1600
        cols = np.arange(size)
        u = ((cols + 0.5) * f - 2048) / 1600
        for i, zz in enumerate(z):
            if bottom <= zz <= top:
                half = r * np.sqrt(max(0.0, 1 - max(0.0, zz - zc) ** 2 / (top - zc) ** 2))
                lab[i, np.abs(u) <= half] = 1
    views = {}
    for name in sf.VIEW_FILES:
        lab = np.zeros((size, size), np.uint8)
        paint(name, lab)
        views[name] = sf.View(name, lab, 2048.0, 3712.0, 1600.0, f)
    return views


def test_outline_and_hull_exit():
    views = views_with_dome()
    o = molds.Outline(views, 1, smooth=0)
    a, yf, yb = o.extents(np.array([1.80]))
    assert abs(a[0] - 0.12) < 0.02 and abs(yf[0] + 0.12) < 0.02 and abs(yb[0] - 0.12) < 0.02
    t = molds.hull_exit(np.array([0, 0, 1.85]), np.array([[1.0, 0, 0], [0, -1.0, 0]]), o)
    assert np.allclose(t, 0.12, atol=0.02)


def test_track_transitions_full_and_partial():
    K = 12
    per = [[0.30, 0.40] if k in (0, 1, 11) else [0.30] for k in range(K)]
    tr = molds.track_transitions(per, smooth=0)
    full = tr[~np.isnan(tr).any(1)]
    part = tr[np.isnan(tr).any(1)]
    assert len(full) == 1 and np.allclose(full[0], 0.30)
    assert len(part) == 1 and np.isnan(part[0, 5]) and abs(part[0, 0] - 0.40) < 1e-9


def test_ring_params_keeps_feature_rings_and_fills_partials():
    K = 4
    feats = np.array([[0.5, 0.5, 0.5, 0.5], [0.7, np.nan, np.nan, 0.7]])
    S = molds.ring_params(feats, spacing_s=0.2)
    assert S.shape[1] == K and np.allclose(S[0], 0) and np.allclose(S[-1], 1)
    assert any(np.allclose(r, 0.5) for r in S)                       # feature ring kept exactly
    assert np.all(np.diff(S, axis=0) >= -1e-12)                      # ordered on every meridian


def test_dome_cage_is_clean():
    views = views_with_dome()
    o = molds.Outline(views, 1, smooth=0)
    r = molds.dome_cage(o, views, [1], details=None, K=16, ring_spacing=0.04)
    faces = r["faces"]
    assert all(len(f) == 4 for f in faces)                           # all quads
    edges = collections.Counter()
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[(min(a, b), max(a, b))] += 1
    assert max(edges.values()) == 2                                   # manifold
    boundary = [e for e, c in edges.items() if c == 1]
    assert len(boundary) == 16                                        # one open hem ring
    v = r["verts"]
    assert v[:, 2].max() <= 2.01 and v[:, 2].min() >= 1.70
    assert abs(np.abs(v[:, 0]).max() - 0.12) < 0.02
    assert "hem" in r["tags"]
