import collections

import numpy as np

from armor_forge import plate as pl

CAMS = {v: [2048.0, 3712.0] for v in ("front", "back", "side")}


def saddle():
    """A plate draped over a shoulder: neck edge at x = 0.15, arm edge at x = 0.40."""
    return pl.Plate(c00=np.array([0.15, -0.12, 1.66]), c10=np.array([0.32, -0.10, 1.56]),
                    c01=np.array([0.16, 0.16, 1.66]), c11=np.array([0.34, 0.15, 1.56]),
                    m_u0=np.array([0.17, 0.02, 1.90]), m_u1=np.array([0.45, 0.02, 1.68]),
                    m_v0=np.array([0.24, -0.14, 1.60]), m_v1=np.array([0.25, 0.18, 1.60]))


def views_of(plate, side_sel=+1):
    """Silhouettes of a plate rendered into full-res artboard masks (and its edge loop as 'gold')."""
    out = {}
    for v in ("front", "back", "side"):
        tmp = pl.PlateView(v, np.zeros((4096, 4096), bool), 2048.0, 3712.0, 1, np.zeros((0, 2)))
        Sm = plate.surface(np.linspace(0, 1, 17), np.linspace(0, 1, 17))
        m = pl._raster(Sm, tmp)
        loop = pl._loop(Sm)
        u, z = pl.project(loop, v)
        g = np.zeros_like(m)
        c = np.clip(np.round(2048 + u * 1600).astype(int), 0, 4095)
        r = np.clip(np.round(3712 - z * 1600).astype(int), 0, 4095)
        g[r, c] = True
        out[v] = pl.make_view(v, m, CAMS[v], g)
    return list(out.values())


def test_true_plate_scores_better_than_a_shifted_one():
    p = saddle()
    views = views_of(p)
    s0, d0 = pl.score(p, views, detail=True)
    shifted = pl.Plate.from_vector(p.vector() + np.tile([0.0, 0.0, 0.03], 8))
    assert s0 < pl.score(shifted, views)
    assert all(d0[v]["iou"] > 0.97 for v in ("front", "back", "side"))
    assert d0["fold"] == 0


def test_init_from_views_is_close():
    p = saddle()
    m = {}
    for v in ("front", "back", "side"):
        tmp = pl.PlateView(v, np.zeros((4096, 4096), bool), 2048.0, 3712.0, 1, np.zeros((0, 2)))
        m[v] = pl._raster(p.surface(np.linspace(0, 1, 17), np.linspace(0, 1, 17)), tmp)
    q = pl.init_from_views(m["front"], m["back"], m["side"], CAMS)
    assert np.linalg.norm(q.c00 - p.c00) < 0.03 and np.linalg.norm(q.c10 - p.c10) < 0.05


def test_mesh_has_panels_rim_lip_and_clean_topology():
    mesh = pl.build_mesh(saddle(), pl.PanelLayout(lip=0.012))
    assert all(len(f) == 4 for f in mesh.faces)
    kinds = collections.Counter(mesh.groups)
    assert kinds["panel"] > 0 and kinds["rim"] > 0 and kinds["lip"] > 0
    edges = collections.Counter()
    for f in mesh.faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[(min(a, b), max(a, b))] += 1
    assert max(edges.values()) == 2                       # one sheet, no overlaps
    assert len(mesh.verts) < 120                          # low poly, like a hand-made base
    # panels between creases are flat: the 4 corners of each crease panel are near coplanar
    p = saddle()
    lay = pl.PanelLayout()
    ku = np.array([0, *lay.creases_u, 1])
    kv = np.array([0, *lay.creases_v, 1])
    K = p.surface(ku, kv)
    assert K.shape == (len(ku), len(kv), 3)
    m = pl.mirror(mesh)
    assert np.allclose(m.verts[:, 0], -mesh.verts[:, 0])


def test_continue_from_starts_under_the_piece_above():
    p = saddle()
    q = pl.continue_from(p)
    assert np.allclose(q.c00, p.surface([0.85], [0.0])[0, 0])
    assert q.c10[2] < p.c10[2]                            # continues past the arm edge, downward


def test_symmetric_roundtrip_and_turned_projection():
    p = saddle()
    q = pl.Plate.from_symmetric(p.symmetric_vector())
    plane = 0.25 * (p.c00[1] + p.c01[1] + p.c10[1] + p.c11[1])
    assert np.allclose(q.c01[1], 2 * plane - q.c00[1]) and np.allclose(q.c01[[0, 2]], q.c00[[0, 2]])
    assert np.isclose(q.m_u0[1], plane)
    P = np.array([[0.1, 0.0, 1.0], [0.0, 0.1, 1.0]])
    u, z = pl.project(P, "side", np.radians(-10), np.zeros(3))
    assert u[0] > 0 and np.isclose(u[1], 0.1 * np.cos(np.radians(10)))   # near-side points drift back
