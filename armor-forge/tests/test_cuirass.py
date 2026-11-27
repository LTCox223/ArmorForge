import collections

import numpy as np

from armor_forge import cuirass as cu


def profiles():
    z = np.linspace(1.28, 1.74, 93)
    a = np.interp(z, [1.28, 1.45, 1.74], [0.18, 0.22, 0.22])
    return cu.Profiles(z, a, np.full_like(z, -0.20), np.full_like(z, 0.16), 1.28, 1.74, 1.45, (-0.05, 0.12))


def test_half_barrel_with_armhole_yoke_and_mirror_seam():
    p = cu.CuirassParams()
    v, f, groups, info = cu.build(profiles(), p)
    assert v[:, 0].min() >= -1e-9                           # left half only
    assert all(len(q) == 4 for q in f)
    assert set(groups) == {"steel", "trim"}
    edges = collections.Counter()
    for q in f:
        for a, b in zip(q, q[1:] + q[:1]):
            edges[(min(a, b), max(a, b))] += 1
    assert max(edges.values()) == 2
    boundary = [e for e, n in edges.items() if n == 1]
    on_seam = [e for e in boundary if abs(v[e[0], 0]) < 1e-9 and abs(v[e[1], 0]) < 1e-9]
    open_edges = [e for e in boundary if e not in on_seam]
    # open edges: waist + neck (each C edges) + the armhole loop
    adj = collections.defaultdict(set)
    for a, b in open_edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, loops = set(), 0
    for s in adj:
        if s in seen:
            continue
        loops += 1
        stack = [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x] - seen)
    assert loops == 3                                       # waist, neck, armhole (half)
    # the armhole is on the side, between the armpit and the top
    c0, c1 = info["armhole_cols"]
    assert c1 > c0
    top = v[:, 2].max()
    assert abs(top - 1.74) < 1e-6
    # the neck opening is lower at the front than the back
    neck = v[-(p.cols + 1):]
    assert neck[0, 2] < neck[-1, 2]


def test_mirror_full_doubles():
    v, f, _, _ = cu.build(profiles())
    V, F = cu.mirror_full(v, f)
    assert len(V) == 2 * len(v) and len(F) == 2 * len(f)
    assert np.allclose(V[len(v):, 0], -v[:, 0])
