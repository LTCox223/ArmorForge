"""Turn labelled, inflated body faces into separate armor pieces (one low-poly shell per part)."""
from __future__ import annotations

import numpy as np

from .labels import BARE, NAMES

# extra stand-off (m) so plates sit proud of what they overlap; higher = further out
LAYER = {
    "Helmet": 0.010, "Gorget": 0.012, "Cuirass": 0.008, "Pauldrons": 0.022, "Shoulder lames": 0.014,
    "Padded sleeves": 0.002, "Vambraces": 0.012, "Gauntlets": 0.010, "Belt": 0.016,
    "Upper tassets": 0.014, "Lower tassets": 0.010, "Split tabard": 0.006, "Thigh plates": 0.012,
    "Knee guards": 0.020, "Greaves": 0.010, "Armored boots": 0.008,
}


def build_shells(verts, faces, normals, t, labels, wall_in=0.004, layer=LAYER):
    """Return {part_name: (verts, quads)} with outer skin + side walls down to the body.

    verts/faces: the (welded, subdivided) body; t: inflate push per vertex; labels: part per face.
    Each shell is a thin solid skin: the outer surface sits at body + t + layer offset, and every
    boundary edge of the part gets a wall quad dropping to just under the body surface, so each
    piece has real edge thickness and can be moved or edited on its own.
    """
    faces = np.asarray(faces)
    out = {}
    for pi, name in enumerate(NAMES):
        if pi == BARE:
            continue
        sel = np.nonzero(labels == pi)[0]
        if not len(sel):
            continue
        f = faces[sel]
        used = np.unique(f)
        remap = -np.ones(len(verts), int)
        remap[used] = np.arange(len(used))
        off = t[used] + layer.get(name, 0.01)
        outer = verts[used] + normals[used] * off[:, None]
        inner = verts[used] - normals[used] * wall_in
        nf = remap[f]
        # boundary edges = edges used by exactly one selected face
        cnt = {}
        for q in nf:
            for k in range(4):
                a, b = int(q[k]), int(q[(k + 1) % 4])
                key = (min(a, b), max(a, b))
                cnt.setdefault(key, []).append((a, b))
        n = len(used)
        walls = []
        for lst in cnt.values():
            if len(lst) == 1:
                a, b = lst[0]
                walls.append([a, b, b + n, a + n])      # outer a,b -> inner b,a
        quads = np.vstack([nf, np.array(walls, int).reshape(-1, 4)]) if walls else nf
        out[name] = (np.vstack([outer, inner]), quads)
    return out


def split_lr(shells):
    """Split every part into left (x>0) and right (x<0) pieces by face centroid; centre pieces stay whole."""
    res = {}
    for name, (v, q) in shells.items():
        cx = v[q][:, :, 0].mean(1)
        spread = v[:, 0].max() - v[:, 0].min()
        if v[:, 0].min() < -0.04 and v[:, 0].max() > 0.04 and spread > 0.12 and name not in ("Helmet", "Gorget", "Cuirass", "Belt", "Split tabard"):
            for side, m in (("L", cx > 0), ("R", cx <= 0)):
                if m.any():
                    res[f"{name}.{side}"] = _compact(v, q[m])
        else:
            res[name] = (v, q)
    return res


def _compact(v, q):
    used = np.unique(q)
    remap = -np.ones(len(v), int)
    remap[used] = np.arange(len(used))
    return v[used], remap[q]


def write_obj_multi(path, parts, y_up=True):
    """One OBJ, one object per part (Blender default axes: world (x,y,z) -> file (x,z,-y))."""
    lines, base = [], 1
    for name, (v, q) in parts.items():
        lines.append(f"o {name.replace(' ', '_')}")
        w = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1) if y_up else v
        lines += [f"v {a:.5f} {b:.5f} {c:.5f}" for a, b, c in w]
        lines.append("s off")
        lines += ["f " + " ".join(str(int(i) + base) for i in f) for f in q]
        base += len(v)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
