"""Write detail label grids (visor slit, trim bands) by projecting a modelled piece into the mask views.

Stand-in for the detail colours the image AI is meant to paint on the piece map. Rays are cast
from each view onto the given objects; the material of the first face hit decides the detail id.

    blender -b blender/Embermine_Modular_Geometry.blend -P armor-forge/blender/project_details.py -- \
        --objects "Head / Embermine" --material-ids "Fit study • Slit=20" "Fit study • Edge=21" \
        --labels input --cams armor-forge/cams.json --colors input/Part_Colors.json --out input

Writes <View>_details.npy (uint8, 0 = no detail) for Front, Back, Side, next to the labels.
"""
import argparse
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge import shellfit as sf  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--material-ids", nargs="+", required=True, help="'material name=id'")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cams", default=None)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def world_matrix(o):
    m = o.matrix_basis.copy()
    while o.parent:
        m = o.parent.matrix_basis @ o.matrix_parent_inverse @ m
        o = o.parent
    return m


def main():
    a = args()
    mat_id = {}
    for s in a.material_ids:
        name, i = s.rsplit("=", 1)
        mat_id[name] = int(i)
    V, F, ids = [], [], []
    for n in a.objects:
        o = bpy.data.objects[n]
        mw = world_matrix(o)
        vs = [mw @ v.co for v in o.data.vertices]
        mirrored = any(m.type == "MIRROR" for m in o.modifiers)
        for flip in ([False, True] if mirrored else [False]):
            base = len(V)
            V += [Vector((-c.x, c.y, c.z)) if flip else c for c in vs]
            for p in o.data.polygons:
                F.append([base + i for i in (reversed(p.vertices) if flip else p.vertices)])
                m = o.data.materials[p.material_index] if o.data.materials else None
                ids.append(mat_id.get(m.name if m else "", 0))
    tree = BVHTree.FromPolygons(V, F)
    P = np.array([v[:] for v in V])
    lo, hi = P.min(0) - 0.01, P.max(0) + 0.01
    views = sf.load_views(a.labels, a.cams, a.colors)
    for name in ("front", "back", "side"):
        v = views[name]
        grid = np.zeros_like(v.labels)
        d = Vector(-v.to_cam)
        # pixel range covering the objects
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        cols, rows = v.project(corners)
        for r in range(max(int(rows.min()), 0), min(int(rows.max()) + 2, grid.shape[0])):
            for c in range(max(int(cols.min()), 0), min(int(cols.max()) + 2, grid.shape[1])):
                u = ((c + 0.5) * v.f - v.c0) / v.s * v.sign
                z = (v.r0 - (r + 0.5) * v.f) / v.s
                start = Vector((u, -3.0, z)) if name == "front" else Vector((u, 3.0, z)) if name == "back" \
                    else Vector((3.0, u, z))
                hit = tree.ray_cast(start, d, 10.0)
                if hit[0] is not None:
                    grid[r, c] = ids[hit[2]]
        stem = sf.VIEW_FILES[name]
        np.save(os.path.join(a.out, f"{stem}_details.npy"), grid)
        print(f"{stem}: detail ids {np.unique(grid[grid > 0]).tolist()} px {(grid > 0).sum()}")


main()
