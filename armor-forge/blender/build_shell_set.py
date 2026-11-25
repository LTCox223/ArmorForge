"""Build a whole armor set from a piece map, using the mannequin's own surface (ArmorForge step 1).

    blender -b blender/Embermine_Modular_Geometry.blend -P armor-forge/blender/build_shell_set.py -- \
        --labels input --cams armor-forge/cams.json --colors input/Part_Colors.json \
        --out out/step1/embermine_shells.blend

The opened .blend is never overwritten: the result is saved to --out. Pieces go into a new collection
"ArmorForge · <set>" in the scene that holds the mannequin. Each piece is one half of the body
(+X) with a Mirror modifier, a Solidify modifier and a flat material in its piece-map colour.

Labels are the <View>_labels.npy grids from tools/masks_to_labels.py. See armor_forge/shellfit.py
for how pieces are chosen, snapped to the outline and lifted.
"""
import argparse
import json
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge import shellfit as sf  # noqa: E402

DEFAULTS = {
    "body": ["GunHound Fitting Body", "GunHound Fitting Body.003", "SkullTarget"],
    "body_no_arms": "GunHound Fitting Body.001",
    "subdiv": 1,
    "name": "Embermine",
}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="folder with <View>_labels.npy")
    ap.add_argument("--cams", default=None, help="calibrated view origins (cams.json)")
    ap.add_argument("--colors", required=True, help="Part_Colors.json")
    ap.add_argument("--out", required=True, help=".blend to write (the opened file is not touched)")
    ap.add_argument("--body", nargs="+", default=DEFAULTS["body"])
    ap.add_argument("--body-no-arms", default=DEFAULTS["body_no_arms"])
    ap.add_argument("--subdiv", type=int, default=DEFAULTS["subdiv"])
    ap.add_argument("--name", default=DEFAULTS["name"])
    ap.add_argument("--only", nargs="*", type=int, default=None, help="build only these part ids")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- body
def world_matrix(o):
    """matrix_world from the parent chain (matrix_world is stale for objects outside the active scene)."""
    m = o.matrix_basis.copy()
    while o.parent:
        m = o.parent.matrix_basis @ o.matrix_parent_inverse @ m
        o = o.parent
    return m


def half_mesh(names, subdiv):
    """Base (unmirrored, x >= 0) meshes of the body objects in world space, joined and subdivided."""
    bm = bmesh.new()
    for n in names:
        o = bpy.data.objects[n]
        tmp = bmesh.new()
        tmp.from_mesh(o.data)
        tmp.transform(world_matrix(o))
        m = bpy.data.meshes.new("_af_tmp")
        tmp.to_mesh(m)
        tmp.free()
        bm.from_mesh(m)
        bpy.data.meshes.remove(m)
    for _ in range(subdiv):
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    verts = np.array([v.co[:] for v in bm.verts])
    faces = [[v.index for v in f.verts] for f in bm.faces]
    vnorm = np.array([v.normal[:] for v in bm.verts])
    fnorm = np.array([f.normal[:] for f in bm.faces])
    centers = np.array([f.calc_center_median()[:] for f in bm.faces])
    bm.free()
    seam = np.abs(verts[:, 0]) < 1e-4
    vnorm[seam, 0] = 0.0
    vnorm /= np.maximum(np.linalg.norm(vnorm, axis=1, keepdims=True), 1e-9)
    return verts, faces, vnorm, fnorm, centers, seam


def full_bvh(names):
    """BVH of the mirrored (whole) body, for visibility tests."""
    verts, polys = [], []
    for n in names:
        o = bpy.data.objects[n]
        has_mirror = any(m.type == "MIRROR" for m in o.modifiers)
        base = len(verts)
        mw = world_matrix(o)
        vs = [mw @ v.co for v in o.data.vertices]
        verts += vs
        polys += [[base + i for i in p.vertices] for p in o.data.polygons]
        if has_mirror:
            base2 = len(verts)
            verts += [Vector((-v.x, v.y, v.z)) for v in vs]
            polys += [[base2 + i for i in reversed(p.vertices)] for p in o.data.polygons]
    return BVHTree.FromPolygons(verts, polys, epsilon=1e-6)


def visibility(centers, normals, views, bvh_full, bvh_noarm):
    vis = {}
    for name, v in views.items():
        d = Vector(v.to_cam)
        tree = bvh_noarm if name == "side_noarm" else bvh_full
        out = np.zeros(len(centers), bool)
        for i, (c, n) in enumerate(zip(centers, normals)):
            if n @ v.to_cam <= 0:
                continue
            hit = tree.ray_cast(Vector(c) + Vector(n) * 1e-3 + d * 1e-3, d)
            out[i] = hit[0] is None
        vis[name] = out
    return vis


def arm_flags(points, bvh_noarm, dist=0.02):
    out = np.zeros(len(points), bool)
    for i, p in enumerate(points):
        loc, _, _, d = bvh_noarm.find_nearest(Vector(p))
        out[i] = loc is None or d > dist
    return out


# ----------------------------------------------------------------------------- objects
def material(pid, rgb):
    name = f"AF {pid:02d}"
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = (*[c / 255 for c in rgb], 1.0)
    return m


def make_object(name, verts, faces, coll, mat, thickness):
    used = np.unique(np.concatenate([np.asarray(f) for f in faces]))
    remap = {int(o): i for i, o in enumerate(used)}
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(verts[i]) for i in used], [], [[remap[i] for i in f] for f in faces])
    me.validate()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    mi = ob.modifiers.new("Mirror", "MIRROR")
    mi.use_clip = True
    mi.use_mirror_merge = True
    mi.merge_threshold = 1e-4
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = thickness
    so.offset = 1.0
    so.use_even_offset = False   # even offset spikes at sharp hem corners
    so.use_rim = True
    return ob


# ----------------------------------------------------------------------------- main
def main():
    a = args()
    t0 = time.time()
    meta = json.load(open(a.colors))
    colors = {int(p["id"]): p["rgb"] for p in meta["parts"]}
    views = sf.load_views(a.labels, a.cams, a.colors)

    verts, faces, vnorm, fnorm, centers, seam = half_mesh(a.body, a.subdiv)
    print(f"armor-forge: body half {len(verts)} verts, {len(faces)} faces (subdiv {a.subdiv})")
    bvh_full = full_bvh(a.body)
    bvh_noarm = full_bvh([a.body_no_arms] + [n for n in a.body if n != a.body[0]])
    f_arm = arm_flags(centers, bvh_noarm)
    v_arm = arm_flags(verts, bvh_noarm)

    vis = visibility(centers, fnorm, views, bvh_full, bvh_noarm)
    votes = sf.face_votes(centers, fnorm, views, vis, f_arm)
    adj = sf.face_adjacency(faces)
    hanging = [p for p, c in sf.PIECES.items() if c["kind"] == "hanging"]
    lab = sf.assign_labels(votes, adj, skip_ids=hanging)
    vnames, _ = sf.best_views(vnorm, v_arm, views)
    vnb = sf.vertex_neighbors(len(verts), faces)

    scene = bpy.data.objects[a.body[0]].users_scene[0]
    coll = bpy.data.collections.new(f"ArmorForge · {a.name}")
    scene.collection.children.link(coll)

    report = {}
    for pid, cfg in sf.PIECES.items():
        if a.only and pid not in a.only:
            continue
        if cfg["kind"] == "shell":
            pf = np.flatnonzero(lab == pid)
        elif cfg["kind"] == "base":
            pf = np.flatnonzero(~np.isin(lab, cfg.get("skip", ())))
        else:
            continue
        if not len(pf):
            print(f"armor-forge: {pid:02d} {cfg['key']}: no faces")
            continue
        snapped = (sf.snap_boundary(verts, faces, pf, pid, views, vnames, seam)
                   if cfg["kind"] == "shell" else verts.copy())
        pv = np.unique(np.concatenate([np.asarray(faces[i]) for i in pf]))
        t = np.zeros(len(verts))
        t[pv] = sf.lift_distance(snapped[pv], vnorm[pv], views, pid, cfg["lift"], v_arm[pv])
        # smooth the lift over the piece only
        local = {int(v): i for i, v in enumerate(pv)}
        nbl = [np.array([local[int(n)] for n in vnb[v] if int(n) in local], int) for v in pv]
        t[pv] = sf.smooth_scalar(t[pv], nbl, iters=10)
        # the outer face (after Solidify) should sit on the drawn outline, and never closer than gap
        offset = np.maximum(cfg["gap"], t - cfg["thickness"])
        moved = snapped + vnorm * offset[:, None]
        moved[seam, 0] = 0.0
        name = f"{pid:02d} {cfg['key']}"
        make_object(name, moved, [faces[i] for i in pf], coll, material(pid, colors[pid]), cfg["thickness"])
        report[name] = dict(faces=int(len(pf)), lift_mean_mm=round(float(t[pv].mean()) * 1000, 1),
                            lift_max_mm=round(float(t[pv].max()) * 1000, 1))
        print(f"armor-forge: {name}: {len(pf)} faces, lift mean {report[name]['lift_mean_mm']} mm, "
              f"max {report[name]['lift_max_mm']} mm")

    # hanging cloth last, so it can hang in front of the pieces built above
    tris_v, tris_f = [], []
    for ob in coll.objects:
        me = ob.data
        base = len(tris_v)
        vs = [v.co.copy() for v in me.vertices]
        tris_v += vs + [Vector((-c.x, c.y, c.z)) for c in vs]
        n = len(vs)
        for p in me.polygons:
            tris_f.append([base + i for i in p.vertices])
            tris_f.append([base + n + i for i in reversed(p.vertices)])
    bvh_pieces = BVHTree.FromPolygons(tris_v, tris_f) if tris_f else None

    def under_fn(sign):
        def f(x, z):
            hits = []
            for tree in (bvh_full, bvh_pieces):
                if tree is None:
                    continue
                h = tree.ray_cast(Vector((x, sign * 2.0, z)), Vector((0, -sign, 0)), 4.0)
                if h[0] is not None:
                    hits.append(h[0].y)
            if not hits:
                return np.nan
            return min(hits) if sign < 0 else max(hits)
        return f

    for pid, cfg in sf.PIECES.items():
        if cfg["kind"] != "hanging" or (a.only and pid not in a.only):
            continue
        parts_v, parts_f = [], []
        for face, sign in (("front", -1), ("back", 1)):
            pv_, pf_ = sf.hanging_panel(views, pid, face, under_fn(sign), cfg["gap"])
            if pv_ is None:
                continue
            off = sum(len(x) for x in parts_v)
            parts_v.append(pv_)
            parts_f += [[i + off for i in q] for q in pf_]
        if not parts_v:
            print(f"armor-forge: {pid:02d} {cfg['key']}: not drawn")
            continue
        pverts = np.vstack(parts_v)
        name = f"{pid:02d} {cfg['key']}"
        make_object(name, pverts, parts_f, coll, material(pid, colors[pid]), cfg["thickness"])
        report[name] = dict(faces=len(parts_f))
        print(f"armor-forge: {name}: {len(parts_f)} faces (hanging panels)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out), copy=True)
    print("JSON>>" + json.dumps(report))
    print(f"armor-forge: wrote {a.out} in {time.time() - t0:.1f}s")


main()
