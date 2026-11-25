"""Build molded pieces (clean cage shaped to the drawn outline, features on edge loops).

    blender -b blender/Embermine_Modular_Geometry.blend -P armor-forge/blender/build_molded.py -- \
        --labels input --cams armor-forge/cams.json --colors input/Part_Colors.json \
        --out out/step2/embermine_molded.blend

Detail grids (<View>_details.npy: 20 = visor slit, 21 = trim band) are read from --labels if present.
Features: trim and hem bands are extruded outward, slit faces are inset and pushed in. The piece gets a
Solidify modifier (inward, so the outside stays on the outline) and a Subdivision Surface modifier that
is switched off (turn it on for a smooth version).
"""
import argparse
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge import molds  # noqa: E402
from armor_forge import shellfit as sf  # noqa: E402

PIECES = {  # pid: settings
    1: dict(name="helmet", kind="dome", ids=[1], exp=2.6, K=24, spacing=0.045, gap=0.012,
            thickness=0.004, band=0.004, slit_depth=0.014, slit_inset=0.004, body="SkullTarget"),
}
MATS = {"plate": ("Fit study • Steel", (0.30, 0.38, 0.45)), "trim": ("Fit study • Edge", (0.55, 0.45, 0.3)),
        "slit": ("Fit study • Slit", (0.05, 0.05, 0.05))}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cams", default=None)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="Embermine")
    ap.add_argument("--only", nargs="*", type=int, default=None)
    return ap.parse_args(argv)


def world_matrix(o):
    m = o.matrix_basis.copy()
    while o.parent:
        m = o.parent.matrix_basis @ o.matrix_parent_inverse @ m
        o = o.parent
    return m


def body_bvh(name):
    o = bpy.data.objects[name]
    mw = world_matrix(o)
    vs = [mw @ v.co for v in o.data.vertices]
    V, F = list(vs), [list(p.vertices) for p in o.data.polygons]
    if any(m.type == "MIRROR" for m in o.modifiers):
        n = len(vs)
        V += [Vector((-c.x, c.y, c.z)) for c in vs]
        F += [[n + i for i in reversed(p.vertices)] for p in o.data.polygons]
    return BVHTree.FromPolygons(V, F)


def material(key):
    name, col = MATS[key]
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.diffuse_color = (*col, 1)
    return m


def region_normals(bm, faces):
    acc = {}
    for f in faces:
        for v in f.verts:
            acc.setdefault(v, Vector())
            acc[v] += f.normal
    return {v: n.normalized() for v, n in acc.items()}


def apply_features(bm, tag_layer, cfg):
    """Extrude trim/hem bands outward, inset slits and push them in. Material slots: 0 plate,
    1 trim, 2 slit."""
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    band = [f for f in bm.faces if f[tag_layer] in (1, 2)]
    if band:
        ret = bmesh.ops.extrude_face_region(bm, geom=band)
        bmesh.ops.delete(bm, geom=band, context="FACES_ONLY")   # extrude keeps the originals
        new_faces = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMFace)]
        nrm = region_normals(bm, new_faces)
        for v, n in nrm.items():
            v.co += n * cfg["band"]
        for f in new_faces:
            f.material_index = 1
        # the side walls created by the extrusion are trim too
        for g in ret["geom"]:
            if isinstance(g, bmesh.types.BMFace):
                g.material_index = 1
    bm.normal_update()
    slit = [f for f in bm.faces if f[tag_layer] == 3]
    if slit:
        bmesh.ops.inset_region(bm, faces=slit, thickness=cfg["slit_inset"], depth=0.0, use_even_offset=True)
        bm.normal_update()
        nrm = region_normals(bm, slit)
        for v, n in nrm.items():
            v.co -= n * cfg["slit_depth"]
        for f in slit:
            f.material_index = 2


def build_dome(pid, cfg, views, details, coll):
    outline = molds.Outline(views, pid, cfg["ids"])
    tree = body_bvh(cfg["body"])

    def min_dist(center, dirs):
        out = np.zeros(len(dirs))
        for i, d in enumerate(dirs):
            hit = tree.ray_cast(Vector(center), Vector(d), 1.0)
            out[i] = (hit[3] if hit[0] is not None else 0.0) + cfg["gap"] + cfg["thickness"]
        return out

    r = molds.dome_cage(outline, views, cfg["ids"], details, n_exp=cfg["exp"], K=cfg["K"],
                        ring_spacing=cfg["spacing"], min_dist_fn=min_dist)
    verts = molds.symmetrize(r["verts"])
    me = bpy.data.meshes.new(f"{pid:02d} {cfg['name']}")
    me.from_pydata([tuple(v) for v in verts], [], r["faces"])
    me.validate()
    for key in ("plate", "trim", "slit"):
        me.materials.append(material(key))
    bm = bmesh.new()
    bm.from_mesh(me)
    tag = bm.faces.layers.int.new("af_tag")
    code = {"": 0, "trim": 1, "hem": 2, "slit": 3}
    bm.faces.ensure_lookup_table()
    for f, t in zip(bm.faces, r["tags"]):
        f[tag] = code[t]
    apply_features(bm, tag, cfg)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(me.name, me)
    coll.objects.link(ob)
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = cfg["thickness"]
    so.offset = -1.0
    so.use_rim = True
    so.material_offset_rim = 1
    sub = ob.modifiers.new("Subdivision (optional)", "SUBSURF")
    sub.levels = sub.render_levels = 1
    sub.show_viewport = sub.show_render = False
    n_tags = {t: r["tags"].count(t) for t in set(r["tags"])}
    print(f"armor-forge: {me.name}: {len(me.vertices)} verts, {len(me.polygons)} faces, "
          f"{r['rings']} rings x {r['K']}, tags {n_tags}")
    return ob


def main():
    a = args()
    views = sf.load_views(a.labels, a.cams, a.colors)
    details = {}
    for name in ("front", "back", "side"):
        p = os.path.join(a.labels, f"{sf.VIEW_FILES[name]}_details.npy")
        if os.path.exists(p):
            details[name] = np.load(p)
    scene = bpy.data.objects["GunHound Fitting Body"].users_scene[0]
    coll = bpy.data.collections.new(f"ArmorForge molded · {a.name}")
    scene.collection.children.link(coll)
    for pid, cfg in PIECES.items():
        if a.only and pid not in a.only:
            continue
        if cfg["kind"] == "dome":
            build_dome(pid, cfg, views, details, coll)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out), copy=True)
    print(f"armor-forge: wrote {a.out}")


main()
