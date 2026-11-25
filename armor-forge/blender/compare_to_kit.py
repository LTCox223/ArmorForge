"""Score a built set against the drawing and against a hand-made kit (read-only).

    blender -b out/step1/embermine_shells.blend -P armor-forge/blender/compare_to_kit.py -- \
        --set "ArmorForge · Embermine" --kit-scene "Embermine modular fitting" --kit-suffix "Embermine" \
        --labels input --cams armor-forge/cams.json --colors input/Part_Colors.json --out out/step1/compare

1. Silhouettes: renders the set (on the mannequin) and the kit with orthographic cameras framed like
   the piece map, then reports how well each matches the drawn outline (IoU) in front/back/side,
   and per piece for the built set.
2. Distance: for every kit piece, how far its surface is from the built set (mean / 90th percentile),
   i.e. roughly how much the rough-in has to move to become the hand-made piece.
Writes <out>/report.json and silhouette images.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge import shellfit as sf  # noqa: E402

BODY_ID = 50                                     # silhouette id for the bare mannequin
BODY = ["GunHound Fitting Body", "GunHound Fitting Body.003", "SkullTarget"]
ORTHO = {"front": ((0, -10, 0), (np.pi / 2, 0, 0)), "back": ((0, 10, 0), (np.pi / 2, 0, np.pi)),
         "side": ((10, 0, 0), (np.pi / 2, 0, np.pi / 2))}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--kit-scene", required=True)
    ap.add_argument("--kit-suffix", default="Embermine")
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


def tri_soup(objs):
    """World-space verts/polys with Mirror modifiers applied by hand (no depsgraph needed)."""
    V, F = [], []
    for o in objs:
        mw = world_matrix(o)
        vs = [mw @ v.co for v in o.data.vertices]
        base = len(V)
        V += vs
        F += [[base + i for i in p.vertices] for p in o.data.polygons]
        if any(m.type == "MIRROR" for m in o.modifiers):
            b2 = len(V)
            V += [Vector((-c.x, c.y, c.z)) for c in vs]
            F += [[b2 + i for i in reversed(p.vertices)] for p in o.data.polygons]
    return V, F


def surface_samples(V, F, per_face=4, rng=np.random.default_rng(0)):
    P = np.array([v[:] for v in V])
    out = []
    for f in F:
        q = P[f]
        for _ in range(per_face):
            w = rng.dirichlet(np.ones(len(f)))
            out.append(w @ q)
    return np.array(out)


def silhouettes(objs_by_id, views, res=1024):
    """Render objects flat-coloured by id; -> {view: (res,res) id image}, -1 = empty."""
    sc = bpy.data.scenes.new("AF sil")
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "FLAT"
    sc.display.shading.color_type = "OBJECT"
    sc.display.render_aa = "OFF"
    sc.display.shading.show_object_outline = False
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "Standard"
    sc.world = bpy.data.worlds.new("AF black")
    sc.world.color = (0, 0, 0)
    code = {}
    for pid, objs in objs_by_id.items():
        col = ((pid * 37 % 251) / 255, (pid * 91 % 251) / 255, 0.5)   # unique flat colour per id
        lin = np.array(col[:2])
        srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        code[pid] = np.round(srgb * 255)                               # what the PNG will hold
        for o in objs:
            c = o.copy()
            c.color = (*col, 1)
            c.parent = None
            c.matrix_world = world_matrix(o)
            for m in list(c.modifiers):
                if m.type == "ARMATURE":
                    c.modifiers.remove(m)
            sc.collection.objects.link(c)
    cam = bpy.data.objects.new("AF ortho", bpy.data.cameras.new("AF ortho"))
    cam.data.type = "ORTHO"
    sc.collection.objects.link(cam)
    sc.camera = cam
    out = {}
    v0 = views["front"]
    cz = (v0.r0 - 4096 / 2) / v0.s                    # world z at the image centre
    cam.data.ortho_scale = 4096 / v0.s
    for name, (loc, rot) in ORTHO.items():
        v = views[name]
        cu = (4096 / 2 - v.c0) / v.s * v.sign             # world u at the image centre
        x, y, _ = loc
        if v.axis == 0:
            cam.location = (cu, y, cz)
        else:
            cam.location = (x, cu, cz)
        cam.rotation_euler = rot
        path = os.path.join(os.environ.get("AF_TMP", bpy.app.tempdir or "/tmp"), f"af_sil_{name}.png")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True, scene=sc.name)
        img = bpy.data.images.load(path)
        px = np.array(img.pixels[:]).reshape(res, res, 4)[::-1, :, :3]
        bpy.data.images.remove(img)
        rg = np.round(px[..., :2] * 255)
        ids = np.full((res, res), -1, int)
        for pid, c in code.items():
            ids[(np.abs(rg - c) <= 2).all(-1)] = pid
        out[name] = ids
    bpy.data.scenes.remove(sc)
    return out


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main():
    a = args()
    os.makedirs(a.out, exist_ok=True)
    views = sf.load_views(a.labels, a.cams, a.colors)
    built = [o for o in bpy.data.collections[a.set].objects if o.type == "MESH"]
    kit_scene = bpy.data.scenes[a.kit_scene]
    kit = [o for o in kit_scene.objects if o.type == "MESH" and o.name.endswith(a.kit_suffix)
           and not o.hide_render]
    body = [bpy.data.objects[n] for n in BODY]
    report = {"built": [o.name for o in built], "kit": [o.name for o in kit]}

    # ---- silhouettes
    by_id = {int(o.name[:2]): [o] for o in built}
    by_id[BODY_ID] = body
    sil_built = silhouettes(by_id, views)
    sil_kit = silhouettes({1: kit, BODY_ID: body}, views)
    sil = {}
    for name in ORTHO:
        lab = views[name].labels
        res = lab.shape[0]
        drawn = lab > 0
        sil[name] = {"built_iou": round(iou(drawn, sil_built[name] >= 0), 3),
                     "kit_iou": round(iou(drawn, sil_kit[name] >= 0), 3)}
        # per piece (only visible pixels count, like the drawing)
        per = {}
        for pid in sorted(by_id):
            if pid == BODY_ID:
                continue
            per[f"{pid:02d} {sf.PIECES[pid]['key']}"] = round(iou(lab == pid, sil_built[name] == pid), 3)
        sil[name]["built_per_piece"] = per
        rgb = np.zeros((res, res, 3), np.uint8)
        rgb[drawn & (sil_built[name] >= 0)] = (200, 200, 200)
        rgb[drawn & (sil_built[name] < 0)] = (230, 60, 60)      # drawn, missing in the build
        rgb[~drawn & (sil_built[name] >= 0)] = (60, 120, 230)   # built, not drawn
        bpy_img = bpy.data.images.new(f"sil_{name}", res, res)
        bpy_img.pixels = np.concatenate([rgb[::-1].reshape(-1, 3) / 255, np.ones((res * res, 1))], 1).ravel()
        bpy_img.filepath_raw = os.path.join(os.path.abspath(a.out), f"silhouette_{name}.png")
        bpy_img.file_format = "PNG"
        bpy_img.save()
    report["silhouette"] = sil

    # ---- distance from each kit piece to the built set
    V, F = tri_soup(built)
    tree = BVHTree.FromPolygons(V, F)
    dist = {}
    for o in kit:
        kv, kf = tri_soup([o])
        pts = surface_samples(kv, kf)
        d = np.array([tree.find_nearest(Vector(p))[3] for p in pts])
        dist[o.name] = {"mean_mm": round(float(d.mean()) * 1000, 1),
                        "p90_mm": round(float(np.percentile(d, 90)) * 1000, 1)}
    report["kit_to_built"] = dist
    # and the other way: built pieces far from any kit surface (extra / misplaced material)
    KV, KF = tri_soup(kit + body)
    ktree = BVHTree.FromPolygons(KV, KF)
    back = {}
    for o in built:
        bv, bf = tri_soup([o])
        pts = surface_samples(bv, bf, per_face=2)
        d = np.array([ktree.find_nearest(Vector(p))[3] for p in pts])
        back[o.name] = {"mean_mm": round(float(d.mean()) * 1000, 1),
                        "p90_mm": round(float(np.percentile(d, 90)) * 1000, 1)}
    report["built_to_kit"] = back
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print("JSON>>" + json.dumps(report))


main()
