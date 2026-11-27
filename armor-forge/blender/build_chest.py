"""Build the cuirass: one barrel round the torso with a flat top band, neck opening and armholes
(armor_forge/cuirass.py), as a left half with a Mirror modifier.

    blender -b -P armor-forge/blender/build_chest.py -- --masks input/Masks --cams armor-forge/cams.json \
        --art "reference/Embermine Heavy Armor Turnaround.png" --trace-meta out/helmet/traced/trace.json \
        --mannequin armor-forge/mannequin.json --out out/chest

Writes <out>/chest.blend (collection "Cuirass"; "Reference" holds the torso), renders,
concept_vs_chest.png (the built outline in red on the concept) and chest.json.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
import scipy.ndimage as ndi
from mathutils import Vector
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import build_helmet as bh  # noqa: E402
import build_pauldrons as bp  # noqa: E402
from armor_forge import concept as cc  # noqa: E402
from armor_forge import cuirass as cu  # noqa: E402
from armor_forge import helmet_recipe as hr  # noqa: E402
from armor_forge import helmet_parts as hp  # noqa: E402
from armor_forge import plate as pl  # noqa: E402

COLOURS = {"steel": (0.36, 0.40, 0.46), "trim": (0.66, 0.50, 0.24)}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--trace-meta", required=True)
    ap.add_argument("--mannequin", required=True)
    ap.add_argument("--rgb", default="54,117,222", help="cuirass colour")
    ap.add_argument("--sleeve-rgb", default="146,99,76")
    ap.add_argument("--param", nargs="*", default=[], help="CuirassParams k=v")
    ap.add_argument("--side-turn", type=float, default=None)
    ap.add_argument("--thickness", type=float, default=0.005)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def outline_overlay(art, reg, cam, view, mask_full, centre_z, size=450, half=0.36):
    ppm = reg.scale * 1600
    cx, cy = reg.to_art(cam[0], cam[1] - centre_z * 1600)
    h = half * ppm
    k = size / (2 * h)
    crop = art.transform((size, size), Image.EXTENT, (cx - h, cy - h, cx + h, cy + h), Image.BICUBIC)
    d = ImageDraw.Draw(crop)
    e = mask_full & ~ndi.binary_erosion(mask_full, iterations=2)
    r, c = np.nonzero(e)
    x, y = reg.to_art(c + .5, r + .5)
    for X, Y in zip((x - cx + h) * k, (y - cy + h) * k):
        d.point((X, Y), fill=(230, 20, 20))
    return crop


def main():
    a = args()
    cams = json.load(open(a.cams))
    meta = json.load(open(a.trace_meta))
    regs = {v: cc.Registration(**meta[v]["registration"]) for v in ("front", "back", "side")}
    turn = 0.0
    if a.side_turn is not None:
        turn = np.radians(a.side_turn)
    else:
        sp = os.path.join(os.path.dirname(os.path.abspath(a.trace_meta)), "sheet.json")
        if os.path.exists(sp):
            turn = np.radians(json.load(open(sp)).get("side_turn_deg", 0.0))
    rgb = tuple(int(v) for v in a.rgb.split(","))
    srgb = tuple(int(v) for v in a.sleeve_rgb.split(","))
    path = lambda v: os.path.join(a.masks, f"{cc.MASK_FILES[v]}.png")  # noqa: E731
    m = {v: bp.colour_mask(path(v), rgb) for v in ("front", "back", "side")}
    sleeve = {v: bp.colour_mask(path(v), srgb) for v in ("front", "side")}
    other = {}
    for v in ("front", "back", "side"):
        img = np.asarray(Image.open(path(v)).convert("RGBA"))
        other[v] = (img[..., 3] > 128) & ~m[v]
    V, F = bp.load_torso(a.mannequin)
    T = [(f[0], f[i], f[i + 1]) for f in F for i in range(1, len(f) - 1)]
    body = pl.body_from_mesh(np.asarray(V), T, (1.0, 1.75))

    p = cu.CuirassParams()
    for kv in a.param:
        k, v = kv.split("=")
        setattr(p, k, type(getattr(p, k))(v))
    pr = cu.measure(m["front"], m["back"], m["side"], sleeve["front"], sleeve["side"], cams, body, p)
    verts, faces, groups, info = cu.build(pr, p)
    print(f"armor-forge: cuirass z {pr.z_bot:.3f}..{pr.z_top:.3f}, armpit {pr.z_armpit:.3f}, armhole y "
          f"{pr.arm_y[0]:+.3f}..{pr.arm_y[1]:+.3f}; {info['rings']} rings x {info['cols']} columns (half); "
          f"{len(verts)} verts, {len(faces)} faces per half")

    # fit check: the whole cuirass against its colour (pieces drawn over it may hide it)
    FV, FF = cu.mirror_full(verts, faces)
    iou, proj = {}, {}
    for v in ("front", "back", "side"):
        if v == "side" and turn:
            b = hr.turned_side_mask([(FV, FF)], cams["side"], m[v].shape, turn)
        else:
            b = hp.project_mask([(FV, FF)], v, cams[v], m[v].shape)
        extra = b & ~m[v] & ~other[v]
        iou[v] = float((b & m[v]).sum() / max(m[v].sum() + extra.sum(), 1))
        proj[v] = b
    print("armor-forge: silhouette IoU (hidden parts allowed)", {k: round(v_, 3) for k, v_ in iou.items()})

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    ref = bpy.data.collections.new("Reference")
    sc.collection.children.link(ref)
    me = bpy.data.meshes.new("Torso")
    me.from_pydata([tuple(v) for v in V], [], F)
    mt = bpy.data.materials.new("Torso")
    mt.diffuse_color = (0.75, 0.72, 0.68, 1)
    me.materials.append(mt)
    ref.objects.link(bpy.data.objects.new("Torso", me))
    coll = bpy.data.collections.new("Cuirass")
    sc.collection.children.link(coll)
    me = bpy.data.meshes.new("Cuirass")
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    for key in ("steel", "trim"):
        mat = bpy.data.materials.new(f"Cuirass {key}")
        mat.diffuse_color = (*COLOURS[key], 1)
        me.materials.append(mat)
    for poly, g in zip(me.polygons, groups):
        poly.material_index = 0 if g == "steel" else 1
        poly.use_smooth = False
    ob = bpy.data.objects.new("Cuirass", me)
    coll.objects.link(ob)
    mi = ob.modifiers.new("Mirror", "MIRROR")
    mi.use_clip = True
    mi.use_mirror_merge = True
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = a.thickness
    so.offset = -1
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out, "chest.blend")))
    json.dump(dict(iou=iou, info=info, side_turn_deg=float(np.degrees(turn)),
                   armpit=pr.z_armpit, armhole_y=pr.arm_y, z=[pr.z_bot, pr.z_top]),
              open(os.path.join(a.out, "chest.json"), "w"), indent=1)

    cam = bh.setup_render(a.res)
    cam.data.ortho_scale = 0.65
    centre = Vector((0.0, 0.0, 1.47))
    for view in ("front", "side", "back", "three_quarter"):
        bh.render_view(cam, view, os.path.join(a.out, f"chest_{view}.png"), centre=centre)
    art = Image.open(a.art).convert("RGB")
    Tn = 450
    W = Image.new("RGB", (Tn * 4, Tn * 2), "white")
    for i, v in enumerate(("front", "side", "back")):
        W.paste(outline_overlay(art, regs[v], cams[v], v, proj[v], 1.47, Tn), (i * Tn, 0))
    for i, v in enumerate(("front", "side", "back", "three_quarter")):
        W.paste(Image.open(os.path.join(a.out, f"chest_{v}.png")).convert("RGB").resize((Tn, Tn)), (i * Tn, Tn))
    W.save(os.path.join(a.out, "concept_vs_chest.png"))
    print("armor-forge: wrote", a.out)


main()
