"""Build the pauldrons (cap + lames, left and right) from the piece map and the mannequin.

    blender -b -P armor-forge/blender/build_pauldrons.py -- --masks input/Masks \
        --cams armor-forge/cams.json --mannequin armor-forge/mannequin.json \
        --art "reference/Embermine Heavy Armor Turnaround.png" --trace-meta out/helmet/traced/trace.json \
        --out out/pauldrons

Piece colours default to the Embermine piece map (pauldrons 137,87,212; shoulder lames 220,157,242).
Writes <out>/pauldrons.blend (collection "Pauldrons": "Pauldron cap L/R", "Pauldron lame 1 L/R";
the mannequin in "Reference", hidden from render), renders and concept_vs_pauldrons.png.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import build_helmet as bh  # noqa: E402
from armor_forge import concept as cc  # noqa: E402
from armor_forge import helmet_parts as hp  # noqa: E402
from armor_forge import pauldron as pd  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--mannequin", required=True)
    ap.add_argument("--cap-rgb", default="137,87,212")
    ap.add_argument("--lame-rgb", default="220,157,242")
    ap.add_argument("--arm-rgb", default="146,99,76;2,191,166",
                    help="arm pieces as drawn, sleeve first (';'-separated r,g,b)")
    ap.add_argument("--art", default=None)
    ap.add_argument("--trace-meta", default=None)
    ap.add_argument("--param", nargs="*", default=[])
    ap.add_argument("--subd", type=int, default=1)
    ap.add_argument("--res", type=int, default=700)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def part(path, rgb):
    a = np.asarray(Image.open(path).convert("RGBA"))
    return (a[..., 0] == rgb[0]) & (a[..., 1] == rgb[1]) & (a[..., 2] == rgb[2]) & (a[..., 3] > 128)


def views(masks, rgb, cams):
    f = {v: part(os.path.join(masks, f"{cc.MASK_FILES[v]}.png"), rgb) for v in ("front", "back", "side")}
    return pd.Views(f["front"], f["back"], f["side"], cams)


def mesh_object(name, verts, faces, colour, coll, subd=1, thickness=0.005):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    me.shade_smooth()
    mat = bpy.data.materials.get(name.rsplit(" ", 1)[0]) or bpy.data.materials.new(name.rsplit(" ", 1)[0])
    mat.diffuse_color = (*colour, 1)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if thickness:
        so = ob.modifiers.new("Solidify", "SOLIDIFY")
        so.thickness = thickness
        so.offset = -1
    if subd:
        sd = ob.modifiers.new("Subdivision", "SUBSURF")
        sd.levels = sd.render_levels = subd
    return ob


def arm_tube(axis, t0, t1, r, n=24, rows=8):
    verts, faces = [], []
    phis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for t in np.linspace(t0, t1, rows + 1):
        verts.extend(axis.point(t) + r * axis.direction(phis))
    for i in range(rows):
        for j in range(n):
            a_, b_ = i * n + j, i * n + (j + 1) % n
            faces.append([a_, b_, b_ + n, a_ + n])
    return np.array(verts), faces


def outline_on_concept(art, reg, cam, view, meshes, centre_u, centre_z, size=400, half=0.25):
    """Concept crop with the built pieces' outline in red."""
    import scipy.ndimage as ndi
    from PIL import ImageDraw
    ppm = reg.scale * 1600
    cx, cy = reg.to_art(cam[0] + centre_u * 1600, cam[1] - centre_z * 1600)
    h = half * ppm
    crop = art.transform((size, size), Image.EXTENT, (cx - h, cy - h, cx + h, cy + h), Image.BICUBIC)
    m = hp.project_mask(meshes, view, cam, (4096, 4096))
    e = m & ~ndi.binary_erosion(m, iterations=2)
    r, c = np.nonzero(e)
    x, y = reg.to_art(c + 0.5, r + 0.5)
    k = size / (2 * h)
    d = ImageDraw.Draw(crop)
    for X, Y in zip((x - cx + h) * k, (y - cy + h) * k):
        d.point((X, Y), fill=(230, 20, 20))
    return crop


def main():
    a = args()
    cams = json.load(open(a.cams))
    rgb = lambda s: tuple(int(v) for v in s.split(","))  # noqa: E731
    vc, vl = views(a.masks, rgb(a.cap_rgb), cams), views(a.masks, rgb(a.lame_rgb), cams)
    body = pd.load_mannequin(a.mannequin)
    # the arm as drawn: sleeves + vambraces (+ the pauldron pieces) give the axis, the sleeve the radius
    arm_rgbs = [rgb(s) for s in a.arm_rgb.split(";")]
    fa = np.zeros_like(vc.front)
    sa = np.zeros_like(vc.side)
    for c in arm_rgbs:
        fa |= part(os.path.join(a.masks, "Front.png"), c)
        sa |= part(os.path.join(a.masks, "Side.png"), c)
    axis = pd.axis_from_views(fa, sa, cams)
    r_arm = pd.arm_radius(part(os.path.join(a.masks, "Front.png"), arm_rgbs[0]), cams, axis)
    print(f"armor-forge: arm radius {r_arm:.3f} (from the drawn sleeve)")
    p = pd.PauldronParams()
    for kv in a.param:
        k, v = kv.split("=")
        setattr(p, k, type(getattr(p, k))(v))
    P = pd.build(vc, vl, r_arm, axis, p)
    print("armor-forge: arm axis", np.round(axis.origin, 3).tolist(), "dir", np.round(axis.d, 3).tolist())
    for name, v, f in P.pieces:
        print(f"armor-forge: {name}: {len(v)} verts, {len(f)} faces")
    # silhouette check (left side only)
    left = [(v, f) for _, v, f in P.pieces]
    iou = {}
    for view, m in (("front", vc.front | vl.front), ("back", vc.back | vl.back), ("side", vc.side | vl.side)):
        c0 = cams[view][0]
        side_mask = np.zeros_like(m)
        if view == "front":
            side_mask[:, int(c0):] = True
        elif view == "back":
            side_mask[:, :int(c0)] = True
        else:
            side_mask[:] = True
        mm = m & side_mask
        b = hp.project_mask(left, view, cams[view], m.shape)
        iou[view] = float((b & mm).sum() / max((b | mm).sum(), 1))
    print("armor-forge: silhouette IoU (left)", {k: round(v, 3) for k, v in iou.items()})

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    ref = bpy.data.collections.new("Reference")
    sc.collection.children.link(ref)
    try:
        V, T = pd.load_mannequin(a.mannequin, "GunHound Fitting Body.001")   # torso without arms
    except KeyError:
        V, T = body
    mesh_object("Mannequin", V, T.tolist(), (0.75, 0.72, 0.68), ref, subd=0, thickness=0)
    # the arm as drawn (the concept's arm need not match the mannequin's pose)
    J = np.array(P.info["joint"])
    tj, _, _ = axis.to_local(J)
    tv, tf = arm_tube(axis, tj, tj + 0.55, r_arm)
    mesh_object("Arm (as drawn)", tv, tf, (0.62, 0.50, 0.42), ref, subd=0, thickness=0)
    coll = bpy.data.collections.new("Pauldrons")
    sc.collection.children.link(coll)
    colours = {"Pauldron cap": (0.33, 0.36, 0.40), "Pauldron lame 1": (0.52, 0.47, 0.40)}
    for name, v, f in P.pieces:
        mesh_object(f"{name} L", v, f, colours[name], coll, a.subd)
        mv, mf = pd.mirror(v, f)
        mesh_object(f"{name} R", mv, mf, colours[name], coll, a.subd)
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out, "pauldrons.blend")))
    json.dump(dict(iou=iou, axis=dict(origin=axis.origin.tolist(), d=axis.d.tolist()), info=P.info),
              open(os.path.join(a.out, "pauldrons.json"), "w"), indent=1)

    cam = bh.setup_render(a.res)
    cam.data.ortho_scale = 0.5
    centre = Vector((0.30, 0.03, 1.55))
    shots = {"front": centre, "back": Vector((0.30, 0.03, 1.55)), "side": Vector((0.30, 0.03, 1.55)),
             "three_quarter": centre}
    for name, c in shots.items():
        bh.render_view(cam, name, os.path.join(a.out, f"pauldron_{name}.png"), centre=c)
    if a.art and a.trace_meta:
        meta = json.load(open(a.trace_meta))
        art = Image.open(a.art).convert("RGB")
        T_ = 400
        W = Image.new("RGB", (T_ * 4, T_ * 2), "white")
        meshes = [(v, f) for _, v, f in P.pieces]
        for i, v in enumerate(("front", "side", "back")):
            reg = cc.Registration(**meta[v]["registration"])
            u = {"front": 0.30, "back": -0.30, "side": 0.03}[v]
            W.paste(outline_on_concept(art, reg, cams[v], v, meshes, u, 1.55, T_), (i * T_, 0))
        for i, v in enumerate(("front", "side", "back", "three_quarter")):
            W.paste(Image.open(os.path.join(a.out, f"pauldron_{v}.png")).convert("RGB").resize((T_, T_)),
                    (i * T_, T_))
        W.save(os.path.join(a.out, "concept_vs_pauldrons.png"))
    print("armor-forge: wrote", a.out)


main()
