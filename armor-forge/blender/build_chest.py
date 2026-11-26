"""Build the cuirass as two trimmed plates (breastplate + backplate) fitted to the concept.

    blender -b -P armor-forge/blender/build_chest.py -- --masks input/Masks --cams armor-forge/cams.json \
        --art "reference/Embermine Heavy Armor Turnaround.png" --trace-meta out/helmet/traced/trace.json \
        --mannequin armor-forge/mannequin.json --out out/chest

The cuirass colour covers both plates; the breastplate is fitted to the front view and the front half
of the side view, the backplate to the back view and the back half. Plates are symmetric left/right,
the side view is taken as drawn turned (sheet.json), anything drawn over the plate (gorget,
pauldrons, belt, arms) may hide it, and only gold near the plate's outline counts as its trim.
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
import build_pauldrons as bp  # noqa: E402
from armor_forge import concept as cc  # noqa: E402
from armor_forge import plate as pl  # noqa: E402

CUIRASS = (54, 117, 222)
LAYOUT = dict(creases_u=(0.4,), creases_v=(0.3, 0.5, 0.7), lip=0.0)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--trace-meta", required=True)
    ap.add_argument("--mannequin", default=None)
    ap.add_argument("--rgb", default="54,117,222")
    ap.add_argument("--side-turn", type=float, default=None)
    ap.add_argument("--thickness", type=float, default=0.005)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


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
    art_rgb = cc.load_rgb(a.art)
    full, other = {}, {}
    for v in ("front", "back", "side"):
        img = np.asarray(Image.open(os.path.join(a.masks, f"{cc.MASK_FILES[v]}.png")).convert("RGBA"))
        full[v] = bp.colour_mask(os.path.join(a.masks, f"{cc.MASK_FILES[v]}.png"), rgb)
        other[v] = (img[..., 3] > 128) & ~full[v]
    # the torso under the plates (the arm hides most of the side view, so the body guides the depth)
    torso = bp.load_torso(a.mannequin) if a.mannequin else None
    body = None
    if torso is not None:
        V, F = torso
        T = [(f[0], f[i], f[i + 1]) for f in F for i in range(1, len(f) - 1)]
        body = pl.body_from_mesh(np.asarray(V), T, (1.0, 1.7))
        chest = (body.z > 1.2) & (body.z < 1.5)
        y_mid = float(np.median(body.yc[chest]))
    else:
        r, c = np.nonzero(full["side"])
        y_mid = float(np.median((c + .5 - cams["side"][0]) / 1600))
    cols = np.arange(full["side"].shape[1])
    ycol = (cols + .5 - cams["side"][0]) / 1600
    halves = {"front": full["side"] & (ycol < y_mid)[None, :], "back": full["side"] & (ycol > y_mid)[None, :]}
    print(f"armor-forge: side view split at y = {y_mid:+.3f}; side view turn {np.degrees(turn):+.1f} deg")

    built, report = [], {}
    for name, view in (("Breastplate", "front"), ("Backplate", "back")):
        face, side_half = full[view], halves[view]
        views = [pl.make_view(view, face, cams[view], bp.gold_in_frame(art_rgb, regs[view], face), side=0,
                              in_front=other[view], edge_band=0.02),
                 pl.make_view("side", side_half, cams["side"], None, side=0, in_front=other["side"] | halves[
                     "back" if view == "front" else "front"])]
        views[1].turn = turn
        init = pl.init_wrap(face, side_half, cams, view, y_side=y_mid)
        # (pl.seam_rule would make the plates meet at the sides, but a four-sided plate can't also
        # make the armholes: it squashes the plate. Left off until the cuirass has its own model.)
        plate, _ = pl.fit(views, init, symmetric=True, sym_axis=0, body=body)
        _, detail = pl.score(plate, views, detail=True, body=body)
        layout = pl.PanelLayout(**LAYOUT)
        mesh = pl.build_mesh(plate, layout, outward=np.array([0.0, y_mid, plate.centre()[2]]))
        print(f"armor-forge: {name}: " + ", ".join(f"{v} IoU {d['iou']:.3f}" + (f" trim {d['trim_mm']:.1f} mm" if d['trim_mm'] else "")
                                                     for v, d in detail.items() if isinstance(d, dict))
              + f"; {len(mesh.verts)} verts, {len(mesh.faces)} faces")
        built.append((name, plate, layout, mesh, view))
        report[name] = dict(scores=detail, plate=plate.vector().reshape(8, 3).tolist())

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    if a.mannequin:
        torso = bp.load_torso(a.mannequin)
        if torso is not None:
            ref = bpy.data.collections.new("Reference")
            sc.collection.children.link(ref)
            me = bpy.data.meshes.new("Torso")
            me.from_pydata([tuple(v) for v in torso[0]], [], torso[1])
            mt = bpy.data.materials.new("Torso")
            mt.diffuse_color = (0.75, 0.72, 0.68, 1)
            me.materials.append(mt)
            ref.objects.link(bpy.data.objects.new("Torso", me))
    coll = bpy.data.collections.new("Cuirass")
    sc.collection.children.link(coll)
    for name, plate, layout, mesh, view in built:
        bp.make_object(name, mesh, coll, a.thickness)
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out, "chest.blend")))
    report["side_split_y"] = y_mid
    report["side_turn_deg"] = float(np.degrees(turn))
    json.dump(report, open(os.path.join(a.out, "chest.json"), "w"), indent=1)

    cam = bh.setup_render(a.res)
    cam.data.ortho_scale = 0.75
    centre = Vector((0.0, 0.0, 1.38))
    for view in ("front", "side", "back", "three_quarter"):
        bh.render_view(cam, view, os.path.join(a.out, f"chest_{view}.png"), centre=centre)
    art = Image.open(a.art).convert("RGB")
    T = 450
    W = Image.new("RGB", (T * 4, T * 2), "white")
    for i, v in enumerate(("front", "side", "back")):
        plates = [(p_, l_) for _, p_, l_, _, vw in built if v == "side" or vw == v]
        W.paste(bp.overlay(art, regs[v], cams[v], v, plates, 0.0, 1.38, T, half=0.36, turn=turn), (i * T, 0))
    for i, v in enumerate(("front", "side", "back", "three_quarter")):
        W.paste(Image.open(os.path.join(a.out, f"chest_{v}.png")).convert("RGB").resize((T, T)), (i * T, T))
    W.save(os.path.join(a.out, "concept_vs_chest.png"))
    print("armor-forge: wrote", a.out)


main()
