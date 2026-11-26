"""Build a helmet with the standard structure: "Helmet skull" + "Helmet face plate", joined at the
face seam (see armor_forge/helmet_recipe.py).

    blender -b -P armor-forge/blender/build_helmet_recipe.py -- --masks out/helmet/traced \
        --cams armor-forge/cams.json --art "reference/Embermine Heavy Armor Turnaround.png" \
        --trace-meta out/helmet/traced/trace.json --face-mode separate --out out/helmet_recipe/embermine

--face-mode: separate (visor / plate over the opening), integral (welded), open (rim around an open
face). Without --art the seam goes to default places. Writes <out>/helmet.blend and renders; the
face plate is drawn in a lighter colour. Both objects carry a "seam" vertex group and a "weld"
property (True: select both, join, Merge by Distance).
"""
import argparse
import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import build_helmet as bh  # noqa: E402
from armor_forge import concept as cc  # noqa: E402
from armor_forge import helmet as hm  # noqa: E402
from armor_forge import helmet_recipe as hr  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--art", default=None)
    ap.add_argument("--trace-meta", default=None)
    ap.add_argument("--face-mode", default="separate", choices=("separate", "integral", "open"))
    ap.add_argument("--param", nargs="*", default=[], help="HelmetParams k=v")
    ap.add_argument("--face", nargs="*", default=[], help="FaceParams k=v")
    ap.add_argument("--subd", type=int, default=1)
    ap.add_argument("--res", type=int, default=700)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def set_params(obj, kvs):
    for kv in kvs:
        k, v = kv.split("=")
        setattr(obj, k, type(getattr(obj, k))(v))
    return obj


def make_piece(piece, name, colour, subd, weld, coll):
    ob = bh.make_object(piece.verts, piece.faces, {"crease": piece.crease}, subd, name=name)
    bpy.context.scene.collection.objects.unlink(ob)
    coll.objects.link(ob)
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*colour, 1)
    ob.data.materials.clear()
    ob.data.materials.append(m)
    vg = ob.vertex_groups.new(name="seam")
    vg.add(piece.seam, 1.0, "REPLACE")
    ob["weld"] = bool(weld)
    return ob


def main():
    a = args()
    cams = json.load(open(a.cams))
    masks = {v: hm.load_mask(os.path.join(a.masks, f"{cc.MASK_FILES[v]}.png")) for v in ("front", "back", "side")}
    regs, art, side_real = None, None, True
    if a.trace_meta:
        meta = json.load(open(a.trace_meta))
        regs = {v: cc.Registration(**meta[v]["registration"]) for v in ("front", "back", "side") if v in meta}
        side_real = not meta.get("side_stand_in", False)
    if a.art:
        art = cc.load_rgb(a.art)
    fp = set_params(hr.FaceParams(mode=a.face_mode), a.face)
    p = set_params(hm.HelmetParams(), a.param)
    helmet = hr.build(masks["front"], masks["back"], masks["side"], cams, fp, p, art, regs, side_real)
    info = helmet.info
    seam = info["seam_def"]
    print(f"armor-forge: seam brow z={float(seam.brow_fn(0)):.3f} at the centre, "
          f"{float(seam.brow_fn(seam.half_width)):.3f} at the cheeks; cheek edges at |x|={seam.half_width:.3f}; "
          f"cues: {'gold (front+side)' if info['lines'] else 'gold (front)' if info['front_lines'] else 'defaults'}")
    print(f"armor-forge: skull {len(helmet.skull.verts)} verts, face plate {len(helmet.face.verts)} verts, "
          f"seam {len(helmet.skull.seam)} points, mode {helmet.mode}, weld={helmet.weld}")
    view_masks = {v: masks[v] for v in ("front", "back")}
    if side_real:
        view_masks["side"] = masks["side"]
    iou = hr.silhouette_iou(helmet, view_masks, cams)
    print("armor-forge: silhouette IoU", {k: round(v, 3) for k, v in iou.items()})

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    coll = bpy.data.collections.new("Helmet")
    bpy.context.scene.collection.children.link(coll)
    make_piece(helmet.skull, "Helmet skull", (0.33, 0.36, 0.40), a.subd, helmet.weld, coll)
    make_piece(helmet.face, "Helmet face plate", (0.62, 0.55, 0.42), a.subd, helmet.weld, coll)
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out, "helmet.blend")))
    json.dump(dict(mode=helmet.mode, weld=helmet.weld, iou=iou, seam_points=len(helmet.skull.seam),
                   brow_centre=float(seam.brow_fn(0)), cheek_half_width=seam.half_width),
              open(os.path.join(a.out, "helmet.json"), "w"), indent=1)
    cam = bh.setup_render(a.res)
    for name in ("front", "side", "back", "three_quarter"):
        bh.render_view(cam, name, os.path.join(a.out, f"helmet_{name}.png"))
    print("armor-forge: wrote", a.out)


main()
