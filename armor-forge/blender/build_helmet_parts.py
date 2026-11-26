"""Build the helmet as separate parts (wedge visor + skull) found from the concept's gold trim.

    blender -b -P armor-forge/blender/build_helmet_parts.py -- \
        --masks out/helmet/traced --cams armor-forge/cams.json \
        --art "reference/Embermine Heavy Armor Turnaround.png" --trace-meta out/helmet/traced/trace.json \
        --out out/helmet_parts

Writes <out>/helmet_parts.blend, renders (front, side, back, three-quarter), lines.png (the fitted visor
edges drawn on the concept) and lines.json.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import build_helmet as bh  # noqa: E402
from armor_forge import concept as cc  # noqa: E402
from armor_forge import helmet as hm  # noqa: E402
from armor_forge import helmet_parts as hp  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--trace-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--param", nargs="*", default=[], help="skull HelmetParams k=v")
    ap.add_argument("--visor", nargs="*", default=[], help="VisorParams k=v")
    ap.add_argument("--subd", type=int, default=1)
    ap.add_argument("--res", type=int, default=700)
    return ap.parse_args(argv)


def set_params(obj, kvs):
    for kv in kvs:
        k, v = kv.split("=")
        setattr(obj, k, type(getattr(obj, k))(v))
    return obj


def draw_lines(art, regs, cams, lines, front_sil, side_sil, path, tile=600):
    """The fitted visor edges (magenta) on the concept, front and side."""
    tiles = []
    for v in ("front", "side"):
        reg = regs[v]
        ppm = reg.scale * 1600
        c0, r0 = cams[v]

        def A(u, z):
            return reg.to_art(c0 + u * 1600, r0 - z * 1600)
        cx, cy = A(-0.03 if v == "side" else 0.0, 1.90)
        h = 0.2 * ppm
        k = tile / (2 * h)
        crop = art.transform((tile, tile), Image.EXTENT, (cx - h, cy - h, cx + h, cy + h), Image.BICUBIC)
        d = ImageDraw.Draw(crop)

        def P(u, z):
            x, y = A(u, z)
            return float((x - cx + h) * k), float((y - cy + h) * k)
        sil = side_sil if v == "side" else front_sil
        rows = np.nonzero(sil.any(1))[0]
        zlo, zhi = (r0 - rows.max()) / 1600, (r0 - rows.min()) / 1600
        zs = np.linspace(zlo, zlo + 0.75 * (zhi - zlo), 20)
        if v == "side":
            d.line([P(lines.y_rear(z), z) for z in zs], fill=(255, 0, 200), width=3)
            ys = np.linspace(lines.top_side[2], lines.y_rear(lines.z_top_side(lines.y_rear(1.95))), 10)
            d.line([P(y, lines.z_top_side(y)) for y in ys], fill=(255, 0, 200), width=3)
        else:
            for sgn in (1, -1):
                d.line([P(sgn * lines.w(z), z) for z in zs], fill=(255, 0, 200), width=3)
            xs = np.linspace(-lines.w(1.9), lines.w(1.9), 21)
            d.line([P(x, lines.z_top_front(x)) for x in xs], fill=(255, 0, 200), width=3)
        tiles.append(crop)
    W = Image.new("RGB", (tile * 2, tile), "white")
    for i, t in enumerate(tiles):
        W.paste(t, (i * tile, 0))
    W.save(path)


def main():
    a = args()
    cams = json.load(open(a.cams))
    meta = json.load(open(a.trace_meta))
    regs = {v: cc.Registration(**meta[v]["registration"]) for v in ("front", "side", "back")}
    f = hm.load_mask(os.path.join(a.masks, "Front.png"))
    b = hm.load_mask(os.path.join(a.masks, "Back.png"))
    sd = hm.load_mask(os.path.join(a.masks, "Side.png"))
    art_rgb = cc.load_rgb(a.art)
    vp = set_params(hp.VisorParams(), a.visor)
    sp = set_params(hp.skull_params(), a.param)
    parts = hp.build_parts(f, b, sd, art_rgb, regs, cams, vp, sp)
    L = parts.lines
    print("armor-forge: visor lines", json.dumps(L.as_dict()))
    os.makedirs(a.out, exist_ok=True)
    draw_lines(Image.open(a.art).convert("RGB"), regs, cams, L, f, sd, os.path.join(a.out, "lines.png"))

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sv, sf_, sinfo = parts.skull
    skull = bh.make_object(sv, sf_, sinfo, a.subd, name="Helmet skull")
    vv, vf, vcrease = parts.visor
    visor = bh.make_object(vv, vf, {"crease": vcrease}, a.subd, name="Helmet visor", thickness=0.003)
    m = bpy.data.materials.new("Visor steel")
    m.diffuse_color = (0.42, 0.40, 0.36, 1)
    visor.data.materials.clear()
    visor.data.materials.append(m)
    iou = hp.silhouette_iou(parts, {"front": f, "side": sd, "back": b}, cams)
    print("armor-forge: silhouette IoU vs traced outlines", {k: round(v, 3) for k, v in iou.items()})
    json.dump(dict(lines=L.as_dict(), iou=iou), open(os.path.join(a.out, "lines.json"), "w"), indent=1)
    print(f"armor-forge: skull {len(sv)} verts, brow z={sinfo['brow']:.3f}; visor {len(vv)} verts")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out, "helmet_parts.blend")))
    cam = bh.setup_render(a.res)
    for name in ("front", "side", "back", "three_quarter"):
        bh.render_view(cam, name, os.path.join(a.out, f"helmet_{name}.png"))
    print("armor-forge: wrote", a.out)


main()
