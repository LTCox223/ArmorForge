"""Render a pick-by-eye sheet of helmet variants next to the concept.

The outlines fix the silhouette in the front, side and back views, but not how round or boxy the
helmet is in between. This builds a 3 x 3 grid over two shape parameters (everything else at the
defaults or --param), renders each from three-quarter and draws plan cuts at visor and skull heights, and puts the concept's
front and side views on the left for reference. Cells are labelled A1..C3 with their values.

    blender -b -P armor-forge/blender/helmet_variants.py -- \
        --masks out/helmet/traced --cams armor-forge/cams.json \
        --art "reference/Embermine Heavy Armor Turnaround.png" --trace-meta out/helmet/traced/trace.json \
        --rows beak=0.2,0.45,0.7 --cols n_visor=2.5,4,6 --out out/helmet/variants_visor.png
"""
import argparse
import json
import os
import sys
import tempfile

import bpy
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import build_helmet as bh  # noqa: E402
from armor_forge import concept as cc  # noqa: E402
from armor_forge import helmet as hm  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--trace-meta", required=True, help="trace.json from concept.trace_views (registration)")
    ap.add_argument("--rows", required=True, help="param=v1,v2,v3")
    ap.add_argument("--cols", required=True, help="param=v1,v2,v3")
    ap.add_argument("--param", nargs="*", default=[])
    ap.add_argument("--subd", type=int, default=1)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def axis(spec):
    k, vs = spec.split("=")
    return k, [float(v) for v in vs.split(",")]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def concept_crop(art, reg, cam_origin, view, size, half=0.20, centre_z=1.88):
    ppm = reg.scale * 1600.0
    cx, cy = reg.to_art(cam_origin[0], cam_origin[1] - centre_z * 1600)
    if view == "side":
        cx -= 0.03 * ppm
    h = half * ppm
    return art.transform((size, size), Image.EXTENT, (cx - h, cy - h, cx + h, cy + h), Image.BICUBIC)


SECTIONS = ((-0.07, (200, 60, 40), "visor, 7 cm below brow"), (-0.02, (230, 140, 20), "visor, 2 cm below brow"),
            (0.03, (40, 120, 200), "skull, 3 cm above brow"), (0.08, (90, 60, 170), "skull, 8 cm above brow"))


def sections(o, p, size, span=0.40):
    """Plan cuts through the helmet at fixed heights relative to the brow, front at the bottom."""
    P = hm.shape_profiles(o, p)
    cut = hm.slicer(P, p, K=200)
    img = Image.new("RGB", (size, size), (246, 246, 246))
    d = ImageDraw.Draw(img)
    k = size / span
    cy = size / 2 + k * 0.5 * (P["yf"].min() + P["yb"].max())
    for i in range(-3, 4):          # 5 cm grid
        g = size / 2 + i * 0.05 * k
        d.line((g, 0, g, size), fill=(225, 225, 225))
        d.line((0, cy + i * 0.05 * k, size, cy + i * 0.05 * k), fill=(225, 225, 225))
    for dz, col, _ in SECTIONS:
        z = P["brow"] + dz
        if not (P["z"][0] <= z <= P["z"][-1]):
            continue
        X, Y = cut(z)
        pts = [(size / 2 + x * k, cy - y * k) for x, y in zip(X, Y)]
        d.line(pts + pts[:1], fill=col, width=2)
    d.text((6, size - 18), "front", fill=(120, 120, 120), font=font(13))
    return img


def main():
    a = args()
    rk, rv = axis(a.rows)
    ck, cv = axis(a.cols)
    o, base = bh.load(a)
    tmp = tempfile.mkdtemp()
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    cam = bh.setup_render(a.cell)
    tiles = {}
    for i, r in enumerate(rv):
        for j, c in enumerate(cv):
            p = hm.HelmetParams(**vars(base))
            setattr(p, rk, type(getattr(p, rk))(r))
            setattr(p, ck, type(getattr(p, ck))(c))
            v, f, info = hm.build(o, p)
            ob = bh.make_object(v, f, info, a.subd, name=f"v{i}{j}")
            path = os.path.join(tmp, f"{i}{j}.png")
            bh.render_view(cam, "three_quarter", path)
            tiles[i, j, "three_quarter"] = Image.open(path).convert("RGB")
            tiles[i, j, "top"] = sections(o, p, a.cell)
            bpy.data.objects.remove(ob)
    # sheet: concept column on the left, then the grid (each cell = three-quarter over plan cuts)
    S = a.cell
    head = 40
    lab_w = 70
    W = S + lab_w + len(cv) * S
    H = head + len(rv) * 2 * S
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    meta = json.load(open(a.trace_meta))
    cams = json.load(open(a.cams))
    art = Image.open(a.art).convert("RGB")
    for k, view in enumerate(("front", "side")):
        reg = cc.Registration(**meta[view]["registration"])
        sheet.paste(concept_crop(art, reg, cams[view], view, S), (0, head + k * S))
    d.text((6, 8), "concept", fill=(0, 0, 0), font=font(18))
    d.text((S + lab_w, 8), f"rows: {rk}    columns: {ck}    (each cell: three-quarter view above, cuts seen"
           f" from above below, front at the bottom, 5 cm grid)", fill=(0, 0, 0), font=font(16))
    for n, (_, col, label) in enumerate(SECTIONS):
        d.text((8, head + 2 * S + 10 + 22 * n), "cut: " + label, fill=col, font=font(15))
    for i, r in enumerate(rv):
        y = head + i * 2 * S
        d.text((S + 6, y + S - 20), f"{'ABC'[i] if i < 3 else i}\n{rk}\n= {r:g}", fill=(0, 0, 0), font=font(15))
        for j, c in enumerate(cv):
            x = S + lab_w + j * S
            sheet.paste(tiles[i, j, "three_quarter"], (x, y))
            sheet.paste(tiles[i, j, "top"], (x, y + S))
            d.rectangle((x, y, x + S - 1, y + 2 * S - 1), outline=(180, 180, 180))
            d.text((x + 6, y + 6), f"{'ABC'[i] if i < 3 else i}{j + 1}   {ck} = {c:g}", fill=(0, 0, 0),
                   font=font(16))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sheet.save(a.out)
    print("armor-forge: wrote", a.out)


main()
