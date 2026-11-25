#!/usr/bin/env python3
"""Tile the renders from blender/render_compare.py into one sheet: a row per set, a column per camera.

    python tools/contact_blender.py out/blender --sets embermine v2 v3 --out out/blender_compare.png
    python tools/contact_blender.py out/blender --crop 0.45 1.0 --out out/blender_legs.png   (lower body)
"""
import argparse
import os

from PIL import Image, ImageDraw

CAMS = ("Three-quarter_front", "Front_calibrated", "Side_calibrated", "Back_calibrated", "Three-quarter_rear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--sets", nargs="+", default=["embermine", "v2", "v3"])
    ap.add_argument("--crop", nargs=2, type=float, default=(0.0, 1.0), help="vertical band, 0=top 1=bottom")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for s in a.sets:
        tiles = []
        for c in CAMS:
            p = os.path.join(a.dir, f"{s}_{c}.png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            w, h = im.size
            im = im.crop((int(w * 0.12), int(h * a.crop[0]), int(w * 0.88), int(h * a.crop[1])))
            tiles.append(im.resize((int(im.width * a.scale), int(im.height * a.scale)), Image.LANCZOS))
        row = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height + 26), "white")
        x = 0
        for t in tiles:
            row.paste(t, (x, 26))
            x += t.width
        ImageDraw.Draw(row).text((8, 7), s, fill=(0, 0, 0))
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(a.out)
    print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
