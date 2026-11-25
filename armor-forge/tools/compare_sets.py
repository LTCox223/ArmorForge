#!/usr/bin/env python3
"""Side-by-side: v2 mask-derived fitter vs v3 templates, focused on the converted pieces.

    python tools/compare_sets.py --input input --body input/body_geometry.json --out out
Writes compare.png (top row v2, bottom row templates) and prints per-part IoU and triangles.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.fit import Body, Fitter  # noqa: E402
from armor_forge.preview import render  # noqa: E402
from armor_forge.score import score  # noqa: E402
from armor_forge.shells import write_obj_multi  # noqa: E402
from armor_forge.templates import TemplateFitter  # noqa: E402
from score_set import load  # noqa: E402

VIEWS = ((-35, 12), (145, 12), (90, 0), (180, 0))


def flatten(parts, focus, rgb, body=None):
    V, Q, C = [], [], []
    items = list(parts.items())
    if body is not None:
        items = [("17 body", (body.v, body.faces))] + items
    base = 0
    for name, (pv, pf) in items:
        pid = int(name[:2])
        col = rgb[pid] if pid in focus else ((200, 200, 205) if pid != 17 else (150, 120, 110))
        V.append(np.asarray(pv))
        for f in pf:
            for k in range(1, len(f) - 1):
                Q.append([f[0] + base, f[k] + base, f[k + 1] + base, f[k + 1] + base])
                C.append(col)
        base += len(pv)
    return np.vstack(V), np.array(Q), np.array(C)


def row(parts, focus, rgb, body, label):
    V, Q, C = flatten(parts, focus, rgb, body)
    imgs = [render(V, Q, a, e, face_colors=C, smooth=False, size=(460, 760)) for a, e in VIEWS]
    s = Image.new("RGB", (sum(i.width for i in imgs), imgs[0].height + 28), (255, 255, 255))
    x = 0
    for i in imgs:
        s.paste(i, (x, 28))
        x += i.width
    ImageDraw.Draw(s).text((10, 8), label, fill=(20, 20, 20))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--focus", default="12,13,15")
    a = ap.parse_args()
    focus = {int(x) for x in a.focus.split(",")}
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.input, "Part_Colors.json")))
    rgb = {p["id"]: tuple(p["rgb"]) for p in meta["parts"]}
    d, b = load(a.input, a.body)
    bodies = {k: Body(v) for k, v in b.items()}
    sets = {"v2 (mask-derived)": Fitter(d, b).build(), "v3 templates": TemplateFitter(d, b).build()}
    rows = []
    for label, parts in sets.items():
        res, sil = score(parts, bodies, d, verbose=False)
        tri = {p: sum(len(f) - 2 for n, (_, fs) in parts.items() if int(n[:2]) == p for f in fs) for p in focus}
        mean = 100 * np.mean([res[p] for p in res if p < 17])
        print(f"{label}:  silhouette {100 * sil:.1f}%  mean part IoU {mean:.1f}%  total tris "
              f"{sum(len(f) - 2 for _, fs in parts.values() for f in fs)}")
        for p in sorted(focus):
            print(f"    {p:2d} {meta['parts'][p - 1]['name']:15s} IoU {100 * res[p]:5.1f}%  tris {tri[p]}")
        rows.append(row(parts, focus, rgb, bodies["body"],
                        f"{label}   silhouette {100 * sil:.1f}%   mean part IoU {mean:.1f}%   "
                        + "   ".join(f"{meta['parts'][p - 1]['name']} {100 * res[p]:.0f}%" for p in sorted(focus))))
        if "templates" in label:
            write_obj_multi(os.path.join(a.out, "armor_v3.obj"), parts)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(os.path.join(a.out, "compare.png"))
    print("wrote", os.path.join(a.out, "compare.png"))


if __name__ == "__main__":
    main()
