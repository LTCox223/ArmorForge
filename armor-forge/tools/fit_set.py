#!/usr/bin/env python3
"""Build a full armor set from registered part masks + the mannequin geometry (ArmorForge v2).

    python tools/fit_set.py --input work/input --body body_geometry.json --out fit_out
`input` holds <View>_labels.npy (1/4-res part-ID grids) and Part_Colors.json.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.fit import VIEWS, Drawing, Fitter  # noqa: E402
from armor_forge.preview import render  # noqa: E402
from armor_forge.shells import write_obj_multi  # noqa: E402


def sheet(V, Q, C=None, views=((-35, 12), (145, 12), (90, 0))):
    imgs = [render(V, Q, a, e, face_colors=C, smooth=False) for a, e in views]
    s = Image.new("RGB", (sum(i.width for i in imgs), imgs[0].height))
    x = 0
    for i in imgs:
        s.paste(i, (x, 0))
        x += i.width
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", default="fit_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.input, "Part_Colors.json")))
    labels = {v: np.load(os.path.join(a.input, f"{v}_labels.npy")) for v in VIEWS}
    g = json.load(open(a.body))
    bodies = {"body": g["GunHound Fitting Body"], "armless": g["GunHound Fitting Body.001"],
              "head": g["SkullTarget"], "feet": g["GunHound Fitting Body.003"]}
    parts = Fitter(Drawing(labels, meta), bodies).build()
    tris = 0
    V, Q, C = [], [], []
    rgb = {p["id"]: p["rgb"] for p in meta["parts"]}
    for name, (pv, pf) in parts.items():
        t = sum(len(f) - 2 for f in pf)
        tris += t
        print(f"  {name:28s} {len(pf):4d} faces {t:5d} tris")
        base = sum(len(x) for x in V)
        V.append(pv)
        for f in pf:
            f = [i + base for i in f]
            for k in range(1, len(f) - 1):
                Q.append([f[0], f[k], f[k + 1], f[k + 1]])
                C.append(rgb[int(name[:2])])
    print(f"total: {len(parts)} pieces, {tris} triangles")
    write_obj_multi(os.path.join(a.out, "armor_v2.obj"), parts)
    V, Q, C = np.vstack(V), np.array(Q), np.array(C)
    sheet(V, Q, C).save(os.path.join(a.out, "preview.png"))
    sheet(V, Q).save(os.path.join(a.out, "preview_plain.png"))


if __name__ == "__main__":
    main()
