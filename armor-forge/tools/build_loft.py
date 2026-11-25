#!/usr/bin/env python3
"""Build every armor part as its own lofted piece, sized from that part's mask.

    python tools/build_loft.py --body-obj mannequin_yup.obj --labels label_out/face_parts_subdiv.npy \
        --cams label_out/cams.json --front f.png --back b.png --side s.png --side-noarm sn.png --out loft_out
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.inflate import subdivide, weld  # noqa: E402
from armor_forge.labels import NAMES, PALETTE, ViewCam, default_cams, load_mask  # noqa: E402
from armor_forge.loft import build_parts  # noqa: E402
from armor_forge.meshio import read_obj  # noqa: E402
from armor_forge.preview import render  # noqa: E402
from armor_forge.shells import write_obj_multi  # noqa: E402


def sheet(V, Q, C=None):
    imgs = [render(V, Q, a, e, face_colors=C, smooth=False) for a, e in ((-35, 12), (145, 12), (90, 0))]
    s = Image.new("RGB", (sum(i.width for i in imgs), imgs[0].height))
    x = 0
    for i in imgs:
        s.paste(i, (x, 0))
        x += i.width
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-obj", required=True)
    ap.add_argument("--body-obj-zup", action="store_true")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cams", required=True)
    for k in ("front", "back", "side", "side-noarm"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--seg", type=int, default=16, help="segments around each ring")
    ap.add_argument("--subdivide", type=int, default=1)
    ap.add_argument("--out", default="loft_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    v, f = read_obj(a.body_obj, y_up=not a.body_obj_zup)
    v, f, _ = weld(v, f)
    for _ in range(a.subdivide):
        v, f = subdivide(v, f)
    f = np.array(f)
    lab = np.load(a.labels)
    cj = json.load(open(a.cams))
    cams = {}
    for k, c in default_cams().items():
        cams[k] = ViewCam(k, c.axis, c.sign, cj[k][0], cj[k][1], c.s, c.depth_sign)
    masks = {k: load_mask(p) for k, p in {"front": a.front, "back": a.back, "side": a.side,
                                          "side_noarm": getattr(a, "side_noarm")}.items()}
    parts = build_parts(v, f, lab, masks, cams, seg=a.seg)
    tris = 0
    V, Q, C = [], [], []
    for name, (pv, pp) in parts.items():
        t = sum(1 if len(p) == 3 else 2 for p in pp)
        tris += t
        print(f"  {name:22s} {len(pp):4d} faces  ~{t:4d} tris")
        base = sum(len(x) for x in V)
        V.append(pv)
        for p in pp:
            Q.append([i + base for i in (p if len(p) == 4 else p + [p[-1]])])
            C.append(PALETTE[NAMES.index(name.rsplit('.', 1)[0] if name[-2] == '.' else name)])
    print(f"total: {len(parts)} pieces, ~{tris} triangles")
    write_obj_multi(os.path.join(a.out, "armor_loft.obj"), parts)
    V, Q, C = np.vstack(V), np.array(Q), np.array(C)
    sheet(V, Q, C).save(os.path.join(a.out, "preview.png"))
    sheet(V, Q).save(os.path.join(a.out, "preview_plain.png"))


if __name__ == "__main__":
    main()
