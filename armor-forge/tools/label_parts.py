#!/usr/bin/env python3
"""Label the base mesh faces by armor part from four colour-coded part masks.

    python tools/label_parts.py armor_sheet.png --body-obj mannequin_yup.obj \
        --front front.png --back back.png --side side.png --side-noarm side_noarm.png --out label_out
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.inflate import inflate, subdivide, weld  # noqa: E402
from armor_forge.labels import (NAMES, PALETTE, default_cams, fit_cam, label_faces, load_mask)  # noqa: E402
from armor_forge.meshio import read_obj  # noqa: E402
from armor_forge.preview import contact_sheet  # noqa: E402
from armor_forge.sheet import load_sheet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armor_sheet")
    ap.add_argument("--body-obj", required=True)
    ap.add_argument("--body-obj-zup", action="store_true")
    for k in ("front", "back", "side", "side-noarm"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--out", default="label_out")
    ap.add_argument("--subdivide", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    v, f = read_obj(a.body_obj, y_up=not a.body_obj_zup)
    v, f, _ = weld(v, f)
    nbase = len(f)
    for _ in range(a.subdivide):
        v, f = subdivide(v, f)
    nv, t = inflate(v, f, load_sheet(a.armor_sheet))
    f = np.array(f)

    paths = {"front": a.front, "back": a.back, "side": a.side, "side_noarm": a.side_noarm}
    masks = {k: load_mask(p) for k, p in paths.items()}
    cams = {}
    for k, c in default_cams().items():
        cams[k], iou = fit_cam(nv, f, c, masks[k])
        print(f"{k:10s} cam offset c0={cams[k].c0:.0f} r0={cams[k].r0:.0f}  silhouette IoU {100 * iou:.1f}%")

    lab, raw, conf = label_faces(nv, f, cams, masks)
    print(f"faces: {len(f)}; labelled directly {int((raw >= 0).sum())}, filled {int(((raw < 0) & (lab >= 0)).sum())}, "
          f"still unlabelled {int((lab < 0).sum())}")
    print("mean vote agreement %.2f" % conf[raw >= 0].mean())
    for i, n in enumerate(NAMES):
        print(f"  {n:15s} {int((lab == i).sum()):5d} faces")

    # labels back onto the original (welded) base-mesh faces: 4**subdivide children each
    per = 4 ** a.subdivide
    base_lab = np.array([np.bincount(lab[i * per:(i + 1) * per][lab[i * per:(i + 1) * per] >= 0],
                                     minlength=len(NAMES)).argmax()
                         if (lab[i * per:(i + 1) * per] >= 0).any() else -1 for i in range(nbase)])
    np.save(os.path.join(a.out, "face_parts_subdiv.npy"), lab)
    np.save(os.path.join(a.out, "face_parts_base.npy"), base_lab)
    json.dump({"names": NAMES, "bare": NAMES.index("Undersuit")}, open(os.path.join(a.out, "parts.json"), "w"))
    cols = PALETTE[np.maximum(lab, 0)]
    contact_sheet(nv, f, views=((-35, 12), (145, 12), (90, 0), (-90, 0)), face_colors=cols).save(
        os.path.join(a.out, "labels_inflated.png"))
    contact_sheet(v, f, views=((-35, 12), (145, 12), (90, 0), (-90, 0)), face_colors=cols).save(
        os.path.join(a.out, "labels_body.png"))


if __name__ == "__main__":
    main()
