#!/usr/bin/env python3
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.fit import VIEWS, Body, Drawing, Fitter  # noqa: E402
from armor_forge.score import score  # noqa: E402


def load(inp, body):
    meta = json.load(open(os.path.join(inp, "Part_Colors.json")))
    labels = {v: np.load(os.path.join(inp, f"{v}_labels.npy")) for v in VIEWS}
    g = json.load(open(body))
    bodies = {"body": g["GunHound Fitting Body"], "armless": g["GunHound Fitting Body.001"],
              "head": g["SkullTarget"], "feet": g["GunHound Fitting Body.003"]}
    return Drawing(labels, meta), bodies


if __name__ == "__main__":
    d, b = load(sys.argv[1], sys.argv[2])
    parts = Fitter(d, b).build()
    score(parts, {k: Body(v) for k, v in b.items()}, d)
