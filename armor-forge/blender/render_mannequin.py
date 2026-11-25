"""Render the base body from the front and back with a fixed orthographic camera.

Outputs (in <blend folder>/mannequin_renders by default):
  front.png, back.png   flat-shaded grey renders, transparent background
  camera.json           the exact camera settings -- fit_shell.py needs this to map masks back

Feed front.png / back.png to the image generator as the pose/shape reference, then paint or
segment the armor piece you want into a mask image of the same size.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

# Make the sibling helper importable. Open this file from disk in Blender's Text Editor
# (don't paste it into a new text block) so Blender knows where it lives.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    raise RuntimeError("Open this script from disk (Text Editor > Open) instead of pasting it.")
import _common  # noqa: E402

DEFAULTS = {
    "body": "GunHound Fitting Body",
    "width": 1024,
    "height": 2048,
    "margin": 1.08,
    "out": None,  # default: <blend folder>/mannequin_renders
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=DEFAULTS["body"])
    ap.add_argument("--width", type=int, default=DEFAULTS["width"])
    ap.add_argument("--height", type=int, default=DEFAULTS["height"])
    ap.add_argument("--margin", type=float, default=DEFAULTS["margin"])
    ap.add_argument("--out", default=DEFAULTS["out"])
    args = ap.parse_args(_common.script_args())

    out_dir = args.out or os.path.join(_common.blend_dir(), "mannequin_renders")
    os.makedirs(out_dir, exist_ok=True)

    scene = bpy.context.scene
    body = bpy.data.objects[args.body]
    rig = body.parent if body.parent and body.parent.type == "ARMATURE" else None

    old_pose = rig.data.pose_position if rig else None
    old_hide = {o: o.hide_render for o in scene.objects}
    old_cam, old_engine = scene.camera, scene.render.engine
    old_film = scene.render.film_transparent
    cam = cam_data = None
    try:
        # Only the body renders, in rest pose, so the images are repeatable
        for o in scene.objects:
            o.hide_render = o != body
        if rig:
            rig.data.pose_position = "REST"
        bpy.context.view_layer.update()

        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg)
        pts = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
        mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        center = (mn + mx) / 2
        width, depth, height = mx.x - mn.x, mx.y - mn.y, mx.z - mn.z

        # ortho_scale spans the LARGER image side; size it so the whole body fits with margin
        ortho_scale = max(args.width, args.height) * max(width / args.width,
                                                         height / args.height) * args.margin

        cam_data = bpy.data.cameras.new("MannequinCam")
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho_scale
        cam_data.clip_start, cam_data.clip_end = 0.01, 100
        cam = bpy.data.objects.new("MannequinCam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam

        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = True
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "SINGLE"
        scene.display.shading.single_color = (0.6, 0.6, 0.6)

        dist = depth + 2.0
        views = {
            "front": ((center.x, center.y - dist, center.z), (math.pi / 2, 0, 0)),
            "back": ((center.x, center.y + dist, center.z), (math.pi / 2, 0, math.pi)),
        }
        meta = {"body": args.body, "res": [args.width, args.height], "ortho_scale": ortho_scale,
                "bounds_min": list(mn), "bounds_max": list(mx), "views": {}}
        for name, (loc, rot) in views.items():
            cam.location, cam.rotation_euler = loc, rot
            scene.render.filepath = os.path.join(out_dir, f"{name}.png")
            bpy.ops.render.render(write_still=True)
            meta["views"][name] = {"location": list(loc), "rotation": list(rot)}

        with open(os.path.join(out_dir, "camera.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("armor-forge: wrote renders to", out_dir)
    finally:
        for o, h in old_hide.items():
            o.hide_render = h
        if rig:
            rig.data.pose_position = old_pose
        scene.camera, scene.render.engine = old_cam, old_engine
        scene.render.film_transparent = old_film
        if cam:
            bpy.data.objects.remove(cam)
        if cam_data:
            bpy.data.cameras.remove(cam_data)


main()
