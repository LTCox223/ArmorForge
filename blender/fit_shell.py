"""Turn a painted mask into a fitted, rigged armor shell on the base body.

For each mask (front and/or back) we find the body faces that
  1. face the camera,
  2. are not hidden behind other parts of the body (BVH ray test), and
  3. land inside the mask,
then copy those faces, push them off the skin, and give them thickness (Solidify).
The shell inherits the body's vertex groups, is parented to the same rig and gets an Armature
modifier, so it deforms like the body right away. Treat it as a starting point: refine the
shape and skin weights by hand.

Headless example:
  blender -b GunHound.blend -P blender/fit_shell.py -- --name Chest \
      --front-mask masks/chest_front.png --back-mask masks/chest_back.png --thickness 0.015

Masks must be the same pixel size as the renders from render_mannequin.py. White (or opaque,
if the image has transparency) = armor.
"""
import argparse
import json
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

# Open this file from disk in Blender's Text Editor (don't paste it) so imports resolve.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    raise RuntimeError("Open this script from disk (Text Editor > Open) instead of pasting it.")
import _common  # noqa: E402

_common.setup_path()
from armor_forge.projection import mask_from_pixels, view_direction  # noqa: E402
from armor_forge.selection import (face_adjacency, face_centers_normals,  # noqa: E402
                                   grow_selection, select_faces_for_view)

# GUI users: edit these, then press Run. CLI flags override them.
DEFAULTS = {
    "name": "Chest",
    "front_mask": None,
    "back_mask": None,
    "renders": None,      # folder with camera.json; default <blend folder>/mannequin_renders
    "thickness": 0.012,   # shell thickness, meters
    "gap": 0.004,         # distance between skin and the inside of the shell
    "grow": 1,            # extra face rings added around the selection (catches edge-on faces)
}


def load_mask(path, expected_res):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
        w, h = img.size
        if (w, h) != tuple(expected_res):
            raise RuntimeError(f"Mask {os.path.basename(path)} is {w}x{h} but the renders are "
                               f"{expected_res[0]}x{expected_res[1]}. Masks must match exactly.")
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        return mask_from_pixels(px, w, h)
    finally:
        bpy.data.images.remove(img)


def camera_matrix(view):
    return Matrix.Translation(view["location"]) @ Euler(view["rotation"], "XYZ").to_matrix().to_4x4()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=DEFAULTS["name"])
    ap.add_argument("--front-mask", default=DEFAULTS["front_mask"])
    ap.add_argument("--back-mask", default=DEFAULTS["back_mask"])
    ap.add_argument("--renders", default=DEFAULTS["renders"])
    ap.add_argument("--thickness", type=float, default=DEFAULTS["thickness"])
    ap.add_argument("--gap", type=float, default=DEFAULTS["gap"])
    ap.add_argument("--grow", type=int, default=DEFAULTS["grow"])
    args = ap.parse_args(_common.script_args())

    masks = {"front": args.front_mask, "back": args.back_mask}
    masks = {k: v for k, v in masks.items() if v}
    if not masks:
        raise RuntimeError("Give at least one of --front-mask / --back-mask.")

    renders = args.renders or os.path.join(_common.blend_dir(), "mannequin_renders")
    with open(os.path.join(renders, "camera.json")) as f:
        meta = json.load(f)
    res, ortho_scale = meta["res"], meta["ortho_scale"]

    body = bpy.data.objects[meta["body"]]
    rig = body.parent if body.parent and body.parent.type == "ARMATURE" else None
    old_pose = rig.data.pose_position if rig else None
    try:
        if rig:
            rig.data.pose_position = "REST"  # build in rest pose, the armature modifier handles posing
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg)
        mesh = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)

        mw = body.matrix_world
        verts_local = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts_local)
        verts_local = verts_local.reshape(-1, 3)
        mw_np = np.array(mw)
        verts_world = verts_local @ mw_np[:3, :3].T + mw_np[:3, 3]
        faces = [tuple(p.vertices) for p in mesh.polygons]
        centers, normals = face_centers_normals(verts_world, faces)

        bvh = BVHTree.FromPolygons([Vector(v) for v in verts_world], faces)
        selected = np.zeros(len(faces), dtype=bool)
        for view_name, mask_path in masks.items():
            cam = camera_matrix(meta["views"][view_name])
            cam_np = np.array(cam)
            to_cam = -view_direction(cam_np)
            occluded = np.zeros(len(faces), dtype=bool)
            for i, c in enumerate(centers):
                origin = Vector(c + to_cam * 1e-4)
                hit = bvh.ray_cast(origin, Vector(to_cam))
                occluded[i] = hit[0] is not None
            mask = load_mask(mask_path, res)
            chosen = select_faces_for_view(centers, normals, cam_np, ortho_scale, res, mask,
                                           occluded=occluded)
            print(f"armor-forge: {view_name} mask selects {int(chosen.sum())} faces")
            selected |= chosen

        if not selected.any():
            raise RuntimeError("No faces landed inside the mask(s). Check that the mask was drawn "
                               "on the renders from render_mannequin.py (same size/framing).")
        if args.grow > 0:
            selected = grow_selection(face_adjacency(faces), selected, args.grow)
        print(f"armor-forge: {int(selected.sum())} of {len(faces)} faces in the shell")

        # Keep only selected faces, then lift them off the skin
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        doomed = [f for f in bm.faces if not selected[f.index]]
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.normal_update()
        for v in bm.verts:
            v.co += v.normal * args.gap
        bm.to_mesh(mesh)
        bm.free()
        mesh.name = args.name

        # New object: same transform and parenting as the body so it lines up and rides the rig
        obj = bpy.data.objects.new(args.name, mesh)
        (body.users_collection[0] if body.users_collection else bpy.context.scene.collection) \
            .objects.link(obj)
        obj.matrix_world = body.matrix_world
        if rig:
            obj.parent = rig
            obj.matrix_parent_inverse = body.matrix_parent_inverse.copy()

        # Make sure vertex groups survived the copy (names must match the rig's DEF bones)
        if not obj.vertex_groups:
            for g in body.vertex_groups:
                obj.vertex_groups.new(name=g.name)

        solid = obj.modifiers.new("Solidify", "SOLIDIFY")
        solid.thickness = args.thickness
        solid.offset = 1.0
        solid.use_even_offset = True
        if rig:
            arm = obj.modifiers.new("Armature", "ARMATURE")
            arm.object = rig
            arm.use_vertex_groups = True
        print(f"armor-forge: created '{obj.name}'")
    finally:
        if rig:
            rig.data.pose_position = old_pose


main()
