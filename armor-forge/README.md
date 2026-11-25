# armor-forge

Turn concept art of armor on a mannequin into fitted, pre-rigged armor shells in Blender.

The idea: armor on a character is a shell around a body you already have. So instead of
reconstructing 3D from a picture, we take the picture's *outline* of each armor piece, find the
body surface underneath it, lift that surface off the skin, and give it thickness. The result
already shares the body's skin weights and rig. You then refine it by hand.

## Rough-in from four-view sheets (current approach)

Inputs: a gridded four-view sheet (Front, Back, Side, Side without arms; each panel 2 m x 2 m,
x -1..+1 m, z 0..2 m) of the AI-drawn armor, the same sheet rendered for the bare mannequin, and
the mannequin as an OBJ.

    pip install -r requirements.txt
    python tools/carve_sheet.py armor_sheet.png mannequin_sheet.png \
        --body-obj mannequin.obj --out carve_out

* Export `mannequin.obj` from Blender with default OBJ settings and *Apply Modifiers* on (body,
  skull stand-in and feet, mirrored).
* The tool reads the silhouettes at metric scale (it ignores the printed "px per m" caption and
  uses the plot rectangle), carves a 3D volume from the four views, limits it to a band (default
  10 cm) around the body surface, and writes `carve_out/armor_rough.obj` plus `preview.png`.
  Import the OBJ into Blender with default settings.
* It prints a silhouette-match score per view (how well the volume reproduces the sheet).
* Result: one fused, chunky armor volume. Cut it into pieces, retopo and refine by hand.
  Next step: a flat-color-coded sheet so each piece is carved separately.

`armor_forge/sheet.py` holds the sheet layout fractions. If a future sheet template changes its
layout, update `LAYOUT` there.

## Inflate the base mesh (keeps the body's curves and topology)

    python tools/inflate_sheet.py armor_sheet.png --body-obj mannequin.obj --out inflate_out

Pushes every vertex of the base mesh outward along its own normal until it would leave an armor
silhouette (front, back, side), then smooths the push distances. The output has the base mesh's
faces and vertex order, so its curves, UVs and weights can be kept; only positions change.
`--subdivide 1` (default) adds a linear subdivision first so the surface can follow the outline.
Silhouettes alone give a smooth, padded look; hard plate edges need the per-piece masks.

## Alternative: mask-to-shell (surface fitting; not yet run in Blender)

## Workflow

1. **Render the mannequin** (fixed orthographic front/back camera, rest pose):

       blender -b MyFile.blend -P blender/render_mannequin.py

   Writes `mannequin_renders/front.png`, `back.png`, `camera.json` next to the .blend.
   Change the body object with `-- --body "Object Name"`.

2. **Generate concept art on those renders.** Feed `front.png`/`back.png` to your image tool as the
   pose/shape reference (edge or depth conditioning keeps the armor aligned to the body).

3. **Make a mask per armor piece.** Same pixel size as the renders. White (or opaque, if the PNG has
   transparency) = armor. Paint it over the concept art or use a segmentation tool.

4. **Fit the shell:**

       blender -b MyFile.blend -P blender/fit_shell.py -- --name Chest \
           --front-mask masks/chest_front.png --back-mask masks/chest_back.png \
           --thickness 0.015 --gap 0.004 --grow 1

   Creates an object named `Chest`, parented to the rig, with the body's vertex groups, a Solidify
   modifier, and an Armature modifier. Then touch up shape and weights.

Prefer the GUI? Open the script from disk in Blender's Text Editor (Text > Open, don't paste it),
edit the `DEFAULTS` block at the top, and press Run.

## Good fits and bad fits

Works well: breastplates, greaves, bracers, gauntlets, tabards, anything that hugs the body.
Weak: protruding pieces (big pauldrons, spikes, horns, crests, capes). A front/back outline can't tell
depth. Model those by hand, or add a side-view step later. Put surface detail (trim, rivets,
engraving) in textures/normal maps rather than geometry.

## Audit tool (no Blender needed)

    python tools/blend_audit.py MyFile.blend

Reads the raw .blend and reports likely rig problems: Rest Position left on, helper bones that
deform, vertex groups with no bone, vertex weights that don't sum to 1, broken constraint or
driver targets. Read-only. Works on Blender 3.x/4.x files (classic header). Compressed files need
libzstd on the system.

## Layout

    armor_forge/   pure Python/numpy: sheet reading, carving, meshing, projection, selection; unit-tested
    blender/       scripts that run inside Blender (render_mannequin.py, fit_shell.py)
    tools/         standalone tools (carve_sheet.py, inflate_sheet.py, blend_audit.py)
    tests/         pytest suite for armor_forge/

Run tests: `pip install -r requirements-dev.txt && pytest`

## Status

- `armor_forge/` (sheet reading, carving, meshing) has unit tests; `tools/carve_sheet.py` was run on a real
  sheet pair. `tools/blend_audit.py` reads Blender 3.x-5.x files.
- `blender/*.py` are written against the Blender 4.3 Python API but have **not been run inside
  Blender yet**. Expect to fix small API issues on first run.

## Roadmap

- Side-view masks for depth, and a depth/height map to vary thickness per piece
- Bake the concept art onto the shell as a texture (projection) for surface detail
- Per-piece presets (gap/thickness) and batch fitting for whole sets
- Rig repair script (turn off Deform on non-DEF bones, normalize weights)
