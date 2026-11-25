# armor-forge

Turn concept art of armor on a mannequin into fitted, pre-rigged armor shells in Blender.

The idea: armor on a character is a shell around a body you already have. So instead of
reconstructing 3D from a picture, we take the picture's *outline* of each armor piece, find the
body surface underneath it, lift that surface off the skin, and give it thickness. The result
already shares the body's skin weights and rig. You then refine it by hand.

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

    armor_forge/   pure Python/numpy math (projection, mask sampling, face selection), unit-tested
    blender/       scripts that run inside Blender (render_mannequin.py, fit_shell.py)
    tools/         standalone tools (blend_audit.py)
    tests/         pytest suite for armor_forge/

Run tests: `pip install -r requirements-dev.txt && pytest`

## Status

- `armor_forge/` and `tools/blend_audit.py` are tested (the audit was run against a real rig).
- `blender/*.py` are written against the Blender 4.3 Python API but have **not been run inside
  Blender yet**. Expect to fix small API issues on first run.

## Roadmap

- Side-view masks for depth, and a depth/height map to vary thickness per piece
- Bake the concept art onto the shell as a texture (projection) for surface detail
- Per-piece presets (gap/thickness) and batch fitting for whole sets
- Rig repair script (turn off Deform on non-DEF bones, normalize weights)
