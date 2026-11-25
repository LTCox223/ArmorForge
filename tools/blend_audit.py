#!/usr/bin/env python3
"""Audit a rigged .blend WITHOUT Blender: reads the raw file and reports likely rig problems.

    python tools/blend_audit.py path/to/file.blend

Checks: rest-position flag, bones that deform but aren't DEF- bones, vertex groups with no bone,
vertex weights that don't sum to ~1, constraint and driver targets that don't exist.

Supports Blender 3.x/4.x files (the classic 'BLENDER-v4xx' header, zstd or gzip compressed or raw).
Read-only: never writes to the .blend. Needs numpy-free pure Python + libzstd for compressed files.
"""
import collections
import ctypes
import ctypes.util
import gzip
import re
import struct
import sys

BONE_NO_DEFORM = 1 << 6
ARM_RESTPOS = 1 << 0
DTAR_FLAG_INVALID = 1 << 1


# ---------------------------------------------------------------- decompression
def _zstd_decompress_frames(data):
    lib = ctypes.CDLL(ctypes.util.find_library("zstd") or "libzstd.so.1")
    lib.ZSTD_findFrameCompressedSize.restype = ctypes.c_size_t
    lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    lib.ZSTD_decompress.restype = ctypes.c_size_t
    out, p = [], 0
    while p < len(data):
        magic = struct.unpack("<I", data[p:p + 4])[0]
        if 0x184D2A50 <= magic <= 0x184D2A5F:  # skippable frame (seek table)
            p += 8 + struct.unpack("<I", data[p + 4:p + 8])[0]
            continue
        buf = data[p:]
        frame = lib.ZSTD_findFrameCompressedSize(buf, len(buf))
        size = lib.ZSTD_getFrameContentSize(buf, frame)
        dst = ctypes.create_string_buffer(size)
        n = lib.ZSTD_decompress(dst, size, buf, frame)
        out.append(dst.raw[:n])
        p += frame
    return b"".join(out)


def read_blend(path):
    data = open(path, "rb").read()
    if data[:4] == b"\x28\xb5\x2f\xfd":
        data = _zstd_decompress_frames(data)
    elif data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if not data.startswith(b"BLENDER-v"):
        raise SystemExit(f"Unsupported .blend header {data[:12]!r} (need classic 64-bit "
                         "little-endian 'BLENDER-v###', i.e. Blender 3.x-4.x).")
    return data


# ---------------------------------------------------------------- DNA parsing
class Blend:
    def __init__(self, data):
        self.d = data
        self.version = int(data[9:12])
        self.blocks, p = [], 12
        while p < len(data):
            code = data[p:p + 4]
            size, ptr, sdna, cnt = struct.unpack("<IQII", data[p + 4:p + 24])
            self.blocks.append((code, size, ptr, sdna, cnt, p + 24))
            p += 24 + size
            if code == b"ENDB":
                break
        self._read_sdna()
        self.byptr = {}
        for b in self.blocks:
            if b[2]:
                self.byptr.setdefault(b[2], b)
        self._layouts = {}

    def _read_sdna(self):
        d = self.d
        q = [b for b in self.blocks if b[0] == b"DNA1"][0][5] + 4

        def strings(q):
            n = struct.unpack("<I", d[q + 4:q + 8])[0]
            q += 8
            out = []
            for _ in range(n):
                e = d.index(b"\0", q)
                out.append(d[q:e].decode())
                q = e + 1
            return out, (q + 3) & ~3

        names, q = strings(q)
        types, q = strings(q)
        n = len(types)
        tlen = struct.unpack("<%dH" % n, d[q + 4:q + 4 + 2 * n])
        q = (q + 4 + 2 * n + 3) & ~3
        ns = struct.unpack("<I", d[q + 4:q + 8])[0]
        q += 8
        self.structs = []
        for _ in range(ns):
            t, nf = struct.unpack("<HH", d[q:q + 4])
            q += 4
            fields = []
            for _ in range(nf):
                a, b = struct.unpack("<HH", d[q:q + 4])
                q += 4
                fields.append((types[a], names[b]))
            self.structs.append((types[t], fields))
        self.sindex = {s[0]: i for i, s in enumerate(self.structs)}
        self.tsize = dict(zip(types, tlen))

    def layout(self, sname):
        if sname in self._layouts:
            return self._layouts[sname]
        off, res = 0, {}
        for t, nm in self.structs[self.sindex[sname]][1]:
            isptr = nm.startswith("*") or nm.startswith("(*")
            base = re.sub(r"\[.*", "", re.sub(r"[*()]", "", nm))
            cnt = 1
            for x in re.findall(r"\[(\d+)\]", nm):
                cnt *= int(x)
            sz = 8 if isptr else self.tsize[t]
            res[base] = (t, isptr, cnt, off, sz)
            off += sz * cnt
        self._layouts[sname] = (res, off)
        return self._layouts[sname]

    def size_of(self, sname):
        return self.layout(sname)[1]

    def read(self, sname, off):
        """Read a struct at absolute file offset -> dict. Embedded structs/ListBases come back
        as ('<Type>', absolute_offset)."""
        d, o = self.d, {}
        for base, (t, isptr, cnt, fo, sz) in self.layout(sname)[0].items():
            a = off + fo
            if isptr:
                v = struct.unpack("<%dQ" % cnt, d[a:a + 8 * cnt])
            elif t == "char" and cnt > 1:
                o[base] = d[a:a + cnt].split(b"\0")[0].decode("latin1")
                continue
            elif t in ("char", "uchar"):
                v = struct.unpack("<%d%s" % (cnt, "b" if t == "char" else "B"), d[a:a + cnt])
            elif t in ("short", "ushort"):
                v = struct.unpack("<%d%s" % (cnt, "h" if t == "short" else "H"), d[a:a + 2 * cnt])
            elif t in ("int", "uint"):
                v = struct.unpack("<%d%s" % (cnt, "i" if t == "int" else "I"), d[a:a + 4 * cnt])
            elif t == "float":
                v = struct.unpack("<%df" % cnt, d[a:a + 4 * cnt])
            else:
                o[base] = ("<%s>" % t, a)
                continue
            o[base] = v if cnt > 1 else v[0]
        return o

    def blocks_of(self, sname):
        i = self.sindex[sname]
        return [b for b in self.blocks if b[3] == i]

    def listbase_first(self, field):
        return struct.unpack("<Q", self.d[field[1]:field[1] + 8])[0]

    def walk(self, first, sname):
        out, p = [], first
        while p and p in self.byptr:
            off = self.byptr[p][5]
            o = self.read(sname, off)
            o["_off"], o["_ptr"] = off, p
            out.append(o)
            p = o["next"]
        return out

    def id_name(self, sname, off):
        return self.read("ID", off + self.layout(sname)[0]["id"][3])["name"][2:]

    def obj_name(self, ptr):
        b = self.byptr.get(ptr) if ptr else None
        return self.id_name("Object", b[5]) if b else None


# ---------------------------------------------------------------- audit
def audit(path):
    B = Blend(read_blend(path))
    out = []

    def say(s=""):
        out.append(s)

    objs = {}
    for b in B.blocks_of("Object"):
        o = B.read("Object", b[5])
        o["_name"], o["_ptr"], o["_off"] = B.id_name("Object", b[5]), b[2], b[5]
        objs[b[2]] = o
    rigs = [o for o in objs.values() if o["type"] == 25 or (o["type"] == 11)]
    meshes = [o for o in objs.values()
              if o["type"] == 1 and not o["_name"].startswith("WGT")]
    say(f"Blender file version {B.version}; {len(objs)} objects, {len(rigs)} armature(s), "
        f"{len(meshes)} non-widget mesh(es)")

    for rig in rigs:
        say(f"\n=== Armature object '{rig['_name']}' ===")
        arm = B.read("bArmature", B.byptr[rig["data"]][5])
        bones = {}

        def walk_bones(first, parent=None):
            for b in B.walk(first, "Bone"):
                b["_parent"] = parent
                bones[b["name"]] = b
                walk_bones(B.listbase_first(b["childbase"]), b["name"])

        walk_bones(B.listbase_first(arm["bonebase"]))
        say(f"{len(bones)} bones")

        if arm["flag"] & ARM_RESTPOS:
            say("! Pose Position is set to 'Rest Position' (armature flag). Poses and controls will "
                "appear to do nothing. Armature tab -> Pose Position -> 'Pose Position'.")
        else:
            say("ok: armature is in Pose Position mode")

        deform = [n for n, b in bones.items() if not b["flag"] & BONE_NO_DEFORM]
        stray = [n for n in deform if not n.startswith("DEF-")]
        if stray:
            say(f"! {len(stray)} non-DEF bones have Deform enabled (Rigify normally allows only "
                f"DEF- bones). Automatic weights will leak onto them. e.g. {stray[:5]}")
        else:
            say("ok: only DEF- bones deform")

        # constraints
        pose = B.read("bPose", B.byptr[rig["pose"]][5])
        chans = B.walk(B.listbase_first(pose["chanbase"]), "bPoseChannel")
        n_con, bad = 0, []
        for ch in chans:
            for c in B.walk(B.listbase_first(ch["constraints"]), "bConstraint"):
                n_con += 1
                blk = B.byptr.get(c["data"])
                if not blk:
                    continue
                sname = B.structs[blk[3]][0]
                if sname not in B.sindex:
                    continue
                dd = B.read(sname, blk[5])
                sub = dd.get("subtarget")
                if isinstance(sub, str) and sub and sub not in bones:
                    bad.append(f"{ch['name']}/{c['name']} -> missing bone '{sub}'")
        say(f"{n_con} constraints; " + ("ok: all bone targets exist" if not bad
                                         else f"! {len(bad)} point at missing bones: {bad[:5]}"))

        # drivers
        bad_drv, n_drv = [], 0
        vlay = B.layout("DriverVar")[0]
        tsz = B.size_of("DriverTarget")
        for blk in B.blocks_of("FCurve"):
            fc = B.read("FCurve", blk[5])
            drv_blk = B.byptr.get(fc["driver"])
            if not drv_blk:
                continue
            n_drv += 1
            drv = B.read("ChannelDriver", drv_blk[5])
            for v in B.walk(B.listbase_first(drv["variables"]), "DriverVar"):
                for i in range(v["num_targets"]):
                    t = B.read("DriverTarget", v["_off"] + vlay["targets"][3] + i * tsz)
                    if not t["id"] or (t["pchan_name"] and t["pchan_name"] not in bones) \
                            or t["flag"] & DTAR_FLAG_INVALID:
                        bad_drv.append(v["name"])
        say(f"{n_drv} drivers; " + ("ok: all targets valid" if not bad_drv
                                    else f"! {len(bad_drv)} invalid driver targets"))

        # meshes that use this rig
        for m in meshes:
            if m["parent"] != rig["_ptr"]:
                continue
            say(f"\n--- mesh '{m['_name']}' ---")
            mesh = B.read("Mesh", B.byptr[m["data"]][5])
            groups = [g["name"] for g in
                      B.walk(B.listbase_first(mesh["vertex_group_names"]), "bDeformGroup")]
            orphan = [g for g in groups if g not in bones]
            say(f"{mesh['totvert']} verts, {len(groups)} vertex groups; " +
                ("ok: every group matches a bone" if not orphan
                 else f"! groups with no bone: {orphan}"))
            cd = B.read("CustomData", mesh["vdata"][1])
            lsz = B.size_of("CustomDataLayer")
            layers = [B.read("CustomDataLayer", B.byptr[cd["layers"]][5] + i * lsz)
                      for i in range(cd["totlayer"])]
            dv = [l for l in layers if l["type"] == 2]  # CD_MDEFORMVERT
            if not dv:
                say("! no vertex weights at all")
                continue
            base = B.byptr[dv[0]["data"]][5]
            unweighted, off_sum, per = 0, [], collections.Counter()
            for i in range(mesh["totvert"]):
                dw, tw, _ = struct.unpack("<QiI", B.d[base + 16 * i:base + 16 * i + 16])
                if not tw:
                    unweighted += 1
                    continue
                ob, s = B.byptr[dw][5], 0.0
                for k in range(tw):
                    gi, w = struct.unpack("<If", B.d[ob + 8 * k:ob + 8 * k + 8])
                    s += w
                    per[groups[gi] if gi < len(groups) else "?BAD INDEX"] += 1
                if abs(s - 1) > 0.05:
                    off_sum.append((i, round(s, 2)))
            say(f"unweighted vertices: {unweighted}")
            if off_sum:
                lo = sorted(off_sum, key=lambda t: t[1])[:5]
                say(f"! {len(off_sum)} vertices whose weights don't sum to ~1 "
                    f"(lowest: {lo}). Fix with Weights -> Normalize All.")
            else:
                say("ok: all weights sum to ~1")
            say(f"groups actually used: {len(per)} of {len(groups)}")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(audit(sys.argv[1]))
