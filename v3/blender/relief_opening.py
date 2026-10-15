"""Render the relief-wall opening stinger with Blender (direction B).

Run headless:

    Blender -b --factory-startup --python-exit-code 1 \
        --python v3/blender/relief_opening.py -- \
        --crest crest.png --opponent opponent.png --primary "#1E3160" \
        --accent "#5AA4D0" --out frames/ [--vertical]

The scene has no text. The wall is a grid of machined plates in the team
colour. The plates carry debossed crests of both teams and the engraved rink
markings, and every cut edge has a bright chamfer that catches the light.
Timing matches the 135-frame opening sound (4.5 s at 30 fps):

- 0-24: the camera tracks fast along the wall; a light bar races ahead and
  the plates slam into place behind it (blade scrape).
- 26: impact. The arena lights snap up and the camera whips to the centre.
- 26-42: light runs along the engraved rink lines (crackle).
- 46: the medallion punches out of the centre circle, the ring flashes and a
  shock wave runs through the plates (crack).
- 46-58: the medallion spins one turn and locks face-on.
- 62: a light bar glints across the medallion (shing); 66: lock-in (thunk).
- 66-112: the camera keeps orbiting; a second light bar crosses the wall and
  lifts the plates it passes.
- 112-134: the camera pushes hard into the crest.
"""
import argparse
import math
import os
import re
import struct
import sys
import tempfile
import zlib

import bpy
import numpy as np
from mathutils import Quaternion, Vector

FRAMES = 135
FPS = 30
PX = 60                          # height map pixels per wall unit
PLATE = 2.4                      # plate pitch in wall units
PLATE_PX = int(PLATE * PX)
NX, NZ = 19, 11                  # plates across and down (odd: one plate is centred)
WALL_W, WALL_H = NX * PLATE, NZ * PLATE
RECESS_R, RECESS_D = 2.95, 1.2   # centre circle well
MEDAL_R, MEDAL_D = 2.6, 0.5
MEDAL_OUT = -0.35                # medallion centre after the rise (towards camera)
MEDAL_IN = 0.95                  # at the bottom of the well
HIT = 46                         # medallion burst


# ---------------------------------------------------------------- arguments

def parse_args(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="relief_opening")
    p.add_argument("--crest", required=True, help="focus-team crest, transparent PNG")
    p.add_argument("--opponent", required=True, help="opponent logo, transparent PNG")
    p.add_argument("--primary", default="#1E3160", help="focus-team primary colour")
    p.add_argument("--accent", default="#5AA4D0", help="focus-team accent colour")
    p.add_argument("--out", required=True, help="output directory for PNG frames")
    p.add_argument("--vertical", action="store_true", help="1080x1920 instead of 1920x1080")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=FRAMES - 1)
    p.add_argument("--scale", type=int, default=100, help="resolution percentage")
    a = p.parse_args(argv)
    for name in ("primary", "accent"):
        if not re.fullmatch(r"#?[0-9A-Fa-f]{6}", getattr(a, name)):
            p.error(f"--{name} must be a #RRGGBB colour")
    for name in ("crest", "opponent"):
        path = getattr(a, name)
        if not os.path.isfile(path) or not path.lower().endswith(".png"):
            p.error(f"--{name} must be an existing PNG file")
    if not 1 <= a.samples <= 1024:
        p.error("--samples must be 1-1024")
    if not 1 <= a.scale <= 100:
        p.error("--scale must be 1-100")
    if not 0 <= a.start <= a.end < FRAMES:
        p.error(f"--start/--end must satisfy 0 <= start <= end < {FRAMES}")
    return a


# ---------------------------------------------------------------- helpers

def srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb(hexcode, k=1.0):
    h = hexcode.lstrip("#")
    lin = srgb_to_linear([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return (*(float(v) * k for v in lin), 1.0)


def clamp01(t):
    return max(0.0, min(1.0, t))


def smooth(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def out_cubic(t):
    return 1 - (1 - clamp01(t)) ** 3


def out_quart(t):
    return 1 - (1 - clamp01(t)) ** 4


def in_cubic(t):
    return clamp01(t) ** 3


def in_out_cubic(t):
    t = clamp01(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def out_back(t, s=1.6):
    t = clamp01(t) - 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def lerp(a, b, t):
    return a + (b - a) * t


def erode(a):
    """One-pixel grey erosion (cross-shaped minimum filter)."""
    p = np.pad(a, 1, mode="edge")
    return np.minimum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])


def chamfer(cov, px):
    """Ramp 0 -> 1 over `px` pixels inside an anti-aliased coverage mask.

    Cut edges become a short straight bevel instead of a blurred slope, so the
    geometry stays crisp and the bevel catches a thin highlight.
    """
    acc, cur = cov.copy(), cov
    for _ in range(px - 1):
        cur = erode(cur)
        acc += cur
    return acc / px


def coverage(d, half_width):
    """Anti-aliased coverage of the band |d| < half_width (d in wall units)."""
    return np.clip((half_width - d) * PX + 0.5, 0, 1).astype(np.float32)


def load_rgba(path, max_side):
    """Return a bottom-up float RGBA array, linear colour, straight alpha."""
    img = bpy.data.images.load(path)
    if img.channels < 4:
        raise SystemExit(f"[opening] {path} has no alpha channel")
    w, h = img.size
    s = max_side / max(w, h)
    img.scale(max(1, round(w * s)), max(1, round(h * s)))
    w, h = img.size
    arr = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(arr)
    arr = arr.reshape(h, w, 4)
    if arr[..., 3].min() > 0.99:
        raise SystemExit(f"[opening] {path} has no transparent pixels")
    if not img.is_float:
        arr[..., :3] = srgb_to_linear(arr[..., :3])
    bpy.data.images.remove(img)
    return arr


def crest_masks(path, side_px, ss=4):
    """Anti-aliased silhouette and light-area coverage at side_px, from a 4x supersample."""
    rgba = load_rgba(path, side_px * ss)
    luma = rgba[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    solid = rgba[..., 3] > 0.5
    light = solid & (luma > 0.18)
    h, w = solid.shape
    h2, w2 = -(-h // ss) * ss, -(-w // ss) * ss

    def down(m):
        m = np.pad(m.astype(np.float32), ((0, h2 - h), (0, w2 - w)))
        return m.reshape(h2 // ss, ss, w2 // ss, ss).mean(axis=(1, 3))
    return down(solid), down(light)


def place(dst, stamp, cx, cy, combine):
    h, w = stamp.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xs, ys = max(0, -x0), max(0, -y0)
    xe, ye = min(w, dst.shape[1] - x0), min(h, dst.shape[0] - y0)
    if xe <= xs or ye <= ys:
        return
    region = dst[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
    region[...] = combine(region, stamp[ys:ye, xs:xe])


def float_image(name, arr):
    """arr: (h, w) grey or (h, w, 4) RGBA, bottom-up, linear values 0..1.

    Generated float images sample as zero at render time in Blender 5.2, so
    each map is written to a 16-bit PNG and loaded back as a file image.
    """
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr, np.ones_like(arr)])
    h, w = arr.shape[:2]
    data = np.round(np.clip(arr[::-1], 0, 1) * 65535).astype(">u2")
    raw = b"".join(b"\x00" + row.tobytes() for row in data.reshape(h, w * 4))

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))

    path = os.path.join(tempfile.mkdtemp(prefix="relief_"), f"{name}.png")
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 6, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = "Non-Color"
    img.alpha_mode = "STRAIGHT"
    img.pack()
    return img


def grid_mesh(name, heights, step, uv_origin=(0.0, 0.0), uv_step=None, keep=None):
    """Mesh with one vertex per height-map pixel, centred on the origin.

    keep: optional (h-1, w-1) bool array; quads where it is False are left out.

    The Displace modifier samples generated images as zero in this Blender
    build, so the heights are written to the vertices directly.
    """
    h, w = heights.shape
    xs = (np.arange(w, dtype=np.float32) - (w - 1) / 2) * step
    ys = (np.arange(h, dtype=np.float32) - (h - 1) / 2) * step
    X, Y = np.meshgrid(xs, ys)
    co = np.stack([X, Y, heights.astype(np.float32)], -1).reshape(-1, 3)
    idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    if keep is not None:
        quads = quads[keep.ravel()]
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", co.ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", quads.ravel())
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    us, vs = uv_step or (1 / (w - 1), 1 / (h - 1))
    U, V = np.meshgrid(uv_origin[0] + np.arange(w) * us, uv_origin[1] + np.arange(h) * vs)
    uv = np.stack([U, V], -1).reshape(-1, 2).astype(np.float32)
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", uv[quads.ravel()].ravel())
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(quads), bool))
    me.update()
    return me


def link_object(name, data):
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def key_frames(fn):
    for f in range(FRAMES):
        fn(f)


def key_value(node, curve):
    def k(f):
        node.outputs[0].default_value = curve(f)
        node.outputs[0].keyframe_insert("default_value", frame=f)
    key_frames(k)


# ---------------------------------------------------------------- shading helpers

class Nodes:
    def __init__(self, name):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.nt = self.mat.node_tree
        self.nt.nodes.clear()
        self.out = self.new("ShaderNodeOutputMaterial")
        self.bsdf = self.new("ShaderNodeBsdfPrincipled")
        self.link(self.bsdf.outputs[0], self.out.inputs["Surface"])

    def new(self, kind, **props):
        node = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(node, k, v)
        return node

    def link(self, a, b):
        self.nt.links.new(a, b)

    def set(self, sock, value):
        if hasattr(value, "is_linked") or hasattr(value, "links"):
            self.link(value, sock)
        else:
            sock.default_value = value

    def math(self, op, a, b=0.0, clamp=False):
        node = self.new("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.set(node.inputs[0], a)
        self.set(node.inputs[1], b)
        return node.outputs[0]

    def mix(self, fac, a, b, colour=True):
        node = self.new("ShaderNodeMix", data_type="RGBA" if colour else "FLOAT")
        self.set(node.inputs[0], fac)
        ia, ib = (6, 7) if colour else (2, 3)
        self.set(node.inputs[ia], a)
        self.set(node.inputs[ib], b)
        return node.outputs[2 if colour else 0]

    def image(self, img, interpolation="Linear"):
        node = self.new("ShaderNodeTexImage", image=img, interpolation=interpolation,
                        extension="EXTEND")
        uv = self.new("ShaderNodeTexCoord")
        self.link(uv.outputs["UV"], node.inputs["Vector"])
        return node

    def value(self, curve):
        node = self.new("ShaderNodeValue")
        key_value(node, curve)
        return node.outputs[0]

    def coord(self, output, axis):
        tc = self.new("ShaderNodeTexCoord")
        sep = self.new("ShaderNodeSeparateXYZ")
        self.link(tc.outputs[output], sep.inputs[0])
        return sep.outputs[axis]

    def machined(self, face, face_rough, floor, floor_rough, floor_depth, metallic=0.55):
        """Anodised plate: bright bare-metal chamfers, darker bead-blasted floors.

        Edge = how far the shading normal tilts away from the plate normal, so
        every cut edge gets a thin silver bevel highlight.
        """
        nz = self.coord("Normal", "Z")
        edge = self.math("MULTIPLY", self.math("SUBTRACT", 0.985, nz), 9.0, clamp=True)
        depth = self.math("MULTIPLY", self.coord("Object", "Z"), -1.0 / floor_depth, clamp=True)
        silver = (0.80, 0.83, 0.87, 1)
        self.link(self.mix(edge, self.mix(depth, face, floor), silver), self.bsdf.inputs["Base Color"])
        rough = self.mix(depth, face_rough, floor_rough, colour=False)
        self.link(self.mix(edge, rough, 0.1, colour=False), self.bsdf.inputs["Roughness"])
        self.link(self.mix(edge, metallic, 1.0, colour=False), self.bsdf.inputs["Metallic"])
        self.bsdf.inputs["Coat Roughness"].default_value = 0.05
        self.link(self.mix(depth, 0.6, 0.0, colour=False), self.bsdf.inputs["Coat Weight"])
        return edge, depth


# ---------------------------------------------------------------- scene

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, FRAMES - 1
    scene.render.fps = FPS
    w = bpy.data.worlds.new("Arena")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Color"].default_value = (0.004, 0.006, 0.012, 1)
    # Reflections see a dim studio (a soft horizon band and a top glow) so the
    # steel reads as metal; the camera itself still sees a dark void.
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.004, 0.005, 0.008, 1)
    els[1].position, els[1].color = 1.0, (0.35, 0.42, 0.52, 1)
    band = els.new(0.52)
    band.color = (0.22, 0.26, 0.32, 1)
    dip = els.new(0.62)
    dip.color = (0.02, 0.025, 0.035, 1)
    studio = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(ramp.outputs["Color"], studio.inputs["Color"])
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(studio.outputs[0], mix.inputs[1])
    nt.links.new(bg.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["World Output"].inputs["Surface"])
    return scene


def rink_lines(X, Y):
    """Anti-aliased coverage of the rink markings (1 wall unit = 5 ft)."""
    w = 0.08
    hx, hz, rc = 20.0, 8.5, 5.6
    qx, qz = np.abs(X) - (hx - rc), np.abs(Y) - (hz - rc)
    corner = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - rc
    cov = coverage(np.abs(corner), w)
    inside = np.clip(-corner * PX + 0.5, 0, 1)
    for x0 in (0.0, -5.0, 5.0, -17.8, 17.8):
        cov = np.maximum(cov, coverage(np.abs(X - x0), w * (1.6 if x0 == 0 else 1.2)) * inside)
    for cx in (-13.8, 13.8):
        for cz in (-4.4, 4.4):
            d = np.hypot(X - cx, Y - cz)
            cov = np.maximum(cov, coverage(np.abs(d - 3.0), w))
            cov = np.maximum(cov, coverage(d, 0.2))
    for gx in (-17.8, 17.8):
        d = np.hypot(X - gx, Y)
        cov = np.maximum(cov, coverage(np.abs(d - 1.2), w) * (np.sign(-gx) * (X - gx) > 0))
    return cov


def wall_maps(crest, opponent):
    """Height (0 = plate face, negative = cut in) and glow mask, bottom-up."""
    W, H = NX * PLATE_PX, NZ * PLATE_PX
    xs = (np.arange(W, dtype=np.float32) + 0.5) / PX - WALL_W / 2
    ys = (np.arange(H, dtype=np.float32) + 0.5) / PX - WALL_H / 2
    X, Y = np.meshgrid(xs, ys)
    height = np.zeros((H, W), np.float32)

    # Debossed crests on alternate plates: silhouette cut 0.09, light areas
    # stepped back up to -0.04, so each crest has two clear depth steps.
    stamps = []
    for path in (crest, opponent):
        solid, light = crest_masks(path, int(PLATE_PX * 0.74))
        stamps.append(-0.09 * chamfer(solid, 2) + 0.05 * chamfer(light, 2))
    for j in range(NZ):
        for i in range(NX):
            if abs(i - NX // 2) <= 1 and abs(j - NZ // 2) <= 1:
                continue                                  # centre circle plates
            place(height, stamps[(i + j) % 2], (i + 0.5) * PLATE_PX, (j + 0.5) * PLATE_PX, np.minimum)

    lines = rink_lines(X, Y)
    height = np.minimum(height, -0.07 * chamfer(lines, 2))
    glow = erode(lines)

    # Centre circle: a lip step, then the deep well the medallion comes from.
    r = np.hypot(X, Y)
    height = np.minimum(height, -0.05 * chamfer(coverage(r, RECESS_R + 0.2), 2))
    height = np.minimum(height, -RECESS_D * chamfer(coverage(r, RECESS_R), 3))

    # Every plate has a chamfered edge that drops into the joint.
    lx = np.arange(W) % PLATE_PX
    ly = np.arange(H) % PLATE_PX
    ex = np.minimum(lx, PLATE_PX - 1 - lx)
    ey = np.minimum(ly, PLATE_PX - 1 - ly)
    edge = np.minimum(ex[None, :], ey[:, None]).astype(np.float32)
    height = np.minimum(height, -0.12 * (1 - np.clip(edge / 4, 0, 1)))
    return height, glow


def plate_motion(x, z, f):
    """Plate offset into the wall (+y) and tilts (radians) at frame f."""
    seed = math.sin(x * 12.9898 + z * 78.233) * 43758.5453
    jitter = (seed - math.floor(seed)) * 2.5
    # Assembly: plates slam in left to right behind the first light bar.
    t0 = (x + WALL_W / 2) / WALL_W * 18 - 3 + jitter
    e = out_back((f - t0) / 7, 1.2)
    push = 1.1 * (1 - e)
    tilt_z = 0.55 * (1 - e)
    tilt_x = -0.25 * (1 - e) * math.copysign(1, z or 1)
    # Shock wave from the medallion burst.
    r = math.hypot(x, z)
    if f >= HIT and r > 4.0:
        front = 3.0 + (f - HIT) * 1.5
        wave = 0.3 * math.exp(-(f - HIT) / 14) * math.exp(-((r - front) / 1.4) ** 2)
        push += wave
        tilt_z -= math.copysign(wave * 0.35, x or 1)
        tilt_x += math.copysign(wave * 0.35, z or 1)
    # The second light bar lifts the plates it passes.
    bar = sweep_c_x(f)
    if bar is not None:
        push -= 0.07 * math.exp(-((x - bar) / 2.4) ** 2)
    return push, tilt_x, tilt_z


def sweep_c_x(f):
    """x position of the late light bar, or None when it is off."""
    if not 80 <= f <= 108:
        return None
    return 26 - 52 * in_out_cubic((f - 80) / 28)


def wall(args):
    height, glow = wall_maps(args.crest, args.opponent)
    g_img = float_image("WallGlow", glow)
    m = Nodes("Plate")
    m.machined(rgb(args.primary, 1.25), 0.24, rgb(args.primary, 0.32), 0.6, 0.035)
    grain = m.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 1400.0
    bump = m.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.012
    bump.inputs["Distance"].default_value = 0.002
    m.link(grain.outputs["Fac"], bump.inputs["Height"])
    m.link(bump.outputs["Normal"], m.bsdf.inputs["Normal"])

    # Light runs along the engraved lines from left to right (26-42), and the
    # lines flare on the medallion burst.
    u = m.coord("UV", "X")
    front = m.value(lambda f: -0.15 + 1.3 * smooth((f - 26) / 16))
    ahead = m.math("SUBTRACT", front, u)
    reach = m.math("MULTIPLY", ahead, 12.0, clamp=True)
    head = m.math("SUBTRACT", 1.0, m.math("MULTIPLY", m.math("ABSOLUTE", ahead), 16.0), clamp=True)
    lit = m.math("MULTIPLY", m.math("ADD", reach, m.math("MULTIPLY", head, 6.0)),
                 m.image(g_img).outputs["Color"])
    glow_k = m.value(lambda f: 0.0 if f < 26 else
                     1.1 + 3.0 * math.exp(-(f - 26) / 10) + (6.0 * math.exp(-(f - HIT) / 5) if f >= HIT else 0))
    m.link(m.math("MULTIPLY", lit, glow_k), m.bsdf.inputs["Emission Strength"])
    m.bsdf.inputs["Emission Color"].default_value = rgb(args.accent)

    W = NX * PLATE_PX
    H = NZ * PLATE_PX
    plates = []
    for j in range(NZ):
        for i in range(NX):
            patch = height[j * PLATE_PX:(j + 1) * PLATE_PX, i * PLATE_PX:(i + 1) * PLATE_PX]
            # The well has no floor: it opens onto the dark backing.
            floor = patch <= -RECESS_D * 0.98
            keep = ~(floor[:-1, :-1] & floor[:-1, 1:] & floor[1:, :-1] & floor[1:, 1:])
            me = grid_mesh(f"Plate{i:02d}_{j:02d}", patch, 1 / PX,
                           ((i * PLATE_PX + 0.5) / W, (j * PLATE_PX + 0.5) / H), (1 / W, 1 / H), keep)
            me.materials.append(m.mat)
            obj = link_object(me.name, me)
            x, z = (i + 0.5) * PLATE - WALL_W / 2, (j + 0.5) * PLATE - WALL_H / 2
            plates.append((obj, x, z))

    def k(f):
        for obj, x, z in plates:
            push, tx, tz = plate_motion(x, z, f)
            obj.location = (x, push, z)
            obj.rotation_euler = (math.pi / 2 + tx, 0, tz)  # face -Y, towards the camera
            obj.keyframe_insert("location", frame=f)
            obj.keyframe_insert("rotation_euler", frame=f)
    key_frames(k)

    # Dark backing behind the joints and behind plates that are still flying in.
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 1.6, 0), rotation=(math.pi / 2, 0, 0))
    back = bpy.context.object
    back.scale = (WALL_W + 8, WALL_H + 8, 1)
    b = Nodes("Backing")
    b.bsdf.inputs["Base Color"].default_value = (0.004, 0.005, 0.008, 1)
    b.bsdf.inputs["Roughness"].default_value = 0.8
    back.data.materials.append(b.mat)


def medal_maps(crest, size=900):
    """Medallion face height and colour; face units are the face side (0..1)."""
    u = (np.arange(size, dtype=np.float32) + 0.5) / size - 0.5
    U, V = np.meshgrid(u, u)
    r = np.hypot(U, V)
    px = 1 / size
    cov = lambda d, hw: np.clip((hw - d) / px + 0.5, 0, 1).astype(np.float32)
    height = -0.03 * chamfer(cov(r, 0.47), 2)                   # enamel field (wall units)
    height = np.minimum(height, -0.045 * chamfer(cov(np.abs(r - 0.488), 0.006), 2))
    solid, light = crest_masks(crest, int(size * 0.78))
    lift = np.zeros((size, size), np.float32)
    place(lift, 0.06 * chamfer(solid, 3) + 0.025 * chamfer(light, 2), size / 2, size / 2 + size * 0.01, np.maximum)
    rgba = load_rgba(crest, int(size * 0.78))
    colour = np.zeros((size, size, 4), np.float32)
    place(colour, rgba, size / 2, size / 2 + size * 0.01, lambda a, b: b)
    return height + lift, colour


def medal_motion(f):
    """Medallion depth (+y into the wall) and spin about the vertical axis."""
    if f < HIT - 2:
        return MEDAL_IN, 0.0
    t = f - (HIT - 2)
    # Punch out in 4 frames with an overshoot towards the camera, then a hard
    # settle; a small lock-in kick lands on the thunk at frame 66.
    y = lerp(MEDAL_IN, MEDAL_OUT, out_back(t / 5, 2.4))
    y -= 0.25 * math.exp(-max(0, t - 5) / 3) * math.sin(max(0, t - 5) * 0.9)
    if f >= 66:
        y += 0.06 * math.exp(-(f - 66) / 3) * math.sin((f - 66) * 1.4 + 0.6)
    spin = 2 * math.pi * (1 - out_quart((f - HIT) / 12)) if f >= HIT else 2 * math.pi
    return y, spin


def medallion(args):
    rig = bpy.data.objects.new("Medal", None)
    bpy.context.collection.objects.link(rig)

    bpy.ops.mesh.primitive_cylinder_add(vertices=256, radius=MEDAL_R, depth=MEDAL_D,
                                        rotation=(math.pi / 2, 0, 0))
    body = bpy.context.object
    bev = body.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments, bev.limit_method = 0.035, 2, "ANGLE"
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=0.6)
    except (AttributeError, RuntimeError):
        bpy.ops.object.shade_smooth()
    steel = Nodes("Steel")
    steel.bsdf.inputs["Base Color"].default_value = (0.62, 0.66, 0.72, 1)
    steel.bsdf.inputs["Metallic"].default_value = 1.0
    steel.bsdf.inputs["Roughness"].default_value = 0.16
    body.data.materials.append(steel.mat)
    body.parent = rig

    height, colour = medal_maps(args.crest)
    c_img = float_image("MedalColour", colour)
    side = 2 * MEDAL_R * 0.69
    n = height.shape[0]
    c = (np.arange(n - 1) + 1 - n / 2) / n                       # quad centres, face units
    disc = np.hypot(*np.meshgrid(c, c)) < 0.497                   # round face, no square seam
    face = link_object("MedalFace", grid_mesh("MedalFace", height, side / (n - 1), keep=disc))
    face.rotation_euler = (math.pi / 2, 0, 0)
    face.location = (0, -MEDAL_D / 2 - 0.004, 0)
    face.parent = rig

    m = Nodes("MedalFace")
    edge, _ = m.machined((0.6, 0.64, 0.7, 1), 0.2, rgb(args.primary, 0.75), 0.35, 0.025, metallic=0.9)
    col = m.image(c_img)
    alpha = col.outputs["Alpha"]
    base = m.bsdf.inputs["Base Color"].links[0].from_socket
    m.link(m.mix(m.math("MULTIPLY", alpha, m.math("SUBTRACT", 1.0, edge)), base, col.outputs["Color"]),
           m.bsdf.inputs["Base Color"])
    # The crest glows faintly so it reads on the dark enamel.
    m.link(col.outputs["Color"], m.bsdf.inputs["Emission Color"])
    m.link(m.math("MULTIPLY", alpha, m.value(lambda f: 0.0 if f < HIT - 2 else 0.9)),
           m.bsdf.inputs["Emission Strength"])
    face.data.materials.append(m.mat)

    # Glow ring just inside the mouth of the well, around the medallion.
    bpy.ops.mesh.primitive_torus_add(major_radius=RECESS_R - 0.09, minor_radius=0.045,
                                     major_segments=256, rotation=(math.pi / 2, 0, 0),
                                     location=(0, 0.12, 0))
    ring = bpy.context.object
    g = Nodes("Ring")
    g.bsdf.inputs["Base Color"].default_value = (0.01, 0.015, 0.02, 1)
    g.bsdf.inputs["Emission Color"].default_value = rgb(args.accent)

    def ring_k(f):
        if f < HIT - 1:
            return 0.0
        v = 5.0 + 70.0 * math.exp(-(f - HIT + 1) / 4)
        return v + (18.0 * math.exp(-(f - 66) / 4) if f >= 66 else 0)
    g.link(g.value(ring_k), g.bsdf.inputs["Emission Strength"])
    ring.data.materials.append(g.mat)

    def k(f):
        y, spin = medal_motion(f)
        rig.location = (0, y, 0)
        rig.rotation_euler = (0, 0, spin)
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)
        for part in (body, face):
            part.hide_render = f < HIT - 3
            part.keyframe_insert("hide_render", frame=f)
    key_frames(k)


def light(name, kind, loc, aim, colour, size=(1, 1), glossy=False):
    data = bpy.data.lights.new(name, kind)
    data.color = colour[:3]
    if kind == "AREA":
        data.shape = "RECTANGLE"
        data.size, data.size_y = size
    obj = link_object(name, data)
    obj.location = loc
    target = link_object(f"{name}Aim", None)
    target.location = aim
    con = obj.constraints.new("TRACK_TO")
    con.target, con.track_axis, con.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    obj.visible_glossy = glossy
    return obj, target


def key_energy(obj, curve):
    def k(f):
        obj.data.energy = curve(f)
        obj.data.keyframe_insert("energy", frame=f)
    key_frames(k)


def moving_bar(name, colour, size, energy, path):
    """A strip light whose position and aim follow path(f) -> (loc, aim) or None."""
    bar, aim = light(name, "AREA", (0, -3, 0), (0, 0, 0), colour, size, True)

    def k(f):
        p = path(f)
        if p is not None:
            bar.location, aim.location = p
        bar.data.energy = energy(f) if p is not None else 0.0
        for obj in (bar, aim):
            obj.keyframe_insert("location", frame=f)
        bar.data.keyframe_insert("energy", frame=f)
    key_frames(k)


def lights(args):
    # Lit from frame 0; the impact at 26 snaps everything up with a flash.
    snap = lambda f: 0.7 if f < 26 else 1.0 + 0.6 * math.exp(-(f - 26) / 3)
    hit = lambda f: 1.0 + (0.5 * math.exp(-(f - HIT) / 3) if f >= HIT else 0)
    key, _ = light("Key", "AREA", (-26, -4.0, 3.0), (0, 0.3, 0), (1.0, 0.97, 0.93), (1.2, 12))
    key_energy(key, lambda f: 26000 * snap(f) * hit(f))
    rim, _ = light("Rim", "AREA", (26, -4.0, -2.0), (0, 0.3, 0), rgb(args.accent), (1.2, 12))
    key_energy(rim, lambda f: 16000 * snap(f) * hit(f))
    top, _ = light("Top", "AREA", (0, -6, 16), (0, 0, 0), (0.75, 0.85, 1.0), (20, 3), True)
    key_energy(top, lambda f: 2500 * snap(f))
    # A front light lets the raised medallion face read after it rises.
    face, _ = light("Face", "AREA", (3.0, -14, 5), (0, 0, 0), (0.92, 0.96, 1.0), (4, 4))
    key_energy(face, lambda f: 1800 * smooth((f - HIT + 2) / 6))

    # Bar A races left to right ahead of the plates (0-24).
    def path_a(f):
        if f > 30:
            return None
        x = -22 + 30 * (f / 30) ** 0.8
        return (x, -3.0, 0), (x + 3.5, 0.5, 0)
    moving_bar("BarA", (0.9, 0.95, 1.0), (0.3, 22), lambda f: 3000 * (1 - smooth((f - 22) / 8)), path_a)

    # Bar B glints diagonally across the medallion (56-70, shing at 62).
    def path_b(f):
        if not 54 <= f <= 72:
            return None
        x = -6 + 12 * in_out_cubic((f - 54) / 16)
        return (x, -3.4, 0), (x + 1.2, 0, 0)
    moving_bar("BarB", (1, 1, 1), (0.25, 9), lambda f: 1400 * smooth((f - 54) / 3) * (1 - smooth((f - 69) / 3)), path_b)

    # Bar C crosses the whole wall right to left (80-108).
    def path_c(f):
        x = sweep_c_x(f)
        return None if x is None else ((x, -2.6, 0.5), (x - 3.0, 0.5, 0))
    moving_bar("BarC", (0.85, 0.93, 1.0), (0.3, 20),
               lambda f: 7000 * smooth((f - 80) / 4) * (1 - smooth((f - 104) / 4)), path_c)


# ---------------------------------------------------------------- camera

def camera_pose(f):
    """Target, azimuth (deg, + = camera left of target), distance, height, roll (deg)."""
    if f < 26:                       # fast track along the wall, slowing into the impact
        t = f / 26
        e = 1 - (1 - t) ** 1.6
        tx, az, dist, hgt, roll = lerp(-15, -5.5, e), lerp(64, 50, e), lerp(5.0, 6.2, e), lerp(1.2, 0.8, e), lerp(-7, -2, e)
        tz = 0.6
    elif f < HIT:                    # whip to the centre, landing on the burst
        e = in_out_cubic((f - 26) / (HIT - 26))
        tx, az, dist, hgt, roll = lerp(-5.5, 0, e), lerp(50, 6, e), lerp(6.2, 12.0, e), lerp(0.8, 0.2, e), lerp(-2, 0, e)
        roll += 5 * math.sin(math.pi * e)
        tz = lerp(0.6, 0, e)
    elif f < 112:                    # steady orbit with parallax over the plates
        e = (f - HIT) / (112 - HIT)
        tx, tz = 0.0, 0.0
        az = lerp(6, -10, smooth(e) * 0.4 + e * 0.6)
        dist = lerp(12.0, 10.6, e)
        hgt = lerp(0.2, 1.0, e)
        roll = 0.0
        if f < HIT + 10:             # recoil from the burst
            k = math.exp(-(f - HIT) / 3)
            dist += 0.7 * k
            tx += 0.08 * k * math.sin((f - HIT) * 2.6)
            tz += 0.06 * k * math.cos((f - HIT) * 3.1)
    else:                            # hard push into the crest
        e = in_cubic((f - 112) / 22)
        tx, tz = 0.0, 0.0
        az, dist, hgt, roll = lerp(-10, -2, e), lerp(10.6, 3.4, e), lerp(1.0, 0.3, e), lerp(0, 4, e)
    ty = MEDAL_OUT - MEDAL_D / 2 if f >= HIT else 0.0
    return Vector((tx, ty, tz)), az, dist, hgt, roll


def camera(scene, args):
    data = bpy.data.cameras.new("Camera")
    data.sensor_fit = "HORIZONTAL"
    data.lens = 46 if args.vertical else 24
    # Keep the medallion above the matchup rail Remotion draws later.
    data.shift_y = -0.08 if args.vertical else -0.09
    data.clip_start, data.clip_end = 0.05, 200
    cam = link_object("Camera", data)
    cam.rotation_mode = "QUATERNION"
    scene.camera = cam
    data.dof.use_dof = True
    data.dof.aperture_fstop = 5.6
    prev = [None]

    def k(f):
        target, az, dist, hgt, roll = camera_pose(f)
        a = math.radians(az)
        loc = target + Vector((-dist * math.sin(a), -dist * math.cos(a), hgt))
        q = (target - loc).to_track_quat("-Z", "Y") @ Quaternion((0, 0, 1), math.radians(roll))
        if prev[0] is not None and q.dot(prev[0]) < 0:
            q.negate()
        prev[0] = q
        cam.location, cam.rotation_quaternion = loc, q
        data.dof.focus_distance = (target - loc).length
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        data.dof.keyframe_insert("focus_distance", frame=f)
    key_frames(k)
    return cam


def render_settings(scene, args):
    scene.render.engine = "BLENDER_EEVEE"
    ee = scene.eevee
    ee.taa_render_samples = args.samples
    for attr in ("use_raytracing", "use_shadows"):
        if hasattr(ee, attr):
            setattr(ee, attr, True)
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.4
    scene.render.resolution_x, scene.render.resolution_y = (1080, 1920) if args.vertical else (1920, 1080)
    scene.render.resolution_percentage = args.scale
    scene.render.filter_size = 1.0
    transforms = scene.view_settings.bl_rna.properties["view_transform"].enum_items.keys()
    scene.view_settings.view_transform = "Khronos PBR Neutral" if "Khronos PBR Neutral" in transforms \
        else "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = os.path.join(os.path.abspath(args.out), "frame_")
    scene.frame_start, scene.frame_end = args.start, args.end
    compositor(scene)


def compositor(scene):
    """Bloom on the light lines, the ring and the brightest edge glints only."""
    tree = bpy.data.node_groups.new("Opening", "CompositorNodeTree") \
        if hasattr(scene, "compositing_node_group") else None
    if tree is None:
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
    else:
        scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    for name, value in [("glare_type", "BLOOM"), ("quality", "HIGH")]:
        if hasattr(glare, name):
            setattr(glare, name, value)
    for name, value in [("Type", "Bloom"), ("Threshold", 1.4), ("Strength", 0.4), ("Size", 0.5)]:
        sock = glare.inputs.get(name)
        if sock is not None:
            try:
                sock.default_value = value
            except TypeError:
                pass
    tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
    if hasattr(tree, "interface"):
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        out = tree.nodes.new("NodeGroupOutput")
        tree.links.new(glare.outputs["Image"], out.inputs[0])
    else:
        out = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(glare.outputs["Image"], out.inputs["Image"])


def build(args):
    scene = reset_scene()
    wall(args)
    medallion(args)
    lights(args)
    camera(scene, args)
    render_settings(scene, args)
    return scene


def main():
    args = parse_args(sys.argv)
    os.makedirs(args.out, exist_ok=True)
    build(args)
    bpy.ops.render.render(animation=True)
    expected = [os.path.join(args.out, f"frame_{f:04d}.png") for f in range(args.start, args.end + 1)]
    missing = [p for p in expected if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"[opening] {len(missing)} frames missing, first {missing[0]}")
    print(f"[opening] rendered frames {args.start}-{args.end} to {args.out}")


if __name__ == "__main__":
    main()
