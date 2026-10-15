"""Render the Black Ice opening stinger with Blender Cycles (spec 003, hhg-3r5.86).

Run headless:

    Blender -b --factory-startup --python-exit-code 1 \
        --python v3/blender/black_ice_opening.py -- \
        --crest crest.png --out frames/ [--vertical]

A black-ice pond at the base of the Wasatch range, at dusk, with snow falling.
The old (blue) crest is frozen under the ice. No text. Timing (135 frames,
4.5 s at 30 fps):

- 0-16    a skate blade crosses the ice over the frozen crest and carves it.
- 10-30   the camera pulls back: pond, snow banks, mountains.
- 24-70   the crest wakes: it heats from cold blue to red-hot and shoves
          against the ice three times (30, 44, 58); each shove cracks it.
- 70      it breaks through; shards fly.
- 70-84   it rises and turns to face the camera, front and centre.
- 84-134  hold with a slow push. Snowflakes land on the hot crest and fizzle
          (88, 98, 107, 117). The crest cools from orange-hot to red.

The red crest is not delivered yet: the heat is procedural (the blue crest's
light and dark areas mapped to hot colours) and is a placeholder.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector

FRAMES = 135
FPS = 30
ICE_T = 0.3                  # ice thickness; the top face is z = 0
POND_R = 13.0                # pond radius; snow banks rise beyond it
CREST = 2.4                  # crest size (longest side)
HOME = Vector((0.0, 1.6, 0.0))    # where the crest is frozen (x, y)
SHOVES = (30, 44, 58)
BREAK = 70
HITS = (88, 98, 107, 117)    # snowflakes that land on the hot crest
VIEW = (40.665, -111.905)    # the pond: Salt Lake valley floor, about 10 km west of Mount Olympus


# ---------------------------------------------------------------- arguments

def parse_args(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="black_ice_opening")
    p.add_argument("--crest", required=True, help="focus-team crest, transparent PNG")
    p.add_argument("--out", required=True, help="output directory for PNG frames")
    p.add_argument("--vertical", action="store_true", help="1080x1920 instead of 1920x1080")
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=FRAMES - 1)
    p.add_argument("--scale", type=int, default=100, help="resolution percentage")
    p.add_argument("--cpu", action="store_true", help="render on the CPU instead of Metal")
    p.add_argument("--dem", help="Wasatch elevation grid from wasatch_dem.py (.npz); none = no mountains")
    a = p.parse_args(argv)
    if not os.path.isfile(a.crest) or not a.crest.lower().endswith(".png"):
        p.error("--crest must be an existing PNG file")
    if a.dem and not os.path.isfile(a.dem):
        p.error("--dem must be an existing .npz file")
    if not 1 <= a.samples <= 4096:
        p.error("--samples must be 1-4096")
    if not 1 <= a.scale <= 100:
        p.error("--scale must be 1-100")
    if not 0 <= a.start <= a.end < FRAMES:
        p.error(f"--start/--end must satisfy 0 <= start <= end < {FRAMES}")
    return a


# ---------------------------------------------------------------- helpers

def clamp01(t):
    return max(0.0, min(1.0, t))


def smooth(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def out_cubic(t):
    return 1 - (1 - clamp01(t)) ** 3


def in_cubic(t):
    return clamp01(t) ** 3


def in_out(t):
    t = clamp01(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def out_back(t, s=1.7):
    t = clamp01(t) - 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def lerp(a, b, t):
    return a + (b - a) * t


def key_frames(fn):
    for f in range(FRAMES):
        fn(f)


def link_object(name, data):
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def heat(f):
    """0 = cold blue crest, 1 = orange-hot. Each shove adds a jump."""
    h = 0.55 * smooth((f - 20) / 50)
    for s in SHOVES:
        h += 0.15 * smooth((f - s) / 4)
    if f >= BREAK:
        h += 0.35 * math.exp(-(f - BREAK) / 6)
    return h


def cool(f):
    """After the break the crest cools from orange-hot to a steady red."""
    return smooth((f - 80) / 40)


def blade_x(f):
    return lerp(-4.5, 4.5, f / 16) if f <= 16 else 4.5 + (f - 16) * 0.55


def blade_y():
    return HOME.y - 0.25


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
        if hasattr(value, "links"):
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

    def value(self, curve):
        node = self.new("ShaderNodeValue")

        def k(f):
            node.outputs[0].default_value = curve(f)
            node.outputs[0].keyframe_insert("default_value", frame=f)
        key_frames(k)
        return node.outputs[0]

    def obj_xyz(self):
        tc = self.new("ShaderNodeTexCoord")
        sep = self.new("ShaderNodeSeparateXYZ")
        self.link(tc.outputs["Object"], sep.inputs[0])
        return tc, sep.outputs["X"], sep.outputs["Y"], sep.outputs["Z"]

    def band(self, d, inner, outer):
        """1 inside |d| < inner, falling to 0 at outer."""
        node = self.new("ShaderNodeMapRange", clamp=True)
        self.link(self.math("ABSOLUTE", d), node.inputs["Value"])
        node.inputs["From Min"].default_value = inner
        node.inputs["From Max"].default_value = outer
        node.inputs["To Min"].default_value = 1.0
        node.inputs["To Max"].default_value = 0.0
        return node.outputs[0]

    def ramp(self, fac, stops):
        node = self.new("ShaderNodeValToRGB")
        self.set(node.inputs["Fac"], fac)
        els = node.color_ramp.elements
        els[0].position, els[0].color = stops[0]
        els[1].position, els[1].color = stops[-1]
        for pos, c in stops[1:-1]:
            e = els.new(pos)
            e.color = c
        return node.outputs["Color"]


def noise2(shape, scale, seed, octaves=6, ridged=False):
    """Fractal value noise on a grid, 0..1."""
    rng = np.random.default_rng(seed)
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = max(2, int(scale * 2 ** o))
        g = rng.random((n + 1, n + 1)).astype(np.float32)
        ys = np.linspace(0, n, h, dtype=np.float32)
        xs = np.linspace(0, n, w, dtype=np.float32)
        y0, x0 = np.floor(ys).astype(int).clip(0, n - 1), np.floor(xs).astype(int).clip(0, n - 1)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx
        b = g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx
        layer = a * (1 - fy) + b * fy
        if ridged:
            layer = 1 - np.abs(layer * 2 - 1)
            layer = layer * layer
        out += layer * amp
        total += amp
        amp *= 0.5
    return out / total


def grid(name, X, Y, Z):
    """Mesh from a height grid (world coordinates)."""
    h, w = Z.shape
    co = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", co.ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", quads.ravel())
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(quads), bool))
    me.update()
    return link_object(name, me)


# ---------------------------------------------------------------- scene

SKY_LOW = (0.05, 0.04, 0.034, 1)      # light pollution from the valley, low on the horizon
SKY_HIGH = (0.004, 0.005, 0.008, 1)   # overcast night overhead: no stars


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, FRAMES - 1
    scene.render.fps = FPS
    w = bpy.data.worlds.new("Night")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    els = ramp.color_ramp.elements
    # World 'Generated' Z is the view direction (-1..1); the horizon is at 0.
    els[0].position, els[0].color = 0.0, SKY_LOW
    els[1].position, els[1].color = 0.45, SKY_HIGH
    e = els.new(0.08)
    e.color = (0.02, 0.019, 0.02, 1)
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    return scene


def mountains(args):
    """The real Wasatch front (public elevation data, see wasatch_dem.py), seen
    east from the valley floor. Haze and depth of field keep it a hint."""
    if not args.dem:
        return
    d = np.load(args.dem)
    elev = d["elev"]
    north, south, west, east = (float(d[k]) for k in ("north", "south", "west", "east"))
    step = 3
    elev = elev[::step, ::step]
    rows, cols = elev.shape
    lats = north + (np.arange(rows) + 0.5) / rows * (south - north)
    lons = west + (np.arange(cols) + 0.5) / cols * (east - west)
    keep = lons >= VIEW[1] - 0.01                      # nothing behind the camera
    elev, lons = elev[:, keep], lons[keep]
    lat0, lon0 = VIEW
    # Local metres: +y = east (the camera looks east), +x = south.
    Yg = ((lons - lon0) * 111320 * math.cos(math.radians(lat0)))[None, :].repeat(rows, 0)
    Xg = ((lat0 - lats) * 110574)[:, None].repeat(len(lons), 1)
    r0 = int(np.argmin(np.abs(lats - lat0)))
    c0 = int(np.argmin(np.abs(lons - lon0)))
    ground = float(np.median(elev[max(0, r0 - 3):r0 + 4, max(0, c0 - 3):c0 + 4]))
    Z = elev - ground - 1.5
    # The local pond and snow banks replace the terrain right around the camera.
    near = (np.abs(Xg) < 60) & (Yg > -40) & (Yg < 90)
    Z = np.where(near, -6.0, Z)
    obj = grid("Wasatch", Xg.astype(np.float32), Yg.astype(np.float32), Z.astype(np.float32))
    m = Nodes("Mountain")
    geo = m.new("ShaderNodeNewGeometry")
    sep = m.new("ShaderNodeSeparateXYZ")
    m.link(geo.outputs["Normal"], sep.inputs[0])
    _, _, _, z = m.obj_xyz()
    tex = m.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 0.004
    tex.inputs["Detail"].default_value = 14
    tex.inputs["Roughness"].default_value = 0.65
    # Snow holds on slopes gentler than about 40 degrees and above the foothills;
    # steep rock and the scrub-oak foothills stay dark.
    flat = m.math("MULTIPLY", m.math("SUBTRACT", m.math("ADD", sep.outputs["Z"],
                                                          m.math("MULTIPLY", tex.outputs["Fac"], 0.35)), 0.82), 5.0, clamp=True)
    high = m.math("MULTIPLY", m.math("SUBTRACT", m.math("ADD", z, m.math("MULTIPLY", tex.outputs["Fac"], 300)), 500), 0.004,
                  clamp=True)
    snow = m.math("MULTIPLY", flat, high)
    valley = m.math("SUBTRACT", 1.0, m.math("MULTIPLY", z, 0.02), clamp=True)     # snowy valley floor
    snow = m.math("MAXIMUM", snow, m.math("MULTIPLY", valley, 0.7))
    colour = m.mix(snow, (0.012, 0.012, 0.012, 1), (0.72, 0.75, 0.8, 1))
    cam = m.new("ShaderNodeCameraData")
    haze = m.math("MULTIPLY", m.math("SUBTRACT", cam.outputs["View Distance"], 2500), 1 / 14000, clamp=True)
    m.link(m.mix(m.math("ADD", m.math("MULTIPLY", haze, 0.55), 0.25), colour, SKY_LOW), m.bsdf.inputs["Base Color"])
    m.bsdf.inputs["Roughness"].default_value = 0.85
    obj.data.materials.append(m.mat)
    # Night light on the range: a low, cool moon from the south-west.
    sun = bpy.data.lights.new("Moon", "SUN")
    sun.color = (0.72, 0.8, 1.0)
    sun.energy = 0.35
    sun.angle = math.radians(0.6)
    so = link_object("Moon", sun)
    so.rotation_euler = (Vector((0, 0, 0)) - Vector((60, -40, 30))).to_track_quat("-Z", "Y").to_euler()


def banks():
    """Pond floor inside, snow banks and a snowy shore outside."""
    n = 220
    xs = np.linspace(-45, 45, n, dtype=np.float32)
    ys = np.linspace(-30, 70, n, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X - HOME.x * 0.3, (Y - 3) * 0.85)
    lump = noise2((n, n), 5, 3, 6)
    edge = np.clip((r - POND_R) / 2.5, 0, 1)
    Z = np.where(r < POND_R, -2.2 - 0.3 * lump, -2.2 + 2.45 * edge + 0.35 * lump * edge)
    obj = grid("Banks", X, Y, Z)
    m = Nodes("Snow")
    _, _, _, z = m.obj_xyz()
    floor = m.math("MULTIPLY", m.math("SUBTRACT", -1.2, z), 2.0, clamp=True)
    m.link(m.mix(floor, (0.62, 0.68, 0.8, 1), (0.006, 0.008, 0.012, 1)), m.bsdf.inputs["Base Color"])
    m.bsdf.inputs["Roughness"].default_value = 0.7
    obj.data.materials.append(m.mat)


def ice():
    """Skated pond ice: clear and dark, scuffed by skates, with trapped bubbles,
    long natural pressure cracks, a frosted underside, heat cracks that spread
    on each shove, and the hole where the crest breaks out."""
    centre = (HOME.x * 0.3, 3.0)
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=POND_R + 1.5, depth=ICE_T,
                                        location=(centre[0], centre[1], -ICE_T / 2))
    slab = bpy.context.object
    slab.name = "Ice"
    m = Nodes("BlackIce")
    b = m.bsdf
    b.inputs["Base Color"].default_value = (0.9, 0.95, 1.0, 1)
    b.inputs["IOR"].default_value = 1.31
    tc, x, y, z = m.obj_xyz()
    wx = m.math("ADD", x, centre[0])
    wy = m.math("ADD", y, centre[1])

    # Warped coordinates make the cracks wander like real ice.
    warp = m.new("ShaderNodeTexNoise")
    warp.inputs["Scale"].default_value = 0.6
    warp.inputs["Detail"].default_value = 4
    m.link(tc.outputs["Object"], warp.inputs["Vector"])
    wv = m.new("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
    m.link(warp.outputs["Color"], wv.inputs[0])
    wv.inputs[1].default_value = (0.9, 0.9, 0.0)
    m.link(tc.outputs["Object"], wv.inputs[2])
    warped = wv.outputs[0]

    def cracks(scale, keep_level, width):
        vor = m.new("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
        vor.inputs["Scale"].default_value = scale
        m.link(warped, vor.inputs["Vector"])
        keep = m.new("ShaderNodeTexNoise")
        keep.inputs["Scale"].default_value = scale * 0.9
        m.link(tc.outputs["Object"], keep.inputs["Vector"])
        mask = m.math("MULTIPLY", m.math("SUBTRACT", keep.outputs["Fac"], keep_level), 10.0, clamp=True)
        return m.math("MULTIPLY", m.band(vor.outputs["Distance"], width * 0.3, width), mask)

    # Long pressure cracks that were always there (white, sparse).
    pressure = cracks(0.25, 0.6, 0.004)

    # Skate scratches: long, thin, random strokes in many directions (stretched
    # Voronoi cell edges, so nothing repeats), patchy like a well-used pond.
    scratches = None
    for k, (rot, sx, sy) in enumerate([(0.15, 0.12, 1.6), (0.95, 0.1, 1.3), (-0.65, 0.14, 1.8), (1.7, 0.11, 1.5),
                                       (2.5, 0.13, 1.2), (-1.3, 0.1, 1.4)]):
        mp = m.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, 0, rot)
        mp.inputs["Scale"].default_value = (sx, sy, 1)
        m.link(warped, mp.inputs["Vector"])
        vor = m.new("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
        vor.inputs["Scale"].default_value = 1.0 + 0.2 * k
        m.link(mp.outputs[0], vor.inputs["Vector"])
        line = m.band(vor.outputs["Distance"], 0.0015, 0.005)
        patch = m.new("ShaderNodeTexNoise")
        patch.inputs["Scale"].default_value = 0.7 + 0.25 * k
        m.link(tc.outputs["Object"], patch.inputs["Vector"])
        line = m.math("MULTIPLY", line, m.math("MULTIPLY", m.math("SUBTRACT", patch.outputs["Fac"], 0.5), 3.0, clamp=True))
        scratches = line if scratches is None else m.math("MAXIMUM", scratches, line)

    # Light snow dust in the low spots.
    dust_n = m.new("ShaderNodeTexNoise")
    dust_n.inputs["Scale"].default_value = 1.5
    dust_n.inputs["Detail"].default_value = 8
    m.link(tc.outputs["Object"], dust_n.inputs["Vector"])
    dust = m.math("MULTIPLY", m.math("SUBTRACT", dust_n.outputs["Fac"], 0.7), 4.0, clamp=True)

    # The fresh carve over the crest.
    carve = m.math("MULTIPLY", m.band(m.math("SUBTRACT", wy, blade_y()), 0.006, 0.025),
                   m.math("LESS_THAN", wx, m.value(blade_x)))

    # Heat cracks: finer, inside a radius that grows on each shove.
    dx = m.math("SUBTRACT", wx, HOME.x)
    dy = m.math("SUBTRACT", wy, HOME.y)
    dist = m.math("SQRT", m.math("ADD", m.math("MULTIPLY", dx, dx), m.math("MULTIPLY", dy, dy)))
    reach = m.value(lambda f: sum(r * out_cubic((f - s) / 3) for s, r in zip(SHOVES, (0.7, 0.4, 0.5)))
                    + (0.5 * out_cubic((f - BREAK) / 4) if f >= BREAK else 0))
    edge = m.new("ShaderNodeMapRange", clamp=True)
    m.link(m.math("SUBTRACT", reach, dist), edge.inputs["Value"])
    edge.inputs["From Min"].default_value = 0.0
    edge.inputs["From Max"].default_value = 0.4
    heat_cracks = m.math("MULTIPLY", cracks(1.4, 0.5, 0.006), edge.outputs[0])

    white = m.math("MAXIMUM", m.math("MAXIMUM", m.math("MULTIPLY", pressure, 0.8), m.math("MULTIPLY", scratches, 0.3)),
                   m.math("MAXIMUM", m.math("MULTIPLY", dust, 0.35), carve))
    white = m.math("MAXIMUM", white, heat_cracks)
    # Frosted underside: the crest below looks soft, as if seen through ice.
    under = m.math("LESS_THAN", z, -ICE_T / 2 + 0.01)
    rough = m.math("ADD", m.math("MULTIPLY", under, 0.16), 0.01)
    m.link(m.math("ADD", rough, m.math("MULTIPLY", white, 0.55), clamp=True), b.inputs["Roughness"])
    m.link(m.math("SUBTRACT", 1.0, m.math("MULTIPLY", white, 0.9), clamp=True), b.inputs["Transmission Weight"])
    b.inputs["Emission Color"].default_value = (1.0, 0.32, 0.06, 1)
    glow = m.math("MULTIPLY", heat_cracks, m.value(lambda f: (2.0 * heat(f) if f >= SHOVES[0] else 0.0) * (1 - 0.9 * cool(f))))
    m.link(glow, b.inputs["Emission Strength"])

    jag = m.new("ShaderNodeTexNoise")
    jag.inputs["Scale"].default_value = 2.2
    jag.inputs["Detail"].default_value = 3
    m.link(warped, jag.inputs["Vector"])
    hole_r = m.value(lambda f: 0.0 if f < BREAK else 1.6 * out_cubic((f - BREAK) / 3))
    gone = m.math("LESS_THAN", m.math("ADD", dist, m.math("MULTIPLY", jag.outputs["Fac"], 1.2)), hole_r)
    clear = m.new("ShaderNodeBsdfTransparent")
    mixs = m.new("ShaderNodeMixShader")
    m.link(gone, mixs.inputs["Fac"])
    m.link(b.outputs[0], mixs.inputs[1])
    m.link(clear.outputs[0], mixs.inputs[2])
    m.link(mixs.outputs[0], m.out.inputs["Surface"])
    slab.data.materials.append(m.mat)
    bubbles()


def bubbles():
    """Trapped air in sheets and trails at different depths; denser over the crest,
    so it reads as frozen in under them. Bubbles over the hole go with the break."""
    rng = np.random.default_rng(2011)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
    proto = bpy.context.object
    me = proto.data
    m = Nodes("Bubble")
    m.bsdf.inputs["Base Color"].default_value = (0.85, 0.9, 0.95, 1)
    m.bsdf.inputs["Roughness"].default_value = 0.2
    m.bsdf.inputs["Transmission Weight"].default_value = 0.5
    m.bsdf.inputs["IOR"].default_value = 1.0
    me.materials.append(m.mat)
    bpy.data.objects.remove(proto)
    centres = [Vector((HOME.x + rng.uniform(-1.4, 1.4), HOME.y + rng.uniform(-1.0, 1.0), 0)) for _ in range(7)]
    centres += [Vector((rng.uniform(-9, 9), rng.uniform(-3, 12), 0)) for _ in range(22)]
    over_hole = []
    for c_i, c in enumerate(centres):
        depth = rng.uniform(0.04, ICE_T - 0.05)
        for j in range(int(rng.integers(12, 40))):
            p = c + Vector((rng.normal(0, 0.35), rng.normal(0, 0.25), 0))
            r = rng.uniform(0.004, 0.022) * (2.2 if rng.random() < 0.06 else 1)
            o = link_object(f"Bubble{c_i}_{j}", me)
            o.location = (p.x, p.y, -depth - rng.uniform(-0.03, 0.03))
            o.scale = (r, r * rng.uniform(0.6, 1.4), r * 0.35)
            if (p - HOME).length < 2.0:
                over_hole.append(o)
    for f in (BREAK - 1, BREAK):
        for o in over_hole:
            o.hide_render = f >= BREAK
            o.keyframe_insert("hide_render", frame=f)


def props():
    """Pond-hockey details: a net with red posts, a stick in the snowbank, pucks."""
    red = Nodes("NetRed")
    red.bsdf.inputs["Base Color"].default_value = (0.55, 0.01, 0.015, 1)
    red.bsdf.inputs["Roughness"].default_value = 0.3
    white = Nodes("NetMesh")
    white.bsdf.inputs["Base Color"].default_value = (0.8, 0.82, 0.85, 1)
    white.bsdf.inputs["Roughness"].default_value = 0.8
    net = link_object("Net", None)
    w, h, d = 1.83, 1.22, 0.9

    def pipe(name, a, b, r=0.03):
        a, b = Vector(a), Vector(b)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=r, depth=(b - a).length, location=(a + b) / 2)
        o = bpy.context.object
        o.name = name
        o.rotation_euler = (b - a).to_track_quat("Z", "Y").to_euler()
        o.data.materials.append(red.mat)
        o.parent = net
        return o
    pipe("PostL", (-w / 2, 0, 0), (-w / 2, 0, h))
    pipe("PostR", (w / 2, 0, 0), (w / 2, 0, h))
    pipe("Bar", (-w / 2, 0, h), (w / 2, 0, h))
    pipe("BaseL", (-w / 2, 0, 0), (-w / 2, d, 0))
    pipe("BaseR", (w / 2, 0, 0), (w / 2, d, 0))
    pipe("BaseB", (-w / 2, d, 0), (w / 2, d, 0))
    for name, verts in [("MeshBack", [(-w / 2, d, 0), (w / 2, d, 0), (w / 2, 0.15, h), (-w / 2, 0.15, h)]),
                        ("MeshL", [(-w / 2, 0, h), (-w / 2, 0.15, h), (-w / 2, d, 0), (-w / 2, 0, 0)]),
                        ("MeshR", [(w / 2, 0, h), (w / 2, 0.15, h), (w / 2, d, 0), (w / 2, 0, 0)])]:
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], [(0, 1, 2, 3)])
        o = link_object(name, me)
        o.parent = net
        bpy.context.view_layer.objects.active = o
        sub = o.modifiers.new("Sub", "SUBSURF")
        sub.subdivision_type = "SIMPLE"
        sub.levels = sub.render_levels = 5
        wire = o.modifiers.new("Wire", "WIREFRAME")
        wire.thickness = 0.006
        me.materials.append(white.mat)
    net.location = (HOME.x + 3.6, HOME.y + 4.2, 0)
    net.rotation_euler = (0, 0, math.radians(-25))

    wood = Nodes("Stick")
    wood.bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.022, 1)
    wood.bsdf.inputs["Roughness"].default_value = 0.4
    tape = Nodes("Tape")
    tape.bsdf.inputs["Base Color"].default_value = (0.75, 0.76, 0.78, 1)
    stick = link_object("Stick", None)
    bpy.ops.mesh.primitive_cube_add(size=1)
    shaft = bpy.context.object
    shaft.scale = (0.03, 0.022, 1.55)
    shaft.location = (0, 0, 0.78)
    shaft.data.materials.append(wood.mat)
    shaft.parent = stick
    bpy.ops.mesh.primitive_cube_add(size=1)
    blade = bpy.context.object
    blade.scale = (0.3, 0.012, 0.075)
    blade.location = (0.13, 0, 0.03)
    blade.data.materials.append(tape.mat)
    blade.parent = stick
    stick.location = (HOME.x - 4.2, HOME.y + 5.0, 0.0)
    stick.rotation_euler = (math.radians(-20), math.radians(12), math.radians(30))

    puck = Nodes("Puck")
    puck.bsdf.inputs["Base Color"].default_value = (0.01, 0.01, 0.01, 1)
    puck.bsdf.inputs["Roughness"].default_value = 0.5
    for i, (px, py) in enumerate([(-1.7, HOME.y - 0.9), (2.3, HOME.y + 1.9), (-3.0, HOME.y + 3.2)]):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.038, depth=0.025, location=(px, py, 0.0125))
        bpy.context.object.data.materials.append(puck.mat)


def crest_pose(f):
    """Crest centre and tilt (deg about x; 0 = face up, 90 = facing the camera)."""
    if f < BREAK:
        t = smooth((f - 22) / 48)
        x = HOME.x + 0.35 * math.sin(f * 0.11) * t
        y = HOME.y + 0.25 * math.sin(f * 0.07 + 1.0) * t
        tilt = 32 + 6 * math.sin(f * 0.13) * t
        z = -1.05
        for s in SHOVES:                    # each shove drives the top edge into the ice
            k = max(0.0, f - s)
            kick = math.exp(-k / 4) * smooth((f - s + 3) / 3)
            z += 0.28 * kick
            tilt += 9 * kick
        return Vector((x, y, z)), tilt
    t = (f - BREAK) / 14
    z = lerp(-0.6, 1.55, out_back(t, 1.3))
    tilt = lerp(45, 90, out_back((f - BREAK - 2) / 12, 1.6))
    y = lerp(HOME.y, HOME.y - 0.3, out_cubic(t))
    return Vector((HOME.x, y, z)), tilt


def crest_world(f, local):
    """World position of a point (x, y) on the crest face at frame f."""
    c, tilt = crest_pose(f)
    rot = Euler((math.radians(tilt), 0, 0)).to_matrix()
    return c + rot @ Vector((local[0], local[1], 0.0))


def crest(args):
    """The crest as a stack of cut-out layers (front face plus a hot glowing edge)."""
    img = bpy.data.images.load(args.crest)
    aspect = img.size[0] / max(1, img.size[1])
    w, h = (CREST, CREST / aspect) if aspect >= 1 else (CREST * aspect, CREST)
    rig = link_object("Crest", None)

    # Front face: the blue crest, desaturated and dim when cold; hot colours as it heats.
    face = Nodes("CrestFace")
    tex = face.new("ShaderNodeTexImage", image=img, interpolation="Cubic", extension="CLIP")
    hsv = face.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = 0.6
    hsv.inputs["Value"].default_value = 1.15
    face.link(tex.outputs["Color"], hsv.inputs["Color"])
    luma = face.new("ShaderNodeRGBToBW")
    face.link(tex.outputs["Color"], luma.inputs[0])
    hot = face.ramp(luma.outputs[0], [(0.0, (0.3, 0.01, 0.0, 1)), (0.35, (0.95, 0.12, 0.02, 1)),
                                      (0.75, (1.0, 0.42, 0.08, 1)), (1.0, (1.0, 0.78, 0.45, 1))])
    red = face.ramp(luma.outputs[0], [(0.0, (0.03, 0.0, 0.0, 1)), (0.3, (0.38, 0.0, 0.03, 1)),
                                      (0.7, (0.75, 0.03, 0.06, 1)), (1.0, (0.95, 0.9, 0.9, 1))])
    hval = face.value(heat)
    target = face.mix(face.value(cool), hot, red)
    colour = face.mix(face.math("MINIMUM", hval, 1.0), hsv.outputs["Color"], target)
    face.link(colour, face.bsdf.inputs["Base Color"])
    face.link(target, face.bsdf.inputs["Emission Color"])
    glow_k = face.value(lambda f: (1.6 * min(heat(f), 1.0) ** 1.5) * (1 - 0.5 * cool(f)))
    face.link(face.math("MULTIPLY", glow_k, face.math("ADD", luma.outputs[0], 0.25)),
              face.bsdf.inputs["Emission Strength"])
    face.bsdf.inputs["Roughness"].default_value = 0.35
    face.bsdf.inputs["Metallic"].default_value = 0.3
    face.link(tex.outputs["Alpha"], face.bsdf.inputs["Alpha"])

    edge = Nodes("CrestEdge")
    etex = edge.new("ShaderNodeTexImage", image=img, interpolation="Cubic", extension="CLIP")
    edge.bsdf.inputs["Base Color"].default_value = (0.02, 0.025, 0.035, 1)
    edge.bsdf.inputs["Roughness"].default_value = 0.4
    edge.bsdf.inputs["Emission Color"].default_value = (1.0, 0.25, 0.04, 1)
    edge.link(edge.value(lambda f: 3.5 * min(heat(f), 1.1) ** 1.5 * (1 - 0.5 * cool(f))), edge.bsdf.inputs["Emission Strength"])
    edge.link(etex.outputs["Alpha"], edge.bsdf.inputs["Alpha"])

    for i in range(12):
        bpy.ops.mesh.primitive_plane_add(size=1)
        p = bpy.context.object
        p.name = f"CrestLayer{i}"
        p.scale = (w, h, 1)
        p.location = (0, 0, -0.012 * i)                 # layers stack behind the face (local -z)
        p.data.materials.append(face.mat if i == 0 else edge.mat)
        p.parent = rig


    def k(f):
        c, tilt = crest_pose(f)
        # Local +z (the face) points up at tilt 0 and at the camera (-y) at tilt 90.
        rig.location = c
        rig.rotation_euler = (math.radians(tilt), 0, 0)
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)
    key_frames(k)
    return w, h


def shards():
    """Ice shards thrown out of the hole on the break."""
    rng = np.random.default_rng(70)
    m = Nodes("Shard")
    m.bsdf.inputs["Base Color"].default_value = (0.9, 0.95, 1.0, 1)
    m.bsdf.inputs["Transmission Weight"].default_value = 0.9
    m.bsdf.inputs["Roughness"].default_value = 0.08
    m.bsdf.inputs["IOR"].default_value = 1.31
    pieces = []
    for i in range(70):
        a, r = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random()) * 1.6
        s = rng.uniform(0.05, 0.22)
        v = [(0, 0, 0), (s, rng.uniform(-0.3, 0.3) * s, 0), (rng.uniform(0, 1) * s, s * rng.uniform(0.6, 1.2), 0),
             (0, 0, -0.08), (s, 0, -0.08), (s * 0.5, s, -0.08)]
        faces = [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
        me = bpy.data.meshes.new(f"Shard{i}")
        me.from_pydata(v, [], faces)
        me.materials.append(m.mat)
        o = link_object(me.name, me)
        p0 = Vector((HOME.x + r * math.cos(a), HOME.y + r * math.sin(a), 0.0))
        vel = Vector((math.cos(a) * rng.uniform(0.8, 3.0), math.sin(a) * rng.uniform(0.8, 3.0) - 0.6,
                      rng.uniform(2.0, 5.5)))
        spin = Vector(rng.uniform(-12, 12, 3))
        pieces.append((o, p0, vel, spin))

    def k(f):
        for o, p0, vel, spin in pieces:
            t = max(0.0, (f - BREAK) / FPS)
            p = p0 + vel * t + Vector((0, 0, -4.9 * t * t))
            alive = f >= BREAK and p.z > -0.05
            o.location = p if f >= BREAK else p0
            o.rotation_euler = spin * t
            o.scale = (1, 1, 1) if alive else (0, 0, 0)
            for path in ("location", "rotation_euler", "scale"):
                o.keyframe_insert(path, frame=f)
    key_frames(k)


def blade():
    """A skate blade on a black holder that crosses the ice over the crest (0-16)."""
    steel = Nodes("Steel")
    steel.bsdf.inputs["Base Color"].default_value = (0.85, 0.87, 0.9, 1)
    steel.bsdf.inputs["Metallic"].default_value = 1.0
    steel.bsdf.inputs["Roughness"].default_value = 0.12
    boot = Nodes("Holder")
    boot.bsdf.inputs["Base Color"].default_value = (0.012, 0.012, 0.014, 1)
    boot.bsdf.inputs["Roughness"].default_value = 0.35
    rig = link_object("Skate", None)
    n = 48
    xs = np.linspace(-0.55, 0.55, n)
    bottom = 0.08 * (xs / 0.55) ** 2
    verts = []
    for side in (-0.007, 0.007):
        verts += [(x, side, b) for x, b in zip(xs, bottom)]
        verts += [(x, side, 0.13) for x in xs]
    faces = []
    for s in (0, 2 * n):
        for i in range(n - 1):
            faces.append((s + i, s + i + 1, s + n + i + 1, s + n + i))
    for i in range(n - 1):
        faces.append((i, 2 * n + i, 2 * n + i + 1, i + 1))
        faces.append((n + i, n + i + 1, 3 * n + i + 1, 3 * n + i))
    me = bpy.data.meshes.new("Blade")
    me.from_pydata(verts, [], faces)
    me.materials.append(steel.mat)
    b = link_object("Blade", me)
    b.parent = rig
    bpy.ops.mesh.primitive_cube_add(size=1)
    holder = bpy.context.object
    holder.scale = (1.0, 0.07, 0.2)
    holder.location = (0, 0, 0.23)
    holder.data.materials.append(boot.mat)
    holder.parent = rig

    def k(f):
        rig.location = (blade_x(f), blade_y(), -0.004)
        rig.rotation_euler = (math.radians(-12), 0, 0)
        for o in (rig, b, holder):
            o.hide_render = f > 22
            o.keyframe_insert("hide_render", frame=f)
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)
    key_frames(k)


def flake_mesh():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    proto = bpy.context.object
    me = proto.data
    m = Nodes("Snowflake")
    m.bsdf.inputs["Base Color"].default_value = (0.9, 0.94, 1.0, 1)
    m.bsdf.inputs["Roughness"].default_value = 0.6
    m.bsdf.inputs["Emission Color"].default_value = (0.8, 0.88, 1.0, 1)
    m.bsdf.inputs["Emission Strength"].default_value = 0.04
    me.materials.append(m.mat)
    bpy.data.objects.remove(proto)
    return me


def snow(crest_size):
    """Flurries everywhere, blade spray at the start, and flakes that fizzle on the crest."""
    rng = np.random.default_rng(2011)
    me = flake_mesh()
    flakes = []
    for i in range(320):
        near = i < 200                      # a denser, larger layer close to the camera
        p0 = Vector((rng.uniform(-5, 5) if near else rng.uniform(-9, 9),
                     rng.uniform(-1.0, 4.5) if near else rng.uniform(-1.0, 12), rng.uniform(-0.5, 6)))
        o = link_object(f"Flake{i}", me)
        flakes.append((o, p0, rng.uniform(0.5, 1.1), rng.uniform(0, 2 * math.pi),
                       rng.uniform(0.015, 0.03) if near else rng.uniform(0.008, 0.02)))

    def k(f):
        t = f / FPS
        for o, p0, speed, phase, r in flakes:
            z = (p0.z - speed * t + 0.5) % 7.5 - 0.5
            o.location = (p0.x + 0.25 * math.sin(t * 1.3 + phase) + 0.4 * t, p0.y + 0.15 * math.cos(t + phase), z)
            o.scale = (r, r, r * 0.4) if z > 0 else (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
    key_frames(k)

    # Blade spray (0-16): chips on ballistic paths from the contact point.
    chips = []
    for i in range(160):
        born = rng.uniform(1, 15)
        v = Vector((rng.uniform(-0.3, 1.4), rng.choice([-1, 1]) * rng.uniform(0.5, 2.0), rng.uniform(0.6, 2.4)))
        o = link_object(f"Chip{i}", me)
        chips.append((o, born, Vector((blade_x(born), blade_y(), 0.01)), v, rng.uniform(0.004, 0.012)))

    def c(f):
        for o, born, p0, v, r in chips:
            t = (f - born) / FPS
            p = p0 + v * t + Vector((0, 0, -4.9 * t * t)) if t > 0 else p0
            o.location = p
            o.scale = (r, r, r) if t > 0 and p.z > 0 else (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
    key_frames(c)

    # Flakes that land on the hot crest: each one fizzles into a small flash and steam.
    w, h = crest_size
    for i, hit in enumerate(HITS):
        local = (rng.uniform(-0.3, 0.3) * w, rng.uniform(-0.25, 0.3) * h)
        target = crest_world(hit, local) + Vector((0, -0.02, 0))
        start = target + Vector((-0.5, -0.6, 1.4))
        o = link_object(f"HitFlake{i}", me)
        # Steam: a soft volume (no surface) that flashes warm, then thins out.
        puff_mat = Nodes(f"Puff{i}")
        puff_mat.nt.nodes.remove(puff_mat.bsdf)
        vol = puff_mat.new("ShaderNodeVolumePrincipled")
        vol.inputs["Color"].default_value = (0.95, 0.97, 1.0, 1)
        vol.inputs["Emission Color"].default_value = (1.0, 0.6, 0.35, 1)
        tcp = puff_mat.new("ShaderNodeTexCoord")
        wisp = puff_mat.new("ShaderNodeTexNoise")
        wisp.inputs["Scale"].default_value = 2.5
        puff_mat.link(tcp.outputs["Object"], wisp.inputs["Vector"])
        dens = puff_mat.value(lambda f, hit=hit: 0.0 if f < hit else 6.0 * math.exp(-(f - hit) / 7))
        puff_mat.link(puff_mat.math("MULTIPLY", dens, puff_mat.math("SUBTRACT", wisp.outputs["Fac"], 0.35, clamp=True)),
                      vol.inputs["Density"])
        puff_mat.link(puff_mat.value(lambda f, hit=hit: 4.0 * math.exp(-max(0, f - hit) / 2) if f >= hit else 0.0),
                      vol.inputs["Emission Strength"])
        puff_mat.link(vol.outputs[0], puff_mat.out.inputs["Volume"])
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1.0)
        puff = bpy.context.object
        puff.name = f"Puff{i}"
        puff.data.materials.append(puff_mat.mat)
        bpy.ops.object.shade_smooth()

        def k2(f, o=o, hit=hit, start=start, target=target, puff=puff):
            t = clamp01((f - (hit - 28)) / 28)
            o.location = start.lerp(target, t) + Vector((0.08 * math.sin(f * 0.5), 0, 0)) * (1 - t)
            o.scale = (0.02, 0.02, 0.008) if hit - 28 <= f < hit else (0, 0, 0)
            puff.location = target + Vector((0, -0.05, 0.1 * clamp01((f - hit) / 10)))
            s = 0.06 + 0.3 * out_cubic((f - hit) / 12) if f >= hit else 0.0
            puff.scale = (s, s * 0.7, s)
            for obj, path in ((o, "location"), (o, "scale"), (puff, "location"), (puff, "scale")):
                obj.keyframe_insert(path, frame=f)
        key_frames(k2)


def lights():
    def area(name, loc, aim, colour, energy, size):
        data = bpy.data.lights.new(name, "AREA")
        data.shape = "RECTANGLE"
        data.size, data.size_y = size
        data.color = colour
        data.energy = energy
        obj = link_object(name, data)
        obj.location = loc
        obj.rotation_euler = (Vector(aim) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        obj.visible_camera = False
        return obj

    def flood(name, loc, aim, energy):
        data = bpy.data.lights.new(name, "SPOT")
        data.color = (1.0, 0.92, 0.8)
        data.energy = energy
        data.spot_size = math.radians(80)
        data.spot_blend = 0.4
        data.shadow_soft_size = 0.25
        obj = link_object(name, data)
        obj.location = loc
        obj.rotation_euler = (Vector(aim) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        obj.visible_glossy = False         # its mirror image read as a blob on the ice
        return obj

    # Two pond-rink floodlights on poles: hard, warm, raking the ice.
    flood("FloodL", (-9, HOME.y + 7, 8), (0, HOME.y, 0), 9000)
    flood("FloodR", (8, HOME.y - 6, 7), (0, HOME.y + 1, 0), 4000)
    # A raking strip low over the ice for the blade pass (0-20).
    rake = area("Rake", (-8, 0.5, 0.8), (0, 1.4, 0), (0.9, 0.94, 1.0), 0, (5, 0.4))
    rake.visible_glossy = False

    def k(f):
        rake.data.energy = 2000 * (1 - smooth((f - 14) / 8))
        rake.data.keyframe_insert("energy", frame=f)
    key_frames(k)
    # A rim from behind the crest once it stands up.
    rim = area("Rim", (0, 6, 4.5), (0, 1.3, 1.5), (0.85, 0.9, 1.0), 0, (2.5, 0.6))
    rim.visible_glossy = False

    def r(f):
        rim.data.energy = 1200 * smooth((f - BREAK - 4) / 10)
        rim.data.keyframe_insert("energy", frame=f)
    key_frames(r)


def camera(scene, args):
    data = bpy.data.cameras.new("Camera")
    data.sensor_fit = "HORIZONTAL"
    data.clip_start, data.clip_end = 0.01, 60000
    data.dof.use_dof = True
    data.dof.aperture_fstop = 2.8
    data.lens = 34 if args.vertical else 24
    cam = link_object("Camera", data)
    cam.rotation_mode = "QUATERNION"
    scene.camera = cam

    def pose(f):
        # 0-12 high, looking down through the ice at the crest as the blade passes;
        # 8-38 crane down and back while it wakes; 38-70 slow push; 70-92 drop to
        # eye level as it stands up; then a slow push; 120-134 fast push.
        c, _ = crest_pose(min(f, BREAK - 1))
        hi_loc, hi_aim = Vector((0.4, HOME.y - 1.6, 4.4)), Vector((0.0, HOME.y + 0.2, -0.6))
        mid_loc, mid_aim = Vector((0.0, HOME.y - 5.0, 2.7)), Vector((0.0, HOME.y + 0.6, -0.2))
        if f < 38:
            e = in_out((f - 8) / 30)
            loc, aim = hi_loc.lerp(mid_loc, e), hi_aim.lerp(mid_aim, e)
            aim.x += 0.3 * c.x * e
        elif f < BREAK:
            e = (f - 38) / (BREAK - 38)
            loc = mid_loc.lerp(Vector((0.0, HOME.y - 4.3, 2.4)), e)
            aim = mid_aim + Vector((0.4 * c.x, 0, 0))
        elif f < 92:
            e = in_out((f - BREAK) / 22)
            loc = Vector((0.0, HOME.y - 4.3, 2.4)).lerp(Vector((0.0, HOME.y - 5.6, 1.25)), e)
            aim = mid_aim.lerp(Vector((0.0, HOME.y, 1.75)), e)
        elif f < 120:
            e = (f - 92) / 28
            loc = Vector((0.0, lerp(HOME.y - 5.6, HOME.y - 5.0, e), 1.25))
            aim = Vector((0.0, HOME.y, 1.75))
        else:
            e = in_cubic((f - 120) / 14)
            loc = Vector((0.0, lerp(HOME.y - 5.0, HOME.y - 2.2, e), lerp(1.25, 1.55, e)))
            aim = Vector((0.0, HOME.y, 1.75))
        focus = Vector((blade_x(min(f, 16)), blade_y(), 0)) if f < 10 else crest_pose(f)[0]
        return loc, aim, focus

    def k(f):
        loc, aim, focus = pose(f)
        cam.location = loc
        cam.rotation_quaternion = (aim - loc).to_track_quat("-Z", "Y")
        data.dof.focus_distance = (focus - loc).length
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        data.dof.keyframe_insert("focus_distance", frame=f)
    key_frames(k)
def render_settings(scene, args):
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if not args.cpu:
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "METAL"
        cy.device = "GPU"
    cy.samples = args.samples
    cy.use_denoising = True
    cy.max_bounces = 12
    cy.transmission_bounces = 12
    cy.glossy_bounces = 6
    cy.transparent_max_bounces = 24
    cy.caustics_reflective = cy.caustics_refractive = False
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.35
    scene.render.resolution_x, scene.render.resolution_y = (1080, 1920) if args.vertical else (1920, 1080)
    scene.render.resolution_percentage = args.scale
    transforms = scene.view_settings.bl_rna.properties["view_transform"].enum_items.keys()
    # Khronos PBR Neutral keeps the crest colours; AgX washes them out.
    scene.view_settings.view_transform = "Khronos PBR Neutral" if "Khronos PBR Neutral" in transforms \
        else "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = os.path.join(os.path.abspath(args.out), "frame_")
    scene.frame_start, scene.frame_end = args.start, args.end
    compositor(scene)


def compositor(scene):
    """Bloom on the hot crest, the glowing cracks, and the steam flashes."""
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
    for name, value in [("Type", "Bloom"), ("Threshold", 1.0), ("Strength", 0.6), ("Size", 0.7)]:
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
    mountains(args)
    banks()
    ice()
    props()
    size = crest(args)
    shards()
    blade()
    snow(size)
    lights()
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
