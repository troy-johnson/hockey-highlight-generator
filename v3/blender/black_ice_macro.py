"""Black Ice opening, macro version (spec 003, hhg-3r5.86). Built scene by scene.

    Blender -b --factory-startup --python-exit-code 1 --python v3/blender/black_ice_macro.py -- \
        --crest-svg ".../Logos/icepak_crest_3d.svg" --out frames/ [--vertical] [--scene 1]

Frame numbers below are story frames (what the animation is keyed on). The output is
retimed from them (RETIME): Scene 1 shows story 12-34 as output 0-21.

Scene 1 "Wake-up" (story frames 0-34), agreed storyboard:
- 0-18  dormant: the rusty blue crest lies about 15-20 cm under black pond ice; the
        camera is locked over it and creeps in; snow dust drifts across the surface.
- 18-23 a puck shoots across, over the crest.
- 22-30 two skater shadows sweep over the crest (the skaters stay out of frame);
        a taped stick blade cuts through the top of the frame; ice flecks spray in.
- 28    a cold blue pulse runs through the crest.
- 30-34 the flecks settle; the camera carries on into Scene 2 (no cut).
Falling snow throughout. Units are metres; the ice top is z = 0.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crest3d  # noqa: E402

FPS = 30
S1_END = 34
S2_A, S2_B, S2_END = 34, 56, 78   # Scene 2 'Heat': angle A from above, angle B low across the ice
S3_A, S3_END = 78, 100            # Scene 3 'Breakthrough': the first cut, low across the ice
S4_A, S4_END = 100, 164          # Scene 4 'Hero': the same shot, no cut; it cools to the red kit
KEY_END = S4_END
FIZZLES = (104, 112, 119, 126)   # snowflakes land on the crest and fizzle
S4_AIM_Z = 0.335                 # camera aim height at the end of Scene 4 (keeps the bottom quarter clear)
# The scenes above are keyed in story frames. The output is retimed (user, after the
# v14 review): less wake-up and heat, a slower, heavier breakthrough, a long hero hold.
# Knots (output frame, story frame), linear between them; fractional story frames are
# rendered as subframes.
RETIME = ((0, 12), (22, 34),                       # Scene 1: from just before the puck
          (28, 46), (55, 46 + 27 * 32 / 28),       # Scene 2: the quiet start quickly, then about 1.1x
          # The cut skips story 77-80, the dark beat under the ice: Scene 3 opens as the
          # crest starts up (v7 review: the energy dropped at the cut).
          (56, 80),
          (83, 102),                               # Scene 3: one steady 0.8x slow motion (a speed
                                                   # ramp mid-flight made gravity look wrong, v8)
          (147, 164))                              # Scene 4: about real time, a long hold
RETIME_CUT = 55                                    # the knot pair (55, 56) is the cut, not a ramp
OUT_END = {1: 21, 2: 55, 3: 81, 4: 147}            # last output frame of each scene (4.93 s)
HOLD_FROM = 29                   # story frames into Scene 4 when the crest and camera lock (output 111)
HERO_KEY = 40.0                  # W, the crest-only key light that comes up as its glow dies
RISE_TIP = 0.10                  # the crest's lower tip above the ice at the end of Scene 3 (m)
ICE_SIZE = 40.0                  # pond ice extent: the low Scene 3 camera sees to the horizon
SHOVES = (44, 60, 72)
ICE_T = 0.12                     # pond ice thickness
CREST_W = 0.6                    # crest width
CREST_BASE = -0.19               # crest back face; its top sheet is ~4 cm under the ice, in the water
FLOOR_Z = -2.5                   # pond floor: deep, so it falls away into blue-black
PUCK_R, PUCK_H = 0.038, 0.0254
CAM_ELEV = 60                    # camera angle above the ice, degrees
# Reflections: a real night HDRI (Poly Haven, CC0) seen only by glossy rays.
HDRI = os.path.expanduser("~/Library/Caches/hockey-highlight-generator/polyhaven/kloppenheim_04_4k.hdr")
HDRI_ROT, HDRI_STRENGTH = 20.0, 1.8
# Scenes 1-2 ice colour, matched to the cool blue-black ice of Scenes 3-4.
S1_FLOOD = (0.78, 0.88, 1.0)                       # flood colour before the cut
FLOOR_DARK, FLOOR_LIGHT = (0.016, 0.019, 0.022, 1), (0.07, 0.075, 0.08, 1)
MIRROR_TINT = (0.6, 0.8, 1.25, 1)                  # cools the HDRI the ice mirrors
SNOW_K = 0.55                                      # snow patch whiteness (the flood burns it out at 1)
SNOW_MELT = (46, 82)                               # frames: snow over the crest melts away
SNOW_CLEAR = 0.06                                  # fewer snow patches over the crest
DRIFT_AT, DRIFT_SIZE = (0.05, 0.45, 0.0), (0.5, 0.13, 1.0)   # the drift whose shadow crosses the crest
SNOW_EDGE = 0.62                                   # snow patch threshold (higher = less snow)
CRACK_K = 0.35                                     # brightness of the fracture sheets in the ice
CRACK_LAYERS = (                                   # depth (m), cells per metre, brightness, seed
    (0.006, 1.1, 1.0, 0.0), (0.025, 2.2, 0.75, 1.7), (0.05, 3.5, 0.55, 3.1), (0.09, 5.5, 0.4, 4.6))


# ---------------------------------------------------------------- helpers

def clamp01(t):
    return max(0.0, min(1.0, t))


def smooth(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


WIND_DROP = 12   # frames: the wind dies as the hero shot holds (Scene 4)


def wind(f):
    """Wind strength, 1 until Scene 4, then dropping to 0."""
    return 1.0 - smooth((f - S4_A) / WIND_DROP)


def wind_t(f, f0=0):
    """Seconds of wind-driven travel from frame f0 to f (stops as the wind drops):
    specks blown across the ice behind the crest read as light moving on water."""
    if f <= S4_A:
        return (f - f0) / FPS
    k = min(f, S4_A + WIND_DROP) - S4_A
    return (S4_A - f0 + sum(wind(S4_A + i + 0.5) for i in range(k))) / FPS


def lerp(a, b, t):
    return a + (b - a) * t


def link(name, data):
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def keys(frames, fn):
    for f in frames:
        fn(f)


def heat(f):
    """0 = dormant, 1 = orange-hot (end of Scene 2). Each shove adds a short flare."""
    h = smooth((f - S2_A) / (S2_END - S2_A)) ** 1.3
    for i, s in enumerate(SHOVES):
        if f >= s:
            h += (0.08 + 0.05 * i) * math.exp(-(f - s) / 4.0)
    if f > S2_END:                     # Scene 3: a last flare as it breaks out, then it cools in the air
        h += 0.12 * math.exp(-(f - S2_END) / 3.0) - 0.15 * smooth((f - 86) / 40)
    return min(h, 1.15)


def crest_motion(f):
    """(dx, dy, dz, tilt_x_deg, tilt_y_deg, dyaw_deg): dormant, then a ramp of shoves."""
    dx = dy = dz = tx = ty = yaw = 0.0
    if f >= 38:                                         # a tiny twitch
        k = math.exp(-(f - 38) / 2.5)
        dx += 0.005 * k * math.sin((f - 38) * 2.2)
    if f >= 44:                                         # first shove: a sideways jolt
        k = 1 - math.exp(-(f - 44) / 1.5)
        dx += 0.02 * k
        ty += 2.0 * math.exp(-(f - 44) / 5) * k
    if f >= 46:                                         # uneasy drift, searching
        t = (f - 46) / FPS
        amp = smooth((f - 46) / 10)
        dx += 0.008 * amp * math.sin(t * 5.0)
        dy += 0.006 * amp * math.sin(t * 3.7 + 1.0)
        yaw += 1.5 * amp * math.sin(t * 4.2)
    if f >= 60:                                         # second shove: tilts up and rises
        k = 1 - math.exp(-(f - 60) / 2.0)
        tx += 1.5 * k                                   # mostly a flat push up, not a tilt
        dz += 0.014 * k
    if f >= 72:                                         # third: drives its peak into the ice
        k = 1 - math.exp(-(f - 72) / 1.2)
        tx += 1.0 * k
        dz += 0.018 * k                                 # the top sheet meets the underside of the ice
        dx += 0.004 * math.exp(-(f - 72) / 2) * math.sin((f - 72) * 3.0)
    return dx, dy, dz, tx, ty, yaw


class Mat:
    """Small node-building helper for Principled materials."""

    def __init__(self, name):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.nt = self.mat.node_tree
        self.bsdf = self.nt.nodes["Principled BSDF"]
        self.out = self.nt.nodes["Material Output"]
        self.tc = self.nt.nodes.new("ShaderNodeTexCoord")

    def new(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=0.0, clamp=False):
        n = self.new("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, x in enumerate((a, b)):
            if hasattr(x, "links"):
                self.link(x, n.inputs[i])
            else:
                n.inputs[i].default_value = x
        return n.outputs[0]

    def noise(self, scale, detail=4.0, rough=0.5, vector=None):
        n = self.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        self.link(vector or self.tc.outputs["Object"], n.inputs["Vector"])
        return n.outputs["Fac"]

    def ramp(self, fac, lo, hi):
        """Map fac from [lo, hi] to [0, 1], clamped."""
        n = self.new("ShaderNodeMapRange", clamp=True)
        self.link(fac, n.inputs["Value"])
        n.inputs["From Min"].default_value = lo
        n.inputs["From Max"].default_value = hi
        return n.outputs[0]

    def value(self, frames, curve):
        n = self.new("ShaderNodeValue")
        for f in frames:
            n.outputs[0].default_value = curve(f)
            n.outputs[0].keyframe_insert("default_value", frame=f)
        return n.outputs[0]


# ---------------------------------------------------------------- the pond

def far_soft(m):
    """0 near the camera, 1 far off, from the Scene 3 cut only. From the low camera,
    fine snow detail far off goes below a pixel and crawls like light on water."""
    cam = m.new("ShaderNodeCameraData")
    after = m.value(range(0, KEY_END + 1), lambda f: 1.0 if f >= S3_A else 0.0)
    return m.math("MULTIPLY", m.ramp(cam.outputs["View Distance"], 1.5, 6.0), after)


def snow_cover(m, co, heat_melt=True):
    """Hard-edged snow patches drawn out by the wind (0-1), at pond position co (metres,
    the ice top at z = ICE_T / 2). Thinner over the crest, where people skate. Shared by
    the ice and by the shadow layer, so the shadows match the snow."""
    rad = m.new("ShaderNodeVectorMath", operation="LENGTH")
    flat = m.new("ShaderNodeVectorMath", operation="MULTIPLY")
    m.link(co, flat.inputs[0])
    flat.inputs[1].default_value = (1.0, 1.4, 0.0)
    m.link(flat.outputs[0], rad.inputs[0])
    worn = m.ramp(m.math("ADD", rad.outputs["Value"], m.math("MULTIPLY", m.noise(1.5, 3.0, 0.5, vector=co), 0.25)),
                  0.45, 0.85)
    wmap = m.new("ShaderNodeMapping")
    wmap.inputs["Rotation"].default_value = (0, 0, 0.35)          # the wind, as the drifts
    wmap.inputs["Scale"].default_value = (0.7, 1.8, 1.0)
    m.link(co, wmap.inputs["Vector"])
    soft = far_soft(m)
    edge_jag = m.math("MULTIPLY", m.noise(45.0, 6.0, 0.7, vector=co),
                      m.math("MULTIPLY", m.math("SUBTRACT", 1.0, m.math("MULTIPLY", soft, 0.8)), 0.09))   # flaky, broken edges
    # From the low camera the small snow islands drew out into bright dashes that
    # sparkled like water: from the cut, fewer fine octaves (same large patches).
    after = m.value(range(0, KEY_END + 1), lambda f: 1.0 if f >= S3_A else 0.0)
    fine = m.noise(1.3, 8.0, 0.62, vector=wmap.outputs[0])
    coarse = m.noise(1.3, 2.5, 0.62, vector=wmap.outputs[0])
    lump = m.math("ADD", fine, m.math("MULTIPLY", m.math("SUBTRACT", coarse, fine), after))
    lump = m.math("ADD", lump, edge_jag)
    # From the low camera, bare far ice is only soft grey bands and reads as a lake:
    # beyond the cleared rink the pond lies under wind-packed snow.
    lump = m.math("ADD", lump, m.math("MULTIPLY", soft, 0.3))
    lump = m.math("SUBTRACT", lump, m.math("MULTIPLY", m.math("SUBTRACT", 1.0, worn), SNOW_CLEAR))
    if heat_melt:
        # Over the heating crest the snow melts from its edges inwards (Scene 2), so it
        # does not hide the glow; the drift beyond the crest stays.
        nearc = m.ramp(rad.outputs["Value"], 0.6, 0.35)
        melt = m.value(range(0, KEY_END + 1), lambda f: clamp01((f - SNOW_MELT[0]) / (SNOW_MELT[1] - SNOW_MELT[0])) ** 1.5)   # slow, then faster
        lump = m.math("SUBTRACT", lump, m.math("MULTIPLY", m.math("MULTIPLY", nearc, melt), 0.09))
    width = m.math("ADD", 0.012, m.math("MULTIPLY", soft, 0.025))                 # softer edges far off
    patch = m.math("DIVIDE", m.math("SUBTRACT", lump, SNOW_EDGE), width, clamp=True)
    # A drift just beyond the crest: the low flood throws its shadow across the crest
    # under the ice (the shadow lands ~0.5 m nearer the camera than the snow).
    dm = m.new("ShaderNodeMapping", vector_type="TEXTURE")
    dm.inputs["Location"].default_value = DRIFT_AT
    dm.inputs["Rotation"].default_value = (0, 0, 0.35)
    dm.inputs["Scale"].default_value = DRIFT_SIZE
    m.link(co, dm.inputs["Vector"])
    flat2 = m.new("ShaderNodeVectorMath", operation="MULTIPLY")
    m.link(dm.outputs[0], flat2.inputs[0])
    flat2.inputs[1].default_value = (1.0, 1.0, 0.0)
    rr = m.new("ShaderNodeVectorMath", operation="LENGTH")
    m.link(flat2.outputs[0], rr.inputs[0])
    wob = m.math("MULTIPLY", m.math("SUBTRACT", m.noise(3.0, 8.0, 0.65, vector=co), 0.5), 0.9)
    drift = m.ramp(m.math("ADD", m.math("ADD", rr.outputs["Value"], wob), m.math("MULTIPLY", edge_jag, 4.0)), 1.0, 0.97)
    patch = m.math("MAXIMUM", patch, drift)
    return m.math("MULTIPLY", patch, m.math("ADD", 0.8, m.math("MULTIPLY", m.noise(12.0, 4.0, 0.6, vector=co), 0.2)))


def snow_shadows():
    """The snow on the ice shades the crest under it (the clear ice itself casts no
    shadow): a shadow-only layer just under the ice top, opaque where the snow lies.
    Scenes 1-2 only: in the low shot the hole is there and the crest is out."""
    bpy.ops.mesh.primitive_plane_add(size=ICE_SIZE, location=(0, 0, -0.001))
    o = bpy.context.object
    o.name = "SnowShadow"
    for v in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter"):
        setattr(o, v, False)
    m = Mat("SnowShadow")
    geo = m.new("ShaderNodeNewGeometry")
    co = m.new("ShaderNodeVectorMath", operation="ADD")
    m.link(geo.outputs["Position"], co.inputs[0])
    co.inputs[1].default_value = (0, 0, ICE_T / 2)                 # the ice top, as the ice sees it
    clear = m.new("ShaderNodeBsdfTransparent")
    mix = m.new("ShaderNodeMixShader")
    m.link(snow_cover(m, co.outputs[0]), mix.inputs["Fac"])
    m.link(clear.outputs[0], mix.inputs[1])
    m.link(m.bsdf.outputs[0], mix.inputs[2])
    m.link(mix.outputs[0], m.out.inputs["Surface"])
    o.data.materials.append(m.mat)
    for f in (0, S3_A - 1, S3_A):
        o.visible_shadow = f < S3_A
        o.keyframe_insert("visible_shadow", frame=f)


def ice():
    """Clear black pond ice: glossy top with fine skate scratches and snow dust,
    a slightly frosted underside, and trapped bubbles."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -ICE_T / 2))
    slab = bpy.context.object
    slab.name = "Ice"
    slab.scale = (ICE_SIZE, ICE_SIZE, ICE_T)
    m = Mat("Ice")
    b = m.bsdf
    b.inputs["Base Color"].default_value = (0.92, 0.96, 1.0, 1)
    b.inputs["IOR"].default_value = 1.31
    b.inputs["Transmission Weight"].default_value = 1.0
    # Object coordinates of the slab: scale them back up to metres.
    metres = m.new("ShaderNodeVectorMath", operation="MULTIPLY")
    m.link(m.tc.outputs["Object"], metres.inputs[0])
    metres.inputs[1].default_value = (ICE_SIZE, ICE_SIZE, ICE_T)
    co = metres.outputs[0]
    sep = m.new("ShaderNodeSeparateXYZ")
    m.link(co, sep.inputs[0])
    # The top face only (it was the whole upper half, which put snow dust on the hole
    # walls; the top-down Scenes 1-2 only ever see the top face, so they are unchanged).
    top = m.math("GREATER_THAN", sep.outputs["Z"], ICE_T / 2 - 0.002)
    # Skate marks: long, gentle arcs (pieces of very large circles) crossing in many
    # directions, thin and faint, patchy like a well-used pond.
    rng = np.random.default_rng(7)
    scratches = None
    for k in range(90):
        centre = Vector((rng.uniform(-6, 6), rng.uniform(-6, 6), 0))
        radius = rng.uniform(1.5, 7.0)
        off = m.new("ShaderNodeVectorMath", operation="SUBTRACT")
        m.link(co, off.inputs[0])
        off.inputs[1].default_value = centre
        ring = m.new("ShaderNodeVectorMath", operation="LENGTH")
        m.link(off.outputs[0], ring.inputs[0])
        d = m.math("ABSOLUTE", m.math("SUBTRACT", ring.outputs["Value"], radius))
        wob = m.math("MULTIPLY", m.noise(3.0, 2.0, 0.5, vector=co), 0.004)
        line = m.ramp(m.math("ADD", d, wob), 0.0022, 0.0004)
        patch = m.ramp(m.noise(0.8 + 0.15 * k, vector=co), 0.45, 0.58)
        line = m.math("MULTIPLY", line, patch)
        scratches = line if scratches is None else m.math("MAXIMUM", scratches, line)
    # Where people skate: scratches build up away from the clear patch
    # over the crest (reference: pond hockey at Lake Nokomis / Stirling Mill Pond).
    rad = m.new("ShaderNodeVectorMath", operation="LENGTH")
    flat = m.new("ShaderNodeVectorMath", operation="MULTIPLY")
    m.link(co, flat.inputs[0])
    flat.inputs[1].default_value = (1.0, 1.4, 0.0)
    m.link(flat.outputs[0], rad.inputs[0])
    worn = m.ramp(m.math("ADD", rad.outputs["Value"], m.math("MULTIPLY", m.noise(1.5, 3.0, 0.5, vector=co), 0.25)),
                  0.45, 0.85)
    scr = m.math("MULTIPLY", scratches, m.math("ADD", 0.25, m.math("MULTIPLY", worn, 0.75)))
    # The look (references: Lake Baikal black ice; a winter pond): clear blue-black ice,
    # no haze. Snow lies on it in bright, hard-edged patches drawn out by the wind (the
    # skater shadows fall on them), and white fracture sheets run down inside the ice.
    snow_patch = m.math("MULTIPLY", snow_cover(m, co), SNOW_K)
    # Wet and grey as it melts over the crest.
    rc = m.new("ShaderNodeVectorMath", operation="LENGTH")
    fc = m.new("ShaderNodeVectorMath", operation="MULTIPLY")
    m.link(co, fc.inputs[0])
    fc.inputs[1].default_value = (1.0, 1.4, 0.0)
    m.link(fc.outputs[0], rc.inputs[0])
    over_crest = m.ramp(rc.outputs["Value"], 0.6, 0.35)
    wet = m.value(range(0, KEY_END + 1), lambda f: smooth((f - SNOW_MELT[0] + 2) / 16))
    snow_patch = m.math("MULTIPLY", snow_patch, m.math("SUBTRACT", 1.0, m.math("MULTIPLY", m.math("MULTIPLY", over_crest, wet), 0.55)))
    # Fracture sheets at several depths, seen through the top: each layer is looked up
    # where the view ray reaches that depth (refraction roughly folded in), so the layers
    # slide over each other as the camera moves and the cracks read as going down.
    geo = m.new("ShaderNodeNewGeometry")
    sepi = m.new("ShaderNodeSeparateXYZ")
    m.link(geo.outputs["Incoming"], sepi.inputs[0])
    iz = m.math("MAXIMUM", sepi.outputs["Z"], 0.02)
    # Snell (n 1.31): the sideways reach at depth d is d * tan(t); per unit of the view
    # vector's xy that is d / (1.31 cos t). From above it is ~0.76 d; even at grazing it
    # stays under 1.2 d (a 1/cos(i) guess swam like water in the low shot).
    cos_t = m.math("SQRT", m.math("SUBTRACT", 1.0, m.math("DIVIDE", m.math("SUBTRACT", 1.0, m.math("MULTIPLY", iz, iz)), 1.716)))
    xy_only = m.new("ShaderNodeVectorMath", operation="MULTIPLY")
    m.link(geo.outputs["Incoming"], xy_only.inputs[0])
    xy_only.inputs[1].default_value = (1.0, 1.0, 0.0)
    # Far off the thin lines go below a pixel and shimmer: fade them out with distance.
    cam = m.new("ShaderNodeCameraData")
    near_cam = m.ramp(cam.outputs["View Distance"], 6.0, 2.5)
    inner = None
    for depth, scale, bright, seed in CRACK_LAYERS:
        shift = m.new("ShaderNodeVectorMath", operation="SCALE")
        m.link(xy_only.outputs[0], shift.inputs[0])
        m.link(m.math("DIVIDE", -depth / 1.31, cos_t), shift.inputs["Scale"])
        at = m.new("ShaderNodeVectorMath", operation="ADD")
        m.link(co, at.inputs[0])
        m.link(shift.outputs[0], at.inputs[1])
        mp = m.new("ShaderNodeMapping")
        mp.inputs["Location"].default_value = (seed, 2 * seed, 0)
        mp.inputs["Rotation"].default_value = (0, 0, seed)
        mp.inputs["Scale"].default_value = (1.0, 1.7, 1.0)           # long cracks, as on Baikal
        m.link(at.outputs[0], mp.inputs["Vector"])
        wv = m.new("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
        m.link(m.new("ShaderNodeTexNoise").outputs["Color"], wv.inputs[0])
        wn = wv.inputs[0].links[0].from_node
        wn.inputs["Scale"].default_value = 2.0 * scale
        m.link(mp.outputs[0], wn.inputs["Vector"])
        wv.inputs[1].default_value = (0.35 / scale, 0.35 / scale, 0.0)
        m.link(mp.outputs[0], wv.inputs[2])
        vor = m.new("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE", voronoi_dimensions="2D")
        vor.inputs["Scale"].default_value = scale
        m.link(wv.outputs[0], vor.inputs["Vector"])
        dist = vor.outputs["Distance"]
        bold = 1.0 + 1.5 * (bright >= 1.0)                     # the top layer: the main cracks, bolder
        line = m.ramp(dist, 0.0025 * bold, 0.0004 * bold)
        sheet = m.math("MULTIPLY", m.ramp(dist, 0.06, 0.0), m.ramp(m.noise(8.0, 6.0, 0.7, vector=at.outputs[0]), 0.35, 0.75))
        layer = m.math("MULTIPLY", m.math("MAXIMUM", line, m.math("MULTIPLY", sheet, 0.18 * bold)), bright)
        layer = m.math("MULTIPLY", layer, m.ramp(m.noise(0.9 + seed, 2.0, 0.5, vector=co), 0.3, 0.5))   # patchy
        inner = layer if inner is None else m.math("MAXIMUM", inner, layer)
    white = m.math("MAXIMUM", m.math("MULTIPLY", scr, 0.45), snow_patch)
    surface_white = m.math("MULTIPLY", white, top)          # on the surface: matte; the cracks are inside
    # The packed snow behind the plates (Scene 3 on), lit by the warm flood, read as a
    # flat cream strip that pulled the eye from the crest: darker and cooler from the cut.
    tint = m.new("ShaderNodeMix", data_type="RGBA")
    m.link(m.value(range(0, KEY_END + 1), lambda f: 0.85 if f >= S3_A else 0.0), tint.inputs["Factor"])
    tint.inputs[6].default_value = (0.92, 0.96, 1.0, 1)
    tint.inputs[7].default_value = (0.3, 0.37, 0.5, 1)
    # From above (Scenes 1-2) the snow and the white cracks under the flood clipped to
    # white and outshone the crest (v7 review): about 1.3 stops darker until the cut.
    snow_dim = m.new("ShaderNodeMix", data_type="RGBA")
    snow_dim.inputs[7].default_value = (0.37, 0.38, 0.4, 1)
    m.link(tint.outputs[2], snow_dim.inputs[6])
    m.link(snow_dim.outputs[2], b.inputs["Base Color"])
    # As the heat cracks grow (Scene 2) the white fractures over the crest dim, so the
    # glowing cracks read first.
    calm = m.value(range(0, KEY_END + 1), lambda f: 1.0 - 0.65 * smooth((f - 44) / 20))
    calm = m.math("ADD", m.math("MULTIPLY", over_crest, m.math("SUBTRACT", calm, 1.0)), 1.0)
    inner = m.math("MULTIPLY", m.math("MULTIPLY", inner, calm), near_cam)
    white = m.math("MAXIMUM", white, m.math("MULTIPLY", m.math("MULTIPLY", inner, CRACK_K), m.math("ADD", 0.7, m.math("MULTIPLY", worn, 0.3))))
    # Away from the crest (only the low Scene 3 camera sees it): wind-blown snow
    # drifts in long streaks, powder patches and more skate marks, so the pond out to
    # the banks reads as used.
    fl = m.new("ShaderNodeVectorMath", operation="LENGTH")
    fxy = m.new("ShaderNodeVectorMath", operation="MULTIPLY")
    m.link(co, fxy.inputs[0])
    fxy.inputs[1].default_value = (1.0, 1.0, 0.0)
    m.link(fxy.outputs[0], fl.inputs[0])
    far = m.ramp(fl.outputs["Value"], 1.6, 3.5)
    wind = m.new("ShaderNodeMapping")
    wind.inputs["Rotation"].default_value = (0, 0, 0.35)
    wind.inputs["Scale"].default_value = (0.5, 3.5, 1.0)
    m.link(co, wind.inputs["Vector"])
    drift = m.ramp(m.noise(0.9, 6.0, 0.62, vector=wind.outputs[0]), 0.5, 0.72)
    streak_map = m.new("ShaderNodeMapping")
    streak_map.inputs["Rotation"].default_value = (0, 0, 0.35)
    streak_map.inputs["Scale"].default_value = (1.0, 30.0, 1.0)
    m.link(co, streak_map.inputs["Vector"])
    streaks = m.math("MULTIPLY", m.ramp(m.noise(1.5, 4.0, 0.6, vector=streak_map.outputs[0]), 0.56, 0.7),
                     m.ramp(m.noise(0.6, 2.0, 0.5, vector=co), 0.4, 0.6))
    # From the low camera the streaks lie across the view and read as wave crests.
    streaks = m.math("MULTIPLY", streaks, m.value(range(0, KEY_END + 1), lambda f: 0.0 if f >= S3_A else 1.0))
    frng = np.random.default_rng(8)
    far_scr = None
    for k in range(40):
        centre = Vector((frng.uniform(-14, 14), frng.uniform(-6, 18), 0))
        off = m.new("ShaderNodeVectorMath", operation="SUBTRACT")
        m.link(co, off.inputs[0])
        off.inputs[1].default_value = centre
        ring = m.new("ShaderNodeVectorMath", operation="LENGTH")
        m.link(off.outputs[0], ring.inputs[0])
        d = m.math("ABSOLUTE", m.math("SUBTRACT", ring.outputs["Value"], frng.uniform(2.0, 9.0)))
        line = m.math("MULTIPLY", m.ramp(d, 0.004, 0.0008), m.ramp(m.noise(0.5 + 0.1 * k, vector=co), 0.4, 0.55))
        far_scr = line if far_scr is None else m.math("MAXIMUM", far_scr, line)
    far_white = m.math("MAXIMUM", m.math("MAXIMUM", m.math("MULTIPLY", drift, 0.6), m.math("MULTIPLY", streaks, 0.45)),
                       m.math("MULTIPLY", far_scr, 0.5))
    white = m.math("MAXIMUM", white, m.math("MULTIPLY", far_white, far))
    # A plate that went under the water comes up wet: the snow on it turns to clear slush
    # and the top goes glossy, so the floating plates read apart from the dry sheet.
    soaked = m.new("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="wet").outputs["Fac"]
    dry = m.math("SUBTRACT", 1.0, m.math("MULTIPLY", soaked, 0.8))
    white = m.math("MULTIPLY", m.math("MULTIPLY", white, top), dry)
    m.link(m.math("MULTIPLY", m.math("DIVIDE", white, SNOW_K, clamp=True),
                  m.value(range(0, KEY_END + 1), lambda f: 0.0 if f >= S3_A else 1.0)), snow_dim.inputs["Factor"])
    # Gentle unevenness of pond ice plus a fine grain from skating (no ripple).
    wave = m.math("ADD", m.noise(0.9, 2.0, 0.4, vector=co), m.math("MULTIPLY", m.noise(160.0, 6.0, 0.6, vector=co), 0.04))
    bump = m.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = 0.002
    m.link(m.math("MULTIPLY", wave, top), bump.inputs["Height"])
    # From the low camera the long swells mirror the flood as ripples: flatten them.
    # From above, the swells refracted the crest into a wobble as the camera crept in
    # (v7 review): gentler there too.
    m.link(m.value(range(0, KEY_END + 1), lambda f: 0.25 if f >= S3_A else 0.35), bump.inputs["Strength"])
    m.link(bump.outputs["Normal"], b.inputs["Normal"])
    # Skated ice is never glass: a faint matte haze everywhere on top (scuffs too fine
    # to see), more where the frost and scratches are.
    haze = m.math("ADD", 0.03, m.math("MULTIPLY", m.noise(2.0, 3.0, 0.5, vector=co), 0.04))
    # At the grazing Scene 3 angle a near-mirror top reads as water: scuffed, frosty
    # ice gives a broad sheen, not sharp streaks of reflected light.
    haze = m.math("ADD", haze, m.value(range(0, KEY_END + 1), lambda f: 0.22 if f >= S3_A else 0.0))
    rough = m.math("MULTIPLY", m.math("MULTIPLY", top, haze), dry)
    m.link(m.math("ADD", rough, m.math("MULTIPLY", surface_white, 0.6), clamp=True), b.inputs["Roughness"])
    m.link(m.math("SUBTRACT", 1.0, m.math("MULTIPLY", white, 0.9), clamp=True), b.inputs["Transmission Weight"])
    # Ice itself absorbs a little (blue): deeper bubbles and the crest fade with depth.
    iv = m.new("ShaderNodeVolumeAbsorption")
    iv.inputs["Color"].default_value = (0.55, 0.72, 0.9, 1)
    iv.inputs["Density"].default_value = 2.5
    m.link(iv.outputs[0], m.out.inputs["Volume"])
    slab.data.materials.append(m.mat)
    # Clear ice must not block lamp light: Cycles treats refractive surfaces as opaque
    # to direct light, which left the crest unlit and hid the skater shadows on it.
    slab.visible_shadow = False

    # Water under the ice: a little murk, so the crest reads as submerged.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, (-ICE_T + FLOOR_Z) / 2))
    water = bpy.context.object
    water.name = "Water"
    water.scale = (ICE_SIZE, ICE_SIZE, -ICE_T - FLOOR_Z - 0.002)
    w = Mat("Water")
    w.nt.nodes.remove(w.bsdf)
    vol = w.new("ShaderNodeVolumeAbsorption")
    vol.inputs["Color"].default_value = (0.25, 0.42, 0.62, 1)          # deep blue-black
    vol.inputs["Density"].default_value = 9.0
    scat = w.new("ShaderNodeVolumeScatter")                # fine suspended silt
    scat.inputs["Color"].default_value = (0.4, 0.5, 0.52, 1)
    scat.inputs["Density"].default_value = 0.0        # no haze: the hot glow lit it into a pink fog
    add = w.new("ShaderNodeAddShader")
    w.link(vol.outputs[0], add.inputs[0])
    w.link(scat.outputs[0], add.inputs[1])
    w.link(add.outputs[0], w.out.inputs["Volume"])
    water.data.materials.append(w.mat)

    bpy.ops.mesh.primitive_plane_add(size=ICE_SIZE, location=(0, 0, FLOOR_Z))
    floor = bpy.context.object
    floor.name = "PondFloor"
    f = Mat("PondFloor")
    # Silt and small stones on the pond bottom: light enough that the puck and the
    # skater shadows read against it through the clear ice.
    grain = f.noise(9.0, 10.0, 0.65)
    stones = f.new("ShaderNodeTexVoronoi")
    stones.inputs["Scale"].default_value = 60.0
    f.link(f.tc.outputs["Object"], stones.inputs["Vector"])
    tone = f.math("ADD", f.math("MULTIPLY", grain, 0.7), f.math("MULTIPLY", stones.outputs["Distance"], 0.5))
    col = f.new("ShaderNodeValToRGB")
    f.link(tone, col.inputs["Fac"])
    e = col.color_ramp.elements
    e[0].position, e[0].color = 0.3, FLOOR_DARK
    e[1].position, e[1].color = 0.8, FLOOR_LIGHT
    f.link(col.outputs["Color"], f.bsdf.inputs["Base Color"])
    f.bsdf.inputs["Roughness"].default_value = 0.9
    bmp = f.new("ShaderNodeBump")
    bmp.inputs["Distance"].default_value = 0.004
    f.link(stones.outputs["Distance"], bmp.inputs["Height"])
    f.link(bmp.outputs["Normal"], f.bsdf.inputs["Normal"])
    floor.data.materials.append(f.mat)
    snow_shadows()
    bubbles()
    fractures()


def bubbles():
    """Trapped air as in real lake ice (reference: Abraham Lake photos): opaque white,
    flat discs of 0.5-3 cm, stacked in vertical 'towers' at a few spots, at several
    depths. Not spread evenly, not round."""
    rng = np.random.default_rng(2011)
    m = Mat("AirInIce")
    b = m.bsdf
    b.inputs["Base Color"].default_value = (0.85, 0.88, 0.93, 1)
    b.inputs["Roughness"].default_value = 0.12
    lw = m.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.2
    # Clearer in the middle, milky towards the rim, like the reference discs.
    # Mostly clear air pocket; only a thin rim catches light.
    m.link(m.math("SUBTRACT", 0.97, m.math("MULTIPLY", m.math("POWER", lw.outputs["Facing"], 3.0), 0.5), clamp=True),
           b.inputs["Transmission Weight"])
    b.inputs["IOR"].default_value = 1.0 / 1.31
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1.0, depth=1.0)
    proto = bpy.context.object
    me = proto.data
    bev = None
    me.materials.append(m.mat)
    bpy.data.objects.remove(proto)
    # Towers sit around the crest and over its edges, not over its middle. Each is a
    # dense cluster of many small discs stacked in columns, fading with depth (the
    # water/ice absorbs the deeper ones), with a loose scatter around it.
    towers = [(-0.42, 0.24), (0.43, 0.3), (-0.45, -0.22), (0.4, -0.25), (-0.12, 0.38), (0.2, -0.36)]
    n = 0
    for cx, cy in towers:
        spread = rng.uniform(0.05, 0.09)
        for column in range(int(rng.integers(3, 8))):
            base = Vector((cx + rng.normal(0, spread), cy + rng.normal(0, spread * 0.8), 0))
            r0 = rng.uniform(0.003, 0.022) if rng.random() < 0.3 else rng.uniform(0.003, 0.009)
            for level in range(int(rng.integers(3, 9))):
                depth = 0.006 + level * rng.uniform(0.006, 0.012)
                if depth > ICE_T - 0.006:
                    break
                r = r0 * rng.uniform(0.7, 1.15)
                o = link(f"Disc{n}", me)
                n += 1
                o.location = base + Vector((rng.normal(0, r0 * 0.6), rng.normal(0, r0 * 0.6), -depth))
                o.rotation_euler = (rng.normal(0, 0.06), rng.normal(0, 0.06), 0)
                o.scale = (r, r * rng.uniform(0.85, 1.15), r * 0.03)
        for j in range(int(rng.integers(8, 16))):
            r = rng.uniform(0.002, 0.008)
            o = link(f"Disc{n}", me)
            n += 1
            o.location = (cx + rng.normal(0, spread * 2.2), cy + rng.normal(0, spread * 1.8),
                          -rng.uniform(0.006, ICE_T - 0.006))
            o.scale = (r, r, r * 0.04)


def fractures():
    """Real cracks (reference: Fishhook Lake): branching paths across the ice. Each is
    a thin frosted sheet that goes down into the ice (it has depth and shifts with the
    camera), with snow packed along the top in an uneven strip."""
    rng = np.random.default_rng(91)
    sheet = Mat("CrackSheet")
    sb = sheet.bsdf
    sb.inputs["Base Color"].default_value = (0.8, 0.85, 0.9, 1)
    sb.inputs["Roughness"].default_value = 0.2
    sb.inputs["Transmission Weight"].default_value = 0.92
    sb.inputs["IOR"].default_value = 1.0 / 1.31
    snow = Mat("CrackSnow")
    sn = snow.bsdf
    sn.inputs["Base Color"].default_value = (0.88, 0.9, 0.93, 1)
    sn.inputs["Roughness"].default_value = 0.85
    clump = snow.ramp(snow.noise(60.0, 6.0, 0.7), 0.42, 0.6)
    edge = snow.new("ShaderNodeSeparateXYZ")
    uv = snow.new("ShaderNodeUVMap")
    snow.link(uv.outputs["UV"], edge.inputs[0])
    centre = snow.math("SUBTRACT", 1.0, snow.math("MULTIPLY", snow.math("ABSOLUTE", snow.math("SUBTRACT", edge.outputs["Y"], 0.5)), 2.0))
    cover = snow.math("MULTIPLY", clump, snow.math("POWER", centre, 0.6))
    hold = snow.new("ShaderNodeMixShader")
    clear = snow.new("ShaderNodeBsdfTransparent")
    snow.link(cover, hold.inputs["Fac"])
    snow.link(clear.outputs[0], hold.inputs[1])
    snow.link(sn.outputs[0], hold.inputs[2])
    snow.link(hold.outputs[0], snow.out.inputs["Surface"])

    def walk(start, heading, length, step=0.02):
        pts, p, h = [start], Vector(start), heading
        for _ in range(int(length / step)):
            h += rng.normal(0, 0.12)
            p = p + Vector((math.cos(h), math.sin(h), 0)) * step
            pts.append(p.copy())
        return pts

    def ribbon(name, pts, mat, depth=None, width=None):
        verts, faces, uvs = [], [], []
        for i, p in enumerate(pts):
            if depth is not None:                        # vertical sheet down into the ice
                d = depth * (0.6 + 0.4 * math.sin(i * 0.3 + rng.uniform(0, 3)))
                lean = rng.normal(0, 0.003)
                verts += [(p.x, p.y, -0.0005), (p.x + lean, p.y + lean, -d)]
            else:                                         # flat strip on top
                if i + 1 < len(pts):
                    t = (pts[i + 1] - p).normalized()
                n = Vector((-t.y, t.x, 0)) * width * (0.5 + rng.random())
                verts += [tuple(p + n + Vector((0, 0, 0.0004))), tuple(p - n + Vector((0, 0, 0.0004)))]
            if i:
                a = 2 * (i - 1)
                faces.append((a, a + 2, a + 3, a + 1))
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        uvl = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices):
                uvl.data[li].uv = ((vi // 2) / max(1, len(pts) - 1), vi % 2)
        me.materials.append(mat)
        return link(name, me)

    mains = [((-1.3, -0.15, 0), 0.08, 2.8), ((0.9, 0.9, 0), -1.9, 1.8), ((-1.0, 0.85, 0), -0.7, 1.2)]
    n = 0
    for start, heading, length in mains:
        pts = walk(Vector(start), heading, length)
        paths = [pts]
        for _ in range(int(rng.integers(1, 3))):             # a branch or two
            at = pts[int(rng.integers(len(pts) // 4, 3 * len(pts) // 4))]
            paths.append(walk(at, heading + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1), rng.uniform(0.2, 0.6)))
        for path in paths:
            ribbon(f"CrackSheet{n}", path, sheet.mat, depth=rng.uniform(0.03, 0.08))
            ribbon(f"CrackSnow{n}", path, snow.mat, width=rng.uniform(0.004, 0.009))
            n += 1


# ---------------------------------------------------------------- scene 1 actors

def crest(args):
    rig, set_heat = crest3d.build(args.crest_svg, CREST_W, bolts=[
        (-0.17, 0.30), (0.27, 0.18), (-0.17, -0.30), (0.21, -0.28), (0.0, 0.215), (0.0, -0.205),
        (-0.38, 0.02), (0.31, 0.0), (0.004, 0.02)])
    # Lying face up under the ice, turned a little so it is not square to the frame.
    rig.location = (0.0, 0.0, CREST_BASE)
    rig.rotation_euler = (0, 0, math.radians(-8))
    def face_glow(h):
        return 0.2 + 0.35 * smooth((clamp01(h) - 0.3) / 0.6)    # edges still brighter: the letters read

    def heat_keys(f):
        h = heat(f)
        # Blackbody temperature, glow, cold->hot colour blend, rust burning off.
        # Metal shows no glow until it is very hot: glow rises steeply (~T^4), so the
        # colour shifts first (blue -> dull grey -> dark red), then the glow takes over.
        # Cherry red, not orange: peak ~1000 K. The paint chars dark as it heats and the
        # glow comes through unevenly (hot steel, reference-style), no shift to pink.
        if f <= S2_END:
            # The enamel shifts to the hot kit's red and chars only part way, and the
            # whole plate glows, not only its edges: with dark faces only the glowing
            # edges read and the crest looked like a wire outline (v7 review).
            set_heat(f, 740 + 240 * clamp01(h), 1.5 * clamp01(h) ** 2.4, 0.9 * smooth((h - 0.25) / 0.5),
                     1.0 - 0.8 * clamp01(h), 0.5 * smooth((h - 0.05) / 0.55), face=face_glow(h))
        else:
            # Scene 3: out of the ice it cools towards the red/black kit, the outline
            # first; the middle still glows dull red when the shot hands over.
            c = smooth((f - (S3_A + 3)) / (S3_END - S3_A - 3))
            # Scene 4: the last of the glow dies in the middle and it settles on the red kit.
            d = smooth((f - S4_A) / 22)
            # It flares as the ice gives and the water flashes off it: from the low cut the
            # glow must read through the ice as brightly as it did from above (v7 review).
            flare = 1 + 1.5 * math.exp(-max(f - S3_A, 0) / 3.0)
            set_heat(f, 740 + 240 * clamp01(h) - 120 * c - 60 * d,
                     1.5 * clamp01(h) ** 2.4 * (1 - 0.6 * c) * (1 - d) * flare,
                     0.85 * c + 0.15 * d, 0.2, smooth((h - 0.05) / 0.55) * (1 - 0.85 * c) * (1 - d), rim=c,
                     face=0.2 + (face_glow(h) - 0.2) * math.exp(-(f - S3_A) / 4.0))   # the faces cool first
    keys(range(0, KEY_END + 1), heat_keys)
    # The cold blue pulse at frame 28: a blue light just above the crest, under the ice.
    data = bpy.data.lights.new("Pulse", "AREA")
    data.color = (0.35, 0.62, 1.0)
    data.size = 0.7
    pulse = link("Pulse", data)
    pulse.location = (0, 0, CREST_BASE + 0.05)
    pulse.rotation_euler = (math.pi, 0, 0)          # pointing down at the crest
    pulse.visible_camera = pulse.visible_glossy = pulse.visible_transmission = False

    def k(f):
        data.energy = 0.0 if f < 27 else 45.0 * math.exp(-(f - 28) ** 2 / 6.0)
        data.keyframe_insert("energy", frame=f)
    keys(range(0, KEY_END + 1), k)

    # The peak (top point of the crest) in world space at rest, for Scene 2 framing.
    bpy.context.view_layer.update()
    top = max((o.matrix_world @ v.co for o in rig.children if o.type == "MESH" for v in o.data.vertices),
              key=lambda p: p.y)
    peak = Vector((top.x, top.y, -ICE_T))

    def move(f):
        dx, dy, dz, tx, ty, yaw = crest_motion(f)
        rig.location = (dx, dy, CREST_BASE + dz)
        rig.rotation_euler = (math.radians(tx), math.radians(ty), math.radians(-8 + yaw))
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)
    keys(range(0, KEY_END + 1), move)

    # Heat light from the crest, pointing up through the ice (it lights the bubbles
    # and the cracks from below).
    hl = bpy.data.lights.new("HeatUp", "AREA")
    hl.shape, hl.size, hl.size_y = "RECTANGLE", 0.55, 0.6
    hl.color = (1.0, 0.16, 0.03)
    ho = link("HeatUp", hl)
    ho.location = (0, 0, CREST_BASE + 0.045)
    ho.rotation_euler = (math.pi, 0, 0)              # area lights shine along -Z; flip to shine up
    ho.visible_camera = ho.visible_glossy = ho.visible_transmission = False

    def light(f):
        hl.energy = 3.0 * heat(f) ** 2
        hl.keyframe_insert("energy", frame=f)
        dx, dy, dz, *_ = crest_motion(f)
        ho.location = (dx, dy, CREST_BASE + 0.045 + dz)
        ho.keyframe_insert("location", frame=f)
    keys(range(0, KEY_END + 1), light)
    return rig, peak


def boil(rig):
    """From frame 40, streams of small bubbles rise off the hot crest and collect as
    a flattened layer under the ice (hot metal in cold water)."""
    rng = np.random.default_rng(40)
    m = Mat("BoilBubble")
    b = m.bsdf
    b.inputs["Base Color"].default_value = (1, 1, 1, 1)
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["IOR"].default_value = 1.0 / 1.33
    b.inputs["Roughness"].default_value = 0.03
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=1.0)
    proto = bpy.context.object
    me = proto.data
    me.materials.append(m.mat)
    bpy.ops.object.shade_smooth()
    bpy.data.objects.remove(proto)
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ v.co for o in rig.children if o.type == "MESH" for v in o.data.vertices]
    top_z = max(p.z for p in pts)
    surface = [p for p in pts if p.z > top_z - 0.012]
    ceiling = -ICE_T
    items = []
    for i in range(260):
        born = 40 + (S2_END - 40) * rng.random() ** 0.6        # more as it heats
        src = surface[int(rng.integers(len(surface)))]
        r = rng.uniform(0.0008, 0.0035)
        speed = rng.uniform(0.12, 0.25)
        o = link(f"Boil{i}", me)
        items.append((o, born, Vector(src), r, speed, rng.uniform(0, 6.3)))

    def k(f):
        for o, born, src, r, speed, ph in items:
            age = (f - born) / FPS
            if age < 0:
                o.scale = (0, 0, 0)
                o.location = src
            else:
                z = src.z + speed * age
                wob = 0.002 * math.sin(age * 25 + ph)
                if z < ceiling - r:
                    o.location = (src.x + wob, src.y + wob * 0.5, z)
                    o.scale = (r, r, r)
                else:                                          # pooled under the ice: flat, spreading
                    # Pooled under the ice: small and flat, then they slide off (fade).
                    spread = min(1.0, (z - ceiling) / 0.05)
                    keep = max(0.0, 1.0 - (z - ceiling) / 0.12)
                    o.location = (src.x, src.y, ceiling - r * 0.25)
                    o.scale = (r * (1 + 0.5 * spread) * keep, r * (1 + 0.5 * spread) * keep, r * 0.25 * keep)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
    keys(range(0, KEY_END + 1), k)


def heat_cracks(peak, rig):
    """Returns the main crack paths (Scene 3 breaks the ice into plates along them)."""
    """Cracks driven out from where the crest's peak presses the ice: two short ones at
    the first shove (44), one racing towards the lens at the second (60), a star at the
    third (72). Each grows fast along its length, and orange light leaks through it."""
    rng = np.random.default_rng(72)

    def walk(start, heading, length, step=0.008):
        pts, p, h = [Vector(start)], Vector(start), heading
        for _ in range(max(2, int(length / step))):
            h += rng.normal(0, 0.18)
            p = p + Vector((math.cos(h), math.sin(h), 0)) * step
            pts.append(p.copy())
        return pts

    def crack_mat(name, t0, glow_k=1.0):
        m = Mat(name)
        b = m.bsdf
        b.inputs["Base Color"].default_value = (0.92, 0.95, 0.98, 1)
        b.inputs["IOR"].default_value = 1.0 / 1.31
        # Real fracture planes are feathered: silvery-white where the break scatters
        # light, clear elsewhere (reference: Baikal / Fishhook Lake).
        uvf = m.new("ShaderNodeUVMap")
        mapping = m.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (40.0, 7.0, 1.0)
        m.link(uvf.outputs["UV"], mapping.inputs["Vector"])
        # Mostly clear, with fine silvery feathering only (not crumpled foil).
        feather = m.ramp(m.noise(1.0, 9.0, 0.72, vector=mapping.outputs[0]), 0.62, 0.76)
        m.link(m.math("SUBTRACT", 1.0, m.math("MULTIPLY", feather, 0.45)), b.inputs["Transmission Weight"])
        m.link(m.math("ADD", 0.02, m.math("MULTIPLY", feather, 0.2)), b.inputs["Roughness"])
        b.inputs["IOR"].default_value = 1.0 / 1.31
        b.inputs["Emission Color"].default_value = (1.0, 0.17, 0.03, 1)
        uv = m.new("ShaderNodeUVMap")
        sep = m.new("ShaderNodeSeparateXYZ")
        m.link(uv.outputs["UV"], sep.inputs[0])
        grow = m.value(range(0, KEY_END + 1), lambda f: 1 - (1 - clamp01((f - t0) / 4)) ** 3 if f >= t0 else -0.1)
        shown = m.math("LESS_THAN", sep.outputs["X"], grow)
        ragged = m.math("ADD", sep.outputs["Y"], m.math("MULTIPLY", m.noise(14.0, 4.0, 0.6, vector=uv.outputs["UV"]), 0.35))
        shown = m.math("MULTIPLY", shown, m.math("LESS_THAN", ragged, 1.05))
        # Light leaks most near the bottom of the sheet (the hot side) and near the origin.
        # Light leaks along the deep (hot) edge of the crack, fading up the sheet and
        # along its length; the top edge stays a thin silver line.
        leak = m.math("MULTIPLY", m.math("POWER", sep.outputs["Y"], 14.0), m.math("SUBTRACT", 1.1, sep.outputs["X"], clamp=True))
        # A thin silver line where the crack meets the surface.
        top_line = m.ramp(sep.outputs["Y"], 0.06, 0.0)
        b.inputs["Coat Weight"].default_value = 0.0
        m.link(top_line, b.inputs["Specular IOR Level"])
        glow = m.value(range(0, KEY_END + 1), lambda f: glow_k * (3.0 * heat(f) + (8.0 * math.exp(-(f - t0) / 3) if f >= t0 else 0)))
        m.link(m.math("MULTIPLY", leak, glow), b.inputs["Emission Strength"])
        mix = m.new("ShaderNodeMixShader")
        clear = m.new("ShaderNodeBsdfTransparent")
        m.link(shown, mix.inputs["Fac"])
        m.link(clear.outputs[0], mix.inputs[1])
        m.link(b.outputs[0], mix.inputs[2])
        m.link(mix.outputs[0], m.out.inputs["Surface"])
        return m.mat

    ROWS = 4

    def sheet(name, pts, mat, depth):
        """A fracture plane: leaning (not vertical), with a jagged, uneven bottom edge."""
        lean = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])         # tan of the lean angle
        verts, faces = [], []
        n = len(pts)
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
            side = Vector((-t.y, t.x, 0))
            d = depth * (0.45 + 0.55 * math.cos(i / n * math.pi / 2)) * (0.6 + 0.8 * rng.random())
            for row in range(ROWS):
                z = -0.0004 - d * row / (ROWS - 1)
                q = p + side * (lean * (-z)) + side * rng.normal(0, 0.0003) * row
                verts.append((q.x, q.y, z))
            if i:
                for row in range(ROWS - 1):
                    a, b = (i - 1) * ROWS + row, i * ROWS + row
                    faces.append((a, b, b + 1, a + 1))
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        uvl = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices):
                uvl.data[li].uv = ((vi // ROWS) / max(1, n - 1), (vi % ROWS) / (ROWS - 1))
        me.materials.append(mat)
        return link(name, me)

    def line_mat(name, t0, glow_k=1.0):
        """The crack seen from above: a thin frosted line on the surface, lit orange from below."""
        m = Mat(name)
        b = m.bsdf
        b.inputs["Base Color"].default_value = (0.92, 0.94, 0.96, 1)
        b.inputs["Roughness"].default_value = 0.6
        b.inputs["Emission Color"].default_value = (1.0, 0.2, 0.04, 1)
        uv = m.new("ShaderNodeUVMap")
        sep = m.new("ShaderNodeSeparateXYZ")
        m.link(uv.outputs["UV"], sep.inputs[0])
        grow = m.value(range(0, KEY_END + 1), lambda f: 1 - (1 - clamp01((f - t0) / 4)) ** 3 if f >= t0 else -0.1)
        shown = m.math("LESS_THAN", sep.outputs["X"], grow)
        m.link(m.value(range(0, KEY_END + 1), lambda f: glow_k * (0.6 * heat(f) + (2.5 * math.exp(-(f - t0) / 4) if f >= t0 else 0))),
               b.inputs["Emission Strength"])
        mix = m.new("ShaderNodeMixShader")
        clear = m.new("ShaderNodeBsdfTransparent")
        m.link(shown, mix.inputs["Fac"])
        m.link(clear.outputs[0], mix.inputs[1])
        m.link(b.outputs[0], mix.inputs[2])
        m.link(mix.outputs[0], m.out.inputs["Surface"])
        return m.mat

    def line(name, pts, mat, width=0.0012):
        verts, faces = [], []
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
            w = width * (1.0 - 0.6 * i / len(pts)) * (0.6 + 0.8 * rng.random())
            nrm = Vector((-t.y, t.x, 0)) * w
            verts += [tuple(p + nrm + Vector((0, 0, 0.0006))), tuple(p - nrm + Vector((0, 0, 0.0006)))]
            if i:
                a = 2 * (i - 1)
                faces.append((a, a + 2, a + 3, a + 1))
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        uvl = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices):
                uvl.data[li].uv = ((vi // 2) / max(1, len(pts) - 1), vi % 2)
        me.materials.append(mat)
        return link(name, me)

    origin = Vector((peak.x, peak.y, 0))
    # Angle A looks down with the peak near the top of frame, so the first cracks run
    # down and sideways into the frame.
    groups = [(44, [(a, rng.uniform(0.18, 0.28)) for a in (-0.35 + rng.normal(0, 0.1), -2.8 + rng.normal(0, 0.1))]),
              (60, [(-math.pi / 2 + rng.normal(0, 0.15), 0.38)]),            # towards camera B
              (72, [(a + rng.normal(0, 0.2), rng.uniform(0.2, 0.36)) for a in np.linspace(0, 2 * math.pi, 5)[:-1] + 0.4])]
    n = 0
    paths = []
    for t0, cracks in groups:
        # The first cracks start lower, over the upper mountain: angle A (from above)
        # sees the surface above the crest's tip at the very top edge of its frame.
        start = origin + Vector((0.0, -0.24, 0.0)) if t0 == SHOVES[0] else origin
        mat = crack_mat(f"HeatCrack{t0}", t0)
        lmat = line_mat(f"HeatCrackLine{t0}", t0)
        for heading, length in cracks:
            pts = walk(start, heading, length)
            paths.append(pts)
            sheet(f"HeatCrack{n}", pts, mat, rng.uniform(0.03, 0.06))
            line(f"HeatCrackLine{n}", pts, lmat)
            # Hairline cracks branching off: short, shallow, no surface line.
            for j in range(int(rng.integers(2, 4))):
                at = pts[int(rng.integers(1, len(pts)))]
                hair = walk(at, heading + rng.choice([-1, 1]) * rng.uniform(0.5, 1.4), rng.uniform(0.02, 0.07), step=0.004)
                sheet(f"HeatHair{n}_{j}", hair, mat, rng.uniform(0.008, 0.025))
            n += 1

    # Cracks at the bottom of the frame (the camera side). The heat spreads under the
    # whole crest, so the sheet also fails over its lower tip: one at each shove, then a
    # long, nearly straight sheet crack runs across in front of the crest at the third.
    # Lake-ice sheet cracks are long and gently curved, so these wander less.
    # (Own generator: the approved peak cracks above keep their shapes.)
    rng = np.random.default_rng(144)
    bpy.context.scene.frame_set(0)
    low_pt = min((o.matrix_world @ v.co for o in rig.children if o.type == "MESH" for v in o.data.vertices),
                 key=lambda p: p.y)
    tip = Vector((low_pt.x, low_pt.y, 0))                        # the crest's lower tip, on the surface

    def straight_walk(start, heading, length, step=0.01, wander=0.07):
        pts, p, h = [Vector(start)], Vector(start), heading
        for _ in range(max(2, int(length / step))):
            h += rng.normal(0, wander)
            p = p + Vector((math.cos(h), math.sin(h), 0)) * step
            pts.append(p.copy())
        return pts

    low = [(44, tip + Vector((-0.06, 0.04, 0)), [(-2.1, 0.22)]),
           (60, tip, [(-1.45, 0.3), (-0.6, 0.2)]),
           (72, tip + Vector((0.05, 0.02, 0)), [(-1.9, 0.28), (-1.15, 0.24)]),
           (72, tip + Vector((0.16, 0.07, 0)), [(-0.45, 0.3)]),
           (72, Vector((-0.42, tip.y - 0.05, 0)), [(0.1, 0.86)])]                  # across the bottom
    for t0, start, cracks in low:
        # Further from the heat and nearer the lens: dimmer and finer than at the peak.
        mat = crack_mat(f"HeatCrackLow{t0}_{n}", t0 + 1, glow_k=0.35)
        lmat = line_mat(f"HeatCrackLineLow{t0}_{n}", t0 + 1, glow_k=0.3)
        for heading, length in cracks:
            pts = straight_walk(start, heading + rng.normal(0, 0.08), length)
            paths.append(pts)
            sheet(f"HeatCrack{n}", pts, mat, rng.uniform(0.03, 0.05))
            line(f"HeatCrackLine{n}", pts, lmat, width=0.0006)
            for j in range(int(rng.integers(2, 4))):
                at = pts[int(rng.integers(len(pts) // 4, len(pts)))]
                hair = straight_walk(at, heading + rng.choice([-1, 1]) * rng.uniform(0.4, 1.0),
                                     rng.uniform(0.03, 0.09), step=0.005, wander=0.15)
                sheet(f"HeatHair{n}_{j}", hair, mat, rng.uniform(0.01, 0.03))
            n += 1
    return paths


def puck():
    """A regulation puck: black vulcanised rubber, knurled edge band, scuffed faces."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=PUCK_R, depth=PUCK_H, location=(0, 0, PUCK_H / 2))
    p = bpy.context.object
    p.name = "Puck"
    bev = p.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments = 0.0015, 3
    bpy.ops.object.shade_smooth()
    m = Mat("Puck")
    b = m.bsdf
    b.inputs["Base Color"].default_value = (0.012, 0.012, 0.013, 1)
    sep = m.new("ShaderNodeSeparateXYZ")
    m.link(m.tc.outputs["Object"], sep.inputs[0])
    ang = m.math("ARCTAN2", sep.outputs["Y"], sep.outputs["X"])
    # Knurled band: diagonal cross-hatch on the edge only.
    side = m.math("LESS_THAN", m.math("ABSOLUTE", sep.outputs["Z"]), PUCK_H * 0.32)
    k1 = m.math("SINE", m.math("ADD", m.math("MULTIPLY", ang, 60.0), m.math("MULTIPLY", sep.outputs["Z"], 900.0)))
    k2 = m.math("SINE", m.math("SUBTRACT", m.math("MULTIPLY", ang, 60.0), m.math("MULTIPLY", sep.outputs["Z"], 900.0)))
    knurl = m.math("MULTIPLY", m.math("MULTIPLY", k1, k2), side)
    scuff = m.ramp(m.noise(180.0, 6.0, 0.7), 0.55, 0.7)
    bump = m.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = 0.0004
    m.link(m.math("ADD", knurl, m.math("MULTIPLY", scuff, -0.6)), bump.inputs["Height"])
    m.link(bump.outputs["Normal"], b.inputs["Normal"])
    # Reference (stick-and-puck photos): the rim is worn smoother and greyer, so the
    # top edge catches a thin highlight that makes the puck read on dark ice.
    rim = m.ramp(m.math("ABSOLUTE", sep.outputs["Z"]), PUCK_H * 0.38, PUCK_H * 0.5)
    m.link(m.math("SUBTRACT", m.math("ADD", 0.7, m.math("MULTIPLY", scuff, 0.2)), m.math("MULTIPLY", rim, 0.45)),
           b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = 0.5
    grey = m.new("ShaderNodeMix", data_type="RGBA")
    m.link(m.math("MAXIMUM", m.math("MULTIPLY", scuff, 0.6), m.math("MULTIPLY", rim, 0.7)), grey.inputs[0])
    grey.inputs[6].default_value = (0.012, 0.012, 0.013, 1)
    grey.inputs[7].default_value = (0.09, 0.09, 0.09, 1)
    m.link(grey.outputs[2], b.inputs["Base Color"])
    p.data.materials.append(m.mat)

    start, end = Vector((-1.3, 0.06, 0)), Vector((1.4, -0.14, 0))     # about 9 m/s, over the crest's middle

    def k(f):
        t = clamp01((f - 15) / 13)                   # about 6 m/s: a brisk pass that still reads
        pos = start.lerp(end, t)
        p.location = (pos.x, pos.y, PUCK_H / 2)
        p.rotation_euler = (0, 0, -6.0 * t)
        p.hide_render = not 14 <= f <= 29
        for path in ("location", "rotation_euler", "hide_render"):
            p.keyframe_insert(path, frame=f)
    keys(range(0, KEY_END + 1), k)

    # A thin trail of snow dust kicked up behind the puck: it reads even over dark ice.
    rng = np.random.default_rng(18)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    proto = bpy.context.object
    me = proto.data
    dm = Mat("PuckDust")
    dm.bsdf.inputs["Base Color"].default_value = (0.85, 0.88, 0.92, 1)
    dm.bsdf.inputs["Roughness"].default_value = 0.7
    me.materials.append(dm.mat)
    bpy.data.objects.remove(proto)
    grains = []
    for i in range(120):
        born = rng.uniform(15, 28)
        at = start.lerp(end, clamp01((born - 15) / 13))
        v = Vector((rng.uniform(-0.3, 0.6), rng.uniform(-0.25, 0.25), rng.uniform(0.05, 0.35)))
        grains.append((link(f"PuckDust{i}", me), born, at + Vector((rng.uniform(-0.02, 0.02), rng.uniform(-0.03, 0.03), 0.002)),
                       v, rng.uniform(0.0006, 0.0018), rng.uniform(4, 9)))

    def g(f):
        for o, born, p0, v, r, life in grains:
            age = (f - born) / FPS
            pos = p0 + v * age + Vector((0, 0, -4.9 * age * age))
            alive = 0 <= age and f - born < life
            o.location = (pos.x, pos.y, max(pos.z, 0.001))
            o.scale = (r, r, r) if alive else (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
    keys(range(0, KEY_END + 1), g)


def skaters():
    """Two skaters chasing the puck, never in frame: only their shadows cross the ice.
    Built from rounded mesh capsules (helmet, padded shoulders, gloves, bent knees,
    stick out front) with an alternating sideways stride, so the shadows read as
    players."""
    proxy = Mat("SkaterProxy")
    proxy.bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1)

    def capsule(name, r, parent):
        """A unit capsule along +z (length 1), placed later by place(). Built with
        bmesh: joining primitives through operators left stray spheres behind."""
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=False, segments=12, radius1=r, radius2=r, depth=1.0,
                              matrix=Matrix.Translation((0, 0, 0.5)))
        for z in (0.0, 1.0):
            bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=r,
                                      matrix=Matrix.Translation((0, 0, z)))
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        c = link(name, me)
        c.data.materials.append(proxy.mat)
        c.visible_camera = c.visible_glossy = c.visible_transmission = False
        c.parent = parent
        return c

    def place(c, a, b):
        a, b = Vector(a), Vector(b)
        c.location = a
        c.rotation_euler = (b - a).to_track_quat("Z", "Y").to_euler()
        c.scale = (1, 1, max((b - a).length, 1e-3))

    def body(name, phase):
        rig = link(name, None)
        # Built facing +y (forward), crouched.
        for nm, r, a, b in [("Helmet", 0.13, (0, 0.34, 1.5), (0, 0.38, 1.62)),
                            ("Torso", 0.22, (0, 0.05, 0.95), (0, 0.3, 1.35)),
                            ("Shoulders", 0.13, (-0.24, 0.28, 1.32), (0.24, 0.28, 1.32)),
                            ("Pants", 0.23, (0, 0.0, 0.92), (0, 0.06, 0.78)),
                            ("ArmR", 0.08, (0.24, 0.28, 1.3), (0.2, 0.62, 1.0)),
                            ("ArmL", 0.08, (-0.24, 0.28, 1.3), (-0.05, 0.55, 1.05)),
                            ("Stick", 0.018, (0.1, 0.6, 1.0), (0.35, 1.55, 0.03)),
                            ("StickBlade", 0.03, (0.3, 1.5, 0.03), (0.6, 1.62, 0.03))]:
            place(capsule(f"{name}{nm}", r, rig), a, b)
        legs = [(side, capsule(f"{name}Thigh{side}", 0.1, rig), capsule(f"{name}Shin{side}", 0.08, rig))
                for side in (-1, 1)]

        def stride(f):
            for side, thigh, shin in legs:
                push = max(0.0, math.sin((f + phase) * 2 * math.pi / 11 + (0 if side < 0 else math.pi)))
                out = side * 0.32 * push
                hip = (side * 0.12, 0.0, 0.85)
                knee = (side * 0.14 + out * 0.5, 0.25 - 0.1 * push, 0.5 + 0.05 * push)
                ankle = (side * 0.16 + out, 0.05 - 0.25 * push, 0.1 + 0.08 * (1 - push))
                for c, a, b in ((thigh, hip, knee), (shin, knee, ankle)):
                    place(c, a, b)
                    for path in ("location", "rotation_euler", "scale"):
                        c.keyframe_insert(path, frame=f)
        keys(range(0, KEY_END + 1), stride)
        return rig
    a, b = body("SkaterA", 0), body("SkaterB", 5)
    # They chase the puck's line just beyond the top of the frame; the floodlight is
    # up behind them, so their shadows fall across the crest towards the camera.
    path = lambda f, lag: Vector((lerp(-2.6, 2.2, clamp01((f - 17 - lag) / 16)), 2.6, 0))   # ~9 m/s

    def k(f):
        for rig, lag in ((a, 0.0), (b, 3.0)):
            rig.location = path(f, lag)
            rig.rotation_euler = (0, 0, -math.pi / 2)                     # facing +x, the way they skate
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)
    keys(range(0, KEY_END + 1), k)


def stick_blade():
    """A taped stick blade that cuts through the top edge of the frame (frames 24-28)."""
    rig = link("StickBlade", None)
    tape = Mat("Tape")
    b = tape.bsdf
    sep = tape.new("ShaderNodeSeparateXYZ")
    tape.link(tape.tc.outputs["Object"], sep.inputs[0])
    wraps = tape.math("SINE", tape.math("MULTIPLY", tape.math("ADD", sep.outputs["X"], sep.outputs["Z"]), 160.0))
    fray = tape.ramp(tape.noise(90.0, 8.0, 0.7), 0.62, 0.75)
    snow = tape.ramp(tape.noise(40.0, 6.0, 0.6), 0.6, 0.72)
    col = tape.new("ShaderNodeMix", data_type="RGBA")
    tape.link(snow, col.inputs[0])
    col.inputs[6].default_value = (0.015, 0.015, 0.016, 1)
    col.inputs[7].default_value = (0.75, 0.78, 0.82, 1)
    tape.link(col.outputs[2], b.inputs["Base Color"])
    tape.link(tape.math("ADD", 0.7, tape.math("MULTIPLY", fray, 0.2)), b.inputs["Roughness"])
    bump = tape.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = 0.0003
    tape.link(tape.math("ADD", tape.math("MULTIPLY", wraps, 0.5), fray), bump.inputs["Height"])
    tape.link(bump.outputs["Normal"], b.inputs["Normal"])
    # The blade: a gently curved plate, 30 cm long and 7.5 cm tall.
    bpy.ops.mesh.primitive_cube_add(size=1)
    blade = bpy.context.object
    blade.name = "Blade"
    blade.scale = (0.30, 0.012, 0.075)
    bend = blade.modifiers.new("Bend", "SIMPLE_DEFORM")
    bend.deform_method, bend.angle, bend.deform_axis = "BEND", math.radians(18), "Z"
    sub = blade.modifiers.new("Sub", "SUBSURF")
    sub.levels = sub.render_levels = 2
    blade.data.materials.append(tape.mat)
    blade.parent = rig
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.17, 0, 0.3))
    shaft = bpy.context.object
    shaft.scale = (0.03, 0.022, 0.6)
    shaft.rotation_euler = (0, math.radians(-35), 0)
    shaft.data.materials.append(tape.mat)
    shaft.parent = rig

    def k(f):
        t = clamp01((f - 23) / 6)
        rig.location = (lerp(-0.9, 0.9, t), 0.62, 0.045)
        rig.rotation_euler = (0, 0, math.radians(lerp(-10, 6, t)))
        rig.hide_render = not 22 <= f <= 30
        for o in (rig, blade, shaft):
            o.hide_render = rig.hide_render
            o.keyframe_insert("hide_render", frame=f)
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)
    keys(range(0, KEY_END + 1), k)


def particles(vertical):
    """Falling snow throughout, snow dust drifting on the ice, and the skates' ice flecks."""
    rng = np.random.default_rng(28)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    proto = bpy.context.object
    me = proto.data
    for v in me.vertices:                                   # irregular, flat flakes
        v.co.x *= rng.uniform(0.6, 1.4)
        v.co.y *= rng.uniform(0.6, 1.4)
        v.co.z *= 0.25
    m = Mat("Snow")
    m.bsdf.inputs["Base Color"].default_value = (0.82, 0.85, 0.9, 1)
    m.bsdf.inputs["Roughness"].default_value = 0.5
    me.materials.append(m.mat)
    bpy.data.objects.remove(proto)
    items = []
    # Falling flakes between the camera and the ice.
    for i in range(55):
        p0 = Vector((rng.uniform(-1.1, 1.1), rng.uniform(-0.4, 1.2), rng.uniform(0.05, 0.7)))
        items.append(("flake", link(f"Flake{i}", me), p0, rng.uniform(0.25, 0.5), rng.uniform(0, 6.3),
                      rng.uniform(0.0008, 0.0022)))
    # Snow dust grains skating across the ice in the wind.
    for i in range(140):
        p0 = Vector((rng.uniform(-1.6, 1.2), rng.uniform(-0.8, 0.8), 0.0006))
        items.append(("dust", link(f"Dust{i}", me), p0, rng.uniform(0.04, 0.12), rng.uniform(0, 6.3),
                      rng.uniform(0.0004, 0.0012)))
    # Ice flecks kicked in from the top edge where the skates dig in (frames 24-31).
    for i in range(70):
        born = rng.uniform(24, 29)
        p0 = Vector((rng.uniform(-0.4, 0.6), 0.75, 0.01))
        v = Vector((rng.uniform(-0.3, 1.2), rng.uniform(-1.6, -0.4), rng.uniform(0.3, 1.4)))
        items.append(("fleck", link(f"Fleck{i}", me), p0, born, v, rng.uniform(0.0008, 0.003)))

    def k(f):
        t = f / FPS
        for kind, o, p0, a, b, r in items:
            if kind == "flake":
                z = p0.z - a * t                      # no wrap: a jump would smear under motion blur
                tl = min(t, (p0.z - 0.0004) / a)      # a landed flake stays put
                o.location = (p0.x + 0.05 * math.sin(tl * 1.5 + b) + 0.08 * tl, p0.y, max(z, 0.0004))
                o.scale = (r, r, r * 0.5) if z > -0.02 else (r, r, r * 0.25)
            elif kind == "dust":
                tw = wind_t(f)
                o.location = (p0.x + a * tw + 0.01 * math.sin(tw * 3 + b), p0.y + 0.004 * math.sin(tw * 2 + b), p0.z)
                o.scale = (r, r, r * 0.4)
            else:
                age = (f - a) / FPS
                pos = p0 + b * age + Vector((0, 0, -4.9 * age * age))
                alive = age > 0 and (pos.z > 0 or age < 0.05)
                o.location = (pos.x, pos.y, max(pos.z, 0.0005))
                o.scale = (r, r, r) if alive or (age > 0 and pos.z <= 0) else (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
    keys(range(0, KEY_END + 1), k)


def lights():
    # A pond-rink floodlight on a pole, low and far behind the skaters: warm and hard,
    # so the skater shadows are long and sharp.
    flood = bpy.data.lights.new("Flood", "SPOT")
    flood.color = (1.0, 0.9, 0.76)
    flood.energy = 4.0e4
    flood.spot_size = math.radians(40)
    flood.shadow_soft_size = 0.05
    fo = link("Flood", flood)
    fo.location = (-2.0, 9.0, 3.2)        # low: long, sharp skater shadows across the frame
    # Scenes 1-2 seen from above: a cool-neutral flood keeps the ice blue-black, as it
    # reads in Scenes 3-4 (warm, it turned the dust and the pond floor grey-brown).
    # The low shot keeps the warm flood from the cut.
    for f, c in ((S3_A - 1, S1_FLOOD), (S3_A, (1.0, 0.9, 0.76))):
        flood.color = c
        flood.keyframe_insert("color", frame=f)
    # From the low cut (Scene 3) it is dim: its pool on the far ice, out of focus,
    # read as a bright band of water or steam.
    e3 = flood.energy * 0.85
    # It lit the near ice as brightly as the crest: 0.11, not 0.25 (v7 review: foreground
    # about 1 stop too bright).
    for f, e in ((S2_A + 6, flood.energy), (S3_A - 1, e3), (S3_A, e3 * 0.11)):
        flood.energy = e
        flood.keyframe_insert("energy", frame=f)
    flood.energy = 4.0e4
    aim(fo, (0, 0, 0))
    fo.visible_glossy = False
    # A second, distant rink light whose mirror image sits on the ice in frame.
    pole = bpy.data.lights.new("PoleGlint", "AREA")
    pole.shape, pole.size, pole.size_y = "DISK", 0.6, 0.6
    pole.color = (1.0, 0.88, 0.72)
    pole.energy = 3200                               # 1.5 stops under the first cut: it clipped top centre (v7 review)
    po = link("PoleGlint", pole)
    po.location = (0.25, 3.6, 2.4)
    aim(po, (0, 0, 0))
    po.visible_diffuse = False                       # only seen as a reflection
    # Lower in Scene 2 the camera would see it as a big glint streak: fade it out.
    for f, e in ((S2_A, 3200.0), (S2_A + 12, 0.0)):
        pole.energy = e
        pole.keyframe_insert("energy", frame=f)
    # Light linking: the glint lights the ice only. On the glossy crest it washed out
    # every colour.
    only_ice = bpy.data.collections.new("GlintReceivers")
    only_ice.objects.link(bpy.data.objects["Ice"])
    po.light_linking.receiver_collection = only_ice
    # Night sky fill: cool and dim.
    w = bpy.data.worlds.new("Night")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.002, 0.0025, 0.004, 1)
    # Reflections in the ice see a few distant rink lights and a dark sky.
    nt = w.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.0, (0.05, 0.04, 0.032, 1)     # valley light glow low on the horizon
    e[1].position, e[1].color = 0.45, (0.002, 0.0025, 0.004, 1)
    sky = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(ramp.outputs["Color"], sky.inputs["Color"])
    # What the ice mirrors (glossy rays only, so it does not light the scene): an
    # overcast night sky lit from below by the valley, a dark treeline at the horizon,
    # and a few warm lights in the trees.
    gen = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], gen.inputs[0])
    refl = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(gen.outputs["Z"], refl.inputs["Fac"])
    r = refl.color_ramp.elements
    r[0].position, r[0].color = 0.0, (0.02, 0.02, 0.02, 1)
    r[1].position, r[1].color = 0.9, (0.05, 0.055, 0.065, 1)
    mid = r.new(0.12)
    mid.color = (0.16, 0.14, 0.12, 1)                      # glow low over the valley
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Generated"], noise.inputs["Vector"])
    trees = nt.nodes.new("ShaderNodeMath")                 # treeline: ragged band near the horizon
    trees.operation = "LESS_THAN"
    zs = nt.nodes.new("ShaderNodeMath")
    zs.operation = "MULTIPLY_ADD"
    nt.links.new(noise.outputs["Fac"], zs.inputs[0])
    zs.inputs[1].default_value = 0.12
    zs.inputs[2].default_value = 0.0
    nt.links.new(gen.outputs["Z"], trees.inputs[0])
    nt.links.new(zs.outputs[0], trees.inputs[1])
    dark = nt.nodes.new("ShaderNodeMix")
    dark.data_type = "RGBA"
    nt.links.new(trees.outputs[0], dark.inputs[0])
    nt.links.new(refl.outputs["Color"], dark.inputs[6])
    dark.inputs[7].default_value = (0.004, 0.004, 0.005, 1)
    mirror = nt.nodes.new("ShaderNodeBackground")
    if HDRI and os.path.isfile(HDRI):
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(HDRI)
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, 0, math.radians(HDRI_ROT))
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs[0], env.inputs["Vector"])
        tint = nt.nodes.new("ShaderNodeMix")
        tint.data_type = "RGBA"
        tint.blend_type = "MULTIPLY"
        tint.inputs[0].default_value = 1.0
        nt.links.new(env.outputs["Color"], tint.inputs[6])
        tint.inputs[7].default_value = MIRROR_TINT
        nt.links.new(tint.outputs[2], mirror.inputs["Color"])
        mirror.inputs["Strength"].default_value = HDRI_STRENGTH
    else:
        nt.links.new(dark.outputs[2], mirror.inputs["Color"])
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Glossy Ray"], mix.inputs["Fac"])
    nt.links.new(sky.outputs[0], mix.inputs[1])
    nt.links.new(mirror.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["World Output"].inputs["Surface"])


def camera(scene, args):
    data = bpy.data.cameras.new("Camera")
    data.sensor_fit = "HORIZONTAL"
    data.lens = 50 if not args.vertical else 35
    data.clip_start = 0.01
    data.dof.use_dof = True
    data.dof.aperture_fstop = 2.8
    cam = link("Camera", data)
    scene.camera = cam
    target = Vector((0.0, -0.03, -0.05))

    def k(f):
        # Locked over the crest, about 60 degrees down, with a very slow creep in.
        # It carries on as one continuous shot into Scene 2 (angle A): it keeps
        # creeping in and steepens, so there is no jump cut at frame 34.
        d = lerp(1.62, 1.5, smooth(f / S1_END))
        el = math.radians(CAM_ELEV)
        tgt = target
        focus = Vector((0, 0.05, -0.03))                                           # in the bubble layer
        if f > S1_END:
            # Scenes 1-2 are one unbroken shot: a slow, steady descent to a lower angle
            # (whole crest always in frame) while it heats; the first cut is the
            # breakthrough.
            t = smooth((f - S1_END) / (S2_END - S1_END))
            d = lerp(1.5, 1.38, t)
            el = math.radians(lerp(CAM_ELEV, 40, t))
            tgt = target.lerp(Vector((0.0, 0.0, CREST_BASE + 0.03)), t)
            focus = focus.lerp(Vector((0.0, 0.0, CREST_BASE + 0.03)), t)           # onto the heating crest
        cam.location = tgt + Vector((0, -math.cos(el), math.sin(el))) * d
        aim(cam, tgt)
        data.dof.focus_distance = (focus - cam.location).length
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        data.dof.keyframe_insert("focus_distance", frame=f)
    keys(range(0, KEY_END + 1), k)


def bind_scene1_camera(scene):
    """The Scene 1 camera carries on through Scene 2 (no cut at 34 or 56) up to the
    Scene 3 cut. Bound to a timeline marker, as the Scene 3 camera is."""
    mk = scene.timeline_markers.new("S1", frame=0)
    mk.camera = scene.camera


def crush_puff(peak):
    """Crushed ice puffs out of the crack at the second shove (60)."""
    rng = np.random.default_rng(60)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    proto = bpy.context.object
    me = proto.data
    m = Mat("Crushed")
    m.bsdf.inputs["Base Color"].default_value = (0.9, 0.93, 0.97, 1)
    m.bsdf.inputs["Roughness"].default_value = 0.4
    me.materials.append(m.mat)
    bpy.data.objects.remove(proto)
    items = []
    for i in range(45):
        t0 = rng.choice([60, 72]) + rng.uniform(0, 2)
        v = Vector((rng.normal(0, 0.25), rng.normal(-0.15, 0.25), rng.uniform(0.3, 0.9)))
        items.append((link(f"Crushed{i}", me), t0, Vector((peak.x, peak.y, 0.001)), v, rng.uniform(0.0006, 0.002)))

    def k(f):
        for o, t0, p0, v, r in items:
            age = (f - t0) / FPS
            p = p0 + v * age + Vector((0, 0, -4.9 * age * age))
            o.location = (p.x, p.y, max(p.z, 0.0008))
            o.scale = (r, r, r) if age >= 0 else (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
    keys(range(0, KEY_END + 1), k)


# ---------------------------------------------------------------- scene 3 'Breakthrough'

def in_poly(p, poly):
    """True when the point (x, y) is inside the polygon (even-odd rule)."""
    x, y, inside, j = p[0], p[1], False, len(poly) - 1
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def clip_half(poly, n, c):
    """The part of the polygon where n . p <= c (one Sutherland-Hodgman pass)."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i - 1], poly[i]
        da, db = n[0] * a[0] + n[1] * a[1] - c, n[0] * b[0] + n[1] * b[1] - c
        if (da > 0) != (db > 0):
            t = da / (da - db)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        if db <= 0:
            out.append(b)
    return out


def prism_mesh(name, poly, z0, z1, to_local=lambda v: v):
    """A vertical prism over a counter-clockwise polygon, from z0 up to z1."""
    n = len(poly)
    verts = [to_local((x, y, z1)) for x, y in poly] + [to_local((x, y, z0)) for x, y in poly]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    faces += [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    return me


def show_from(obj, f0, show=True):
    """Key render visibility: hidden before f0 and shown from f0 (or the reverse)."""
    for f in (0, f0 - 1, f0):
        obj.hide_render = (f < f0) == show
        obj.keyframe_insert("hide_render", frame=f)
    for f in range(f0 + 1, KEY_END + 1):        # override any later keys of its own
        if obj.animation_data and obj.animation_data.action:
            obj.hide_render = not show
            obj.keyframe_insert("hide_render", frame=f)


def breakthrough(scene, args, rig, crack_paths):
    """Scene 3 (78-100), the first cut: the cracked ice over the crest breaks into plates
    that lift, tilt and slide away; a hockey-stop fan of ice chips and snow sprays off
    one side; steam comes off the hot metal; and the crest rises through the hole and
    turns to the camera. The shot carries on into Scene 4 with no cut."""
    rng = np.random.default_rng(78)
    # The hole: an uneven oval a little larger than the crest.
    hole = []
    for i in range(40):
        a = 2 * math.pi * i / 40
        r = 1 + 0.08 * math.sin(3 * a + 0.7) + 0.05 * math.sin(7 * a + 2.1) + rng.normal(0, 0.025)
        hole.append((0.52 * r * math.cos(a), 0.02 + 0.44 * r * math.sin(a)))
    # A sheet pushed up from below fails in radial wedges (pie slices) that hinge up and
    # out, around a small crushed zone over the middle of the load (the crest's centre).
    # Where a heat crack reaches out towards the rim, a wedge break runs out through its
    # end; wide gaps get extra radial breaks. Cracks that do not become breaks stay on
    # their plate. The plates are 25-45 cm across, so with the 12 cm thickness they read
    # as slabs, not blocks (the first pass, 10-25 cm Voronoi cells, read as crates).
    load = Vector((0.0, 0.02))
    rays = []
    for path in crack_paths:
        inside = [p.xy for p in path if in_poly(p.xy, hole)]
        if not inside:
            continue
        far = max(inside, key=lambda p: (p - load).length)
        if (far - load).length > 0.25:
            rays.append(math.atan2(far.y - load.y, far.x - load.x) % (2 * math.pi))
    rays.sort()
    merged = []                                           # cracks closer than ~30 deg make one break
    for a in rays:
        if merged and a - merged[-1] < 0.55:
            continue
        merged.append(a)
    if len(merged) > 1 and merged[0] + 2 * math.pi - merged[-1] < 0.55:
        merged.pop()
    if len(merged) < 2:
        merged = [k * 2 * math.pi / 7 for k in range(7)]
    bounds = []
    for k, a in enumerate(merged):
        b = merged[(k + 1) % len(merged)] + (2 * math.pi if k == len(merged) - 1 else 0)
        n_split = max(1, math.ceil((b - a) / 1.05))
        bounds += [a + (b - a) * (j + (rng.uniform(-0.12, 0.12) if j else 0)) / n_split for j in range(n_split)]
    print(f"[fracture] load {tuple(round(v, 3) for v in load)}: {len(rays)} radial cracks, "
          f"{len(bounds)} wedges at {[round(math.degrees(a)) for a in bounds]}")
    # The crushed zone: a small convex polygon with a corner on every radial break.
    hub = [(load.x + r * math.cos(a), load.y + r * math.sin(a))
           for a, r in ((a, rng.uniform(0.1, 0.14)) for a in bounds)]
    cells = []
    for k, a in enumerate(bounds):
        b = bounds[(k + 1) % len(bounds)]
        poly = clip_half(list(hole), (math.sin(a), -math.cos(a)), math.sin(a) * load.x - math.cos(a) * load.y)
        poly = clip_half(poly, (-math.sin(b), math.cos(b)), -math.sin(b) * load.x + math.cos(b) * load.y)
        (px, py), (qx, qy) = hub[k], hub[(k + 1) % len(hub)]
        nrm = (py - qy, qx - px)                          # keeps the side away from the hub
        poly = clip_half(poly, nrm, nrm[0] * px + nrm[1] * py)
        cells.append(poly if len(poly) >= 3 else None)
    cells.append(list(hub))
    n_main = len(cells) - 1                               # the last cell is the crushed hub
    # Neighbouring wedges compute their shared corners separately: snap them together so
    # the jagged edges between them match exactly.
    corners = list(hub)
    for poly in cells:
        if poly is None:
            continue
        for i, p in enumerate(poly):
            near = next((c for c in corners if (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 < 1e-10), None)
            if near is None:
                corners.append(p)
            else:
                poly[i] = near
    flat_cells = [None if p is None else list(p) for p in cells]

    # Real fracture edges are jagged, not straight: each inner edge gets a zigzag that
    # is the same seen from both plates (seeded by the edge itself).
    def on_hole(a, b):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        for k in range(len(hole)):
            (px, py), (qx, qy) = hole[k - 1], hole[k]
            dx, dy = qx - px, qy - py
            t = max(0.0, min(1.0, ((mx - px) * dx + (my - py) * dy) / (dx * dx + dy * dy)))
            if (px + t * dx - mx) ** 2 + (py + t * dy - my) ** 2 < 1e-10:
                return True
        return False

    def jag(a, b):
        if on_hole(a, b):
            return []
        ka, kb = (round(a[0], 5), round(a[1], 5)), (round(b[0], 5), round(b[1], 5))
        lo, hi = (a, b) if ka <= kb else (b, a)
        r2 = np.random.default_rng(abs(hash((min(ka, kb), max(ka, kb)))) % (2 ** 32))
        dx, dy = hi[0] - lo[0], hi[1] - lo[1]
        n_len = math.hypot(dx, dy)
        if n_len < 1e-6:
            return []
        nx, ny = -dy / n_len, dx / n_len
        n = max(1, int(n_len / 0.022))
        pts = []
        for k in range(1, n + 1):
            t = (k + r2.uniform(-0.3, 0.3)) / (n + 1)
            off = r2.normal(0, 0.007) * (1 if k % 2 else -1)
            pts.append((lo[0] + dx * t + nx * off, lo[1] + dy * t + ny * off))
        return pts if lo is a else pts[::-1]

    for i, poly in enumerate(cells):
        if poly is not None:
            cells[i] = [q for k in range(len(poly)) for q in [poly[k]] + jag(poly[k], poly[(k + 1) % len(poly)])]

    # Some wedges are cracked through already and come apart in the throw (v8 review):
    # two break in the air, two as they slam onto the ice. Each splits across, near its
    # middle, along a jagged crack; its pieces are cells of their own (cracks, discs and
    # chips go with the piece they sit on), moving as one plate until it breaks.
    groups = {}                                           # wedge -> (outline, cells, 'air' | 'impact')
    for i, mode in WEDGE_BREAKS.items():
        poly = cells[i] if i < n_main else None
        if poly is None:
            continue
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        a = math.atan2(cy - load.y, cx - load.x) + rng.uniform(-0.45, 0.45)
        nrm = (math.cos(a), math.sin(a))
        cut = nrm[0] * cx + nrm[1] * cy
        pieces = []
        for sgn in (1, -1):
            piece = clip_half(poly, (sgn * nrm[0], sgn * nrm[1]), sgn * cut)
            on_cut = [abs(q[0] * nrm[0] + q[1] * nrm[1] - cut) < 1e-9 for q in piece]
            pieces.append([q for k in range(len(piece)) for q in [piece[k]] +
                           (jag(piece[k], piece[(k + 1) % len(piece)])
                            if on_cut[k] and on_cut[(k + 1) % len(piece)] else [])])
        if min(len(pc) for pc in pieces) < 3:
            continue
        cells[i] = pieces[0]
        cells.append(pieces[1])
        flat_cells[i] = list(pieces[0])
        flat_cells.append(list(pieces[1]))
        groups[i] = (poly, [i, len(cells) - 1], mode)

    def cell_of(x, y):
        if not in_poly((x, y), hole):
            return -1
        for i, poly in enumerate(flat_cells):
            if poly is not None and in_poly((x, y), poly):
                return i
        return min((i for i, p in enumerate(flat_cells) if p is not None),
                   key=lambda i: min((q[0] - x) ** 2 + (q[1] - y) ** 2 for q in flat_cells[i]))

    # The slab gets the hole from the cut on (a boolean, switched on at 78).
    slab = bpy.data.objects["Ice"]
    cutter = link("HoleCutter", prism_mesh("HoleCutter", hole, -ICE_T - 0.05, 0.05))
    cutter.hide_render = True
    cutter.display_type = "WIRE"
    # The hole's wall: invisible while the plates still fill the hole (seen from inside
    # the ice, a wall touching the plates reflects totally and reads as a bright pane),
    # then a fracture face once the rim plates have moved off.
    hw = Mat("HoleWall")
    hw.bsdf.inputs["Base Color"].default_value = (0.9, 0.95, 1.0, 1)
    hw.bsdf.inputs["Transmission Weight"].default_value = 1.0
    hw.bsdf.inputs["IOR"].default_value = 1.31
    hw.bsdf.inputs["Roughness"].default_value = 0.15
    hw_mix = hw.new("ShaderNodeMixShader")
    hw_clear = hw.new("ShaderNodeBsdfTransparent")
    hw.link(hw.value([0, S3_A + 4, S3_A + 6], lambda f: 0.0 if f <= S3_A + 4 else 1.0), hw_mix.inputs["Fac"])
    hw.link(hw_clear.outputs[0], hw_mix.inputs[1])
    hw.link(hw.bsdf.outputs[0], hw_mix.inputs[2])
    hw.link(hw_mix.outputs[0], hw.out.inputs["Surface"])
    cutter.data.materials.append(hw.mat)
    mod = slab.modifiers.new("Hole", "BOOLEAN")
    mod.operation, mod.solver, mod.object = "DIFFERENCE", "EXACT", cutter
    mod.material_mode = "TRANSFER"
    for f in (0, S3_A - 1, S3_A):
        mod.show_render = f >= S3_A
        slab.keyframe_insert('modifiers["Hole"].show_render', frame=f)

    # Plates: prisms in the slab's own object space, so the ice material (frost, snow
    # dust, skate marks) carries over unchanged and moves with each plate.
    # A plate's copy is near colourless: light trapped in a small slab bounces many times
    # through its faces, and the sheet's cool Scene 3 tint (0.3, 0.37, 0.5), compounded,
    # lit the plates' sides saturated blue (v8). Neutral grey keeps the same darkness.
    ice_mat = slab.data.materials[0].copy()
    ice_mat.name = "IcePlate"
    base = ice_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
    snow_dim = base.links[0].from_node                     # off from the cut; the tint feeds it
    snow_dim.inputs[6].links[0].from_node.inputs[7].default_value = (0.42, 0.43, 0.45, 1)
    # Its blue absorption too: along a plate's length, trapped light travels 1-2 m inside
    # it, and at the sheet's density (tuned to tint deep bubbles) that turned it deep blue.
    out = next(n for n in ice_mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
    out.inputs["Volume"].links[0].from_node.inputs["Density"].default_value = 0.5

    def to_slab(v):
        return (v[0] / ICE_SIZE, v[1] / ICE_SIZE, (v[2] + ICE_T / 2) / ICE_T)

    # Fresh fracture faces of lake ice: clear, with vertical columnar grain (candle
    # ice) and a ragged white line of snow and frost along the top edge.
    em = Mat("IceBreak")
    eb = em.bsdf
    eb.inputs["Base Color"].default_value = (0.975, 0.99, 1.0, 1)   # see IcePlate
    eb.inputs["IOR"].default_value = 1.31
    grain_map = em.new("ShaderNodeMapping")
    grain_map.inputs["Scale"].default_value = (ICE_SIZE * 40, ICE_SIZE * 40, 1.5)
    em.link(em.tc.outputs["Object"], grain_map.inputs["Vector"])
    columns = em.ramp(em.noise(1.0, 4.0, 0.6, vector=grain_map.outputs[0]), 0.48, 0.75)
    esep = em.new("ShaderNodeSeparateXYZ")
    em.link(em.tc.outputs["Object"], esep.inputs[0])
    band = em.ramp(em.math("ADD", esep.outputs["Z"], em.math("MULTIPLY", em.noise(300.0, 4.0, 0.6), 0.12)), 0.8, 0.95)
    em.link(em.math("SUBTRACT", 1.0, em.math("MULTIPLY", band, 0.85)), eb.inputs["Transmission Weight"])
    em.link(em.math("MAXIMUM", em.math("ADD", 0.03, em.math("MULTIPLY", columns, 0.3)), em.math("MULTIPLY", band, 0.8)),
            eb.inputs["Roughness"])
    ebump = em.new("ShaderNodeBump")
    ebump.inputs["Distance"].default_value = 0.002
    em.link(columns, ebump.inputs["Height"])
    em.link(ebump.outputs["Normal"], eb.inputs["Normal"])
    spall_rng = np.random.default_rng(83)
    shard_pieces = []

    def split_cell(poly, r):
        """Split a cell's outline into 2-3 pieces by cuts from its middle (jagged like
        the other fracture edges)."""
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        a0 = r.uniform(0, 2 * math.pi)
        n = 2 if r.random() < 0.5 else 3
        angles = [a0 + 2 * math.pi * k / n + (r.uniform(-0.35, 0.35) if n == 3 else 0) for k in range(n)]
        pieces = []
        for k in range(n):
            d0 = (math.cos(angles[k]), math.sin(angles[k]))
            d1 = (math.cos(angles[(k + 1) % n]), math.sin(angles[(k + 1) % n]))
            if n == 2:
                nrm = (d0[1], -d0[0])
                piece = clip_half(poly, nrm, nrm[0] * cx + nrm[1] * cy)
            else:
                n0, n1 = (d0[1], -d0[0]), (-d1[1], d1[0])
                piece = clip_half(clip_half(poly, n0, n0[0] * cx + n0[1] * cy), n1, n1[0] * cx + n1[1] * cy)
            if len(piece) >= 3:
                pieces.append([q for j in range(len(piece)) for q in [piece[j]] +
                               (jag(piece[j], piece[(j + 1) % len(piece)])
                                if math.dist(piece[j], piece[(j + 1) % len(piece)]) > 0.03 else [])])
        return pieces

    plates = []
    for i, poly in enumerate(cells):
        if poly is None:
            plates.append(None)
            continue
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        pivot = link(f"PlatePivot{i}", None)
        pivot.location = (cx, cy, -ICE_T / 2)
        pivot.rotation_mode = "QUATERNION"
        # The small pieces over the middle of the crest spall: a thin top layer of each
        # splits into shards that sit in the ice until the crest throws them up.
        top = 0.0
        if i == n_main:
            top = -float(spall_rng.uniform(0.008, 0.02))
            # The crushed hub's top layer breaks twice over: 4-9 shards a few cm across.
            shard_pieces += [(piece, i, top) for half in split_cell(poly, spall_rng)
                             for piece in split_cell(half, spall_rng)]
        pivot["top"] = top                             # its top face (under 0 where it spalled)
        me = prism_mesh(f"Plate{i}", poly, -ICE_T, top, to_slab)
        me.materials.append(ice_mat)
        me.materials.append(em.mat)
        for k, face in enumerate(me.polygons[2:]):   # the sides are the broken faces,
            a, b = poly[k], poly[(k + 1) % len(poly)]   # except along the hole's wall: those
            face.material_index = 0 if on_hole(a, b) else 1   # stay clear ice until they move
        plate = link(f"Plate{i}", me)
        plate.location = (0, 0, -ICE_T / 2)
        plate.scale = (ICE_SIZE, ICE_SIZE, ICE_T)
        plate.parent = pivot
        plate.matrix_parent_inverse = Matrix.Translation(Vector(pivot.location)).inverted()
        plate.visible_shadow = False
        show_from(plate, S3_A)
        plates.append(pivot)

    shards = []
    for j, (piece, cell, top) in enumerate(shard_pieces):
        me = prism_mesh(f"Shard{j}", piece, top, 0.0, to_slab)
        me.materials.append(ice_mat)
        me.materials.append(em.mat)
        me.polygons[1].material_index = 1             # the underside is the spall face
        for k, face in enumerate(me.polygons[2:]):
            a, b = piece[k], piece[(k + 1) % len(piece)]
            face.material_index = 0 if on_hole(a, b) else 1
        o = link(f"Shard{j}", me)
        o.rotation_mode = "QUATERNION"
        o.visible_shadow = False
        show_from(o, S3_A)
        cx = sum(p[0] for p in piece) / len(piece)
        cy = sum(p[1] for p in piece) / len(piece)
        shards.append((o, cell, Vector((cx, cy, top / 2)), -top, piece))

    def attach(obj, cell):
        pivot = plates[cell]
        if pivot is not None:
            obj.parent = pivot
            obj.matrix_parent_inverse = Matrix.Translation(Vector(pivot.location)).inverted()

    # Cracks inside the hole travel with their plates: each crack mesh is split by
    # plate (the outside part stays on the slab), swapped in at the cut.
    import bmesh
    split = [o for o in list(bpy.data.objects) if o.type == "MESH" and
             o.name.startswith(("HeatCrack", "HeatHair", "CrackSheet", "CrackSnow"))]
    for o in split:
        owner = [cell_of(*(sum((o.data.vertices[v].co for v in poly.vertices), Vector()) / len(poly.vertices)).xy)
                 for poly in o.data.polygons]
        if all(c == -1 for c in owner):
            continue
        for key in sorted(set(owner)):
            bm = bmesh.new()
            bm.from_mesh(o.data)
            bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if owner[f.index] != key], context="FACES")
            me = bpy.data.meshes.new(f"{o.name}_p{key}")
            bm.to_mesh(me)
            bm.free()
            for mt in o.data.materials:
                me.materials.append(mt)
            piece = link(f"{o.name}_p{key}", me)
            if key >= 0:
                attach(piece, key)
            near_lens = key < 0 and not o.name.startswith(("HeatCrackLine", "CrackSnow")) and \
                min(v.co.y for v in me.vertices) < -0.3
            # From the low camera a crack face near the lens reads as a pane of glass;
            # keep only its line on the surface there.
            if not near_lens:
                show_from(piece, S3_A)
            else:
                piece.hide_render = True
        show_from(o, S3_A, show=False)
    # Air discs trapped in the ice go with their plates.
    for o in list(bpy.data.objects):
        if o.name.startswith("Disc"):
            c = cell_of(o.location.x, o.location.y)
            if c >= 0:
                attach(o, c)

    # Water fills the hole, a little below the top of the ice, stirred by the crest.
    wm = Mat("HoleWater")
    wb = wm.bsdf
    wb.inputs["Base Color"].default_value = (1, 1, 1, 1)
    wb.inputs["Transmission Weight"].default_value = 1.0
    wb.inputs["IOR"].default_value = 1.33
    # Stirred and slushy while the crest rises, then open water at night: near black,
    # with sharp reflections, so it reads apart from the matte, snowy ice round it.
    wm.link(wm.value(range(S3_A - 1, KEY_END + 1), lambda f: 0.15 - 0.13 * smooth((f - S4_A + 4) / 16)),
            wb.inputs["Roughness"])
    stir = wm.new("ShaderNodeTexNoise", noise_dimensions="4D")
    stir.inputs["Scale"].default_value = 30.0
    stir.inputs["Detail"].default_value = 3.0
    wm.link(wm.tc.outputs["Object"], stir.inputs["Vector"])
    wm.link(wm.value(range(S3_A - 1, KEY_END + 1), lambda f: f * 0.03), stir.inputs["W"])
    wbump = wm.new("ShaderNodeBump")
    wbump.inputs["Distance"].default_value = 0.004
    # The water settles once the crest is out (small ripples stay, from the floating plates).
    wm.link(wm.value(range(S3_A - 1, KEY_END + 1), lambda f: 1.0 - 0.75 * smooth((f - S4_A) / 20)),
            wbump.inputs["Strength"])
    wm.link(stir.outputs["Fac"], wbump.inputs["Height"])
    wm.link(wbump.outputs["Normal"], wb.inputs["Normal"])
    wv = wm.new("ShaderNodeVolumeAbsorption")
    wv.inputs["Color"].default_value = (0.12, 0.22, 0.34, 1)
    wv.inputs["Density"].default_value = 16.0
    wm.link(wv.outputs[0], wm.out.inputs["Volume"])
    inner = [(x * 0.995, 0.02 + (y - 0.02) * 0.995) for x, y in hole]
    water = link("HoleWater", prism_mesh("HoleWater", inner, -ICE_T - 0.003, -0.014))
    water.data.materials.append(wm.mat)
    show_from(water, S3_A)

    # Scene 1-2 actors that do not belong in this shot.
    gone = [o for o in bpy.data.objects if o.name.startswith(("Puck", "Boil", "Crushed", "Fleck"))]
    for name in ("SkaterA", "SkaterB", "StickBlade"):
        if name in bpy.data.objects:
            gone += [bpy.data.objects[name]] + list(bpy.data.objects[name].children_recursive)
    for o in gone:
        show_from(o, S3_A, show=False)
    # Snow dust and landed flakes over the hole would hang in the air: drop them there.
    specks = [o for o in bpy.data.objects if o.name.startswith(("Dust", "Flake")) and o.animation_data]
    for f in range(S3_A, KEY_END + 1):
        scene.frame_set(f)
        drop = [o for o in specks if o.matrix_world.translation.z < 0.003 and in_poly(o.matrix_world.translation.xy, hole)]
        for o in drop:
            o.scale = (0, 0, 0)
            o.keyframe_insert("scale", frame=f)

    pose = crest_rise(rig)
    breaks = plates_fly(plates, cells, hole, groups)
    motions = {}                                   # cell -> (break-off frame, Motion)
    k = 0
    for i, pivot in enumerate(plates):
        if pivot is not None:
            motions[i] = (breaks[k][1], breaks[k][3])
            k += 1
    ground = Ground(hole, [b[3] for b in breaks])
    shards_fly(shards, motions, ground)
    steam(rig, pose, hole)
    spray(hole, breaks, on_hole, ground)
    splash(hole, breaks, ground, pose)
    rim_marks(hole, breaks)
    s3_snow()
    s3_dust(hole)
    s3_lights(rig, pose)
    s3_world()
    backdrop()
    s3_camera(scene, args, pose)
    return pose, hole


def crest_rise(rig):
    """The crest rises through the hole and turns to the camera (79-98), then hovers.
    Returns pose(f) -> world matrix of the crest rig."""
    from mathutils import Euler
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    m0 = rig.matrix_world.inverted()
    local = [m0 @ (o.matrix_world @ v.co) for o in rig.children if o.type == "MESH"
             for v in list(o.data.vertices)[::7]]
    dx, dy, dz, tx, ty, yaw = crest_motion(S3_A)
    loc0 = Vector((dx, dy, CREST_BASE + dz))
    rot0 = Vector((math.radians(tx), math.radians(ty), math.radians(-8 + yaw)))
    rot1 = Vector((math.radians(94), 0.0, math.radians(-5)))      # upright, leaning back a little
    r1 = Euler(rot1).to_matrix()
    low = min((r1 @ p).z for p in local)
    loc1 = Vector((0.0, 0.02, RISE_TIP - low))

    def kick(t, freq, zeta, v0):
        """A damped spring hit at t = 0: 0 -> 1 with starting speed v0 (rise per second),
        so it bursts out at full speed, slows, overshoots a little and settles."""
        if t <= 0:
            return 0.0
        w = 2 * math.pi * freq
        wd = w * math.sqrt(1 - zeta * zeta)
        return 1 + math.exp(-zeta * w * t) * (-math.cos(wd * t) + (v0 - zeta * w) / wd * math.sin(wd * t))

    def state(f):
        if f <= S3_A:
            return loc0, rot0
        # It bursts up through the hole at ~4 m/s (86% of the way in 0.2 s), slows,
        # overshoots the hover by ~6 cm, drops back and settles; the turn to camera lags.
        # (v8 review: the eased spring, 0.5 s to the peak, did not burst out of the ice.)
        s = kick((f - (S3_A + 1)) / FPS, 1.0, 0.7, 6.0)
        u = kick((f - (S3_A + 2)) / FPS, 1.0, 0.6, 3.0)
        loc = loc0.lerp(loc1, s)
        # Hovering: a slight bob that dies away, so the end card holds still.
        loc.z += 0.004 * math.sin((f - 97) * 0.35) * smooth((f - 94) / 6) * (1 - smooth((f - S4_A) / HOLD_FROM))
        return loc, rot0.lerp(rot1, u)

    def k(f):
        loc, rot = state(f)
        rig.location, rig.rotation_euler = loc, rot
        rig.keyframe_insert("location", frame=f)
        rig.keyframe_insert("rotation_euler", frame=f)
    keys(range(S3_A, KEY_END + 1), k)
    # The heat light under the ice flares with the crest at the cut (it lights the
    # cracks and the bubbles red from below, as from above in Scene 2) and is off once
    # the crest is out.
    hl = bpy.data.lights["HeatUp"]
    for f in range(S3_A, KEY_END + 1):
        hl.energy = 3.0 * heat(f) ** 2 * 4.0 * (1 - smooth((f - S3_A) / 3)) if f < S3_A + 3 else 0.0
        hl.keyframe_insert("energy", frame=f)

    def pose(f):
        loc, rot = state(f)
        return Matrix.LocRotScale(loc, Euler(rot), None)
    pose.local = local
    # The crest surface in rig space, to place things exactly on its face.
    from mathutils.bvhtree import BVHTree
    deps = bpy.context.evaluated_depsgraph_get()
    verts, polys = [], []
    for o in rig.children:
        if o.type != "MESH":
            continue
        me = o.evaluated_get(deps).to_mesh()
        mw = rig.matrix_world.inverted() @ o.matrix_world      # both at the current frame
        base = len(verts)
        verts += [mw @ v.co for v in me.vertices]
        polys += [[base + i for i in p.vertices] for p in me.polygons]
        o.evaluated_get(deps).to_mesh_clear()
    pose.surface = BVHTree.FromPolygons(verts, polys)
    return pose


WATER_Z = -0.014                 # the water's surface in the hole
FLOOD_GLINT = 400.0              # W, the far rink light as the open water mirrors it
ICE_TOP = 0.0                    # the ice sheet's top
SLAB_DRAG = 0.4                  # 1/s, air drag on a flying plate's sideways speed
WEDGE_BREAKS = {2: "air", 7: "air", 3: "impact", 8: "impact"}   # wedges that come apart


class Motion:
    """A plate's rigid motion: carries points that sit on it (shards, crumbs, chips) until
    they break off, and is a surface that loose pieces land on. mats: keyed frame -> pivot
    world matrix; the pivot rests at c; poly is the plate's outline at rest; land is the
    frame it comes to rest (in the water or on the ice)."""

    def __init__(self, c, mats, poly=None):
        self.c, self.mats, self.poly, self.land = Vector(c), mats, poly, None
        self.radius = max(math.dist(p, (c.x, c.y)) for p in poly) if poly else 0.0
        self._inv = {}

    def matrix(self, f):
        """The plate's move from rest at (possibly fractional) frame f."""
        f = min(max(f, S3_A - 1), KEY_END)
        f0 = int(math.floor(f))
        f1 = min(f0 + 1, KEY_END)
        m0, m1 = self.mats[f0], self.mats[f1]
        t = f - f0
        loc = m0.to_translation().lerp(m1.to_translation(), t)
        rot = m0.to_quaternion().slerp(m1.to_quaternion(), t)
        return Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ Matrix.Translation(-self.c)

    def point(self, f, p):
        return self.matrix(f) @ Vector(p)

    def velocity(self, f, p, h=0.5):
        """World velocity (m/s) of a point riding on the plate."""
        return (self.point(f + h, p) - self.point(f - h, p)) * (FPS / (2 * h))

    def top(self, f, xy):
        """The height of the plate's top face over world point xy at frame f, or None if
        xy is not over the plate."""
        m = self.matrix(f)
        centre = m @ Vector((self.c.x, self.c.y, 0.0))
        if (centre.xy - Vector(xy)).length > self.radius + 0.01:
            return None
        inv = m.inverted()
        # Where the vertical line through xy meets the plate's top plane (z = 0 at rest).
        o = inv @ Vector((xy[0], xy[1], 0.0))
        dz = (inv.to_3x3() @ Vector((0, 0, 1)))
        if abs(dz.z) < 0.2:                          # nearly on edge: not a surface to rest on
            return None
        q = o - dz * (o.z / dz.z)
        if not in_poly((q.x, q.y), self.poly):
            return None
        return (m @ Vector((q.x, q.y, 0.0))).z


class Ground:
    """What a falling piece lands on: a plate (the highest one under it), the ice sheet,
    or the water in the hole. land(xy, z_bottom, f) -> (surface height, Motion or None)."""

    def __init__(self, hole, motions):
        self.hole, self.motions = hole, motions

    def under(self, xy, f, z_below):
        best, carrier = None, None
        for mo in self.motions:
            # Only a plate that is down and settled (about 0.1 s after landing) holds a
            # piece; one still flying or slapping flat would throw it off.
            if mo.land is None or f < mo.land + 3:
                continue
            t = mo.top(f, xy)
            if t is not None and t <= z_below + 0.03 and (best is None or t > best):
                best, carrier = t, mo
        base = WATER_Z if in_poly(xy, self.hole) else ICE_TOP
        if best is None or best < base:
            return base, None
        return best, carrier


def ride(carrier, f_land, f):
    """The move of a piece resting on a carrier plate since it landed (identity if none)."""
    if carrier is None:
        return Matrix.Identity(4)
    return carrier.matrix(f) @ carrier.matrix(f_land).inverted()


def plates_fly(plates, cells, hole, groups=None):
    """Every plate pops off the water for a split second (the crest hits the sheet
    hard), tips outwards, and falls back: into the hole, or onto the ice past its rim,
    with a small bounce. Plates over the crest go highest. Kept physical: a plate that
    reaches the hole's wall while still below the ice top hits it and falls back; a plate
    lands where it really is (water or ice) and keeps that floor; on the ice it lies on
    its lowest corner and falls flat; afloat it rights itself and drifts without crossing
    the rim. groups: wedge -> (outline, its piece cells, 'air' | 'impact'): the pieces
    fly as one plate, then come apart on the way up or as it slams onto the ice."""
    from mathutils import Quaternion
    rng = np.random.default_rng(79)
    # Each plate's own kick (its own generator, so the other plate traits stay as they
    # were): they all fell back together; now a few fly farther out and later, and
    # some barely lift.
    kick_rng = np.random.default_rng(81)
    g = 9.8
    groups = groups or {}
    split_rng = np.random.default_rng(85)
    entries = {}                         # cell -> (outline, launch frame, direction, Motion)
    on_ice_spots = []                    # (where, radius) of the plates down on the ice
    # The bodies that fly: a whole plate, or a cracked wedge whose pieces fly as one.
    grouped = {m for _, members, _ in groups.values() for m in members}
    bodies = []                          # (cell, outline, centre at rest, member cells, break mode)
    for i, pivot in enumerate(plates):
        if pivot is None or (i in grouped and i not in groups):
            continue
        if i in groups:
            poly, members, mode = groups[i]
            c = Vector((sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly), -ICE_T / 2))
        else:
            poly, members, mode, c = cells[i], [i], None, Vector(pivot.location)
        bodies.append((i, poly, c, members, mode))
    # Plates thrown clear go to different heights (v8 review: their arcs all peaked
    # alike): round the hole, a high throw, then a low one, and so on (apex, m).
    whole = [b for b in bodies if plates[b[0]].get("top", 0.0) >= 0]
    away = sorted((b for b in whole if b[2].y >= -0.3 * b[2].xy.length), key=lambda b: math.atan2(b[2].y - 0.02, b[2].x))
    apex = {b[0]: (0.62, 0.27, 0.5, 0.33, 0.56, 0.3, 0.44)[k % 7] for k, b in enumerate(away)}
    # One plate on the near side comes at the lens and lands in the foreground, frame
    # left of the crest (the near-left one; v8 review: no plate in the foreground).
    near = [b for b in whole if b[0] not in apex]
    fg = min(near, key=lambda b: b[2].x)[0] if near else None
    for i_plate, poly, c, members, mode in bodies:
        pivot = plates[i_plate]
        r = min(1.0, c.xy.length / 0.5)
        d = c.xy.normalized() if c.xy.length > 0.02 else Vector((rng.normal(), rng.normal())).normalized()
        push = (1 - r) ** 0.7
        vu = (0.9 + 1.4 * push) * rng.uniform(0.8, 1.25)        # 4-30 cm up, in the air 0.2-0.5 s
        vo = (0.25 + 1.1 * push) * rng.uniform(0.8, 1.2)
        w = (1.0 + 2.5 * push) * rng.uniform(0.7, 1.3)          # the inner edge lifts: they tip outwards
        spin = rng.normal(0, 1.2)
        # Afloat in the hole: a gentle bob and rock, each plate at its own rate and phase.
        bob_f, bob_ph, rock = rng.uniform(0.5, 1.1), rng.uniform(0, 6.3), rng.uniform(0.02, 0.05)
        rock_axis = Vector((rng.normal(), rng.normal(), 0)).normalized()
        # Settling differs per plate: how long it keeps turning and sliding, how far it
        # slides, and how long it teeters (some stop at once, some rock for a second).
        t_turn, t_slide, slide = rng.uniform(0.1, 1.2), rng.uniform(0.15, 1.3), rng.uniform(0.01, 0.1)
        teeter, t_teeter, f_teeter = rng.uniform(0.0, 0.08), rng.uniform(0.2, 1.2), rng.uniform(1.5, 3.5)
        float_turn, float_drift = rng.normal(0, 0.06), Vector((rng.normal(0, 0.006), rng.normal(0, 0.006)))
        # The sheet heaves first (the gaps open and the glow shows through), then the
        # plates break off from the middle outwards.
        tl = S3_A + 1.0 + 1.5 * r + rng.uniform(0, 0.8)
        # Most plates kick hard enough to clear the rim, so open water shows in the hole
        # (with 9 big wedges, the ones that fell back covered it all again).
        kick_rng.random()
        big = i_plate in apex or i_plate == fg   # thrown clear; the rest towards the lens fall back
        ku = kick_rng.uniform(1.15, 1.35) if big else kick_rng.uniform(0.6, 1.05)
        ko = kick_rng.uniform(0.9, 1.3) if big else kick_rng.uniform(0.5, 1.1)
        tl += kick_rng.uniform(0, 1.0)           # the burst throws them all within ~0.15 s
        if not big:               # towards the lens: a big kick flew across the crest's face
            ku, ko = min(ku, 0.9), min(ko, 1.0)
        vu, vo = vu * ku, vo * ko
        if i_plate in apex:       # clears the 12 cm wall before it reaches the rim; its own height
            vu = math.sqrt(2 * g * apex[i_plate])
        if i_plate == fg:         # at the lens: over the near rim and down ~0.6 m out, in front of the crest
            d, vu, vo = Vector((-0.5, -1.0)).normalized(), 3.0, 1.6
        heave_tilt = rng.uniform(0.02, 0.06)
        mats = {}
        motion = Motion(c, mats, poly)
        # Plates thrown clear must not land on one another: if this one would come down
        # on a plate already lying on the ice, it is thrown a little to one side.
        # It goes farther first, then a little to either side; failing all, the best try.
        d0, vo0, vu0 = d, vo, vu
        tries = [(d0, k) for k in (1.0, 1.25, 1.5, 1.8)]
        tries += [(Vector((math.cos(a), math.sin(a))), k) for da in (0.25, -0.25, 0.5, -0.5, 0.8, -0.8, 1.1, -1.1)
                  for k in (1.0, 1.3, 1.6)
                  for a in [math.atan2(d0.y, d0.x) + da]]
        tries = [t for t in tries if not big or t[0].y >= -0.3]
        if i_plate == fg:
            tries = [(d0, k) for k in (1.0, 1.15, 1.3)]
        # The crushed hub sits on the crest's peak, clear of the wall: the crest flips it
        # back off its face (not high) and it drops into the open water behind it.
        hub = pivot.get("top", 0.0) < 0
        if hub:
            vu0, vo0, tries = min(vu0, 2.0), 0.9, [(d0 if d0.y > 0.3 else Vector((0.0, 1.0)), 1.0)]
        best = (-1.0, tries[0])
        for n_try in range(len(tries) + 1):
            d, vk = tries[n_try] if n_try < len(tries) else best[1]
            vo = vo0 * vk
            vu = min(vu0 * (1 + 0.5 * (vk - 1)), max(vu0, 2.6))   # farther: a little higher, longer
            axis = Vector((d.y, -d.x, 0))
            corners = [Vector((p[0] - c.x, p[1] - c.y, s * ICE_T / 2)) for p in poly for s in (-1, 1)]
            lead = max(v.xy.dot(d) for v in corners)            # the plate's leading edge, outwards

            def quat(tilt, turn, sway=None):
                q = Quaternion(axis, tilt) @ Quaternion((0, 0, 1), turn)
                return q if sway is None else sway @ q

            def lowest(q):
                return min((q @ v).z for v in corners)

            # A wedge's outer edge is the hole's wall: it cannot move outwards until it has
            # risen clear of the 12 cm wall. Until then it goes straight up and only starts
            # to tip (t_c: the moment its lowest corner clears the ice top; never if the kick
            # is too weak, and then it falls back into the hole).
            # A plate thrown clear tips slowly (a fast tip drove its outer edge back into the
            # wall); one that only lifts tips fast, hinged on the wall.
            k_tip = 0.3 if big else 1.0

            def tilt_at(t, t_c):
                return -heave_tilt - w * (0.2 * min(t, t_c) + k_tip * max(0.0, t - t_c))

            def z_at(t):
                return c.z + 0.012 * (0.4 + push) + vu * t - 0.5 * g * t * t

            t_c = 0.0 if hub else math.inf
            for i in range(1, int(vu / g * 240) + 1 if not hub else 0):
                tt = i / 240
                if z_at(tt) + lowest(quat(tilt_at(tt, math.inf), spin * tt)) > ICE_TOP + 0.01:
                    t_c = tt
                    break

            def flight(t, xy_wall=None):
                # The crest's sloped face pushes it outwards over the first few frames
                # (no instant jump in speed), then air drag slows it, a little: a 10 kg
                # slab (strong drag stopped them dead over the apex: v8 review).
                tau, a_push = max(0.0, t - t_c), max(0.05, vo / 20)
                b = SLAB_DRAG + 1 / a_push
                out = vo * ((1 - math.exp(-SLAB_DRAG * tau)) / SLAB_DRAG - (1 - math.exp(-b * tau)) / b)
                xy = c.xy + d * out if xy_wall is None else xy_wall
                return xy, z_at(t), tilt_at(t, t_c), spin * t

            # Step the flight finely: find where it hits the wall (if it does) and where it lands.
            t_wall, xy_wall, landed = None, None, None
            dt = 1 / 240
            t = 0.0
            xy_prev = c.xy.copy()
            while t < (KEY_END - tl) / FPS + dt:
                xy, z, tilt, turn = flight(t, xy_wall)
                q = quat(tilt, turn)
                bottom = z + lowest(q)
                if xy_wall is None and t > t_c and not in_poly(xy + d * lead * 0.9, hole) and bottom < ICE_TOP - 0.005:
                    t_wall, xy_wall = t, xy_prev.copy()        # hits the hole's wall: falls back inside
                    xy = xy_wall
                falling = t > vu / g
                over_ice = not in_poly(xy, hole)
                if falling and over_ice and bottom <= ICE_TOP:
                    landed = (t, xy.copy(), tilt, turn, True)
                    break
                if falling and not over_ice and z <= c.z + 0.004:
                    landed = (t, xy.copy(), tilt, turn, False)
                    break
                xy_prev = xy.copy()
                t += dt
            if landed is None:
                landed = (t, xy_prev.copy(), tilt, turn, not in_poly(xy_prev, hole))
            t_land, xy0, tilt0, turn0, on_ice = landed
            gap = min([(xy0 - o_xy).length - 0.75 * (motion.radius + o_r) for o_xy, o_r in on_ice_spots] + [1.0])
            if (on_ice and gap > 0) or n_try == len(tries) or not big or hub:
                break
            score = gap + (0.5 if on_ice else 0.0)
            if score > best[0]:
                best = (score, (d, vk))
        motion.land = tl + t_land * FPS
        motion.impact = (vu - g * t_land, on_ice)          # vertical speed as it comes down (m/s)
        # Pivot height afloat (ice 0.917 g/cm3): its top over the water by 8% of its own
        # thickness (thinner where its top layer spalled off as shards).
        top_off = pivot.get("top", 0.0)
        z_float = WATER_Z + (1 - 0.917) * (ICE_T + top_off) - top_off - ICE_T / 2
        drift_xy, blocked = None, False
        frames = [f for f in range(0, KEY_END + 1) if f in (0, S3_A - 1) or f >= S3_A]

        def body_at(t):
            """The flying plate's world matrix t seconds after it broke off."""
            xy, z, tilt, turn = flight(t, xy_wall if t_wall is not None and t >= t_wall else None)
            return Matrix.Translation((xy.x, xy.y, z)) @ quat(tilt, turn).to_matrix().to_4x4()

        for f in frames:
            if f < tl:
                heave = smooth((f - S3_A + 1) / (tl - S3_A + 1))
                xy, z = c.xy, c.z + 0.012 * (0.4 + push) * heave
                q = quat(-heave_tilt * heave, 0.0)
            elif (f - tl) / FPS < t_land:
                xy, z, tilt, turn = flight((f - tl) / FPS, xy_wall if t_wall is not None and (f - tl) / FPS >= t_wall else None)
                q = quat(tilt, turn)
            else:
                tt = (f - tl) / FPS - t_land
                k = 1 - math.exp(-tt / t_slide)
                turn = turn0 + spin * t_turn * (1 - math.exp(-tt / t_turn))       # slows to a stop
                if on_ice:
                    # On the ice: slides outwards a little, slaps flat, teeters, lies still;
                    # always resting on its lowest corner.
                    xy = xy0 + d * slide * k
                    sway = Quaternion(rock_axis, teeter * math.exp(-tt / t_teeter) * math.sin(2 * math.pi * f_teeter * tt))
                    q = quat(tilt0 * math.exp(-tt / 0.08), turn, sway)
                    bounce = vu * 0.012 * math.exp(-tt / 0.07) * abs(math.sin(tt * 27))
                    z = ICE_TOP - lowest(q) + bounce
                else:
                    # Afloat: rights itself, bobs, rocks and drifts slowly; it stops at the rim.
                    settle = 1 - math.exp(-tt / 0.4)
                    cand = xy0 + d * slide * k + float_drift * tt
                    step = cand - (drift_xy if drift_xy is not None else xy0)
                    lead_dir = step.normalized() if step.length > 1e-6 else d
                    reach = max(v.xy.dot(lead_dir) for v in corners)
                    if drift_xy is None or (not blocked and in_poly(cand + lead_dir * reach * 0.9, hole)):
                        drift_xy = cand
                    else:
                        blocked = True                     # against the rim: it stays there
                    xy = drift_xy
                    sway = Quaternion(rock_axis, rock * settle * math.sin(2 * math.pi * 0.8 * bob_f * tt + 2 * bob_ph))
                    q = quat(tilt0 * math.exp(-tt / 0.35), turn + float_turn * tt, sway)
                    # Floats as ice does: 92% under, its top 1 cm over the water (just under the
                    # ice top). It comes down at its landing speed, plunges (the water washes over
                    # its top), bobs back up and settles.
                    vz_in = max(motion.impact[0], -0.5)
                    plunge = (vz_in / 7.0) * math.exp(-tt / 0.3) * math.sin(7.0 * tt)
                    z = (z_float + (c.z + 0.004 - z_float) * math.exp(-tt / 0.3) + plunge
                         + 0.005 * settle * math.sin(2 * math.pi * bob_f * tt + bob_ph))
            mats[f] = Matrix.Translation((xy.x, xy.y, z)) @ q.to_matrix().to_4x4()

        # A cracked wedge comes apart: in the air on the way up (if it flies clear of the
        # wall), or as it slams onto the ice. Before that its pieces ride it exactly.
        t_b = 0.55 * vu / g
        if mode == "air" and ((t_wall is not None and t_wall < t_b + 0.05) or t_b > t_land - 0.1):
            mode = "impact" if on_ice else None
        if mode == "impact" and not on_ice:
            mode = None
        parts = []                                         # (cell, Motion, on the ice)
        for m_cell in members:
            ck = Vector(plates[m_cell].location)
            off = Vector((ck.x - c.x, ck.y - c.y, 0.0))
            piece_poly = cells[m_cell]
            pcorners = [Vector((x - ck.x, y - ck.y, s * ICE_T / 2)) for x, y in piece_poly for s in (-1, 1)]

            def low_k(qk):
                return min((qk @ v).z for v in pcorners)

            mats_k = {f: mats[f] @ Matrix.Translation(off) for f in frames}
            pm = motion if len(members) == 1 else Motion(ck, mats_k, piece_poly)
            pm.land, pm.impact, p_on_ice = motion.land, motion.impact, on_ice
            if mode == "air":
                # It snaps on the way up: each piece leaves with the plate's speed at its own
                # middle, pushed apart by the break, tumbling on its own; then gravity only.
                h = 1 / 240
                w0 = body_at(t_b) @ Matrix.Translation(off)
                p0, r0 = w0.to_translation(), w0.to_quaternion()
                v0 = ((body_at(t_b + h) @ off) - (body_at(t_b - h) @ off)) / (2 * h)
                dq = body_at(t_b + h).to_quaternion() @ body_at(t_b - h).to_quaternion().inverted()
                ax, ang = dq.to_axis_angle()
                omega = Vector(ax) * (ang / (2 * h)) + Vector(split_rng.normal(0, 2.5, 3))
                # Pushed apart mostly sideways (across its flight): pushed back along it, the
                # inner piece fell short, into the rim.
                side = Vector((-d.y, d.x))
                apart = (r0 @ off).xy
                apart = (side if apart.dot(side) >= 0 else -side) * 0.85 + apart.normalized() * 0.15
                v0 = v0 + Vector((apart.x, apart.y, 0.0)) * split_rng.uniform(0.35, 0.6) \
                    + Vector((0, 0, split_rng.uniform(-0.1, 0.3)))

                def free(t):
                    rot = Quaternion(omega.normalized(), omega.length * t) @ r0 if omega.length > 1e-6 else r0
                    return p0 + v0 * t + Vector((0, 0, -0.5 * g * t * t)), rot

                t, dt = 0.0, 1 / 240
                f_b = tl + t_b * FPS
                while True:
                    t += dt
                    pk, rk = free(t)
                    if v0.z - g * t >= 0 and f_b + t * FPS < KEY_END:
                        continue
                    if not in_poly(pk.xy, hole):
                        if pk.z + low_k(rk) <= ICE_TOP:
                            p_on_ice = True
                            break
                    elif pk.z <= -ICE_T / 2 + 0.004:
                        p_on_ice = False
                        break
                    if f_b + t * FPS >= KEY_END:
                        p_on_ice = not in_poly(pk.xy, hole)
                        break
                t_k, p_land, r_land = t, pk, rk
                vh = Vector((v0.x, v0.y, 0.0))
                up = r_land @ Vector((0, 0, 1))
                target = (up if up.z >= 0 else -up).rotation_difference(Vector((0, 0, 1))) @ r_land
                vz_k = v0.z - g * t_k
                zf_k = WATER_Z + (1 - 0.917) * ICE_T - ICE_T / 2
                pm.land, pm.impact = f_b + t_k * FPS, (vz_k, p_on_ice)
                for f in frames:
                    if f < f_b:
                        continue
                    a = (f - f_b) / FPS
                    if a < t_k:
                        pk, rk = free(a)
                    else:
                        tt = a - t_k
                        if p_on_ice:
                            # Down on the ice: slides to a stop, falls flat (either face up).
                            rk = r_land.slerp(target, 1 - math.exp(-tt / 0.08))
                            xy = p_land.xy + vh.xy * 0.12 * (1 - math.exp(-tt / 0.12))
                            bounce = -vz_k * 0.01 * math.exp(-tt / 0.07) * abs(math.sin(tt * 27))
                            pk = Vector((xy.x, xy.y, ICE_TOP - low_k(rk) + bounce))
                        else:
                            rk = r_land.slerp(target, 1 - math.exp(-tt / 0.35))
                            xy = p_land.xy + vh.xy * 0.1 * (1 - math.exp(-tt / 0.1))
                            plunge = (max(vz_k, -0.5) / 7.0) * math.exp(-tt / 0.3) * math.sin(7.0 * tt)
                            pk = Vector((xy.x, xy.y, zf_k + (p_land.z - zf_k) * math.exp(-tt / 0.3) + plunge))
                    mats_k[f] = Matrix.Translation(pk) @ rk.to_matrix().to_4x4()
            elif mode == "impact":
                # It cracks through as it slams down: the pieces spread a few cm apart and
                # turn a little, each dropping onto its own lowest corner.
                w_land = body_at(t_land) @ Matrix.Translation(off)
                z_att, vz_k = w_land.to_translation().z, vu - g * t_land
                s_k, yaw_k = split_rng.uniform(0.04, 0.09), split_rng.uniform(-0.3, 0.3)
                for f in frames:
                    if f < motion.land:
                        continue
                    tt = (f - motion.land) / FPS
                    loc_b, q_b = mats[f].to_translation(), mats[f].to_quaternion()
                    rel = q_b @ off
                    apart = rel.xy.normalized() if rel.xy.length > 1e-4 else d
                    e = 1 - math.exp(-tt / 0.12)
                    rk = Quaternion((0, 0, 1), yaw_k * e) @ q_b
                    xy = loc_b.xy + rel.xy + apart * s_k * e
                    zk = max(ICE_TOP - low_k(rk), z_att + vz_k * tt - 0.5 * g * tt * tt)
                    mats_k[f] = Matrix.Translation((xy.x, xy.y, zk)) @ rk.to_matrix().to_4x4()
            if len(members) > 1:
                pm.mats = mats_k
            parts.append((m_cell, pm, p_on_ice))
            # Keys on the piece's pivot (the whole plate's, if it does not break).
            pv = plates[m_cell]
            pmesh = bpy.data.objects[pv.name.replace("PlatePivot", "Plate")]
            for f in frames:
                pv.location = mats_k[f].to_translation()
                pv.rotation_quaternion = mats_k[f].to_quaternion()
                pv.keyframe_insert("location", frame=f)
                pv.keyframe_insert("rotation_quaternion", frame=f)
                # Wet from the plunge on (it stays wet: the bobbing keeps washing it).
                pmesh["wet"] = 0.0 if p_on_ice or f < pm.land else smooth((f - pm.land) / 3)
                pmesh.keyframe_insert('["wet"]', frame=f)
            pv["born"], pv["land"] = tl, pm.land
            entries[m_cell] = (piece_poly, tl, d, pm)
        for m_cell, pm, p_on_ice in parts:
            if p_on_ice:
                xy_rest = pm.mats[KEY_END].to_translation().xy
                on_ice_spots.append((xy_rest, pm.radius))
        if mode == "air":
            print(f"[plates] wedge {i_plate} breaks in the air at {tl + t_b * FPS:.1f}: pieces land at "
                  + ", ".join(f"{pm.land:.1f} {'ice' if oi else 'water'}" for _, pm, oi in parts))
    return [entries[i] for i in range(len(plates)) if i in entries]


def spray(hole, breaks, on_hole, ground):
    """A hockey-stop fan of ice chips and snow off the right side of the burst: wide,
    fast and sustained, with a snow mist in it; the biggest motion in the scene. Also
    the crumbs that break off the plate edges as each plate goes. Solid chips and crumbs
    are crushed ice that sits in the cracks from the cut, rides with its plate, and
    leaves with the plate's speed; none appears or shrinks away. The snow powder blows
    off the snow on the ice and disperses."""
    rng = np.random.default_rng(80)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
    proto = bpy.context.object
    chip = proto.data
    for v in chip.vertices:
        v.co *= rng.uniform(0.6, 1.3)
    cm = Mat("SprayChip")
    cm.bsdf.inputs["Base Color"].default_value = (0.9, 0.94, 0.98, 1)
    cm.bsdf.inputs["Transmission Weight"].default_value = 0.6
    cm.bsdf.inputs["IOR"].default_value = 1.31
    cm.bsdf.inputs["Roughness"].default_value = 0.4    # frosty crumbs, not glassy glints
    chip.materials.append(cm.mat)
    bpy.data.objects.remove(proto)
    powder = bpy.data.objects["Flake0"].data if "Flake0" in bpy.data.objects else chip
    items = []

    def rim(a):
        """A point on the hole's edge at angle a, and the outward direction."""
        return Vector((0.53 * math.cos(a), 0.02 + 0.45 * math.sin(a), 0.01)), a

    # The chips sit in the cracks between the outer plates, just under the surface (on the
    # hole's outline itself they drew a white ring before the break).
    def outer(c):
        return math.hypot(c.x / 0.52, (c.y - 0.02) / 0.44) > 0.55

    rim_edges = [(Vector(poly[a - 1]), Vector(poly[a]), motion) for poly, tl, d, motion in breaks
                 if outer(motion.c) for a in range(len(poly)) if not on_hole(poly[a - 1], poly[a])]
    rim_len = np.array([(e1 - e0).length for e0, e1, _ in rim_edges])
    rim_w = rim_len / rim_len.sum()

    def launch(motion, seat, born, v):
        """Where a seated piece leaves its plate, and its speed (its own plus the plate's)."""
        return motion.point(born, seat), v + motion.velocity(born, seat)

    for i in range(4400):
        is_chip = i < 260
        el = math.radians(rng.uniform(15, 60) if is_chip else rng.uniform(10, 50))
        sp = rng.uniform(1.4, 3.2) if is_chip else rng.uniform(1.6, 3.2)   # powder stays near the hole
        born = S3_A + 0.5 + (11.0 if is_chip else 7.0) * rng.random() ** 1.5
        r = rng.uniform(0.005, 0.016) if is_chip else rng.uniform(0.002, 0.006)
        drag, grav = (1.2, 9.8) if is_chip else (2.2, 4.0)
        if is_chip:
            # A chip of crushed ice in a rim crack: all the way round, thinner towards the lens.
            while True:
                e0, e1, motion = rim_edges[int(rng.choice(len(rim_edges), p=rim_w))]
                q = e0.lerp(e1, rng.random())
                a = math.atan2(q.y - 0.02, q.x)
                if math.hypot(q.x / 0.52, (q.y - 0.02) / 0.44) > 0.9:   # radial breaks run out to the
                    continue                                          # hole's outline: not on it
                if math.sin(a) >= -0.4 or rng.random() >= 0.7:
                    break
            az = a + rng.normal(0, 0.35)
            v = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))) * sp
            born = min(born, motion.land - 0.5)          # it can only be thrown before its plate is down
            seat = Vector((q.x, q.y, -0.5 * r - 0.001))
            p0, v = launch(motion, seat, born, v)
            o = link(f"Spray{i}", chip)
            o["seat_born"] = [b[1] for b in breaks if b[3] is motion][0]
            items.append((o, p0, v, born, r, drag, grav, Vector(rng.normal(0, 8, 3)), (motion, seat)))
            continue
        a = rng.uniform(0, 2 * math.pi)                   # all the way round: the crest is near round,
        while math.sin(a) < -0.4 and rng.random() < 0.95:  # but next to none towards the lens
            a = rng.uniform(0, 2 * math.pi)
        src, out = rim(a)
        az = out + rng.normal(0, 0.35)
        v = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))) * sp
        p0 = src + Vector((rng.normal(0, 0.03), rng.normal(0, 0.03), 0))
        items.append((link(f"Spray{i}", powder), p0, v, born, r, drag, grav, Vector(rng.normal(0, 8, 3)), None))
    # Crumbs off each plate's edges as it breaks away: crushed ice in its cracks.
    for j, (poly, tl, d, motion) in enumerate(breaks):
        for n in range(12):
            a = int(rng.integers(len(poly)))
            e0, e1 = Vector(poly[a - 1]), Vector(poly[a])
            q = e0.lerp(e1, rng.random())
            v = Vector((d.x * rng.uniform(0.1, 0.6) + rng.normal(0, 0.2), d.y * rng.uniform(0.1, 0.6) + rng.normal(0, 0.2),
                        rng.uniform(0.3, 1.2)))
            born = min(tl + rng.uniform(0, 1.5), motion.land - 0.5)
            r = rng.uniform(0.002, 0.007)
            seat = Vector((q.x, q.y, -0.5 * r - 0.001))
            p0, v = launch(motion, seat, born, v)
            o = link(f"Crumb{j}_{n}", chip)
            o["seat_born"] = tl
            items.append((o, p0, v, born, r, 1.0, 9.8, Vector(rng.normal(0, 10, 3)), (motion, seat)))
    # A snow mist rides in the fan (the cloud of a hard stop).
    mist = bpy.data.objects["Steam0"].data if "Steam0" in bpy.data.objects else None
    clouds = []
    if mist is not None:
        for i in range(24):
            src, out = rim(2 * math.pi * (i + rng.uniform(0, 0.8)) / 24)
            az = out + rng.normal(0, 0.2)
            el = math.radians(rng.uniform(10, 35))
            v = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))) * rng.uniform(1.4, 2.6)
            o = link(f"SprayMist{i}", mist)
            o["fade"] = 0.0
            clouds.append((o, src + Vector((0, rng.normal(0, 0.08), 0.02)), v, S3_A + 0.5 + 8 * rng.random(),
                           rng.uniform(0.04, 0.07)))
    # A plate that slams onto the ice blasts the snow under it out sideways: a low, fast
    # ring of powder and a puff of snow mist from under its edges (v8 review: no snow
    # spray). Kept low and slower towards the lens.
    puff_rng = np.random.default_rng(87)
    for j, (poly, tl, d, motion) in enumerate(breaks):
        vz, on_ice = motion.impact
        if not on_ice or vz > -0.8:
            continue
        hit = min(1.6, -vz / 2.5)
        land = motion.land
        mid = motion.point(land + 2, (motion.c.x, motion.c.y, 0.0))
        for n in range(int(160 * hit)):
            a = int(puff_rng.integers(len(poly)))
            q = Vector(poly[a - 1]).lerp(Vector(poly[a]), puff_rng.random())
            p0 = motion.point(land + 2, (q.x, q.y, -ICE_T))
            out = Vector((p0.x - mid.x, p0.y - mid.y))
            out = out.normalized() if out.length > 1e-4 else d
            p0 = Vector((p0.x, p0.y, 0.004))
            el = math.radians(puff_rng.uniform(4, 28))
            sp = puff_rng.uniform(0.7, 1.8) * hit * (0.5 if out.y < -0.5 and p0.y < -0.5 else 1.0)
            v = Vector((out.x * math.cos(el), out.y * math.cos(el), math.sin(el))) * sp
            items.append((link(f"SprayPuff{j}_{n}", powder), p0, v, land + puff_rng.uniform(0, 1.5),
                          puff_rng.uniform(0.003, 0.007), 2.2, 4.0, Vector(puff_rng.normal(0, 8, 3)), None))
        if mist is not None:
            for n in range(3):
                a = int(puff_rng.integers(len(poly)))
                q = Vector(poly[a - 1]).lerp(Vector(poly[a]), puff_rng.random())
                p0 = motion.point(land + 2, (q.x, q.y, -ICE_T))
                out = Vector((p0.x - mid.x, p0.y - mid.y, 0.0))
                out = out.normalized() if out.length > 1e-4 else Vector((d.x, d.y, 0.0))
                v = (out + Vector((0, 0, 0.35))) * puff_rng.uniform(0.5, 0.9) * hit
                o = link(f"SprayPuffMist{j}_{n}", mist)
                o["fade"] = 0.0
                clouds.append((o, Vector((p0.x, p0.y, 0.02)), v, land + puff_rng.uniform(0, 1.0),
                               puff_rng.uniform(0.03, 0.05)))

    def fly(p0, v, drag, grav, age):
        e = (1 - math.exp(-drag * age)) / drag
        return p0 + v * e + Vector((0, 0, -0.5 * grav * age * age))

    def landing(p0, v, drag, grav):
        """When a chip first touches the ice, and its sideways speed then."""
        t = 0.0
        while t < 6.0:
            t += 1 / 240
            if fly(p0, v, drag, grav, t).z <= 0.0008:
                break
        sv = v * math.exp(-drag * t)
        return t, Vector((sv.x, sv.y, 0))

    def solid_landing(p0, v, drag, born, r):
        """Where a chip of ice comes down: on a plate (then rides it), on the ice (then slides
        to rest), or in the water (floats)."""
        t, dt, g = 0.0, 1 / 120, 9.8
        while True:
            t += dt
            p = fly(p0, v, drag, g, t)
            f = min(born + t * FPS, KEY_END)
            if v.z * math.exp(-drag * t) - g * t < 0:
                h, carrier = ground.under(p.xy, f, p.z)
                if p.z - 0.5 * r <= h or f >= KEY_END:
                    lo, hi = t - dt, t
                    for _ in range(10):
                        mid = (lo + hi) / 2
                        lo, hi = (mid, hi) if fly(p0, v, drag, g, mid).z - 0.5 * r > h else (lo, mid)
                    t = hi
                    p = fly(p0, v, drag, g, t)
                    break
        sv = v * math.exp(-drag * t)
        water = carrier is None and in_poly(p.xy, hole)
        rest = WATER_Z + 0.1 * r if water else h + 0.5 * r
        return t, p, rest, carrier, Vector((sv.x, sv.y, 0)) if (carrier is None and not water) else Vector()

    lands = [solid_landing(p0, v, drag, born, r) if grav > 5 else landing(p0, v, drag, grav) + (None,) * 3
             for _, p0, v, born, r, drag, grav, _, _ in items]
    for (o, _, _, born, _, _, grav, _, _), land in zip(items, lands):
        if grav > 5:
            o["born"], o["land"] = born, born + land[0] * FPS

    def k(f):
        # A chip that lands stops turning and slides to rest in a moment, at its own
        # time: chips that kept spinning on the ice glittered like water.
        for (o, p0, v, born, r, drag, grav, spin, seated), land in zip(items, lands):
            age = (f - born) / FPS
            if seated is not None and (age < 0 or f < S3_A):
                # In its crack, riding with the plate (from the cut on).
                motion, seat = seated
                o.location = motion.point(f, seat)
                o.rotation_euler = (0, 0, 0)
                o.scale = (r, r, r * 0.5) if f >= S3_A else (0, 0, 0)
            elif age < 0 or f < S3_A - 1:
                o.scale = (0, 0, 0)
                o.location = p0
            elif grav > 5:
                # Solid ice: flies, lands on what is under it and stays there (riding a plate,
                # sliding to rest on the ice, or afloat); keeps its size.
                t_land, p_land, rest, carrier, slide_v = land
                if age < t_land:
                    p = fly(p0, v, drag, grav, age)
                else:
                    tt = age - t_land
                    q = Vector((p_land.x, p_land.y, rest)) + slide_v * 0.25 * (1 - math.exp(-tt / 0.25)) \
                        + Vector((0, 0, (p_land.z - rest) * math.exp(-tt / 0.05)))
                    p = ride(carrier, born + t_land * FPS, f) @ q
                o.location = p
                o.rotation_euler = tuple(spin * min(age, t_land))
                o.scale = (r, r, r * 0.5)
            else:
                t_land, slide_v = land[0], land[1]
                if age < t_land:
                    p = fly(p0, v, drag, grav, age)
                else:
                    p = fly(p0, v, drag, grav, t_land) + slide_v * 0.25 * (1 - math.exp(-(age - t_land) / 0.25))
                o.location = (p.x, p.y, max(p.z, 0.0008))
                o.rotation_euler = tuple(spin * min(age, t_land))
                gone = max(0.0, 1 - (age - 0.25) / 0.3)    # powder disperses
                o.scale = tuple(x * min(1.0, gone) for x in (r, r, r * (0.6 if p.z > 0.001 else 0.3)))
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("rotation_euler", frame=f)
            o.keyframe_insert("scale", frame=f)
        for o, p0, v, born, r0 in clouds:
            age = (f - born) / FPS
            if age < 0:
                o.location, o.scale, o["fade"] = p0, (0, 0, 0), 0.0
            else:
                e = (1 - math.exp(-2.2 * age)) / 2.2
                o.location = p0 + v * e
                r = r0 + 0.45 * age
                o.scale = (r * 1.4, r, r * 0.8)
                o["fade"] = 0.9 * smooth(age / 0.08) * math.exp(-age / 0.3) * max(0.0, 1 - age / 0.45)
                if o["fade"] < 0.08 and age > 0.05:  # nearly faded: out of the render (cost, and a grey veil)
                    o.scale = (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
            o.keyframe_insert('["fade"]', frame=f)
    keys([0] + list(range(S3_A - 1, KEY_END + 1)), k)


def splash(hole, breaks, ground, pose):
    """The water a plate throws up as it falls back into the hole: a crown of drops off its
    edges, as high as the plate came down fast. Each drop leaves the water's surface at the
    plate's edge, flies, and goes back into the water (or flattens out on ice or a plate).
    Also the water the burst throws: it sheets off the crest as it bursts out (the crest
    slows harder than gravity, so the water on it flies on up) and runs off the edges of
    the flying plates."""
    rng = np.random.default_rng(83)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
    proto = bpy.context.object
    drop = proto.data
    bpy.data.objects.remove(proto)
    dm = Mat("SplashDrop")
    dm.bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    # Clear water: it reads by its glints (half-frosted drops read as pink pebbles).
    dm.bsdf.inputs["Transmission Weight"].default_value = 1.0
    dm.bsdf.inputs["IOR"].default_value = 1.33
    dm.bsdf.inputs["Roughness"].default_value = 0.02
    drop.materials.append(dm.mat)
    # Foam: the churned water round a plate that came down, breaking up as the bubbles burst.
    fm = Mat("SplashFoam")
    fm.bsdf.inputs["Base Color"].default_value = (0.92, 0.95, 0.98, 1)
    fm.bsdf.inputs["Roughness"].default_value = 0.6
    fade = fm.new("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="fade")
    patch = fm.ramp(fm.noise(55.0, detail=6.0, rough=0.65), 0.42, 0.62)
    mix = fm.new("ShaderNodeMixShader")
    fm.link(fm.math("MULTIPLY", patch, fade.outputs["Fac"]), mix.inputs["Fac"])
    fm.link(fm.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
    fm.link(fm.bsdf.outputs[0], mix.inputs[2])
    fm.link(mix.outputs[0], fm.out.inputs["Surface"])
    foams, puffs = [], []
    mist = bpy.data.objects["Steam0"].data if "Steam0" in bpy.data.objects else None
    g, drag = 9.8, 1.5
    items = []
    for j, (poly, tl, d, motion) in enumerate(breaks):
        vz, on_ice = motion.impact
        if on_ice or vz > -0.6:                    # on the ice, or settles in too gently to splash
            continue
        hit = -vz
        n = int(30 * hit)
        ring = [(x - motion.c.x, y - motion.c.y) for x, y in poly]
        me = bpy.data.meshes.new(f"SplashFoam{j}")
        me.from_pydata([(x, y, 0.0) for x, y in ring], [], [list(range(len(ring)))])
        fo = link(f"SplashFoam{j}", me)
        fo.data.materials.append(fm.mat)
        fo["fade"] = 0.0
        foams.append((fo, motion, hit))
        # The fine spray: a quick white puff off the edges that thins out in the air.
        if mist is not None:
            for i in range(2 + int(hit)):
                a = int(rng.integers(len(poly)))
                q = Vector(poly[a - 1]).lerp(Vector(poly[a]), rng.random())
                p0 = motion.point(motion.land, (q.x, q.y, 0.0))
                if not in_poly(p0.xy, hole):
                    continue
                o = link(f"SplashMist{j}_{i}", mist)
                o["fade"] = 0.0
                puffs.append((o, Vector((p0.x, p0.y, WATER_Z + 0.01)), motion.land + rng.uniform(-0.3, 1.0),
                              hit, rng.uniform(0.025, 0.04)))
        for i in range(n):
            a = int(rng.integers(len(poly)))
            q = Vector(poly[a - 1]).lerp(Vector(poly[a]), rng.random())
            p0 = motion.point(motion.land, (q.x, q.y, 0.0))
            out = (p0.xy - motion.point(motion.land, (motion.c.x, motion.c.y, 0.0)).xy)
            out = out.normalized() if out.length > 1e-4 else Vector((1, 0))
            if not in_poly(p0.xy + out * 0.01, hole):  # against the rim: no open water there
                continue
            p0 = Vector((p0.x + out.x * 0.006, p0.y + out.y * 0.006, WATER_Z))
            el = math.radians(rng.uniform(45, 82))
            sp = hit * rng.uniform(0.35, 0.85)
            v = Vector((out.x * math.cos(el), out.y * math.cos(el), math.sin(el))) * sp
            v.xy += Vector((rng.normal(0, 0.08), rng.normal(0, 0.08)))
            born = motion.land - 0.4 + 1.2 * rng.random() ** 2
            r = rng.uniform(0.0015, 0.004) if rng.random() < 0.85 else rng.uniform(0.004, 0.007)
            items.append((link(f"Splash{j}_{i}", drop), p0, v, born, r, None))
    # Water on the crest: it rides the metal out of the water, then flies off as the
    # crest slows (v8 review: no water spray). Each drop leaves with the speed of its
    # spot on the crest, plus a little scatter.
    wet_rng = np.random.default_rng(86)
    spots = pose.local
    for i in range(1600):
        lp = spots[int(wet_rng.integers(len(spots)))]
        born = S3_A + 2.0 + 6.0 * wet_rng.random() ** 1.3
        p0 = pose(born) @ lp
        if p0.z < WATER_Z + 0.01:                       # still under the water: nothing to throw
            continue
        v = (pose(born + 0.5) @ lp - pose(born - 0.5) @ lp) * FPS
        v = v * wet_rng.uniform(0.85, 1.1) + Vector(wet_rng.normal(0, 0.3, 3))
        r = wet_rng.uniform(0.002, 0.005) if wet_rng.random() < 0.75 else wet_rng.uniform(0.005, 0.009)
        items.append((link(f"SplashCrest{i}", drop), p0, v, born, r, (lambda f, lp=lp: pose(f) @ lp)))
    # The burst churns the water round the crest to white: a quick puff of spray where
    # it breaks the surface.
    if mist is not None:
        for i in range(10):
            a = 2 * math.pi * (i + wet_rng.uniform(0, 0.7)) / 10
            p0 = Vector((0.3 * math.cos(a), 0.02 + 0.16 * math.sin(a), WATER_Z + 0.01))
            o = link(f"SplashBurstMist{i}", mist)
            o["fade"] = 0.0
            puffs.append((o, p0, S3_A + 1.5 + wet_rng.uniform(0, 2.0), 2.4, wet_rng.uniform(0.03, 0.05)))
    # Water running off the edges of the plates thrown clear, as they fly.
    for j, (poly, tl, d, motion) in enumerate(breaks):
        if motion.land - tl < 8:                          # barely lifted: nothing flies off
            continue
        cxy = Vector((motion.c.x, motion.c.y))
        for i in range(30):
            a = int(wet_rng.integers(len(poly)))
            q = Vector(poly[a - 1]).lerp(Vector(poly[a]), wet_rng.random())
            seat = Vector((q.x, q.y, 0.0))
            born = tl + wet_rng.uniform(1.0, min(7.0, max(1.5, (motion.land - tl) * 0.6)))
            out = (q - cxy)
            out = out.normalized() if out.length > 1e-4 else Vector((1, 0))
            p0 = motion.point(born, seat)
            v = motion.velocity(born, seat) + (motion.matrix(born).to_3x3() @ Vector((out.x, out.y, 0.0))) * wet_rng.uniform(0.4, 0.8)
            r = wet_rng.uniform(0.0015, 0.004)
            items.append((link(f"SplashRun{j}_{i}", drop), p0, v, born, r,
                          (lambda f, motion=motion, seat=seat, tl=tl: motion.point(f, seat) if f >= tl else None)))

    def fly(p0, v, age):
        e = (1 - math.exp(-drag * age)) / drag
        return p0 + v * e + Vector((0, 0, -0.5 * g * age * age))

    def end(p0, v, born):
        """When the drop is back down, and whether it fell into open water."""
        t = 0.02
        while t < 3.0:
            t += 1 / 240
            p = fly(p0, v, t)
            if v.z * math.exp(-drag * t) - g * t < 0:
                h, carrier = ground.under(p.xy, born + t * FPS, p.z)
                if p.z <= h:
                    return t, h, carrier is None and in_poly(p.xy, hole)
        return t, WATER_Z, True

    ends = [end(p0, v, born) for _, p0, v, born, _, _ in items]

    def k(f):
        for (o, p0, v, born, r, riding), (t_end, h, water) in zip(items, ends):
            age = (f - born) / FPS
            if age < 0 and riding is not None and f >= S3_A:
                # On the crest or plate it came out with: a bead, unless still under water.
                p = riding(f)
                o.location = p if p is not None else p0
                o.rotation_euler = (0, 0, 0)
                o.scale = (r, r, r * 0.6) if p is not None and p.z > WATER_Z + 0.004 else (0, 0, 0)
            elif age < 0:
                o.location, o.scale = p0, (0, 0, 0)
            elif age < t_end:
                p = fly(p0, v, age)
                vel = v * math.exp(-drag * age) + Vector((0, 0, -g * age))
                o.location = p
                o.rotation_euler = vel.to_track_quat("Z", "Y").to_euler()
                grow = min(1.0, 0.3 + age / 0.04)          # pulled out of the surface as a bead
                stretch = 1.0 + min(2.5, vel.length * 0.9)
                o.scale = (r * grow, r * grow, r * grow * stretch)
            else:
                # Back into the water (gone at once), or flattened out on ice or a plate.
                flat = 0.0 if water else max(0.0, 1 - (age - t_end) / 0.1)
                p = fly(p0, v, t_end)
                o.location = (p.x, p.y, h + 0.0005)
                o.rotation_euler = (0, 0, 0)
                o.scale = (r * 1.6 * flat, r * 1.6 * flat, r * 0.25 * flat)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("rotation_euler", frame=f)
            o.keyframe_insert("scale", frame=f)
        for fo, motion, hit in foams:
            age = (f - motion.land) / FPS
            c = motion.point(max(f, motion.land), (motion.c.x, motion.c.y, 0.0))
            fo.location = (c.x, c.y, WATER_Z + 0.0008)
            spread = 1.05 + 0.35 * (1 - math.exp(-max(age, 0) / 0.25))
            fo.scale = (spread, spread, 1) if age >= -0.5 else (0, 0, 0)
            fo["fade"] = 0.0 if age < 0 else min(1.0, 0.5 * hit) * smooth(age / 0.06) * math.exp(-age / 0.9)
            if age > 0 and fo["fade"] < 0.03:                   # the last bubbles gone
                fo.scale = (0, 0, 0)
            fo.keyframe_insert("location", frame=f)
            fo.keyframe_insert("scale", frame=f)
            fo.keyframe_insert('["fade"]', frame=f)
        for o, p0, born, hit, r0 in puffs:
            age = (f - born) / FPS
            if age < 0:
                o.location, o.scale, o["fade"] = p0, (0, 0, 0), 0.0
            else:
                o.location = p0 + Vector((0, 0, 0.12 * hit * (1 - math.exp(-age / 0.15)) * 0.4 + 0.02 * age))
                r = r0 + 0.12 * (1 - math.exp(-age / 0.3))
                o.scale = (r * 1.3, r * 1.3, r * 0.8)
                o["fade"] = 0.55 * min(1.0, hit / 1.8) * smooth(age / 0.05) * math.exp(-age / 0.25)
                if o["fade"] < 0.04 and age > 0.05:
                    o.scale = (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
            o.keyframe_insert('["fade"]', frame=f)
    keys([0] + list(range(S3_A - 1, KEY_END + 1)), k)
    print(f"[macro] splash: {len(items)} drops from "
          f"{sum(1 for b in breaks if not b[3].impact[1] and b[3].impact[0] <= -0.6)} plates")


def rim_marks(hole, breaks):
    """What shows where the hole's edge is (the water in it stands only ~1 cm under the
    ice top, as it does in floating ice): a ragged white band of crushed ice along the
    break, and beyond it a dark, glossy film of water that spilled over the rim as the
    crest broke through and the plates splashed down."""
    rng = np.random.default_rng(84)
    cx = sum(p[0] for p in hole) / len(hole)
    cy = sum(p[1] for p in hole) / len(hole)
    first_break = min(tl for _, tl, _, _ in breaks)
    first_splash = min((b[3].land for b in breaks if not b[3].impact[1]), default=first_break + 8)

    def strip(name, w_lo, w_hi, z):
        """A band on the ice top from the hole's edge outwards, w_lo..w_hi metres wide (it
        varies round the rim); u runs round the rim, v across it (0 at the edge)."""
        n = 160
        widths = np.convolve(rng.uniform(w_lo, w_hi, n + 8), np.ones(9) / 9, mode="valid")[:n]
        verts, uvs = [], []
        for k in range(n):
            t = k / n * len(hole)
            a, b = Vector(hole[int(t) % len(hole)]), Vector(hole[(int(t) + 1) % len(hole)])
            p = a.lerp(b, t - int(t))
            out = (p - Vector((cx, cy))).normalized()
            verts += [(p.x - out.x * 0.003, p.y - out.y * 0.003, z), (p.x + out.x * widths[k], p.y + out.y * widths[k], z)]
            uvs += [(k / n, 0.0), (k / n, 1.0)]
        faces = [(2 * k, 2 * ((k + 1) % n), 2 * ((k + 1) % n) + 1, 2 * k + 1) for k in range(n)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        uv = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                uv.data[li].uv = uvs[me.loops[li].vertex_index]
        o = link(name, me)
        o["fade"] = 0.0
        o.visible_shadow = False
        return o

    def band_mat(name, edge_lo, edge_hi, setup, gaps=0.0):
        """A material that covers the band from the rim out, with a ragged outer edge, and
        is shown by the object's 'fade'."""
        m = Mat(name)
        setup(m.bsdf)
        uvn = m.new("ShaderNodeUVMap", uv_map="UVMap")
        sep = m.new("ShaderNodeSeparateXYZ")
        m.link(uvn.outputs["UV"], sep.inputs[0])
        # Ragged: fine grain on the outer edge, and gaps along the rim (crushed ice lies in
        # clumps, not a line).
        rag = m.noise(240.0, 6.0, 0.7, vector=m.tc.outputs["Object"])
        clumps = m.ramp(m.noise(45.0, 3.0, 0.6, vector=m.tc.outputs["Object"]), 0.3, 0.5)
        cover = m.ramp(m.math("ADD", sep.outputs["Y"], m.math("MULTIPLY", rag, 0.6)), edge_hi, edge_lo)
        cover = m.math("MULTIPLY", cover, m.math("ADD", 1.0 - gaps, m.math("MULTIPLY", clumps, gaps)))
        fade = m.new("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="fade")
        mix = m.new("ShaderNodeMixShader")
        m.link(m.math("MULTIPLY", cover, fade.outputs["Fac"]), mix.inputs["Fac"])
        m.link(m.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        m.link(m.bsdf.outputs[0], mix.inputs[2])
        m.link(mix.outputs[0], m.out.inputs["Surface"])
        return m.mat

    def crushed(b):
        b.inputs["Base Color"].default_value = (0.88, 0.92, 0.97, 1)
        # Fresh crushed ice is a heap of small, clean facets: smooth, each tilted its own
        # way, so the near rim glitters where a facet catches a light (v7 review: the
        # near rim was lost in the foreground).
        b.inputs["Roughness"].default_value = 0.2
        nt = b.id_data.nodes
        tco = nt.new("ShaderNodeTexCoord")
        cells = nt.new("ShaderNodeTexVoronoi")
        cells.inputs["Scale"].default_value = 250.0          # chips of about 4 mm: they survive the defocus
        b.id_data.links.new(tco.outputs["Object"], cells.inputs["Vector"])
        facets = nt.new("ShaderNodeBump")
        facets.inputs["Strength"].default_value = 1.0
        facets.inputs["Distance"].default_value = 0.002
        b.id_data.links.new(cells.outputs["Distance"], facets.inputs["Height"])
        b.id_data.links.new(facets.outputs["Normal"], b.inputs["Normal"])

    def wet(b):
        b.inputs["Base Color"].default_value = (0.02, 0.03, 0.045, 1)
        # A thin spill over snow-dusted ice is not a mirror: at 0.04 the low end-card
        # camera saw the floodlight in it as a hard, clipped bar along the near rim.
        b.inputs["Roughness"].default_value = 0.3
        b.inputs["IOR"].default_value = 1.33

    film = strip("RimWater", 0.05, 0.16, 0.0003)
    film.data.materials.append(band_mat("RimWater", 0.55, 0.8, wet, gaps=0.3))
    crush = strip("RimCrush", 0.02, 0.05, 0.0006)
    crush.data.materials.append(band_mat("RimCrush", 0.45, 0.75, crushed, gaps=0.6))

    def k(f):
        # The crushed band shows as the rim breaks; the water spills over with the first
        # plates and spreads a little as the splashes come down.
        crush["fade"] = smooth((f - first_break) / 3)
        film["fade"] = 0.6 * smooth((f - first_break - 1) / 6) + 0.4 * smooth((f - first_splash) / 8)
        for o in (crush, film):
            o.keyframe_insert('["fade"]', frame=f)
    keys([0] + list(range(S3_A - 1, KEY_END + 1)), k)
    for o in (crush, film):
        show_from(o, S3_A)


def steam(rig, pose, hole):
    """Steam off the hot metal in the cold air: thin, translucent wisps that peel off
    the crest's edges, stretch upwards as they rise fast, lean with the wind (+x, as
    the snow dust) and thin out within half a second."""
    rng = np.random.default_rng(81)
    m = Mat("Steam")
    m.nt.nodes.remove(m.bsdf)
    vol = m.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.9, 0.89, 0.88, 1)
    vol.inputs["Anisotropy"].default_value = 0.5                    # bright when backlit
    co = m.tc.outputs["Object"]
    ln = m.new("ShaderNodeVectorMath", operation="LENGTH")
    m.link(co, ln.inputs[0])
    falloff = m.math("POWER", m.math("SUBTRACT", 1.0, ln.outputs["Value"], clamp=True), 2.0)
    # Wisps: noise stretched upwards, so it streams rather than billows.
    streak = m.new("ShaderNodeMapping")
    streak.inputs["Scale"].default_value = (1.5, 1.5, 0.7)
    m.link(co, streak.inputs["Vector"])
    wn = m.new("ShaderNodeTexNoise")
    wn.inputs["Scale"].default_value = 1.4
    wn.inputs["Detail"].default_value = 0.5
    wn.inputs["Roughness"].default_value = 0.4
    wn.inputs["Distortion"].default_value = 0.6
    m.link(streak.outputs[0], wn.inputs["Vector"])
    wisps = m.ramp(wn.outputs["Fac"], 0.4, 0.85)
    fade = m.new("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="fade")
    dens = m.math("MULTIPLY", m.math("MULTIPLY", wisps, falloff), fade.outputs["Fac"])
    # Self-lit haze plus a little absorption, no scattering: scattered light in thin
    # wisps stays grainy at preview samples, and this reads the same at a glance.
    m.nt.nodes.remove(vol)
    glow = m.new("ShaderNodeEmission")
    glow.inputs["Color"].default_value = (0.8, 0.8, 0.83, 1)
    m.link(m.math("MULTIPLY", dens, 4.0), glow.inputs["Strength"])
    absorb = m.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = (0.5, 0.5, 0.52, 1)
    m.link(m.math("MULTIPLY", dens, 2.5), absorb.inputs["Density"])
    both = m.new("ShaderNodeAddShader")
    m.link(glow.outputs[0], both.inputs[0])
    m.link(absorb.outputs[0], both.inputs[1])
    m.link(both.outputs[0], m.out.inputs["Volume"])
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
    proto = bpy.context.object
    me = proto.data
    me.materials.append(m.mat)
    bpy.data.objects.remove(proto)
    items = []
    # The outline of the crest (where the metal is thinnest and the water flashes off).
    hx = max(abs(p.x) for p in pose.local)
    hy = max(abs(p.y) for p in pose.local)
    edge = [p for p in pose.local if abs(p.x) / hx + abs(p.y) / hy > 0.75]
    for i in range(100):                                  # off the crest's edges
        born = S3_A + 3 + 19 * rng.random() ** 0.9
        p0 = pose(int(born)) @ edge[int(rng.integers(len(edge)))]
        o = link(f"Steam{i}", me)
        o["fade"] = 0.0
        items.append((o, born, p0, rng.uniform(0.22, 0.45), Vector((rng.uniform(0.06, 0.16), rng.normal(0, 0.03), 0)),
                      rng.uniform(0.02, 0.04), rng.uniform(0, 6.3)))
    for i in range(100, 160):                             # off the hot water in the hole: broader, slower
        born = S3_A + 5 + 17 * rng.random() ** 0.8
        while True:
            q = Vector((rng.uniform(-0.45, 0.45), rng.uniform(-0.35, 0.4), -0.005))
            if in_poly(q.xy, hole):
                break
        o = link(f"Steam{i}", me)
        o["fade"] = 0.0
        items.append((o, born, q, rng.uniform(0.22, 0.4), Vector((rng.uniform(0.05, 0.12), rng.normal(0, 0.03), 0)),
                      rng.uniform(0.045, 0.07), rng.uniform(0, 6.3)))

    def k(f):
        for o, born, p0, up, drift, r0, ph in items:
            age = (f - born) / FPS
            if age < 0:
                o.location, o.scale, o["fade"] = p0, (0, 0, 0), 0.0
            else:
                o.location = p0 + Vector((0, 0, up * age)) + drift * age
                r = r0 + 0.09 * age
                o.scale = (r, r, r * (1.3 + 1.2 * age))                 # stretches as it rises
                o.rotation_euler = (0, 0, ph + 0.8 * age)
                o["fade"] = smooth(age / 0.1) * math.exp(-age / 0.35)
                if o["fade"] < 0.08 and age > 0.05:  # nearly faded: out of the render (cost, and a grey veil)
                    o.scale = (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
            o.keyframe_insert("rotation_euler", frame=f)
            o.keyframe_insert('["fade"]', frame=f)
    keys([0] + list(range(S3_A, KEY_END + 1)), k)


def s3_snow():
    """Snow falling through the whole of the low shot (and on into Scene 4)."""
    rng = np.random.default_rng(82)
    me = bpy.data.objects["Flake0"].data
    items = []
    for i in range(320):
        p0 = Vector((rng.uniform(-1.4, 1.4), rng.uniform(-1.1, 2.6), rng.uniform(0.0, 1.8)))
        items.append((link(f"S3Flake{i}", me), p0, rng.uniform(0.18, 0.4), rng.uniform(0, 6.3), rng.uniform(0.001, 0.0028)))

    def k(f):
        t = (f - S3_A) / FPS
        for o, p0, fall, ph, r in items:
            z = p0.z - fall * t
            tl = min(t, max(0.0, p0.z - 0.0004) / fall)   # a landed flake stays put
            o.location = (p0.x + 0.05 * math.sin(tl * 1.5 + ph) + 0.06 * tl, p0.y, max(z, 0.0004))
            o.scale = (r, r, r * 0.5) if f >= S3_A else (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
    keys([0, S3_A - 1] + list(range(S3_A, KEY_END + 1)), k)


DRIFT_FAR = 1.5     # metres: no blowing-snow sheets beyond this (they blur into smoke)


def s3_dust(hole):
    """Snow dust blowing across the ice, as in Scenes 1-2 (same wind: towards +x), seen
    low: fine grains skimming and hopping just above the surface in loose streamers."""
    rng = np.random.default_rng(86)
    me = bpy.data.objects["Flake0"].data
    items = []
    # Streamers: grains bunch along a few wavy lanes, plus a thin scatter everywhere.
    lanes = [(rng.uniform(-0.9, 4.0), rng.uniform(0.2, 0.6)) for _ in range(11)]
    for i in range(3600):
        if rng.random() < 0.7:
            ly, lw = lanes[int(rng.integers(len(lanes)))]
            y = ly + rng.normal(0, 0.06 + 0.04 * lw)
        else:
            y = rng.uniform(-0.9, 4.5)
        x = rng.uniform(-3.0, 2.5)
        speed = rng.uniform(0.25, 0.6) * (1.0 + 0.2 * max(0.0, y))
        hop = rng.uniform(0.0, 0.025) ** 1.5 * 6
        r = rng.uniform(0.002, 0.005) * (1 + 0.3 * max(0.0, y))
        items.append((link(f"S3Dust{i}", me), x, y, speed, hop, rng.uniform(0, 6.3), r))

    def k(f):
        t = wind_t(f, S3_A)
        w = wind(f)
        for o, x0, y0, sp, hop, ph, r in items:
            x = x0 + sp * t
            y = y0 + 0.03 * math.sin(t * 2.0 + ph)
            z = 0.0006 + hop * w * abs(math.sin(t * 6.0 + ph))
            o.location = (x, y, z)
            gone = f < S3_A or in_poly((x, y), hole) or w < 0.02
            # As the wind drops the grains settle and thin out.
            o.scale = (0, 0, 0) if gone else (r * (1 + 1.2 * w) * w, r * w, r * 0.4 * w)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
    keys([0, S3_A - 1] + list(range(S3_A, KEY_END + 1)), k)
    # Ground drift: low, thin sheets of blown snow streaming across the ice.
    dm = Mat("GroundDrift")
    dm.nt.nodes.remove(dm.bsdf)
    dv = dm.new("ShaderNodeVolumePrincipled")
    dv.inputs["Color"].default_value = (0.86, 0.88, 0.92, 1)
    dv.inputs["Anisotropy"].default_value = 0.2       # less glow when backlit by the flood
    dco = dm.tc.outputs["Object"]
    dlen = dm.new("ShaderNodeVectorMath", operation="LENGTH")
    dm.link(dco, dlen.inputs[0])
    dfall = dm.math("POWER", dm.math("SUBTRACT", 1.0, dlen.outputs["Value"], clamp=True), 1.5)
    dmap = dm.new("ShaderNodeMapping")
    dmap.inputs["Scale"].default_value = (0.6, 3.0, 4.0)       # streaks along the wind (object x)
    dm.link(dco, dmap.inputs["Vector"])
    dn = dm.new("ShaderNodeTexNoise")
    dn.inputs["Scale"].default_value = 2.0
    dn.inputs["Detail"].default_value = 3.0
    dn.inputs["Distortion"].default_value = 0.5
    dm.link(dmap.outputs[0], dn.inputs["Vector"])
    dfade = dm.new("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="fade")
    dm.link(dm.math("MULTIPLY", dm.math("MULTIPLY", dm.ramp(dn.outputs["Fac"], 0.48, 0.75), dfall),
                    dm.math("MULTIPLY", dfade.outputs["Fac"], 6.0)), dv.inputs["Density"])
    dm.link(dv.outputs[0], dm.out.inputs["Volume"])
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
    proto = bpy.context.object
    sheet = proto.data
    sheet.materials.append(dm.mat)
    bpy.data.objects.remove(proto)
    drifts = []
    for i in range(34):
        y = rng.uniform(-0.9, 5.0)
        o = link(f"S3Drift{i}", sheet)
        o["fade"] = 0.0
        drifts.append((o, rng.uniform(-3.5, 2.0), y, rng.uniform(0.4, 0.8), rng.uniform(0.25, 0.7) * (1 + 0.2 * y),
                       rng.uniform(0.0, 1.0)))

    def kd(f):
        t = (f - S3_A) / FPS
        for o, x0, y, sp, length, ph in drifts:
            x = x0 + sp * t
            o.location = (x, y, 0.025)
            o.scale = (length, 0.07 + 0.02 * max(0.0, y), 0.045 + 0.01 * max(0.0, y)) if f >= S3_A else (0, 0, 0)
            # Steady (a pulsing density flickered against the backlight at the far edge),
            # and the wind drops as the hero shot holds.
            # None far off: out of focus there, the sheets read as steam or smoke.
            o["fade"] = 0.0 if f < S3_A or in_poly((x, y), hole) or y > DRIFT_FAR \
                else 0.8 * (1 - smooth((f - S4_A) / 12))
            if o["fade"] < 0.02:
                o.scale = (0, 0, 0)                                  # gone: out of the render
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
            o.keyframe_insert('["fade"]', frame=f)
    keys([0, S3_A - 1] + list(range(S3_A, KEY_END + 1)), kd)


def s3_lights(rig, pose):
    """The flood is behind the crest now (rim light on the spray and steam); the hot
    crest lights what is around it; a dim cool fill from the camera side."""
    glow = bpy.data.lights.new("HeatGlow", "POINT")
    glow.color = (1.0, 0.18, 0.04)
    glow.shadow_soft_size = 0.2
    go = link("HeatGlow", glow)
    go.visible_camera = go.visible_glossy = go.visible_transmission = False
    # It lights the steam, plates and spray; the crest is lit by its own glow only.
    skip = bpy.data.collections.new("HeatGlowReceivers")
    for o in rig.children:
        skip.objects.link(o)
    go.light_linking.receiver_collection = skip
    for entry in skip.collection_objects:
        entry.light_linking.link_state = "EXCLUDE"

    def k(f):
        # It dies with the crest's own glow (as the crest cools to the kit colours it was
        # still lighting the water and the plates orange: warm speckles on the ripples).
        c = smooth((f - (S3_A + 3)) / (S3_END - S3_A - 3))
        d = smooth((f - S4_A) / 22)
        glow.energy = 0.0 if f < S3_A else 12.0 * heat(f) ** 2 * (1 - 0.6 * c) * (1 - d)
        go.location = pose(f) @ Vector((0, 0, 0.12))
        glow.keyframe_insert("energy", frame=f)
        go.keyframe_insert("location", frame=f)
    keys([0, S3_A - 1] + list(range(S3_A, KEY_END + 1)), k)
    fill = bpy.data.lights.new("S3Fill", "AREA")
    fill.size = 1.2
    fill.color = (0.6, 0.75, 1.0)
    fo = link("S3Fill", fill)
    fo.location = (-0.9, -1.8, 1.0)
    aim(fo, (0, 0, 0.35))
    for f, e in ((0, 0.0), (S3_A - 1, 0.0), (S3_A, 12.5)):
        fill.energy = e
        fill.keyframe_insert("energy", frame=f)
    # A rink light on a pole across the pond, straight behind the crest (~8 degrees up
    # from the low camera, so the crest hides it from the rise on). Its mirror image falls
    # on the open water just behind the crest's base: a warm streak, broken up by the
    # ripples. It is what makes the hole read as water and not as a pit. As with the
    # Scene 1 pole glint, only its reflection in the water is rendered.
    fo2 = bpy.data.objects["Flood"]
    head = bpy.data.lights.new("FloodGlint", "AREA")
    head.shape, head.size, head.size_y = "DISK", 2.5, 2.5          # a bank of lamps: a broad, soft image
    head.color = (1.0, 0.9, 0.76)
    ho = link("FloodGlint", head)
    ho.location = (0.3, 10.0, 1.9)
    aim(ho, (0, 0, 0))
    ho.visible_diffuse = ho.visible_camera = ho.visible_transmission = False
    wet = bpy.data.collections.new("FloodGlintReceivers")
    for name in ("HoleWater", "RimWater"):
        if name in bpy.data.objects:
            wet.objects.link(bpy.data.objects[name])
    ho.light_linking.receiver_collection = wet
    for f, e in ((0, 0.0), (S3_A - 1, 0.0), (S3_A, FLOOD_GLINT)):
        head.energy = e
        head.keyframe_insert("energy", frame=f)
    # The flood is in front of the lens now: keep it out of refractions through the plates.
    for f in (0, S3_A - 1, S3_A):
        fo2.visible_transmission = f < S3_A
        fo2.keyframe_insert("visible_transmission", frame=f)


def shards_fly(shards, motions, ground):
    """The spalled top layer over the crest: each shard sits in the ice with its plate from
    the cut, is thrown up as the crest hits (with its plate's speed plus its own kick),
    tumbles, and lands on whatever is under it (a plate, the ice, or the water), easing
    flat and then riding that surface. No shard appears, vanishes or moves on its own
    once it is down."""
    from mathutils import Euler
    rng = np.random.default_rng(84)
    rest = Matrix.Translation((0, 0, -ICE_T / 2)) @ Matrix.Diagonal((ICE_SIZE, ICE_SIZE, ICE_T, 1))
    g = 9.8
    for o, cell, g0, th, piece in shards:
        tl, motion = motions[cell]
        corners = [Vector((x - g0.x, y - g0.y, s * th / 2)) for x, y in piece for s in (-1, 1)]

        def low(q):
            return min((q @ v).z for v in corners)

        born = min(S3_A + 1.5 + 3.5 * rng.random(), tl)
        out = Vector((g0.x, g0.y - 0.02, 0))
        out = out.normalized() if out.length > 1e-3 else Vector((1, 0, 0))
        kick = out * rng.uniform(0.3, 1.3) + Vector((rng.normal(0, 0.15), rng.normal(0, 0.15), rng.uniform(1.0, 2.6)))
        spin = Vector(rng.normal(0, 14, 3))
        a_launch = motion.matrix(born)
        p_launch = a_launch @ g0
        v = motion.velocity(born, g0) + kick
        q_launch = a_launch.to_quaternion()

        def fly(t):
            return p_launch + v * t + Vector((0, 0, -0.5 * g * t * t)), Euler(tuple(spin * t)).to_quaternion() @ q_launch

        # Step the fall until the shard's lowest corner meets the surface under it, then
        # find the moment of contact exactly (no sinking into the ice on the landing frame).
        t, dt = 0.0, 1 / 120
        while True:
            t += dt
            p, q = fly(t)
            f = born + t * FPS
            if f >= KEY_END:
                h, carrier = ground.under(p.xy, KEY_END, p.z)
                break
            if v.z - g * t < 0:
                h, carrier = ground.under(p.xy, f, p.z)
                if p.z + low(q) <= h:
                    lo, hi = t - dt, t
                    for _ in range(12):
                        mid = (lo + hi) / 2
                        pm, qm = fly(mid)
                        lo, hi = (mid, hi) if pm.z + low(qm) > h else (lo, mid)
                    t = hi
                    break
        t_land, f_land = t, born + t * FPS
        p_land, q_land = fly(t_land)
        afloat = carrier is None and in_poly(p_land.xy, ground.hole)
        if afloat:
            rest_z = WATER_Z - 0.4 * th                    # afloat, mostly under the surface
            up = Vector((0, 0, 1))
        else:
            rest_z = h + th / 2
            up = (carrier.matrix(f_land).to_3x3() @ Vector((0, 0, 1))).normalized() if carrier else Vector((0, 0, 1))
        own_up = q_land @ Vector((0, 0, 1))
        q_flat = own_up.rotation_difference(up if own_up.dot(up) >= 0 else -up) @ q_land
        on_ice = carrier is None and not in_poly(p_land.xy, ground.hole)
        slide = Vector((v.x, v.y, 0)) * 0.03 if on_ice else Vector()
        landing_pos = Vector((p_land.x, p_land.y, rest_z))

        def pose(f):
            """The shard's move (world matrix applied to its rest pose) at frame f."""
            if f <= born:
                return motion.matrix(f)
            if f < f_land:
                p, q = fly((f - born) / FPS)
            else:
                tt = (f - f_land) / FPS
                k = 1 - math.exp(-tt / 0.12)
                q = q_land.slerp(q_flat, smooth(tt / 0.15))
                if afloat:                                 # sinks in a little and floats
                    z = rest_z + (p_land.z - rest_z) * math.exp(-tt / 0.06)
                else:                                      # falls flat onto its lowest corner
                    z = h - low(q)
                p = Vector((landing_pos.x, landing_pos.y, z)) + slide * k
                return ride(carrier, f_land, f) @ Matrix.Translation(p) @ q.to_matrix().to_4x4() @ Matrix.Translation(-g0)
            return Matrix.Translation(p) @ q.to_matrix().to_4x4() @ Matrix.Translation(-g0)

        o["born"], o["land"], o["seat_born"] = born, f_land, tl
        for f in [0, S3_A - 1] + list(range(S3_A, KEY_END + 1)):
            o.matrix_world = pose(f) @ rest
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("rotation_quaternion", frame=f)
            o.keyframe_insert("scale", frame=f)


# The pond: Salt Lake valley floor, about 10 km west of Mount Olympus, looking east.
VIEW = (40.665, -111.905)
DEM = os.path.expanduser("~/Library/Caches/hockey-highlight-generator/wasatch_dem.npz")
SKY_HORIZON = (0.075, 0.045, 0.032, 1)   # sodium glow of the valley, low on the horizon
SKY_MID = (0.022, 0.02, 0.03, 1)         # mauve haze
SKY_TOP = (0.004, 0.007, 0.018, 1)       # deep night blue overhead


def s3_world():
    """From the cut the world is a night sky over the valley: deep blue overhead,
    a warm sodium glow from the city low on the horizon (camera and reflections)."""
    w = bpy.context.scene.world
    nt = w.node_tree
    out = nt.nodes["World Output"]
    before = out.inputs["Surface"].links[0].from_socket
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.0, SKY_HORIZON
    e[1].position, e[1].color = 0.45, SKY_TOP
    mid = e.new(0.1)
    mid.color = SKY_MID
    # Faint overcast: soft cloud banks lit from below by the city.
    cloud = nt.nodes.new("ShaderNodeTexNoise")
    cloud.inputs["Scale"].default_value = 3.0
    cloud.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Generated"], cloud.inputs["Vector"])
    lift = nt.nodes.new("ShaderNodeMix")
    lift.data_type = "RGBA"
    lift.blend_type = "MULTIPLY"
    lift.inputs[0].default_value = 0.6
    nt.links.new(ramp.outputs["Color"], lift.inputs[6])
    nt.links.new(cloud.outputs["Color"], lift.inputs[7])
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(lift.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    on = nt.nodes.new("ShaderNodeValue")
    for f in (0, S3_A - 1, S3_A):
        on.outputs[0].default_value = 1.0 if f >= S3_A else 0.0
        on.outputs[0].keyframe_insert("default_value", frame=f)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(on.outputs[0], mix.inputs["Fac"])
    nt.links.new(before, mix.inputs[1])
    nt.links.new(bg.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])


def grid_mesh(name, X, Y, Z):
    """A mesh from a height grid (world coordinates)."""
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
    return link(name, me)


def backdrop():
    """What the low Scene 3 camera sees past the crest (all hidden before the cut):
    the real Wasatch front (public elevation data, wasatch_dem.py), moonlit snow and
    dark rock, hazy with distance; town lights along its foot; a shovelled snowbank
    round the rink; a cool moon. Kept hazy and out of focus."""
    rng = np.random.default_rng(84)
    made = []
    if os.path.isfile(DEM):
        d = np.load(DEM)
        north, south, west, east = (float(d[k]) for k in ("north", "south", "west", "east"))
        elev = d["elev"][::2, ::2]
        rows, cols = elev.shape
        lats = north + (np.arange(rows) + 0.5) / rows * (south - north)
        lons = west + (np.arange(cols) + 0.5) / cols * (east - west)
        keep = lons >= VIEW[1] - 0.01
        elev, lons = elev[:, keep], lons[keep]
        lat0, lon0 = VIEW
        # Local metres: +y = east (the camera looks east), +x = south.
        Yg = ((lons - lon0) * 111320 * math.cos(math.radians(lat0)))[None, :].repeat(rows, 0)
        Xg = ((lat0 - lats) * 110574)[:, None].repeat(len(lons), 1)
        r0, c0 = int(np.argmin(np.abs(lats - lat0))), int(np.argmin(np.abs(lons - lon0)))
        ground = float(np.median(elev[max(0, r0 - 3):r0 + 4, max(0, c0 - 3):c0 + 4]))
        Z = np.maximum(elev - ground, 0.0) + 0.2
        # The pond and the rink banks replace the terrain round the camera (it drops
        # under the pond floor there, out of sight behind the banks).
        near = (np.abs(Xg) < 120) & (Yg > -120) & (Yg < 120)
        Z = np.where(near, FLOOR_Z - 1.0, Z).astype(np.float32)
        # The crest's mountain is Twin Peaks (Broads Fork) as seen from this pond: its
        # skyline from here matches the logo's inner mountain (correlation 0.93, top 1%
        # of skyline windows). Turn the range so that summit stands right behind the
        # crest, on the line from the final camera through the crest: the crest rises
        # out of the ice in front of the mountain it was drawn from.
        dist = np.hypot(Xg, Yg)
        az = (90.0 - np.degrees(np.arctan2(-Xg, Yg))) % 360         # compass azimuth (+y east, +x south)
        window = (az > 110) & (az < 124) & (dist > 3000)
        rise = np.where(window, np.degrees(np.arctan2(Z, np.maximum(dist, 1.0))), -90.0)
        sr, sc = np.unravel_index(int(np.argmax(rise)), rise.shape)
        summit = math.atan2(float(Yg[sr, sc]), float(Xg[sr, sc]))
        target = math.atan2(1.52, -0.08)                       # final camera (0.08, -1.5) -> crest (0, 0.02)
        turn = target - summit
        ct, st = math.cos(turn), math.sin(turn)
        Xg, Yg = Xg * ct - Yg * st, Xg * st + Yg * ct
        print(f"[macro] Twin Peaks summit at azimuth {az[sr, sc]:.1f}, {dist[sr, sc] / 1000:.1f} km; range turned "
              f"{math.degrees(turn):.1f} deg")
        mt = grid_mesh("Wasatch", Xg.astype(np.float32), Yg.astype(np.float32), Z)
        m = Mat("Mountain")
        geo = m.new("ShaderNodeNewGeometry")
        nsep = m.new("ShaderNodeSeparateXYZ")
        m.link(geo.outputs["Normal"], nsep.inputs[0])
        wsep = m.new("ShaderNodeSeparateXYZ")
        m.link(geo.outputs["Position"], wsep.inputs[0])
        tex = m.noise(0.004, 14.0, 0.65, vector=geo.outputs["Position"])
        flat = m.math("MULTIPLY", m.math("SUBTRACT", m.math("ADD", nsep.outputs["Z"], m.math("MULTIPLY", tex, 0.35)), 0.8),
                      5.0, clamp=True)
        high = m.math("MULTIPLY", m.math("SUBTRACT", m.math("ADD", wsep.outputs["Z"], m.math("MULTIPLY", tex, 300)), 350),
                      0.004, clamp=True)
        snow = m.math("MAXIMUM", m.math("MULTIPLY", flat, high),
                      m.math("MULTIPLY", m.math("SUBTRACT", 1.0, m.math("MULTIPLY", wsep.outputs["Z"], 0.01), clamp=True), 0.8))
        # Rock ribs and couloirs break up the snow, and conifers speckle the lower slopes.
        ribs = m.new("ShaderNodeTexNoise", noise_type="RIDGED_MULTIFRACTAL")
        ribs.inputs["Scale"].default_value = 0.0025
        ribs.inputs["Detail"].default_value = 8.0
        m.link(geo.outputs["Position"], ribs.inputs["Vector"])
        rock = m.ramp(ribs.outputs["Fac"], 0.55, 0.85)
        trees = m.math("MULTIPLY", m.ramp(m.noise(0.05, 4.0, 0.7, vector=geo.outputs["Position"]), 0.5, 0.62),
                       m.ramp(wsep.outputs["Z"], 900, 300))
        snow = m.math("MULTIPLY", snow, m.math("SUBTRACT", 1.0, m.math("MAXIMUM", m.math("MULTIPLY", rock, 0.75), trees),
                                               clamp=True))
        relief = m.new("ShaderNodeBump")
        relief.inputs["Strength"].default_value = 0.6
        relief.inputs["Distance"].default_value = 25.0
        m.link(m.math("ADD", ribs.outputs["Fac"], m.math("MULTIPLY", m.noise(0.02, 8.0, 0.6, vector=geo.outputs["Position"]), 0.5)),
               relief.inputs["Height"])
        m.link(relief.outputs["Normal"], m.bsdf.inputs["Normal"])
        col = m.new("ShaderNodeMix", data_type="RGBA")
        m.link(snow, col.inputs[0])
        col.inputs[6].default_value = (0.02, 0.018, 0.02, 1)            # scrub oak and rock
        col.inputs[7].default_value = (0.62, 0.68, 0.82, 1)             # moonlit snow, cool
        camd = m.new("ShaderNodeCameraData")
        haze = m.math("ADD", 0.2, m.math("MULTIPLY", m.math("SUBTRACT", camd.outputs["View Distance"], 2500), 1 / 16000,
                                         clamp=True))
        # Haze veils the range with the sky's own light: it dims the surface and adds
        # the glow of the valley, rather than painting it dark.
        haze = m.math("MINIMUM", m.math("ADD", haze, 0.1), 0.8)
        hz = m.new("ShaderNodeMix", data_type="RGBA")
        m.link(haze, hz.inputs[0])
        m.link(col.outputs[2], hz.inputs[6])
        hz.inputs[7].default_value = (0.05, 0.05, 0.06, 1)
        m.link(hz.outputs[2], m.bsdf.inputs["Base Color"])
        m.bsdf.inputs["Roughness"].default_value = 0.85
        m.bsdf.inputs["Emission Color"].default_value = SKY_MID
        m.link(m.math("MULTIPLY", haze, 0.9), m.bsdf.inputs["Emission Strength"])
        mt.data.materials.append(m.mat)
        made.append(mt)
        # Town and house lights along the foot of the range and across the valley.
        lm = Mat("TownLight")
        lm.bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
        lm.bsdf.inputs["Emission Color"].default_value = (1.0, 0.62, 0.3, 1)
        lm.bsdf.inputs["Emission Strength"].default_value = 60.0
        lm2 = Mat("TownLightWhite")
        lm2.bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
        lm2.bsdf.inputs["Emission Color"].default_value = (1.0, 0.88, 0.7, 1)
        lm2.bsdf.inputs["Emission Strength"].default_value = 60.0
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0)
        proto = bpy.context.object
        warm, white = proto.data, proto.data.copy()
        warm.materials.append(lm.mat)
        white.materials.append(lm2.mat)
        bpy.data.objects.remove(proto)
        cand = np.argwhere((Z > 5) & (Z < 220) & (Yg > 1500) & (Yg < 11000) & (np.abs(Xg) < 9000))
        for i in range(min(500, len(cand))):
            rr, cc = cand[int(rng.integers(len(cand)))]
            o = link(f"TownLight{i}", warm if rng.random() < 0.7 else white)
            o.location = (float(Xg[rr, cc]) + rng.uniform(-20, 20), float(Yg[rr, cc]), float(Z[rr, cc]) + 6)
            s_ = rng.uniform(2.5, 6.0) * float(Yg[rr, cc]) / 6000
            o.scale = (s_, s_, s_)
            o.visible_shadow = False
            made.append(o)
    # The snowbank round the shovelled rink: a lumpy ridge across the back.
    xs = np.linspace(-30, 30, 300, dtype=np.float32)
    ys = np.linspace(8.0, 18.0, 70, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    lump = sum(np.sin(X * fx + ph) * a for fx, ph, a in ((0.7, 0.3, 0.08), (1.9, 1.1, 0.05), (4.3, 2.0, 0.025))) \
        + rng.normal(0, 0.01, X.shape)
    ridge = np.exp(-((Y - 10.5) / (1.3 + 0.2 * np.sin(X * 0.4))) ** 2)
    Zb = (0.7 * ridge + lump * ridge + 0.35 * np.clip((Y - 10.5) / 4, 0, 1)).astype(np.float32)
    bank = grid_mesh("SnowBank", X, Y, Zb)
    sm = Mat("Bank")
    sm.bsdf.inputs["Base Color"].default_value = (0.66, 0.7, 0.8, 1)
    sm.bsdf.inputs["Roughness"].default_value = 0.8
    sb = sm.new("ShaderNodeBump")
    sb.inputs["Distance"].default_value = 0.02
    sm.link(sm.noise(8.0, 8.0, 0.65), sb.inputs["Height"])
    sm.link(sb.outputs["Normal"], sm.bsdf.inputs["Normal"])
    bank.data.materials.append(sm.mat)
    made.append(bank)
    # A puck on the ice in front of the crest gives the shot its scale. (A net behind
    # the crest was tried and dropped: it pulled attention from the action.)
    if "Puck" in bpy.data.objects:
        src = bpy.data.objects["Puck"]
        pk = src.copy()
        pk.animation_data_clear()
        bpy.context.collection.objects.link(pk)
        pk.name = "PuckS3"
        pk.hide_render = False
        p0 = Vector((0.32, -0.52, PUCK_H / 2))
        dirn = Vector((0.86, -0.5, 0)).normalized()

        def puck_k(f):
            # The burst knocks it: a small hop, then it slides away across the ice,
            # spinning, and slows on the snow dust.
            t = max(0.0, (f - (S3_A + 4)) / FPS)
            v0, dec = 0.7, 0.9
            t_stop = v0 / dec
            tt = min(t, t_stop)
            pos = p0 + dirn * (v0 * tt - 0.5 * dec * tt * tt)
            hop = 0.018 * math.sin(min(t, 0.12) / 0.12 * math.pi) if t < 0.12 else 0.0
            pk.location = (pos.x, pos.y, PUCK_H / 2 + hop)
            wob = 0.18 * math.exp(-t / 0.15) * math.sin(t * 40) if t > 0 else 0.0
            pk.rotation_euler = (wob, 0.0, 0.6 + 9.0 * (tt - 0.5 * tt * tt / t_stop))
            pk.keyframe_insert("location", frame=f)
            pk.keyframe_insert("rotation_euler", frame=f)
        keys(range(S3_A, KEY_END + 1), puck_k)
        made.append(pk)

    # A low cool moon from the south-east, behind the camera's left.
    moon = bpy.data.lights.new("Moon", "SUN")
    moon.color = (0.62, 0.74, 1.0)
    moon.angle = math.radians(0.6)
    mo = link("Moon", moon)
    mo.rotation_euler = (Vector((0, 0, 0)) - Vector((-50, -60, 35))).to_track_quat("-Z", "Y").to_euler()
    for f, e in ((0, 0.0), (S3_A - 1, 0.0), (S3_A, 0.35)):
        moon.energy = e * 0.25
        moon.keyframe_insert("energy", frame=f)
    # Moonlight on the range only (light linking): the snow on the peaks reads without
    # the moon flattening the pond and the crest.
    if "Wasatch" in bpy.data.objects:
        rm = bpy.data.lights.new("MoonRange", "SUN")
        rm.color = (0.62, 0.74, 1.0)
        rm.angle = math.radians(0.6)
        rmo = link("MoonRange", rm)
        # Low from the south: it rakes across the west face, so ridges and gullies read.
        rmo.rotation_euler = (Matrix.Rotation(turn, 3, "Z") @ (Vector((0, 0, 0)) - Vector((60, 25, 18)))) \
            .to_track_quat("-Z", "Y").to_euler()
        only = bpy.data.collections.new("RangeOnly")
        only.objects.link(bpy.data.objects["Wasatch"])
        # The rink's snowbank too: unlit, its face read as a dark shoreline behind
        # bright ice, like the edge of open water.
        if "SnowBank" in bpy.data.objects:
            only.objects.link(bpy.data.objects["SnowBank"])
        rmo.light_linking.receiver_collection = only
        for f, e in ((0, 0.0), (S3_A - 1, 0.0), (S3_A, 2.0)):
            rm.energy = e
            rm.keyframe_insert("energy", frame=f)
    for o in made:
        show_from(o, S3_A)


def s3_camera(scene, args, pose):
    """Low across the ice, about 12 degrees up, cut in at 78. It cranes up a little as
    the crest rises, ending with the ice in the bottom quarter (the matchup rail)."""
    data = bpy.data.cameras.new("CamS3")
    data.sensor_fit = "HORIZONTAL"
    data.lens = 20 if args.vertical else 30
    data.clip_start = 0.01
    data.clip_end = 60000.0                  # the Wasatch is ~9-25 km away
    data.dof.use_dof = True
    data.dof.aperture_fstop = 2.8
    cam = link("CamS3", data)

    def k(f):
        s = smooth((f - (S3_A + 1)) / 13)                 # it reacts to the burst, then holds
        loc = Vector((0.10, -1.38, 0.3)).lerp(Vector((0.08, -1.5, 0.32)), s)
        tgt = Vector((0.0, 0.05, 0.02)).lerp(Vector((0.0, 0.0, 0.36)), s)
        # Scene 4 (no cut): a slow push-in with a slight arc; the aim lifts so the
        # crest stays clear of the bottom quarter (the matchup rail).
        u = smooth((f - S4_A) / HOLD_FROM)          # it settles, then holds for the end card
        # It also cranes down to 18 cm: from that low the crest stands clear above the
        # Wasatch ridge, which crossed the lettering from 33 cm (v7 review).
        loc = loc.lerp(Vector((0.15, -1.36, 0.18)), u)
        tgt = tgt.lerp(Vector((0.0, 0.0, S4_AIM_Z)), u)
        cam.location = loc
        aim(cam, tgt)
        data.dof.focus_distance = (pose(f) @ Vector((0, 0, 0.01)) - cam.location).length
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        data.dof.keyframe_insert("focus_distance", frame=f)
    keys(range(S3_A, KEY_END + 1), k)
    mk = scene.timeline_markers.new("S3", frame=S3_A)
    mk.camera = cam


# ---------------------------------------------------------------- scene 4 'Hero'

def hero(pose, hole):
    """Scene 4 (story frames 100-164), the same shot with no cut: the crest hovers and cools to the
    red kit (heat keys in crest()); the camera pushes in slowly (s3_camera); the last
    steam peels off as it cools; and four snowflakes land on the face and fizzle in a
    tiny flash and puff of steam (FIZZLES). The bottom quarter stays clear."""
    rng = np.random.default_rng(100)
    steam_me = bpy.data.objects["Steam0"].data
    flake_me = bpy.data.objects["Flake0"].data
    # The last steam, thinning as the metal cools.
    hx = max(abs(p.x) for p in pose.local)
    hy = max(abs(p.y) for p in pose.local)
    edge = [p for p in pose.local if abs(p.x) / hx + abs(p.y) / hy > 0.75]
    wisps = []
    for i in range(50):
        born = S4_A + 26 * rng.random() ** 1.4
        o = link(f"S4Steam{i}", steam_me)
        o["fade"] = 0.0
        wisps.append((o, born, pose(int(born)) @ edge[int(rng.integers(len(edge)))], rng.uniform(0.2, 0.4),
                      Vector((rng.uniform(0.06, 0.14), rng.normal(0, 0.03), 0)), rng.uniform(0.015, 0.03),
                      max(0.0, 1 - (born - S4_A) / 28)))
    # Fizzles: a flake drifts down onto the front face, flashes and is gone in a puff.
    front = max(p.z for p in pose.local)
    face = [p for p in pose.local if p.z > front - 0.004]
    # Hits spread over the face (fractions of the half-width/height), off the centre line,
    # each hotter than the next: the first hisses, the last only melts to a bead.
    spots = [(-0.55, 0.22), (0.58, -0.08), (0.24, 0.5), (-0.34, -0.38)]
    hotness = [1.0, 0.7, 0.45, 0.22]
    flash_m = Mat("Fizzle")
    flash_m.bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    flash_m.bsdf.inputs["Emission Color"].default_value = (1.0, 0.55, 0.3, 1)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
    proto = bpy.context.object
    flash_me = proto.data
    flash_me.materials.append(flash_m.mat)
    bpy.data.objects.remove(proto)
    fizz = []
    for n, (t_hit, (u, v), h) in enumerate(zip(FIZZLES, spots, hotness)):
        # Straight onto the face at (u, v): the first surface seen from the front (the
        # nearest-vertex pick snapped two hits onto the raised centre peak).
        u, v = u * hx, v * hy
        local, _, _, _ = pose.surface.ray_cast(Vector((u, v, front + 0.05)), Vector((0, 0, -1)))
        if local is None:
            local = min(face, key=lambda p: (p.x - u) ** 2 + (p.y - v) ** 2)
        hit = pose(t_hit) @ (local + Vector((0, 0, 0.002)))
        print(f"[macro] fizzle {n} at frame {t_hit}: face ({local.x:.3f}, {local.y:.3f}, {local.z:.3f})")
        flake = link(f"FizzleFlake{n}", flake_me)
        flash = link(f"FizzleFlash{n}", flash_me)
        flash.visible_shadow = False
        flash["glow"] = 0.0
        puffs = []
        for j in range(3):
            o = link(f"FizzlePuff{n}_{j}", steam_me)
            o["fade"] = 0.0
            puffs.append((o, Vector((rng.normal(0, 0.006), rng.normal(0, 0.004), 0)), rng.uniform(0.1, 0.18)))
        # Blown in on the wind (towards +x, as the snow), each on its own slant, with sway and tumble.
        back = Vector((-rng.uniform(0.1, 0.22), -rng.uniform(0.05, 0.14), rng.uniform(0.18, 0.32)))
        sway = (rng.uniform(0.03, 0.05), rng.uniform(5.0, 8.0), rng.uniform(0, 6.3))
        spin = Vector((rng.uniform(2, 5), rng.uniform(2, 5), rng.uniform(3, 6)))
        fizz.append((t_hit, hit, flake, flash, puffs, back, sway, spin, h, rng.uniform(1.0, 1.4)))
    # The flash glows through an object property (one material for all four).
    attr = flash_m.new("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name="glow")
    flash_m.link(attr.outputs["Fac"], flash_m.bsdf.inputs["Emission Strength"])

    def k(f):
        for o, born, p0, up, drift, r0, keep in wisps:
            age = (f - born) / FPS
            if age < 0:
                o.location, o.scale, o["fade"] = p0, (0, 0, 0), 0.0
            else:
                o.location = p0 + Vector((0, 0, up * age)) + drift * age
                r = r0 + 0.09 * age
                o.scale = (r, r, r * (1.3 + 1.2 * age))
                o["fade"] = keep * smooth(age / 0.1) * math.exp(-age / 0.35)
                if o["fade"] < 0.08 and age > 0.05:  # nearly faded: out of the render (cost, and a grey veil)
                    o.scale = (0, 0, 0)
            o.keyframe_insert("location", frame=f)
            o.keyframe_insert("scale", frame=f)
            o.keyframe_insert('["fade"]', frame=f)
        for t_hit, hit, flake, flash, puffs, back, sway, spin, h, fly in fizz:
            dt = (f - t_hit) / FPS
            melt = 0.03 + 0.35 * (1 - h)                    # a cooler face melts it slowly to a bead
            if dt < -fly or dt > melt:                      # in the air for the last second or so
                flake.scale = (0, 0, 0)
                flake.location = hit
            elif dt < 0:
                amp, freq, ph = sway
                a = amp * min(1.0, -dt / 0.3)               # the sway settles as it lands
                flake.location = hit + back * (-dt) + Vector((a * math.sin(freq * dt + ph), 0.4 * a * math.cos(0.7 * freq * dt + ph), 0))
                r = 0.007                                   # a big, in-focus hero flake
                flake.scale = (r, r, r * 0.4)
            else:
                flake.location = hit
                r = 0.0035 * (1 - dt / melt)
                flake.scale = (r, r, r * 0.8)
            flake.rotation_euler = (0.4 + spin.x * dt, 0.3 + spin.y * dt, spin.z * dt) if dt < 0 else (0.4, 0.3, 0)
            if f < S4_A - 1:                                # first seen in the air at the cut
                flake.scale = (0, 0, 0)
            flash.location = hit
            # A sharp flash over about 3 frames, shrinking as it dies (no lingering disc);
            # smaller and dimmer as the metal cools.
            live = 0.0 <= dt < 0.1
            g = 90.0 * h * math.exp(-dt / 0.03) if live else 0.0
            flash["glow"] = g
            fs = 0.009 * (0.4 + 0.6 * h) * (1 - dt / 0.1) if live else 0.0
            flash.scale = (fs, fs, fs * 0.3)
            for o, off, up in puffs:
                if dt < 0:
                    o.location, o.scale, o["fade"] = hit, (0, 0, 0), 0.0
                else:
                    o.location = hit + off + Vector((0.08 * dt, -0.02, up * dt))
                    r = (0.015 + 0.1 * dt) * (0.5 + 0.5 * h)
                    o.scale = (r, r, r * 1.5)
                    o["fade"] = 2.5 * h * smooth(dt / 0.05) * math.exp(-dt / 0.35)
                    if o["fade"] < 0.08 and dt > 0.05:  # nearly faded: out of the render (cost, and a grey veil)
                        o.scale = (0, 0, 0)
                o.keyframe_insert("location", frame=f)
                o.keyframe_insert("scale", frame=f)
                o.keyframe_insert('["fade"]', frame=f)
            for o in (flake, flash):
                o.keyframe_insert("location", frame=f)
                o.keyframe_insert("scale", frame=f)
            flake.keyframe_insert("rotation_euler", frame=f)
            flash.keyframe_insert('["glow"]', frame=f)
    keys([0, S4_A - 2, S4_A - 1] + list(range(S4_A, KEY_END + 1)), k)
    # Once the glow has gone only a dim, cool fill reached the crest's face and its red
    # and white read muted: a soft, neutral key from high front-left, on the crest only
    # (light linking), comes up as the glow dies, plus a little warm rim from behind.
    rig_parts = [o for o in bpy.data.objects if o.parent is not None and o.parent.name == "Crest"]
    only = bpy.data.collections.new("CrestOnly")
    for o in rig_parts:
        only.objects.link(o)
    for name, colour, size, at, power in (("HeroKey", (1.0, 0.97, 0.93), 0.9, (-0.75, -1.25, 1.05), 1.0),
                                         ("HeroRim", (1.0, 0.85, 0.7), 0.5, (0.6, 0.9, 0.9), 0.35)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.size, ld.color = size, colour
        lo = link(name, ld)
        lo.location = at
        aim(lo, (0.0, 0.0, 0.45))
        lo.visible_camera = False             # glossy stays on: highlights on the crest; linking keeps it off the ice
        lo.light_linking.receiver_collection = only

        def kl(f, ld=ld, power=power):
            ld.energy = HERO_KEY * power * smooth((f - (S4_A + 2)) / 20)
            ld.keyframe_insert("energy", frame=f)
        keys([0, S4_A - 1] + list(range(S4_A, KEY_END + 1)), kl)


def render_settings(scene, args):
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "METAL"
    cy = scene.cycles
    cy.device = "GPU"
    cy.samples = args.samples
    cy.use_denoising = True
    cy.max_bounces = 16
    cy.transmission_bounces = 16
    cy.volume_bounces = 2
    cy.caustics_reflective = cy.caustics_refractive = False
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.25
    scene.render.fps = FPS
    scene.render.resolution_x, scene.render.resolution_y = (1080, 1920) if args.vertical else (1920, 1080)
    scene.render.resolution_percentage = args.scale
    transforms = scene.view_settings.bl_rna.properties["view_transform"].enum_items.keys()
    scene.view_settings.view_transform = "Khronos PBR Neutral" if "Khronos PBR Neutral" in transforms else "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(os.path.abspath(args.out), "frame_")
    compositor(scene)
    scene.frame_start, scene.frame_end = args.start, args.end


def compositor(scene):
    """A filmic finish: soft bloom around the hot glow and bright highlights. (Grain and
    a light vignette are added at encode time, see encode notes in the docstring.)"""
    tree = bpy.data.node_groups.new("Finish", "CompositorNodeTree") \
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
    for name, value in [("Type", "Bloom"), ("Threshold", 0.9), ("Strength", 0.35), ("Size", 0.55)]:
        sock = glare.inputs.get(name)
        if sock is not None:
            try:
                sock.default_value = value
            except TypeError:
                pass
    tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
    lens = tree.nodes.new("CompositorNodeLensdist")
    for name, value in [("Distortion", -0.01), ("Dispersion", 0.012)]:
        sock = lens.inputs.get(name)
        if sock is not None:
            sock.default_value = value
    tree.links.new(glare.outputs["Image"], lens.inputs["Image"])
    if hasattr(tree, "interface"):
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        out = tree.nodes.new("NodeGroupOutput")
        tree.links.new(lens.outputs["Image"], out.inputs[0])
    else:
        out = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(lens.outputs["Image"], out.inputs["Image"])


def build(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    scene = bpy.context.scene
    ice()
    rig, peak = crest(args)
    puck()
    skaters()
    stick_blade()
    particles(args.vertical)
    lights()
    camera(scene, args)
    if args.scene >= 2:
        boil(rig)
        crack_paths = heat_cracks(peak, rig)
        crush_puff(peak)
        bind_scene1_camera(scene)
    if args.scene >= 3:
        pose, hole = breakthrough(scene, args, rig, crack_paths)
    if args.scene >= 4:
        hero(pose, hole)
    render_settings(scene, args)
    return scene


def story(g):
    """Output frame -> (story frame, story frames per output frame)."""
    for (g0, s0), (g1, s1) in zip(RETIME, RETIME[1:]):
        if g0 == RETIME_CUT:
            continue                   # a cut has no frames between its two knots
        if g <= g1 or (g1, s1) == RETIME[-1]:
            rate = (s1 - s0) / (g1 - g0)
            return s0 + (g - g0) * rate, rate
    raise ValueError(g)


def render_retimed(scene, args):
    """Each output frame at its story time. The shutter scales with the rate, so the
    motion blur stays the same length on screen in slow motion."""
    shutter = scene.render.motion_blur_shutter
    for g in range(args.start, args.end + 1):
        t, rate = story(g)
        scene.frame_set(int(t), subframe=t - int(t))
        scene.render.motion_blur_shutter = shutter * rate
        scene.render.filepath = os.path.join(os.path.abspath(args.out), f"frame_{g:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[macro] output frame {g}: story frame {t:.2f}")


def parse_args(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="black_ice_macro")
    p.add_argument("--crest-svg", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vertical", action="store_true")
    p.add_argument("--scene", type=int, default=4, choices=(1, 2, 3, 4), help="build through this scene")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--start", type=int, default=0, help="output frame (see RETIME)")
    p.add_argument("--end", type=int, default=None, help="output frame; default: the last frame of --scene")
    a = p.parse_args(argv)
    if not os.path.isfile(a.crest_svg):
        p.error("--crest-svg must exist")
    if a.end is None:
        a.end = OUT_END[a.scene]
    return a


def main():
    args = parse_args(sys.argv)
    os.makedirs(args.out, exist_ok=True)
    scene = build(args)
    render_retimed(scene, args)
    print(f"[macro] rendered frames {args.start}-{args.end} to {args.out}")


if __name__ == "__main__":
    main()
