"""Build a team crest as a 3D badge in Blender from an SVG (spec 003, hhg-3r5.86).

Each SVG path becomes a raised layer, stacked in draw order, with bevelled
edges. Each fill colour gets an enamel-on-metal material. A kit table maps the
drawn colours to the kit's colours. Heat is physical: a blackbody glow that is
strongest on the bevels and edges, while the enamel shifts to the hot kit.

    import crest3d
    rig, heat = crest3d.build(svg_path, width=0.6, kit=crest3d.ICE_PAK_KITS["navy"],
                              hot_kit=crest3d.ICE_PAK_KITS["red"])
    heat(frame, temperature_k, glow, blend)   # key the heat at a frame
"""
import math
import os

import bpy

# Drawn colours in Final_IcePak_Logo.ai, and their colour in each kit.
# The red kit is a placeholder until the red-and-black crest file arrives.
DRAWN = {"#1D3160": "navy", "#5AA6D1": "light", "#AFB0B0": "grey", "#FFFFFF": "white"}
ICE_PAK_KITS = {
    "navy": {"navy": "#1D3160", "light": "#5AA6D1", "grey": "#AFB0B0", "white": "#FFFFFF"},
    "white": {"navy": "#1D3160", "light": "#5AA6D1", "grey": "#AFB0B0", "white": "#FFFFFF"},
    "red": {"navy": "#111111", "light": "#B0102E", "grey": "#C9CDD2", "white": "#FFFFFF"},
}
# A laser-cut sign: one flat sheet per colour run (in draw order), stacked directly
# on top of each other with no gaps.
SHEET = 0.003                # m thickness of each cut sheet
COAT_VALUE = 0.7             # real coat is never pure logo white; this keeps highlights in range
METALLIC_COAT = 0.22         # a little metal flake in the powder coat; more hides small details
BEVEL = 0.0005               # m edge break on the laser-cut edges


def _linear(hexcode):
    h = hexcode.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c) + (1.0,)


def _material(name, cold_hex, hot_hex, role, frame_obj=None):
    """Enamel on metal. Grey trim is bare metal. Heat glows hardest on the bevels."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    values = {}
    for key in ("temp", "glow", "blend", "age", "char", "rim", "hx", "hy", "face"):
        v = nt.nodes.new("ShaderNodeValue")
        v.label = key
        v.outputs[0].default_value = {"temp": 900.0, "glow": 0.0, "blend": 0.0, "age": 0.0, "char": 0.0,
                                      "rim": 0.0, "hx": 0.3, "hy": 0.3, "face": 0.2}[key]
        values[key] = v
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(values["blend"].outputs[0], mix.inputs[0])
    mix.inputs[6].default_value = _linear(cold_hex)
    mix.inputs[7].default_value = _linear(hot_hex)
    metal = role == "grey"

    # Weathering (age 0..1): faded, chipped enamel with rust blooming in the chips
    # and along the edges. It burns off as the crest heats (age is keyed down).
    def node(kind, **props):
        n = nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def math(op, a, b, clamp=False):
        n = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, x in enumerate((a, b)):
            if hasattr(x, "links"):
                nt.links.new(x, n.inputs[i])
            else:
                n.inputs[i].default_value = x
        return n.outputs[0]

    # Rim cooling (rim 0..1): the outline of the crest cools first. Towards the outer
    # edge (a diamond measure in the rig's space, hx/hy = half width/height) the glow
    # fades, the temperature drops, the char lifts and the colour goes to the hot kit.
    # rim = 0 leaves every input exactly as before.
    rc = node("ShaderNodeTexCoord")
    if frame_obj is not None:
        rc.object = frame_obj
    rsep = node("ShaderNodeSeparateXYZ")
    nt.links.new(rc.outputs["Object"], rsep.inputs[0])
    diamond = math("ADD", math("DIVIDE", math("ABSOLUTE", rsep.outputs["X"], 0.0), values["hx"].outputs[0]),
                   math("DIVIDE", math("ABSOLUTE", rsep.outputs["Y"], 0.0), values["hy"].outputs[0]))
    outer = node("ShaderNodeMapRange", clamp=True, interpolation_type="SMOOTHSTEP")
    nt.links.new(diamond, outer.inputs["Value"])
    outer.inputs["From Min"].default_value, outer.inputs["From Max"].default_value = 0.45, 0.95
    cool = math("MULTIPLY", values["rim"].outputs[0], outer.outputs[0])
    keep = math("SUBTRACT", 1.0, cool)
    glow_eff = math("MULTIPLY", values["glow"].outputs[0], keep)
    temp_eff = math("MULTIPLY", values["temp"].outputs[0], math("SUBTRACT", 1.0, math("MULTIPLY", cool, 0.25)))
    char_eff = math("MULTIPLY", values["char"].outputs[0], keep)
    blend_eff = math("ADD", values["blend"].outputs[0],
                     math("MULTIPLY", math("SUBTRACT", 1.0, values["blend"].outputs[0]), cool))
    nt.links.new(blend_eff, mix.inputs[0])

    tco = node("ShaderNodeTexCoord")
    # Every sheet is its own object; a per-object random offset gives each sheet its
    # own rust and stains, so patterns do not line up across sheets.
    info = node("ShaderNodeObjectInfo")
    jitter = node("ShaderNodeVectorMath", operation="MULTIPLY")
    nt.links.new(info.outputs["Random"], jitter.inputs[0])
    jitter.inputs[1].default_value = (37.0, 53.0, 11.0)
    shifted = node("ShaderNodeVectorMath", operation="ADD")
    nt.links.new(tco.outputs["Object"], shifted.inputs[0])
    nt.links.new(jitter.outputs[0], shifted.inputs[1])
    ocoord = shifted.outputs[0]
    blot = node("ShaderNodeTexNoise")
    blot.inputs["Scale"].default_value = 20.0
    blot.inputs["Detail"].default_value = 10.0
    blot.inputs["Roughness"].default_value = 0.7
    nt.links.new(ocoord, blot.inputs["Vector"])
    speck = node("ShaderNodeTexNoise")
    speck.inputs["Scale"].default_value = 160.0
    speck.inputs["Detail"].default_value = 4.0
    nt.links.new(ocoord, speck.inputs["Vector"])
    sepn = node("ShaderNodeSeparateXYZ")
    nt.links.new(tco.outputs["Normal"], sepn.inputs[0])
    rim = math("SUBTRACT", 1.0, math("ABSOLUTE", sepn.outputs["Z"], 0.0), clamp=True)
    chips = math("MULTIPLY", math("SUBTRACT", math("ADD", blot.outputs["Fac"], math("MULTIPLY", rim, 0.7)), 0.555), 40.0,
                 clamp=True)
    chips = math("MAXIMUM", chips, math("MULTIPLY", math("SUBTRACT", speck.outputs["Fac"], 0.68), 12.0, clamp=True))
    # A thin orange bleed just outside each chip, where rust creeps under the enamel.
    bleed = math("MULTIPLY", math("SUBTRACT", math("ADD", blot.outputs["Fac"], math("MULTIPLY", rim, 0.7)), 0.525), 12.0,
                 clamp=True)
    bleed = math("MULTIPLY", math("SUBTRACT", bleed, chips), 0.55, clamp=True)
    rust_mask = math("MULTIPLY", math("MAXIMUM", chips, bleed), values["age"].outputs[0])
    hsv = node("ShaderNodeHueSaturation")
    nt.links.new(mix.outputs[2], hsv.inputs["Color"])
    nt.links.new(math("SUBTRACT", 1.0, math("MULTIPLY", values["age"].outputs[0], 0.18)), hsv.inputs["Saturation"])
    nt.links.new(math("MULTIPLY", COAT_VALUE, math("SUBTRACT", 1.0, math("MULTIPLY", values["age"].outputs[0], 0.15))),
                 hsv.inputs["Value"])
    # Rust colour varies from dark brown to orange, in coarse and fine grain.
    grain = node("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 420.0
    grain.inputs["Detail"].default_value = 8.0
    grain.inputs["Roughness"].default_value = 0.75
    nt.links.new(ocoord, grain.inputs["Vector"])
    tone = math("ADD", math("MULTIPLY", blot.outputs["Fac"], 0.6), math("MULTIPLY", grain.outputs["Fac"], 0.6))
    rust_col = node("ShaderNodeValToRGB")
    nt.links.new(tone, rust_col.inputs["Fac"])
    els = rust_col.color_ramp.elements
    els[0].position, els[0].color = 0.35, (0.035, 0.014, 0.006, 1)
    els[1].position, els[1].color = 0.85, (0.42, 0.16, 0.04, 1)
    mid = els.new(0.6)
    mid.color = (0.16, 0.055, 0.02, 1)
    final = node("ShaderNodeMix", data_type="RGBA")
    nt.links.new(rust_mask, final.inputs[0])
    nt.links.new(hsv.outputs["Color"], final.inputs[6])
    nt.links.new(rust_col.outputs["Color"], final.inputs[7])

    # Relief: chips are recessed into the enamel with a raised, blistered lip, and
    # the rust inside them is grainy. Scaled by age, so it burns off too.
    edge_band = math("MULTIPLY", math("SUBTRACT", 1.0, math("ABSOLUTE", math("SUBTRACT", chips, 0.5), 0.0)), 1.0)
    lip = math("MULTIPLY", math("SUBTRACT", edge_band, 0.6), 2.5, clamp=True)
    relief = math("ADD", math("MULTIPLY", chips, -1.0), math("MULTIPLY", lip, 0.6))
    pits = node("ShaderNodeTexVoronoi")
    pits.inputs["Scale"].default_value = 900.0
    nt.links.new(ocoord, pits.inputs["Vector"])
    pitted = math("MULTIPLY", math("SUBTRACT", 0.18, pits.outputs["Distance"], clamp=True), 4.0)
    relief = math("ADD", relief, math("MULTIPLY", math("SUBTRACT", grain.outputs["Fac"], pitted), chips))
    relief = math("MULTIPLY", relief, values["age"].outputs[0])
    peel = node("ShaderNodeTexNoise")                       # orange-peel texture of the coating
    peel.inputs["Scale"].default_value = 700.0
    peel.inputs["Detail"].default_value = 2.0
    nt.links.new(ocoord, peel.inputs["Vector"])
    relief = math("ADD", relief, math("MULTIPLY", peel.outputs["Fac"], 0.12))
    bump = node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.9
    bump.inputs["Distance"].default_value = 0.0009
    nt.links.new(relief, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # Powder-coated metal: thick satin colour over steel, faces and bevels alike.
    # The coating wears through to bare steel on the sharpest edges; chips go
    # through to steel too, and rust blooms in and around them.
    # Wear only on the bevel line (normals between face and side); side walls stay coated.
    on_bevel = math("MULTIPLY", math("MULTIPLY", math("SUBTRACT", rim, 0.2), 5.0, clamp=True),
                    math("MULTIPLY", math("SUBTRACT", 0.96, rim), 12.0, clamp=True))
    wear = math("MULTIPLY", on_bevel, math("MULTIPLY", math("SUBTRACT", speck.outputs["Fac"], 0.38), 5.0, clamp=True))
    # The grey trim is a grey powder coat too (with a little metallic flake), not bare steel.
    bare = math("MAXIMUM", math("MULTIPLY", wear, values["age"].outputs[0]), 0.0)
    bare = math("MAXIMUM", bare, math("MULTIPLY", chips, values["age"].outputs[0]))
    rust_in_bare = math("MULTIPLY", math("SUBTRACT", math("ADD", grain.outputs["Fac"], math("MULTIPLY", blot.outputs["Fac"], 0.5)),
                                         0.66), 6.0, clamp=True)
    rust_amt = math("MAXIMUM", math("MULTIPLY", math("MULTIPLY", chips, rust_in_bare), values["age"].outputs[0]),
                    math("MULTIPLY", bleed, values["age"].outputs[0]))
    # Large-scale ageing that reads at a distance (scaled by age):
    # grime in the corners where sheets meet, and hand-sized water-stain mottling.
    ao = node("ShaderNodeAmbientOcclusion", samples=8)
    ao.inputs["Distance"].default_value = 0.025
    crevice = math("SUBTRACT", 1.0, ao.outputs["AO"], clamp=True)
    mottle = node("ShaderNodeTexNoise")
    mottle.inputs["Scale"].default_value = 4.0
    mottle.inputs["Detail"].default_value = 6.0
    mottle.inputs["Roughness"].default_value = 0.6
    nt.links.new(ocoord, mottle.inputs["Vector"])
    stain = math("MULTIPLY", math("SUBTRACT", mottle.outputs["Fac"], 0.5), 2.2)
    dark = math("ADD", math("MULTIPLY", crevice, 1.1), math("MULTIPLY", stain, 0.9))
    # All of it scales with age: the heat that burns off the rust bakes the stains off
    # too (the old 0.3 floor left grey mottling on the burnt crest's faces: v7 review).
    dark = math("MULTIPLY", dark, values["age"].outputs[0])
    grimed = node("ShaderNodeMix", data_type="RGBA")
    nt.links.new(math("MAXIMUM", dark, 0.0), grimed.inputs[0])
    nt.links.new(hsv.outputs["Color"], grimed.inputs[6])
    grimed.inputs[7].default_value = (0.05, 0.04, 0.03, 1)
    lighten = node("ShaderNodeMix", data_type="RGBA")         # mineral bloom where the stain is light
    nt.links.new(math("MULTIPLY", math("MULTIPLY", math("SUBTRACT", 0.0, stain), 0.3, clamp=True),
                      values["age"].outputs[0]), lighten.inputs[0])
    nt.links.new(grimed.outputs[2], lighten.inputs[6])
    lighten.inputs[7].default_value = (0.55, 0.56, 0.55, 1)
    coat_col = node("ShaderNodeMix", data_type="RGBA")
    nt.links.new(bare, coat_col.inputs[0])
    nt.links.new(lighten.outputs[2], coat_col.inputs[6])
    coat_col.inputs[7].default_value = (0.6, 0.61, 0.63, 1)
    rusty = node("ShaderNodeMix", data_type="RGBA")
    nt.links.new(rust_amt, rusty.inputs[0])
    nt.links.new(coat_col.outputs[2], rusty.inputs[6])
    nt.links.new(rust_col.outputs["Color"], rusty.inputs[7])
    # Heat chars the paint: hot steel goes dark with oxide scale (char 0..1).
    charred = nt.nodes.new("ShaderNodeMix")
    charred.data_type = "RGBA"
    nt.links.new(char_eff, charred.inputs[0])
    nt.links.new(rusty.outputs[2], charred.inputs[6])
    charred.inputs[7].default_value = (0.018, 0.015, 0.014, 1)
    nt.links.new(charred.outputs[2], bsdf.inputs["Base Color"])
    brushed = node("ShaderNodeTexNoise")
    brushed.inputs["Scale"].default_value = 60.0
    brushed.inputs["Detail"].default_value = 2.0
    stretch = node("ShaderNodeMapping")
    stretch.inputs["Scale"].default_value = (1.0, 40.0, 1.0)
    nt.links.new(ocoord, stretch.inputs["Vector"])
    nt.links.new(stretch.outputs[0], brushed.inputs["Vector"])
    steel_rough = math("ADD", 0.2, math("MULTIPLY", brushed.outputs["Fac"], 0.12))
    rough = node("ShaderNodeMix", data_type="FLOAT")
    nt.links.new(bare, rough.inputs[0])
    nt.links.new(math("ADD", 0.4, math("MULTIPLY", stain, 0.12)), rough.inputs[2])   # satin coat, duller where stained
    nt.links.new(steel_rough, rough.inputs[3])
    nt.links.new(math("ADD", rough.outputs[0], math("MULTIPLY", rust_amt, 0.45), clamp=True), bsdf.inputs["Roughness"])
    # Metallic powder coat: metal flake in the paint, so the colour itself reads as metal.
    flake = node("ShaderNodeTexVoronoi")
    flake.inputs["Scale"].default_value = 2500.0
    nt.links.new(ocoord, flake.inputs["Vector"])
    coat_metal = math("ADD", METALLIC_COAT, math("MULTIPLY", flake.outputs["Color"] if False else flake.outputs["Distance"], 0.25))
    nt.links.new(math("MULTIPLY", math("MAXIMUM", bare, coat_metal), math("SUBTRACT", 1.0, rust_amt)),
                 bsdf.inputs["Metallic"])
    bsdf.inputs["Coat Weight"].default_value = 0.0
    # Edge factor: faces that turn away from the badge face (the bevels and sides).
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Normal"], sep.inputs[0])
    absz = nt.nodes.new("ShaderNodeMath")
    absz.operation = "ABSOLUTE"
    nt.links.new(sep.outputs["Z"], absz.inputs[0])
    edge = nt.nodes.new("ShaderNodeMath")
    edge.operation = "SUBTRACT"
    edge.use_clamp = True
    edge.inputs[0].default_value = 1.0
    nt.links.new(absz.outputs[0], edge.inputs[1])
    weight = nt.nodes.new("ShaderNodeMath")
    weight.operation = "MULTIPLY_ADD"
    nt.links.new(edge.outputs[0], weight.inputs[0])
    weight.inputs[1].default_value = 2.5
    weight.inputs[2].default_value = 0.0
    # Uneven glow: faces glow in hot patches (thin parts heat first), edges hottest.
    hot = nt.nodes.new("ShaderNodeTexNoise")
    hot.inputs["Scale"].default_value = 9.0
    hot.inputs["Detail"].default_value = 4.0
    nt.links.new(tco.outputs["Object"], hot.inputs["Vector"])
    patches = nt.nodes.new("ShaderNodeMapRange")
    patches.inputs["From Min"].default_value, patches.inputs["From Max"].default_value = 0.4, 0.7
    # Glow concentrates on the edges; "face" sets how much the faces glow (0.2: dark).
    nt.links.new(values["face"].outputs[0], patches.inputs["To Max"])
    nt.links.new(hot.outputs["Fac"], patches.inputs["Value"])
    nt.links.new(patches.outputs[0], weight.inputs[2])
    bb = nt.nodes.new("ShaderNodeBlackbody")
    nt.links.new(temp_eff, bb.inputs["Temperature"])
    nt.links.new(bb.outputs[0], bsdf.inputs["Emission Color"])
    strength = nt.nodes.new("ShaderNodeMath")
    strength.operation = "MULTIPLY"
    nt.links.new(weight.outputs[0], strength.inputs[0])
    nt.links.new(glow_eff, strength.inputs[1])
    nt.links.new(strength.outputs[0], bsdf.inputs["Emission Strength"])
    return mat, values


def _fix_nonzero_holes(curve):
    """SVG 'nonzero' fill: an inner outline with the same winding as the outline around
    it is filled, not a hole. Blender's curve fill makes every inner outline a hole,
    so drop those same-winding inner outlines; opposite-winding ones stay real holes."""
    info = []
    for sp in curve.splines:
        pts = [(p.co.x, p.co.y) for p in sp.bezier_points] or [(p.co.x, p.co.y) for p in sp.points]
        area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                   for i in range(len(pts))) / 2
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        info.append((sp, area, (min(xs), min(ys), max(xs), max(ys))))
    drop = []
    for sp, area, (x0, y0, x1, y1) in info:
        for other, oarea, (a0, b0, a1, b1) in info:
            if other is sp or abs(oarea) <= abs(area):
                continue
            inside = a0 <= x0 and b0 <= y0 and x1 <= a1 and y1 <= b1
            if inside and (area > 0) == (oarea > 0):
                drop.append(sp)
                break
    for sp in drop:
        curve.splines.remove(sp)
    return len(drop)


def _bolts(rig, meshes, points, width, mats):
    """Hex bolts with washers through the stack. Each sits on whichever sheet is on top
    at its point (found by casting a ray down onto the sheets)."""
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    trees = []
    for o in meshes:
        bm = bmesh.new()
        bm.from_object(o, depsgraph)
        bm.transform(o.matrix_world)
        trees.append(BVHTree.FromBMesh(bm))
        bm.free()
    # Zinc-plated steel, rusted in patches like the sign around it.
    steel = bpy.data.materials.new("BoltSteel")
    steel.use_nodes = True
    nt = steel.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    info = nt.nodes.new("ShaderNodeObjectInfo")
    off = nt.nodes.new("ShaderNodeVectorMath")
    off.operation = "MULTIPLY_ADD"
    nt.links.new(info.outputs["Random"], off.inputs[0])
    off.inputs[1].default_value = (19.0, 23.0, 29.0)
    nt.links.new(tc.outputs["Object"], off.inputs[2])
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 260.0
    n.inputs["Detail"].default_value = 8.0
    nt.links.new(off.outputs[0], n.inputs["Vector"])
    m = nt.nodes.new("ShaderNodeMapRange")
    m.inputs["From Min"].default_value, m.inputs["From Max"].default_value = 0.45, 0.62
    nt.links.new(n.outputs["Fac"], m.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(m.outputs[0], mix.inputs[0])
    mix.inputs[6].default_value = (0.45, 0.44, 0.42, 1)
    mix.inputs[7].default_value = (0.2, 0.07, 0.025, 1)
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(m.outputs[0], inv.inputs[1])
    nt.links.new(inv.outputs[0], b.inputs["Metallic"])
    rough = nt.nodes.new("ShaderNodeMath")
    rough.operation = "MULTIPLY_ADD"
    nt.links.new(m.outputs[0], rough.inputs[0])
    rough.inputs[1].default_value = 0.45
    rough.inputs[2].default_value = 0.38
    nt.links.new(rough.outputs[0], b.inputs["Roughness"])
    for i, (u, v) in enumerate(points):
        x, y = u * width, v * width
        hits = [(t.ray_cast(Vector((x, y, 1.0)), Vector((0, 0, -1)))[0], o) for o, t in zip(meshes, trees)]
        hits = [(h.z, o) for h, o in hits if h is not None]
        if not hits:
            print(f"[crest] bolt {i} at {u:.3f},{v:.3f} misses the crest; skipped")
            continue
        top, piece = max(hits, key=lambda h: h[0])
        # Painted to match the sheet it fixes: the sheet's own material, so the bolt
        # gets the same coat, edge wear, rust, and heat.
        paint = piece.active_material
        print(f"[crest] bolt {i} at {u:.3f},{v:.3f} on {piece.name} ({piece['role']}, sheet {piece['sheet']})")
        r = width * 0.007
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=r * 1.6, depth=r * 0.18,
                                            location=(x, y, top + r * 0.09))
        washer = bpy.context.object
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=r, depth=r * 0.7,
                                            location=(x, y, top + r * 0.18 + r * 0.35))
        head = bpy.context.object
        bev = head.modifiers.new("Bevel", "BEVEL")
        bev.width, bev.segments = r * 0.12, 2
        for o in (washer, head):
            o.data.materials.append(paint or steel)
            o.name = f"Bolt{i}_{o.name}"
            mw = o.matrix_world.copy()
            o.parent = rig
            o.matrix_world = mw


def build(svg_path, width=0.6, kit=None, hot_kit=None, name="Crest", bolts=None):
    """Import the SVG, extrude and stack it, and return (rig, heat).

    bolts: optional list of (u, v) points in crest widths from the centre (v up)."""
    kit = kit or ICE_PAK_KITS["navy"]
    hot_kit = hot_kit or ICE_PAK_KITS["red"]
    before = set(bpy.data.objects)
    bpy.ops.import_curve.svg(filepath=os.path.abspath(svg_path))
    curves = [o for o in bpy.data.objects if o not in before and o.type == "CURVE"]
    if not curves:
        raise SystemExit(f"[crest] no paths imported from {svg_path}")
    # The importer keeps the SVG order in the object names (p00, p01, ...).
    curves.sort(key=lambda o: o.name)
    xs, ys = [], []
    for o in curves:
        for s in o.data.splines:
            for p in s.bezier_points:
                co = o.matrix_world @ p.co
                xs.append(co.x)
                ys.append(co.y)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    scale = width / (max(xs) - min(xs))

    rig = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(rig)
    mats = {}
    meshes = []
    level, last_role = -1, None
    for i, o in enumerate(curves):
        src = o.active_material
        hexcode = None
        if src is not None:
            c = src.diffuse_color
            enc = lambda v: round((v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055) * 255)
            hexcode = "#%02X%02X%02X" % (enc(c[0]), enc(c[1]), enc(c[2]))
        role = DRAWN.get(hexcode or "") or min(DRAWN.items(), key=lambda kv: sum(
            (int(kv[0][j:j + 2], 16) - int((hexcode or "#000000")[j:j + 2], 16)) ** 2 for j in (1, 3, 5)))[1]
        if role != last_role:                       # a new colour run starts a new sheet
            level += 1
            last_role = role
        if role not in mats:
            mats[role] = _material(f"{name}_{role}", kit[role], hot_kit[role], role, frame_obj=rig)
        cu = o.data
        _fix_nonzero_holes(cu)
        cu.dimensions = "2D"
        cu.fill_mode = "BOTH"
        cu.extrude = (SHEET / 2 - BEVEL) / scale
        cu.bevel_depth = BEVEL / scale
        cu.bevel_resolution = 1
        cu.offset = -BEVEL / scale
        cu.resolution_u = 24
        o.data.materials.clear()
        o.data.materials.append(mats[role][0])
        o.location.x -= cx
        o.location.y -= cy
        o.location.z += level * SHEET / scale
        o["sheet"] = level
        o["role"] = role
        meshes.append(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.convert(target="MESH")
    for o in meshes:
        o.location = (o.location.x * scale, o.location.y * scale, o.location.z * scale)
        o.scale = (scale, scale, scale)
        o.parent = rig
        try:
            o.data.shade_smooth_by_angle(angle=math.radians(35))
        except (AttributeError, TypeError):
            pass

    if bolts:
        _bolts(rig, meshes, bolts, width, mats)
    # Half extents of the crest in the rig's space, for rim cooling.
    bpy.context.view_layer.update()
    pts = [rig.matrix_world.inverted() @ (o.matrix_world @ v.co) for o in meshes for v in o.data.vertices]
    for _, values in mats.values():
        values["hx"].outputs[0].default_value = max(abs(p.x) for p in pts)
        values["hy"].outputs[0].default_value = max(abs(p.y) for p in pts)

    def heat(frame, temperature, glow, blend, age=0.0, char=0.0, rim=0.0, face=0.2):
        """Key every material: blackbody temperature (K), glow strength, cold->hot colour blend,
        weathering (1 = old and rusty, 0 = burnt clean), char (paint darkened by heat), and
        rim cooling (0..1: the outline cools first, towards the hot kit's colours), and face
        glow (0.2: the faces stay dark and the edges glow)."""
        for _, values in mats.values():
            for key, v in (("temp", temperature), ("glow", glow), ("blend", blend), ("age", age), ("char", char),
                           ("rim", rim), ("face", face)):
                values[key].outputs[0].default_value = v
                values[key].outputs[0].keyframe_insert("default_value", frame=frame)
    return rig, heat
