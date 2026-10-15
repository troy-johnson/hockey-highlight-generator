"""Render the Black Ice opening stinger with Blender.

Run headless:

    Blender -b --factory-startup --python v3/blender/opening_stinger.py -- \
        --lockup lockup.png --primary "#1E3160" --out frames/ [--vertical]

The scene has no text. Remotion draws the matchup rail, the date, and the
ice flash over these frames. Timing matches the 135-frame Remotion opening
(4.5 s at 30 fps) so the generated opening sound still fits:

- 4-26: the slab slides in and hockey-stops; ice spray at 20-30.
- 24-46: frost grows from the blade contact corner.
- 44-67: the lockup breaks through the frost.
- 60-70: a light sweeps across the face.
- 114-134: the camera pushes into the lockup.
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

FRAMES = 135
FPS = 30
SLAB_W, SLAB_H, SLAB_D = 10.0, 2.9, 0.55
LIFT = SLAB_H / 2 + 0.03          # slab centre height above the ice


def parse_args(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="opening_stinger")
    p.add_argument("--lockup", required=True, help="transparent PNG for the slab face")
    p.add_argument("--primary", default="#1E3160", help="team primary colour")
    p.add_argument("--out", required=True, help="output directory for PNG frames")
    p.add_argument("--vertical", action="store_true", help="1080x1920 instead of 1920x1080")
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=FRAMES - 1)
    p.add_argument("--scale", type=int, default=100, help="resolution percentage")
    return p.parse_args(argv)


def rgb(hexcode):
    h = hexcode.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, 1.0)


def ease_out(t):
    return 1 - (1 - t) ** 2


def clamp01(t):
    return max(0.0, min(1.0, t))


def motion(f):
    """Slab position and rotation at frame f (same curve as the Remotion opening)."""
    slide = ease_out(clamp01((f - 4) / 22))
    drift = clamp01((f - 26) / 82)
    x = 24 * (1 - slide)
    yaw = -0.35 * (1 - slide) + 0.18 - 0.06 * drift
    t = clamp01((f - 26) / 16)
    back = 1 + 3.2 * (t - 1) ** 3 + 2.2 * (t - 1) ** 2   # overshoot settle
    lean = 0.1 * clamp01((f - 18) / 8) * (1 - back)
    push = clamp01((f - 114) / 20)
    push = push * push * (3 - 2 * push)
    yaw *= 1 - push
    return x, yaw, lean, drift, push


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, FRAMES - 1
    scene.render.fps = FPS
    return scene


def node_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return mat, nt, out


def key_value(node, frames_values):
    for f, v in frames_values:
        node.outputs[0].default_value = v
        node.outputs[0].keyframe_insert("default_value", frame=f)


def world(scene):
    w = bpy.data.worlds.new("Studio")
    scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.0015, 0.0022, 0.004, 1)
    bg.inputs["Strength"].default_value = 1.0


def ice(scene):
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 30, 0))
    floor = bpy.context.object
    floor.name = "Ice"
    floor.scale = (220, 120, 1)
    mat, nt, out = node_material("Ice")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1, 18, 1)       # long skate cuts
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    cuts = nt.nodes.new("ShaderNodeTexNoise")
    cuts.inputs["Scale"].default_value = 140
    cuts.inputs["Detail"].default_value = 6
    nt.links.new(mapping.outputs["Vector"], cuts.inputs["Vector"])
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 900
    nt.links.new(coord.outputs["Object"], fine.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.06, 0.07, 0.085, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.16, 0.19, 0.22, 1)
    nt.links.new(cuts.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.05
    rough.inputs["To Max"].default_value = 0.32
    nt.links.new(fine.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    nt.links.new(cuts.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Coat Weight"].default_value = 0.6
    bsdf.inputs["Coat Roughness"].default_value = 0.04
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    floor.data.materials.append(mat)
    # A blue line passing under the slab gives the stop a sense of travel.
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 4, 0.002))
    line = bpy.context.object
    line.name = "BlueLine"
    line.scale = (0.6, 120, 1)
    line.rotation_euler.z = math.pi / 2
    lm, lnt, lout = node_material("BlueLine")
    lb = lnt.nodes.new("ShaderNodeBsdfPrincipled")
    lb.inputs["Base Color"].default_value = (0.01, 0.03, 0.12, 1)
    lb.inputs["Roughness"].default_value = 0.15
    lb.inputs["Alpha"].default_value = 0.55
    lnt.links.new(lb.outputs["BSDF"], lout.inputs["Surface"])
    line.data.materials.append(lm)


def slab(args):
    root = bpy.data.objects.new("SlabRig", None)
    bpy.context.collection.objects.link(root)

    bpy.ops.mesh.primitive_cube_add(size=1)
    body = bpy.context.object
    body.name = "Slab"
    body.scale = (SLAB_W, SLAB_D, SLAB_H)
    bpy.ops.object.transform_apply(scale=True)
    bev = body.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments = 0.08, 4
    body.parent = root
    mat, nt, out = node_material("Graphite")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (0.012, 0.016, 0.022, 1)
    b.inputs["Metallic"].default_value = 0.55
    b.inputs["Roughness"].default_value = 0.32
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.08
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    body.data.materials.append(mat)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -SLAB_D / 2 - 0.02, -SLAB_H / 2 + 0.1))
    strip = bpy.context.object
    strip.name = "TeamStrip"
    strip.scale = (SLAB_W - 0.1, 0.06, 0.12)
    strip.parent = root
    sm, snt, sout = node_material("TeamStrip")
    em = snt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgb(args.primary)
    strength = snt.nodes.new("ShaderNodeValue")
    key_value(strength, [(0, 2.0), (26, 2.0), (27, 14.0), (40, 4.0)])
    snt.links.new(strength.outputs[0], em.inputs["Strength"])
    snt.links.new(em.outputs["Emission"], sout.inputs["Surface"])
    strip.data.materials.append(sm)

    face = face_plane("Lockup", -SLAB_D / 2 - 0.012, root)
    face.data.materials.append(lockup_material(args.lockup))
    frost = face_plane("Frost", -SLAB_D / 2 - 0.02, root)
    frost.data.materials.append(frost_material())
    return root


def face_plane(name, y, parent):
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, y, 0), rotation=(math.pi / 2, 0, 0))
    plane = bpy.context.object
    plane.name = name
    plane.scale = (SLAB_W - 0.5, SLAB_H - 0.4, 1)
    plane.parent = parent
    return plane


def lockup_material(path):
    mat, nt, out = node_material("Lockup")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.extension = "CLIP"
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    aspect = tex.image.size[0] / max(1, tex.image.size[1])
    box = (SLAB_W - 0.5) / (SLAB_H - 0.4)
    sx, sy = (1, aspect / box) if aspect < box else (box / aspect, 1)
    sx, sy = sx * 0.78, sy * 0.78                      # margin inside the slab face
    mapping.inputs["Scale"].default_value = (1 / sx, 1 / sy, 1)
    mapping.inputs["Location"].default_value = (0.5 - 0.5 / sx, 0.5 - 0.5 / sy, 0)
    nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Coat Weight"].default_value = 0.8
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    glow = nt.nodes.new("ShaderNodeValue")
    key_value(glow, [(43, 0.0), (52, 1.6), (62, 0.35)])
    nt.links.new(glow.outputs[0], bsdf.inputs["Emission Strength"])
    reveal = nt.nodes.new("ShaderNodeValue")
    key_value(reveal, [(43, 0.0), (62, 1.0)])
    alpha = nt.nodes.new("ShaderNodeMath")
    alpha.operation = "MULTIPLY"
    nt.links.new(tex.outputs["Alpha"], alpha.inputs[0])
    nt.links.new(reveal.outputs[0], alpha.inputs[1])
    nt.links.new(alpha.outputs[0], bsdf.inputs["Alpha"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def frost_material():
    """Frost grows from the bottom-left contact corner, then clears from the centre."""
    mat, nt, out = node_material("Frost")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["UV"], sep.inputs["Vector"])
    aspect = (SLAB_W - 0.5) / (SLAB_H - 0.4)

    def math_node(op, a, b=None):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                nt.links.new(v, n.inputs[i])
        return n.outputs[0]

    ux = math_node("MULTIPLY", sep.outputs["X"], aspect)
    contact = math_node("DIVIDE", math_node("SQRT", math_node(
        "ADD", math_node("POWER", ux, 2), math_node("POWER", sep.outputs["Y"], 2))),
        math.hypot(aspect, 1))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9
    noise.inputs["Detail"].default_value = 12
    noise.inputs["Roughness"].default_value = 0.68
    nt.links.new(coord.outputs["UV"], noise.inputs["Vector"])
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 22
    nt.links.new(coord.outputs["UV"], vor.inputs["Vector"])

    growth = nt.nodes.new("ShaderNodeValue")
    key_value(growth, [(24, -0.1), (46, 1.15)])
    field = math_node("ADD", contact, math_node("MULTIPLY", math_node(
        "SUBTRACT", noise.outputs["Fac"], 0.5), 0.5))
    frozen = math_node("SUBTRACT", growth.outputs[0], field)
    frozen_mask = math_node("MAXIMUM", math_node("MULTIPLY", frozen, 18), 0)
    frozen_mask = math_node("MINIMUM", frozen_mask, 1)
    front = math_node("MULTIPLY", math_node("SUBTRACT", 1, math_node(
        "MINIMUM", math_node("ABSOLUTE", math_node("MULTIPLY", frozen, 25)), 1)), 1)

    centre_x = math_node("MULTIPLY", math_node("SUBTRACT", sep.outputs["X"], 0.5), aspect)
    centre_y = math_node("SUBTRACT", sep.outputs["Y"], 0.5)
    radius = math_node("SQRT", math_node("ADD", math_node("POWER", centre_x, 2),
                                         math_node("POWER", centre_y, 2)))
    radius = math_node("ADD", radius, math_node("MULTIPLY", noise.outputs["Fac"], 0.35))
    clear = nt.nodes.new("ShaderNodeValue")
    key_value(clear, [(44, -0.2), (67, 2.4)])
    cleared = math_node("MINIMUM", math_node("MAXIMUM", math_node(
        "MULTIPLY", math_node("SUBTRACT", clear.outputs[0], radius), 6), 0), 1)
    edge_x = math_node("MINIMUM", ux, math_node("SUBTRACT", aspect, ux))
    edge = math_node("MINIMUM", edge_x, math_node("MINIMUM", sep.outputs["Y"], math_node(
        "SUBTRACT", 1, sep.outputs["Y"])))
    residue = math_node("SUBTRACT", 1, math_node("MINIMUM", math_node(
        "MULTIPLY", math_node("ADD", edge, math_node("MULTIPLY", noise.outputs["Fac"], 0.12)), 7), 1))

    crystals = math_node("SUBTRACT", 1, math_node("MINIMUM", math_node(
        "MULTIPLY", vor.outputs["Distance"], 30), 1))
    # Vary the coat with noise so the frost is not a uniform cracked tile.
    patches = math_node("MINIMUM", math_node("MAXIMUM", math_node("MULTIPLY", math_node(
        "SUBTRACT", noise.outputs["Fac"], 0.45), 4.0), 0), 1)
    coat = math_node("ADD", math_node("MULTIPLY", patches, 0.18), math_node("MULTIPLY", math_node(
        "MULTIPLY", crystals, 0.8), math_node("ADD", 0.08, patches)))
    keep = math_node("MAXIMUM", math_node("SUBTRACT", 1, cleared), residue)
    alpha = math_node("MINIMUM", math_node("MULTIPLY", math_node(
        "MULTIPLY", frozen_mask, keep), coat), 0.92)
    alpha = math_node("MAXIMUM", alpha, math_node("MULTIPLY", front, 0.9))

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.75, 0.86, 0.95, 1)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Transmission Weight"].default_value = 0.3
    bsdf.inputs["Emission Color"].default_value = (0.55, 0.8, 1.0, 1)
    nt.links.new(math_node("ADD", math_node("MULTIPLY", front, 6), math_node(
        "MULTIPLY", crystals, 0.25)), bsdf.inputs["Emission Strength"])
    nt.links.new(alpha, bsdf.inputs["Alpha"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def spray(root):
    """Ice spray thrown from the leading (left) blade edge during the stop."""
    bpy.ops.mesh.primitive_plane_add(size=1, location=(-SLAB_W / 2 + 0.3, 0, 0.05))
    emitter = bpy.context.object
    emitter.name = "SprayEmitter"
    emitter.scale = (1.2, 0.8, 1)
    # A render-hidden emitter also hides its particles; hide only the mesh.
    emitter.show_instancer_for_render = False
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1)
    grain = bpy.context.object
    grain.name = "Grain"
    grain.location = (0, 0, -50)
    gm, gnt, gout = node_material("Grain")
    ge = gnt.nodes.new("ShaderNodeEmission")
    ge.inputs["Color"].default_value = (0.8, 0.9, 1.0, 1)
    ge.inputs["Strength"].default_value = 3.0
    gnt.links.new(ge.outputs["Emission"], gout.inputs["Surface"])
    grain.data.materials.append(gm)

    ps_mod = emitter.modifiers.new("Spray", "PARTICLE_SYSTEM")
    ps = ps_mod.particle_system.settings
    emitter.particle_systems[0].seed = 1731
    ps.count = 4000
    ps.frame_start, ps.frame_end = 20, 30
    ps.lifetime, ps.lifetime_random = 34, 0.5
    ps.emit_from = "FACE"
    ps.normal_factor = 4.0
    ps.object_align_factor = (-9.0, -2.0, 1.5)
    ps.factor_random = 3.5
    ps.effector_weights.gravity = 0.55
    ps.drag_factor = 0.25
    ps.render_type = "OBJECT"
    ps.instance_object = grain
    ps.particle_size, ps.size_random = 0.04, 0.8
    return emitter


def lights():
    def area(name, loc, energy, size, color=(1, 1, 1), rot=None):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.size, data.color = energy, size, color
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        bpy.context.collection.objects.link(obj)
        track = obj.constraints.new("TRACK_TO")
        target = bpy.data.objects.get("SlabRig")
        track.target = target
        return obj

    # Fixed lights do not show as bright rectangles on the ice.
    for obj in (area("Key", (-7, -10, 9), 2600, 6, (0.88, 0.95, 1.0)),
                area("Rim", (6, 8, 6), 3200, 4, (0.55, 0.78, 1.0)),
                area("Under", (0, -6, 0.4), 260, 10, (0.45, 0.7, 1.0))):
        obj.visible_glossy = False
    sweep = area("Sweep", (-9, -4, 3.5), 0, 1.2, (1, 1, 1))
    for f, x, e in [(58, -9, 0), (60, -8, 2400), (70, 8, 2400), (72, 9, 0)]:
        sweep.location.x = x
        sweep.data.energy = e
        sweep.keyframe_insert("location", index=0, frame=f)
        sweep.data.keyframe_insert("energy", frame=f)


def camera(scene, vertical):
    data = bpy.data.cameras.new("Camera")
    data.lens = 50 if not vertical else 62
    if vertical:
        # Fit the slab to the frame width and place it in the upper third.
        data.sensor_fit = "HORIZONTAL"
        data.shift_y = -0.2
    data.dof.use_dof = True
    data.dof.aperture_fstop = 2.8
    cam = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    target = bpy.data.objects.new("CameraTarget", None)
    bpy.context.collection.objects.link(target)
    data.dof.focus_object = target
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    return cam, target


def animate(root, cam, target, emitter, vertical):
    base = 17.0 if vertical else 16.0
    for f in range(FRAMES):
        x, yaw, lean, drift, push = motion(f)
        root.location = (x, 0, LIFT + 5.5 * abs(math.sin(lean)) * 0.12)
        root.rotation_euler = (0, lean, yaw)
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_euler", frame=f)
        emitter.location.x = x - SLAB_W / 2 + 0.3
        emitter.keyframe_insert("location", frame=f)
        settle = 1 - ease_out(clamp01(f / 26))
        shake = math.exp(-(f - 26) / 5) * 0.06 if f >= 26 else 0
        dist = base - 1.2 * drift
        dist = dist + (2.2 - dist) * push
        cam.location = (-1.5 * settle + math.sin(f * 2.1) * shake,
                        -dist, LIFT + 1.0 + 0.6 * settle - 0.9 * push + math.cos(f * 2.7) * shake)
        cam.keyframe_insert("location", frame=f)
        target.location = (x * 0.35 * (1 - push), 0, LIFT)
        target.keyframe_insert("location", frame=f)


def render_settings(scene, args):
    scene.render.engine = "BLENDER_EEVEE"
    ee = scene.eevee
    ee.taa_render_samples = args.samples
    for attr, value in [("use_raytracing", True), ("use_shadows", True)]:
        if hasattr(ee, attr):
            setattr(ee, attr, value)
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    scene.render.resolution_x, scene.render.resolution_y = (1080, 1920) if args.vertical else (1920, 1080)
    scene.render.resolution_percentage = args.scale
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = os.path.join(os.path.abspath(args.out), "frame_")
    scene.frame_start, scene.frame_end = args.start, args.end
    compositor(scene)


def compositor(scene):
    """Bloom and a soft vignette in the compositor."""
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
    for name, value in [("Type", "Bloom"), ("Threshold", 1.0), ("Strength", 0.6), ("Size", 0.6)]:
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


def main():
    args = parse_args(sys.argv)
    if not os.path.isfile(args.lockup):
        print(f"[opening] lockup not found: {args.lockup}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(args.out, exist_ok=True)
    scene = reset_scene()
    world(scene)
    ice(scene)
    root = slab(args)
    emitter = spray(root)
    lights()
    cam, target = camera(scene, args.vertical)
    animate(root, cam, target, emitter, args.vertical)
    render_settings(scene, args)
    bpy.ops.render.render(animation=True)
    print(f"[opening] rendered frames {args.start}-{args.end} to {args.out}")


if __name__ == "__main__":
    main()
