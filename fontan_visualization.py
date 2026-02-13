import bpy
import bmesh    
import mathutils
from mathutils import Vector
import math
from math import radians
import json
import os
import glob
import subprocess

# --- Load config ---
with open("my_config.json") as f:
    config = json.load(f)

artery_path = config["artery_path"][0]
scaffold_dir = config["scaffold_dir"]

distance_factor = config.get("camera_distance_factor", 2.0)
height_factor = config.get("camera_height_factor", 0.15)
orientation = config.get("camera_orientation", "diagonal")

image_resolution_scaling = config.get("image_resolution_scaling", 100)
number_of_samples =config.get("number_of_render_samples", 100)
light_amplification = config.get("light_amplification", 1.0)
light_distance = config.get("light_distance", 1.0)

make_animation = config.get("make_animation", False)
zoom_camera_in = config.get("zoom_camera_in", False)
number_of_frames= config.get("animation_number_of_frames", 250)
close_up = config.get("close_up", True)
render_test = config.get("render_test", False)
target_name = config.get("target","Coil1") ## target for the camera and lighting

def clear_scene():
    # Delete all objects
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # Delete all meshes
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)

    # Delete all materials
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
        
    # Delete all collections (except master)
    for coll in list(bpy.data.collections):
        if coll.name != "Collection":
            bpy.data.collections.remove(coll, do_unlink=True)

def import_mesh(filepath):
    # Import STL mesh
    bpy.ops.import_mesh.stl(filepath=filepath)
    return bpy.context.selected_objects[0]

def scale_tuple(t, factor):
    return tuple(factor * x for x in t)

def parent_to(target, children):
    # If a single object is passed, wrap it in a list
    if not isinstance(children, (list, tuple)):
        children = [children]
    
    for child in children:
        child.parent = target
        child.matrix_parent_inverse = target.matrix_world.inverted()

# --- 3-point lighting ---
def setup_three_point_lighting(key=800, fill=400, back=300,
        light_distance=1.0, light_amplification=1.0, target_name=None, enabled=True):
    if not enabled:
        return

    # Remove existing lights
    for obj in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj)

    # Get target object (fallback to origin if not found)
    target = bpy.data.objects.get(target_name) if target_name else None
    target_loc = target.location if target else mathutils.Vector((0,0,0))

    def add_light(name, loc, energy):
        light_data = bpy.data.lights.new(name=name, type='AREA')
        light_data.energy = energy
        light_data.size = 5
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = loc

        # Point the light toward target (or origin)
        direction = target_loc - light_obj.location
        light_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        return light_obj

    # Key, Fill, Back
    add_light("KeyLight",  scale_tuple((10, -10, 10), light_distance), key * light_amplification)
    add_light("FillLight", scale_tuple((-12, -5, 6), light_distance), fill * light_amplification)
    add_light("BackLight", scale_tuple((0, 15, 12), light_distance), back * light_amplification)

def center_object_at_origin(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    # Local centroid
    center = sum((v.co for v in bm.verts), Vector()) / len(bm.verts)
    bm.free()

    # Convert to world space
    world_center = obj.matrix_world @ center

    # Shift object so centroid is at origin
    obj.location -= world_center

# --- World HDRI lighting ---
def add_world_lighting(hdr_path=None, strength=1.0, rotation=0.0, enabled=True):
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear nodes
    for node in nodes:
        nodes.remove(node)

    if not enabled:
        # Just plain dark background
        bg_plain = nodes.new(type='ShaderNodeBackground')
        bg_plain.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1)
        output = nodes.new(type='ShaderNodeOutputWorld')
        links.new(bg_plain.outputs['Background'], output.inputs['Surface'])
        return

    # Background shaders
    bg_plain = nodes.new(type='ShaderNodeBackground')
    bg_plain.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1)

    bg_hdri = nodes.new(type='ShaderNodeBackground')
    bg_hdri.inputs['Strength'].default_value = strength
    bg_hdri.location = (0, -200)

    # Load HDRI image
    env_tex = nodes.new(type='ShaderNodeTexEnvironment')
    env_tex.image = bpy.data.images.load(hdr_path) if hdr_path else None
    env_tex.location = (-800, 0)

    # Mapping + coords
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.inputs['Rotation'].default_value[2] = rotation
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-400, -200)
    mapping.location = (-200, -200)

    # Light path + mix
    light_path = nodes.new(type='ShaderNodeLightPath')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    output = nodes.new(type='ShaderNodeOutputWorld')

    # Link
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
    links.new(env_tex.outputs['Color'], bg_hdri.inputs['Color'])
    links.new(bg_hdri.outputs['Background'], mix_shader.inputs[1])   # HDRI
    links.new(bg_plain.outputs['Background'], mix_shader.inputs[2]) # dark bg
    links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

# --- Set up lightinh with config parameters ---
def setup_lighting_from_config(params):
    use_hdri = params.get("use_hdri", True)
    hdr_path = params.get("hdr_image_path", None)
    hdr_strength = params.get("hdr_strength", 1.0)
    hdr_rotation = params.get("hdr_rotation", 0.0)

    use_three_point = params.get("use_three_point", True)

    add_world_lighting(hdr_path=hdr_path, strength=hdr_strength,
                       rotation=hdr_rotation, enabled=use_hdri)
    setup_three_point_lighting(light_distance=light_distance, light_amplification=light_amplification,
                                enabled=use_three_point,target_name=target_name)

def set_camera_orientation(cam_obj, target, orientation="diagonal", distance_factor=4.0):
    # Compute distance from bounding box of target
    bbox = [target.matrix_world @ v.co for v in target.data.vertices]
    size = max((v - target.location).length for v in bbox)
    dist = size * distance_factor

    if orientation == "diagonal":
        cam_obj.location = (dist, dist, dist)
    elif orientation == "side":
        cam_obj.location = (dist, 0, 0)
    elif orientation == "tilted20":
        # Convert 20° above and 20° to the right into radians
        elev = math.radians(20)   # elevation
        azim = math.radians(20)   # azimuth
        cam_obj.location = (
            dist * math.cos(elev) * math.sin(azim),  # X
            dist * math.cos(elev) * math.cos(azim),  # Y
            dist * math.sin(elev)                    # Z
        )

def compute_centroid(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.data

    sum_pos = mathutils.Vector((0.0, 0.0, 0.0))
    for v in mesh.vertices:
        sum_pos += obj.matrix_world @ v.co

    return sum_pos / len(mesh.vertices)

def point_camera_at(camera, target_obj):
    """Rotate camera to look at target object."""
    direction = target_obj - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')  # Blender camera looks along -Z
    camera.rotation_euler = rot_quat.to_euler()

def place_camera():

    cam_obj = bpy.data.objects.get("Camera")
    if cam_obj is None:
        bpy.ops.object.camera_add()
        cam_obj = bpy.context.object
        cam_obj.name = "Camera"

    centroid = compute_centroid(artery)

    #print ("centroid",centroid)

    # Compute bounding box size
    bbox = [artery.matrix_world @ mathutils.Vector(corner) for corner in artery.bound_box]
    size_x = max([v[0] for v in bbox]) - min([v[0] for v in bbox])
    size_y = max([v[1] for v in bbox]) - min([v[1] for v in bbox])
    size_z = max([v[2] for v in bbox]) - min([v[2] for v in bbox])

    max_dim = max(size_x, size_y, size_z)

    #print ("size", size_x, size_y, size_z )

    # Set camera distance (e.g., 2x the largest artery dimension)
    distance_factor = 1.2

    # Place camera at offset along -Y and above Z
    cam_obj.location = centroid + mathutils.Vector((0, max_dim*distance_factor, max_dim*0.5))

    point_camera_at(bpy.data.objects["Camera"], centroid)

def get_or_create_material(name):
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    return mat


def assign_artery_material(obj, name="ArteryMaterial", 
                           color=(0.85, 0.25, 0.25, 0.35)):
    """
    High-quality translucent artery material.
    
    color = (R, G, B, Alpha)
    Alpha controls transparency.
    """

    mat = get_or_create_material(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    # --- Base Color ---
    bsdf.inputs["Base Color"].default_value = color

    # --- Subsurface Scattering (soft tissue effect) ---
#    bsdf.inputs["Subsurface"].default_value = 0.45
#    bsdf.inputs["Subsurface Radius"].default_value = Vector((1.2, 0.5, 0.4))
#   bsdf.inputs["Subsurface Color"].default_value = (1.0, 0.3, 0.3, 1)

    # --- Surface response ---
    bsdf.inputs["Metallic"].default_value = 0.0
 #   bsdf.inputs["Specular"].default_value = 0.4
    bsdf.inputs["Roughness"].default_value = 0.25

    # --- Transparency ---
    bsdf.inputs["Alpha"].default_value = color[3]

    mat.blend_method = 'BLEND'
#    mat.shadow_method = 'HASHED'

    # Assign material
    if mat.name not in [m.name for m in obj.data.materials]:
        obj.data.materials.append(mat)

    index = obj.data.materials.find(mat.name)
    for poly in obj.data.polygons:
        poly.material_index = index

def assign_CoCr_material(obj, name="CoCr_material"):


    """
    Assign a chromium–cobalt (metallic) material to a Blender object.
    
    Parameters:
        obj (bpy.types.Object): The mesh object to assign the material to.
        name (str): Name of the material.
    """
    # --- Get existing material or create a new one ---
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True

    # --- Set up Principled BSDF ---
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (450, 400)
    
    # Co–Cr is a true metal
    bsdf.inputs["Metallic"].default_value = 1.0
    # Polished but not mirror-like
    bsdf.inputs["Roughness"].default_value = 0.25
    # Neutral steel-grey (slightly cool)
    bsdf.inputs["Base Color"].default_value = (0.70, 0.72, 0.75, 1.0)
    # Fully opaque
    bsdf.inputs["Alpha"].default_value = 1.0

    # --- Assign material to object ---
    if mat.name not in [m.name for m in obj.data.materials]:
        obj.data.materials.append(mat)
    
    # Apply material to all faces
    index = obj.data.materials.find(mat.name)
    for poly in obj.data.polygons:
        poly.material_index = index

def set_render_settigs ():
    bpy.context.scene.render.engine = 'CYCLES'
    major, minor, patch = bpy.app.version
    if major <= 4:
        bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'

    # --- Device: GPU if available ---
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"  # or "OPTIX" (NVIDIA RTX), or "HIP" (AMD)
    bpy.context.scene.cycles.device = "GPU"

    # --- Sampling ---
    bpy.context.scene.cycles.samples = number_of_samples            # final render samples
    bpy.context.scene.cycles.preview_samples = 100     # viewport samples
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.01

    # --- Denoising ---
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'  

    # --- Clamp to reduce fireflies ---
    bpy.context.scene.cycles.sample_clamp_direct = 0.0
    bpy.context.scene.cycles.sample_clamp_indirect = 2.0

    c = bpy.context.scene.cycles

    # Light paths
    c.max_bounces = 4
    c.diffuse_bounces = 2
    c.glossy_bounces = 2
    c.transmission_bounces = 4
    c.volume_bounces = 0
    c.transparent_max_bounces = 4

    # No caustics
    c.caustics_reflective = False
    c.caustics_refractive = False

    # Motion blur
    bpy.context.scene.render.use_motion_blur = False

    # --- Output settings ---
    bpy.context.scene.render.resolution_x = 4000
    bpy.context.scene.render.resolution_y = 3000
    bpy.context.scene.render.resolution_percentage = image_resolution_scaling  # scale (100% = full res)

    # Enable transparent background
    bpy.context.scene.render.film_transparent = True

    # --- Set output file format to PNG with alpha ---
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # include alpha channel
    bpy.context.scene.render.image_settings.color_depth = '8'     # or '16' for higher quality


# Main loop

set_render_settigs()
clear_scene()
setup_lighting_from_config(config)

# --- Import artery ---
bpy.ops.import_mesh.stl(filepath=artery_path)
artery = bpy.context.selected_objects[0]

center_object_at_origin(artery)

assign_artery_material(artery, "TranslucentArtery", color=(0.85, 0.25, 0.25, 0.35) )
                       
place_camera()

# --- Get all STL files in the scaffold directory ---
scaffold_files = sorted(glob.glob(os.path.join(scaffold_dir, "*.stl")))

# --- Import scaffolds and store as objects ---
scaffolds = []

# py.ops.import_mesh.stl(filepath=scaffold_files[0])

for i, f in enumerate(scaffold_files[:1]):
    bpy.ops.import_mesh.stl(filepath=f)
    obj = bpy.context.selected_objects[0]
    obj.name = f"scaffold_{i:03d}"
    obj.hide_render = False 
    obj.hide_viewport = False # visible
    scaffolds.append(obj)

scale_factor = 1000.0

for obj in scaffolds:
    obj.scale = (obj.scale[0] * scale_factor,
                 obj.scale[1] * scale_factor,
                 obj.scale[2] * scale_factor)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(scale=True)
    obj.select_set(False)
     # Assign Co-Cr material  
    assign_CoCr_material(obj)

if (make_animation):
    # --- Animate scaffolds over time ---
    frame_start = 1
    frame_gap = 20  # frames between each timestep

    # Hide all scaffolds at frame 1
    for obj in scaffolds:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert("hide_viewport", frame=frame_start)
        obj.keyframe_insert("hide_render", frame=frame_start)


    for i, obj in enumerate(scaffolds):

        f_on  = frame_start + i * frame_gap
        f_off = f_on + frame_gap

        # Show
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert("hide_viewport", frame=f_on)
        obj.keyframe_insert("hide_render", frame=f_on)

        # Hide again
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert("hide_viewport", frame=f_off)
        obj.keyframe_insert("hide_render", frame=f_off)