import sys
sys.path.append("/Users/daviduva/.local/lib/python3.11/site-packages")
import subprocess
import bpy
import bmesh    
import mathutils
from mathutils import Vector, Matrix
import numpy as np
import math
from math import radians
import json
import os
import glob
import subprocess
import vtk

with open("my_config.json") as f:
    config = json.load(f)

artery_path = config["artery_path"][0]
scaffold_dir = config["scaffold_dir"]
centerline_csv_paths = config.get("centerline_csv_path", [])
min_points = config.get("centerline_treshold_points", 10)

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

def scale_tuple(t, factor):
    return tuple(factor * x for x in t)

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

def import_mesh(filepath):
    # Import STL mesh

    bpy.ops.import_mesh.stl(filepath=filepath)
    return bpy.context.selected_objects[0]

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
    
# main loop
set_render_settigs()
clear_scene()
setup_lighting_from_config(config)