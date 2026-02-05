import bpy
import bmesh    
import mathutils
from mathutils import Vector
import math
from math import radians
import json
import os
import subprocess

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
    

def parent_to(target, children):
    # If a single object is passed, wrap it in a list
    if not isinstance(children, (list, tuple)):
        children = [children]
    
    for child in children:
        child.parent = target
        child.matrix_parent_inverse = target.matrix_world.inverted()

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

def assign_artery_material(obj, material_name, color):
    # Create material if it doesn't exist    
    mat = get_or_create_material("aneurysm_material")
    # set up bsdf
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
   # Base color
    bsdf.inputs[0].default_value = color        # Base Color
    bsdf.inputs[8].default_value = 0.2          # Subsurface
    bsdf.inputs[9].default_value = Vector((1.0,0.6,0.6)) # Subsurface Radius
    bsdf.inputs[1].default_value = 0.0          # Metallic
    bsdf.inputs[2].default_value = 0.4          # Roughness
    bsdf.inputs[4].default_value = color[3]    # Alpha

    mat.blend_method = 'BLEND'

    # Append material if not already in the object
    if mat.name not in [m.name for m in obj.data.materials]:
        obj.data.materials.append(mat)

    # Assign this material to all faces
    index = obj.data.materials.find(mat.name)
    for poly in obj.data.polygons:
        poly.material_index = index

def setup_medical_lighting(strength_key=800, strength_fill=400, strength_back=300):
    # Remove old lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)
    for obj in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj)

    # --- Key light (main, strong, directional) ---
    key_light_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_light_data.energy = strength_key
    key_light_data.size = 5
    key_light = bpy.data.objects.new(name="KeyLight", object_data=key_light_data)
    bpy.context.scene.collection.objects.link(key_light)
    key_light.location = (10, -10, 10)

    # --- Fill light (softens shadows) ---
    fill_light_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_light_data.energy = strength_fill
    fill_light_data.size = 5
    fill_light = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
    bpy.context.scene.collection.objects.link(fill_light)
    fill_light.location = (-12, -5, 6)

    # --- Back light (adds rim highlight) ---
    back_light_data = bpy.data.lights.new(name="BackLight", type='AREA')
    back_light_data.energy = strength_back
    back_light_data.size = 6
    back_light = bpy.data.objects.new(name="BackLight", object_data=back_light_data)
    bpy.context.scene.collection.objects.link(back_light)
    back_light.location = (0, 15, 12)

    print("3-point lighting setup added.")

    # Clean scene

clear_scene()

setup_lighting_from_config(params)

# --- Import sac and arteries ---
arteries = import_mesh(aneurysm_paths[0])
arteries.name = "Fontan_arteries"
assign_artery_material(aneurysm_sac, "TranslucentMaterial", (1.0, 0.2, 0.2, 0.25))  # brighter red

arteries = bpy.data.objects.get("AneurysmSac")


    