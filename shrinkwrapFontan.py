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

# ---------------------------
# Create collection
# ---------------------------
collection = bpy.data.collections.new("FontanScene")
bpy.context.scene.collection.children.link(collection)

# ---------------------------
# Import Fontan model
# ---------------------------
bpy.ops.import_mesh.stl(filepath="CompleteModel_Fontan_47mm.stl")
fontan = bpy.context.active_object
collection.objects.link(fontan)

# ---------------------------
# Create cylinder scaffold
# ---------------------------
bpy.ops.mesh.primitive_cylinder_add(radius=5, depth=40)
cylinder = bpy.context.active_object
cylinder.name = "Cylinder"
collection.objects.link(cylinder)

# ---------------------------
# Create lattice
# ---------------------------
bpy.ops.object.add(type='LATTICE')
lattice = bpy.context.active_object
lattice.name = "Lattice"
collection.objects.link(lattice)

# Scale lattice
lattice.scale = (10,10,20)

# ---------------------------
# Add modifiers to cylinder
# ---------------------------
lat_mod = cylinder.modifiers.new("Lattice", type='LATTICE')
lat_mod.object = lattice

solid_mod = cylinder.modifiers.new("Solidify", type='SOLIDIFY')
solid_mod.thickness = 1.0

# ---------------------------
# Shrinkwrap lattice to Fontan
# ---------------------------
shrink = lattice.modifiers.new("Shrinkwrap", type='SHRINKWRAP')
shrink.target = fontan