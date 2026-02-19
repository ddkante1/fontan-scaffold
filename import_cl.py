import bpy
import bmesh
import mathutils
from mathutils import Vector
import numpy as np
import os

# -------------------------------
# CONFIG
# -------------------------------
centerline_csv = "/Users/daviduva/Fontan_data/centerline_points.csv"
scaffold_name = "Scaffold"   # Name of your scaffold object in Blender
frame_start = 1              # Start frame if you want to animate
frame_step = 1               # Frame step for animation (optional)

# -------------------------------
# LOAD CENTERLINE
# -------------------------------
centerline = np.loadtxt(centerline_csv, delimiter=",", skiprows=1)
points = [Vector(p) for p in centerline]

print(f"Loaded {len(points)} points from centerline.")