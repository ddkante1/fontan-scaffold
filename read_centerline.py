import bpy
import pyvista as pv
#print(pv.__version__)

import os

# ------------------------------
# Path to your .vtk centerline
# ------------------------------
vtk_file = "/Users/daviduva/Fontan_data/centerlines.vtk/"

if not os.path.exists(vtk_file):
    raise FileNotFoundError(f"VTK file not found: {vtk_file}")

# ------------------------------
# Read points from the .vtk
# ------------------------------
centerline = pv.read(vtk_file)
points = centerline.points  # Nx3 array

if len(points) == 0:
    raise ValueError("No points found in VTK file.")

print(f"Read {len(points)} points from centerline.")

# ------------------------------
# Create a Blender curve
# ------------------------------
curve_data = bpy.data.curves.new("CenterlineCurve", type='CURVE')
curve_data.dimensions = '3D'

spline = curve_data.splines.new('POLY')
spline.points.add(len(points) - 1)  # Blender adds 1 point by default

# Assign points
for i, p in enumerate(points):
    x, y, z = p
    spline.points[i].co = (x, y, z, 1)  # w = 1

# Create object and link to scene
curve_obj = bpy.data.objects.new("Centerline", curve_data)
bpy.context.collection.objects.link(curve_obj)

# Optional: bevel for visualization
curve_data.bevel_depth = 0.002  # small radius
curve_data.resolution_u = 64    # smooth curve

print("Centerline curve created in Blender successfully!")