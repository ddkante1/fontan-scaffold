import vtk
import numpy as np
import os

def load_centerline(path):
    """Load VTK or VTP centerline and return Nx3 numpy array of points."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError("Unsupported file extension: " + ext)

    reader.SetFileName(path)
    reader.Update()

    polydata = reader.GetOutput()
    if polydata is None:
        raise ValueError(f"Failed to read file: {path}")

    points = polydata.GetPoints()
    if points is None:
        raise ValueError(f"No points found in {path}")

    centerline = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    return centerline


def export_centerline_points(centerline, output_path):
    """Export only the centerline points to CSV: columns x,y,z"""
    header = "x,y,z"
    np.savetxt(output_path, centerline, delimiter=",", header=header, comments="")
    print(f"Saved centerline points to {output_path}")


if __name__ == "__main__":
    # CHANGE THESE PATHS
    input_path = "/Users/daviduva/Fontan_data/centerlines.vtk"
    output_csv = "/Users/daviduva/Fontan_data/centerline_points.csv"

    cl = load_centerline(input_path)
    export_centerline_points(cl, output_csv)