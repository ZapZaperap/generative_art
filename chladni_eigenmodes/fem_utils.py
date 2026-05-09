import h5py
import numpy as np
import meshio
import tempfile
import os
from pathlib import Path
from skfem import MeshTri


def meshio_to_points_triangles(m):
    """Convert meshio mesh to points and triangles arrays."""
    tri_blocks = [c.data for c in m.cells if c.type == "triangle"]
    if not tri_blocks:
        raise ValueError("No triangle cells found.")

    triangles = np.vstack(tri_blocks)
    used = np.unique(triangles.ravel())

    new_index = -np.ones(m.points.shape[0], dtype=int)
    new_index[used] = np.arange(len(used))

    points = m.points[used, :2]
    triangles = new_index[triangles]

    return points, triangles


def mesh_to_points_triangles(mesh):
    """Extract points and triangles from a meshio or skfem mesh."""
    if hasattr(mesh, "p") and hasattr(mesh, "t"):
        points = np.asarray(mesh.p.T)
        triangles = np.asarray(mesh.t.T)
    elif isinstance(mesh, meshio.Mesh):
        points, triangles = meshio_to_points_triangles(mesh)
    else:
        raise TypeError(f"Unsupported mesh type for export: {type(mesh)}")
    return points, triangles


def hdf5_filename_for_source(mesh_file, output_dir=None):
    path = Path(mesh_file)
    name = path.stem + "_fem.h5"
    if output_dir is not None:
        return Path(output_dir) / name
    return path.with_name(name)


def save_fem_solution_hdf5(output_file, mesh, eigenvalues, eigenvectors, source_mesh_file=None):
    points, triangles = mesh_to_points_triangles(mesh)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as f:
        mesh_grp = f.create_group("mesh")
        mesh_grp.create_dataset("points", data=points, compression="gzip")
        mesh_grp.create_dataset("triangles", data=triangles, compression="gzip")

        modes_grp = f.create_group("modes")
        modes_grp.create_dataset("eigenvalues", data=np.asarray(eigenvalues), compression="gzip")
        modes_grp.create_dataset("eigenvectors", data=np.asarray(eigenvectors), compression="gzip")

        if source_mesh_file is not None:
            f.attrs["source_mesh_file"] = str(source_mesh_file)
            f.attrs["source_mesh_name"] = Path(source_mesh_file).stem
        f.attrs["n_points"] = points.shape[0]
        f.attrs["n_triangles"] = triangles.shape[0]
        f.attrs["n_modes"] = np.asarray(eigenvectors).shape[0]

    return output_file


def load_fem_solution_hdf5(filename):
    with h5py.File(filename, "r") as f:
        points = np.array(f["mesh/points"])
        triangles = np.array(f["mesh/triangles"])
        eigenvalues = np.array(f["modes/eigenvalues"])
        eigenvectors = np.array(f["modes/eigenvectors"])
        metadata = {
            "source_mesh_file": f.attrs.get("source_mesh_file", ""),
            "source_mesh_name": f.attrs.get("source_mesh_name", ""),
            "n_points": int(f.attrs.get("n_points", points.shape[0])),
            "n_triangles": int(f.attrs.get("n_triangles", triangles.shape[0])),
            "n_modes": int(f.attrs.get("n_modes", eigenvectors.shape[0])),
        }
    return {
        "points": points,
        "triangles": triangles,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "metadata": metadata,
    }


def load_skfem_mesh_from_hdf5(filename):
    data = load_fem_solution_hdf5(filename)
    points = np.ascontiguousarray(data["points"].T)
    triangles = np.ascontiguousarray(data["triangles"].T)
    return MeshTri(points, triangles)
