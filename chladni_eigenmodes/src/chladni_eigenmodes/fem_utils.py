import h5py
import numpy as np
import meshio
import tempfile
import os
from pathlib import Path
from skfem import MeshTri
import matplotlib.pyplot as plt
from matplotlib import tri

from shapely.geometry import Polygon
from shapely.geometry import Point, LineString
from shapely.ops import polygonize, unary_union, nearest_points
from shapely.prepared import prep

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

    points = np.ascontiguousarray(points)
    triangles = np.ascontiguousarray(triangles)

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

def plot_mesh(m, lines=True, triangles=True, points=True):
    """
    Plot a meshio mesh object.

    Parameters
    ----------
    m : meshio.Mesh
        Mesh to plot
    """
    _, ax = plt.subplots()
    if points:
        ax.scatter(m.points[:, 0], m.points[:, 1], s=1)
    if "line" in m.cells_dict and lines:
        for l in m.cells_dict["line"]:
            ax.plot(m.points[l, 0], m.points[l, 1], "k-", lw=0.5)
    if "triangle" in m.cells_dict and triangles:
        for t in m.cells_dict["triangle"]:
            ax.plot(m.points[t[[0, 1, 2, 0]], 0], m.points[t[[0, 1, 2, 0]], 1], "r-", lw=0.5)

    ax.set_aspect("equal")
    plt.show()

class EigenmodePotentialField:
    """
    Piecewise-linear representation of an eigenmode over a mesh, with precomputed
    triangle coefficients and gradient data for the potential V = |psi|^2.
    """
    #TODO: Unify with definition in particle_simulations.ipynb and move to a common module
    def __init__(self, points, triangles, eigenvector):
        self.points = points
        self.triangles = triangles
        self.eigenvector = eigenvector
        self.V = np.abs(self.eigenvector) ** 2

        self.triang = tri.Triangulation(points[:, 0], points[:, 1], triangles)
        self.trifinder = self.triang.get_trifinder()
        self.triangle_data = self._precompute_triangle_data()

    def _precompute_triangle_data(self):
        data = []
        for tri_indices in self.triangles:
            p = self.points[tri_indices]
            psi = self.eigenvector[tri_indices]

            A = np.column_stack([np.ones(3), p[:, 0], p[:, 1]])
            a, b, c = np.linalg.solve(A, psi)

            data.append({
                "coeff": np.array([a, b, c]),
                "grad_psi": np.array([b, c]),
            })

        return data

    def _find_triangle(self, x, y):
        return self.trifinder(x, y)

    def psi_at(self, x, y):
        tri_idx = self._find_triangle(x, y)
        if tri_idx == -1:
            return 0.0

        a, b, c = self.triangle_data[tri_idx]["coeff"]
        return a + b * x + c * y

    def gradient(self, x, y):
        tri_idx = self._find_triangle(x, y)
        if tri_idx == -1:
            return np.zeros(2)

        psi_xy = self.psi_at(x, y)
        grad_psi = self.triangle_data[tri_idx]["grad_psi"]
        return 2 * np.real(np.conj(psi_xy) * grad_psi)

    def force_at(self, x, y):
        return -self.gradient(x, y)

class MeshDomain:
    def __init__(self, geometry):
        """
        geometry: shapely Polygon or MultiPolygon.
        May contain holes.
        """
        self.geometry = geometry
        self.boundary = geometry.boundary
        self.prepared = prep(geometry)
        self.minx, self.miny, self.maxx, self.maxy = geometry.bounds

    def contains(self, p):
        return self.prepared.contains(Point(float(p[0]), float(p[1])))

    def signed_distance(self, p):
        point = Point(float(p[0]), float(p[1]))
        d = point.distance(self.boundary)

        if self.prepared.contains(point):
            return d
        else:
            return -d

    def sample(self, n):
        pts = []
        while len(pts) < n:
            p = np.array([
                np.random.uniform(self.minx, self.maxx),
                np.random.uniform(self.miny, self.maxy),
            ])
            if self.contains(p):
                pts.append(p)

        return np.array(pts)

    def nearest_boundary_direction(self, p):
        """
        Returns approximate inward normal direction using nearest boundary point.
        """
        point = Point(float(p[0]), float(p[1]))
        _, nearest = nearest_points(point, self.boundary)

        q = np.array([nearest.x, nearest.y])
        v = np.asarray(p) - q
        norm = np.linalg.norm(v)

        if norm < 1e-12:
            return np.zeros(2)

        # If p is inside, v points inward away from boundary.
        # If p is outside, -v usually points back toward boundary/interior.
        if self.contains(p):
            return v / norm
        else:
            return -v / norm

    def draw(self, ax):
        """
        Draw polygon / multipolygon including holes.
        """

        geometries = getattr(self.geometry, "geoms", [self.geometry])

        for geom in geometries:

            # Outer boundary
            x, y = geom.exterior.xy
            ax.plot(x, y, color="black")

            # Holes
            for interior in geom.interiors:
                xh, yh = interior.xy
                ax.plot(xh, yh, color="black")

        ax.set_aspect("equal")

        pad_x = 0.05 * (self.maxx - self.minx)
        pad_y = 0.05 * (self.maxy - self.miny)

        ax.set_xlim(self.minx - pad_x, self.maxx + pad_x)
        ax.set_ylim(self.miny - pad_y, self.maxy + pad_y)

def load_domain_from_msh(filename, simplify_tol=0.0):
    mesh = meshio.read(filename)
    points = mesh.points[:, :2]

    triangles = []
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles.append(cell_block.data)

    if not triangles:
        raise ValueError("No triangle cells found in mesh.")

    triangles = np.vstack(triangles)

    tri_polygons = []
    for tri in triangles:
        coords = points[tri]
        poly = Polygon(coords)

        if poly.is_valid and poly.area > 0:
            tri_polygons.append(poly)

    if not tri_polygons:
        raise RuntimeError("No valid triangle polygons constructed.")

    geometry = unary_union(tri_polygons)

    if simplify_tol > 0:
        geometry = geometry.simplify(simplify_tol, preserve_topology=True)

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    return MeshDomain(geometry)
