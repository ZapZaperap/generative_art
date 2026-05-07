"""
Standalone script to generate randomized shapes with holes and triangulate them.
Outputs a mesh file that can be used for FEM analysis.
"""

import numpy as np
import gmsh
import meshio
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def random_wobbly_curve(
    center=(0.0, 0.0),
    r_mean=1.0,
    n_harmonics=6,
    noise_strength=0.15,
    n_points=200,
    seed=None,
):
    """
    Generate a smooth random closed curve in polar form.

    Parameters
    ----------
    center : tuple of float
        Center of the curve (x, y)
    r_mean : float
        Mean radius
    n_harmonics : int
        Number of harmonic components
    noise_strength : float
        Amplitude of the noise relative to r_mean
    n_points : int
        Number of points to generate
    seed : int, optional
        Random seed

    Returns
    -------
    pts : ndarray, shape (n_points, 2)
        Points on the curve
    """
    rng = np.random.default_rng(seed)

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    r = np.ones_like(theta) * r_mean

    for k in range(2, n_harmonics + 2):
        amp = noise_strength * r_mean / k
        phase = rng.uniform(0, 2 * np.pi)

        r += amp * np.cos(k * theta + phase)

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.column_stack([x, y])


def curve_is_inside(inner_pts_xy, outer_pts_xy, margin=0.0):
    """
    Check if inner curve is completely inside outer curve.

    Parameters
    ----------
    inner_pts_xy : ndarray, shape (n, 2)
        Points on inner curve
    outer_pts_xy : ndarray, shape (m, 2)
        Points on outer curve
    margin : float, optional
        Additional margin to check

    Returns
    -------
    is_inside : bool
    """
    outer_path = Path(outer_pts_xy)

    if not outer_path.contains_points(inner_pts_xy).all():
        return False

    if margin > 0:
        # crude but useful: also check slightly inflated inner points
        center = inner_pts_xy.mean(axis=0)
        directions = inner_pts_xy - center
        norms = np.linalg.norm(directions, axis=1)

        inflated = inner_pts_xy + margin * directions / norms[:, None]

        if not outer_path.contains_points(inflated).all():
            return False

    return True


def plot_mesh(m):
    """
    Plot a meshio mesh object.

    Parameters
    ----------
    m : meshio.Mesh
        Mesh to plot
    """
    _, ax = plt.subplots()
    ax.scatter(m.points[:, 0], m.points[:, 1], s=1)
    if "line" in m.cells_dict:
        for l in m.cells_dict["line"]:
            ax.plot(m.points[l, 0], m.points[l, 1], "k-", lw=0.5)
    if "triangle" in m.cells_dict:
        for t in m.cells_dict["triangle"]:
            ax.plot(m.points[t[[0, 1, 2, 0]], 0], m.points[t[[0, 1, 2, 0]], 1], "r-", lw=0.5)

    ax.set_aspect("equal")
    plt.show()


def generate_random_shape_mesh(
    output_path,
    outer_r_mean=1.0,
    outer_n_harmonics=7,
    outer_noise_strength=0.25,
    outer_n_points=200,
    outer_lc=0.08,
    inner_hole_min_radius=0.25,
    inner_hole_max_radius=0.40,
    inner_n_harmonics=5,
    inner_noise_strength=0.18,
    inner_n_points=150,
    inner_lc=0.05,
    mesh_char_length=0.03,
    max_attempts=1000,
    seed=None,
    show_plot=False,
):
    """
    Generate a random shape with a hole and triangulate it using gmsh.

    Parameters
    ----------
    output_path : str
        Path to save the mesh file (should end in .msh)
    outer_r_mean : float
        Mean radius of outer curve
    outer_n_harmonics : int
        Number of harmonics for outer curve
    outer_noise_strength : float
        Noise strength for outer curve
    outer_n_points : int
        Number of points on outer curve
    outer_lc : float
        Characteristic length for outer curve
    inner_hole_min_radius : float
        Minimum radius for inner hole
    inner_hole_max_radius : float
        Maximum radius for inner hole
    inner_n_harmonics : int
        Number of harmonics for inner hole
    inner_noise_strength : float
        Noise strength for inner hole
    inner_n_points : int
        Number of points on inner hole curve
    inner_lc : float
        Characteristic length for inner hole
    mesh_char_length : float
        Characteristic mesh length
    max_attempts : int
        Maximum attempts to generate valid inner hole
    seed : int, optional
        Random seed
    show_plot : bool
        Whether to show the mesh plot

    Returns
    -------
    mesh : meshio.Mesh
        The generated mesh
    """

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("random_shape_with_hole")
    occ = gmsh.model.occ

    # Generate outer curve
    outer_pts_xy = random_wobbly_curve(
        center=(0.0, 0.0),
        r_mean=outer_r_mean,
        n_harmonics=outer_n_harmonics,
        noise_strength=outer_noise_strength,
        n_points=outer_n_points,
        seed=seed,
    )

    outer_pts = [occ.addPoint(x, y, 0.0, outer_lc) for x, y in outer_pts_xy]
    outer_pts.append(outer_pts[0])
    outer_curve = occ.addBSpline(outer_pts)
    outer_loop = occ.addCurveLoop([outer_curve])

    # Generate inner hole with validation
    rng = np.random.default_rng(seed)

    for attempt in range(max_attempts):
        hole_center = rng.uniform(-0.2, 0.2, size=2)
        hole_radius = rng.uniform(inner_hole_min_radius, inner_hole_max_radius)

        inner_pts_xy = random_wobbly_curve(
            center=hole_center,
            r_mean=hole_radius,
            n_harmonics=inner_n_harmonics,
            noise_strength=inner_noise_strength,
            n_points=inner_n_points,
            seed=rng.integers(1_000_000_000),
        )

        if curve_is_inside(inner_pts_xy, outer_pts_xy, margin=0.03):
            break
    else:
        gmsh.finalize()
        raise RuntimeError(
            f"Could not generate a valid inner hole after {max_attempts} attempts."
        )

    inner_pts = [occ.addPoint(x, y, 0.0, inner_lc) for x, y in inner_pts_xy]
    inner_pts.append(inner_pts[0])
    inner_curve = occ.addBSpline(inner_pts)
    inner_loop = occ.addCurveLoop([inner_curve])

    # Create surface with hole
    surface = occ.addPlaneSurface([outer_loop, inner_loop])
    occ.synchronize()

    # Mesh settings
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_char_length)

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Save mesh
    gmsh.write(output_path)
    gmsh.finalize()

    # Read mesh with meshio
    mesh = meshio.read(output_path)

    if show_plot:
        plot_mesh(mesh)

    return mesh


def meshio_to_points_triangles(m):
    """
    Convert meshio mesh to points and triangles arrays.

    Parameters
    ----------
    m : meshio.Mesh
        Input mesh

    Returns
    -------
    points : ndarray, shape (n_points, 2)
        Point coordinates
    triangles : ndarray, shape (n_triangles, 3)
        Triangle connectivity
    """
    # collect all triangle cell blocks
    tri_blocks = [c.data for c in m.cells if c.type == "triangle"]
    if not tri_blocks:
        raise ValueError("No triangle cells found.")

    triangles = np.vstack(tri_blocks)  # shape (ntri, 3)

    # find actually used point indices
    used = np.unique(triangles.ravel())

    # build old->new index map
    new_index = -np.ones(m.points.shape[0], dtype=int)
    new_index[used] = np.arange(len(used))

    # prune points and renumber triangles
    points = m.points[used, :2]  # shape (nused, 2)
    triangles = new_index[triangles]  # shape (ntri, 3)

    return points, triangles


if __name__ == "__main__":
    # Example usage
    output_path = "/home/jlangbehn/documents/private/gen_art/chladni/output/random_shape.msh"

    mesh = generate_random_shape_mesh(
        output_path=output_path,
        show_plot=True,
    )

    # Extract points and triangles
    points, triangles = meshio_to_points_triangles(mesh)

    print(f"Generated mesh with {len(points)} points and {len(triangles)} triangles")
    print(f"Mesh saved to {output_path}")
