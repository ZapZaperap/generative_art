"""
Standalone script to solve FEM eigenvalue problem on a mesh and plot eigenmodes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib
import meshio
from scipy.sparse.linalg import eigsh
from skfem import MeshTri, Basis, ElementTriP1, asm, condense
from skfem.models.poisson import laplace, mass
import argparse


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

    triangles = np.vstack(tri_blocks)   # shape (ntri, 3)

    # find actually used point indices
    used = np.unique(triangles.ravel())

    # build old->new index map
    new_index = -np.ones(m.points.shape[0], dtype=int)
    new_index[used] = np.arange(len(used))

    # prune points and renumber triangles
    points = m.points[used, :2]              # shape (nused, 2)
    triangles = new_index[triangles]           # shape (ntri, 3)

    return points, triangles


def solve_fem(m, n_modes, verbose=False):
    """
    Solve finite element eigenvalue problem on a mesh.

    Parameters
    ----------
    m : meshio.Mesh
        Input mesh
    n_modes : int
        Number of eigenmodes to compute
    verbose : bool
        Whether to print eigenvalues

    Returns
    -------
    mesh : skfem.MeshTri
        FEM mesh
    I : ndarray
        Interior node indices
    evals : ndarray
        Eigenvalues
    full_evecs : ndarray
        Eigenvectors (full mesh)
    """
    # --------------------------------------------------
    # 1. Read mesh from meshio
    # --------------------------------------------------

    points, triangles = meshio_to_points_triangles(m)

    points_c = np.ascontiguousarray(points.T)
    triangles_c = np.ascontiguousarray(triangles.T)
    mesh = MeshTri(points_c, triangles_c)

    # --------------------------------------------------
    # 2. Define finite-element basis
    # --------------------------------------------------
    element = ElementTriP1()   # piecewise linear basis
    basis = Basis(mesh, element)

    # --------------------------------------------------
    # 3. Assemble stiffness and mass matrices
    # --------------------------------------------------
    K = asm(laplace, basis)   # stiffness matrix
    M = asm(mass, basis)      # mass matrix

    # --------------------------------------------------
    # 4. Impose Dirichlet boundary conditions
    #    Here: all boundary nodes are fixed, u=0
    # --------------------------------------------------
    D = basis.get_dofs().all()
    Kc, Mc, _, I = condense(K, M, D=D)

    # --------------------------------------------------
    # 5. Solve generalized eigenvalue problem
    #    Kc @ u = lambda * Mc @ u
    # --------------------------------------------------
    evals, evecs = eigsh(Kc, M=Mc, k=n_modes, sigma=0.0, which="LM")

    # sort just to be safe
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    if verbose:
        print("Eigenvalues:")
        print(evals)

    # --------------------------------------------------
    # 6. Lift one eigenvector back to full mesh
    # --------------------------------------------------
    full_evecs = np.zeros((K.shape[0], n_modes))
    full_evecs[I, :] = evecs
    full_evecs = full_evecs.T

    return mesh, I, evals, full_evecs


def extract_zero_loops(points, triangles, u, level=0.0):
    """
    Extract zero contour loops from triangulated data.

    Parameters
    ----------
    points : ndarray, shape (n_points, 2)
        Point coordinates
    triangles : ndarray, shape (n_triangles, 3)
        Triangle connectivity
    u : ndarray
        Field values at points
    level : float
        Contour level

    Returns
    -------
    loops : list of ndarray
        List of contour loops
    """
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    fig, ax = plt.subplots()
    cs = ax.tricontour(triang, u, levels=[level])
    _x = cs.get_paths()[0]
    _loops = np.split(_x.vertices, np.where(_x.codes == 79)[0])

    loops = []
    for loop in _loops:
        if len(loop) < 3:
            continue
        c_loop = loop[1:]
        c_loop = np.vstack((c_loop, c_loop[0,None]))
        loops.append(c_loop)

    plt.close(fig)
    return loops


def plot_eigenmode(mesh, evals, evecs, mode_idx, show_zero_contours=True, cmap_name="YlGn"):
    """
    Plot an eigenmode.

    Parameters
    ----------
    mesh : skfem.MeshTri
        FEM mesh
    evals : ndarray
        Eigenvalues
    evecs : ndarray
        Eigenvectors
    mode_idx : int
        Index of mode to plot
    show_zero_contours : bool
        Whether to show zero contours
    cmap_name : str
        Name of matplotlib colormap to use
    """
    points = mesh.p.T
    triangles = mesh.t.T

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_axis_off()
    #ax.set_title('.6f')

    u = evecs[mode_idx, :]

    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    # Get colormap
    cmap = matplotlib.colormaps[cmap_name]

    # Define diverging colormaps
    diverging_cmaps = {
        'RdBu', 'RdYlBu', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
        'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    }

    if cmap_name in diverging_cmaps:
        # For diverging colormaps, center around zero
        max_abs = np.abs(u).max()
        ax.tricontourf(triang, u, levels=100, cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        # For other colormaps, plot absolute value
        ax.tricontourf(triang, np.abs(u), levels=100, cmap=cmap)

    if show_zero_contours:
        # Add zero contours
        loops = extract_zero_loops(points=points, triangles=triangles, u=u, level=1e-5)
        loops += extract_zero_loops(points=points, triangles=triangles, u=u, level=-1e-5)
        for loop in loops:
            ax.plot(loop[:, 0], loop[:, 1], lw=2, color=cmap(1.0))

    plt.tight_layout()
    plt.savefig(f"output/mode_{mode_idx:02d}.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Solve FEM eigenvalue problem and plot eigenmodes"
    )
    parser.add_argument(
        "mesh_file",
        help="Path to the mesh file (.msh)"
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=50,
        help="Number of eigenmodes to compute (default: 50)"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Index of mode to plot (default: 0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print eigenvalues"
    )
    parser.add_argument(
        "--no-zero-contours",
        action="store_true",
        help="Don't show zero contours on the plot"
    )
    parser.add_argument(
        "--colormap",
        default="YlGn",
        help="Matplotlib colormap to use (default: YlGn)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode >= args.n_modes:
        parser.error(f"Mode index ({args.mode}) must be less than number of modes ({args.n_modes})")

    # Load mesh
    print(f"Loading mesh from {args.mesh_file}")
    try:
        m = meshio.read(args.mesh_file)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # Solve FEM problem
    print(f"Solving FEM problem for {args.n_modes} modes...")
    try:
        fem_mesh, I, fem_evals, fem_evecs = solve_fem(m, args.n_modes, verbose=args.verbose)
    except Exception as e:
        print(f"Error solving FEM: {e}")
        return

    # Plot the specified mode
    print(f"Plotting mode {args.mode}...")
    try:
        plot_eigenmode(
            fem_mesh,
            fem_evals,
            fem_evecs,
            args.mode,
            show_zero_contours=not args.no_zero_contours,
            cmap_name=args.colormap
        )
    except Exception as e:
        print(f"Error plotting: {e}")
        return

    print("Done!")


if __name__ == "__main__":
    main()