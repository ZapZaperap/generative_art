"""Load saved FEM mode data and plot a selected eigenmode."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib
from pathlib import Path

from fem_utils import load_fem_solution_hdf5, load_skfem_mesh_from_hdf5


def extract_zero_loops(points, triangles, u, level=0.0):
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
        c_loop = np.vstack((c_loop, c_loop[0, None]))
        loops.append(c_loop)

    plt.close(fig)
    return loops


def plot_eigenmode(mesh, evals, evecs, mode_idx, show_zero_contours=True, cmap_name="YlGn", save_path=None):
    points = mesh.p.T
    triangles = mesh.t.T

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_axis_off()

    u = evecs[mode_idx, :]
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    cmap = matplotlib.colormaps[cmap_name]
    diverging_cmaps = {
        'RdBu', 'RdYlBu', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
        'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    }

    if cmap_name in diverging_cmaps:
        max_abs = np.abs(u).max()
        ax.tricontourf(triang, u, levels=100, cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        ax.tricontourf(triang, np.abs(u), levels=100, cmap=cmap)

    if show_zero_contours:
        loops = extract_zero_loops(points=points, triangles=triangles, u=u, level=1e-5)
        loops += extract_zero_loops(points=points, triangles=triangles, u=u, level=-1e-5)
        for loop in loops:
            ax.plot(loop[:, 0], loop[:, 1], lw=2, color=cmap(1.0))

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot a saved FEM eigenmode from HDF5 mode data.")
    parser.add_argument("hdf5_file", help="Path to the saved FEM HDF5 file")
    parser.add_argument("--mode", type=int, default=0, help="Index of mode to plot (default: 0)")
    parser.add_argument("--no-zero-contours", action="store_true", help="Don't show zero contours on the plot")
    parser.add_argument("--colormap", default="YlGn", help="Matplotlib colormap to use (default: YlGn)")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")

    args = parser.parse_args()

    data = load_fem_solution_hdf5(args.hdf5_file)
    if args.mode >= data["eigenvectors"].shape[0]:
        parser.error(f"Mode index ({args.mode}) must be less than number of modes ({data['eigenvectors'].shape[0]})")

    mesh = load_skfem_mesh_from_hdf5(args.hdf5_file)
    save_path = args.save
    if save_path is None:
        save_path = Path(args.hdf5_file).parent / f"mode_{args.mode}.png"
        print (save_path)
    plot_eigenmode(
        mesh,
        data["eigenvalues"],
        data["eigenvectors"],
        args.mode,
        show_zero_contours=not args.no_zero_contours,
        cmap_name=args.colormap,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
