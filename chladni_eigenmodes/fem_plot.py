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


def plot_eigenmode(mesh, evals, evecs, mode_indices, show_zero_contours=True, cmap_name="YlGn", save_path=None, scale=None):
    """Plot eigenmode(s) from FEM solution.
    
    Parameters
    ----------
    mode_indices : int or list of int
        Single mode index or list of indices to sum
    scale : float, optional
        Scale factor for arcsinh compression. Smaller values give more amplification.
        Defaults: 0.02 for diverging colormaps, 0.05 for others.
    """
    points = mesh.p.T
    triangles = mesh.t.T

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Handle both single mode and mode ranges
    if isinstance(mode_indices, int):
        u = evecs[mode_indices, :]
    else:
        u = np.sum(evecs[mode_indices], axis=0)
    
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    
    # Set axis limits to mesh bounds with small margin
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    margin = 0.01 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    cmap = matplotlib.colormaps[cmap_name]
    diverging_cmaps = {
        'RdBu', 'RdYlBu', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy',
        'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    }
    
    import matplotlib.colors as colors
    if cmap_name in diverging_cmaps:
        # Signed eigenmode: keep sign, but compress large amplitudes
        v = np.percentile(np.abs(u), 99.5)

        # Default or user-provided scale
        if scale is None:
            scale = 0.02 * v
        else:
            scale = scale * v

        u_plot = np.arcsinh(u / scale)

        v_plot = np.percentile(np.abs(u_plot), 99.5)

        ax.tricontourf(
            triang,
            u_plot,
            levels=100,
            cmap=cmap,
            vmin=-v_plot,
            vmax=v_plot,
        )

    else:
        # Amplitude plot: compress dynamic range
        amp = np.abs(u)

        v = np.percentile(amp, 99.5)
        
        # Default or user-provided scale
        if scale is None:
            scale = 0.05 * v
        else:
            scale = scale * v

        amp_plot = np.arcsinh(amp / scale)

        vmax = np.percentile(amp_plot, 99.5)

        ax.tricontourf(
            triang,
            amp_plot,
            levels=100,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )

    if show_zero_contours:
        loops = extract_zero_loops(points=points, triangles=triangles, u=u, level=1e-5)
        loops += extract_zero_loops(points=points, triangles=triangles, u=u, level=-1e-5)
        for loop in loops:
            ax.plot(loop[:, 0], loop[:, 1], lw=1,
                    color=cmap(1.0),
                    )

    plt.tight_layout(pad=0.0)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot a saved FEM eigenmode from HDF5 mode data.")
    parser.add_argument("hdf5_file", help="Path to the saved FEM HDF5 file")
    parser.add_argument("--mode", default="0", help="Mode index or range (e.g., '5' or '900:940') to plot (default: 0)")
    parser.add_argument("--scale", type=float, default=None, help="Scale factor for arcsinh compression (smaller = more amplification). Defaults: 0.02 for diverging, 0.05 for others")
    parser.add_argument("--no-zero-contours", action="store_true", help="Don't show zero contours on the plot")
    parser.add_argument("--colormap", default="YlGn", help="Matplotlib colormap to use (default: YlGn)")
    parser.add_argument("--save", default=None, help="Optional path to save the plotted image")

    args = parser.parse_args()

    # Parse mode argument: can be a single int or a range like "900:940"
    if ":" in args.mode:
        mode_parts = args.mode.split(":")
        mode_start = int(mode_parts[0])
        mode_end = int(mode_parts[1])
        mode_indices = list(range(mode_start, mode_end))
        mode_label = f"modes_{mode_start}_{mode_end}"
    else:
        mode_indices = int(args.mode)
        mode_label = f"mode_{mode_indices}"

    data = load_fem_solution_hdf5(args.hdf5_file)
    
    # Validation
    if isinstance(mode_indices, int):
        if mode_indices >= data["eigenvectors"].shape[0]:
            parser.error(f"Mode index ({mode_indices}) must be less than number of modes ({data['eigenvectors'].shape[0]})")
    else:
        if max(mode_indices) >= data["eigenvectors"].shape[0]:
            parser.error(f"Mode range end ({max(mode_indices)}) must be less than number of modes ({data['eigenvectors'].shape[0]})")

    mesh = load_skfem_mesh_from_hdf5(args.hdf5_file)
    save_path = args.save
    if save_path is None:
        hdf5_path = Path(args.hdf5_file)
        save_path = hdf5_path.with_name(f"{hdf5_path.stem}_{mode_label}.png")
    plot_eigenmode(
        mesh,
        data["eigenvalues"],
        data["eigenvectors"],
        mode_indices,
        show_zero_contours=not args.no_zero_contours,
        cmap_name=args.colormap,
        save_path=save_path,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
