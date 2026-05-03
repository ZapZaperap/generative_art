import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Plot attractor map from HDF5.")
    parser.add_argument("run_name", type=str, help="Name of the run (without .h5)")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap name")
    parser.add_argument("--border-width", type=float, default=0.05, help="Border thickness")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "output"

    h5_path = out_dir / f"{args.run_name}.h5"
    png_path = out_dir / f"{args.run_name}.png"

    # --- load data ---
    with h5py.File(h5_path, "r") as file:
        out_arr = file["attractor_map"][:]

    # --- colormap ---
    cmap = matplotlib.colormaps[args.cmap]

    # pick a color *outside* normal range → not used in imshow
    border_color = cmap(-0.1)

    # --- plot ---
    rand = args.border_width
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(
        out_arr,
        origin="lower",
        cmap=cmap,
        extent=[0, 1, 0, 1],
        vmin=0, vmax=out_arr.max(),
        zorder=100,
    )

    facecolor=(0,0,0,1)
    #facecolor=cmap(0.0)
    rect = patches.Rectangle(
        (-rand, -rand),
        1 + rand,
        1 + rand,
        linewidth=0,
        edgecolor="none",
        facecolor=facecolor,
        transform=ax.transAxes,
        zorder=0,
    )

    ax.add_patch(rect)

    ax.set_xlim(-rand, 1 + rand)
    ax.set_ylim(-rand, 1 + rand)
    ax.axis("off")

    plt.savefig(png_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()