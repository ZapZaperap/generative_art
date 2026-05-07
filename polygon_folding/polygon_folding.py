#!/usr/bin/env python3
"""Animate a polygon-folding style generative-art construction.

The animation starts from points on a regular n-gon. In the first phase, the odd
vertices are radially contracted. In the second phase, pairs of vertices are
folded/rotated into a new configuration. At each frame, all pairwise connecting
lines are drawn.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


def positive_int(value: str) -> int:
    """Argparse type for strictly positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """Argparse type for strictly positive floats."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def nonnegative_float(value: str) -> float:
    """Argparse type for non-negative floats."""
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed

def even_int_at_least_4(value):
    n = int(value)

    if n < 4:
        raise argparse.ArgumentTypeError(
            "n must be at least 4"
        )

    if n % 2 != 0:
        raise argparse.ArgumentTypeError(
            "n must be an even number"
        )

    return n

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an animation in which points on a regular polygon are "
            "contracted and folded while all pairwise connecting lines are drawn."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--n-points",
        type=even_int_at_least_4,
        default=20,
        help="Number of polygon vertices (even integer >= 4).",
    )
    parser.add_argument(
        "--contract-frames",
        type=positive_int,
        default=100,
        help="Number of frames for the radial contraction phase.",
    )
    parser.add_argument(
        "--fold-frames",
        type=positive_int,
        default=100,
        help="Number of frames for the folding/rotation phase.",
    )
    parser.add_argument(
        "--min-radius",
        type=positive_float,
        default=1e-5,
        help="Final radius of the contracted odd vertices in the first phase.",
    )
    parser.add_argument(
        "--contraction-power",
        type=positive_float,
        default=0.75,
        help="Power-law exponent controlling the radial contraction curve.",
    )
    parser.add_argument(
        "--fold-turns",
        type=float,
        default=1.0,
        help="Final folding angle in units of 2π/n. For example, 1 means 2π/n.",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=30,
        help="Frames per second of the saved animation.",
    )
    parser.add_argument(
        "--interval-ms",
        type=positive_int,
        default=30,
        help="Delay between frames in milliseconds for interactive playback.",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=200,
        help="Resolution of the saved animation.",
    )
    parser.add_argument(
        "--figsize",
        type=positive_float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(6.4, 4.8),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--line-width",
        type=nonnegative_float,
        default=1.0,
        help="Width of the connecting lines.",
    )
    parser.add_argument(
        "--line-color",
        default="black",
        help="Matplotlib-compatible line color.",
    )
    parser.add_argument(
        "--margin",
        type=nonnegative_float,
        default=0.1,
        help="Extra plot margin around the unit circle.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/animation.mp4"),
        help="Output filename. The parent directory is created automatically.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the animation window instead of only saving the file.",
    )

    return parser


def polygon_vertices(n_points: int) -> np.ndarray:
    """Return vertices of a regular polygon as complex numbers on the unit circle."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False) - np.pi / n_points
    return np.exp(1j * angles)


def smooth_step(n_frames: int) -> np.ndarray:
    """Smooth interpolation from 0 to 1 with zero initial slope."""
    t = np.linspace(0.0, 1.0, n_frames)
    return 1.0 - np.cos(0.5 * np.pi * t) ** 2


def contracted_points(
    base_vertices: np.ndarray,
    frame: int,
    n_frames: int,
    min_radius: float,
    contraction_power: float,
) -> np.ndarray:
    """Return polygon points during the contraction phase."""
    radii = np.linspace(1.0, min_radius, n_frames) ** contraction_power
    points = base_vertices.copy()
    points[1::2] *= radii[frame]
    return points


def folded_points(
    base_vertices: np.ndarray,
    frame: int,
    n_frames: int,
    fold_turns: float,
) -> np.ndarray:
    """Return polygon points during the folding/rotation phase.

    The construction intentionally mirrors the original sketch: even vertices are
    copied into the odd slots, while the even vertices are rotated by a smooth
    angle. For an odd number of points, the final unpaired odd slot remains at 0.
    """
    n_points = base_vertices.size
    points = np.zeros(n_points, dtype=complex)

    even_vertices = base_vertices[::2]
    points[::2] = even_vertices
    points[1::2] = even_vertices[: points[1::2].size]

    final_angle = fold_turns * 2.0 * np.pi / n_points
    theta = smooth_step(n_frames)[frame] * final_angle
    points[::2] *= np.exp(1j * theta)
    return points


def points_for_frame(
    frame: int,
    base_vertices: np.ndarray,
    contract_frames: int,
    fold_frames: int,
    min_radius: float,
    contraction_power: float,
    fold_turns: float,
) -> np.ndarray:
    """Return the complex points for an animation frame."""
    if frame < contract_frames:
        return contracted_points(
            base_vertices=base_vertices,
            frame=frame,
            n_frames=contract_frames,
            min_radius=min_radius,
            contraction_power=contraction_power,
        )

    return folded_points(
        base_vertices=base_vertices,
        frame=frame - contract_frames,
        n_frames=fold_frames,
        fold_turns=fold_turns,
    )


def line_segments(points: Iterable[complex]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return all pairwise connecting line segments between complex points."""
    return [
        ((z1.real, z1.imag), (z2.real, z2.imag))
        for z1, z2 in itertools.combinations(points, 2)
    ]


def build_animation(args: argparse.Namespace) -> tuple[FuncAnimation, plt.Figure]:
    """Create the matplotlib animation and return it together with the figure."""
    base_vertices = polygon_vertices(args.n_points)
    total_frames = args.contract_frames + args.fold_frames

    fig, ax = plt.subplots(figsize=args.figsize)
    line_collection = LineCollection(
        [], linewidths=args.line_width, colors=args.line_color
    )
    ax.add_collection(line_collection)

    limit = 1.0 + args.margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_axis_off()

    def update(frame: int) -> tuple[LineCollection]:
        points = points_for_frame(
            frame=frame,
            base_vertices=base_vertices,
            contract_frames=args.contract_frames,
            fold_frames=args.fold_frames,
            min_radius=args.min_radius,
            contraction_power=args.contraction_power,
            fold_turns=args.fold_turns,
        )
        line_collection.set_segments(line_segments(points))
        return (line_collection,)

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=args.interval_ms,
        blit=True,
    )
    return animation, fig


def main() -> None:
    args = create_argument_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    animation, fig = build_animation(args)
    animation.save(args.output, fps=args.fps, dpi=args.dpi)

    #from matplotlib.animation import PillowWriter
    #animation.save("output/test.gif", writer=PillowWriter(fps=args.fps), dpi=args.dpi,)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved animation to {args.output}")


if __name__ == "__main__":
    main()