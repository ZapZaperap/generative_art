"""Render the basin structure of a damped magnetic pendulum.

The script integrates the pendulum ODE for a rectangular grid of initial
positions and stores the nearest final attractor for each initial condition in
an HDF5 file.  Work is split into row chunks and evaluated with a
``ProcessPoolExecutor`` to keep multiprocessing explicit and easy to inspect.

Example
-------
Run with the defaults::

    python render_pendulum.py

Use fewer grid points for a quick test::

    python render_pendulum.py --grid-size 200 --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import RK45

from pendulum_tools import PendulumParameters, make_ode, nearest_attractor_index, regular_polygon_magnets


def iter_row_chunks(n_rows: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    """Yield half-open row intervals ``[start, stop)``."""

    for start in range(0, n_rows, chunk_size):
        yield start, min(start + chunk_size, n_rows)


def integrate_initial_condition(
    x: float,
    y: float,
    t_bound: float,
    parameters: PendulumParameters,
) -> int:
    """Integrate one initial condition and classify its final attractor."""

    rhs = make_ode(parameters)
    solver = RK45(rhs, t0=0.0, y0=np.array([x, y, 0.0, 0.0], dtype=float), t_bound=t_bound)

    while solver.status == "running":
        solver.step()

    return nearest_attractor_index(solver.y[:2], parameters.magnet_positions)


def compute_row_chunk(
    row_range: tuple[int, int],
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    t_bound: float,
    parameters: PendulumParameters,
) -> tuple[int, NDArray[np.uint16]]:
    """Compute one block of rows for the output image."""

    start, stop = row_range
    block = np.empty((stop - start, len(y_values)), dtype=np.uint16)

    rhs = make_ode(parameters)
    attractors = np.vstack((np.zeros((1, 2), dtype=float), parameters.magnet_positions))

    for local_ix, x in enumerate(x_values[start:stop]):
        for iy, y in enumerate(y_values):
            solver = RK45(rhs, t0=0.0, y0=np.array([x, y, 0.0, 0.0], dtype=float), t_bound=t_bound)
            while solver.status == "running":
                solver.step()

            distances = np.linalg.norm(attractors - solver.y[:2], axis=1)
            block[local_ix, iy] = np.argmin(distances)

    return start, block


def save_hdf5(
    output_path: Path,
    attractor_map: NDArray[np.uint16],
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    parameters: PendulumParameters,
    t_bound: float,
) -> None:
    """Save simulation output and metadata to an HDF5 file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        h5.attrs["description"] = "Magnetic pendulum attractor map"
        h5.attrs["t_bound"] = t_bound

        h5.create_dataset("attractor_map", data=attractor_map, compression="gzip", shuffle=True)
        h5.create_dataset("x_values", data=x_values)
        h5.create_dataset("y_values", data=y_values)

        parameter_group = h5.create_group("parameters")
        parameter_group.attrs["magnet_height"] = parameters.magnet_height
        parameter_group.attrs["mass"] = parameters.mass
        parameter_group.attrs["restoring_force"] = parameters.restoring_force
        parameter_group.attrs["damping"] = parameters.damping
        parameter_group.create_dataset("magnet_positions", data=parameters.magnet_positions)
        parameter_group.create_dataset("magnet_strengths", data=parameters.magnet_strengths)


def build_parameters(n_magnets: int, magnet_radius: float) -> PendulumParameters:
    """Create the default magnetic pendulum configuration."""

    magnet_positions = regular_polygon_magnets(n_magnets=n_magnets, radius=magnet_radius)
    return PendulumParameters(
        magnet_height=0.2,
        mass=1.0,
        restoring_force=0.5,
        magnet_positions=magnet_positions,
        magnet_strengths=np.ones(n_magnets, dtype=float),
        damping=0.15,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=1000, help="Number of x and y samples.")
    parser.add_argument("--extent", type=float, default=8.0, help="Sample x and y from [-extent, extent].")
    parser.add_argument("--t-bound", type=float, default=50.0, help="Final integration time.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes. Defaults to os.cpu_count().")
    parser.add_argument("--chunk-size", type=int, default=8, help="Number of x rows per worker task.")
    parser.add_argument("--n-magnets", type=int, default=6, help="Number of magnets on the regular polygon.")
    parser.add_argument("--magnet-radius", type=float, default=3.2, help="Radius of the magnet polygon.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "pendulum_attractors.h5",
        help="Output HDF5 path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the simulation."""

    args = parse_args()
    parameters = build_parameters(n_magnets=args.n_magnets, magnet_radius=args.magnet_radius)

    x_values = np.linspace(-args.extent, args.extent, args.grid_size, dtype=float)
    y_values = np.linspace(-args.extent, args.extent, args.grid_size, dtype=float)
    attractor_map = np.empty((len(x_values), len(y_values)), dtype=np.uint16)

    row_chunks = list(iter_row_chunks(len(x_values), args.chunk_size))
    completed_rows = 0

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(compute_row_chunk, row_range, x_values, y_values, args.t_bound, parameters)
                for row_range in row_chunks
            ]

            for future in as_completed(futures):
                start, block = future.result()
                stop = start + block.shape[0]
                attractor_map[start:stop, :] = block

                completed_rows += block.shape[0]
                print(f"Completed {completed_rows}/{len(x_values)} rows", end="\r", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted before writing output.")
        raise SystemExit(130)

    print("\nWriting output...")
    save_hdf5(args.output, attractor_map, x_values, y_values, parameters, args.t_bound)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()