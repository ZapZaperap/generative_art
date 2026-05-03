"""Numerical model for a damped magnetic pendulum.

The state vector is ``(x, y, vx, vy)``.  The pendulum is subject to

* a linear restoring force,
* viscous damping, and
* attractive inverse-square-like forces from fixed magnets.

The functions in this module are deliberately independent of plotting,
parallelization, and file I/O so they can be reused from scripts, notebooks,
or tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

State = NDArray[np.float64]
VectorArray = NDArray[np.float64]
OdeFunction = Callable[[float, State], State]


@dataclass(frozen=True)
class PendulumParameters:
    """Physical parameters for the magnetic pendulum model."""

    magnet_height: float
    mass: float
    restoring_force: float
    magnet_positions: VectorArray
    magnet_strengths: NDArray[np.float64]
    damping: float

    def __post_init__(self) -> None:
        positions = np.asarray(self.magnet_positions, dtype=float)
        strengths = np.asarray(self.magnet_strengths, dtype=float)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("magnet_positions must have shape (n_magnets, 2).")
        if strengths.ndim != 1 or strengths.shape[0] != positions.shape[0]:
            raise ValueError("magnet_strengths must have length n_magnets.")

        object.__setattr__(self, "magnet_positions", positions)
        object.__setattr__(self, "magnet_strengths", strengths)

    @property
    def n_magnets(self) -> int:
        """Number of external magnets."""

        return int(self.magnet_positions.shape[0])


def regular_polygon_magnets(
    n_magnets: int,
    radius: float,
    phase: float = 0.0,
) -> VectorArray:
    """Return magnet positions on a regular polygon around the origin."""

    angles = np.linspace(0.0, 2.0 * np.pi, n_magnets, endpoint=False) + phase
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def energy_components(state: ArrayLike, parameters: PendulumParameters) -> tuple[float, float, float]:
    """Return kinetic, restoring-potential, and magnetic-potential energy."""

    state_arr = np.asarray(state, dtype=float)
    position = state_arr[:2]
    velocity = state_arr[2:]

    kinetic = parameters.mass * np.linalg.norm(velocity) ** 2 / 2.0
    restoring = parameters.restoring_force * np.linalg.norm(position) ** 2 / 2.0

    magnetic = 0.0
    for strength, magnet_position in zip(parameters.magnet_strengths, parameters.magnet_positions):
        displacement_3d = np.array(
            [
                position[0] - magnet_position[0],
                position[1] - magnet_position[1],
                parameters.magnet_height,
            ],
            dtype=float,
        )
        magnetic -= parameters.mass * strength / np.linalg.norm(displacement_3d)

    return float(kinetic), float(restoring), float(magnetic)


def make_ode(parameters: PendulumParameters) -> OdeFunction:
    """Create the ODE right-hand side for ``scipy.integrate.RK45``.

    Parameters
    ----------
    parameters:
        Model parameters. Values are captured in local NumPy arrays so worker
        processes do not need to repeatedly access a mutable global dictionary.
    """

    magnet_height = parameters.magnet_height
    mass = parameters.mass
    restoring_force = parameters.restoring_force
    damping = parameters.damping
    magnet_strengths = parameters.magnet_strengths
    magnet_positions = parameters.magnet_positions

    def rhs(_time: float, state: State) -> State:
        position = state[:2]
        velocity = state[2:]

        derivative = np.zeros(4, dtype=float)
        derivative[:2] = velocity

        # Linear restoring force and viscous damping.
        derivative[2:] += -(restoring_force / mass) * position
        derivative[2:] += -(damping / mass) * velocity

        # Attractive magnetic forces.
        for strength, magnet_position in zip(magnet_strengths, magnet_positions):
            displacement = position - magnet_position
            denominator = (np.linalg.norm(displacement) ** 2 + magnet_height**2) ** 1.5
            derivative[2:] += -strength * displacement / denominator

        return derivative

    return rhs

def nearest_attractor_index(position: ArrayLike, magnet_positions: VectorArray) -> int:
    """Return the nearest attractor index for a final position.

    Index ``0`` corresponds to the origin.  Magnet indices start at ``1`` in
    the order supplied by ``magnet_positions``.
    """

    attractors = np.vstack((np.zeros((1, 2), dtype=float), np.asarray(magnet_positions, dtype=float)))
    distances = np.linalg.norm(attractors - np.asarray(position, dtype=float), axis=1)
    return int(np.argmin(distances))