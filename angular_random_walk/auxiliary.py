"""Small geometry and display utilities used by the generative-art scripts.

The mathematical helpers in this module are intentionally lightweight and have no
side effects. Display helpers are optional conveniences for interactive use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.interpolate as interpolate
from PIL import Image


def show_image(file_path: str | Path) -> None:
    """Open an image with the default PIL image viewer.

    This is mainly useful during local experimentation. Repository scripts should
    not depend on it for rendering.
    """

    Image.open(file_path).show()


def bspline(
    control_vertices: np.ndarray,
    n_samples: int = 100,
    degree: int = 3,
    periodic: bool = False,
) -> np.ndarray:
    """Sample a B-spline curve through a set of control vertices.

    Parameters
    ----------
    control_vertices:
        Array of shape ``(n_vertices, dimension)`` containing the control points.
    n_samples:
        Number of points sampled along the spline.
    degree:
        Degree of the spline. The value is clipped to a valid range for the
        supplied number of vertices.
    periodic:
        If ``True``, return a closed periodic spline. Otherwise, return an open
        spline.

    Returns
    -------
    np.ndarray
        Sampled points with shape ``(n_samples, dimension)``.
    """

    vertices = np.asarray(control_vertices, dtype=float)
    if vertices.ndim != 2:
        raise ValueError("control_vertices must be a two-dimensional array")

    count = len(vertices)
    if count < 2:
        raise ValueError("at least two control vertices are required")

    if periodic:
        factor, fraction = divmod(count + degree + 1, count)
        vertices = np.concatenate((vertices,) * factor + (vertices[:fraction],))
        count = len(vertices)
        degree = int(np.clip(degree, 1, degree))
        knot_vector = np.arange(0 - degree, count + degree + degree - 1, dtype=int)
        sample_positions = np.linspace(1, count - degree, n_samples)
    else:
        degree = int(np.clip(degree, 1, count - 1))
        knot_vector = np.concatenate(
            ([0] * degree, np.arange(count - degree + 1), [count - degree] * degree)
        )
        sample_positions = np.linspace(0, count - degree, n_samples)

    return np.array(
        interpolate.splev(sample_positions, (knot_vector, vertices.T, degree))
    ).T


def round_to_odd(value: float) -> int:
    """Round up to the next odd integer."""

    return int(np.ceil(value) // 2 * 2 + 1)


def round_to_even(value: float) -> int:
    """Round up to the next even integer."""

    return int(np.ceil(value / 2.0) * 2)


def rotation_matrix(angle: float) -> np.ndarray:
    """Return the two-dimensional rotation matrix for ``angle`` radians."""

    return np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )


def mirror_matrix(line_vector: Optional[np.ndarray]) -> np.ndarray:
    """Return the matrix that mirrors points at the line spanned by a vector.

    Passing ``None`` returns the identity matrix.
    """

    if line_vector is None:
        return np.eye(2)

    vector = np.asarray(line_vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("line_vector must be non-zero")

    matrix = np.array(
        [
            [vector[0] ** 2 - vector[1] ** 2, 2 * vector[0] * vector[1]],
            [2 * vector[0] * vector[1], vector[1] ** 2 - vector[0] ** 2],
        ],
        dtype=float,
    )
    return matrix / norm**2


# Backward-compatible aliases for older sketches.
show_img_pil = show_image
rot_matrix = rotation_matrix