"""Generate angular random-walk artwork.

The script creates a collection of PNG files by drawing random walks composed of
alternating circular arcs. Output files are written to an ``output`` directory
located next to this script.

Example:
    python angular_random_walk.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cairo
import matplotlib
import numpy as np
from matplotlib.colors import Colormap

# Canvas dimensions in pixels. Drawing coordinates are normalized to [0, 1].
WIDTH = 1000
HEIGHT = 1000
BACKGROUND_RGB = (0.0, 0.0, 0.0)
CLIP_RADIUS = 0.45
CENTER = np.array([0.5, 0.5])


@dataclass(frozen=True)
class WalkParameters:
    """Parameters controlling one family of random walks."""

    steps_per_walk: int
    number_of_walks: int
    radius: float
    max_step_angle: float


@dataclass(frozen=True)
class ColorMapConfig:
    """Configuration for coloring one generated image."""

    name: str
    invert: bool = False


COLOR_MAPS: tuple[ColorMapConfig, ...] = (
    ColorMapConfig("cividis", invert=True),
    ColorMapConfig("YlGnBu"),
    ColorMapConfig("spring"),
    ColorMapConfig("twilight_shifted", invert=True),
    ColorMapConfig("tab20"),
    ColorMapConfig("pink", invert=True),
    ColorMapConfig("viridis", invert=True),
)

WALK_PARAMETER_SETS: tuple[WalkParameters, ...] = (
    WalkParameters(steps_per_walk=50, number_of_walks=200, radius=0.02, max_step_angle=1.5),
    WalkParameters(steps_per_walk=100, number_of_walks=200, radius=0.02, max_step_angle=0.5),
    WalkParameters(steps_per_walk=100, number_of_walks=200, radius=0.02, max_step_angle=np.pi),
)

DrawCallback = Callable[["ArcWalkContext"], None]


class ArcWalkContext:
    """Stateful wrapper around a Cairo context for drawing arc walks.

    Some pycairo/cairocffi installations do not support subclassing
    ``cairo.Context`` reliably. This class therefore uses composition: it owns a
    real ``cairo.Context`` and forwards unknown attributes/methods to it.
    """

    def __init__(self, surface: cairo.Surface, color_map: Colormap, invert_color_map: bool = False) -> None:
        self.context = cairo.Context(surface)
        self.current_angle = 0.0
        self.reference_radius: float | None = None
        self.current_center = CENTER.copy()
        self.arc_counter = 0
        self.total_arc_length = 0.0
        self.current_path_length = 0.0
        self.color_map = color_map
        self.invert_color_map = invert_color_map

    def __getattr__(self, name: str):
        """Forward Cairo drawing calls to the wrapped context."""
        return getattr(self.context, name)

    def draw_arc_segment(
        self,
        radius: float,
        angle: float,
        *,
        start_angle: float | None = None,
        center: np.ndarray | None = None,
        relative_angle: bool = True,
        callback: DrawCallback | None = None,
        new_sub_path: bool = False,
    ) -> None:
        """Draw one arc segment and update the internal walk state.

        Args:
            radius: Radius of the arc in normalized canvas units.
            angle: Arc angle. Interpreted relative to the current tangent angle
                unless ``relative_angle`` is false.
            start_angle: Optional starting angle for the first segment.
            center: Optional starting center for the first segment.
            relative_angle: Whether ``angle`` is relative to the current tangent.
            callback: Optional function called after creating the arc path.
            new_sub_path: Start the arc as a new Cairo sub-path.
        """
        if start_angle is not None:
            self.current_angle = start_angle
        if center is not None:
            self.current_center = center.copy()
        if self.reference_radius is None:
            self.reference_radius = radius

        arc_function = self.arc_negative if self.arc_counter % 2 == 0 else self.arc

        if relative_angle:
            next_angle = self.current_angle - (-1) ** self.arc_counter * angle
        else:
            next_angle = angle

        previous_center = center if center is not None else np.array(self.get_current_point())
        self.current_center -= (1 + radius / self.reference_radius) * (self.current_center - previous_center)

        if new_sub_path:
            self.new_sub_path()

        arc_function(*self.current_center, radius, self.current_angle, next_angle)

        if callback is not None:
            callback(self)

        self.current_angle = np.mod(next_angle, 2 * np.pi)
        self.current_angle += np.pi if self.current_angle > np.pi else -np.pi
        self.arc_counter += 1

    def draw_arc_path(
        self,
        radii: float | Sequence[float],
        angles: float | Sequence[float],
        *,
        start_angle: float | None = None,
        center: np.ndarray | None = None,
        relative_angle: bool = True,
        initial_arc_counter: int | str = 0,
        segment_callback: DrawCallback | None = None,
        new_sub_path: bool = False,
    ) -> None:
        """Draw a path consisting of multiple arc segments.

        ``radii`` and ``angles`` may either both be sequences of equal length or
        one may be a scalar, in which case it is repeated to match the other.
        """
        radius_array, angle_array = _broadcast_walk_inputs(radii, angles)
        self.total_arc_length = float(np.sum(radius_array * angle_array))
        self.current_path_length = 0.0

        if initial_arc_counter == "random":
            self.arc_counter = int(np.random.randint(2))
        elif isinstance(initial_arc_counter, int):
            self.arc_counter = initial_arc_counter
        else:
            raise ValueError("initial_arc_counter must be an int or 'random'.")

        for index, (radius, angle) in enumerate(zip(radius_array, angle_array, strict=True)):
            self.draw_arc_segment(
                float(radius),
                float(angle),
                start_angle=start_angle if index == 0 else None,
                center=center if index == 0 else None,
                relative_angle=relative_angle,
                callback=segment_callback,
                new_sub_path=new_sub_path,
            )
            self.current_path_length += float(radius * angle)

        self.arc_counter = 0


def _broadcast_walk_inputs(
    radii: float | Sequence[float], angles: float | Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return radius and angle arrays of equal length."""
    radii_is_scalar = np.isscalar(radii)
    angles_is_scalar = np.isscalar(angles)

    if radii_is_scalar and angles_is_scalar:
        raise ValueError("At least one of radii or angles must be a sequence.")

    if radii_is_scalar:
        angle_array = np.asarray(angles, dtype=float)
        radius_array = np.full_like(angle_array, float(radii), dtype=float)
    elif angles_is_scalar:
        radius_array = np.asarray(radii, dtype=float)
        angle_array = np.full_like(radius_array, float(angles), dtype=float)
    else:
        radius_array = np.asarray(radii, dtype=float)
        angle_array = np.asarray(angles, dtype=float)
        if radius_array.shape != angle_array.shape:
            raise ValueError("radii and angles must have the same length.")

    return radius_array, angle_array


def stroke_segment_with_fading_color(ctx: ArcWalkContext) -> None:
    """Stroke the current segment with a color and opacity based on path length."""
    current_point = ctx.get_current_point()
    path_fraction = min(ctx.current_path_length / ctx.total_arc_length, 1.0)
    opacity = (1.0 - path_fraction) ** 0.25
    color_fraction = 1.0 - path_fraction if ctx.invert_color_map else path_fraction
    color = ctx.color_map(color_fraction)

    ctx.set_source_rgba(*color[:-1], opacity)
    ctx.set_line_width(0.002)
    ctx.stroke()
    ctx.move_to(*current_point)


def paint_background(ctx: cairo.Context) -> None:
    """Fill the full canvas with the configured background color."""
    ctx.save()
    ctx.set_source_rgb(*BACKGROUND_RGB)
    ctx.paint()
    ctx.restore()


def apply_circular_clip(ctx: cairo.Context, center: np.ndarray = CENTER, radius: float = CLIP_RADIUS) -> None:
    """Restrict subsequent drawing to a circle."""
    ctx.arc(*center, radius, 0.0, 2.0 * np.pi)
    ctx.clip()


def draw_center_disc(ctx: ArcWalkContext, center: np.ndarray, radius: float) -> None:
    """Draw the central black disc and a colored outline."""
    ctx.arc(*center, radius, 0.0, 2.0 * np.pi)
    ctx.set_source_rgba(*BACKGROUND_RGB, 1.0)
    ctx.fill()

    outline_color = ctx.color_map(1.0 if ctx.invert_color_map else 0.0)
    ctx.arc(*center, radius, 0.0, 2.0 * np.pi)
    ctx.set_source_rgba(*outline_color[:-1], 1.0)
    ctx.set_line_width(0.004)
    ctx.stroke()


def render_image(parameters: WalkParameters, color_config: ColorMapConfig) -> cairo.ImageSurface:
    """Render one random-walk artwork and return the Cairo surface."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    color_map = matplotlib.colormaps[color_config.name]
    ctx = ArcWalkContext(surface, color_map=color_map, invert_color_map=color_config.invert)
    ctx.scale(WIDTH, HEIGHT)

    paint_background(ctx)
    apply_circular_clip(ctx)

    for _ in range(parameters.number_of_walks):
        start_angle = np.random.random() * 2.0 * np.pi
        random_angles = np.random.random(parameters.steps_per_walk) * parameters.max_step_angle

        ctx.set_dash([1.0, 0.0])
        ctx.draw_arc_path(
            parameters.radius,
            random_angles,
            center=CENTER,
            start_angle=start_angle,
            initial_arc_counter=0,
            segment_callback=stroke_segment_with_fading_color,
            new_sub_path=True,
        )

    draw_center_disc(ctx, CENTER, parameters.radius)
    return surface


def output_directory() -> Path:
    """Return the output directory next to this script and create it if needed."""
    directory = Path(__file__).resolve().parent / "output"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main() -> None:
    """Generate all configured images."""
    out_dir = output_directory()

    for parameter_index, parameters in enumerate(WALK_PARAMETER_SETS):
        for color_config in COLOR_MAPS:
            surface = render_image(parameters, color_config)
            output_path = out_dir / f"{parameter_index}_{color_config.name}.png"
            surface.write_to_png(str(output_path))
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()