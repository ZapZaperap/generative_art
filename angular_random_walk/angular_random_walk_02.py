"""Generate radial angular random-walk artwork.

The script draws random angular walks made of circular arcs with increasing
radius. The resulting PNG is written to an ``output`` directory located next to
this script.

Example:
    python angular_radial_walk.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cairo
import numpy as np

# Canvas dimensions in pixels. Drawing coordinates are normalized to [0, 1].
WIDTH = 1000
HEIGHT = 1000
CENTER = np.array([0.5, 0.5])
BACKGROUND_RGB = (0.0, 0.0, 0.0)
STROKE_RGBA = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class WalkConfig:
    """Parameters controlling the generated walk image."""

    steps_per_walk: int = 50
    number_of_walks: int = 100
    radius_step: float = 0.02
    step_angle: float = 2.0 * np.pi / 10.0
    line_width: float = 0.002
    output_filename: str = "angular_random_2.png"


def script_directory() -> Path:
    """Return the directory containing this script.

    ``Path(__file__).resolve().parent`` is the standard way to make paths
    relative to the script location instead of the current working directory.
    """

    return Path(__file__).resolve().parent


def output_directory() -> Path:
    """Create and return the output directory next to this script."""

    directory = script_directory() / "output"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def paint_background(ctx: cairo.Context) -> None:
    """Fill the complete canvas with the background color."""

    ctx.save()
    ctx.set_source_rgb(*BACKGROUND_RGB)
    ctx.paint()
    ctx.restore()


def random_signed_steps(number_of_steps: int, step_angle: float) -> np.ndarray:
    """Return random clockwise/counter-clockwise angular steps.

    Each step is either ``+step_angle`` or ``-step_angle`` with equal
    probability.
    """

    signs = 2 * np.random.randint(2, size=number_of_steps) - 1
    return signs * step_angle


def draw_single_walk(ctx: cairo.Context, config: WalkConfig) -> None:
    """Draw one angular random walk centered around ``CENTER``."""

    current_angle = np.random.random() * 2.0 * np.pi
    angular_steps = random_signed_steps(config.steps_per_walk, config.step_angle)

    ctx.move_to(*CENTER)

    for index, step in enumerate(angular_steps):
        radius = index * config.radius_step
        next_angle = current_angle + step

        if step > 0.0:
            ctx.arc(*CENTER, radius, current_angle, next_angle)
        else:
            ctx.arc_negative(*CENTER, radius, current_angle, next_angle)

        current_angle = np.mod(next_angle, 2.0 * np.pi)

    ctx.set_source_rgba(*STROKE_RGBA)
    ctx.set_line_width(config.line_width)
    ctx.stroke()


def render_image(config: WalkConfig) -> cairo.ImageSurface:
    """Render the complete image and return the Cairo surface."""

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    ctx.scale(WIDTH, HEIGHT)

    paint_background(ctx)

    for _ in range(config.number_of_walks):
        draw_single_walk(ctx, config)

    return surface


def main() -> None:
    """Generate the artwork and save it as a PNG file."""

    config = WalkConfig()
    output_path = output_directory() / config.output_filename

    surface = render_image(config)
    surface.write_to_png(str(output_path))

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()