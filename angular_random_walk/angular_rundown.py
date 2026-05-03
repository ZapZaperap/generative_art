"""Generate an angular rundown image.

The script writes ``output/angular_rundown.png`` relative to this file's
location, making it safe to run from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cairo
import numpy as np

from context_extension import ArcWalkContext


@dataclass(frozen=True)
class RenderConfig:
    """Parameters controlling the rendered image."""

    width: int = 1000
    height: int = 1000
    n_steps: int = 1000
    n_walks: int = 20
    radius: float = 0.05
    max_step_angle: float = 0.5
    output_filename: str = "angular_rundown.png"


def script_dir() -> Path:
    """Return the directory containing this script."""

    return Path(__file__).resolve().parent


def output_dir() -> Path:
    """Return the output directory next to this script, creating it if needed."""

    directory = script_dir() / "output"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def draw_fading_segment(ctx: ArcWalkContext) -> None:
    """Stroke the current arc segment with opacity decreasing along the path."""

    current_point = ctx.get_current_point()
    progress = min(ctx.arc_counter / ctx.arc_n, 1.0)
    opacity = 1.0 - progress**10

    ctx.set_source_rgba(1, 1, 1, opacity)
    ctx.set_line_width(0.002)
    ctx.stroke()
    ctx.move_to(*current_point)


def render(config: RenderConfig) -> cairo.ImageSurface:
    """Render the artwork and return the Cairo surface."""

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, config.width, config.height)
    ctx = ArcWalkContext(surface)
    ctx.scale(config.width, config.height)

    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()

    step_scaling = np.linspace(0.0, 1.0, config.n_steps) ** 10
    centre_x_values = np.linspace(0.1, 0.9, config.n_walks + 2)[1:-1]

    for centre_x in centre_x_values:
        centre = np.array([centre_x, 0.1], dtype=float)
        initial_arc_centre = np.array([centre_x - config.radius, 0.1], dtype=float)

        ctx.arc(*centre, 0.01, 0, 2 * np.pi)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()

        random_steps = np.random.random(config.n_steps) * step_scaling * config.max_step_angle
        ctx.arc_path(
            config.radius,
            random_steps,
            current_centre=initial_arc_centre,
            current_angle=0.0,
            parity=1,
            inner_callback=draw_fading_segment,
            new_sub_path=True,
        )

    return surface


def main() -> None:
    """Render and save the image."""

    config = RenderConfig()
    path = output_dir() / config.output_filename
    render(config).write_to_png(str(path))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()