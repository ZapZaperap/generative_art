"""Generate a contiguous letter shape from text and export it as a Gmsh geometry file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except ImportError as exc:
    raise ImportError(
        "This script requires shapely. Install it with `pip install shapely`."
    ) from exc


def resolve_font(font: str | None = None) -> FontProperties:
    """Return a matplotlib FontProperties object for a family or font file."""
    if font is None:
        return FontProperties(fname=findfont(FontProperties(family="DejaVu Sans Mono")))

    if os.path.isfile(font):
        return FontProperties(fname=font)

    try:
        return FontProperties(family=font)
    except Exception:
        return FontProperties(fname=findfont(FontProperties(family="DejaVu Sans")))


def signed_area(coords: np.ndarray) -> float:
    """Return the signed area of a closed polygon ring."""
    if len(coords) < 3:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def text_to_shapely(
    text: str,
    font: str | None = None,
    size: float = 1.0,
    x_scale: float = 1.0,
    underline: bool = False,
    underline_height: float = 0.05,
    underline_margin: float = 0.02,
) -> Polygon:
    """Convert text into a contiguous Shapely polygon shape."""
    prop = resolve_font(font)
    prop.set_weight("bold")
    print(f"Font name: {prop.get_name()}\t weight: {prop.get_weight()}")

    polygon_contours = []
    current_x = 0.0
    for char in text:
        if char == " ":
            char_path = TextPath((0, 0), "A", size=1.0, prop=prop)
            extents = char_path.get_extents()
            char_width = float(extents.width)
            current_x += char_width * x_scale
            continue
        char_path = TextPath((0, 0), char, size=1.0, prop=prop)
        extents = char_path.get_extents()
        char_width = float(extents.width)
        transform = Affine2D().translate(current_x, 0).scale(size, size)
        for poly in char_path.transformed(transform).to_polygons():
            polygon_contours.append(poly)
        current_x += char_width * x_scale

    if not polygon_contours:
        raise ValueError(f"No contours could be generated for text: {text!r}")

    rings = []
    for contour in polygon_contours:
        if len(contour) < 3:
            continue

        coords = np.asarray(contour, dtype=float)
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        if len(coords) < 4:
            continue

        ring_poly = Polygon(coords)
        if not ring_poly.is_valid or ring_poly.area < 1e-8:
            continue

        rings.append({
            "coords": coords,
            "poly": ring_poly,
            "area": ring_poly.area,
        })

    if not rings:
        raise ValueError("No valid polygon rings were extracted from the text path.")

    parent_index = [None] * len(rings)
    for idx, ring in enumerate(rings):
        children = []
        for jdx, other in enumerate(rings):
            if idx == jdx:
                continue
            if other["poly"].covers(ring["poly"]):
                children.append((other["area"], jdx))
        if children:
            parent_index[idx] = min(children)[1]

    interior_groups = {idx: [] for idx in range(len(rings))}
    exteriors = []
    for idx, ring in enumerate(rings):
        if parent_index[idx] is None:
            exteriors.append(idx)
        else:
            interior_groups[parent_index[idx]].append(ring["coords"][:-1].tolist())

    exterior_polygons = []
    for ext_idx in exteriors:
        ext_coords = rings[ext_idx]["coords"]
        holes = interior_groups.get(ext_idx, [])
        ext_poly = Polygon(ext_coords, holes=holes)
        if ext_poly.is_valid and ext_poly.area > 0:
            exterior_polygons.append(ext_poly)

    if not exterior_polygons:
        raise ValueError("Could not assemble any valid text polygons.")

    shape = unary_union(exterior_polygons)

    if underline:
        minx, miny, maxx, maxy = shape.bounds
        underline_rect = Polygon([
            (minx, miny - underline_height),
            (maxx, miny - underline_height),
            (maxx, miny + underline_margin),
            (minx, miny + underline_margin),
            (minx, miny - underline_height),
        ])
        shape = unary_union([shape, underline_rect])

    shape = shape.buffer(0.0)
    if shape.is_empty:
        raise ValueError("The generated shape is empty after unioning contours.")

    return shape


def shapely_to_geo(
    geom,
    filepath: str,
    characteristic_length: float = 0.1,
    include_mesh: bool = False,
) -> None:
    """Write a Shapely polygon or multipolygon into a Gmsh .geo file."""
    if geom.is_empty:
        raise ValueError("Cannot write empty geometry to geo file.")

    polygons = []
    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        raise ValueError(
            f"Unsupported geometry type for export: {geom.geom_type}."
        )

    lines: list[str] = ["lc = %g;" % characteristic_length, ""]
    point_id = 1
    line_id = 1
    loop_id = 1
    surface_id = 1
    geo_lines: list[str] = []
    surface_lines: list[str] = []

    for poly in polygons:
        exterior_coords = list(poly.exterior.coords)
        exterior_point_ids = list(range(point_id, point_id + len(exterior_coords) - 1))
        for coord, pid in zip(exterior_coords[:-1], exterior_point_ids):
            lines.append("Point(%d) = {%g, %g, 0, lc};" % (pid, coord[0], coord[1]))

        exterior_line_ids = []
        for a, b in zip(exterior_point_ids, exterior_point_ids[1:] + [exterior_point_ids[0]]):
            lines.append("Line(%d) = {%d, %d};" % (line_id, a, b))
            exterior_line_ids.append(line_id)
            line_id += 1

        geo_lines.append(
            "Line Loop(%d) = {%s};" % (loop_id, ", ".join(str(i) for i in exterior_line_ids))
        )
        outer_loop_id = loop_id
        loop_id += 1
        point_id += len(exterior_point_ids)

        hole_loop_ids: list[int] = []
        for interior in poly.interiors:
            interior_coords = list(interior.coords)
            interior_point_ids = list(range(point_id, point_id + len(interior_coords) - 1))
            for coord, pid in zip(interior_coords[:-1], interior_point_ids):
                lines.append("Point(%d) = {%g, %g, 0, lc};" % (pid, coord[0], coord[1]))

            hole_line_ids: list[int] = []
            for a, b in zip(interior_point_ids, interior_point_ids[1:] + [interior_point_ids[0]]):
                lines.append("Line(%d) = {%d, %d};" % (line_id, a, b))
                hole_line_ids.append(line_id)
                line_id += 1

            geo_lines.append(
                "Line Loop(%d) = {%s};" % (loop_id, ", ".join(str(i) for i in hole_line_ids))
            )
            hole_loop_ids.append(loop_id)
            loop_id += 1
            point_id += len(interior_point_ids)

        surface_loops = [str(outer_loop_id)] + [f"-{i}" for i in hole_loop_ids]
        surface_lines.append(
            "Plane Surface(%d) = {%s};" % (surface_id, ", ".join(surface_loops))
        )
        surface_id += 1

    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines + [""] + geo_lines + [""] + surface_lines))

    if include_mesh:
        try:
            import gmsh
        except ImportError as exc:
            raise ImportError(
                "gmsh is not installed. Install gmsh to generate a mesh from the .geo file."
            ) from exc
        gmsh.initialize()
        gmsh.open(filepath)
        gmsh.model.mesh.generate(2)
        msh_path = str(Path(filepath).with_suffix(".msh"))
        gmsh.write(msh_path)
        gmsh.finalize()
        print(f"Also wrote mesh to {msh_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a text string into a Gmsh-compatible geometry file."
    )
    parser.add_argument("text", help="The text to convert into a single shape.")
    parser.add_argument(
        "--font",
        help="Font family name or path to a font file.",
        default=None,
    )
    parser.add_argument(
        "--size",
        type=float,
        default=1.0,
        help="Base font size for the generated text shape.",
    )
    parser.add_argument(
        "--x-scale",
        type=float,
        default=1.0,
        help="Horizontal letter-advance multiplier; use values <1 to bring letters closer.",
    )
    parser.add_argument(
        "--underline",
        action="store_true",
        help="Add a baseline underline to make separate letters contiguous.",
    )
    parser.add_argument(
        "--underline-height",
        type=float,
        default=0.05,
        help="Height of the underline rectangle below the text baseline.",
    )
    parser.add_argument(
        "--underline-margin",
        type=float,
        default=0.02,
        help="Horizontal margin for the underline beyond the text extents.",
    )
    parser.add_argument(
        "--characteristic-length",
        type=float,
        default=0.1,
        help="Mesh size used in the exported .geo file.",
    )
    parser.add_argument(
        "--output",
        default="text_shape.geo",
        help="Output .geo filename.",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="If gmsh is installed, also generate an accompanying .msh mesh file.",
    )
    args = parser.parse_args()

    shape = text_to_shapely(
        args.text,
        font=args.font,
        size=args.size,
        x_scale=args.x_scale,
        underline=args.underline,
        underline_height=args.underline_height,
        underline_margin=args.underline_margin,
    )

    shapely_to_geo(shape, args.output, characteristic_length=args.characteristic_length, include_mesh=args.mesh)
    print(f"Wrote Gmsh geometry to: {args.output}")


if __name__ == "__main__":
    main()
