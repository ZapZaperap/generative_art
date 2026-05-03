"""Extended Cairo drawing context for circular arc paths and hex-grid walks.

``ArcWalkContext`` wraps a real ``cairo.Context`` instead of subclassing it.
This avoids compatibility issues with pycairo versions where ``cairo.Context``
cannot be safely subclassed in Python.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

import cairo
import networkx as nx
import numpy as np

import auxiliary as aux

Point = Sequence[float] | np.ndarray
DrawCallback = Callable[["ArcWalkContext"], None]
TerminationCallback = Callable[["ArcWalkContext", Any], bool]


class ArcWalkContext:
    """Cairo context wrapper with helpers for alternating circular arcs.

    The canvas is assumed to be normalized to the unit square by calling
    ``ctx.scale(width, height)`` before drawing. Methods not explicitly defined
    here are forwarded to the underlying ``cairo.Context``.
    """

    def __init__(self, surface: cairo.Surface) -> None:
        self._ctx = cairo.Context(surface)
        self.current_angle = 0.0
        self.current_radius: float | None = None
        self.current_centre = np.array([0.5, 0.5], dtype=float)
        self.arc_counter = 0
        self.parity = 0
        self.arc_n = 0
        self.hex_data: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes/methods to the underlying Cairo context."""

        return getattr(self._ctx, name)

    def _advance_counter(self) -> None:
        self.arc_counter += 1
        self.parity = (self.parity + 1) % 2

    def _add_arc(
        self,
        radius: float,
        angle_step: float,
        *,
        current_angle: float | None = None,
        current_centre: Point | None = None,
        relative_angle: bool = True,
        callback: DrawCallback | None = None,
        new_sub_path: bool = False,
    ) -> None:
        """Add one arc segment to the current path."""

        if current_angle is not None:
            self.current_angle = current_angle
        if current_centre is not None:
            self.current_centre = np.asarray(current_centre, dtype=float).copy()
        if self.current_radius is None:
            self.current_radius = radius

        draw_arc = self.arc_negative if self.parity == 0 else self.arc

        if relative_angle:
            new_angle = self.current_angle - (-1) ** self.parity * angle_step
        else:
            new_angle = angle_step

        if current_centre is not None:
            previous_point = np.asarray(current_centre, dtype=float)
        else:
            previous_point = np.asarray(self.get_current_point(), dtype=float)

        direction = previous_point - self.current_centre
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        self.current_centre = previous_point + radius * direction

        if new_sub_path:
            self.new_sub_path()

        draw_arc(*self.current_centre, radius, self.current_angle, new_angle)

        if callback is not None:
            callback(self)

        self.current_angle = np.mod(new_angle, 2 * np.pi)
        self.current_angle += np.pi if self.current_angle > np.pi else -np.pi
        self._advance_counter()

    @staticmethod
    def _as_matching_arrays(
        radii: float | Sequence[float], angle_steps: float | Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return radius and angle arrays of equal length."""

        radii_is_iterable = isinstance(radii, Iterable) and not isinstance(radii, (str, bytes))
        angles_are_iterable = isinstance(angle_steps, Iterable) and not isinstance(
            angle_steps, (str, bytes)
        )

        if not radii_is_iterable and not angles_are_iterable:
            raise ValueError("at least radii or angle_steps must be iterable")

        if radii_is_iterable:
            radius_array = np.asarray(radii, dtype=float)
        else:
            radius_array = None

        if angles_are_iterable:
            angle_array = np.asarray(angle_steps, dtype=float)
        else:
            angle_array = None

        if radius_array is not None and angle_array is not None:
            if len(radius_array) != len(angle_array):
                raise ValueError("radii and angle_steps must have the same length")
            return radius_array, angle_array

        if radius_array is not None:
            return radius_array, np.full(len(radius_array), float(angle_steps))

        assert angle_array is not None
        return np.full(len(angle_array), float(radii)), angle_array

    def arc_path(
        self,
        radii: float | Sequence[float],
        angle_steps: float | Sequence[float],
        *,
        current_angle: float | None = None,
        current_centre: Point | None = None,
        relative_angle: bool = True,
        parity: Literal[0, 1, "random"] | None = None,
        inner_callback: DrawCallback | None = None,
        outer_callback: DrawCallback | None = None,
        new_sub_path: bool = False,
    ) -> None:
        """Draw a path made from alternating circular arcs.

        ``radii`` and ``angle_steps`` may each be either scalars or sequences;
        at least one of them must be a sequence. If one argument is scalar, it is
        broadcast to match the length of the other.
        """

        radius_array, angle_array = self._as_matching_arrays(radii, angle_steps)
        if len(radius_array) == 0:
            return

        self.arc_n = len(radius_array)
        self.arc_counter = 0

        if parity == "random":
            self.parity = int(np.random.randint(2))
        elif parity in (0, 1):
            self.parity = int(parity)
        elif parity is None:
            self.parity = 0
        else:
            raise ValueError("parity must be 0, 1, 'random', or None")

        self._add_arc(
            radius_array[0],
            angle_array[0],
            current_angle=current_angle,
            current_centre=current_centre,
            relative_angle=relative_angle,
            callback=inner_callback,
            new_sub_path=new_sub_path,
        )
        if outer_callback is not None:
            outer_callback(self)

        for radius, angle_step in zip(radius_array[1:], angle_array[1:]):
            self._add_arc(
                radius,
                angle_step,
                relative_angle=relative_angle,
                callback=inner_callback,
                new_sub_path=new_sub_path,
            )
            if outer_callback is not None:
                outer_callback(self)

        self.arc_counter = 0

    def point(
        self,
        x: float,
        y: float,
        radius: float = 0.01,
        color: tuple[float, float, float, float] = (1, 1, 1, 1),
    ) -> None:
        """Draw a filled circular point."""

        self.arc(x, y, radius, 0, 2 * np.pi)
        self.set_source_rgba(*color)
        self.fill()

    def init_hex_net(
        self,
        spacing: float = 1.0,
        rotation: float = 0.0,
        origin: tuple[float, float] = (0, 0),
    ) -> dict[str, Any]:
        """Initialize a centered hexagonal graph for graph-walk sketches."""

        dx = spacing * (1 + np.sqrt(3) / 4)
        grid_size = aux.round_to_odd(np.sqrt(2) / dx)
        graph = nx.hexagonal_lattice_graph(
            m=grid_size,
            n=grid_size,
            periodic=False,
            with_positions=True,
        )
        nx.set_node_attributes(graph, values=0, name="visited")
        nx.set_edge_attributes(graph, values=0, name="visited")

        nodes = dict(graph.nodes(data=True))
        positions = np.empty((len(nodes), 2), dtype=float)
        for index, (node, data) in enumerate(nodes.items()):
            positions[index] = np.asarray(data["pos"], dtype=float)
            graph.nodes[node]["pos"] = positions[index]

        scaled_spacing = spacing / 5
        vec_a = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)]) * np.sqrt(3) * scaled_spacing
        vec_b = np.array([np.cos(-np.pi / 6), np.sin(-np.pi / 6)]) * np.sqrt(3) * scaled_spacing
        vec_0 = np.array([1, np.sqrt(3) / 2]) * scaled_spacing

        n_half = grid_size // 2
        centre = n_half * (vec_a - vec_b)
        centre += np.ceil(n_half / 2) * vec_a
        centre += (n_half - np.ceil(n_half / 2)) * vec_b
        centre += vec_0

        positions *= scaled_spacing
        positions -= centre
        positions[:] = positions @ aux.rotation_matrix(rotation)
        positions -= origin[0] * vec_a + origin[1] * vec_b - np.array([0.5, 0.5])

        self.hex_data = {
            "dx": dx,
            "grid_size": grid_size,
            "positions": positions,
            "graph": graph,
            "rotation": rotation,
            "origin": origin,
        }
        return self.hex_data

    @staticmethod
    def point_in_canvas(point: Point, pad_lr: float = 0.0, pad_tb: float = 0.0) -> bool:
        """Return whether a point lies inside the normalized canvas."""

        x, y = point
        return pad_lr <= x <= 1 - pad_lr and pad_tb <= y <= 1 - pad_tb

    @staticmethod
    def nodes_in_canvas(hex_data: dict[str, Any], pad_lr: float = 0.0, pad_tb: float = 0.0) -> list[Any]:
        """Return all graph nodes whose positions lie inside the canvas."""

        graph = hex_data["graph"]
        return [
            node
            for node, data in graph.nodes(data=True)
            if ArcWalkContext.point_in_canvas(data["pos"], pad_lr=pad_lr, pad_tb=pad_tb)
        ]

    @staticmethod
    def unvisited_nodes(hex_data: dict[str, Any], nodes: Iterable[Any] | None = None) -> list[Any]:
        """Return all unvisited nodes in a hex graph."""

        graph = hex_data["graph"]
        candidate_nodes = list(graph.nodes()) if nodes is None else list(nodes)
        return [node for node in candidate_nodes if graph.nodes[node]["visited"] == 0]

    def hex_random_walk(
        self,
        node: Any | None = None,
        *,
        termination_callback: TerminationCallback,
        draw_callback: DrawCallback | None = None,
    ) -> int:
        """Draw a random walk over unvisited neighboring nodes of the hex graph."""

        graph = self.hex_data["graph"]
        if node is None:
            available_nodes = self.unvisited_nodes(self.hex_data)
            if not available_nodes:
                raise ValueError("no unvisited nodes are available")
            node = available_nodes[int(np.random.choice(len(available_nodes)))]

        graph.nodes[node]["visited"] += 1
        self.save()
        self.move_to(*graph.nodes[node]["pos"])

        steps = 0
        while not termination_callback(self, node):
            neighbours = self.unvisited_nodes(self.hex_data, graph.neighbors(node))
            if not neighbours:
                if steps == 0:
                    self.restore()
                break

            next_node = neighbours[int(np.random.choice(len(neighbours)))]
            self.line_to(*graph.nodes[next_node]["pos"])
            node = next_node
            steps += 1
            graph.nodes[next_node]["visited"] += 1

        if draw_callback is not None:
            draw_callback(self)

        return steps

    def hex_draw_all(self, draw_callback: DrawCallback | None = None) -> None:
        """Draw all edges of the initialized hex graph."""

        graph = self.hex_data["graph"]
        for node_a, node_b in graph.edges():
            self.move_to(*graph.nodes[node_a]["pos"])
            self.line_to(*graph.nodes[node_b]["pos"])
            if draw_callback is not None:
                draw_callback(self)


# Backward-compatible class name for older sketches.
MyContext = ArcWalkContext