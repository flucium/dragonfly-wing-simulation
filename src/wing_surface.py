from __future__ import annotations

import math

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from .geometry import Vector3,WingGeometry,wing_local_to_body,wing_planform_at_span
from .vibration import cantilever_mode_shape


__all__ = [
    "WingSurfaceMesh",
    "WingSurfacePanel",
    "multimode_deformed_wing_surface",
    "wing_surface_mesh",
    "wing_surface_mesh_body",
    "wing_surface_panels"
]


@dataclass(frozen=True)
class WingSurfacePanel:
    span_fraction_from_root: float

    chord_fraction_from_leading_edge: float

    center_m: Vector3

    surface_normal: Vector3

    area_m2: float


@dataclass(frozen=True)
class WingSurfaceMesh:
    span_fraction_from_root: tuple[float, ...]

    chord_fraction_from_leading_edge: tuple[float, ...]

    x_m: tuple[tuple[float, ...], ...]

    y_m: tuple[tuple[float, ...], ...]

    z_m: tuple[tuple[float, ...], ...]


def _finite_number(value: Real,name: str) -> float:

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _positive_integer(value: int,name: str,*,minimum: int = 1) -> int:

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")

    return value


def _vector3(value: Sequence[Real],name: str) -> Vector3:

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of three real numbers")

    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")

    return (_finite_number(value[0],f"{name}[0]"),_finite_number(value[1],f"{name}[1]"),_finite_number(value[2],f"{name}[2]"))


def _validated_wing(wing: WingGeometry) -> WingGeometry:

    if not isinstance(wing, WingGeometry):
        raise TypeError("wing must be a WingGeometry")

    if wing.length_m <= 0.0 or wing.chord_m <= 0.0:
        raise ValueError("wing length and chord must be positive")

    return wing


def _panel_strip_area_m2(wing: WingGeometry,start: float,stop: float) -> float:

    boundaries = [start]

    boundaries.extend(station.span_fraction_from_root for station in wing.planform_stations if start < station.span_fraction_from_root < stop)

    boundaries.append(stop)

    return wing.length_m * sum(0.5 * (wing_planform_at_span(wing, first).chord_m + wing_planform_at_span(wing, second).chord_m) * (second - first)for first, second in zip(boundaries, boundaries[1:]))


def wing_surface_panels(wing: WingGeometry,*,n_span: int = 32,n_chord: int = 8,wing_root_position_m: Sequence[Real] = (0.0, 0.0, 0.0)) -> tuple[WingSurfacePanel, ...]:

    wing = _validated_wing(wing)

    span_count = _positive_integer(n_span, "n_span")

    chord_count = _positive_integer(n_chord, "n_chord")

    root = _vector3(wing_root_position_m, "wing_root_position_m")

    panels = []

    for span_index in range(span_count):
        span_start = span_index / span_count

        span_stop = (span_index + 1) / span_count

        span_fraction = 0.5 * (span_start + span_stop)

        section = wing_planform_at_span(wing, span_fraction)

        panel_area = _panel_strip_area_m2(wing, span_start, span_stop) / chord_count

        normal = (0.0,-math.sin(section.twist_rad),math.cos(section.twist_rad))

        for chord_index in range(chord_count):
            chord_fraction = (chord_index + 0.5) / chord_count

            chord_offset = (chord_fraction - 0.5) * section.chord_m

            mid_chord = 0.5 * (section.leading_edge_m + section.trailing_edge_m)

            panels.append(WingSurfacePanel(span_fraction,chord_fraction,(root[0] + span_fraction * wing.length_m,root[1] + mid_chord + chord_offset * math.cos(section.twist_rad),root[2] + chord_offset * math.sin(section.twist_rad)),normal,panel_area))

    return tuple(panels)


def _surface_mesh(wing: WingGeometry,modal_tip_displacements_m: Sequence[float],n_span: int,n_chord: int,root: Vector3,deformation_scale: float) -> WingSurfaceMesh:

    span_fractions = tuple(index / (n_span - 1)for index in range(n_span))

    chord_fractions = tuple(index / (n_chord - 1)for index in range(n_chord))

    x_rows = []

    y_rows = []

    z_rows = []

    for span_fraction in span_fractions:
        section = wing_planform_at_span(wing, span_fraction)

        displacement = deformation_scale * sum(value * cantilever_mode_shape(span_fraction, mode_index)for mode_index, value in enumerate(modal_tip_displacements_m, start=1))

        x_row = []

        y_row = []

        z_row = []

        for chord_fraction in chord_fractions:
            chord_offset = (chord_fraction - 0.5) * section.chord_m

            mid_chord = 0.5 * (section.leading_edge_m + section.trailing_edge_m)

            x_row.append(root[0] + span_fraction * wing.length_m)

            y_row.append(root[1] + mid_chord + chord_offset * math.cos(section.twist_rad))

            z_row.append(root[2] + chord_offset * math.sin(section.twist_rad) + displacement)

        x_rows.append(tuple(x_row))

        y_rows.append(tuple(y_row))

        z_rows.append(tuple(z_row))

    return WingSurfaceMesh(span_fractions,chord_fractions,tuple(x_rows),tuple(y_rows),tuple(z_rows))


def wing_surface_mesh(wing: WingGeometry,*,n_span: int = 33,n_chord: int = 9,wing_root_position_m: Sequence[Real] = (0.0, 0.0, 0.0)) -> WingSurfaceMesh:

    wing = _validated_wing(wing)

    span_count = _positive_integer(n_span, "n_span", minimum=2)

    chord_count = _positive_integer(n_chord, "n_chord", minimum=2)

    return _surface_mesh(wing,(),span_count,chord_count,_vector3(wing_root_position_m,"wing_root_position_m"),1.0)


def multimode_deformed_wing_surface(wing: WingGeometry,modal_tip_displacements_m: Sequence[Real],*,n_span: int = 33,n_chord: int = 9,deformation_scale: Real = 1.0) -> WingSurfaceMesh:

    if not modal_tip_displacements_m:
        raise ValueError("modal_tip_displacements_m must not be empty")

    wing = _validated_wing(wing)

    modal_values = tuple(_finite_number(value,f"modal_tip_displacements_m[{index}]")for index, value in enumerate(modal_tip_displacements_m))

    scale = _finite_number(deformation_scale, "deformation_scale")

    if scale <= 0.0:
        raise ValueError("deformation_scale must be positive")

    return _surface_mesh(wing,modal_values,_positive_integer(n_span,"n_span",minimum=2),_positive_integer(n_chord,"n_chord",minimum=2),(0.0, 0.0, 0.0),scale)


def wing_surface_mesh_body(wing: WingGeometry,mesh: WingSurfaceMesh) -> WingSurfaceMesh:

    wing = _validated_wing(wing)

    if not isinstance(mesh, WingSurfaceMesh):
        raise TypeError("mesh must be a WingSurfaceMesh")

    x_rows = []

    y_rows = []

    z_rows = []

    for x_row, y_row, z_row in zip(mesh.x_m, mesh.y_m, mesh.z_m):
        points = tuple(wing_local_to_body(wing, point)for point in zip(x_row, y_row, z_row))

        x_rows.append(tuple(point[0]for point in points))

        y_rows.append(tuple(point[1]for point in points))

        z_rows.append(tuple(point[2]for point in points))

    return WingSurfaceMesh(mesh.span_fraction_from_root,mesh.chord_fraction_from_leading_edge,tuple(x_rows),tuple(y_rows),tuple(z_rows))
