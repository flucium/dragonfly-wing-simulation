from __future__ import annotations

import math

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profiles import Profiles


__all__ = [
    "BodyGeometry",
    "MG_TO_KG",
    "MM_TO_M",
    "WingGeometry",
    "WingInstance",
    "WingPlanformStation",
    "WingSection",
    "all_wings",
    "body_geometry",
    "body_to_wing_local",
    "body_vector_to_wing_local",
    "mg_to_kg",
    "mm_to_m",
    "point_on_wing",
    "select_wing",
    "target_position_body_m",
    "target_position_m",
    "target_wing",
    "wing_local_to_body",
    "wing_local_vector_to_body",
    "wing_planform_at_span"
]


MM_TO_M = 1.0e-3

MG_TO_KG = 1.0e-6

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class BodyGeometry:
    geometry_model: str

    length_m: float

    width_m: float

    thickness_m: float

    mass_kg: float


@dataclass(frozen=True)
class WingPlanformStation:
    span_fraction_from_root: float

    leading_edge_m: float

    trailing_edge_m: float

    thickness_m: float

    twist_rad: float = 0.0

    @property
    def chord_m(self) -> float:

        return self.trailing_edge_m - self.leading_edge_m


@dataclass(frozen=True)
class WingInstance:
    wing_type: str

    side: str

    root_position_body_m: Vector3

    root_orientation_body_euler_xyz_rad: Vector3


@dataclass(frozen=True)
class WingSection:
    leading_edge_m: float

    trailing_edge_m: float

    thickness_m: float

    twist_rad: float

    @property
    def chord_m(self) -> float:

        return self.trailing_edge_m - self.leading_edge_m


@dataclass(frozen=True)
class WingGeometry:
    wing_type: str

    side: str

    length_m: float

    chord_m: float

    thickness_m: float

    mass_kg: float

    planform_stations: tuple[WingPlanformStation, ...] = ()

    root_position_body_m: Vector3 = (0.0, 0.0, 0.0)

    root_orientation_body_euler_xyz_rad: Vector3 = (0.0, 0.0, 0.0)

    @property
    def area_m2(self) -> float:

        if not self.planform_stations:
            return self.length_m * self.chord_m

        return self.length_m * sum(0.5 * (first.chord_m + second.chord_m) * (second.span_fraction_from_root - first.span_fraction_from_root)for first, second in zip(self.planform_stations, self.planform_stations[1:]))


def _finite_number(value: Real, name: str) -> float:

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _positive_number(value: Real, name: str) -> float:

    result = _finite_number(value, name)

    if result <= 0.0:
        raise ValueError(f"{name} must be positive")

    return result


def _fraction(value: Real, name: str) -> float:

    result = _finite_number(value, name)

    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")

    return result


def mm_to_m(value_mm: Real) -> float:

    return _finite_number(value_mm, "value_mm") * MM_TO_M


def mg_to_kg(value_mg: Real) -> float:

    return _finite_number(value_mg, "value_mg") * MG_TO_KG


def _validated_planform(stations: tuple[WingPlanformStation, ...]) -> tuple[WingPlanformStation, ...]:

    if len(stations) < 2:
        raise ValueError("planform_stations must contain at least two stations")

    if stations[0].span_fraction_from_root != 0.0 or stations[-1].span_fraction_from_root != 1.0:
        raise ValueError("planform_stations must start at 0 and end at 1")

    previous = -1.0

    for station in stations:
        if station.span_fraction_from_root <= previous:
            raise ValueError("planform span fractions must be strictly increasing")

        if station.chord_m <= 0.0 or station.thickness_m <= 0.0:
            raise ValueError("planform chord and thickness must be positive")

        previous = station.span_fraction_from_root

    return stations


def _wing_instance(dragonfly: Profiles.Dragonfly, wing_type: str, side: str) -> WingInstance:

    matches = tuple(instance for instance in dragonfly.wing_instances if instance.wing_type == wing_type and instance.side == side)

    if len(matches) != 1:
        raise ValueError(f"expected one {wing_type} {side} wing instance")

    return matches[0]


def select_wing(dragonfly: Profiles.Dragonfly,wing_type: str,side: str) -> WingGeometry:

    if wing_type not in {"fore", "hind"}:
        raise ValueError("wing_type must be 'fore' or 'hind'")

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    if wing_type == "fore":
        length_mm = dragonfly.fore_wing_length_mm

        chord_mm = dragonfly.fore_wing_width_mm

        thickness_mm = dragonfly.fore_wing_thickness_mm

        mass_mg = dragonfly.fore_wing_mass_mg

        stations = dragonfly.fore_wing_planform_stations

    else:
        length_mm = dragonfly.hind_wing_length_mm

        chord_mm = dragonfly.hind_wing_width_mm

        thickness_mm = dragonfly.hind_wing_thickness_mm

        mass_mg = dragonfly.hind_wing_mass_mg

        stations = dragonfly.hind_wing_planform_stations

    instance = _wing_instance(dragonfly, wing_type, side)

    return WingGeometry(
        wing_type=wing_type,

        side=side,

        length_m=mm_to_m(_positive_number(length_mm, "wing length_mm")),

        chord_m=mm_to_m(_positive_number(chord_mm, "wing width_mm")),

        thickness_m=mm_to_m(_positive_number(thickness_mm, "wing thickness_mm")),

        mass_kg=mg_to_kg(_positive_number(mass_mg, "wing mass_mg")),

        planform_stations=_validated_planform(stations),

        root_position_body_m=instance.root_position_body_m,

        root_orientation_body_euler_xyz_rad=instance.root_orientation_body_euler_xyz_rad
    )


def all_wings(profile: Profiles) -> tuple[WingGeometry, ...]:

    return tuple(select_wing(profile.dragonfly, wing_type, side)for wing_type in ("fore", "hind")for side in ("right", "left"))


def body_geometry(profile: Profiles) -> BodyGeometry:

    dragonfly = profile.dragonfly

    return BodyGeometry(
        geometry_model=dragonfly.body_geometry_model,

        length_m=mm_to_m(dragonfly.body_length_mm),

        width_m=mm_to_m(dragonfly.body_width_mm),

        thickness_m=mm_to_m(dragonfly.body_thickness_mm),

        mass_kg=mg_to_kg(dragonfly.body_mass_mg)
    )


def _interpolate(first: float, second: float, fraction: float) -> float:

    return first + fraction * (second - first)


def wing_planform_at_span(wing: WingGeometry, span_fraction_from_root: Real) -> WingSection:

    span_fraction = _fraction(span_fraction_from_root, "span_fraction_from_root")

    if not wing.planform_stations:
        return WingSection(-0.5 * wing.chord_m,0.5 * wing.chord_m,wing.thickness_m,0.0)

    stations = _validated_planform(wing.planform_stations)

    for first, second in zip(stations, stations[1:]):
        if span_fraction <= second.span_fraction_from_root:
            interval = second.span_fraction_from_root - first.span_fraction_from_root

            local_fraction = (span_fraction - first.span_fraction_from_root) / interval

            return WingSection(
                leading_edge_m=_interpolate(first.leading_edge_m, second.leading_edge_m, local_fraction),

                trailing_edge_m=_interpolate(first.trailing_edge_m, second.trailing_edge_m, local_fraction),

                thickness_m=_interpolate(first.thickness_m, second.thickness_m, local_fraction),

                twist_rad=_interpolate(first.twist_rad, second.twist_rad, local_fraction)
            )

    final = stations[-1]

    return WingSection(final.leading_edge_m,final.trailing_edge_m,final.thickness_m,final.twist_rad)


def point_on_wing(wing: WingGeometry,span_fraction_from_root: Real,chord_fraction_from_leading_edge: Real,surface: str = "midplane") -> Vector3:

    span_fraction = _fraction(span_fraction_from_root, "span_fraction_from_root")

    chord_fraction = _fraction(chord_fraction_from_leading_edge,"chord_fraction_from_leading_edge")

    section = wing_planform_at_span(wing, span_fraction)

    mid_chord = 0.5 * (section.leading_edge_m + section.trailing_edge_m)

    chord_offset = (chord_fraction - 0.5) * section.chord_m

    point = (span_fraction * wing.length_m,mid_chord + chord_offset * math.cos(section.twist_rad),chord_offset * math.sin(section.twist_rad))

    normal = (0.0,-math.sin(section.twist_rad),math.cos(section.twist_rad))

    if surface == "midplane":
        normal_offset = 0.0

    elif surface == "dorsal":
        normal_offset = 0.5 * section.thickness_m

    elif surface == "ventral":
        normal_offset = -0.5 * section.thickness_m

    else:
        raise ValueError("surface must be 'midplane', 'dorsal', or 'ventral'")

    return tuple(point[index] + normal_offset * normal[index]for index in range(3))


def _rotate_xyz(vector: Vector3, angles: Vector3) -> Vector3:

    x, y, z = vector

    rx, ry, rz = angles

    y, z = y * math.cos(rx) - z * math.sin(rx),y * math.sin(rx) + z * math.cos(rx)

    x, z = x * math.cos(ry) + z * math.sin(ry),-x * math.sin(ry) + z * math.cos(ry)

    x, y = x * math.cos(rz) - y * math.sin(rz),x * math.sin(rz) + y * math.cos(rz)

    return (x, y, z)


def _inverse_rotate_xyz(vector: Vector3, angles: Vector3) -> Vector3:

    x, y, z = vector

    rx, ry, rz = angles

    x, y = x * math.cos(rz) + y * math.sin(rz),-x * math.sin(rz) + y * math.cos(rz)

    x, z = x * math.cos(ry) - z * math.sin(ry),x * math.sin(ry) + z * math.cos(ry)

    y, z = y * math.cos(rx) + z * math.sin(rx),-y * math.sin(rx) + z * math.cos(rx)

    return (x, y, z)


def wing_local_vector_to_body(wing: WingGeometry, vector_wing_local: Vector3) -> Vector3:

    side_sign = 1.0 if wing.side == "right" else -1.0

    base_vector = (vector_wing_local[1],side_sign * vector_wing_local[0],vector_wing_local[2])

    return _rotate_xyz(base_vector, wing.root_orientation_body_euler_xyz_rad)


def body_vector_to_wing_local(wing: WingGeometry, vector_body: Vector3) -> Vector3:

    base_vector = _inverse_rotate_xyz(vector_body, wing.root_orientation_body_euler_xyz_rad)

    side_sign = 1.0 if wing.side == "right" else -1.0

    return (side_sign * base_vector[1],base_vector[0],base_vector[2])


def wing_local_to_body(wing: WingGeometry, point_wing_local_m: Vector3) -> Vector3:

    rotated = wing_local_vector_to_body(wing, point_wing_local_m)

    return tuple(wing.root_position_body_m[index] + rotated[index]for index in range(3))


def body_to_wing_local(wing: WingGeometry, point_body_m: Vector3) -> Vector3:

    relative = tuple(point_body_m[index] - wing.root_position_body_m[index]for index in range(3))

    return body_vector_to_wing_local(wing, relative)


def target_wing(profile: Profiles) -> WingGeometry:

    return select_wing(profile.dragonfly,profile.simulation.target_wing_type,profile.simulation.target_wing_side)


def target_position_m(profile: Profiles, surface: str = "midplane") -> Vector3:

    wing = target_wing(profile)

    return point_on_wing(wing,profile.simulation.target_wing_span_fraction_from_root,profile.simulation.target_wing_chord_fraction_from_leading_edge,surface)


def target_position_body_m(profile: Profiles, surface: str = "midplane") -> Vector3:

    wing = target_wing(profile)

    return wing_local_to_body(wing, target_position_m(profile, surface))
