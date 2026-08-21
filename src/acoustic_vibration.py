from __future__ import annotations

import math

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from .acoustics import PressureSample,point_source_pressure_field
from .geometry import Vector3,WingGeometry
from .vibration import FirstBendingMode,cantilever_mode_shape,multimode_displacement_at_span,steady_multimode_displacement_phasors
from .wing_surface import WingSurfacePanel,wing_surface_panels


__all__ = [
    "SteadyMultimodeAcousticVibration",
    "WingPressureField",
    "WingPressureSample",
    "modal_forces_from_pressure_field",
    "point_source_wing_pressure_field",
    "steady_multimode_point_source_response"
]


@dataclass(frozen=True)
class WingPressureSample:
    span_fraction_from_root: float

    chord_fraction_from_leading_edge: float

    position_m: Vector3

    surface_normal: Vector3

    area_m2: float

    distance_m: float

    incidence_factor: float

    pressure_rms_phasor_pa: complex

    spl_db: float

    directivity_factor: float = 1.0


@dataclass(frozen=True)
class WingPressureField:
    frequency_hz: float

    source_position_m: Vector3

    surface_normal: Vector3

    samples: tuple[WingPressureSample, ...]

    source_axis: Vector3 | None = None

    directivity_exponent: float = 0.0

    @property
    def spl_range_db(self) -> tuple[float, float]:

        values = tuple(sample.spl_db for sample in self.samples)

        return min(values),max(values)

    @property
    def distance_range_m(self) -> tuple[float, float]:

        values = tuple(sample.distance_m for sample in self.samples)

        return min(values),max(values)


@dataclass(frozen=True)
class SteadyMultimodeAcousticVibration:
    frequency_hz: float

    modes: tuple[FirstBendingMode, ...]

    modal_force_rms_phasors_n: tuple[complex, ...]

    modal_tip_displacement_rms_phasors_m: tuple[complex, ...]

    output_span_fraction_from_root: float

    pressure_field: WingPressureField

    @property
    def tip_displacement_rms_phasor_m(self) -> complex:

        return sum(self.modal_tip_displacement_rms_phasors_m)

    @property
    def output_displacement_rms_phasor_m(self) -> complex:

        return multimode_displacement_at_span(self.modal_tip_displacement_rms_phasors_m,self.output_span_fraction_from_root)


def _finite_number(value: Real,name: str) -> float:

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _positive_number(value: Real,name: str) -> float:

    result = _finite_number(value, name)

    if result <= 0.0:
        raise ValueError(f"{name} must be positive")

    return result


def _fraction(value: Real,name: str) -> float:

    result = _finite_number(value, name)

    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")

    return result


def _positive_integer(value: int,name: str) -> int:

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _vector3(value: Sequence[Real],name: str) -> Vector3:

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of three real numbers")

    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")

    return (_finite_number(value[0],f"{name}[0]"),_finite_number(value[1],f"{name}[1]"),_finite_number(value[2],f"{name}[2]"))


def _unit_vector3(value: Sequence[Real],name: str) -> Vector3:

    vector = _vector3(value, name)

    magnitude = math.sqrt(sum(component * component for component in vector))

    if magnitude == 0.0:
        raise ValueError(f"{name} must not be the zero vector")

    return (vector[0] / magnitude,vector[1] / magnitude,vector[2] / magnitude)


def _pressure_sample(panel: WingSurfacePanel,acoustic: PressureSample,normal: Vector3) -> WingPressureSample:

    sine = -panel.surface_normal[1]

    cosine = panel.surface_normal[2]

    local_normal = _unit_vector3((normal[0],normal[1] * cosine - normal[2] * sine,normal[1] * sine + normal[2] * cosine),"surface_normal")

    incidence = sum(direction * component for direction, component in zip(acoustic.direction_from_source, local_normal))

    return WingPressureSample(panel.span_fraction_from_root,panel.chord_fraction_from_leading_edge,panel.center_m,local_normal,panel.area_m2,acoustic.distance_m,incidence,acoustic.pressure_rms_phasor_pa,acoustic.spl_db,acoustic.directivity_factor)


def point_source_wing_pressure_field(wing: WingGeometry,source_position_m: Sequence[Real],reference_pressure_rms_pa: Real,frequency_hz: Real,reference_distance_m: Real,sound_speed_m_s: Real,reference_sound_pressure_pa: Real,*,wing_root_position_m: Sequence[Real] = (0.0, 0.0, 0.0),surface_normal: Sequence[Real] = (0.0, 0.0, 1.0),source_axis: Sequence[Real] | None = None,directivity_exponent: Real = 0.0,reference_phase_rad: Real = 0.0,n_span: int = 32,n_chord: int = 8) -> WingPressureField:

    if not isinstance(wing, WingGeometry):
        raise TypeError("wing must be a WingGeometry")

    source = _vector3(source_position_m, "source_position_m")

    normal = _unit_vector3(surface_normal, "surface_normal")

    axis = None if source_axis is None else _unit_vector3(source_axis, "source_axis")

    exponent = _finite_number(directivity_exponent, "directivity_exponent")

    if exponent < 0.0:
        raise ValueError("directivity_exponent must be non-negative")

    panels = wing_surface_panels(wing,n_span=_positive_integer(n_span,"n_span"),n_chord=_positive_integer(n_chord,"n_chord"),wing_root_position_m=wing_root_position_m)

    acoustic_samples = point_source_pressure_field(source,tuple(panel.center_m for panel in panels),reference_pressure_rms_pa,frequency_hz,reference_distance_m,sound_speed_m_s,reference_sound_pressure_pa,reference_phase_rad,source_axis=axis,directivity_exponent=exponent)

    return WingPressureField(_positive_number(frequency_hz,"frequency_hz"),source,normal,tuple(_pressure_sample(panel, acoustic, normal)for panel, acoustic in zip(panels, acoustic_samples)),axis,exponent)


def modal_forces_from_pressure_field(pressure_field: WingPressureField,modes: Sequence[FirstBendingMode]) -> tuple[complex, ...]:

    if not isinstance(pressure_field, WingPressureField):
        raise TypeError("pressure_field must be a WingPressureField")

    if not modes:
        raise ValueError("modes must contain at least one mode")

    return tuple(sum(sample.pressure_rms_phasor_pa * sample.incidence_factor * sample.area_m2 * cantilever_mode_shape(sample.span_fraction_from_root,mode.mode_index)for sample in pressure_field.samples)for mode in modes)


def steady_multimode_point_source_response(modes: Sequence[FirstBendingMode],wing: WingGeometry,source_position_m: Sequence[Real],reference_pressure_rms_pa: Real,frequency_hz: Real,reference_distance_m: Real,sound_speed_m_s: Real,reference_sound_pressure_pa: Real,*,output_span_fraction_from_root: Real = 1.0,wing_root_position_m: Sequence[Real] = (0.0, 0.0, 0.0),surface_normal: Sequence[Real] = (0.0, 0.0, 1.0),source_axis: Sequence[Real] | None = None,directivity_exponent: Real = 0.0,reference_phase_rad: Real = 0.0,n_span: int = 32,n_chord: int = 8) -> SteadyMultimodeAcousticVibration:

    mode_tuple = tuple(modes)

    if not mode_tuple:
        raise ValueError("modes must contain at least one mode")

    frequency = _positive_number(frequency_hz, "frequency_hz")

    field = point_source_wing_pressure_field(wing,source_position_m,reference_pressure_rms_pa,frequency,reference_distance_m,sound_speed_m_s,reference_sound_pressure_pa,wing_root_position_m=wing_root_position_m,surface_normal=surface_normal,source_axis=source_axis,directivity_exponent=directivity_exponent,reference_phase_rad=reference_phase_rad,n_span=n_span,n_chord=n_chord)

    modal_forces = modal_forces_from_pressure_field(field, mode_tuple)

    return SteadyMultimodeAcousticVibration(frequency,mode_tuple,modal_forces,steady_multimode_displacement_phasors(mode_tuple,modal_forces,frequency),_fraction(output_span_fraction_from_root,"output_span_fraction_from_root"),field)
