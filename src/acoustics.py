from __future__ import annotations

import cmath
import math

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real


__all__ = [
    "PressureSample",
    "point_source_pressure_field",
    "spl_to_pressure_rms"
]


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class PressureSample:
    position_m: Vector3

    distance_m: float

    direction_from_source: Vector3

    pressure_rms_phasor_pa: complex

    spl_db: float

    directivity_factor: float = 1.0


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

    return tuple(component / magnitude for component in vector)


def spl_to_pressure_rms(spl_db: Real,reference_pressure_pa: Real) -> float:

    level = _finite_number(spl_db, "spl_db")

    reference = _positive_number(reference_pressure_pa, "reference_pressure_pa")

    return reference * 10.0 ** (level / 20.0)


def _pressure_rms_to_spl(pressure_rms_pa: Real,reference_pressure_pa: Real) -> float:

    pressure = _finite_number(pressure_rms_pa, "pressure_rms_pa")

    reference = _positive_number(reference_pressure_pa, "reference_pressure_pa")

    if pressure < 0.0:
        raise ValueError("pressure_rms_pa must be non-negative")

    if pressure == 0.0:
        return -math.inf

    return 20.0 * math.log10(pressure / reference)


def _source_geometry(source: Vector3,target: Vector3) -> tuple[float, Vector3]:

    displacement = (target[0] - source[0],target[1] - source[1],target[2] - source[2])

    distance = math.sqrt(sum(component * component for component in displacement))

    if distance == 0.0:
        raise ValueError("source and target positions must be different")

    return distance,(displacement[0] / distance,displacement[1] / distance,displacement[2] / distance)


def point_source_pressure_field(source_position_m: Sequence[Real],target_positions_m: Sequence[Sequence[Real]],reference_pressure_rms_pa: Real,frequency_hz: Real,reference_distance_m: Real,sound_speed_m_s: Real,reference_sound_pressure_pa: Real,reference_phase_rad: Real = 0.0,*,source_axis: Sequence[Real] | None = None,directivity_exponent: Real = 0.0) -> tuple[PressureSample, ...]:

    source = _vector3(source_position_m, "source_position_m")

    pressure = _finite_number(reference_pressure_rms_pa, "reference_pressure_rms_pa")

    if pressure < 0.0:
        raise ValueError("reference_pressure_rms_pa must be non-negative")

    frequency = _positive_number(frequency_hz, "frequency_hz")

    reference_distance = _positive_number(reference_distance_m, "reference_distance_m")

    sound_speed = _positive_number(sound_speed_m_s, "sound_speed_m_s")

    reference_pressure = _positive_number(reference_sound_pressure_pa, "reference_sound_pressure_pa")

    phase = _finite_number(reference_phase_rad, "reference_phase_rad")

    axis = None if source_axis is None else _unit_vector3(source_axis, "source_axis")

    exponent = _finite_number(directivity_exponent, "directivity_exponent")

    if exponent < 0.0:
        raise ValueError("directivity_exponent must be non-negative")

    wavenumber = 2.0 * math.pi * frequency / sound_speed

    reference_phasor = pressure * cmath.exp(1j * phase)

    samples = []

    for index, value in enumerate(target_positions_m):
        position = _vector3(value,f"target_positions_m[{index}]")

        distance, direction = _source_geometry(source, position)

        if axis is None:
            directivity = 1.0

        else:
            cosine = max(0.0,sum(component * axis_component for component, axis_component in zip(direction, axis)))

            directivity = cosine**exponent if cosine > 0.0 else 0.0

        transfer = directivity * reference_distance / distance * cmath.exp(-1j * wavenumber * (distance - reference_distance))

        local_pressure = reference_phasor * transfer

        samples.append(PressureSample(position,distance,direction,local_pressure,_pressure_rms_to_spl(abs(local_pressure),reference_pressure),directivity))

    if not samples:
        raise ValueError("target_positions_m must contain at least one position")

    return tuple(samples)
