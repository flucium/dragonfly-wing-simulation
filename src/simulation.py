from __future__ import annotations

import cmath
import math

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from .acoustic_vibration import SteadyMultimodeAcousticVibration,WingPressureField,steady_multimode_point_source_response
from .geometry import WingGeometry
from .vibration import FirstBendingMode,cantilever_mode_shape


__all__ = [
    "DriveEnvelopeSpec",
    "SimulationResult",
    "simulate_multimode"
]


@dataclass(frozen=True)
class DriveEnvelopeSpec:
    start_time_s: float = 0.0

    stop_time_s: float | None = None

    rise_time_s: float = 0.0

    fall_time_s: float = 0.0


@dataclass(frozen=True)
class SimulationResult:
    wing: WingGeometry

    modes: tuple[FirstBendingMode, ...]

    steady_response: SteadyMultimodeAcousticVibration

    integration_method: str

    time_step_s: float

    time_s: tuple[float, ...]

    modal_displacement_by_mode_m: tuple[tuple[float, ...], ...]

    tip_displacement_m: tuple[float, ...]

    output_displacement_m: tuple[float, ...]

    @property
    def pressure_field(self) -> WingPressureField:

        return self.steady_response.pressure_field

    @property
    def frequency_hz(self) -> float:

        return self.steady_response.frequency_hz


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


def _validated_envelope(envelope: DriveEnvelopeSpec,duration_s: float) -> DriveEnvelopeSpec:

    if not isinstance(envelope, DriveEnvelopeSpec):
        raise TypeError("drive_envelope_spec must be a DriveEnvelopeSpec")

    start = _finite_number(envelope.start_time_s, "start_time_s")

    rise = _finite_number(envelope.rise_time_s, "rise_time_s")

    fall = _finite_number(envelope.fall_time_s, "fall_time_s")

    if start < 0.0 or rise < 0.0 or fall < 0.0 or start > duration_s:
        raise ValueError("drive envelope times must be non-negative and within duration")

    stop = envelope.stop_time_s

    if stop is not None:
        stop = _positive_number(stop, "stop_time_s")

        if not start < stop <= duration_s:
            raise ValueError("stop_time_s must be after start_time_s and within duration")

    return DriveEnvelopeSpec(start,stop,rise,fall)


def _envelope_value(time_s: float,duration_s: float,envelope: DriveEnvelopeSpec) -> float:

    stop = duration_s if envelope.stop_time_s is None else envelope.stop_time_s

    if time_s < envelope.start_time_s or time_s > stop:
        return 0.0

    if envelope.rise_time_s > 0.0 and time_s < envelope.start_time_s + envelope.rise_time_s:
        progress = min(1.0,(time_s - envelope.start_time_s) / envelope.rise_time_s)

        return 0.5 - 0.5 * math.cos(math.pi * progress)

    if envelope.fall_time_s > 0.0 and time_s > stop - envelope.fall_time_s:
        progress = min(1.0,(stop - time_s) / envelope.fall_time_s)

        return 0.5 - 0.5 * math.cos(math.pi * progress)

    return 1.0


def _harmonic_force(force_rms_phasor_n: complex,frequency_hz: float,time_s: float,duration_s: float,envelope: DriveEnvelopeSpec) -> float:

    rotation = cmath.exp(1j * 2.0 * math.pi * frequency_hz * time_s)

    return math.sqrt(2.0) * (force_rms_phasor_n * rotation).real * _envelope_value(time_s,duration_s,envelope)


def _derivative(time_s: float,displacement_m: float,velocity_m_s: float,mode: FirstBendingMode,force_phasor_n: complex,drive_frequency_hz: float,duration_s: float,envelope: DriveEnvelopeSpec) -> tuple[float, float]:

    force = _harmonic_force(force_phasor_n,drive_frequency_hz,time_s,duration_s,envelope)

    acceleration = (force - mode.damping_n_s_m * velocity_m_s - mode.stiffness_n_m * displacement_m) / mode.modal_mass_kg

    return velocity_m_s,acceleration


def _rk4_step(time_s: float,time_step_s: float,displacement_m: float,velocity_m_s: float,mode: FirstBendingMode,force_phasor_n: complex,drive_frequency_hz: float,duration_s: float,envelope: DriveEnvelopeSpec) -> tuple[float, float]:

    arguments = (mode,force_phasor_n,drive_frequency_hz,duration_s,envelope)

    k1_q, k1_v = _derivative(time_s,displacement_m,velocity_m_s,*arguments)

    k2_q, k2_v = _derivative(time_s + 0.5 * time_step_s,displacement_m + 0.5 * time_step_s * k1_q,velocity_m_s + 0.5 * time_step_s * k1_v,*arguments)

    k3_q, k3_v = _derivative(time_s + 0.5 * time_step_s,displacement_m + 0.5 * time_step_s * k2_q,velocity_m_s + 0.5 * time_step_s * k2_v,*arguments)

    k4_q, k4_v = _derivative(time_s + time_step_s,displacement_m + time_step_s * k3_q,velocity_m_s + time_step_s * k3_v,*arguments)

    return (
        displacement_m + time_step_s * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q) / 6.0,
        velocity_m_s + time_step_s * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    )


def simulate_multimode(wing: WingGeometry,modes: Sequence[FirstBendingMode],*,source_position_m: Sequence[Real],reference_pressure_rms_pa: Real,frequency_hz: Real,reference_distance_m: Real,sound_speed_m_s: Real,reference_sound_pressure_pa: Real,duration_s: Real,time_step_s: Real,output_span_fraction_from_root: Real = 1.0,wing_root_position_m: Sequence[Real] = (0.0, 0.0, 0.0),surface_normal: Sequence[Real] = (0.0, 0.0, 1.0),source_axis: Sequence[Real] | None = None,directivity_exponent: Real = 0.0,reference_phase_rad: Real = 0.0,drive_envelope_spec: DriveEnvelopeSpec = DriveEnvelopeSpec(),n_span: int = 32,n_chord: int = 8) -> SimulationResult:

    mode_tuple = tuple(modes)

    if not mode_tuple or any(not isinstance(mode, FirstBendingMode)for mode in mode_tuple):
        raise ValueError("modes must contain FirstBendingMode values")

    duration = _positive_number(duration_s, "duration_s")

    requested_time_step = _positive_number(time_step_s, "time_step_s")

    envelope = _validated_envelope(drive_envelope_spec, duration)

    response = steady_multimode_point_source_response(mode_tuple,wing,source_position_m,reference_pressure_rms_pa,frequency_hz,reference_distance_m,sound_speed_m_s,reference_sound_pressure_pa,output_span_fraction_from_root=output_span_fraction_from_root,wing_root_position_m=wing_root_position_m,surface_normal=surface_normal,source_axis=source_axis,directivity_exponent=directivity_exponent,reference_phase_rad=reference_phase_rad,n_span=n_span,n_chord=n_chord)

    requested_step_count = duration / requested_time_step

    nearest_step_count = round(requested_step_count)

    number_of_steps = (
        nearest_step_count
        if math.isclose(requested_step_count,nearest_step_count,rel_tol=1.0e-12,abs_tol=1.0e-12)
        else math.ceil(requested_step_count)
    )

    actual_time_step = duration / number_of_steps

    time = tuple(index * actual_time_step for index in range(number_of_steps + 1))

    maximum_frequency = max(mode.angular_frequency_rad_s for mode in mode_tuple)

    if maximum_frequency * actual_time_step > 2.5:
        raise ValueError(f"time_step_s is too large for stable RK4 integration; use {2.5 / maximum_frequency:.6g} s or smaller")

    displacements = [[0.0]for _ in mode_tuple]

    velocities = [0.0 for _ in mode_tuple]

    for sample_index, sample_time in enumerate(time[:-1]):
        for mode_index, (mode, force_phasor) in enumerate(zip(mode_tuple,response.modal_force_rms_phasors_n)):
            displacement, velocity = _rk4_step(sample_time,actual_time_step,displacements[mode_index][-1],velocities[mode_index],mode,force_phasor,response.frequency_hz,duration,envelope)

            displacements[mode_index].append(displacement)

            velocities[mode_index] = velocity

    tip_displacement = tuple(sum(values)for values in zip(*displacements))

    output_span = response.output_span_fraction_from_root

    output_displacement = tuple(sum(displacements[mode_index][sample_index] * cantilever_mode_shape(output_span,mode.mode_index)for mode_index, mode in enumerate(mode_tuple))for sample_index in range(len(time)))

    return SimulationResult(wing,mode_tuple,response,"rk4_modal",actual_time_step,time,tuple(tuple(values)for values in displacements),tip_displacement,output_displacement)
