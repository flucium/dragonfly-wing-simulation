from __future__ import annotations

import cmath
import math

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from numbers import Real
from .acoustic_vibration import SteadyMultimodeAcousticVibration,steady_multimode_point_source_response
from .acoustics import spl_to_pressure_rms
from .geometry import Vector3
from .profiles import ProfileBundle
from .simulation import DriveEnvelopeSpec,SimulationResult,simulate_multimode
from .vibration import FIRST_MODE_MASS_RATIO,FirstBendingMode,build_bending_modes


__all__ = [
    "ConvergenceSample",
    "ConvergenceStudy",
    "ExperimentSpec",
    "FrequencyResponseSample",
    "FrequencySweepResult",
    "PointSourceSpec",
    "SamplingSpec",
    "SensitivitySample",
    "SensitivityStudy",
    "VibrationSpec",
    "experiment_spec_from_profile",
    "run_convergence_study",
    "run_experiment",
    "run_frequency_sweep",
    "run_parameter_sensitivity",
    "run_profile_experiment"
]


@dataclass(frozen=True)
class VibrationSpec:
    damping_ratio: float

    natural_frequency_hz: float | None = None

    young_modulus_pa: float | None = None

    modal_mass_ratio: float = FIRST_MODE_MASS_RATIO

    n_modes: int = 1


@dataclass(frozen=True)
class PointSourceSpec:
    position_m: Vector3

    pressure_rms_pa_at_reference_distance: float

    reference_distance_m: float

    phase_rad: float = 0.0

    axis: Vector3 | None = None

    directivity_exponent: float = 0.0

    illumination_scope: str = "all_wings"


@dataclass(frozen=True)
class SamplingSpec:
    duration_s: float

    time_step_s: float

    n_span: int = 32

    n_chord: int = 8


@dataclass(frozen=True)
class ExperimentSpec:
    vibration: VibrationSpec

    source: PointSourceSpec

    sampling: SamplingSpec

    frequency_hz: float

    sound_speed_m_s: float

    reference_sound_pressure_pa: float

    drive_envelope: DriveEnvelopeSpec

    surface_normal: Vector3


@dataclass(frozen=True)
class FrequencyResponseSample:
    frequency_hz: float

    tip_displacement_rms_phasor_m: complex

    output_displacement_rms_phasor_m: complex

    @property
    def tip_displacement_rms_m(self) -> float:

        return abs(self.tip_displacement_rms_phasor_m)

    @property
    def output_displacement_rms_m(self) -> float:

        return abs(self.output_displacement_rms_phasor_m)

    @property
    def output_phase_rad(self) -> float:

        return cmath.phase(self.output_displacement_rms_phasor_m)


@dataclass(frozen=True)
class FrequencySweepResult:
    samples: tuple[FrequencyResponseSample, ...]

    @property
    def maximum_sample(self) -> FrequencyResponseSample:

        return max(self.samples,key=lambda sample: sample.output_displacement_rms_m)


@dataclass(frozen=True)
class ConvergenceSample:
    parameter: str

    value: str

    peak_output_displacement_m: float

    steady_output_displacement_rms_m: float

    relative_peak_error: float

    is_reference: bool


@dataclass(frozen=True)
class ConvergenceStudy:
    samples: tuple[ConvergenceSample, ...]

    @property
    def maximum_relative_peak_error(self) -> float:

        return max(sample.relative_peak_error for sample in self.samples)


@dataclass(frozen=True)
class SensitivitySample:
    parameter: str

    value: float

    factor: float

    output_displacement_rms_m: float

    ratio_to_nominal: float


@dataclass(frozen=True)
class SensitivityStudy:
    frequency_hz: float

    nominal_output_displacement_rms_m: float

    samples: tuple[SensitivitySample, ...]


def _positive_number(value: Real,name: str) -> float:

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    result = float(value)

    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")

    return result


def _positive_integer(value: int,name: str) -> int:

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _modes(bundle: ProfileBundle,spec: ExperimentSpec) -> tuple[FirstBendingMode, ...]:

    return build_bending_modes(bundle.wing,spec.vibration.damping_ratio,n_modes=spec.vibration.n_modes,young_modulus_pa=spec.vibration.young_modulus_pa,first_natural_frequency_hz=spec.vibration.natural_frequency_hz,first_modal_mass_ratio=spec.vibration.modal_mass_ratio)


def _steady_response(bundle: ProfileBundle,spec: ExperimentSpec,frequency_hz: float,modes: Sequence[FirstBendingMode] | None = None) -> SteadyMultimodeAcousticVibration:

    resolved_modes = _modes(bundle,spec) if modes is None else tuple(modes)

    return steady_multimode_point_source_response(
        resolved_modes,

        bundle.wing,

        spec.source.position_m,

        spec.source.pressure_rms_pa_at_reference_distance,

        frequency_hz,

        spec.source.reference_distance_m,

        spec.sound_speed_m_s,

        spec.reference_sound_pressure_pa,

        output_span_fraction_from_root=bundle.profile.simulation.target_wing_span_fraction_from_root,

        surface_normal=spec.surface_normal,

        source_axis=spec.source.axis,

        directivity_exponent=spec.source.directivity_exponent,

        reference_phase_rad=spec.source.phase_rad,

        n_span=spec.sampling.n_span,

        n_chord=spec.sampling.n_chord
    )


def run_experiment(bundle: ProfileBundle,spec: ExperimentSpec) -> SimulationResult:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be an ExperimentSpec")

    modes = _modes(bundle,spec)

    return simulate_multimode(
        bundle.wing,

        modes,

        source_position_m=spec.source.position_m,

        reference_pressure_rms_pa=spec.source.pressure_rms_pa_at_reference_distance,

        frequency_hz=spec.frequency_hz,

        reference_distance_m=spec.source.reference_distance_m,

        sound_speed_m_s=spec.sound_speed_m_s,

        reference_sound_pressure_pa=spec.reference_sound_pressure_pa,

        duration_s=spec.sampling.duration_s,

        time_step_s=spec.sampling.time_step_s,

        output_span_fraction_from_root=bundle.profile.simulation.target_wing_span_fraction_from_root,

        surface_normal=spec.surface_normal,

        source_axis=spec.source.axis,

        directivity_exponent=spec.source.directivity_exponent,

        reference_phase_rad=spec.source.phase_rad,

        drive_envelope_spec=spec.drive_envelope,

        n_span=spec.sampling.n_span,

        n_chord=spec.sampling.n_chord
    )


def experiment_spec_from_profile(bundle: ProfileBundle) -> ExperimentSpec:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    profile = bundle.profile

    simulation = profile.simulation

    drive = simulation.drive

    structure = simulation.structure

    sampling = simulation.sampling

    return ExperimentSpec(
        vibration=VibrationSpec(structure.damping_ratio,structure.first_natural_frequency_hz,structure.young_modulus_pa,FIRST_MODE_MASS_RATIO,structure.mode_count),

        source=PointSourceSpec(bundle.source_position_m,spl_to_pressure_rms(drive.spl_db_at_reference,profile.acoustics.reference_sound_pressure_pa),drive.reference_distance_m,drive.phase_rad,simulation.source.axis,simulation.source.directivity_exponent,simulation.source.illumination_scope),

        sampling=SamplingSpec(sampling.duration_s,sampling.time_step_s,sampling.spanwise_panel_count,sampling.chordwise_panel_count),

        frequency_hz=drive.frequency_hz,

        sound_speed_m_s=profile.acoustics.sound_speed_m_s,

        reference_sound_pressure_pa=profile.acoustics.reference_sound_pressure_pa,

        drive_envelope=DriveEnvelopeSpec(drive.start_time_s,drive.stop_time_s,drive.rise_time_s,drive.fall_time_s),

        surface_normal=(0.0, 0.0, 1.0)
    )


def run_profile_experiment(bundle: ProfileBundle) -> SimulationResult:

    return run_experiment(bundle,experiment_spec_from_profile(bundle))


def run_frequency_sweep(bundle: ProfileBundle,spec: ExperimentSpec,frequencies_hz: Sequence[Real]) -> FrequencySweepResult:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be an ExperimentSpec")

    frequencies = tuple(_positive_number(value,f"frequencies_hz[{index}]")for index, value in enumerate(frequencies_hz))

    if not frequencies:
        raise ValueError("frequencies_hz must not be empty")

    if any(second <= first for first, second in zip(frequencies,frequencies[1:])):
        raise ValueError("frequencies_hz must be strictly increasing")

    modes = _modes(bundle,spec)

    samples = []

    for frequency in frequencies:
        response = _steady_response(bundle,spec,frequency,modes)

        samples.append(FrequencyResponseSample(frequency,response.tip_displacement_rms_phasor_m,response.output_displacement_rms_phasor_m))

    return FrequencySweepResult(tuple(samples))


def _convergence_group(bundle: ProfileBundle,parameter: str,cases: Sequence[tuple[str, ExperimentSpec]]) -> tuple[ConvergenceSample, ...]:

    results = []

    for value, case_spec in cases:
        result = run_experiment(bundle,case_spec)

        peak = max(abs(displacement)for displacement in result.output_displacement_m)

        steady = abs(result.steady_response.output_displacement_rms_phasor_m)

        results.append((value,peak,steady))

    reference_peak = results[-1][1]

    return tuple(
        ConvergenceSample(
            parameter,

            value,

            peak,

            steady,

            abs(peak - reference_peak) / reference_peak if reference_peak > 0.0 else 0.0,

            index == len(results) - 1
        )
        for index, (value, peak, steady) in enumerate(results)
    )


def run_convergence_study(bundle: ProfileBundle,spec: ExperimentSpec,*,mode_counts: Sequence[int] | None = None,panel_counts: Sequence[tuple[int, int]] | None = None,time_steps_s: Sequence[Real] | None = None) -> ConvergenceStudy:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be an ExperimentSpec")

    if mode_counts is None:
        mode_counts = tuple(sorted({max(1,spec.vibration.n_modes // 5),max(1,spec.vibration.n_modes // 2),spec.vibration.n_modes}))

    resolved_modes = tuple(sorted({_positive_integer(value,f"mode_counts[{index}]")for index, value in enumerate(mode_counts)}))

    if not resolved_modes:
        raise ValueError("mode_counts must not be empty")

    if panel_counts is None:
        panel_counts = tuple((count,spec.sampling.n_chord)for count in sorted({max(4,spec.sampling.n_span // 4),max(4,spec.sampling.n_span // 2),spec.sampling.n_span}))

    resolved_panels = []

    for index, panel_count in enumerate(panel_counts):
        if not isinstance(panel_count, Sequence) or len(panel_count) != 2:
            raise TypeError(f"panel_counts[{index}] must contain spanwise and chordwise counts")

        resolved_panels.append((_positive_integer(panel_count[0],f"panel_counts[{index}][0]"),_positive_integer(panel_count[1],f"panel_counts[{index}][1]")))

    resolved_panels = sorted(set(resolved_panels),key=lambda value: (value[0] * value[1],value))

    if not resolved_panels:
        raise ValueError("panel_counts must not be empty")

    if time_steps_s is None:
        time_steps_s = (spec.sampling.time_step_s,spec.sampling.time_step_s / 2.0)

    resolved_time_steps = tuple(sorted({_positive_number(value,f"time_steps_s[{index}]")for index, value in enumerate(time_steps_s)},reverse=True))

    if not resolved_time_steps:
        raise ValueError("time_steps_s must not be empty")

    mode_cases = tuple((str(count),replace(spec,vibration=replace(spec.vibration,n_modes=count)))for count in resolved_modes)

    panel_cases = tuple((f"{span}x{chord}",replace(spec,sampling=replace(spec.sampling,n_span=span,n_chord=chord)))for span, chord in resolved_panels)

    time_step_cases = tuple((f"{time_step:.6g}",replace(spec,sampling=replace(spec.sampling,time_step_s=time_step)))for time_step in resolved_time_steps)

    samples = _convergence_group(bundle,"mode_count",mode_cases) + _convergence_group(bundle,"panel_count",panel_cases) + _convergence_group(bundle,"time_step_s",time_step_cases)

    return ConvergenceStudy(samples)


def _sensitivity_factors(values: Sequence[Real],name: str) -> tuple[float, ...]:

    factors = {_positive_number(value,f"{name}[{index}]")for index, value in enumerate(values)}

    factors.add(1.0)

    return tuple(sorted(factors))


def run_parameter_sensitivity(bundle: ProfileBundle,spec: ExperimentSpec,*,factors: Sequence[Real] = (0.75, 1.0, 1.25)) -> SensitivityStudy:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be an ExperimentSpec")

    resolved_factors = _sensitivity_factors(factors, "factors")

    frequency = _positive_number(spec.frequency_hz, "spec.frequency_hz")

    nominal_response = _steady_response(bundle,spec,frequency)

    nominal_output = abs(nominal_response.output_displacement_rms_phasor_m)

    if nominal_output == 0.0:
        raise ValueError("nominal output displacement must be non-zero for sensitivity ratios")

    samples = []

    if spec.vibration.young_modulus_pa is not None:
        structural_parameter = "young_modulus_pa"

        structural_nominal = _positive_number(spec.vibration.young_modulus_pa,"spec.vibration.young_modulus_pa")

        structural_specs = tuple(replace(spec,vibration=replace(spec.vibration,young_modulus_pa=structural_nominal * factor))for factor in resolved_factors)

    else:
        structural_parameter = "first_natural_frequency_hz"

        structural_nominal = _positive_number(spec.vibration.natural_frequency_hz,"spec.vibration.natural_frequency_hz")

        structural_specs = tuple(replace(spec,vibration=replace(spec.vibration,natural_frequency_hz=structural_nominal * factor))for factor in resolved_factors)

    damping_nominal = _positive_number(spec.vibration.damping_ratio,"spec.vibration.damping_ratio")

    pressure_nominal = _positive_number(spec.source.pressure_rms_pa_at_reference_distance,"spec.source.pressure_rms_pa_at_reference_distance")

    if damping_nominal * max(resolved_factors) >= 1.0:
        raise ValueError("scaled damping ratios must be smaller than one")

    groups = (
        (structural_parameter,structural_nominal,structural_specs),

        ("damping_ratio",damping_nominal,tuple(replace(spec,vibration=replace(spec.vibration,damping_ratio=damping_nominal * factor))for factor in resolved_factors)),

        ("reference_pressure_rms_pa",pressure_nominal,tuple(replace(spec,source=replace(spec.source,pressure_rms_pa_at_reference_distance=pressure_nominal * factor))for factor in resolved_factors))
    )

    for parameter, nominal_value, group_specs in groups:
        for factor, group_spec in zip(resolved_factors,group_specs):
            response = _steady_response(bundle,group_spec,frequency)

            output = abs(response.output_displacement_rms_phasor_m)

            ratio = output / nominal_output

            samples.append(SensitivitySample(parameter,nominal_value * factor,factor,output,ratio))

    return SensitivityStudy(frequency,nominal_output,tuple(samples))
