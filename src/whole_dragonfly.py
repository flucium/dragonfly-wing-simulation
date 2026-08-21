from __future__ import annotations

import math

from dataclasses import dataclass
from .experiment import ExperimentSpec
from .geometry import (
    BodyGeometry,
    Vector3,
    WingGeometry,
    body_to_wing_local,
    body_vector_to_wing_local,
    point_on_wing,
    wing_local_to_body,
    wing_local_vector_to_body,
)
from .profiles import ProfileBundle
from .simulation import SimulationResult,simulate_multimode
from .vibration import FirstBendingMode,build_bending_modes


__all__ = [
    "WholeDragonflySimulationResult",
    "WingExperimentResult",
    "run_whole_dragonfly_experiment",
]


ComplexVector3 = tuple[complex, complex, complex]


@dataclass(frozen=True)
class WingExperimentResult:
    wing: WingGeometry

    simulation: SimulationResult

    output_position_local_m: Vector3

    output_position_body_m: Vector3

    source_position_local_m: Vector3

    source_axis_local: Vector3 | None

    is_selected: bool

    is_illuminated: bool

    @property
    def acoustic_force_rms_phasor_body_n(self) -> ComplexVector3:

        local_force = tuple(
            sum(
                sample.pressure_rms_phasor_pa
                * sample.incidence_factor
                * sample.area_m2
                * sample.surface_normal[component]
                for sample in self.simulation.pressure_field.samples
            )
            for component in range(3)
        )

        real_body = wing_local_vector_to_body(
            self.wing,tuple(value.real for value in local_force)
        )

        imaginary_body = wing_local_vector_to_body(
            self.wing,tuple(value.imag for value in local_force)
        )

        return tuple(
            complex(real_body[index],imaginary_body[index])for index in range(3)
        )

    @property
    def peak_output_displacement_m(self) -> float:

        return max(abs(value)for value in self.simulation.output_displacement_m)


@dataclass(frozen=True)
class WholeDragonflySimulationResult:
    body: BodyGeometry

    source_position_body_m: Vector3

    source_axis_body: Vector3 | None

    body_boundary: str

    wings: tuple[WingExperimentResult, ...]

    @property
    def time_s(self) -> tuple[float, ...]:

        return self.wings[0].simulation.time_s

    @property
    def selected_wing(self) -> WingExperimentResult:

        return next(result for result in self.wings if result.is_selected)

    @property
    def maximum_wing(self) -> WingExperimentResult:

        return max(self.wings,key=lambda result: result.peak_output_displacement_m)

    @property
    def net_acoustic_force_rms_phasor_body_n(self) -> ComplexVector3:

        forces = tuple(result.acoustic_force_rms_phasor_body_n for result in self.wings)

        return tuple(sum(force[index]for force in forces)for index in range(3))

    @property
    def fixed_body_reaction_rms_phasor_n(self) -> ComplexVector3:

        return tuple(-value for value in self.net_acoustic_force_rms_phasor_body_n)

    def wing_result(self,wing_type: str,side: str) -> WingExperimentResult:

        matches = tuple(
            result
            for result in self.wings
            if result.wing.wing_type == wing_type and result.wing.side == side
        )

        if len(matches) != 1:
            raise ValueError(f"expected one {wing_type} {side} wing result")

        return matches[0]


def _bending_frequency_factor(wing: WingGeometry) -> float:

    second_moment = wing.chord_m * wing.thickness_m**3 / 12.0

    linear_mass_density = wing.mass_kg / wing.length_m

    return math.sqrt(second_moment / linear_mass_density) / wing.length_m**2


def _modes_for_wing(bundle: ProfileBundle,spec: ExperimentSpec,wing: WingGeometry) -> tuple[FirstBendingMode, ...]:

    vibration = spec.vibration

    if vibration.young_modulus_pa is not None:
        return build_bending_modes(
            wing,
            vibration.damping_ratio,
            n_modes=vibration.n_modes,
            young_modulus_pa=vibration.young_modulus_pa,
            first_modal_mass_ratio=vibration.modal_mass_ratio,
        )

    if vibration.natural_frequency_hz is None:
        raise ValueError("the vibration specification has no frequency calibration")

    reference_factor = _bending_frequency_factor(bundle.wing)

    wing_frequency = (
        vibration.natural_frequency_hz
        * _bending_frequency_factor(wing)
        / reference_factor
    )

    return build_bending_modes(
        wing,
        vibration.damping_ratio,
        n_modes=vibration.n_modes,
        first_natural_frequency_hz=wing_frequency,
        first_modal_mass_ratio=vibration.modal_mass_ratio,
    )


def run_whole_dragonfly_experiment(bundle: ProfileBundle,spec: ExperimentSpec) -> WholeDragonflySimulationResult:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be an ExperimentSpec")

    source_position_body = wing_local_to_body(bundle.wing,spec.source.position_m)

    source_axis_body = (
        None
        if spec.source.axis is None
        else wing_local_vector_to_body(bundle.wing,spec.source.axis)
    )

    simulation_profile = bundle.profile.simulation

    wing_results = []

    for wing in bundle.wings:
        is_selected = (
            wing.wing_type == bundle.wing.wing_type and wing.side == bundle.wing.side
        )

        if spec.source.illumination_scope not in {"target_wing", "all_wings"}:
            raise ValueError(
                "source illumination_scope must be 'target_wing' or 'all_wings'"
            )

        is_illuminated = (
            spec.source.illumination_scope == "all_wings" or is_selected
        )

        source_position_local = body_to_wing_local(wing,source_position_body)

        source_axis_local = (
            None
            if source_axis_body is None
            else body_vector_to_wing_local(wing,source_axis_body)
        )

        output_position_local = point_on_wing(
            wing,
            simulation_profile.target_wing_span_fraction_from_root,
            simulation_profile.target_wing_chord_fraction_from_leading_edge,
        )

        result = simulate_multimode(
            wing,
            _modes_for_wing(bundle,spec,wing),
            source_position_m=source_position_local,
            reference_pressure_rms_pa=(
                spec.source.pressure_rms_pa_at_reference_distance
                if is_illuminated
                else 0.0
            ),
            frequency_hz=spec.frequency_hz,
            reference_distance_m=spec.source.reference_distance_m,
            sound_speed_m_s=spec.sound_speed_m_s,
            reference_sound_pressure_pa=spec.reference_sound_pressure_pa,
            duration_s=spec.sampling.duration_s,
            time_step_s=spec.sampling.time_step_s,
            output_span_fraction_from_root=simulation_profile.target_wing_span_fraction_from_root,
            surface_normal=spec.surface_normal,
            source_axis=source_axis_local,
            directivity_exponent=spec.source.directivity_exponent,
            reference_phase_rad=spec.source.phase_rad,
            drive_envelope_spec=spec.drive_envelope,
            n_span=spec.sampling.n_span,
            n_chord=spec.sampling.n_chord,
        )

        wing_results.append(
            WingExperimentResult(
                wing=wing,
                simulation=result,
                output_position_local_m=output_position_local,
                output_position_body_m=wing_local_to_body(wing,output_position_local),
                source_position_local_m=source_position_local,
                source_axis_local=source_axis_local,
                is_selected=is_selected,
                is_illuminated=is_illuminated,
            )
        )

    time = wing_results[0].simulation.time_s

    if any(result.simulation.time_s != time for result in wing_results[1:]):
        raise ArithmeticError("whole-dragonfly wing histories are not time aligned")

    return WholeDragonflySimulationResult(
        body=bundle.body,
        source_position_body_m=source_position_body,
        source_axis_body=source_axis_body,
        body_boundary="fixed",
        wings=tuple(wing_results),
    )
