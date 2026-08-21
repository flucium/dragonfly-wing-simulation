from __future__ import annotations

import math

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from .geometry import WingGeometry


__all__ = [
    "FIRST_MODE_MASS_RATIO",
    "FirstBendingMode",
    "build_bending_modes",
    "cantilever_mode_shape",
    "multimode_displacement_at_span",
    "steady_multimode_displacement_phasors"
]


FIRST_MODE_MASS_RATIO = 0.236

_CANTILEVER_BETAS = (
    1.875104068711961,
    4.694091132974174,
    7.854757438237612,
    10.995540734875467,
    14.13716839104647
)


@dataclass(frozen=True)
class FirstBendingMode:
    natural_frequency_hz: float

    angular_frequency_rad_s: float

    modal_mass_kg: float

    stiffness_n_m: float

    damping_n_s_m: float

    damping_ratio: float

    mode_index: int

    beta: float


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


def _positive_integer(value: int,name: str) -> int:

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _finite_complex(value: complex,name: str) -> complex:

    if isinstance(value, bool) or not isinstance(value, (Real, complex)):
        raise TypeError(f"{name} must be a real or complex number")

    result = complex(value)

    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")

    return result


def _validated_wing(wing: WingGeometry) -> WingGeometry:

    if not isinstance(wing, WingGeometry):
        raise TypeError("wing must be a WingGeometry")

    _positive_number(wing.length_m, "wing.length_m")

    _positive_number(wing.chord_m, "wing.chord_m")

    _positive_number(wing.thickness_m, "wing.thickness_m")

    _positive_number(wing.mass_kg, "wing.mass_kg")

    return wing


@lru_cache(maxsize=None)
def _mode_beta(mode_index: int) -> float:

    index = _positive_integer(mode_index, "mode_index")

    if index <= len(_CANTILEVER_BETAS):
        return _CANTILEVER_BETAS[index - 1]

    return (index - 0.5) * math.pi


@lru_cache(maxsize=262144)
def cantilever_mode_shape(span_fraction_from_root: Real,mode_index: int = 1) -> float:

    span_fraction = _finite_number(span_fraction_from_root, "span_fraction_from_root")

    if not 0.0 <= span_fraction <= 1.0:
        raise ValueError("span_fraction_from_root must be between 0 and 1")

    beta = _mode_beta(mode_index)

    denominator = math.sinh(beta) + math.sin(beta)

    sigma = (math.cosh(beta) + math.cos(beta)) / denominator

    one_minus_sigma = (-math.exp(-beta) + math.sin(beta) - math.cos(beta)) / denominator

    one_plus_sigma = (math.exp(beta) + math.sin(beta) + math.cos(beta)) / denominator

    def unnormalized(position: float) -> float:

        angle = beta * position

        hyperbolic_part = 0.5 * (one_minus_sigma * math.exp(angle) + one_plus_sigma * math.exp(-angle))

        return hyperbolic_part - math.cos(angle) + sigma * math.sin(angle)

    tip_value = unnormalized(1.0)

    if abs(tip_value) < 1.0e-14:
        raise ArithmeticError("cantilever mode normalization failed")

    return unnormalized(span_fraction) / tip_value


@lru_cache(maxsize=None)
def _modal_mass_ratio(mode_index: int,n_integration_points: int = 512) -> float:

    index = _positive_integer(mode_index, "mode_index")

    point_count = _positive_integer(n_integration_points, "n_integration_points")

    return sum(cantilever_mode_shape((point + 0.5) / point_count,index) ** 2 for point in range(point_count)) / point_count


def _natural_frequency_hz(wing: WingGeometry,young_modulus_pa: float,mode_index: int) -> float:

    beta = _mode_beta(mode_index)

    second_moment = wing.chord_m * wing.thickness_m**3 / 12.0

    linear_mass_density = wing.mass_kg / wing.length_m

    angular_frequency = beta**2 / wing.length_m**2 * math.sqrt(young_modulus_pa * second_moment / linear_mass_density)

    return angular_frequency / (2.0 * math.pi)


def _build_mode(wing: WingGeometry,mode_index: int,frequency_hz: float,damping_ratio: float,mass_ratio: float) -> FirstBendingMode:

    modal_mass = mass_ratio * wing.mass_kg

    angular_frequency = 2.0 * math.pi * frequency_hz

    return FirstBendingMode(
        natural_frequency_hz=frequency_hz,

        angular_frequency_rad_s=angular_frequency,

        modal_mass_kg=modal_mass,

        stiffness_n_m=modal_mass * angular_frequency**2,

        damping_n_s_m=2.0 * damping_ratio * modal_mass * angular_frequency,

        damping_ratio=damping_ratio,

        mode_index=mode_index,

        beta=_mode_beta(mode_index)
    )


def build_bending_modes(wing: WingGeometry,damping_ratio: Real,*,n_modes: int = 3,young_modulus_pa: Real | None = None,first_natural_frequency_hz: Real | None = None,first_modal_mass_ratio: Real = FIRST_MODE_MASS_RATIO) -> tuple[FirstBendingMode, ...]:

    wing = _validated_wing(wing)

    damping = _finite_number(damping_ratio, "damping_ratio")

    if not 0.0 <= damping < 1.0:
        raise ValueError("damping_ratio must be between 0 and 1")

    count = _positive_integer(n_modes, "n_modes")

    first_mass_ratio = _positive_number(first_modal_mass_ratio, "first_modal_mass_ratio")

    if (young_modulus_pa is None) == (first_natural_frequency_hz is None):
        raise ValueError("provide exactly one of young_modulus_pa or first_natural_frequency_hz")

    if young_modulus_pa is not None:
        young_modulus = _positive_number(young_modulus_pa, "young_modulus_pa")

        frequencies = tuple(_natural_frequency_hz(wing,young_modulus,index)for index in range(1, count + 1))

    else:
        first_frequency = _positive_number(first_natural_frequency_hz, "first_natural_frequency_hz")

        first_beta = _mode_beta(1)

        frequencies = tuple(first_frequency * (_mode_beta(index) / first_beta) ** 2 for index in range(1, count + 1))

    return tuple(_build_mode(wing,index,frequency,damping,first_mass_ratio if index == 1 else _modal_mass_ratio(index))for index, frequency in enumerate(frequencies, start=1))


def steady_multimode_displacement_phasors(modes: Sequence[FirstBendingMode],modal_force_rms_phasors_n: Sequence[complex],frequency_hz: Real) -> tuple[complex, ...]:

    if len(modes) != len(modal_force_rms_phasors_n) or not modes:
        raise ValueError("modes and modal forces must have equal non-zero length")

    angular_frequency = 2.0 * math.pi * _positive_number(frequency_hz, "frequency_hz")

    return tuple(_finite_complex(force,f"modal_force_rms_phasors_n[{index}]") / complex(mode.stiffness_n_m - mode.modal_mass_kg * angular_frequency**2,mode.damping_n_s_m * angular_frequency)for index, (mode, force) in enumerate(zip(modes, modal_force_rms_phasors_n)))


def multimode_displacement_at_span(modal_tip_displacement_phasors_m: Sequence[complex],span_fraction_from_root: Real) -> complex:

    if not modal_tip_displacement_phasors_m:
        raise ValueError("modal_tip_displacement_phasors_m must not be empty")

    return sum(_finite_complex(value,f"modal_tip_displacement_phasors_m[{index - 1}]") * cantilever_mode_shape(span_fraction_from_root,index)for index, value in enumerate(modal_tip_displacement_phasors_m, start=1))
