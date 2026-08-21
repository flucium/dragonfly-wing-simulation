from __future__ import annotations

import json
import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .geometry import BodyGeometry,Vector3,WingGeometry,WingInstance,WingPlanformStation,all_wings,body_geometry,target_position_body_m,target_position_m,target_wing,wing_local_to_body


__all__ = [
    "ProfileBundle",
    "Profiles",
    "load",
    "load_profile",
    "load_profile_bundle"
]


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_NOTEBOOK_PROFILE_PATH = _PROJECT_ROOT / "profile" / "notebook.json"


@dataclass(frozen=True)
class Profiles:
    @dataclass(frozen=True)

    class Notebook:
        version: str

        notebooks: str

        result_storage: str

        result_storage_path: str | None

        write_result_files: bool

        show_resolved_profile: bool

        acoustic_file: str

        dragonfly_file: str

        simulation_file: str

    @dataclass(frozen=True)
    class Dragonfly:
        @dataclass(frozen=True)

        class Reference:
            scientific_name: str

            japanese_name: str

            url: tuple[str, ...]

        version: str

        name: str

        body_geometry_model: str

        body_length_mm: float

        body_width_mm: float

        body_thickness_mm: float

        body_mass_mg: float

        wing_number: int

        fore_wing_length_mm: float

        fore_wing_width_mm: float

        fore_wing_thickness_mm: float

        fore_wing_mass_mg: float

        hind_wing_length_mm: float

        hind_wing_width_mm: float

        hind_wing_thickness_mm: float

        hind_wing_mass_mg: float

        fore_wing_planform_stations: tuple[WingPlanformStation, ...]

        hind_wing_planform_stations: tuple[WingPlanformStation, ...]

        wing_instances: tuple[WingInstance, ...]

        reference: Reference

    @dataclass(frozen=True)
    class Acoustics:
        @dataclass(frozen=True)

        class Reference:
            name: str

            product_code: str

            url: tuple[str, ...]

        version: str

        name: str

        reference_sound_pressure_pa: float

        sound_speed_m_s: float

        electrical_frequency_hz: float

        electrical_duty_cycle: float

        electrical_supply_voltage_v: float

        electrical_trigger_polarity: str

        pcb_size_vertical_mm: float

        pcb_size_horizontal_mm: float

        pcb_size_height_mm: float

        pcb_buzzer_size_diameter_mm: float

        pcb_buzzer_size_radius_mm: float

        pcb_buzzer_size_height_mm: float

        reference: Reference

        @property
        def pcb_size_without_buzzer_height_mm(self) -> float:

            return self.pcb_size_height_mm - self.pcb_buzzer_size_height_mm

    @dataclass(frozen=True)
    class Simulation:
        @dataclass(frozen=True)

        class Source:
            position_relative_to_target_m: Vector3

            axis: Vector3

            directivity_exponent: float

            @property
            def surface_normal(self) -> Vector3:

                return self.axis

        @dataclass(frozen=True)
        class Drive:
            frequency_hz: float

            spl_db_at_reference: float

            reference_distance_m: float

            phase_rad: float

            start_time_s: float

            stop_time_s: float | None

            rise_time_s: float

            fall_time_s: float

        @dataclass(frozen=True)
        class Structure:
            model: str

            mode_count: int

            young_modulus_pa: float | None

            first_natural_frequency_hz: float | None

            damping_ratio: float

        @dataclass(frozen=True)
        class Sampling:
            duration_s: float

            time_step_s: float

            spanwise_panel_count: int

            chordwise_panel_count: int

        version: str

        target_wing_type: str

        target_wing_side: str

        target_wing_span_fraction_from_root: float

        target_wing_chord_fraction_from_leading_edge: float

        source: Source

        drive: Drive

        structure: Structure

        sampling: Sampling

    notebook: Notebook

    dragonfly: Dragonfly

    acoustics: Acoustics

    simulation: Simulation


@dataclass(frozen=True)
class ProfileBundle:
    profile: Profiles

    body: BodyGeometry

    wings: tuple[WingGeometry, ...]

    wing: WingGeometry

    output_position_m: Vector3

    source_position_m: Vector3

    output_position_body_m: Vector3

    source_position_body_m: Vector3


def _load_json(path: Path) -> dict[str, Any]:

    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc

    if not isinstance(document, dict):
        raise ValueError(f"profile root must be an object: {path}")

    return document


def _object(document: dict[str, Any], key: str, location: str) -> dict[str, Any]:

    value = document.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"{location}.{key} must be an object")

    return value


def _string(document: dict[str, Any], key: str, location: str) -> str:

    value = document.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty string")

    return value


def _number(document: dict[str, Any], key: str, location: str) -> float:

    value = document.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be a number")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{location}.{key} must be finite")

    return result


def _positive_number(document: dict[str, Any], key: str, location: str) -> float:

    value = _number(document, key, location)

    if value <= 0.0:
        raise ValueError(f"{location}.{key} must be positive")

    return value


def _non_negative_number(document: dict[str, Any], key: str, location: str) -> float:

    value = _number(document, key, location)

    if value < 0.0:
        raise ValueError(f"{location}.{key} must be non-negative")

    return value


def _optional_positive_number(document: dict[str, Any], key: str, location: str) -> float | None:

    if document.get(key) is None:
        return None

    return _positive_number(document, key, location)


def _positive_integer(document: dict[str, Any], key: str, location: str) -> int:

    value = document.get(key)

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location}.{key} must be a positive integer")

    return value


def _boolean(document: dict[str, Any], key: str, location: str) -> bool:

    value = document.get(key)

    if not isinstance(value, bool):
        raise ValueError(f"{location}.{key} must be a boolean")

    return value


def _string_tuple(document: dict[str, Any], key: str, location: str) -> tuple[str, ...]:

    value = document.get(key)

    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):

        raise ValueError(f"{location}.{key} must be a non-empty string array")

    return tuple(value)


def _vector3(document: dict[str, Any], key: str, location: str) -> Vector3:

    value = document.get(key)

    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{location}.{key} must contain three numbers")

    result = []

    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f"{location}.{key}[{index}] must be a number")

        number = float(component)

        if not math.isfinite(number):
            raise ValueError(f"{location}.{key}[{index}] must be finite")

        result.append(number)

    return (result[0], result[1], result[2])


def _decode_notebook_json(document: dict[str, Any]) -> Profiles.Notebook:

    location = "notebook"

    behavior = _object(document, "notebook_behavior", location)

    acoustic = _object(document, "acoustic", location)

    dragonfly = _object(document, "dragonfly", location)

    simulation = _object(document, "simulation", location)

    storage_path = behavior.get("result_storage_path")

    if storage_path == "null":
        storage_path = None

    if storage_path is not None and not isinstance(storage_path, str):
        raise ValueError(
            "notebook.notebook_behavior.result_storage_path must be a string or null"
        )

    return Profiles.Notebook(
        version=_string(document, "version", location),
        notebooks=_string(document, "notebooks", location),
        result_storage=_string(behavior, "result_storage", f"{location}.notebook_behavior"),
        result_storage_path=storage_path,
        write_result_files=_boolean(
            behavior, "write_result_files", f"{location}.notebook_behavior"
        ),
        show_resolved_profile=_boolean(
            behavior, "show_resolved_profile", f"{location}.notebook_behavior"
        ),
        acoustic_file=_string(acoustic, "file", f"{location}.acoustic"),
        dragonfly_file=_string(dragonfly, "file", f"{location}.dragonfly"),
        simulation_file=_string(simulation, "file", f"{location}.simulation"),
    )


def _decode_planform_stations(document: dict[str, Any], location: str) -> tuple[WingPlanformStation, ...]:

    values = document.get("planform_stations")

    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"{location}.planform_stations must contain at least two stations")

    stations = []

    for index, value in enumerate(values):
        station_location = f"{location}.planform_stations[{index}]"

        if not isinstance(value, dict):
            raise ValueError(f"{station_location} must be an object")

        stations.append(
            WingPlanformStation(
                span_fraction_from_root=_number(value, "span_fraction", station_location),

                leading_edge_m=1.0e-3 * _number(value, "leading_edge_mm", station_location),

                trailing_edge_m=1.0e-3 * _number(value, "trailing_edge_mm", station_location),

                thickness_m=1.0e-3 * _positive_number(value, "thickness_mm", station_location),

                twist_rad=math.radians(_number(value, "twist_deg", station_location))
            )
        )

    if stations[0].span_fraction_from_root != 0.0 or stations[-1].span_fraction_from_root != 1.0:
        raise ValueError(f"{location}.planform_stations must start at 0 and end at 1")

    for first, second in zip(stations, stations[1:]):
        if second.span_fraction_from_root <= first.span_fraction_from_root:
            raise ValueError(f"{location}.planform station span fractions must increase")

    if any(station.trailing_edge_m <= station.leading_edge_m for station in stations):
        raise ValueError(f"{location}.planform stations must have positive chord")

    return tuple(stations)


def _decode_wing_instances(document: dict[str, Any], location: str) -> tuple[WingInstance, ...]:

    instances = _object(document, "wing_instances", location)

    result = []

    for instance_id, value in instances.items():
        instance_location = f"{location}.wing_instances.{instance_id}"

        if not isinstance(value, dict):
            raise ValueError(f"{instance_location} must be an object")

        wing_type = _string(value, "wing_type", instance_location)

        side = _string(value, "side", instance_location)

        if wing_type not in {"fore", "hind"} or side not in {"left", "right"}:
            raise ValueError(f"{instance_location} has an unsupported wing type or side")

        root_mm = _vector3(value, "root_position_body_mm", instance_location)

        orientation_deg = _vector3(value, "root_orientation_body_euler_xyz_deg", instance_location)

        result.append(
            WingInstance(
                wing_type=wing_type,

                side=side,

                root_position_body_m=tuple(component * 1.0e-3 for component in root_mm),

                root_orientation_body_euler_xyz_rad=tuple(math.radians(component)for component in orientation_deg)
            )
        )

    combinations = {(instance.wing_type, instance.side)for instance in result}

    expected = {(wing_type, side)for wing_type in ("fore", "hind")for side in ("left", "right")}

    if combinations != expected:
        raise ValueError(f"{location}.wing_instances must define fore/hind left/right")

    return tuple(result)


def _decode_dragonfly_json(document: dict[str, Any]) -> Profiles.Dragonfly:

    location = "dragonfly"

    body = _object(document, "body", location)

    wing = _object(document, "wing", location)

    fore = _object(wing, "fore_wing", f"{location}.wing")

    hind = _object(wing, "hind_wing", f"{location}.wing")

    reference = _object(document, "reference", location)

    wing_number = _positive_integer(wing, "number", f"{location}.wing")

    return Profiles.Dragonfly(
        version=_string(document, "version", location),
        name=_string(document, "name", location),
        body_geometry_model=_string(body, "geometry_model", f"{location}.body"),
        body_length_mm=_positive_number(body, "length_mm", f"{location}.body"),
        body_width_mm=_positive_number(body, "width_mm", f"{location}.body"),
        body_thickness_mm=_positive_number(body, "thickness_mm", f"{location}.body"),
        body_mass_mg=_positive_number(body, "mass_mg", f"{location}.body"),
        wing_number=wing_number,
        fore_wing_length_mm=_positive_number(
            fore, "length_mm", f"{location}.wing.fore_wing"
        ),
        fore_wing_width_mm=_positive_number(
            fore, "width_mm", f"{location}.wing.fore_wing"
        ),
        fore_wing_thickness_mm=_positive_number(
            fore, "thickness_mm", f"{location}.wing.fore_wing"
        ),
        fore_wing_mass_mg=_positive_number(
            fore, "mass_mg", f"{location}.wing.fore_wing"
        ),
        hind_wing_length_mm=_positive_number(
            hind, "length_mm", f"{location}.wing.hind_wing"
        ),
        hind_wing_width_mm=_positive_number(
            hind, "width_mm", f"{location}.wing.hind_wing"
        ),
        hind_wing_thickness_mm=_positive_number(
            hind, "thickness_mm", f"{location}.wing.hind_wing"
        ),
        hind_wing_mass_mg=_positive_number(
            hind, "mass_mg", f"{location}.wing.hind_wing"
        ),
        fore_wing_planform_stations=_decode_planform_stations(fore, f"{location}.wing.fore_wing"),
        hind_wing_planform_stations=_decode_planform_stations(hind, f"{location}.wing.hind_wing"),
        wing_instances=_decode_wing_instances(document, location),
        reference=Profiles.Dragonfly.Reference(
            scientific_name=_string(
                reference, "scientific_name", f"{location}.reference"
            ),
            japanese_name=_string(reference, "japanese_name", f"{location}.reference"),
            url=_string_tuple(reference, "url", f"{location}.reference"),
        ),
    )


def _decode_acoustics_json(document: dict[str, Any]) -> Profiles.Acoustics:

    location = "acoustic"

    electrical = _object(document, "electrical", location)

    sound = _object(document, "sound", location)

    pcb_size = _object(document, "pcb_size", location)

    buzzer_size = _object(pcb_size, "buzzer_size", f"{location}.pcb_size")

    reference = _object(document, "reference", location)

    duty_cycle = _number(electrical, "duty_cycle", f"{location}.electrical")

    if not 0.0 <= duty_cycle <= 1.0:
        raise ValueError("acoustic.electrical.duty_cycle must be between 0 and 1")

    diameter = _positive_number(
        buzzer_size, "diameter_mm", f"{location}.pcb_size.buzzer_size"
    )

    radius = _positive_number(
        buzzer_size, "radius_mm", f"{location}.pcb_size.buzzer_size"
    )

    pcb_height = _positive_number(pcb_size,"height_mm",f"{location}.pcb_size")

    buzzer_height = _positive_number(buzzer_size,"height_mm",f"{location}.pcb_size.buzzer_size")

    if abs(diameter - 2.0 * radius) > 1.0e-9:
        raise ValueError("acoustic buzzer diameter_mm must equal twice radius_mm")

    if buzzer_height >= pcb_height:
        raise ValueError("acoustic buzzer height_mm must be smaller than pcb height_mm")

    return Profiles.Acoustics(
        version=_string(document, "version", location),
        name=_string(document, "name", location),
        reference_sound_pressure_pa=_positive_number(
            sound, "pressure_pa", f"{location}.sound"
        ),
        sound_speed_m_s=_positive_number(sound, "speed_ms", f"{location}.sound"),
        electrical_frequency_hz=_positive_number(
            electrical, "frequency_hz", f"{location}.electrical"
        ),
        electrical_duty_cycle=duty_cycle,
        electrical_supply_voltage_v=_positive_number(
            electrical, "supply_voltage_v", f"{location}.electrical"
        ),
        electrical_trigger_polarity=_string(
            electrical, "trigger_polarity", f"{location}.electrical"
        ),
        pcb_size_vertical_mm=_positive_number(
            pcb_size, "vertical_mm", f"{location}.pcb_size"
        ),
        pcb_size_horizontal_mm=_positive_number(
            pcb_size, "horizontal_mm", f"{location}.pcb_size"
        ),
        pcb_size_height_mm=pcb_height,
        pcb_buzzer_size_diameter_mm=diameter,
        pcb_buzzer_size_radius_mm=radius,
        pcb_buzzer_size_height_mm=buzzer_height,
        reference=Profiles.Acoustics.Reference(
            name=_string(reference, "name", f"{location}.reference"),
            product_code=_string(
                reference, "product_code", f"{location}.reference"
            ),
            url=_string_tuple(reference, "url", f"{location}.reference"),
        ),
    )


def _decode_simulation_json(document: dict[str, Any]) -> Profiles.Simulation:

    location = "simulation"

    target = _object(document, "target_wing", location)

    position = _object(target, "position_on", f"{location}.target_wing")

    source = _object(document, "source", location)

    drive = _object(document, "drive", location)

    structure = _object(document, "structure", location)

    sampling = _object(document, "sampling", location)

    wing_type = _string(target, "type", f"{location}.target_wing")

    if wing_type not in {"fore", "hind"}:
        raise ValueError("simulation.target_wing.type must be 'fore' or 'hind'")

    side = _string(target, "side", f"{location}.target_wing")

    if side not in {"left", "right"}:
        raise ValueError("simulation.target_wing.side must be 'left' or 'right'")

    span_fraction = _number(
        position, "span_fraction_from_root", f"{location}.target_wing.position_on"
    )

    chord_fraction = _number(
        position,
        "chord_fraction_from_leading_edge",
        f"{location}.target_wing.position_on",
    )

    if not 0.0 <= span_fraction <= 1.0 or not 0.0 <= chord_fraction <= 1.0:
        raise ValueError("simulation target-wing position fractions must be between 0 and 1")

    axis_key = "axis" if "axis" in source else "surface_normal"

    normal = _vector3(source, axis_key, f"{location}.source")

    if math.sqrt(sum(component * component for component in normal)) == 0.0:
        raise ValueError("simulation.source.axis must not be zero")

    directivity_exponent = 0.0

    if "directivity_exponent" in source:
        directivity_exponent = _non_negative_number(
            source, "directivity_exponent", f"{location}.source"
        )

    stop_value = drive.get("stop_time_s")

    stop_time = None

    if stop_value is not None:
        stop_time = _positive_number(drive, "stop_time_s", f"{location}.drive")

    young_modulus = _optional_positive_number(
        structure, "young_modulus_pa", f"{location}.structure"
    )

    first_frequency = _optional_positive_number(
        structure, "first_natural_frequency_hz", f"{location}.structure"
    )

    if (young_modulus is None) == (first_frequency is None):
        raise ValueError(
            "simulation.structure must define exactly one of young_modulus_pa "
            "or first_natural_frequency_hz"
        )

    damping = _non_negative_number(
        structure, "damping_ratio", f"{location}.structure"
    )

    if damping >= 1.0:
        raise ValueError("simulation.structure.damping_ratio must be less than 1")

    model = _string(structure, "model", f"{location}.structure")

    if model != "cantilever_multimode":
        raise ValueError("only the 'cantilever_multimode' structural model is supported")

    duration = _positive_number(sampling, "duration_s", f"{location}.sampling")

    start_time = _non_negative_number(drive, "start_time_s", f"{location}.drive")

    if start_time > duration:
        raise ValueError("simulation.drive.start_time_s must not exceed duration_s")

    if stop_time is not None and not start_time < stop_time <= duration:
        raise ValueError("simulation.drive.stop_time_s must be after start and within duration")

    return Profiles.Simulation(
        version=_string(document, "version", location),
        target_wing_type=wing_type,
        target_wing_side=side,
        target_wing_span_fraction_from_root=span_fraction,
        target_wing_chord_fraction_from_leading_edge=chord_fraction,
        source=Profiles.Simulation.Source(
            position_relative_to_target_m=_vector3(
                source, "position_relative_to_target_m", f"{location}.source"
            ),
            axis=normal,
            directivity_exponent=directivity_exponent,
        ),
        drive=Profiles.Simulation.Drive(
            frequency_hz=_positive_number(
                drive, "frequency_hz", f"{location}.drive"
            ),
            spl_db_at_reference=_number(
                drive, "spl_db_at_reference", f"{location}.drive"
            ),
            reference_distance_m=_positive_number(
                drive, "reference_distance_m", f"{location}.drive"
            ),
            phase_rad=_number(drive, "phase_rad", f"{location}.drive"),
            start_time_s=start_time,
            stop_time_s=stop_time,
            rise_time_s=_non_negative_number(
                drive, "rise_time_s", f"{location}.drive"
            ),
            fall_time_s=_non_negative_number(
                drive, "fall_time_s", f"{location}.drive"
            ),
        ),
        structure=Profiles.Simulation.Structure(
            model=model,
            mode_count=_positive_integer(
                structure, "mode_count", f"{location}.structure"
            ),
            young_modulus_pa=young_modulus,
            first_natural_frequency_hz=first_frequency,
            damping_ratio=damping,
        ),
        sampling=Profiles.Simulation.Sampling(
            duration_s=duration,
            time_step_s=_positive_number(
                sampling, "time_step_s", f"{location}.sampling"
            ),
            spanwise_panel_count=_positive_integer(
                sampling, "spanwise_panel_count", f"{location}.sampling"
            ),
            chordwise_panel_count=_positive_integer(
                sampling, "chordwise_panel_count", f"{location}.sampling"
            ),
        ),
    )


def _resolve_profile_path(project_root: Path, profile_file: str) -> Path:

    path = Path(profile_file).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def load(notebook_profile_path: str | Path = _NOTEBOOK_PROFILE_PATH) -> Profiles:

    notebook_path = Path(notebook_profile_path).expanduser().resolve()

    if notebook_path.is_dir():
        notebook_path = notebook_path / "profile" / "notebook.json"

    notebook_document = _load_json(notebook_path)

    notebook = _decode_notebook_json(notebook_document)

    project_root = notebook_path.parent.parent

    return Profiles(
        notebook=notebook,
        dragonfly=_decode_dragonfly_json(
            _load_json(_resolve_profile_path(project_root, notebook.dragonfly_file))
        ),
        acoustics=_decode_acoustics_json(
            _load_json(_resolve_profile_path(project_root, notebook.acoustic_file))
        ),
        simulation=_decode_simulation_json(
            _load_json(_resolve_profile_path(project_root, notebook.simulation_file))
        ),
    )


def load_profile_bundle(notebook_profile_path: str | Path | None = None,) -> ProfileBundle:

    profile = load() if notebook_profile_path is None else load(notebook_profile_path)

    wing = target_wing(profile)

    output_position = target_position_m(profile)

    offset = profile.simulation.source.position_relative_to_target_m

    source_position = (
        output_position[0] + offset[0],
        output_position[1] + offset[1],
        output_position[2] + offset[2],
    )

    output_position_body = wing_local_to_body(wing, output_position)

    source_position_body = wing_local_to_body(wing, source_position)

    return ProfileBundle(
        profile=profile,

        body=body_geometry(profile),

        wings=all_wings(profile),

        wing=wing,

        output_position_m=output_position,

        source_position_m=source_position,

        output_position_body_m=output_position_body,

        source_position_body_m=source_position_body,
    )


load_profile = load
