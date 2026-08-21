from __future__ import annotations

import math

from collections.abc import Sequence
from numbers import Real
from typing import Any
from .geometry import wing_local_to_body
from .profiles import ProfileBundle
from .simulation import SimulationResult
from .whole_dragonfly import WholeDragonflySimulationResult
from .wing_surface import multimode_deformed_wing_surface,wing_surface_mesh,wing_surface_mesh_body


__all__ = [
    "animate_dragonfly_3d",
    "animate_whole_dragonfly_3d",
    "plot_dragonfly_3d",
    "plot_whole_dragonfly_3d",
]


def _pyplot() -> Any:

    try:
        import matplotlib.pyplot as plt

    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib; install it before plotting") from exc

    return plt


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


def _sample_index(result: SimulationResult,index: int) -> int:

    if not isinstance(result, SimulationResult) or not result.time_s:
        raise TypeError("result must be a non-empty SimulationResult")

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("sample_index must be an integer")

    resolved = index if index >= 0 else len(result.time_s) + index

    if not 0 <= resolved < len(result.time_s):
        raise IndexError("sample_index is outside the simulation result")

    return resolved


def _whole_sample_index(result: WholeDragonflySimulationResult,index: int) -> int:

    if not isinstance(result, WholeDragonflySimulationResult) or not result.time_s:
        raise TypeError("result must be a non-empty WholeDragonflySimulationResult")

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("sample_index must be an integer")

    resolved = index if index >= 0 else len(result.time_s) + index

    if not 0 <= resolved < len(result.time_s):
        raise IndexError("sample_index is outside the whole-dragonfly result")

    return resolved


def _body_surface(bundle: ProfileBundle) -> tuple[tuple[tuple[float, ...], ...], ...]:

    longitude = tuple(2.0 * math.pi * index / 32 for index in range(33))

    latitude = tuple(-0.5 * math.pi + math.pi * index / 16 for index in range(17))

    x_rows = []

    y_rows = []

    z_rows = []

    for lat in latitude:
        x_rows.append(tuple(0.5 * bundle.body.length_m * math.cos(lat) * math.cos(lon)for lon in longitude))

        y_rows.append(tuple(0.5 * bundle.body.width_m * math.cos(lat) * math.sin(lon)for lon in longitude))

        z_rows.append(tuple(0.5 * bundle.body.thickness_m * math.sin(lat)for _ in longitude))

    return tuple(x_rows),tuple(y_rows),tuple(z_rows)


def _surface_faces(x_rows: Sequence[Sequence[Real]],y_rows: Sequence[Sequence[Real]],z_rows: Sequence[Sequence[Real]]) -> tuple[tuple[tuple[float, float, float], ...], ...]:

    return tuple(
        tuple(
            (float(x_rows[row][column]),float(y_rows[row][column]),float(z_rows[row][column]))
            for row, column in (
                (row_index, column_index),
                (row_index + 1, column_index),
                (row_index + 1, column_index + 1),
                (row_index, column_index + 1)
            )
        )
        for row_index in range(len(x_rows) - 1)
        for column_index in range(len(x_rows[row_index]) - 1)
    )


def _selected_wing(first: Any,second: Any) -> bool:

    return first.wing_type == second.wing_type and first.side == second.side


def _wing_mesh(wing: Any,result: SimulationResult | None,sample_index: int,deformation_scale: float) -> Any:

    if result is not None and _selected_wing(wing, result.wing):
        displacements = tuple(history[sample_index]for history in result.modal_displacement_by_mode_m)

        mesh = multimode_deformed_wing_surface(wing,displacements,n_span=51,n_chord=11,deformation_scale=deformation_scale)

    else:
        mesh = wing_surface_mesh(wing,n_span=51,n_chord=11)

    return wing_surface_mesh_body(wing, mesh)


def _simulation_for_wing(wing: Any,result: SimulationResult | WholeDragonflySimulationResult | None) -> SimulationResult | None:

    if result is None:
        return None

    if isinstance(result, SimulationResult):
        return result if _selected_wing(wing,result.wing) else None

    if isinstance(result, WholeDragonflySimulationResult):
        matches = tuple(
            wing_result.simulation
            for wing_result in result.wings
            if _selected_wing(wing,wing_result.wing)
        )

        if len(matches) != 1:
            raise ValueError("whole-dragonfly result does not match the geometry")

        return matches[0]

    raise TypeError("result must be a SimulationResult or WholeDragonflySimulationResult")


def _wing_coordinates(wing: Any) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:

    mesh = wing_surface_mesh_body(wing,wing_surface_mesh(wing,n_span=11,n_chord=3))

    return tuple(tuple(value for row in rows for value in row)for rows in (mesh.x_m, mesh.y_m, mesh.z_m))


def _padded_limits(values: Sequence[Real]) -> tuple[float, float]:

    minimum = float(min(values))

    maximum = float(max(values))

    padding = max(0.05 * (maximum - minimum),1.0e-4)

    return minimum - padding,maximum + padding


def _vector_add(first: tuple[float, float, float],second: tuple[float, float, float]) -> tuple[float, float, float]:

    return tuple(first[index] + second[index]for index in range(3))


def _vector_scale(vector: tuple[float, float, float],scale: float) -> tuple[float, float, float]:

    return tuple(component * scale for component in vector)


def _dot(first: tuple[float, float, float],second: tuple[float, float, float]) -> float:

    return sum(first[index] * second[index]for index in range(3))


def _cross(first: tuple[float, float, float],second: tuple[float, float, float]) -> tuple[float, float, float]:

    return (first[1] * second[2] - first[2] * second[1],first[2] * second[0] - first[0] * second[2],first[0] * second[1] - first[1] * second[0])


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:

    magnitude = math.sqrt(_dot(vector, vector))

    if magnitude == 0.0:
        raise ValueError("source position must differ from output position")

    return _vector_scale(vector,1.0 / magnitude)


def _ring(center: tuple[float, float, float],radius: float,first_axis: tuple[float, float, float],second_axis: tuple[float, float, float],count: int = 24) -> tuple[tuple[float, float, float], ...]:

    return tuple(
        _vector_add(center,_vector_add(_vector_scale(first_axis,radius * math.cos(2.0 * math.pi * index / count)),_vector_scale(second_axis,radius * math.sin(2.0 * math.pi * index / count))))
        for index in range(count)
    )


def _cylinder_faces(start: tuple[float, float, float],stop: tuple[float, float, float],radius: float,first_axis: tuple[float, float, float],second_axis: tuple[float, float, float]) -> tuple[tuple[tuple[float, float, float], ...], ...]:

    start_ring = _ring(start,radius,first_axis,second_axis)

    stop_ring = _ring(stop,radius,first_axis,second_axis)

    sides = tuple((start_ring[index],start_ring[(index + 1) % len(start_ring)],stop_ring[(index + 1) % len(stop_ring)],stop_ring[index])for index in range(len(start_ring)))

    return (start_ring,tuple(reversed(stop_ring))) + sides


def _box_faces(start: tuple[float, float, float],stop: tuple[float, float, float],half_width: float,half_height: float,first_axis: tuple[float, float, float],second_axis: tuple[float, float, float]) -> tuple[tuple[tuple[float, float, float], ...], ...]:

    def corners(center: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:

        return tuple(_vector_add(center,_vector_add(_vector_scale(first_axis,width_sign * half_width),_vector_scale(second_axis,height_sign * half_height)))for width_sign, height_sign in ((-1.0, -1.0),(1.0, -1.0),(1.0, 1.0),(-1.0, 1.0)))

    first = corners(start)

    second = corners(stop)

    return (first,tuple(reversed(second))) + tuple((first[index],first[(index + 1) % 4],second[(index + 1) % 4],second[index])for index in range(4))


def _source_module_faces(bundle: ProfileBundle) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...],tuple[tuple[tuple[float, float, float], ...], ...],tuple[tuple[float, float, float], ...]]:

    acoustics = bundle.profile.acoustics

    source = bundle.source_position_m

    outward = _unit(tuple(source[index] - bundle.output_position_m[index]for index in range(3)))

    reference = (1.0, 0.0, 0.0) if abs(outward[0]) < 0.9 else (0.0, 1.0, 0.0)

    first_axis = _unit(tuple(reference[index] - _dot(reference,outward) * outward[index]for index in range(3)))

    second_axis = _unit(_cross(outward,first_axis))

    buzzer_height = acoustics.pcb_buzzer_size_height_mm * 1.0e-3

    module_height = acoustics.pcb_size_height_mm * 1.0e-3

    buzzer_stop = _vector_add(source,_vector_scale(outward,buzzer_height))

    module_stop = _vector_add(source,_vector_scale(outward,module_height))

    buzzer_local = _cylinder_faces(source,buzzer_stop,acoustics.pcb_buzzer_size_radius_mm * 1.0e-3,first_axis,second_axis)

    pcb_local = _box_faces(buzzer_stop,module_stop,0.5 * acoustics.pcb_size_horizontal_mm * 1.0e-3,0.5 * acoustics.pcb_size_vertical_mm * 1.0e-3,first_axis,second_axis)

    buzzer_body = tuple(tuple(wing_local_to_body(bundle.wing,point)for point in face)for face in buzzer_local)

    pcb_body = tuple(tuple(wing_local_to_body(bundle.wing,point)for point in face)for face in pcb_local)

    points = tuple(point for faces in (buzzer_body,pcb_body)for face in faces for point in face)

    return buzzer_body,pcb_body,points


def _draw(axis: Any,bundle: ProfileBundle,result: SimulationResult | WholeDragonflySimulationResult | None,sample_index: int,deformation_scale: float) -> None:

    try:
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib; install it before plotting") from exc

    body_x, body_y, body_z = _body_surface(bundle)

    axis.add_collection3d(Poly3DCollection(_surface_faces(body_x,body_y,body_z),facecolor="0.25",alpha=0.55,linewidth=0.0))

    for wing in bundle.wings:
        mesh = _wing_mesh(wing,_simulation_for_wing(wing,result),sample_index,deformation_scale)

        selected = _selected_wing(wing, bundle.wing)

        axis.add_collection3d(Poly3DCollection(_surface_faces(mesh.x_m,mesh.y_m,mesh.z_m),facecolor="tab:blue" if selected else "0.65",alpha=0.85 if selected else 0.55,linewidth=0.15,edgecolor="0.35"))

    buzzer_faces, pcb_faces, module_points = _source_module_faces(bundle)

    axis.add_collection3d(Poly3DCollection(pcb_faces,facecolor="seagreen",alpha=0.55,linewidth=0.2,edgecolor="0.25"))

    axis.add_collection3d(Poly3DCollection(buzzer_faces,facecolor="tab:orange",alpha=0.85,linewidth=0.2,edgecolor="0.25"))

    if isinstance(result, WholeDragonflySimulationResult):
        output_positions = tuple(
            wing_result.output_position_body_m
            for wing_result in result.wings
            if wing_result.is_illuminated
        )

        source_position = result.source_position_body_m

    else:
        output_positions = (bundle.output_position_body_m,)

        source_position = bundle.source_position_body_m

    for output_index, output_position in enumerate(output_positions):
        axis.scatter(*output_position,color="tab:red",s=28,label="wing outputs" if output_index == 0 else None)

    axis.scatter(*source_position,color="tab:orange",s=32,label="buzzer opening")

    for output_position in output_positions:
        axis.plot((source_position[0],output_position[0]),(source_position[1],output_position[1]),(source_position[2],output_position[2]),color="0.4",linestyle="--",linewidth=0.8)

    x_values = [-0.5 * bundle.body.length_m,0.5 * bundle.body.length_m,source_position[0]] + [point[0]for point in output_positions]

    y_values = [-0.5 * bundle.body.width_m,0.5 * bundle.body.width_m,source_position[1]] + [point[1]for point in output_positions]

    z_values = [-0.5 * bundle.body.thickness_m,0.5 * bundle.body.thickness_m,source_position[2]] + [point[2]for point in output_positions]

    for wing in bundle.wings:
        wing_x, wing_y, wing_z = _wing_coordinates(wing)

        x_values.extend(wing_x)

        y_values.extend(wing_y)

        z_values.extend(wing_z)

    x_values.extend(point[0]for point in module_points)

    y_values.extend(point[1]for point in module_points)

    z_values.extend(point[2]for point in module_points)

    limits = (_padded_limits(x_values),_padded_limits(y_values),_padded_limits(z_values))

    axis.set_xlim(*limits[0])

    axis.set_ylim(*limits[1])

    axis.set_zlim(*limits[2])

    if hasattr(axis, "set_box_aspect"):
        axis.set_box_aspect(tuple(stop - start for start, stop in limits))

    for coordinate_axis in (axis.xaxis,axis.yaxis,axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=5))
        coordinate_axis.set_major_formatter(FuncFormatter(lambda value,position: f"{value * 1000.0:g}"))

    axis.set_xlabel("body x [mm]",labelpad=12)

    axis.set_ylabel("body y [mm]",labelpad=12)

    axis.set_zlabel("body z [mm]",labelpad=10)

    axis.tick_params(axis="x",labelsize=9,pad=1)

    axis.tick_params(axis="y",labelsize=9,pad=1)

    axis.tick_params(axis="z",labelsize=9,pad=3)

    axis.view_init(elev=24.0, azim=-62.0)

    axis.legend(loc="upper right")


def plot_dragonfly_3d(bundle: ProfileBundle,result: SimulationResult | None = None,*,sample_index: int = -1,deformation_scale: Real = 1.0) -> Any:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    scale = _positive_number(deformation_scale, "deformation_scale")

    index = 0 if result is None else _sample_index(result, sample_index)

    plt = _pyplot()

    figure = plt.figure(figsize=(10, 7))

    axis = figure.add_subplot(111, projection="3d")

    _draw(axis,bundle,result,index,scale)

    axis.set_title(f"{bundle.profile.dragonfly.name}: 3D geometry" if result is None else f"t = {result.time_s[index]:.6g} s, selected wing deformation = {scale:g}x",pad=16)

    figure.subplots_adjust(left=0.03,right=0.90,bottom=0.10,top=0.88)

    return figure


def animate_dragonfly_3d(bundle: ProfileBundle,result: SimulationResult,*,deformation_scale: Real = 1.0,frame_step: int = 1,interval_ms: Real = 33.0) -> Any:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    _sample_index(result, 0)

    scale = _positive_number(deformation_scale, "deformation_scale")

    step = _positive_integer(frame_step, "frame_step")

    interval = _positive_number(interval_ms, "interval_ms")

    try:
        from matplotlib.animation import FuncAnimation

    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib; install it before animating") from exc

    plt = _pyplot()

    frame_indices = tuple(range(0, len(result.time_s), step))

    if frame_indices[-1] != len(result.time_s) - 1:
        frame_indices += (len(result.time_s) - 1,)

    figure = plt.figure(figsize=(10, 7))

    figure.subplots_adjust(left=0.03,right=0.90,bottom=0.10,top=0.88)

    axis = figure.add_subplot(111, projection="3d")

    def update(frame_number: int) -> tuple[Any, ...]:

        index = frame_indices[frame_number]

        axis.clear()

        _draw(axis,bundle,result,index,scale)

        axis.set_title(f"t = {result.time_s[index]:.6g} s, selected wing deformation = {scale:g}x",pad=16)

        return tuple(axis.collections) + tuple(axis.lines)

    return FuncAnimation(figure,update,frames=len(frame_indices),interval=interval,blit=False)


def plot_whole_dragonfly_3d(bundle: ProfileBundle,result: WholeDragonflySimulationResult,*,sample_index: int = -1,deformation_scale: Real = 1.0) -> Any:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    index = _whole_sample_index(result,sample_index)

    scale = _positive_number(deformation_scale,"deformation_scale")

    plt = _pyplot()

    figure = plt.figure(figsize=(10, 7))

    axis = figure.add_subplot(111, projection="3d")

    _draw(axis,bundle,result,index,scale)

    axis.set_title(f"t = {result.time_s[index]:.6g} s, all four wing deformations = {scale:g}x",pad=16)

    figure.subplots_adjust(left=0.03,right=0.90,bottom=0.10,top=0.88)

    return figure


def animate_whole_dragonfly_3d(bundle: ProfileBundle,result: WholeDragonflySimulationResult,*,deformation_scale: Real = 1.0,frame_step: int = 1,interval_ms: Real = 33.0) -> Any:

    if not isinstance(bundle, ProfileBundle):
        raise TypeError("bundle must be a ProfileBundle")

    _whole_sample_index(result,0)

    scale = _positive_number(deformation_scale,"deformation_scale")

    step = _positive_integer(frame_step,"frame_step")

    interval = _positive_number(interval_ms,"interval_ms")

    try:
        from matplotlib.animation import FuncAnimation

    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib; install it before animating") from exc

    plt = _pyplot()

    frame_indices = tuple(range(0,len(result.time_s),step))

    if frame_indices[-1] != len(result.time_s) - 1:
        frame_indices += (len(result.time_s) - 1,)

    figure = plt.figure(figsize=(10, 7))

    figure.subplots_adjust(left=0.03,right=0.90,bottom=0.10,top=0.88)

    axis = figure.add_subplot(111, projection="3d")

    def update(frame_number: int) -> tuple[Any, ...]:

        index = frame_indices[frame_number]

        axis.clear()

        _draw(axis,bundle,result,index,scale)

        axis.set_title(f"t = {result.time_s[index]:.6g} s, all four wing deformations = {scale:g}x",pad=16)

        return tuple(axis.collections) + tuple(axis.lines)

    return FuncAnimation(figure,update,frames=len(frame_indices),interval=interval,blit=False)
