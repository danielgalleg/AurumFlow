from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from .metrics import compute_metrics
from .physics import ClassifierSimulation


class VideoRecorder:
    def __init__(self, output_dir: str = "recordings", fps: int = 30) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.output_path = self.output_dir / f"classifier_{timestamp}.mp4"
        self.frames_dir = self.output_dir / f".frames_{timestamp}"
        self.fps = fps
        self.recording = False
        self.frame_count = 0
        self.last_message = "REC OFF"

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.recording = True
        self.frame_count = 0
        self.last_message = "REC ON"
        print(f"Grabacion iniciada: {self.output_path}")

    def stop(self) -> None:
        if not self.recording:
            return
        self.recording = False
        if self.frame_count == 0:
            self.last_message = "REC sin frames"
            return
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(self.output_path, fps=self.fps, codec="libx264", quality=8) as writer:
                for frame_path in sorted(self.frames_dir.glob("frame_*.png")):
                    writer.append_data(imageio.imread(frame_path))
            self.last_message = f"Guardado: {self.output_path}"
            print(self.last_message)
        except ImportError:
            self.last_message = "Falta imageio[ffmpeg]"
            print("Instala dependencias: python3 -m pip install -r requirements.txt")
        finally:
            shutil.rmtree(self.frames_dir, ignore_errors=True)

    def toggle(self) -> None:
        if self.recording:
            self.stop()
        else:
            self.start()

    def next_frame_path(self) -> str:
        path = self.frames_dir / f"frame_{self.frame_count:06d}.png"
        self.frame_count += 1
        return str(path)


def normalize_positions(positions: np.ndarray, sim: ClassifierSimulation) -> np.ndarray:
    g = sim.geometry
    shifted = positions.copy()
    shifted[:, 0] = (shifted[:, 0] + g.half_width_m) / g.width_m
    shifted[:, 1] = shifted[:, 1] / g.height_m
    shifted[:, 2] = (shifted[:, 2] + g.half_depth_m) / g.depth_m
    return shifted.astype(np.float32)


def project_oblique_side(
    positions: np.ndarray,
    sim: ClassifierSimulation,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> np.ndarray:
    g = sim.geometry
    radial_scale = max(g.cylinder_radius_m, 1e-6)
    x = np.clip(positions[:, 0] / radial_scale, -1.0, 1.0)
    z = np.clip(positions[:, 2] / radial_scale, -1.0, 1.0)
    y = np.clip(positions[:, 1] / g.height_m, 0.0, 1.0)

    screen = np.zeros((positions.shape[0], 2), dtype=np.float32)
    screen[:, 0] = left + width * np.clip(0.5 + 0.38 * z + 0.18 * x, 0.02, 0.98)
    screen[:, 1] = bottom + height * np.clip(y + 0.035 * x, 0.0, 1.0)
    return screen


def run_headless(sim: ClassifierSimulation, steps: int, substeps: int = 4) -> None:
    for _ in range(max(1, steps)):
        sim.step(substeps)


def _video_button_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    return (0.76, 0.93), (0.98, 0.98)


def _handle_video_button(gui: object, ti: object, recorder: VideoRecorder) -> None:
    top_left, bottom_right = _video_button_bounds()
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key != ti.GUI.LMB:
            continue
        x, y = gui.get_cursor_pos()
        if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
            recorder.toggle()


def _draw_video_button(gui: object, recorder: VideoRecorder) -> None:
    top_left, bottom_right = _video_button_bounds()
    color = 0xB82020 if recorder.recording else 0x255A36
    text = "DETENER Y GUARDAR MP4" if recorder.recording else "GRABAR MP4"
    gui.rect(top_left, bottom_right, radius=2, color=color)
    gui.text(text, pos=(top_left[0] + 0.012, top_left[1] + 0.012), font_size=15, color=0xFFFFFF)
    gui.text(recorder.last_message, pos=(0.76, 0.90), font_size=14, color=0xF0F6FF)


def _show_gui_frame(gui: object, recorder: VideoRecorder) -> None:
    if recorder.recording:
        gui.show(recorder.next_frame_path())
    else:
        gui.show()


def run_interactive(
    sim: ClassifierSimulation,
    max_frames: int = 2_000,
    substeps: int = 4,
    particle_radius: float = 0.006,
    arch_name: str = "auto",
    view_name: str = "front",
    display_name: str = "schematic",
) -> None:
    try:
        import taichi as ti
    except ImportError as exc:
        raise RuntimeError(
            "Taichi no esta instalado. Ejecuta: python3 -m pip install -r requirements.txt"
        ) from exc

    _init_taichi(ti, arch_name)
    if display_name == "schematic":
        _run_schematic_interactive(ti, sim, max_frames, substeps)
        return
    if display_name == "top":
        _run_top_interactive(ti, sim, max_frames, substeps)
        return
    if display_name != "3d":
        raise ValueError("--display debe ser 'schematic', 'top' o '3d'")

    _run_3d_interactive(ti, sim, max_frames, substeps, particle_radius, view_name)


def _run_3d_interactive(
    ti: object,
    sim: ClassifierSimulation,
    max_frames: int,
    substeps: int,
    particle_radius: float,
    view_name: str,
) -> None:
    window = ti.ui.Window("Clasificador hidraulico 3D", (1280, 800), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    _set_camera(camera, view_name)
    positions_field = ti.Vector.field(3, dtype=ti.f32, shape=sim.positions.shape[0])
    colors_field = ti.Vector.field(3, dtype=ti.f32, shape=sim.colors.shape[0])
    corners_field = ti.Vector.field(3, dtype=ti.f32, shape=8)
    corner_colors_field = ti.Vector.field(3, dtype=ti.f32, shape=8)
    guide_field = ti.Vector.field(3, dtype=ti.f32, shape=6)
    guide_colors_field = ti.Vector.field(3, dtype=ti.f32, shape=6)
    corners, corner_colors = _box_corners()
    guide_points, guide_colors = _orientation_guide()
    corners_field.from_numpy(corners)
    corner_colors_field.from_numpy(corner_colors)
    guide_field.from_numpy(guide_points)
    guide_colors_field.from_numpy(guide_colors)

    last_print = time.monotonic()
    frame = 0
    while window.running and frame < max_frames:
        camera.track_user_inputs(window, movement_speed=0.015, hold_key=ti.ui.RMB)
        sim.step(substeps)

        positions = normalize_positions(sim.positions, sim)
        positions_field.from_numpy(positions)
        colors_field.from_numpy(sim.colors)
        scene.set_camera(camera)
        scene.ambient_light((0.65, 0.65, 0.65))
        scene.point_light(pos=(0.5, 1.8, 1.4), color=(1.0, 1.0, 1.0))
        scene.particles(
            positions_field,
            per_vertex_color=colors_field,
            radius=particle_radius,
        )
        _draw_box(scene, corners_field, corner_colors_field)
        scene.particles(guide_field, per_vertex_color=guide_colors_field, radius=0.014)
        canvas.scene(scene)
        window.show()

        now = time.monotonic()
        if now - last_print > 2.0:
            metrics = compute_metrics(sim.status, sim.material_ids, sim.materials)
            print(f"frame={frame} {metrics.summary()}")
            last_print = now
        frame += 1


def _run_schematic_interactive(
    ti: object,
    sim: ClassifierSimulation,
    max_frames: int,
    substeps: int,
) -> None:
    gui = ti.GUI("Clasificador hidraulico - corte frontal", res=(1280, 800), background_color=0x101820)
    recorder = VideoRecorder()
    frame = 0
    last_print = time.monotonic()

    while gui.running and frame < max_frames:
        _handle_video_button(gui, ti, recorder)
        sim.step(substeps)
        _draw_schematic(gui, sim, frame)
        _draw_video_button(gui, recorder)
        _show_gui_frame(gui, recorder)

        now = time.monotonic()
        if now - last_print > 2.0:
            metrics = compute_metrics(sim.status, sim.material_ids, sim.materials)
            print(f"frame={frame} {metrics.summary()}")
            last_print = now
        frame += 1
    recorder.stop()


def _run_top_interactive(
    ti: object,
    sim: ClassifierSimulation,
    max_frames: int,
    substeps: int,
) -> None:
    gui = ti.GUI("Clasificador hidraulico - vista superior", res=(900, 900), background_color=0x101820)
    recorder = VideoRecorder()
    frame = 0
    last_print = time.monotonic()

    while gui.running and frame < max_frames:
        _handle_video_button(gui, ti, recorder)
        sim.step(substeps)
        _draw_top_view(gui, sim, frame)
        _draw_video_button(gui, recorder)
        _show_gui_frame(gui, recorder)

        now = time.monotonic()
        if now - last_print > 2.0:
            metrics = compute_metrics(sim.status, sim.material_ids, sim.materials)
            print(f"frame={frame} {metrics.summary()}")
            last_print = now
        frame += 1
    recorder.stop()


def _draw_top_view(gui: object, sim: ClassifierSimulation, frame: int) -> None:
    g = sim.geometry
    center = np.array([0.44, 0.50], dtype=np.float32)
    radius = 0.31

    gui.text("VISTA SUPERIOR: CICLON / CAMARA CIRCULAR", pos=(0.05, 0.94), font_size=24, color=0xF0F6FF)
    gui.text("Entrada tangencial", pos=(0.03, center[1] + radius + 0.03), font_size=16, color=0x9FE8FF)
    gui.text("Tubo central de rebalse", pos=(0.72, center[1] + 0.03), font_size=16, color=0x8AD7FF)
    gui.text("Trampa central", pos=(center[0] - 0.08, center[1] - 0.08), font_size=15, color=0xFFD15C)

    _draw_circle(gui, center, radius, 0xFFFFFF, point_count=96, line_radius=2)
    _draw_circle(
        gui,
        center,
        radius * (g.body_top_radius_m / g.cylinder_radius_m),
        0xD8E6F3,
        point_count=96,
        line_radius=1,
    )
    _draw_circle(
        gui,
        center,
        radius * (g.body_bottom_radius_m / g.cylinder_radius_m),
        0x607C92,
        point_count=80,
        line_radius=1,
    )
    _draw_circle(gui, center, radius * 0.38, 0x30485C, point_count=64, line_radius=1)
    _draw_circle(gui, center, radius * 0.15, 0xFFD15C, point_count=48, line_radius=2)
    _draw_circle(
        gui,
        center,
        radius * (g.overflow_tube_radius_m / g.cylinder_radius_m),
        0x8AD7FF,
        point_count=48,
        line_radius=3,
    )

    inlet_start = (center[0] - radius - 0.13, center[1] + radius * 0.70)
    inlet_end = (center[0] - radius * 0.52, center[1] + radius * 0.70)
    gui.line(inlet_start, inlet_end, radius=4, color=0x9FE8FF)
    gui.triangle(inlet_end, (inlet_end[0] - 0.032, inlet_end[1] + 0.018), (inlet_end[0] - 0.012, inlet_end[1] - 0.020), color=0x9FE8FF)
    outlet_start = (center[0] + radius * (g.overflow_tube_radius_m / g.cylinder_radius_m), center[1])
    outlet_end = (center[0] + radius + 0.13, center[1])
    gui.line(outlet_start, outlet_end, radius=4, color=0x8AD7FF)
    gui.triangle(outlet_end, (outlet_end[0] - 0.030, outlet_end[1] + 0.018), (outlet_end[0] - 0.030, outlet_end[1] - 0.018), color=0x8AD7FF)

    normalized = normalize_positions(sim.positions, sim)
    screen = np.zeros((sim.positions.shape[0], 2), dtype=np.float32)
    screen[:, 0] = center[0] + radius * (sim.positions[:, 2] / g.cylinder_radius_m)
    screen[:, 1] = center[1] + radius * (sim.positions[:, 0] / g.cylinder_radius_m)

    overflow = sim.status == 2
    screen[overflow, 0] = center[0] + radius * 0.05 * normalized[overflow, 2]
    screen[overflow, 1] = center[1] + radius * 0.05 * normalized[overflow, 0]

    active_or_done = sim.status != 2
    for material in sim.materials:
        mask = active_or_done & (sim.material_ids == material.id)
        if np.any(mask):
            gui.circles(screen[mask], radius=_point_radius(material.name), color=_rgb_to_hex(material.color_rgb))

    _draw_circle(gui, center, radius, 0xFFFFFF, point_count=96, line_radius=2)
    _draw_circle(
        gui,
        center,
        radius * (g.overflow_tube_radius_m / g.cylinder_radius_m),
        0x8AD7FF,
        point_count=48,
        line_radius=3,
    )
    metrics = compute_metrics(sim.status, sim.material_ids, sim.materials)
    gui.text(f"frame {frame}", pos=(0.80, 0.76), font_size=17, color=0xF0F6FF)
    gui.text(f"recuperacion oro: {metrics.target_recovery_pct:5.1f}%", pos=(0.80, 0.71), font_size=17, color=0xFFD15C)
    gui.text(f"perdida oro:      {metrics.target_loss_pct:5.1f}%", pos=(0.80, 0.67), font_size=17, color=0xFF6B6B)
    gui.text(f"contaminacion:    {metrics.trapped_contamination_pct:5.1f}%", pos=(0.80, 0.63), font_size=17, color=0xE8E8E8)

    legend_y = 0.48
    for idx, material in enumerate(sim.materials):
        y = legend_y - idx * 0.045
        gui.circles(np.array([[0.82, y]], dtype=np.float32), radius=6, color=_rgb_to_hex(material.color_rgb))
        gui.text(material.name, pos=(0.84, y - 0.01), font_size=15, color=0xF0F6FF)


def _draw_schematic(gui: object, sim: ClassifierSimulation, frame: int) -> None:
    g = sim.geometry
    left, right = 0.16, 0.70
    bottom, top = 0.08, 0.92
    width = right - left
    height = top - bottom

    gui.text("CORTE VERTICAL OBLICUO: VORTICE HELICOIDAL", pos=(0.05, 0.96), font_size=24, color=0xF0F6FF)
    gui.text("ARRIBA: rebalse / livianos", pos=(0.76, 0.88), font_size=18, color=0x8AD7FF)
    gui.text("ABAJO: trampa de pesados", pos=(0.76, 0.12), font_size=18, color=0xFFD15C)
    gui.text("Entrada agua + material", pos=(0.03, 0.37), font_size=16, color=0x9FE8FF)

    trap_y = bottom + height * (g.trap_height_m / g.height_m)
    cone_y = bottom + height * (g.cone_top_height_m / g.height_m)
    outlet_y = bottom + height * (g.outlet_height_m / g.height_m)
    inlet_y = bottom + height * (g.inlet_height_m / g.height_m)
    throat_left = _screen_x_from_z(-g.cone_neck_half_depth_m, g, left, width)
    throat_right = _screen_x_from_z(g.cone_neck_half_depth_m, g, left, width)

    _draw_classifier_walls(gui, g, left, right, bottom, top, trap_y, cone_y)
    _draw_flow_guides(gui, left, right, bottom, top, inlet_y, outlet_y)
    _draw_overflow_tube_side(gui, g, left, width, bottom, top)

    gui.line((throat_left, trap_y), (throat_right, trap_y), radius=5, color=0xFFD15C)
    gui.line((right, outlet_y), (right + 0.12, outlet_y), radius=4, color=0x8AD7FF)
    gui.line((left - 0.12, inlet_y), (left, inlet_y), radius=4, color=0x9FE8FF)
    gui.triangle((left - 0.01, inlet_y), (left - 0.035, inlet_y + 0.018), (left - 0.035, inlet_y - 0.018), color=0x9FE8FF)
    gui.triangle((right + 0.12, outlet_y), (right + 0.09, outlet_y + 0.018), (right + 0.09, outlet_y - 0.018), color=0x8AD7FF)
    gui.text("trampa", pos=(throat_right + 0.015, trap_y - 0.01), font_size=14, color=0xFFD15C)
    gui.text("rebalse", pos=(right + 0.025, outlet_y + 0.02), font_size=14, color=0x8AD7FF)

    for fraction in (0.25, 0.50, 0.75):
        y = bottom + height * fraction
        gui.line((left + 0.02, y), (right - 0.02, y), radius=1, color=0x30485C)

    normalized = normalize_positions(sim.positions, sim)
    screen = project_oblique_side(sim.positions, sim, left, bottom, width, height)

    overflow = sim.status == 2
    screen[overflow, 0] = 0.82 + 0.13 * normalized[overflow, 2]
    screen[overflow, 1] = outlet_y + 0.11 * (normalized[overflow, 1] - 0.5)

    pending_feed = sim.status == -1
    if np.any(pending_feed):
        pipe_start = -1.55 * g.cylinder_radius_m
        pipe_end = -0.97 * g.cylinder_radius_m
        pipe_progress = np.clip(
            (sim.positions[pending_feed, 2] - pipe_start) / max(1e-6, pipe_end - pipe_start),
            0.0,
            1.0,
        )
        screen[pending_feed, 0] = left - 0.12 + 0.11 * pipe_progress
        screen[pending_feed, 1] = bottom + height * np.clip(
            sim.positions[pending_feed, 1] / g.height_m,
            0.0,
            1.0,
        )

    active_or_done = sim.status != 2
    for material in sim.materials:
        mask = active_or_done & (sim.material_ids == material.id)
        if np.any(mask):
            gui.circles(screen[mask], radius=_point_radius(material.name), color=_rgb_to_hex(material.color_rgb))

    _draw_classifier_outline(gui, g, left, right, bottom, top, trap_y, cone_y)
    _draw_overflow_tube_side(gui, g, left, width, bottom, top)
    metrics = compute_metrics(sim.status, sim.material_ids, sim.materials)
    gui.text(f"frame {frame}", pos=(0.76, 0.78), font_size=17, color=0xF0F6FF)
    gui.text(f"recuperacion oro: {metrics.target_recovery_pct:5.1f}%", pos=(0.76, 0.73), font_size=17, color=0xFFD15C)
    gui.text(f"perdida oro:      {metrics.target_loss_pct:5.1f}%", pos=(0.76, 0.69), font_size=17, color=0xFF6B6B)
    gui.text(f"contaminacion:    {metrics.trapped_contamination_pct:5.1f}%", pos=(0.76, 0.65), font_size=17, color=0xE8E8E8)
    gui.text(f"arena negra trap: {metrics.trapped_black_sand_pct:5.1f}%", pos=(0.76, 0.61), font_size=17, color=0xA8A8A8)

    legend_y = 0.48
    for idx, material in enumerate(sim.materials):
        y = legend_y - idx * 0.045
        gui.circles(np.array([[0.78, y]], dtype=np.float32), radius=6, color=_rgb_to_hex(material.color_rgb))
        gui.text(material.name, pos=(0.80, y - 0.01), font_size=15, color=0xF0F6FF)


def _draw_classifier_walls(
    gui: object,
    g: object,
    left: float,
    right: float,
    bottom: float,
    top: float,
    trap_y: float,
    cone_y: float,
) -> None:
    mid = 0.5 * (left + right)
    top_left = _screen_x_from_z(-g.body_top_radius_m, g, left, right - left)
    top_right = _screen_x_from_z(g.body_top_radius_m, g, left, right - left)
    body_left = _screen_x_from_z(-g.body_bottom_radius_m, g, left, right - left)
    body_right = _screen_x_from_z(g.body_bottom_radius_m, g, left, right - left)
    neck_left = _screen_x_from_z(-g.cone_neck_half_depth_m, g, left, right - left)
    neck_right = _screen_x_from_z(g.cone_neck_half_depth_m, g, left, right - left)
    bottom_left = _screen_x_from_z(-g.trap_bottom_half_depth_m, g, left, right - left)
    bottom_right = _screen_x_from_z(g.trap_bottom_half_depth_m, g, left, right - left)
    tube_left = _screen_x_from_z(-g.overflow_tube_radius_m, g, left, right - left)
    tube_right = _screen_x_from_z(g.overflow_tube_radius_m, g, left, right - left)
    wall = 0xD8E6F3
    water = 0x18384F

    gui.triangle((neck_left, trap_y), (neck_right, trap_y), (mid, bottom), color=0x3A2F1F)
    gui.triangle((body_left, cone_y), (body_right, cone_y), (mid, trap_y), color=water)
    gui.triangle((top_left, top), (top_right, top), (body_left, cone_y), color=water)
    gui.triangle((top_right, top), (body_left, cone_y), (body_right, cone_y), color=water)
    gui.line((top_left, top), (tube_left, top), radius=2, color=wall)
    gui.line((tube_right, top), (top_right, top), radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, g.cone_top_height_m, g.height_m, -1.0, radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, g.cone_top_height_m, g.height_m, 1.0, radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, g.trap_height_m, g.cone_top_height_m, -1.0, radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, g.trap_height_m, g.cone_top_height_m, 1.0, radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, 0.0, g.trap_height_m, -1.0, radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, 0.0, g.trap_height_m, 1.0, radius=2, color=wall)
    _draw_trap_floor(gui, g, left, right - left, radius=2, color=wall)


def _draw_overflow_tube_side(
    gui: object,
    g: object,
    left: float,
    width: float,
    bottom: float,
    top: float,
) -> None:
    tube_left = _screen_x_from_z(-g.overflow_tube_radius_m, g, left, width)
    tube_right = _screen_x_from_z(g.overflow_tube_radius_m, g, left, width)
    tube_bottom = bottom + (top - bottom) * (g.overflow_tube_bottom_height_m / g.height_m)
    color = 0x8AD7FF
    gui.line((tube_left, top + 0.08), (tube_left, top), radius=3, color=color)
    gui.line((tube_right, top + 0.08), (tube_right, top), radius=3, color=color)
    gui.line((tube_left, top), (tube_left, tube_bottom), radius=3, color=color)
    gui.line((tube_right, top), (tube_right, tube_bottom), radius=3, color=color)
    tube_mid = 0.5 * (tube_left + tube_right)
    gui.line((tube_mid, tube_bottom - 0.045), (tube_mid, tube_bottom + 0.045), radius=2, color=color)
    gui.triangle(
        (tube_mid, tube_bottom + 0.052),
        (tube_mid - 0.010, tube_bottom + 0.030),
        (tube_mid + 0.010, tube_bottom + 0.030),
        color=color,
    )
    gui.text("tubo abierto", pos=(tube_right + 0.012, tube_bottom + 0.015), font_size=13, color=color)


def _draw_classifier_outline(
    gui: object,
    g: object,
    left: float,
    right: float,
    bottom: float,
    top: float,
    trap_y: float,
    cone_y: float,
) -> None:
    top_left = _screen_x_from_z(-g.body_top_radius_m, g, left, right - left)
    top_right = _screen_x_from_z(g.body_top_radius_m, g, left, right - left)
    body_left = _screen_x_from_z(-g.body_bottom_radius_m, g, left, right - left)
    body_right = _screen_x_from_z(g.body_bottom_radius_m, g, left, right - left)
    neck_left = _screen_x_from_z(-g.cone_neck_half_depth_m, g, left, right - left)
    neck_right = _screen_x_from_z(g.cone_neck_half_depth_m, g, left, right - left)
    bottom_left = _screen_x_from_z(-g.trap_bottom_half_depth_m, g, left, right - left)
    bottom_right = _screen_x_from_z(g.trap_bottom_half_depth_m, g, left, right - left)
    tube_left = _screen_x_from_z(-g.overflow_tube_radius_m, g, left, right - left)
    tube_right = _screen_x_from_z(g.overflow_tube_radius_m, g, left, right - left)
    wall = 0xFFFFFF

    gui.line((top_left, top), (tube_left, top), radius=2, color=wall)
    gui.line((tube_right, top), (top_right, top), radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, g.cone_top_height_m, g.height_m, -1.0, radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, g.cone_top_height_m, g.height_m, 1.0, radius=2, color=wall)
    _draw_radius_wall(gui, g, left, right - left, g.trap_height_m, g.cone_top_height_m, -1.0, radius=3, color=wall)
    _draw_radius_wall(gui, g, left, right - left, g.trap_height_m, g.cone_top_height_m, 1.0, radius=3, color=wall)
    _draw_radius_wall(gui, g, left, right - left, 0.0, g.trap_height_m, -1.0, radius=3, color=wall)
    _draw_radius_wall(gui, g, left, right - left, 0.0, g.trap_height_m, 1.0, radius=3, color=wall)
    _draw_trap_floor(gui, g, left, right - left, radius=3, color=wall)


def _screen_x_from_z(z_m: float, g: object, left: float, width: float) -> float:
    return left + width * ((z_m + g.half_depth_m) / g.depth_m)


def _screen_y_from_height(height_m: float, g: object, bottom: float = 0.08, top: float = 0.92) -> float:
    return bottom + (top - bottom) * (height_m / g.height_m)


def _draw_radius_wall(
    gui: object,
    g: object,
    left: float,
    width: float,
    y0_m: float,
    y1_m: float,
    side: float,
    radius: int,
    color: int,
) -> None:
    ys = np.linspace(y0_m, y1_m, 18)
    points = [
        (
            _screen_x_from_z(side * g.allowed_radius_at_height(float(y_m)), g, left, width),
            _screen_y_from_height(float(y_m), g),
        )
        for y_m in ys
    ]
    for start, end in zip(points, points[1:]):
        gui.line(start, end, radius=radius, color=color)


def _draw_trap_floor(
    gui: object,
    g: object,
    left: float,
    width: float,
    radius: int,
    color: int,
) -> None:
    xs = np.linspace(-g.trap_bottom_half_depth_m, g.trap_bottom_half_depth_m, 25)
    points = [
        (
            _screen_x_from_z(float(x_m), g, left, width),
            _screen_y_from_height(g.trap_floor_height_at_radius(abs(float(x_m))), g),
        )
        for x_m in xs
    ]
    for start, end in zip(points, points[1:]):
        gui.line(start, end, radius=radius, color=color)


def _draw_flow_guides(
    gui: object,
    left: float,
    right: float,
    bottom: float,
    top: float,
    inlet_y: float,
    outlet_y: float,
) -> None:
    mid = 0.5 * (left + right)
    for offset in (0.0, 0.07, 0.14):
        x0 = left + 0.08 + offset
        x1 = x0 + 0.08
        y = inlet_y + 0.03 + offset * 0.8
        gui.line((x0, y), (x1, y + 0.04), radius=1, color=0x4A9FC7)
        gui.triangle((x1, y + 0.04), (x1 - 0.018, y + 0.043), (x1 - 0.009, y + 0.025), color=0x4A9FC7)
    gui.line((mid, bottom + 0.12), (mid, top - 0.10), radius=1, color=0x4A9FC7)
    gui.triangle((mid, top - 0.08), (mid - 0.012, top - 0.11), (mid + 0.012, top - 0.11), color=0x4A9FC7)
    gui.line((right - 0.08, outlet_y), (right + 0.02, outlet_y), radius=1, color=0x4A9FC7)


def _draw_rect(gui: object, left: float, bottom: float, right: float, top: float, color: int, radius: int) -> None:
    gui.line((left, bottom), (right, bottom), radius=radius, color=color)
    gui.line((right, bottom), (right, top), radius=radius, color=color)
    gui.line((right, top), (left, top), radius=radius, color=color)
    gui.line((left, top), (left, bottom), radius=radius, color=color)


def _draw_circle(
    gui: object,
    center: np.ndarray,
    radius: float,
    color: int,
    point_count: int,
    line_radius: int,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, point_count + 1)
    points = np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        )
    ).astype(np.float32)
    for idx in range(point_count):
        gui.line(points[idx], points[idx + 1], radius=line_radius, color=color)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> int:
    red = int(np.clip(rgb[0], 0.0, 1.0) * 255)
    green = int(np.clip(rgb[1], 0.0, 1.0) * 255)
    blue = int(np.clip(rgb[2], 0.0, 1.0) * 255)
    return (red << 16) + (green << 8) + blue


def _point_radius(material_name: str) -> float:
    if "oro" in material_name:
        return 4.0
    if "magnetita" in material_name:
        return 3.0
    if "limo" in material_name:
        return 2.0
    return 2.5


def _set_camera(camera: object, view_name: str) -> None:
    if view_name == "front":
        # Vista tipo elevacion: Y fisico queda vertical en pantalla.
        camera.position(0.5, 0.5, 2.25)
        camera.lookat(0.5, 0.5, 0.5)
        camera.fov(38)
        return
    if view_name == "isometric":
        camera.position(1.25, 0.85, 1.65)
        camera.lookat(0.5, 0.42, 0.5)
        camera.fov(45)
        return
    if view_name == "side":
        camera.position(2.2, 0.5, 0.5)
        camera.lookat(0.5, 0.5, 0.5)
        camera.fov(38)
        return
    valid = "front, isometric, side"
    raise ValueError(f"--view debe ser uno de: {valid}")


def _init_taichi(ti: object, arch_name: str) -> None:
    archs = {
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "opengl": ti.opengl,
        "cpu": ti.cpu,
    }
    if arch_name != "auto":
        if arch_name not in archs:
            valid = ", ".join(("auto", *archs.keys()))
            raise ValueError(f"--arch debe ser uno de: {valid}")
        ti.init(arch=archs[arch_name])
        print(f"Taichi backend: {arch_name}")
        return

    errors = []
    for candidate in ("vulkan", "opengl", "cpu"):
        try:
            ti.init(arch=archs[candidate])
            print(f"Taichi backend: {candidate}")
            return
        except Exception as exc:  # Taichi levanta RuntimeError distinto segun backend.
            errors.append(f"{candidate}: {exc}")

    details = "\n".join(errors)
    raise RuntimeError(f"No se pudo inicializar Taichi con vulkan/opengl/cpu:\n{details}")


def _box_corners() -> tuple[np.ndarray, np.ndarray]:
    # Taichi GGUI no dibuja lineas 3D portables en todas las versiones; las esferas
    # pequenas marcan esquinas para mantener una referencia espacial simple.
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    colors = np.full((8, 3), 0.35, dtype=np.float32)
    return corners, colors


def _orientation_guide() -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [
            [0.5, 0.02, 0.5],
            [0.5, 0.20, 0.5],
            [0.5, 0.40, 0.5],
            [0.5, 0.60, 0.5],
            [0.5, 0.80, 0.5],
            [0.5, 0.98, 0.5],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [0.20, 0.20, 0.20],
            [0.30, 0.30, 0.30],
            [0.45, 0.45, 0.45],
            [0.25, 0.55, 0.90],
            [0.15, 0.75, 0.95],
            [0.05, 0.95, 1.00],
        ],
        dtype=np.float32,
    )
    return points, colors


def _draw_box(scene: object, corners_field: object, corner_colors_field: object) -> None:
    scene.particles(corners_field, per_vertex_color=corner_colors_field, radius=0.01)

