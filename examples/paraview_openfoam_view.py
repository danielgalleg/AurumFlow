#!/usr/bin/env python3
"""
Vista OpenFOAM en ParaView por script: lector .foam, contexto geometrico,
Glyph (flechas U) y StreamTracer (lineas de corriente).

Ubuntu/Debian: el binario pvpython a veces no incluye site-packages; se añaden rutas tipicas.

  # Captura sin GUI (si hay display, puede abrir ventana breve segun version)
  PYTHONPATH=/usr/lib/python3/dist-packages \\
    pvpython examples/paraview_openfoam_view.py \\
      --foam cfd_cases/ga_curved_f3000_simple/ga_curved_f3000_simple.foam \\
      --screenshot salida.png

  # Interactivo (ventana ParaView hasta cerrar)
  PYTHONPATH=/usr/lib/python3/dist-packages \\
    pvpython examples/paraview_openfoam_view.py --foam .../case.foam --interactive
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _ensure_paraview_import_path() -> None:
    major = sys.version_info.major
    minor = sys.version_info.minor
    candidates = (
        str(Path.home() / f".local/lib/python{major}.{minor}/site-packages"),
        "/usr/lib/python3/dist-packages",
        "/usr/lib/python3.10/dist-packages",
        "/usr/lib/python3.11/dist-packages",
        "/usr/lib/python3.12/dist-packages",
    )
    for p in candidates:
        if Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)


_ensure_paraview_import_path()

from paraview.simple import (  # noqa: E402
    CellDatatoPointData,
    ColorBy,
    CSVReader,
    ExtractSurface,
    GetActiveViewOrCreate,
    GetAnimationScene,
    Glyph,
    OpenFOAMReader,
    Outline,
    Render,
    ResetCamera,
    SaveScreenshot,
    SetActiveSource,
    Show,
    Slice,
    STLReader,
    StreamTracer,
    TableToPoints,
)


def _pick_time(reader: object, time_value: float | None) -> float:
    reader.UpdatePipelineInformation()
    times = list(reader.TimestepValues)
    if not times:
        return 0.0
    if time_value is None:
        return float(times[-1])
    available = min(times, key=lambda t: abs(float(t) - float(time_value)))
    return float(available)


def _stream_ends_from_bounds(bounds: tuple[float, ...]) -> tuple[list[float], list[float]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y_mid = 0.5 * (ymin + ymax)
    z_mid = 0.5 * (zmin + zmax)
    span_x = xmax - xmin
    inset = max(1e-6, 0.12 * span_x)
    p1 = [xmax - inset, y_mid, z_mid]
    p2 = [xmin + inset, y_mid, z_mid]
    return p1, p2


def _default_stl_path(foam: Path) -> Path | None:
    tri_surface = foam.parent / "constant" / "triSurface"
    if not tri_surface.is_dir():
        return None
    matches = sorted(tri_surface.glob("*_internal_volume.stl"))
    if not matches:
        matches = sorted(tri_surface.glob("*.stl"))
    return matches[0] if matches else None


def _set_display(display: object, **kwargs: object) -> None:
    for name, value in kwargs.items():
        try:
            setattr(display, name, value)
        except AttributeError:
            pass


def _bounds_center_and_span(bounds: tuple[float, ...]) -> tuple[list[float], float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]
    span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-6)
    return center, span


def _apply_camera(view: object, bounds: tuple[float, ...], preset: str, azimuth_rad: float = math.radians(45.0)) -> None:
    center, span = _bounds_center_and_span(bounds)
    distance = 2.6 * span
    if preset == "front":
        position = [center[0], center[1], center[2] + distance]
        view_up = [0.0, 1.0, 0.0]
    elif preset == "side":
        position = [center[0] + distance, center[1], center[2]]
        view_up = [0.0, 1.0, 0.0]
    elif preset == "top":
        position = [center[0], center[1] + distance, center[2]]
        view_up = [0.0, 0.0, -1.0]
    else:
        position = [
            center[0] + distance * math.cos(azimuth_rad),
            center[1] + 0.42 * distance,
            center[2] + distance * math.sin(azimuth_rad),
        ]
        view_up = [0.0, 1.0, 0.0]
    _set_display(view, CameraFocalPoint=center, CameraPosition=position, CameraViewUp=view_up)


def _write_orbit_animation(
    view: object,
    bounds: tuple[float, ...],
    frames: int,
    resolution: list[int],
    orbit_dir: Path | None,
    orbit_mp4: Path | None,
    fps: int,
) -> None:
    _set_display(view, ViewSize=[int(resolution[0]), int(resolution[1])])
    frame_dir = orbit_dir or (orbit_mp4.with_suffix("").resolve() if orbit_mp4 else Path("paraview_orbit_frames").resolve())
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    total = max(1, int(frames))
    for i in range(total):
        angle = 2.0 * math.pi * i / total
        _apply_camera(view, bounds, "iso", angle)
        Render(view)
        frame_path = frame_dir / f"orbit_{i:04d}.png"
        _set_display(view, ViewSize=[int(resolution[0]), int(resolution[1])])
        SaveScreenshot(str(frame_path), view, ImageResolution=resolution)
        frame_paths.append(frame_path)
    print(f"Frames de orbita: {frame_dir} ({total} PNG)")

    if orbit_mp4:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise SystemExit(
                "No pude importar imageio para crear MP4. "
                "Instala/activa imageio o usa --orbit-dir para dejar solo PNG."
            ) from exc
        def normalized_frame(path: Path, target_shape: tuple[int, int, int] | None) -> tuple[object, tuple[int, int, int]]:
            image = imageio.imread(path)
            if image.ndim == 2:
                image = image[:, :, None]
            if image.shape[2] == 4:
                image = image[:, :, :3]
            if target_shape is None:
                return image, image.shape
            if image.shape == target_shape:
                return image, target_shape
            import numpy as np

            target_h, target_w, target_c = target_shape
            canvas = np.zeros(target_shape, dtype=image.dtype)
            copy_h = min(target_h, image.shape[0])
            copy_w = min(target_w, image.shape[1])
            copy_c = min(target_c, image.shape[2])
            src_y = max(0, (image.shape[0] - copy_h) // 2)
            src_x = max(0, (image.shape[1] - copy_w) // 2)
            dst_y = max(0, (target_h - copy_h) // 2)
            dst_x = max(0, (target_w - copy_w) // 2)
            canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w, :copy_c] = image[
                src_y : src_y + copy_h, src_x : src_x + copy_w, :copy_c
            ]
            return canvas, target_shape

        with imageio.get_writer(
            str(orbit_mp4.resolve()),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        ) as writer:
            target_shape = None
            for frame_path in frame_paths:
                frame, target_shape = normalized_frame(frame_path, target_shape)
                writer.append_data(frame)
        print(f"Video de orbita: {orbit_mp4.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ParaView: Glyph + StreamTracer para caso OpenFOAM (.foam).")
    p.add_argument("--foam", type=Path, required=True, help="Ruta al archivo .foam dentro del caso.")
    p.add_argument(
        "--time",
        type=float,
        default=None,
        help="Instante mas cercano a este valor (por defecto el ultimo escrito).",
    )
    p.add_argument("--no-glyph", action="store_true", help="No añadir filtro Glyph.")
    p.add_argument("--no-stream", action="store_true", help="No añadir StreamTracer.")
    p.add_argument("--no-context", action="store_true", help="No mostrar STL, superficie, corte ni contorno.")
    p.add_argument("--stl", type=Path, default=None, help="STL de referencia; por defecto constant/triSurface/*.stl.")
    p.add_argument(
        "--slice",
        choices=("vertical", "horizontal", "none"),
        default=None,
        help="Plano de corte coloreado por U. Por defecto: vertical en PNG, none en orbita.",
    )
    p.add_argument("--glyph-scale", type=float, default=0.0015, help="ScaleFactor del Glyph.")
    p.add_argument(
        "--glyph-max-points",
        type=int,
        default=6000,
        help="MaximumNumberOfSamplePoints del Glyph (menos = mas rapido).",
    )
    p.add_argument("--stream-resolution", type=int, default=24, help="Puntos en la semilla tipo Line.")
    p.add_argument(
        "--stream-length",
        type=float,
        default=0.6,
        help="MaximumStreamlineLength (metros en el caso).",
    )
    p.add_argument("--screenshot", type=Path, default=None, help="Guardar PNG y salir (si no --interactive).")
    p.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[1280, 720],
        help="Resolucion del screenshot.",
    )
    p.add_argument(
        "--camera",
        choices=("iso", "front", "side", "top"),
        default="iso",
        help="Preset de camara para PNG/ventana interactiva.",
    )
    p.add_argument("--orbit-dir", type=Path, default=None, help="Guardar una secuencia PNG rotando la camara.")
    p.add_argument("--orbit-mp4", type=Path, default=None, help="Crear un MP4 con una orbita 3D.")
    p.add_argument("--orbit-frames", type=int, default=120, help="Cantidad de frames de la orbita.")
    p.add_argument("--orbit-fps", type=int, default=24, help="FPS del MP4 de orbita.")
    p.add_argument("--particles-csv", type=Path, default=None, help="CSV generado por analyze_openfoam_particles.py.")
    p.add_argument("--particle-size", type=float, default=7.0, help="Tamano visual de puntos de particulas.")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Mantener ventana abierta (Interact) hasta cerrarla.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    foam = args.foam.resolve()
    if not foam.is_file():
        raise SystemExit(f"No existe el archivo .foam: {foam}")

    reader = OpenFOAMReader(FileName=str(foam))
    t = _pick_time(reader, args.time)
    GetAnimationScene().AnimationTime = t
    reader.UpdatePipeline()
    bounds = tuple(reader.GetDataInformation().GetBounds())

    c2p = CellDatatoPointData(Input=reader)
    view = GetActiveViewOrCreate("RenderView")
    _set_display(view, OrientationAxesVisibility=1, Background=[0.05, 0.06, 0.07])
    slice_mode = args.slice
    if slice_mode is None:
        slice_mode = "none" if (args.orbit_dir or args.orbit_mp4) else "vertical"

    last = c2p
    if not args.no_context:
        outline = Outline(Input=c2p)
        outline_display = Show(outline, view)
        _set_display(outline_display, DiffuseColor=[0.0, 0.0, 0.0], LineWidth=2.0)

        surface = ExtractSurface(Input=c2p)
        surface_display = Show(surface, view)
        _set_display(
            surface_display,
            Representation="Wireframe",
            DiffuseColor=[0.72, 0.72, 0.72],
            Opacity=0.22,
            LineWidth=0.5,
        )

        stl_path = args.stl.resolve() if args.stl else _default_stl_path(foam)
        if stl_path and stl_path.is_file():
            stl = STLReader(FileNames=[str(stl_path)])
            stl_display = Show(stl, view)
            _set_display(
                stl_display,
                Representation="Surface With Edges",
                DiffuseColor=[0.82, 0.82, 0.78],
                Opacity=0.18,
                LineWidth=0.6,
            )
            try:
                ColorBy(stl_display, None)
            except Exception:
                pass

        if slice_mode != "none":
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            cut = Slice(Input=c2p)
            cut.SliceType = "Plane"
            if slice_mode == "vertical":
                cut.SliceType.Normal = [0.0, 0.0, 1.0]
                cut.SliceType.Origin = [0.0, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]
            else:
                cut.SliceType.Normal = [0.0, 1.0, 0.0]
                cut.SliceType.Origin = [0.0, 0.5 * (ymin + ymax), 0.0]
            cut_display = Show(cut, view)
            _set_display(cut_display, Opacity=0.48)
            try:
                ColorBy(cut_display, ("POINTS", "U", "Magnitude"))
            except Exception:
                try:
                    ColorBy(cut_display, ("POINTS", "U"))
                except Exception:
                    pass

    if args.no_glyph and args.no_stream and args.no_context:
        Show(c2p, view)
    if not args.no_glyph:
        g = Glyph(Input=c2p, GlyphType="Arrow")
        g.OrientationArray = ["POINTS", "U"]
        g.ScaleArray = ["POINTS", "U"]
        g.VectorScaleMode = "Scale by Magnitude"
        g.ScaleFactor = float(args.glyph_scale)
        g.MaximumNumberOfSamplePoints = int(args.glyph_max_points)
        glyph_display = Show(g, view)
        _set_display(glyph_display, DiffuseColor=[1.0, 0.72, 0.1], Opacity=0.85)
        try:
            ColorBy(glyph_display, ("POINTS", "U", "Magnitude"))
        except Exception:
            pass
        last = g

    if not args.no_stream:
        bounds = reader.GetDataInformation().GetBounds()
        p1, p2 = _stream_ends_from_bounds(tuple(bounds))
        st = StreamTracer(Input=c2p, Vectors=["POINTS", "U"])
        st.SeedType = "Line"
        st.SeedType.Point1 = p1
        st.SeedType.Point2 = p2
        st.SeedType.Resolution = int(args.stream_resolution)
        st.MaximumStreamlineLength = float(args.stream_length)
        stream_display = Show(st, view)
        _set_display(stream_display, DiffuseColor=[0.0, 0.75, 1.0], LineWidth=3.0)
        try:
            ColorBy(stream_display, None)
        except Exception:
            pass
        last = st

    if args.particles_csv:
        csv_path = args.particles_csv.resolve()
        if not csv_path.is_file():
            raise SystemExit(f"No existe el CSV de particulas: {csv_path}")
        particle_table = CSVReader(FileName=[str(csv_path)])
        particle_points = TableToPoints(Input=particle_table)
        particle_points.XColumn = "x"
        particle_points.YColumn = "y"
        particle_points.ZColumn = "z"
        particle_display = Show(particle_points, view)
        _set_display(
            particle_display,
            Representation="Point Gaussian",
            PointSize=float(args.particle_size),
            Opacity=0.95,
        )
        try:
            ColorBy(particle_display, ("POINTS", "density_kg_m3"))
        except Exception:
            try:
                ColorBy(particle_display, ("POINTS", "target"))
            except Exception:
                pass
        last = particle_points

    SetActiveSource(last)
    ResetCamera(view)
    _apply_camera(view, bounds, args.camera)
    Render(view)

    if args.orbit_dir or args.orbit_mp4:
        _write_orbit_animation(
            view=view,
            bounds=bounds,
            frames=args.orbit_frames,
            resolution=args.resolution,
            orbit_dir=args.orbit_dir,
            orbit_mp4=args.orbit_mp4,
            fps=args.orbit_fps,
        )

    if args.screenshot:
        SaveScreenshot(str(args.screenshot.resolve()), view, ImageResolution=args.resolution)
        print(f"Captura: {args.screenshot.resolve()} (tiempo {t})")
    elif not args.interactive:
        print(f"Pipeline lista (tiempo {t}). Use --screenshot o --interactive.")

    if args.interactive:
        try:
            from paraview.simple import Interact  # type: ignore[attr-defined]
        except ImportError:
            Interact = None  # type: ignore[misc, assignment]
        if Interact is not None:
            Interact()
        else:
            print("Este ParaView no expone Interact(); abre el caso .foam desde la GUI.", file=sys.stderr)


if __name__ == "__main__":
    main()
