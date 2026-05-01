from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_sim import ClassifierGeometry, ClassifierSimulation, SimulationConfig
from visual_sim.metrics import compute_metrics
from visual_sim.viewer import run_headless, run_interactive


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PRESETS = {
    "baseline": None,
    "rl-best-f3000": PROJECT_ROOT / "rl_runs" / "sac_geometry_balanced_f3000" / "best_geometry.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulador visual 3D de clasificador hidraulico para separar oro/sedimentos."
    )
    parser.add_argument("--headless", action="store_true", help="Corre sin ventana 3D.")
    parser.add_argument("--particles", type=int, default=8_000, help="Cantidad de particulas.")
    parser.add_argument("--frames", type=int, default=1_500, help="Frames de simulacion.")
    parser.add_argument("--substeps", type=int, default=4, help="Pasos fisicos por frame.")
    parser.add_argument("--feed-duration", type=float, default=5.0, help="Segundos para dosificar toda la alimentacion.")
    parser.add_argument("--width", type=float, default=0.18, help="Ancho de camara en metros.")
    parser.add_argument("--depth", type=float, default=0.12, help="Profundidad de camara en metros.")
    parser.add_argument("--height", type=float, default=0.35, help="Altura de camara en metros.")
    parser.add_argument(
        "--geometry-preset",
        choices=tuple(GEOMETRY_PRESETS),
        default="baseline",
        help="Geometria base o mejor geometria RL conocida.",
    )
    parser.add_argument("--geometry-json", help="Carga una geometria guardada por entrenamiento/evaluacion RL.")
    parser.add_argument(
        "--upflow",
        type=float,
        default=0.065,
        help="Intensidad vertical del flujo en m/s: baja por fuera y sube por el tubo.",
    )
    parser.add_argument(
        "--inlet",
        type=float,
        default=0.22,
        help="Velocidad lateral de entrada en m/s.",
    )
    parser.add_argument(
        "--inlet-height",
        type=float,
        default=0.24,
        help="Altura de entrada de agua/material en metros.",
    )
    parser.add_argument(
        "--turbulence",
        type=float,
        default=0.018,
        help="Ruido de turbulencia simplificada.",
    )
    parser.add_argument(
        "--arch",
        choices=("auto", "vulkan", "opengl", "cpu", "cuda"),
        default="auto",
        help="Backend de Taichi para la ventana 3D. Auto evita CUDA por defecto.",
    )
    parser.add_argument(
        "--view",
        choices=("front", "isometric", "side"),
        default="front",
        help="Vista inicial para --display 3d.",
    )
    parser.add_argument(
        "--display",
        choices=("schematic", "top", "3d"),
        default="schematic",
        help="Schematic muestra corte lateral; top muestra vista superior; 3d muestra nube espacial.",
    )
    return parser.parse_args()


def build_simulation(args: argparse.Namespace) -> ClassifierSimulation:
    preset_path = GEOMETRY_PRESETS[args.geometry_preset]
    geometry_path = Path(args.geometry_json) if args.geometry_json else preset_path
    if geometry_path:
        payload = json.loads(geometry_path.read_text(encoding="utf-8"))
        geometry = ClassifierGeometry(**payload["geometry"])
    else:
        geometry = ClassifierGeometry(
            width_m=args.width,
            depth_m=args.depth,
            height_m=args.height,
            inlet_height_m=args.inlet_height,
            inlet_velocity_m_s=args.inlet,
            upward_velocity_m_s=args.upflow,
            turbulence=args.turbulence,
        )
    config = SimulationConfig(particle_count=args.particles, feed_duration_s=args.feed_duration)
    return ClassifierSimulation(geometry=geometry, config=config)


def main() -> None:
    args = parse_args()
    sim = build_simulation(args)

    if args.headless:
        run_headless(sim, steps=args.frames, substeps=args.substeps)
    else:
        run_interactive(
            sim,
            max_frames=args.frames,
            substeps=args.substeps,
            arch_name=args.arch,
            view_name=args.view,
            display_name=args.display,
        )

    metrics = compute_metrics(sim.status, sim.material_ids, sim.materials)
    print(metrics.summary())


if __name__ == "__main__":
    main()

