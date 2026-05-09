from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_sim import ClassifierGeometry

Point = tuple[float, float, float]
Triangle = tuple[Point, Point, Point]
TriangleGroups = dict[str, list[Triangle]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta geometria axisimetrica del clasificador 'Clepsamia' (reloj de arena)."
    )
    parser.add_argument("--geometry-json", required=True, help="JSON con campo 'geometry'.")
    parser.add_argument("--output-dir", required=True, help="Directorio de exportacion.")
    parser.add_argument("--axial-samples", type=int, default=200, help="Muestras verticales del perfil.")
    parser.add_argument("--angular-segments", type=int, default=96, help="Segmentos de revolucion para STL.")
    parser.add_argument("--name", default="classifier", help="Prefijo de archivos exportados.")
    parser.add_argument(
        "--inlet-angular-width-deg",
        type=float,
        default=28.0,
        help="Ancho angular del patch de entrada en la pared.",
    )
    parser.add_argument(
        "--inlet-patch-height-m",
        type=float,
        default=0.025,
        help="Alto vertical del patch de entrada.",
    )
    return parser.parse_args()


def load_geometry(path: Path) -> ClassifierGeometry:
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometry_data = payload["geometry"] if "geometry" in payload else payload
    geometry = ClassifierGeometry(**geometry_data)
    geometry.validate()
    return geometry


def zone_for_height(y_m: float, geometry: ClassifierGeometry) -> str:
    if y_m < geometry.lower_max_height_m:
        return "lower_bottom"
    if y_m < geometry.neck_height_m:
        return "lower_neck"
    if y_m < geometry.upper_max_height_m:
        return "upper_neck"
    return "upper_top"


def profile_rows(geometry: ClassifierGeometry, axial_samples: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for idx in range(max(2, axial_samples)):
        y_m = geometry.height_m * idx / max(1, axial_samples - 1)
        radius_m = geometry.allowed_radius_at_height(y_m)
        rows.append(
            {
                "index": idx,
                "height_m": y_m,
                "outer_radius_m": radius_m,
                "diameter_m": 2.0 * radius_m,
                "zone": zone_for_height(y_m, geometry),
            }
        )
    return rows


def reference_rows(geometry: ClassifierGeometry) -> list[dict[str, float | str]]:
    return [
        {
            "name": "neck",
            "height_m": geometry.neck_height_m,
            "radius_m": geometry.neck_radius_m,
            "note": "Garganta entre los dos lobulos",
        },
        {
            "name": "upper_max",
            "height_m": geometry.upper_max_height_m,
            "radius_m": geometry.upper_max_radius_m,
            "note": "Punto mas ancho del lobulo superior",
        },
        {
            "name": "lower_max",
            "height_m": geometry.lower_max_height_m,
            "radius_m": geometry.lower_max_radius_m,
            "note": "Punto mas ancho del lobulo inferior",
        },
        {
            "name": "inlet_center",
            "height_m": geometry.inlet_height_m,
            "radius_m": geometry.allowed_radius_at_height(geometry.inlet_height_m),
            "note": "Centro del patch tangencial de entrada",
        },
        {
            "name": "central_tube_top",
            "height_m": geometry.height_m,
            "radius_m": geometry.central_tube_radius_m,
            "note": "Salida superior del tubo central (outlet pasivo)",
        },
        {
            "name": "central_tube_bottom",
            "height_m": geometry.central_tube_bottom_height_m,
            "radius_m": geometry.central_tube_radius_m,
            "note": "Boca inferior del tubo central, donde el fluido entra a el",
        },
    ]


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ring_points(radius_m: float, height_m: float, angular_segments: int) -> list[Point]:
    return [
        (
            radius_m * math.cos(2.0 * math.pi * idx / angular_segments),
            height_m,
            radius_m * math.sin(2.0 * math.pi * idx / angular_segments),
        )
        for idx in range(angular_segments)
    ]


def triangle_normal(a: Point, b: Point, c: Point) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (nx / length, ny / length, nz / length)


def add_triangle(triangles: list[Triangle], a: Point, b: Point, c: Point) -> None:
    if a == b or b == c or a == c:
        return
    triangles.append((a, b, c))


def add_ring_surface(triangles: list[Triangle], lower: list[Point], upper: list[Point]) -> None:
    count = len(lower)
    for idx in range(count):
        nxt = (idx + 1) % count
        add_triangle(triangles, lower[idx], upper[idx], upper[nxt])
        add_triangle(triangles, lower[idx], upper[nxt], lower[nxt])


def angle_difference(a_rad: float, b_rad: float) -> float:
    return math.atan2(math.sin(a_rad - b_rad), math.cos(a_rad - b_rad))


def add_outer_wall_surfaces(
    groups: TriangleGroups,
    rings: list[list[Point]],
    y_values: list[float],
    inlet_height_m: float,
    inlet_angle_deg: float,
    inlet_angular_width_deg: float,
    inlet_patch_height_m: float,
) -> None:
    angular_segments = len(rings[0])
    inlet_angle_rad = math.radians(inlet_angle_deg)
    half_angle_rad = math.radians(inlet_angular_width_deg) * 0.5
    half_height_m = inlet_patch_height_m * 0.5
    for ring_idx, (lower, upper) in enumerate(zip(rings, rings[1:])):
        y_mid = 0.5 * (y_values[ring_idx] + y_values[ring_idx + 1])
        for idx in range(angular_segments):
            nxt = (idx + 1) % angular_segments
            theta_mid = 2.0 * math.pi * (idx + 0.5) / angular_segments
            in_inlet = (
                abs(y_mid - inlet_height_m) <= half_height_m
                and abs(angle_difference(theta_mid, inlet_angle_rad)) <= half_angle_rad
            )
            target = groups["inlet"] if in_inlet else groups["outer_wall"]
            add_triangle(target, lower[idx], upper[idx], upper[nxt])
            add_triangle(target, lower[idx], upper[nxt], lower[nxt])


def build_stl_triangles(
    geometry: ClassifierGeometry,
    axial_samples: int,
    angular_segments: int,
    inlet_angular_width_deg: float,
    inlet_patch_height_m: float,
) -> TriangleGroups:
    groups: TriangleGroups = {
        "outer_wall": [],
        "inlet": [],
        "central_tube_wall": [],
        "central_tube_top": [],
    }

    # Pared exterior: contorno de revolucion del cuerpo. Va desde y=0 (fondo del
    # lobulo inferior) hasta y=height (tope del lobulo superior, donde se encuentra
    # con el tubo central). Toda la pared es curva (sin techo ni piso planos).
    y_values = [geometry.height_m * idx / max(1, axial_samples - 1) for idx in range(max(2, axial_samples))]
    outer_rings = [
        ring_points(geometry.allowed_radius_at_height(y_m), y_m, angular_segments) for y_m in y_values
    ]

    # El primer y ultimo ring se colapsan a un (cuasi-)punto (el fondo del lobulo
    # inferior es un domo cerrado; el tope del lobulo superior es donde empalma con
    # el tubo central de radio = central_tube_radius). Por construccion la geometria
    # ya garantiza esto, asi que las paredes solo necesitan ser triangularizadas.
    add_outer_wall_surfaces(
        groups,
        outer_rings,
        y_values,
        inlet_height_m=geometry.inlet_height_m,
        inlet_angle_deg=geometry.inlet_angle_deg,
        inlet_angular_width_deg=inlet_angular_width_deg,
        inlet_patch_height_m=inlet_patch_height_m,
    )

    # El fondo del lobulo inferior cierra a un punto pequeno; rellenamos un disco hacia el centro
    bottom_ring = outer_rings[0]
    bottom_center = (0.0, 0.0, 0.0)
    for idx in range(angular_segments):
        nxt = (idx + 1) % angular_segments
        add_triangle(groups["outer_wall"], bottom_center, bottom_ring[idx], bottom_ring[nxt])

    # Pared del tubo central: cilindro vertical desde central_tube_bottom_height_m
    # hasta height_m. El tubo es solido; por dentro es donde sale el agua (hueco).
    tube_bottom_y = geometry.central_tube_bottom_height_m
    tube_top_y = geometry.height_m
    tube_y_values = [
        tube_bottom_y + (tube_top_y - tube_bottom_y) * idx / max(1, axial_samples - 1)
        for idx in range(max(2, axial_samples))
    ]
    tube_rings = [
        ring_points(geometry.central_tube_radius_m, y_m, angular_segments) for y_m in tube_y_values
    ]
    # La pared del tubo - normales apuntando hacia afuera del tubo (es decir, hacia el fluido).
    for lower, upper in zip(tube_rings, tube_rings[1:]):
        add_ring_surface(groups["central_tube_wall"], upper, lower)

    # El tope del tubo central es el outlet pasivo (un disco horizontal en y=height_m,
    # del radio del tubo).
    top_tube_ring = tube_rings[-1]
    top_center = (0.0, tube_top_y, 0.0)
    for idx in range(angular_segments):
        nxt = (idx + 1) % angular_segments
        add_triangle(groups["central_tube_top"], top_center, top_tube_ring[nxt], top_tube_ring[idx])

    # La boca inferior del tubo central queda abierta hacia el fluido (no se anade
    # superficie ahi). Es donde el aire/fluido entra al tubo.

    return groups


def write_ascii_stl(path: Path, name: str, groups: TriangleGroups) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for group_name, triangles in groups.items():
            handle.write(f"solid {name}_{group_name}\n")
            for a, b, c in triangles:
                nx, ny, nz = triangle_normal(a, b, c)
                handle.write(f"  facet normal {nx:.8e} {ny:.8e} {nz:.8e}\n")
                handle.write("    outer loop\n")
                for point in (a, b, c):
                    handle.write(f"      vertex {point[0]:.8e} {point[1]:.8e} {point[2]:.8e}\n")
                handle.write("    endloop\n")
                handle.write("  endfacet\n")
            handle.write(f"endsolid {name}_{group_name}\n")


def metadata(
    geometry: ClassifierGeometry,
    source: Path,
    profile_count: int,
    triangle_count: int,
    inlet_angular_width_deg: float,
    inlet_patch_height_m: float,
) -> dict[str, object]:
    return {
        "source_geometry_json": str(source),
        "units": "meters",
        "device_type": "clepsamia_hourglass",
        "coordinate_system": {
            "x": "horizontal",
            "y": "vertical_up",
            "z": "horizontal",
            "axis_of_revolution": "y",
        },
        "geometry": asdict(geometry),
        "derived": {
            "height_m": geometry.height_m,
            "neck_height_m": geometry.neck_height_m,
            "neck_radius_m": geometry.neck_radius_m,
            "upper_max_radius_m": geometry.upper_max_radius_m,
            "upper_max_height_m": geometry.upper_max_height_m,
            "lower_max_radius_m": geometry.lower_max_radius_m,
            "lower_max_height_m": geometry.lower_max_height_m,
            "inlet_height_m": geometry.inlet_height_m,
            "inlet_angle_deg": geometry.inlet_angle_deg,
            "inlet_pitch_deg": geometry.inlet_pitch_deg,
            "inlet_yaw_deg": geometry.inlet_yaw_deg,
            "inlet_angular_width_deg": inlet_angular_width_deg,
            "inlet_patch_height_m": inlet_patch_height_m,
            "central_tube_radius_m": geometry.central_tube_radius_m,
            "central_tube_bottom_height_m": geometry.central_tube_bottom_height_m,
            "max_domain_radius_m": geometry.cylinder_radius_m,
            "profile_samples": profile_count,
            "stl_triangles": triangle_count,
        },
        "notes": [
            "Geometria 'Clepsamia' - reloj de arena con dos lobulos curvos sin superficies planas.",
            "Inlet tangencial en el lobulo superior; salida unica por el tubo central.",
            "El oro debe sedimentar en el fondo del lobulo inferior por gravedad y centrifugacion.",
            "El tubo central actua como vortex finder: fluido + particulas ligeras escapan por aqui.",
        ],
    }


def main() -> None:
    args = parse_args()
    source = Path(args.geometry_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = load_geometry(source)
    rows = profile_rows(geometry, args.axial_samples)
    references = reference_rows(geometry)
    triangle_groups = build_stl_triangles(
        geometry,
        axial_samples=args.axial_samples,
        angular_segments=args.angular_segments,
        inlet_angular_width_deg=args.inlet_angular_width_deg,
        inlet_patch_height_m=args.inlet_patch_height_m,
    )
    triangle_count = sum(len(triangles) for triangles in triangle_groups.values())

    profile_csv = output_dir / f"{args.name}_profile.csv"
    references_csv = output_dir / f"{args.name}_references.csv"
    metadata_json = output_dir / f"{args.name}_metadata.json"
    stl_path = output_dir / f"{args.name}_internal_volume.stl"

    write_csv(profile_csv, rows)
    write_csv(references_csv, references)
    write_ascii_stl(stl_path, args.name, triangle_groups)
    metadata_json.write_text(
        json.dumps(
            metadata(
                geometry,
                source,
                len(rows),
                triangle_count,
                inlet_angular_width_deg=args.inlet_angular_width_deg,
                inlet_patch_height_m=args.inlet_patch_height_m,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Perfil: {profile_csv}")
    print(f"Referencias: {references_csv}")
    print(f"Metadata: {metadata_json}")
    print(f"STL: {stl_path} ({triangle_count} triangulos)")


if __name__ == "__main__":
    main()
