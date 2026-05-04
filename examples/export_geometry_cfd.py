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
        description="Exporta geometria axisimetrica del clasificador para CFD/SPH externo."
    )
    parser.add_argument("--geometry-json", required=True, help="JSON con campo 'geometry'.")
    parser.add_argument("--output-dir", required=True, help="Directorio de exportacion.")
    parser.add_argument("--axial-samples", type=int, default=160, help="Muestras verticales del perfil.")
    parser.add_argument("--radial-samples", type=int, default=32, help="Muestras radiales del piso/trampa.")
    parser.add_argument("--angular-segments", type=int, default=96, help="Segmentos de revolucion para STL.")
    parser.add_argument("--name", default="classifier", help="Prefijo de archivos exportados.")
    parser.add_argument("--inlet-angle-deg", type=float, default=-90.0, help="Angulo central del patch de entrada.")
    parser.add_argument(
        "--inlet-angular-width-deg",
        type=float,
        default=28.0,
        help="Ancho angular del patch de entrada en la pared.",
    )
    parser.add_argument(
        "--inlet-height-m",
        type=float,
        default=0.035,
        help="Alto vertical aproximado del patch de entrada.",
    )
    return parser.parse_args()


def load_geometry(path: Path) -> ClassifierGeometry:
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometry_data = payload["geometry"] if "geometry" in payload else payload
    geometry = ClassifierGeometry(**geometry_data)
    geometry.validate()
    return geometry


def zone_for_height(y_m: float, geometry: ClassifierGeometry) -> str:
    if y_m < geometry.trap_height_m:
        return "trap_wall"
    if y_m < geometry.cone_top_height_m:
        return "cone_transition"
    return "cyclone_body"


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
            "name": "inlet_center",
            "height_m": geometry.inlet_height_m,
            "radius_m": geometry.allowed_radius_at_height(geometry.inlet_height_m),
            "note": "Entrada tangencial aproximada; definir direccion tangencial en CFD.",
        },
        {
            "name": "overflow_tube_wall",
            "height_m": geometry.overflow_tube_bottom_height_m,
            "radius_m": geometry.overflow_tube_radius_m,
            "note": "Boca inferior del tubo central de rebalse.",
        },
        {
            "name": "trap_neck",
            "height_m": geometry.trap_height_m,
            "radius_m": geometry.cone_neck_half_depth_m,
            "note": "Garganta sobre la trampa.",
        },
        {
            "name": "trap_bottom_edge",
            "height_m": geometry.trap_floor_height_at_radius(geometry.trap_bottom_half_depth_m),
            "radius_m": geometry.trap_bottom_half_depth_m,
            "note": "Borde inferior de la trampa.",
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


def ring_points(radius_m: float, height_m: float, angular_segments: int) -> list[tuple[float, float, float]]:
    return [
        (
            radius_m * math.cos(2.0 * math.pi * idx / angular_segments),
            height_m,
            radius_m * math.sin(2.0 * math.pi * idx / angular_segments),
        )
        for idx in range(angular_segments)
    ]


def triangle_normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (nx / length, ny / length, nz / length)


def add_triangle(
    triangles: list[Triangle],
    a: Point,
    b: Point,
    c: Point,
) -> None:
    if a == b or b == c or a == c:
        return
    triangles.append((a, b, c))


def add_ring_surface(
    triangles: list[Triangle],
    lower: list[Point],
    upper: list[Point],
) -> None:
    count = len(lower)
    for idx in range(count):
        nxt = (idx + 1) % count
        add_triangle(triangles, lower[idx], upper[idx], upper[nxt])
        add_triangle(triangles, lower[idx], upper[nxt], lower[nxt])


def angle_difference(a_rad: float, b_rad: float) -> float:
    return math.atan2(math.sin(a_rad - b_rad), math.cos(a_rad - b_rad))


def add_outer_wall_surfaces(
    groups: TriangleGroups,
    lower_rings: list[list[Point]],
    y_values: list[float],
    inlet_height_m: float,
    inlet_angle_deg: float,
    inlet_angular_width_deg: float,
    inlet_patch_height_m: float,
) -> None:
    angular_segments = len(lower_rings[0])
    inlet_angle_rad = math.radians(inlet_angle_deg)
    half_angle_rad = math.radians(inlet_angular_width_deg) * 0.5
    half_height_m = inlet_patch_height_m * 0.5
    for ring_idx, (lower, upper) in enumerate(zip(lower_rings, lower_rings[1:])):
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


def add_disk(
    triangles: list[Triangle],
    center: Point,
    ring: list[Point],
    normal: str,
) -> None:
    for idx in range(len(ring)):
        nxt = (idx + 1) % len(ring)
        if normal == "up":
            add_triangle(triangles, center, ring[nxt], ring[idx])
        elif normal == "down":
            add_triangle(triangles, center, ring[idx], ring[nxt])
        else:
            raise ValueError("normal debe ser 'up' o 'down'")


def build_stl_triangles(
    geometry: ClassifierGeometry,
    axial_samples: int,
    radial_samples: int,
    angular_segments: int,
    inlet_angle_deg: float,
    inlet_angular_width_deg: float,
    inlet_patch_height_m: float,
) -> TriangleGroups:
    groups: TriangleGroups = {
        "outer_wall": [],
        "inlet": [],
        "roof": [],
        "overflow_tube_wall": [],
        "overflow_mouth": [],
        "trap_floor": [],
    }

    y_values = [geometry.height_m * idx / max(1, axial_samples - 1) for idx in range(max(2, axial_samples))]
    outer_rings = [
        ring_points(geometry.allowed_radius_at_height(y_m), y_m, angular_segments) for y_m in y_values
    ]
    add_outer_wall_surfaces(
        groups,
        outer_rings,
        y_values,
        inlet_height_m=geometry.inlet_height_m,
        inlet_angle_deg=geometry.inlet_angle_deg,
        inlet_angular_width_deg=inlet_angular_width_deg,
        inlet_patch_height_m=inlet_patch_height_m,
    )

    # Top annulus: chamber roof excluding the central overflow tube opening.
    top_outer = outer_rings[-1]
    top_inner = ring_points(geometry.overflow_tube_radius_m, geometry.height_m, angular_segments)
    add_ring_surface(groups["roof"], top_outer, top_inner)

    # Overflow tube wall (now curved and tapered)
    tube_y_values = [
        geometry.overflow_tube_bottom_height_m + (geometry.height_m - geometry.overflow_tube_bottom_height_m) * idx / max(1, axial_samples - 1)
        for idx in range(max(2, axial_samples))
    ]
    tube_rings = [
        ring_points(geometry.overflow_tube_radius_at_height(y_m), y_m, angular_segments)
        for y_m in tube_y_values
    ]
    for lower, upper in zip(tube_rings, tube_rings[1:]):
        add_ring_surface(groups["overflow_tube_wall"], upper, lower)  # Note: upper to lower for correct normal (pointing inwards to the fluid)
    
    tube_bottom = tube_rings[0]
    
    add_disk(
        groups["overflow_mouth"],
        (0.0, geometry.overflow_tube_bottom_height_m, 0.0),
        tube_bottom,
        normal="up",
    )

    # Trap floor, including bowl/dome curvature if configured.
    radial_values = [
        geometry.trap_bottom_half_depth_m * idx / max(1, radial_samples - 1)
        for idx in range(max(2, radial_samples))
    ]
    floor_rings = [
        ring_points(radius_m, geometry.trap_floor_height_at_radius(radius_m), angular_segments)
        for radius_m in radial_values
    ]
    center = floor_rings[0][0]
    first_ring = floor_rings[1]
    add_disk(groups["trap_floor"], center, first_ring, normal="down")
    for inner, outer in zip(floor_rings[1:], floor_rings[2:]):
        add_ring_surface(groups["trap_floor"], inner, outer)

    outer_wall_bottom = outer_rings[0]
    floor_outer = floor_rings[-1]
    if any(abs(a[1] - b[1]) > 1e-9 for a, b in zip(outer_wall_bottom, floor_outer)):
        add_ring_surface(groups["trap_floor"], outer_wall_bottom, floor_outer)

    return groups


def write_ascii_stl(
    path: Path,
    name: str,
    groups: TriangleGroups,
) -> None:
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
    inlet_angle_deg: float,
    inlet_angular_width_deg: float,
    inlet_patch_height_m: float,
) -> dict[str, object]:
    return {
        "source_geometry_json": str(source),
        "units": "meters",
        "coordinate_system": {
            "x": "horizontal",
            "y": "vertical_up",
            "z": "horizontal",
            "axis_of_revolution": "y",
        },
        "geometry": asdict(geometry),
        "derived": {
            "height_m": geometry.height_m,
            "max_domain_radius_m": geometry.cylinder_radius_m,
            "body_top_radius_m": geometry.body_top_radius_m,
            "body_bottom_radius_m": geometry.body_bottom_radius_m,
            "cone_neck_radius_m": geometry.cone_neck_half_depth_m,
            "trap_bottom_radius_m": geometry.trap_bottom_half_depth_m,
            "overflow_tube_radius_m": geometry.overflow_tube_radius_m,
            "overflow_tube_bottom_height_m": geometry.overflow_tube_bottom_height_m,
            "inlet_height_m": geometry.inlet_height_m,
            "inlet_angle_deg": geometry.inlet_angle_deg,
            "inlet_pitch_deg": geometry.inlet_pitch_deg,
            "inlet_angular_width_deg": inlet_angular_width_deg,
            "inlet_patch_height_m": inlet_patch_height_m,
            "profile_samples": profile_count,
            "stl_triangles": triangle_count,
        },
        "notes": [
            "STL representa fronteras internas axisimetricas por revolucion del perfil radial.",
            "La entrada tangencial se exporta como referencia, no como boquilla 3D detallada.",
            "El tubo central de rebalse se exporta como pared cilindrica interna y abertura superior.",
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
        radial_samples=args.radial_samples,
        angular_segments=args.angular_segments,
        inlet_angle_deg=args.inlet_angle_deg,
        inlet_angular_width_deg=args.inlet_angular_width_deg,
        inlet_patch_height_m=args.inlet_height_m,
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
                inlet_angle_deg=args.inlet_angle_deg,
                inlet_angular_width_deg=args.inlet_angular_width_deg,
                inlet_patch_height_m=args.inlet_height_m,
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
