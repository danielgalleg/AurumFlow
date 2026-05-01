from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


VECTOR_RE = re.compile(
    r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)(?:\s+[-+0-9]+)?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analiza resultados lagrangianos OpenFOAM por material.")
    parser.add_argument("--cases-root", required=True, help="Directorio con subcasos por material.")
    parser.add_argument("--output", default=None, help="CSV resumen. Por defecto cases-root/particle_metrics.csv.")
    parser.add_argument(
        "--particles-csv",
        default=None,
        help="CSV con posiciones finales para ParaView. Por defecto cases-root/particles_latest.csv.",
    )
    parser.add_argument(
        "--trap-margin-m",
        type=float,
        default=0.012,
        help="Margen vertical sobre trap_height para considerar captura.",
    )
    parser.add_argument(
        "--trap-radius-factor",
        type=float,
        default=2.0,
        help="Factor sobre trap_bottom_radius para considerar captura en la trampa.",
    )
    return parser.parse_args()


def numeric_time_dirs(case_dir: Path) -> list[tuple[float, Path]]:
    times: list[tuple[float, Path]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if (child / "lagrangian" / "kinematicCloud" / "positions").is_file():
            times.append((value, child))
    return sorted(times, key=lambda item: item[0])


def latest_positions_file(case_dir: Path) -> tuple[float, Path | None]:
    times = numeric_time_dirs(case_dir)
    if not times:
        return 0.0, None
    value, path = times[-1]
    return value, path / "lagrangian" / "kinematicCloud" / "positions"


def read_positions(path: Path | None) -> list[tuple[float, float, float]]:
    if path is None:
        return []
    positions: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = VECTOR_RE.search(line)
        if match:
            positions.append(tuple(float(match.group(i)) for i in range(1, 4)))
    return positions


def parse_last_fates(log_path: Path) -> dict[str, dict[str, int]]:
    if not log_path.is_file():
        return {}
    fates: dict[str, dict[str, int]] = {}
    current_patch = "system"
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("Parcel fate:"):
            current_patch = stripped.split("Parcel fate:", 1)[1].strip()
            current_patch = current_patch.removeprefix("patch ").strip()
            fates.setdefault(current_patch, {})
            continue
        if stripped.startswith("- escape") or stripped.startswith("- stick"):
            label, values = stripped[2:].split("=", 1)
            event = label.strip().split()[0]
            count_text = values.split(",", 1)[0].strip()
            try:
                fates.setdefault(current_patch, {})[event] = int(float(count_text))
            except ValueError:
                pass
    return fates


def classify_position(
    position: tuple[float, float, float],
    manifest: dict[str, Any],
    trap_margin_m: float,
    trap_radius_factor: float,
) -> str:
    x, y, z = position
    radius = math.hypot(x, z)
    trap_height = float(manifest["trap_height_m"])
    trap_radius = float(manifest["trap_bottom_radius_m"])
    overflow_radius = float(manifest["overflow_tube_radius_m"])
    overflow_bottom = float(manifest["overflow_tube_bottom_height_m"])
    if y <= trap_height + trap_margin_m and radius <= max(0.01, trap_radius_factor * trap_radius):
        return "trapped_spatial"
    if y >= overflow_bottom and radius <= max(0.015, 2.2 * overflow_radius):
        return "overflow_zone"
    return "in_device"


def material_case_dirs(cases_root: Path) -> list[Path]:
    return sorted(
        child
        for child in cases_root.iterdir()
        if child.is_dir() and (child / "particle_manifest.json").is_file()
    )


def analyze_case(
    case_dir: Path,
    trap_margin_m: float,
    trap_radius_factor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((case_dir / "particle_manifest.json").read_text(encoding="utf-8"))
    material = manifest["material"]
    latest_time, positions_path = latest_positions_file(case_dir)
    positions = read_positions(positions_path)
    fates = parse_last_fates(case_dir / "particleFoam.log")
    overflow_patch = next((name for name in fates if name.endswith("_overflow_mouth")), "")
    trap_patch = next((name for name in fates if name.endswith("_trap_floor")), "")
    overflow_escape = fates.get(overflow_patch, {}).get("escape", 0)
    trap_stick = fates.get(trap_patch, {}).get("stick", 0)
    initial_count = int(manifest["initial_count"])

    particle_rows: list[dict[str, Any]] = []
    spatial_counts: dict[str, int] = {"trapped_spatial": 0, "overflow_zone": 0, "in_device": 0}
    for idx, position in enumerate(positions):
        status = classify_position(position, manifest, trap_margin_m, trap_radius_factor)
        spatial_counts[status] = spatial_counts.get(status, 0) + 1
        x, y, z = position
        particle_rows.append(
            {
                "case": case_dir.name,
                "material": material["name"],
                "label": material["label"],
                "target": int(bool(material["target"])),
                "particle_id": idx,
                "time": latest_time,
                "x": x,
                "y": y,
                "z": z,
                "radius": math.hypot(x, z),
                "status": status,
                "density_kg_m3": material["density_kg_m3"],
                "diameter_m": material["diameter_m"],
            }
        )

    remaining = len(positions)
    missing = max(0, initial_count - remaining)
    trapped = max(trap_stick, spatial_counts["trapped_spatial"])
    overflow = max(overflow_escape, spatial_counts["overflow_zone"], missing if overflow_escape else 0)
    summary = {
        "case": case_dir.name,
        "material": material["name"],
        "label": material["label"],
        "target": int(bool(material["target"])),
        "density_kg_m3": material["density_kg_m3"],
        "diameter_m": material["diameter_m"],
        "initial_count": initial_count,
        "latest_time": latest_time,
        "remaining_count": remaining,
        "missing_count": missing,
        "trapped_count": trapped,
        "overflow_count": overflow,
        "in_device_count": spatial_counts["in_device"],
        "trapped_pct": 100.0 * trapped / max(1, initial_count),
        "overflow_pct": 100.0 * overflow / max(1, initial_count),
        "in_device_pct": 100.0 * spatial_counts["in_device"] / max(1, initial_count),
        "collision_model": manifest.get("collision_model", ""),
        "solver": manifest.get("solver", ""),
    }
    return summary, particle_rows


def combined_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    target_initial = sum(row["initial_count"] for row in rows if row["target"])
    target_trapped = sum(row["trapped_count"] for row in rows if row["target"])
    target_overflow = sum(row["overflow_count"] for row in rows if row["target"])
    non_target_initial = sum(row["initial_count"] for row in rows if not row["target"])
    non_target_trapped = sum(row["trapped_count"] for row in rows if not row["target"])
    non_target_overflow = sum(row["overflow_count"] for row in rows if not row["target"])
    total_trapped = target_trapped + non_target_trapped
    return {
        "target_recovery_pct": 100.0 * target_trapped / max(1, target_initial),
        "target_loss_pct": 100.0 * target_overflow / max(1, target_initial),
        "non_target_rejection_pct": 100.0 * non_target_overflow / max(1, non_target_initial),
        "trapped_contamination_pct": 100.0 * non_target_trapped / max(1, total_trapped),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cases_root = Path(args.cases_root)
    output = Path(args.output) if args.output else cases_root / "particle_metrics.csv"
    particles_csv = Path(args.particles_csv) if args.particles_csv else cases_root / "particles_latest.csv"
    summaries: list[dict[str, Any]] = []
    particles: list[dict[str, Any]] = []
    for case_dir in material_case_dirs(cases_root):
        summary, particle_rows = analyze_case(case_dir, args.trap_margin_m, args.trap_radius_factor)
        summaries.append(summary)
        particles.extend(particle_rows)
    write_csv(output, summaries)
    write_csv(particles_csv, particles)
    combined = combined_metrics(summaries)
    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps({"materials": summaries, "combined": combined}, indent=2),
        encoding="utf-8",
    )
    print(f"Resumen: {output}")
    print(f"Particulas: {particles_csv}")
    print(f"JSON: {json_path}")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
