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
    parser = argparse.ArgumentParser(description="Analiza resultados lagrangianos OpenFOAM por material (Clepsamia).")
    parser.add_argument("--cases-root", required=True, help="Directorio con subcasos por material.")
    parser.add_argument("--output", default=None, help="CSV resumen. Por defecto cases-root/particle_metrics.csv.")
    parser.add_argument(
        "--particles-csv",
        default=None,
        help="CSV con posiciones finales para ParaView. Por defecto cases-root/particles_latest.csv.",
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
) -> str:
    """Clasifica una particula que sigue dentro del dispositivo segun su altura.
    En la geometria Clepsamia hay solo dos zonas: lobulo superior y lobulo inferior,
    separados por el cuello.
    """
    x, y, z = position
    neck_height = float(manifest.get("neck_height_m", 0.0))
    if y < neck_height:
        return "in_lower_lobe"
    return "in_upper_lobe"


def material_case_dirs(cases_root: Path) -> list[Path]:
    return sorted(
        child
        for child in cases_root.iterdir()
        if child.is_dir() and (child / "particle_manifest.json").is_file()
    )


def analyze_case(case_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((case_dir / "particle_manifest.json").read_text(encoding="utf-8"))
    material = manifest["material"]
    latest_time, positions_path = latest_positions_file(case_dir)
    positions = read_positions(positions_path)
    fates = parse_last_fates(case_dir / "particleFoam.log")
    central_tube_patch = next((name for name in fates if name.endswith("_central_tube_top")), "")
    inlet_patch = next((name for name in fates if name.endswith("_inlet")), "")
    central_tube_escape = fates.get(central_tube_patch, {}).get("escape", 0)
    inlet_escape = fates.get(inlet_patch, {}).get("escape", 0)
    initial_count = int(manifest["initial_count"])

    # Calcular masa de una particula
    volume_m3 = (4.0 / 3.0) * math.pi * (float(material["diameter_m"]) / 2.0) ** 3
    particle_mass_kg = volume_m3 * float(material["density_kg_m3"])

    particle_rows: list[dict[str, Any]] = []
    spatial_counts: dict[str, int] = {"in_lower_lobe": 0, "in_upper_lobe": 0}
    for idx, position in enumerate(positions):
        status = classify_position(position, manifest)
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
    # Particulas que escaparon por el tubo central o por el inlet (atras) son consideradas salidas
    escaped = max(central_tube_escape + inlet_escape, missing)
    # Las que siguen dentro: idealmente el oro se queda en el lobulo inferior.
    in_lower = spatial_counts["in_lower_lobe"]
    in_upper = spatial_counts["in_upper_lobe"]

    summary = {
        "case": case_dir.name,
        "material": material["name"],
        "label": material["label"],
        "target": int(bool(material["target"])),
        "particle_mass_kg": particle_mass_kg,
        "density_kg_m3": material["density_kg_m3"],
        "diameter_m": material["diameter_m"],
        "initial_count": initial_count,
        "latest_time": latest_time,
        "remaining_count": remaining,
        "escaped_count": escaped,
        "in_lower_lobe_count": in_lower,
        "in_upper_lobe_count": in_upper,
        "central_tube_escape_count": central_tube_escape,
        "inlet_escape_count": inlet_escape,
        "escaped_pct": 100.0 * escaped / max(1, initial_count),
        "in_lower_lobe_pct": 100.0 * in_lower / max(1, initial_count),
        "in_upper_lobe_pct": 100.0 * in_upper / max(1, initial_count),
        "collision_model": manifest.get("collision_model", ""),
        "solver": manifest.get("solver", ""),
    }
    return summary, particle_rows


def combined_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Metricas para Clepsamia:
    - target_recovery_pct: oro retenido dentro del dispositivo (no escapo).
    - target_loss_pct: oro que escapo por el tubo central o el inlet.
    - non_target_rejection_pct: arena que SI escapo por el tubo central (deseado).
    - trapped_contamination_pct: arena que se quedo dentro del dispositivo (no salio).
    """
    target_initial_mass = sum(row["initial_count"] * row["particle_mass_kg"] for row in rows if row["target"])
    target_escaped_mass = sum(row["escaped_count"] * row["particle_mass_kg"] for row in rows if row["target"])
    target_retained_mass = max(0.0, target_initial_mass - target_escaped_mass)

    non_target_initial_mass = sum(row["initial_count"] * row["particle_mass_kg"] for row in rows if not row["target"])
    non_target_escaped_mass = sum(row["escaped_count"] * row["particle_mass_kg"] for row in rows if not row["target"])
    non_target_retained_mass = max(0.0, non_target_initial_mass - non_target_escaped_mass)

    return {
        "target_recovery_pct": 100.0 * target_retained_mass / max(1e-12, target_initial_mass),
        "target_loss_pct": 100.0 * target_escaped_mass / max(1e-12, target_initial_mass),
        "non_target_rejection_pct": 100.0 * non_target_escaped_mass / max(1e-12, non_target_initial_mass),
        "trapped_contamination_pct": 100.0 * non_target_retained_mass / max(1e-12, target_retained_mass + non_target_retained_mass),
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
        summary, particle_rows = analyze_case(case_dir)
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
