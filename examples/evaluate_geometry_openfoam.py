from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_sim.rl_env import ACTION_NAMES, parameters_to_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua geometrías finalistas con OpenFOAM y particulas nativas."
    )
    parser.add_argument("--geometry-json", action="append", default=[], help="best_geometry.json o JSON con geometry.")
    parser.add_argument("--episodes-csv", default=None, help="episodes.csv de GA/RL para seleccionar top candidatos.")
    parser.add_argument("--top-n", type=int, default=8, help="Cantidad de filas top desde episodes.csv.")
    parser.add_argument("--output-root", default="cfd_sweeps/openfoam_eval", help="Directorio del barrido CFD.")
    parser.add_argument("--name-prefix", default="candidate", help="Prefijo de candidatos.")
    parser.add_argument("--base-cells", type=int, default=24, help="Resolucion base baja para barrido rapido.")
    parser.add_argument("--refinement-level", type=int, default=1, help="Refinamiento snappyHexMesh.")
    parser.add_argument("--cores", type=int, default=14, help="Numero de nucleos para MPI.")
    parser.add_argument("--axial-samples", type=int, default=96, help="Muestras axiales del STL.")
    parser.add_argument("--angular-segments", type=int, default=56, help="Segmentos angulares del STL.")
    parser.add_argument("--particle-end-time", type=float, default=0.1, help="Tiempo de tracking de particulas.")
    parser.add_argument("--particle-write-interval", type=float, default=0.05, help="Intervalo de escritura particulas.")
    parser.add_argument("--parcels-scale", type=float, default=0.05, help="Escala de parcels para barrido rapido.")
    parser.add_argument(
        "--collision-model",
        choices=("none", "pairCollision"),
        default="none",
        help="Modelo de colision para create_openfoam_particle_case.py.",
    )
    parser.add_argument("--docker-image", default="opencfd/openfoam-default:latest", help="Imagen Docker OpenFOAM.")
    parser.add_argument("--skip-flow-run", action="store_true", help="Solo crear caso CFD; no correr OpenFOAM.")
    parser.add_argument("--skip-particles", action="store_true", help="No crear/correr particulas.")
    parser.add_argument("--keep-existing", action="store_true", help="No borrar directorios de candidatos existentes.")
    parser.add_argument("--dry-run", action="store_true", help="Imprime comandos sin ejecutarlos.")
    parser.add_argument("--timeout-flow", type=int, default=None, help="Timeout en segundos para el solver de fluidos.")
    parser.add_argument("--timeout-particles", type=int, default=None, help="Timeout en segundos para el solver de particulas.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(command: list[str], cwd: Path, dry_run: bool = False, timeout: int | None = None) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout)


def docker_run(case_rel: Path, command: str, image: str, cwd: Path, dry_run: bool = False, timeout: int | None = None) -> None:
    uid = os.getuid()
    gid = os.getgid()
    run_command(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{uid}:{gid}",
            "-v",
            f"{cwd}:/work",
            image,
            "bash",
            "-lc",
            f"cd /work/{case_rel.as_posix()} && {command}",
        ],
        cwd=cwd,
        dry_run=dry_run,
        timeout=timeout,
    )


def path_for_docker(path: Path, cwd: Path) -> Path:
    return path.resolve().relative_to(cwd.resolve())


def geometry_payload_from_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "geometry" not in payload:
        payload = {"geometry": payload}
    return {
        "source": str(path),
        "reward": payload.get("reward"),
        "params": payload.get("params"),
        "geometry": payload["geometry"],
        "metrics": payload.get("metrics"),
    }


def params_from_episode_row(row: dict[str, str]) -> dict[str, float]:
    params: dict[str, float] = {}
    for name in ACTION_NAMES:
        value = row.get(name, "")
        if value in ("", None):
            raise ValueError(f"Falta parametro {name} en episodes.csv")
        params[name] = float(value)
    return params


def select_episode_candidates(path: Path, top_n: int) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    valid_rows = [
        row
        for row in rows
        if row.get("invalid", "False").lower() not in ("true", "1", "yes")
        and row.get("reward") not in ("", None)
    ]
    valid_rows.sort(key=lambda row: float(row["reward"]), reverse=True)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for row in valid_rows:
        params = params_from_episode_row(row)
        key = tuple(round(params[name], 4) for name in ACTION_NAMES)
        if key in seen:
            continue
        seen.add(key)
        geometry = parameters_to_geometry(params)
        geometry.validate()
        metrics = {
            key_name: float(row[key_name])
            for key_name in (
                "target_recovery_pct",
                "target_loss_pct",
                "non_target_rejection_pct",
                "trapped_contamination_pct",
                "unprocessed_pct",
            )
            if row.get(key_name) not in ("", None)
        }
        candidates.append(
            {
                "source": str(path),
                "generation": row.get("generation"),
                "individual": row.get("individual"),
                "reward": float(row["reward"]),
                "params": params,
                "geometry": asdict(geometry),
                "metrics": metrics,
            }
        )
        if len(candidates) >= top_n:
            break
    return candidates


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = [geometry_payload_from_file(Path(path)) for path in args.geometry_json]
    if args.episodes_csv:
        candidates.extend(select_episode_candidates(Path(args.episodes_csv), args.top_n))
    if not candidates:
        raise SystemExit("Debes pasar --geometry-json o --episodes-csv")
    return candidates


def write_candidate_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def score_from_combined(combined: dict[str, float], runtime_s: float) -> float:
    recovery = combined.get("target_recovery_pct", 0.0) / 100.0
    rejection = combined.get("non_target_rejection_pct", 0.0) / 100.0
    loss = combined.get("target_loss_pct", 0.0) / 100.0
    contamination = combined.get("trapped_contamination_pct", 0.0) / 100.0
    runtime_penalty = min(1.0, runtime_s / 600.0)
    return float(1.8 * recovery + 1.5 * rejection - 3.0 * loss - 2.4 * contamination - 0.2 * runtime_penalty)


def run_candidate(
    index: int,
    payload: dict[str, Any],
    args: argparse.Namespace,
    root: Path,
    cwd: Path,
) -> dict[str, Any]:
    name = f"{args.name_prefix}_{index:03d}"
    candidate_dir = root / name
    if candidate_dir.exists() and not args.keep_existing:
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    geometry_json = candidate_dir / f"{name}.json"
    write_candidate_json(geometry_json, payload)

    export_dir = candidate_dir / "export"
    case_dir = candidate_dir / "case"
    particles_dir = candidate_dir / "particles"
    stl_name = f"{name}_internal_volume.stl"
    metadata_json = export_dir / f"{name}_metadata.json"

    started = time.perf_counter()
    run_command(
        [
            sys.executable,
            "examples/export_geometry_cfd.py",
            "--geometry-json",
            str(geometry_json),
            "--output-dir",
            str(export_dir),
            "--axial-samples",
            str(args.axial_samples),
            "--angular-segments",
            str(args.angular_segments),
            "--name",
            name,
        ],
        cwd=cwd,
        dry_run=args.dry_run,
    )
    run_command(
        [
            sys.executable,
            "examples/create_openfoam_case.py",
            "--export-dir",
            str(export_dir),
            "--stl-name",
            stl_name,
            "--metadata",
            str(metadata_json),
            "--case-dir",
            str(case_dir),
            "--base-cells",
            str(args.base_cells),
            "--refinement-level",
            str(args.refinement_level),
            "--cores",
            str(args.cores),
        ],
        cwd=cwd,
        dry_run=args.dry_run,
    )
    if not args.skip_flow_run:
        docker_run(path_for_docker(case_dir, cwd), "./Allclean && ./Allrun && ./AllrunFlow", args.docker_image, cwd, args.dry_run, timeout=args.timeout_flow)

    combined: dict[str, float] = {}
    metrics_json = particles_dir / "particle_metrics.json"
    if not args.skip_particles:
        run_command(
            [
                sys.executable,
                "examples/create_openfoam_particle_case.py",
                "--base-case",
                str(case_dir),
                "--metadata",
                str(metadata_json),
                "--output-root",
                str(particles_dir),
                "--end-time",
                str(args.particle_end_time),
                "--write-interval",
                str(args.particle_write_interval),
                "--parcels-scale",
                str(args.parcels_scale),
                "--collision-model",
                args.collision_model,
                "--cores",
                str(args.cores),
            ],
            cwd=cwd,
            dry_run=args.dry_run,
        )
        docker_run(path_for_docker(particles_dir, cwd), "./AllrunParticles", args.docker_image, cwd, args.dry_run, timeout=args.timeout_particles)
        run_command(
            [
                sys.executable,
                "examples/analyze_openfoam_particles.py",
                "--cases-root",
                str(particles_dir),
            ],
            cwd=cwd,
            dry_run=args.dry_run,
        )
        if metrics_json.is_file():
            combined = json.loads(metrics_json.read_text(encoding="utf-8")).get("combined", {})

    runtime_s = time.perf_counter() - started
    score = score_from_combined(combined, runtime_s) if combined else float("nan")
    result = {
        "candidate": name,
        "source": payload.get("source", ""),
        "local_reward": payload.get("reward"),
        "cfd_score": score,
        "runtime_s": runtime_s,
        "geometry_json": str(geometry_json),
        "case_dir": str(case_dir),
        "particles_dir": str(particles_dir),
        "particle_metrics_json": str(metrics_json),
        **combined,
    }
    (candidate_dir / "openfoam_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{name}: cfd_score={score:.4f} runtime_s={runtime_s:.1f}")
    return result


def write_results(root: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    fieldnames: list[str] = []
    for result in results:
        for key in result:
            if key not in fieldnames:
                fieldnames.append(key)
    with (root / "cfd_score.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    sortable = [result for result in results if isinstance(result.get("cfd_score"), float)]
    sortable = [result for result in sortable if result["cfd_score"] == result["cfd_score"]]
    if sortable:
        best = max(sortable, key=lambda item: float(item["cfd_score"]))
        best_payload = json.loads(Path(best["geometry_json"]).read_text(encoding="utf-8"))
        best_payload["cfd_result"] = best
        (root / "best_cfd_geometry.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cwd = repo_root()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args)
    manifest = {
        "args": vars(args),
        "candidate_count": len(candidates),
        "note": "OpenFOAM se usa como juez fisico caro para re-ranking, no como fitness masivo del GA.",
    }
    (output_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results = [run_candidate(idx, payload, args, output_root, cwd) for idx, payload in enumerate(candidates)]
    write_results(output_root, results)
    print(f"Resumen CFD: {output_root / 'cfd_score.csv'}")
    if (output_root / "best_cfd_geometry.json").is_file():
        print(f"Mejor geometria CFD: {output_root / 'best_cfd_geometry.json'}")


if __name__ == "__main__":
    main()
