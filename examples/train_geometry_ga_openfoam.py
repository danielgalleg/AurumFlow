from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_sim.rl_env import (
    ACTION_BOUNDS,
    ACTION_NAMES,
    action_to_parameters,
    parameters_to_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimiza geometria con algoritmo genetico/evolutivo usando OpenFOAM."
    )
    parser.add_argument("--generations", type=int, default=15, help="Generaciones evolutivas.")
    parser.add_argument("--population", type=int, default=16, help="Individuos por generacion.")
    parser.add_argument("--base-cells", type=int, default=16, help="Resolucion base baja de CFD.")
    parser.add_argument("--particle-end-time", type=float, default=2.0, help="Tiempo de sim de particulas.")
    parser.add_argument("--parcels-scale", type=float, default=0.03, help="Escala de inyeccion de particulas.")
    parser.add_argument("--cores-per-eval", type=int, default=2, help="Nucleos MPI por evaluacion.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Evaluaciones CFD en paralelo (n-jobs * cores_per_eval <= CPU threads).")
    parser.add_argument("--elite-fraction", type=float, default=0.20, help="Fraccion de elite conservada.")
    parser.add_argument("--mutation-prob", type=float, default=0.30, help="Probabilidad de mutar cada gen.")
    parser.add_argument("--mutation-sigma", type=float, default=0.25, help="Desviacion estandar de mutacion.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Individuos por torneo.")
    parser.add_argument("--warm-start-jsons", nargs="+", help="Archivos best_geometry.json previos para sembrar la poblacion.")
    parser.add_argument("--output-dir", default="rl_runs/ga_openfoam", help="Directorio de salida.")
    return parser.parse_args()


def params_to_action(params: dict[str, float]) -> np.ndarray:
    params = dict(params)
    if "body_wall_length_ratio" not in params and "cone_top_height_ratio" in params:
        params["body_wall_length_ratio"] = 1.0 - float(params["cone_top_height_ratio"])
    if "trap_wall_length_ratio" not in params and "trap_height_ratio" in params:
        params["trap_wall_length_ratio"] = float(params["trap_height_ratio"])
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    for idx, name in enumerate(ACTION_NAMES):
        if name not in params:
            continue
        low, high = ACTION_BOUNDS[name]
        normalized = 2.0 * (float(params[name]) - low) / (high - low) - 1.0
        action[idx] = np.clip(normalized, -1.0, 1.0)
    return action


def cfd_reward(metrics_combined: dict[str, float]) -> float:
    gold_recovery = metrics_combined.get("target_recovery_pct", 0.0) / 100.0
    gold_loss = metrics_combined.get("target_loss_pct", 100.0) / 100.0
    non_target_rejection = metrics_combined.get("non_target_rejection_pct", 0.0) / 100.0
    contamination = metrics_combined.get("trapped_contamination_pct", 100.0) / 100.0
    return float(
        1.8 * gold_recovery
        + 1.5 * non_target_rejection
        - 3.0 * gold_loss
        - 2.4 * contamination
    )


def evaluate_action_cfd(payload: tuple[int, int, list[float], str, argparse.Namespace]) -> dict[str, Any]:
    generation, individual, action_values, output_dir, args = payload
    action = np.asarray(action_values, dtype=np.float32)
    params = action_to_parameters(action)
    
    cand_id = f"gen{generation:03d}_ind{individual:03d}_{uuid.uuid4().hex[:6]}"
    cand_dir = Path(output_dir) / "evaluations" / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    
    invalid = False
    reward = -5.0
    
    try:
        geometry = parameters_to_geometry(params)
        geometry.validate()
        
        geom_json = cand_dir / "geometry.json"
        geom_json.write_text(json.dumps({"params": params, "geometry": asdict(geometry)}, indent=2))
        
        cmd = [
            sys.executable,
            "examples/evaluate_geometry_openfoam.py",
            "--geometry-json", str(geom_json),
            "--output-root", str(cand_dir),
            "--name-prefix", cand_id,
            "--base-cells", str(args.base_cells),
            "--refinement-level", "1",
            "--particle-end-time", str(args.particle_end_time),
            "--parcels-scale", str(args.parcels_scale),
            "--cores", str(args.cores_per_eval),
            "--timeout-flow", "600",
            "--timeout-particles", "900",
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        
        subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=900  # 15 minutos maximo por evaluacion para evitar cuelgues infinitos
        )
        
        result_file = cand_dir / f"{cand_id}_000" / "openfoam_result.json"
        if not result_file.is_file():
            raise RuntimeError("OpenFOAM did not produce a result JSON")
            
        result = json.loads(result_file.read_text())
        if "target_recovery_pct" not in result:
            raise RuntimeError("OpenFOAM result missing combined metrics")
            
        reward = cfd_reward(result)
        metrics = {
            "target_recovery_pct": result.get("target_recovery_pct"),
            "target_loss_pct": result.get("target_loss_pct"),
            "non_target_rejection_pct": result.get("non_target_rejection_pct"),
            "trapped_contamination_pct": result.get("trapped_contamination_pct"),
        }
        invalid = False
        error = ""
        
        # Cleanup heavy OpenFOAM case data to save disk space
        shutil.rmtree(cand_dir / f"{cand_id}_000" / "case", ignore_errors=True)
        shutil.rmtree(cand_dir / f"{cand_id}_000" / "particles", ignore_errors=True)
        
    except Exception as exc:
        geometry = None
        metrics = None
        reward = -5.0
        error = str(exc)
        invalid = True

    finally:
        # Limpieza agresiva de disco:
        # Siempre borramos las carpetas pesadas de OpenFOAM incluso si hubo error
        shutil.rmtree(cand_dir / f"{cand_id}_000" / "case", ignore_errors=True)
        shutil.rmtree(cand_dir / f"{cand_id}_000" / "particles", ignore_errors=True)
        
        # Si el modelo falló o su puntaje es muy bajo, borramos toda su carpeta para liberar espacio
        if invalid or reward < 2.0:
            shutil.rmtree(cand_dir, ignore_errors=True)

    return {
        "generation": generation,
        "individual": individual,
        "reward": reward,
        "action": action.tolist(),
        "params": params,
        "geometry": asdict(geometry) if geometry else None,
        "metrics": metrics,
        "invalid": invalid,
        "error": error,
        "id": cand_id,
    }


def init_population(args: argparse.Namespace, rng: np.random.Generator) -> list[np.ndarray]:
    population: list[np.ndarray] = []
    
    if args.warm_start_jsons:
        for ws_json in args.warm_start_jsons:
            if Path(ws_json).is_file():
                try:
                    payload = json.loads(Path(ws_json).read_text(encoding="utf-8"))
                    if "action" in payload:
                        action = np.asarray(payload["action"], dtype=np.float32)
                        population.append(action)
                    elif "params" in payload:
                        action = params_to_action(payload["params"])
                        population.append(action)
                    print(f"Warm start inyectado desde {ws_json}")
                    
                    # Generar algunas mutaciones cercanas a esta semilla
                    for _ in range(max(0, min(8, args.population // (4 * len(args.warm_start_jsons))))):
                        mutated = action + rng.normal(0.0, 0.15, size=action.shape).astype(np.float32)
                        population.append(np.clip(mutated, -1.0, 1.0))
                except Exception as exc:
                    print(f"Error cargando warm start {ws_json}: {exc}")

    while len(population) < args.population:
        action = rng.uniform(-1.0, 1.0, size=len(ACTION_NAMES)).astype(np.float32)
        population.append(action)

    return population


def next_generation(
    population: list[np.ndarray],
    rewards: list[float],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    indices = np.argsort(rewards)[::-1]
    elite_count = max(1, int(args.population * args.elite_fraction))
    elites = [population[i].copy() for i in indices[:elite_count]]

    new_population = list(elites)

    def tournament() -> np.ndarray:
        t_size = min(args.tournament_size, len(population))
        participants = rng.choice(len(population), size=t_size, replace=False)
        best_idx = int(participants[np.argmax([rewards[i] for i in participants])])
        return population[best_idx]

    while len(new_population) < args.population:
        parent1 = tournament()
        parent2 = tournament()

        crossover_mask = rng.random(size=len(ACTION_NAMES)) < 0.5
        child = np.where(crossover_mask, parent1, parent2)

        mutation_mask = rng.random(size=len(ACTION_NAMES)) < args.mutation_prob
        mutations = rng.normal(0.0, args.mutation_sigma, size=len(ACTION_NAMES))
        child = np.where(mutation_mask, child + mutations, child)
        child = np.clip(child, -1.0, 1.0).astype(np.float32)

        new_population.append(child)

    return new_population


def save_best(path: Path, result: dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reward": float(result["reward"]),
        "action": result["action"],
        "params": result["params"],
        "geometry": result["geometry"],
        "metrics": result["metrics"],
        "cfd_config": {
            "base_cells": args.base_cells,
            "particle_end_time": args.particle_end_time,
            "parcels_scale": args.parcels_scale,
        },
        "optimizer": "genetic_openfoam",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(7)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = out_dir / "episodes.csv"
    summary_path = out_dir / "generations.csv"
    best_path = out_dir / "best_geometry.json"

    print(f"Optimizacion CFD (OpenFOAM) iniciada.")
    print(f"Poblacion: {args.population}, Generaciones: {args.generations}")
    print(f"Workers CFD paralelos: {args.n_jobs} (x {args.cores_per_eval} MPI cores = {args.n_jobs * args.cores_per_eval} hilos CPU)")

    population = init_population(args, rng)
    best_result: dict[str, Any] | None = None

    for generation in range(args.generations):
        payloads = [
            (generation, i, pop.tolist(), str(out_dir), args)
            for i, pop in enumerate(population)
        ]

        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            for result in executor.map(evaluate_action_cfd, payloads):
                results.append(result)

        rewards = [res["reward"] for res in results]

        write_header = not episodes_path.exists()
        with episodes_path.open("a", newline="", encoding="utf-8") as handle:
            for result in results:
                row = {
                    "generation": result["generation"],
                    "individual": result["individual"],
                    "reward": result["reward"],
                    "invalid": result["invalid"],
                    "id": result["id"],
                    **(result["params"] or {}),
                    **(result["metrics"] or {}),
                }
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

        valid_results = [res for res in results if not res["invalid"]]
        if not valid_results:
            print(f"gen={generation:03d} FATAL: Ningun individuo valido.")
            population = init_population(args, rng)
            continue

        generation_best = max(valid_results, key=lambda r: float(r["reward"]))
        if best_result is None or float(generation_best["reward"]) > float(best_result["reward"]):
            best_result = generation_best
            save_best(best_path, best_result, args)

        valid_count = len(valid_results)
        summary_row = {
            "generation": generation,
            "best_reward": float(generation_best["reward"]),
            "global_best_reward": float(best_result["reward"]) if best_result else float("nan"),
            "mean_reward": float(np.mean(rewards)),
            "valid_count": valid_count,
        }
        write_header = not summary_path.exists()
        with summary_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_row))
            if write_header:
                writer.writeheader()
            writer.writerow(summary_row)

        print(
            f"gen={generation:03d} best={summary_row['best_reward']:.4f} "
            f"global={summary_row['global_best_reward']:.4f} mean={summary_row['mean_reward']:.4f} "
            f"valid={valid_count}/{args.population}"
        )
        population = next_generation(population, rewards, args, rng)

    print(f"Mejor geometria (CFD): {best_path}")

if __name__ == "__main__":
    main()