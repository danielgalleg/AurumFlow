from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_sim.metrics import SimulationMetrics
from visual_sim.rl_env import (
    ACTION_BOUNDS,
    ACTION_NAMES,
    RLEvaluationConfig,
    action_to_parameters,
    balanced_reward,
    evaluate_geometry,
    parameters_to_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimiza geometria con algoritmo genetico/evolutivo paralelo."
    )
    parser.add_argument("--generations", type=int, default=40, help="Generaciones evolutivas.")
    parser.add_argument("--population", type=int, default=64, help="Individuos por generacion.")
    parser.add_argument("--particles", type=int, default=1_600, help="Particulas por evaluacion.")
    parser.add_argument("--frames", type=int, default=550, help="Frames simulados por evaluacion.")
    parser.add_argument("--substeps", type=int, default=3, help="Subpasos fisicos por frame.")
    parser.add_argument("--feed-duration", type=float, default=2.5, help="Segundos de alimentacion.")
    parser.add_argument("--seed", type=int, default=7, help="Semilla base del optimizador.")
    parser.add_argument("--eval-seeds", default="0", help="Offsets de semillas separados por coma.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Evaluaciones paralelas en CPU.")
    parser.add_argument("--elite-fraction", type=float, default=0.15, help="Fraccion de elite conservada.")
    parser.add_argument("--mutation-prob", type=float, default=0.25, help="Probabilidad de mutar cada gen.")
    parser.add_argument("--mutation-sigma", type=float, default=0.18, help="Desviacion estandar de mutacion.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Individuos por torneo.")
    parser.add_argument("--warm-start-json", help="best_geometry.json previo para sembrar la poblacion.")
    parser.add_argument("--output-dir", default="rl_runs/ga_geometry", help="Directorio de salida.")
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


def average_metrics(metrics: list[SimulationMetrics]) -> SimulationMetrics:
    values: dict[str, float] = {}
    for key in asdict(metrics[0]):
        values[key] = float(np.mean([getattr(metric, key) for metric in metrics]))
    return SimulationMetrics(**values)


def evaluate_action(payload: tuple[int, int, list[float], dict[str, Any], list[int]]) -> dict[str, Any]:
    generation, individual, action_values, config_values, seed_offsets = payload
    action = np.asarray(action_values, dtype=np.float32)
    params = action_to_parameters(action)
    try:
        geometry = parameters_to_geometry(params)
        geometry.validate()
        metrics_by_seed: list[SimulationMetrics] = []
        rewards: list[float] = []
        for offset in seed_offsets:
            config = RLEvaluationConfig(**{**config_values, "seed": int(config_values["seed"]) + offset})
            metrics, _ = evaluate_geometry(geometry, config)
            metrics_by_seed.append(metrics)
            rewards.append(balanced_reward(metrics))
        metrics_avg = average_metrics(metrics_by_seed)
        reward = float(np.mean(rewards))
        error = ""
        invalid = False
    except ValueError as exc:
        geometry = None
        metrics_avg = None
        reward = -2.0
        error = str(exc)
        invalid = True

    return {
        "generation": generation,
        "individual": individual,
        "reward": reward,
        "invalid": invalid,
        "error": error,
        "action": action.tolist(),
        "params": params,
        "metrics": asdict(metrics_avg) if metrics_avg else None,
        "geometry": asdict(geometry) if geometry else None,
    }


def tournament_select(
    population: np.ndarray,
    rewards: np.ndarray,
    rng: np.random.Generator,
    tournament_size: int,
) -> np.ndarray:
    contenders = rng.integers(0, len(population), size=max(1, tournament_size))
    winner = contenders[np.argmax(rewards[contenders])]
    return population[winner].copy()


def make_child(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    mutation_prob: float,
    mutation_sigma: float,
) -> np.ndarray:
    mix = rng.uniform(0.0, 1.0, size=parent_a.shape).astype(np.float32)
    child = mix * parent_a + (1.0 - mix) * parent_b
    mutation_mask = rng.random(size=child.shape) < mutation_prob
    child[mutation_mask] += rng.normal(0.0, mutation_sigma, size=np.count_nonzero(mutation_mask))
    return np.clip(child, -1.0, 1.0).astype(np.float32)


def initial_population(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    population = rng.uniform(-1.0, 1.0, size=(args.population, len(ACTION_NAMES))).astype(np.float32)
    population[0] = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    if args.warm_start_json:
        payload = json.loads(Path(args.warm_start_json).read_text(encoding="utf-8"))
        population[1 % args.population] = params_to_action(payload.get("params", {}))
    return population


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "generation",
        "individual",
        "reward",
        "invalid",
        "error",
        *ACTION_NAMES,
        "active_count",
        "pending_count",
        "trapped_count",
        "overflow_count",
        "unprocessed_count",
        "target_recovery_pct",
        "target_loss_pct",
        "non_target_rejection_pct",
        "trapped_contamination_pct",
        "trapped_black_sand_pct",
        "unprocessed_pct",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for result in rows:
            metrics = result["metrics"] or {}
            row = {
                "generation": result["generation"],
                "individual": result["individual"],
                "reward": result["reward"],
                "invalid": result["invalid"],
                "error": result["error"],
                **result["params"],
                **metrics,
            }
            writer.writerow(row)


def save_best(path: Path, result: dict[str, Any], eval_config: RLEvaluationConfig, seed_offsets: list[int]) -> None:
    payload = {
        "reward": result["reward"],
        "action": result["action"],
        "params": result["params"],
        "geometry": result["geometry"],
        "metrics": result["metrics"],
        "eval_config": asdict(eval_config),
        "eval_seed_offsets": seed_offsets,
        "optimizer": "genetic",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def next_generation(
    population: np.ndarray,
    rewards: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    elite_count = max(1, int(round(args.population * args.elite_fraction)))
    elite_indices = np.argsort(rewards)[-elite_count:][::-1]
    children = [population[idx].copy() for idx in elite_indices]
    while len(children) < args.population:
        parent_a = tournament_select(population, rewards, rng, args.tournament_size)
        parent_b = tournament_select(population, rewards, rng, args.tournament_size)
        children.append(make_child(parent_a, parent_b, rng, args.mutation_prob, args.mutation_sigma))
    return np.asarray(children, dtype=np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_offsets = [int(seed.strip()) for seed in args.eval_seeds.split(",") if seed.strip()]
    if not seed_offsets:
        seed_offsets = [0]

    eval_config = RLEvaluationConfig(
        particle_count=args.particles,
        frames=args.frames,
        substeps=args.substeps,
        feed_duration_s=args.feed_duration,
        seed=args.seed,
    )
    config_values = asdict(eval_config)
    rng = np.random.default_rng(args.seed)
    population = initial_population(args, rng)
    best_result: dict[str, Any] | None = None

    episodes_path = output_dir / "episodes.csv"
    best_path = output_dir / "best_geometry.json"
    summary_path = output_dir / "generations.csv"
    for generation in range(args.generations):
        payloads = [
            (generation, individual, population[individual].tolist(), config_values, seed_offsets)
            for individual in range(args.population)
        ]
        with ProcessPoolExecutor(max_workers=max(1, args.n_jobs)) as executor:
            results = list(executor.map(evaluate_action, payloads))
        results.sort(key=lambda item: item["individual"])
        write_rows(episodes_path, results)

        rewards = np.asarray([float(result["reward"]) for result in results], dtype=np.float32)
        generation_best = max(results, key=lambda item: float(item["reward"]))
        if best_result is None or float(generation_best["reward"]) > float(best_result["reward"]):
            best_result = generation_best
            save_best(best_path, best_result, eval_config, seed_offsets)

        valid_count = sum(not result["invalid"] for result in results)
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

    print(f"Mejor geometria: {best_path}")
    print(f"Log de individuos: {episodes_path}")
    print(f"Resumen por generacion: {summary_path}")


if __name__ == "__main__":
    main()
