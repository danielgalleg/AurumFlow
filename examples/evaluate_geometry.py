from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_sim import ClassifierGeometry
from visual_sim.rl_env import RLEvaluationConfig, balanced_reward, evaluate_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua geometria baseline vs geometria RL.")
    parser.add_argument("--geometry-json", help="JSON generado por train_geometry_rl.py")
    parser.add_argument("--particles", type=int, default=8_000, help="Particulas por evaluacion.")
    parser.add_argument("--frames", type=int, default=1_500, help="Frames por evaluacion.")
    parser.add_argument("--substeps", type=int, default=4, help="Subpasos por frame.")
    parser.add_argument("--feed-duration", type=float, default=5.0, help="Segundos de alimentacion.")
    parser.add_argument("--seeds", default="7,13,23", help="Semillas separadas por coma.")
    parser.add_argument("--output", default="rl_runs/evaluation.csv", help="CSV de salida.")
    return parser.parse_args()


def load_geometry(path: str | None) -> ClassifierGeometry:
    if not path:
        return ClassifierGeometry()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClassifierGeometry(**payload["geometry"])


def evaluate_named_geometry(
    name: str,
    geometry: ClassifierGeometry,
    seeds: list[int],
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        config = RLEvaluationConfig(
            particle_count=args.particles,
            frames=args.frames,
            substeps=args.substeps,
            feed_duration_s=args.feed_duration,
            seed=seed,
        )
        metrics, _ = evaluate_geometry(geometry, config)
        rows.append(
            {
                "name": name,
                "seed": seed,
                "reward": balanced_reward(metrics),
                **asdict(metrics),
                **{f"geometry_{key}": value for key, value in asdict(geometry).items()},
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    rows = evaluate_named_geometry("baseline", ClassifierGeometry(), seeds, args)

    if args.geometry_json:
        rows.extend(evaluate_named_geometry("rl_candidate", load_geometry(args.geometry_json), seeds, args))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    summary = df.groupby("name", as_index=False).mean(numeric_only=True)
    print(summary.to_string(index=False))
    print(f"CSV guardado en {output}")


if __name__ == "__main__":
    main()
