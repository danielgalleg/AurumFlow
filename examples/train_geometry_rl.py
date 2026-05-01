from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from visual_sim.rl_env import GeometryOptimizationEnv, RLEvaluationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un agente RL para optimizar geometria del ciclon.")
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac", help="Algoritmo RL.")
    parser.add_argument("--timesteps", type=int, default=300, help="Evaluaciones/steps de entrenamiento.")
    parser.add_argument("--particles", type=int, default=1_600, help="Particulas por episodio de entrenamiento.")
    parser.add_argument("--frames", type=int, default=550, help="Frames simulados por episodio.")
    parser.add_argument("--substeps", type=int, default=3, help="Subpasos fisicos por frame.")
    parser.add_argument("--feed-duration", type=float, default=2.5, help="Segundos de alimentacion por episodio.")
    parser.add_argument("--seed", type=int, default=7, help="Semilla base.")
    parser.add_argument("--n-envs", type=int, default=1, help="Episodios paralelos en CPU.")
    parser.add_argument("--output-dir", default="rl_runs/latest", help="Directorio para modelo, CSV y mejor geometria.")
    return parser.parse_args()


def make_env(eval_config: RLEvaluationConfig, output_dir: Path, env_index: int):
    def _factory() -> GeometryOptimizationEnv:
        if env_index == 0:
            log_path = output_dir / "episodes.csv"
            best_path = output_dir / "best_geometry.json"
        else:
            env_dir = output_dir / f"env_{env_index:02d}"
            log_path = env_dir / "episodes.csv"
            best_path = env_dir / "best_geometry.json"
        return GeometryOptimizationEnv(
            eval_config=RLEvaluationConfig(
                particle_count=eval_config.particle_count,
                frames=eval_config.frames,
                substeps=eval_config.substeps,
                feed_duration_s=eval_config.feed_duration_s,
                seed=eval_config.seed + env_index,
            ),
            log_path=log_path,
            best_path=best_path,
        )

    return _factory


def build_model(algo: str, env, seed: int, n_envs: int):
    if algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            seed=seed,
            learning_starts=max(10, 2 * n_envs),
            batch_size=32,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=1,
        )
    return PPO(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=max(32, 64 // max(1, n_envs)),
        batch_size=32,
        verbose=1,
    )


def collect_best_geometry(output_dir: Path, n_envs: int) -> None:
    candidates = [output_dir / "best_geometry.json"]
    candidates.extend(output_dir / f"env_{idx:02d}" / "best_geometry.json" for idx in range(1, n_envs))
    best_payload = None
    best_path = None
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if best_payload is None or float(payload["reward"]) > float(best_payload["reward"]):
            best_payload = payload
            best_path = path
    if best_payload is None or best_path is None:
        return
    root_best = output_dir / "best_geometry.json"
    if best_path != root_best:
        shutil.copyfile(best_path, root_best)
    print(f"Mejor recompensa global: {float(best_payload['reward']):.4f}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_config = RLEvaluationConfig(
        particle_count=args.particles,
        frames=args.frames,
        substeps=args.substeps,
        feed_duration_s=args.feed_duration,
        seed=args.seed,
    )
    env_factories = [make_env(eval_config, output_dir, idx) for idx in range(max(1, args.n_envs))]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_factories, start_method="fork")
    else:
        env = DummyVecEnv(env_factories)
    model = build_model(args.algo, env, args.seed, max(1, args.n_envs))
    model.learn(total_timesteps=args.timesteps)
    model.save(output_dir / f"{args.algo}_geometry_agent")
    collect_best_geometry(output_dir, max(1, args.n_envs))
    env.close()
    print(f"Modelo guardado en {output_dir}")
    print(f"Mejor geometria: {output_dir / 'best_geometry.json'}")
    print(f"Log de episodios: {output_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()
