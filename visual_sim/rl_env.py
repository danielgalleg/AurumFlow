from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .geometry import ClassifierGeometry
from .metrics import SimulationMetrics, compute_metrics
from .physics import ClassifierSimulation, SimulationConfig


ACTION_NAMES = (
    "height_m",
    "cyclone_top_radius_m",
    "body_wall_angle_deg",
    "body_curve",
    "cone_curve",
    "body_wall_length_ratio",
    "cone_neck_radius_ratio",
    "trap_wall_length_ratio",
    "trap_wall_angle_deg",
    "trap_curve",
    "trap_floor_curve",
    "overflow_tube_radius_ratio",
    "overflow_tube_bottom_height_ratio",
    "inlet_height_ratio",
    "flow_velocity_m_s",
)

ACTION_BOUNDS = {
    "height_m": (0.26, 0.50),
    "cyclone_top_radius_m": (0.045, 0.085),
    "body_wall_angle_deg": (45.0, 135.0),
    "body_curve": (-0.45, 0.45),
    "cone_curve": (-0.45, 0.45),
    "body_wall_length_ratio": (0.45, 0.72),
    "cone_neck_radius_ratio": (0.14, 0.36),
    "trap_wall_length_ratio": (0.08, 0.30),
    "trap_wall_angle_deg": (45.0, 135.0),
    "trap_curve": (-0.45, 0.45),
    "trap_floor_curve": (-0.45, 0.45),
    "overflow_tube_radius_ratio": (0.06, 0.18),
    "overflow_tube_bottom_height_ratio": (0.38, 0.70),
    "inlet_height_ratio": (0.58, 0.82),
    "flow_velocity_m_s": (0.035, 0.12),
}


@dataclass(frozen=True)
class RLEvaluationConfig:
    particle_count: int = 1_600
    frames: int = 550
    substeps: int = 3
    feed_duration_s: float = 2.5
    seed: int = 7


def action_to_parameters(action: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if clipped.size != len(ACTION_NAMES):
        raise ValueError(f"La accion debe tener {len(ACTION_NAMES)} valores")
    params: dict[str, float] = {}
    for value, name in zip(clipped, ACTION_NAMES):
        low, high = ACTION_BOUNDS[name]
        params[name] = float(low + (0.5 * (float(value) + 1.0) * (high - low)))
    return params


def parameters_to_geometry(params: dict[str, float]) -> ClassifierGeometry:
    height = params["height_m"]
    top_radius = params["cyclone_top_radius_m"]
    body_wall_length_ratio = params.get("body_wall_length_ratio", 1.0 - params.get("cone_top_height_ratio", 0.39))
    trap_wall_length_ratio = params.get("trap_wall_length_ratio", params.get("trap_height_ratio", 0.185))
    cone_top_height = (1.0 - body_wall_length_ratio) * height
    min_cone_length = 0.07 * height
    trap_height = min(trap_wall_length_ratio * height, cone_top_height - min_cone_length)
    trap_height = max(0.035 * height, trap_height)
    inlet_height = max(params["inlet_height_ratio"] * height, cone_top_height + 0.04 * height)
    inlet_height = min(0.94 * height, inlet_height)
    outlet_height = min(height, max(inlet_height + 0.03 * height, 0.88 * height))
    flow_velocity = params["flow_velocity_m_s"]
    body_drop = height - cone_top_height
    min_body_bottom_radius = 0.005 * top_radius
    body_slope = np.tan(np.deg2rad(90.0 - params["body_wall_angle_deg"]))
    body_bottom_radius = max(min_body_bottom_radius, top_radius - body_slope * body_drop)
    cone_neck_radius = top_radius * params["cone_neck_radius_ratio"]
    trap_slope = np.tan(np.deg2rad(90.0 - params["trap_wall_angle_deg"]))
    min_trap_bottom_radius = 0.05 * cone_neck_radius
    trap_bottom_radius = max(min_trap_bottom_radius, cone_neck_radius - trap_slope * trap_height)
    body_required_radius = max(top_radius, body_bottom_radius)
    if params["body_curve"] > 0.0:
        for t in np.linspace(0.0, 1.0, 17):
            linear_radius = body_bottom_radius + (top_radius - body_bottom_radius) * float(t)
            curve_fraction = params["body_curve"] * 4.0 * float(t) * (1.0 - float(t))
            body_required_radius = max(body_required_radius, linear_radius / max(1e-6, 1.0 - curve_fraction))
    cone_required_radius = max(cone_neck_radius, body_bottom_radius)
    for t in np.linspace(0.0, 1.0, 17):
        linear_radius = cone_neck_radius + (body_bottom_radius - cone_neck_radius) * float(t)
        curve_radius = params["cone_curve"] * max(cone_neck_radius, body_bottom_radius) * 4.0 * float(t) * (
            1.0 - float(t)
        )
        cone_required_radius = max(cone_required_radius, linear_radius + curve_radius)
    trap_required_radius = max(cone_neck_radius, trap_bottom_radius)
    for t in np.linspace(0.0, 1.0, 17):
        linear_radius = trap_bottom_radius + (cone_neck_radius - trap_bottom_radius) * float(t)
        curve_radius = params["trap_curve"] * cone_neck_radius * 4.0 * float(t) * (1.0 - float(t))
        trap_required_radius = max(trap_required_radius, linear_radius + curve_radius)
    max_radius = max(
        top_radius,
        body_required_radius,
        cone_required_radius,
        trap_required_radius,
    )
    body_top_radius_ratio = top_radius / max(max_radius, 1e-6)
    body_bottom_radius_ratio = body_bottom_radius / max(max_radius, 1e-6)
    cone_neck_radius_ratio = cone_neck_radius / max(max_radius, 1e-6)
    overflow_tube_radius_ratio = (top_radius * params["overflow_tube_radius_ratio"]) / max(max_radius, 1e-6)
    trap_bottom_radius_ratio = trap_bottom_radius / max(cone_neck_radius, 1e-6)
    return ClassifierGeometry(
        width_m=2.0 * max_radius,
        depth_m=2.0 * max_radius,
        height_m=height,
        trap_height_m=trap_height,
        outlet_height_m=outlet_height,
        inlet_height_m=inlet_height,
        inlet_velocity_m_s=flow_velocity,
        upward_velocity_m_s=flow_velocity,
        body_top_radius_ratio=float(body_top_radius_ratio),
        body_bottom_radius_ratio=float(body_bottom_radius_ratio),
        overflow_tube_radius_ratio=float(overflow_tube_radius_ratio),
        overflow_tube_bottom_height_ratio=params["overflow_tube_bottom_height_ratio"],
        cone_top_height_ratio=float(cone_top_height / height),
        cone_neck_radius_ratio=float(cone_neck_radius_ratio),
        trap_bottom_radius_ratio=float(trap_bottom_radius_ratio),
        body_curve=params["body_curve"],
        cone_curve=params["cone_curve"],
        trap_curve=params["trap_curve"],
        trap_floor_curve=params["trap_floor_curve"],
    )


def balanced_reward(metrics: SimulationMetrics) -> float:
    gold_recovery = metrics.target_recovery_pct / 100.0
    gold_loss = metrics.target_loss_pct / 100.0
    non_target_rejection = metrics.non_target_rejection_pct / 100.0
    contamination = metrics.trapped_contamination_pct / 100.0
    unprocessed = metrics.unprocessed_pct / 100.0
    return float(
        1.8 * gold_recovery
        + 1.5 * non_target_rejection
        - 3.0 * gold_loss
        - 2.4 * contamination
        - 0.6 * unprocessed
    )


def evaluate_geometry(
    geometry: ClassifierGeometry,
    config: RLEvaluationConfig,
) -> tuple[SimulationMetrics, ClassifierSimulation]:
    sim_config = SimulationConfig(
        particle_count=config.particle_count,
        seed=config.seed,
        feed_duration_s=config.feed_duration_s,
    )
    sim = ClassifierSimulation(geometry=geometry, config=sim_config)
    sim.step(config.frames * config.substeps)
    return compute_metrics(sim.status, sim.material_ids, sim.materials), sim


class GeometryOptimizationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        eval_config: RLEvaluationConfig | None = None,
        log_path: str | Path | None = None,
        best_path: str | Path | None = None,
    ) -> None:
        self.eval_config = eval_config or RLEvaluationConfig()
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(ACTION_NAMES),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(ACTION_NAMES),),
            dtype=np.float32,
        )
        self.log_path = Path(log_path) if log_path else None
        self.best_path = Path(best_path) if best_path else None
        self.best_reward = -np.inf
        self.episode_index = 0
        self._last_observation = np.zeros(len(ACTION_NAMES), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.eval_config = RLEvaluationConfig(**{**asdict(self.eval_config), "seed": int(seed)})
        self._last_observation = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        return self._last_observation.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        params = action_to_parameters(action)
        try:
            geometry = parameters_to_geometry(params)
            geometry.validate()
            metrics, _ = evaluate_geometry(geometry, self.eval_config)
            reward = balanced_reward(metrics)
            invalid = False
        except ValueError as exc:
            geometry = None
            metrics = None
            reward = -2.0
            invalid = True
            error = str(exc)
        else:
            error = ""

        observation = np.asarray(action, dtype=np.float32).copy()
        info: dict[str, Any] = {
            "episode_index": self.episode_index,
            "reward": reward,
            "invalid": invalid,
            "error": error,
            "params": params,
        }
        if metrics is not None:
            info["metrics"] = asdict(metrics)
            self._log_result(params, metrics, reward)
        if geometry is not None and reward > self.best_reward:
            self.best_reward = reward
            self._save_best(params, geometry, metrics, reward)

        self.episode_index += 1
        self._last_observation = observation
        return observation, reward, True, False, info

    def _log_result(self, params: dict[str, float], metrics: SimulationMetrics, reward: float) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "episode": self.episode_index,
            "reward": reward,
            **params,
            **asdict(metrics),
        }
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_best(
        self,
        params: dict[str, float],
        geometry: ClassifierGeometry,
        metrics: SimulationMetrics | None,
        reward: float,
    ) -> None:
        if self.best_path is None:
            return
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reward": reward,
            "params": params,
            "geometry": asdict(geometry),
            "metrics": asdict(metrics) if metrics else None,
            "eval_config": asdict(self.eval_config),
        }
        self.best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
