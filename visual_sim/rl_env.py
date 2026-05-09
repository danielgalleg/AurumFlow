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


# NOTA: inlet_angle_deg fue ELIMINADO del action space porque la geometria es
# axisimetrica (de revolucion). Rotar el inlet alrededor del eje Y produce un
# resultado fisico identico salvo por una rotacion rigida del campo, asi que
# era un grado de libertad fantasma que solo le agregaba ruido al GA. El
# ClassifierGeometry mantiene el campo con default -90.0 para no romper los
# scripts que generan STL/OpenFOAM cases.
ACTION_NAMES = (
    "height_m",
    "neck_height_ratio",
    "neck_radius_m",
    "upper_max_radius_m",
    "upper_max_position_ratio",
    "lower_max_radius_m",
    "lower_max_position_ratio",
    "inlet_height_ratio",
    "inlet_pitch_deg",
    "inlet_yaw_deg",
    "central_tube_radius_m",
    "central_tube_bottom_ratio",
    "flow_velocity_m_s",
)

ACTION_BOUNDS = {
    "height_m": (0.10, 0.40),
    "neck_height_ratio": (0.10, 0.90),
    "neck_radius_m": (0.002, 0.050),
    "upper_max_radius_m": (0.025, 0.120),
    "upper_max_position_ratio": (0.10, 0.90),
    "lower_max_radius_m": (0.025, 0.120),
    "lower_max_position_ratio": (0.10, 0.90),
    "inlet_height_ratio": (0.05, 0.95),
    "inlet_pitch_deg": (-89.5, 89.5),
    "inlet_yaw_deg": (0.0, 89.5),
    # Tubo central minimo 3mm: tubos sub-3mm rompen el mesh (cell-size local
    # demasiado fino) y no son manufacturables con tolerancia razonable.
    "central_tube_radius_m": (0.003, 0.040),
    "central_tube_bottom_ratio": (0.01, 0.99),
    # Velocidades >2 m/s generan Reynolds turbulento que simpleFoam laminar
    # no resuelve confiable; el campeon usa 0.34 m/s y el optimo tipico de
    # elutriacion esta en 0.1-1.0 m/s. Limitar a 2.0 ahorra exploracion inutil.
    "flow_velocity_m_s": (0.10, 2.00),
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
    """Convierte un dict de parametros normalizados (del GA) en una ClassifierGeometry
    valida del tipo Clepsamia (reloj de arena con 2 lobulos y tubo central).

    Algunos parametros se ajustan con limites duros para garantizar geometrias fisicas:
    - El radio del cuello debe ser menor que ambos radios maximos.
    - El radio del tubo central debe ser menor que el radio del cuello (con margen).
    - La altura inferior del tubo debe estar dentro del dispositivo.
    """
    height = float(params["height_m"])
    neck_height_ratio = float(params["neck_height_ratio"])
    neck_radius = float(params["neck_radius_m"])
    upper_max_r = float(params["upper_max_radius_m"])
    lower_max_r = float(params["lower_max_radius_m"])
    central_tube_radius = float(params["central_tube_radius_m"])

    # Forzar que los radios maximos siempre sean mayores que el cuello (con margen).
    upper_max_r = max(upper_max_r, neck_radius * 1.5)
    lower_max_r = max(lower_max_r, neck_radius * 1.5)

    # Forzar que el tubo central quepa por el cuello con margen.
    central_tube_radius = min(central_tube_radius, neck_radius * 0.7)
    # Pero el tubo central tampoco puede ser tan pequeno que sea cero.
    central_tube_radius = max(central_tube_radius, 0.002)

    # inlet_angle_deg ya no es parte del action space (axisimetria). Si el dict
    # de params lo trae igual (warm-start desde un JSON viejo), lo respetamos;
    # si no, usamos el default de la dataclass (-90.0).
    inlet_angle_deg = float(params.get("inlet_angle_deg", -90.0))

    return ClassifierGeometry(
        height_m=height,
        neck_height_ratio=neck_height_ratio,
        neck_radius_m=neck_radius,
        upper_max_radius_m=upper_max_r,
        upper_max_position_ratio=float(params["upper_max_position_ratio"]),
        lower_max_radius_m=lower_max_r,
        lower_max_position_ratio=float(params["lower_max_position_ratio"]),
        inlet_height_ratio=float(params["inlet_height_ratio"]),
        inlet_pitch_deg=float(params["inlet_pitch_deg"]),
        inlet_angle_deg=inlet_angle_deg,
        inlet_yaw_deg=float(params["inlet_yaw_deg"]),
        central_tube_radius_m=central_tube_radius,
        central_tube_bottom_ratio=float(params["central_tube_bottom_ratio"]),
        inlet_velocity_m_s=float(params["flow_velocity_m_s"]),
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
