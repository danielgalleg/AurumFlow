from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ClassifierGeometry
from .materials import MaterialPreset, default_materials, normalize_fractions


WATER_DENSITY_KG_M3 = 1_000.0
WATER_VISCOSITY_PA_S = 0.001
GRAVITY_M_S2 = 9.80665


def _allowed_radius_for_y(y: np.ndarray, g: ClassifierGeometry) -> np.ndarray:
    y = np.asarray(y)
    radius = np.full_like(y, g.cylinder_radius_m, dtype=np.float32)

    in_trap = y < g.trap_height_m
    if np.any(in_trap):
        t = np.clip(y[in_trap] / max(1e-6, g.trap_height_m), 0.0, 1.0)
        linear_radius = g.trap_bottom_half_depth_m + (
            g.cone_neck_half_depth_m - g.trap_bottom_half_depth_m
        ) * t
        curve_radius = g.trap_curve * g.cone_neck_half_depth_m * 4.0 * t * (1.0 - t)
        radius[in_trap] = np.maximum(1e-6, linear_radius + curve_radius)

    in_cone = (y >= g.trap_height_m) & (y < g.cone_top_height_m)
    if np.any(in_cone):
        t = (y[in_cone] - g.trap_height_m) / max(1e-6, g.cone_top_height_m - g.trap_height_m)
        linear_radius = g.cone_neck_half_depth_m + (
            g.body_bottom_radius_m - g.cone_neck_half_depth_m
        ) * t
        curve_scale = max(g.body_bottom_radius_m, g.cone_neck_half_depth_m)
        curve_radius = g.cone_curve * curve_scale * 4.0 * t * (1.0 - t)
        radius[in_cone] = np.maximum(1e-6, linear_radius + curve_radius)

    in_body = y >= g.cone_top_height_m
    if np.any(in_body):
        t = (y[in_body] - g.cone_top_height_m) / max(1e-6, g.height_m - g.cone_top_height_m)
        t = np.clip(t, 0.0, 1.0)
        linear_radius = g.body_bottom_radius_m + (g.body_top_radius_m - g.body_bottom_radius_m) * t
        curve_radius = g.body_curve * g.cylinder_radius_m * 4.0 * t * (1.0 - t)
        radius[in_body] = np.maximum(1e-6, linear_radius + curve_radius)

    return radius


@dataclass(frozen=True)
class SimulationConfig:
    particle_count: int = 8_000
    dt_s: float = 0.004
    seed: int = 7
    max_particle_radius_m: float = 0.0024
    velocity_relaxation_s: float = 0.08
    settling_scale: float = 0.032
    feed_duration_s: float = 5.0


class ClassifierSimulation:
    """Simulacion 3D simplificada para comparar geometria y parametros."""

    def __init__(
        self,
        geometry: ClassifierGeometry | None = None,
        materials: tuple[MaterialPreset, ...] | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.geometry = geometry or ClassifierGeometry()
        self.geometry.validate()
        self.materials = materials or default_materials()
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.positions = np.zeros((self.config.particle_count, 3), dtype=np.float32)
        self.velocities = np.zeros_like(self.positions)
        self.material_ids = np.zeros(self.config.particle_count, dtype=np.int32)
        self.status = np.zeros(self.config.particle_count, dtype=np.int32)
        self.colors = np.zeros((self.config.particle_count, 3), dtype=np.float32)
        self.visual_radius = np.zeros(self.config.particle_count, dtype=np.float32)
        self._diameter_m = np.zeros(self.config.particle_count, dtype=np.float32)
        self._density_kg_m3 = np.zeros(self.config.particle_count, dtype=np.float32)
        self._shape_factor = np.zeros(self.config.particle_count, dtype=np.float32)
        self._swirl_bias = np.zeros(self.config.particle_count, dtype=np.float32)
        self._inside_overflow_tube = np.zeros(self.config.particle_count, dtype=bool)
        self._released_count = 0
        self.step_index = 0

        self.reset()

    def reset(self) -> None:
        fractions = normalize_fractions(self.materials)
        material_indices = self.rng.choice(
            len(self.materials),
            size=self.config.particle_count,
            p=np.array(fractions),
        )
        g = self.geometry

        pipe_progress = np.linspace(0.0, 1.0, self.config.particle_count, dtype=np.float32)
        self.positions[:, 0] = self.rng.normal(
            0.70 * g.cylinder_radius_m,
            0.08 * g.cylinder_radius_m,
            size=self.config.particle_count,
        )
        self.positions[:, 1] = self.rng.uniform(
            g.inlet_height_m - 0.012,
            g.inlet_height_m + 0.012,
            size=self.config.particle_count,
        )
        self.positions[:, 2] = -g.cylinder_radius_m * (1.55 - 0.50 * pipe_progress)
        self.velocities[:, :] = 0.0
        self.status[:] = -1
        self._inside_overflow_tube[:] = False
        self.material_ids[:] = material_indices

        for idx, material_index in enumerate(material_indices):
            material = self.materials[int(material_index)]
            size_jitter = float(self.rng.lognormal(mean=0.0, sigma=0.22))
            self.colors[idx] = material.color_rgb
            self._diameter_m[idx] = material.diameter_m * size_jitter
            self._density_kg_m3[idx] = material.density_kg_m3
            self._shape_factor[idx] = material.shape_factor
            self._swirl_bias[idx] = self.rng.uniform(-1.0, 1.0)
            visual = self.config.max_particle_radius_m * (0.55 + 0.55 * min(1.0, size_jitter))
            self.visual_radius[idx] = visual

        self.step_index = 0
        self._released_count = 0

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._step_once()

    def active_positions(self) -> np.ndarray:
        return self.positions[self.status == 0]

    def active_colors(self) -> np.ndarray:
        return self.colors[self.status == 0]

    def _release_feed_particles(self) -> None:
        if self.config.feed_duration_s <= 0.0:
            target_released = self.config.particle_count
        else:
            elapsed_s = (self.step_index + 1) * self.config.dt_s
            fraction = min(1.0, elapsed_s / self.config.feed_duration_s)
            target_released = int(np.floor(fraction * self.config.particle_count))

        release_count = max(0, target_released - self._released_count)
        if release_count <= 0:
            return

        g = self.geometry
        start = self._released_count
        stop = min(self.config.particle_count, start + release_count)
        indices = np.arange(start, stop)

        self.positions[indices, 0] = self.rng.normal(
            0.70 * g.cylinder_radius_m,
            0.045 * g.cylinder_radius_m,
            size=indices.size,
        )
        self.positions[indices, 1] = self.rng.normal(
            g.inlet_height_m,
            0.010,
            size=indices.size,
        )
        self.positions[indices, 2] = -0.97 * g.cylinder_radius_m

        radial = np.sqrt(self.positions[indices, 0] ** 2 + self.positions[indices, 2] ** 2)
        self.velocities[indices, :] = 0.0
        self.velocities[indices, 0] = (
            -self.positions[indices, 2] / np.maximum(radial, 1e-6) * g.inlet_velocity_m_s
        )
        self.velocities[indices, 2] = (
            self.positions[indices, 0] / np.maximum(radial, 1e-6) * g.inlet_velocity_m_s
        )
        self.status[indices] = 0
        self._released_count = stop

    def _step_once(self) -> None:
        self._release_feed_particles()
        active = self.status == 0
        if not np.any(active):
            self.step_index += 1
            return

        active_indices = np.flatnonzero(active)
        pos = self.positions[active]
        vel = self.velocities[active]
        previous_pos = pos.copy()
        density = self._density_kg_m3[active]
        diameter = self._diameter_m[active]
        shape = self._shape_factor[active]

        fluid = self._fluid_velocity(pos, self._swirl_bias[active])
        settling = self._settling_velocity(density, diameter, shape)
        density_response = (WATER_DENSITY_KG_M3 / np.maximum(density, 1.0)) ** 0.50
        size_response = (150e-6 / np.maximum(diameter, 1e-9)) ** 0.35
        flow_response = np.clip(1.2 * density_response * size_response, 0.10, 1.0)
        target_velocity = fluid * flow_response[:, None]
        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        radial_x = np.divide(pos[:, 0], np.maximum(radial, 1e-6))
        radial_z = np.divide(pos[:, 2], np.maximum(radial, 1e-6))
        tangential_speed = np.abs(
            fluid[:, 0] * (-radial_z) + fluid[:, 2] * radial_x
        )
        inertia = np.clip(
            ((density - WATER_DENSITY_KG_M3) / WATER_DENSITY_KG_M3) ** 1.6
            * (diameter / 200e-6) ** 0.5,
            0.0,
            18.0,
        )
        centrifugal_slip = np.clip(
            0.0038 * inertia * (tangential_speed / 0.12) ** 2,
            0.0,
            0.055,
        )
        target_velocity[:, 0] += centrifugal_slip * radial_x
        target_velocity[:, 2] += centrifugal_slip * radial_z
        target_velocity[:, 1] -= settling

        alpha = min(1.0, self.config.dt_s / max(1e-4, self.config.velocity_relaxation_s))
        noise = self.rng.normal(0.0, self.geometry.turbulence, size=vel.shape).astype(np.float32)
        noise[:, 0] += self.rng.normal(0.0, self.geometry.turbulence * 4.0, size=vel.shape[0])
        vel += (target_velocity - vel) * alpha + noise * np.sqrt(self.config.dt_s)
        pos += vel * self.config.dt_s

        self._update_overflow_tube_membership(active_indices, previous_pos, pos)
        self._collide_with_walls(pos, vel, active_indices)
        self.positions[active] = pos
        self.velocities[active] = vel
        self._update_status(active)
        self.step_index += 1

    def _settling_velocity(
        self,
        density: np.ndarray,
        diameter: np.ndarray,
        shape: np.ndarray,
    ) -> np.ndarray:
        density_delta = np.maximum(0.0, density - WATER_DENSITY_KG_M3)
        stokes = density_delta * GRAVITY_M_S2 * diameter**2 / (18.0 * WATER_VISCOSITY_PA_S)
        return stokes * np.clip(shape, 0.15, 1.5) * self.config.settling_scale

    def _fluid_velocity(self, pos: np.ndarray, swirl_bias: np.ndarray) -> np.ndarray:
        g = self.geometry
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        y_norm = np.clip(y / g.height_m, 0.0, 1.0)
        inlet_zone = np.exp(-((y - g.inlet_height_m) ** 2) / 0.0025)
        cone_zone = np.clip((g.trap_height_m * 1.8 - y) / max(1e-6, g.trap_height_m * 1.8), 0.0, 1.0)
        radial = np.sqrt(x**2 + z**2)
        safe_radial = np.maximum(radial, 1e-6)
        local_radius = _allowed_radius_for_y(y, g)
        radius_norm = np.clip(radial / np.maximum(1e-6, local_radius), 0.0, 1.0)
        tangent_x = -z / safe_radial
        tangent_z = x / safe_radial
        radial_x = x / safe_radial
        radial_z = z / safe_radial

        angular_speed = (
            (0.13 + 0.12 * radius_norm)
            * np.sin(np.pi * y_norm)
            * g.deflector_strength
            + 0.07 * inlet_zone
        )
        tube_mouth_center = g.overflow_tube_bottom_height_m - 0.06 * g.height_m
        tube_mouth_zone = np.exp(
            -((y - tube_mouth_center) ** 2)
            / max(1e-6, (0.16 * g.height_m) ** 2)
        )
        axial_reversal_zone = 0.35 + 0.65 * np.clip(
            (y - 0.35 * g.trap_height_m)
            / max(1e-6, g.overflow_tube_bottom_height_m - 0.35 * g.trap_height_m),
            0.0,
            1.0,
        )
        air_core = np.exp(
            -((radial / max(1e-6, 1.25 * g.overflow_tube_radius_m)) ** 4)
        )
        vortex_finder_capture = np.exp(
            -((radial / max(1e-6, 1.80 * g.overflow_tube_radius_m)) ** 4)
        )
        radial_pressure_feed = np.exp(
            -((radial / max(1e-6, 2.60 * g.overflow_tube_radius_m)) ** 4)
        )
        tube_suction = 3.0 * g.upward_velocity_m_s * tube_mouth_zone * vortex_finder_capture
        inward_speed = 0.002 + 0.020 * cone_zone + 0.045 * tube_mouth_zone * radial_pressure_feed
        stream_spread = 0.025 * swirl_bias * np.sin(np.pi * y_norm)
        outer_circulation = min(g.upward_velocity_m_s, 0.045)
        outer_downflow = -outer_circulation * (0.55 + 1.35 * radius_norm) * (1.0 - 0.96 * air_core)
        inner_upflow = 7.50 * g.upward_velocity_m_s * air_core * axial_reversal_zone

        fluid = np.zeros_like(pos)
        fluid[:, 0] = angular_speed * tangent_x - inward_speed * radial_x + stream_spread
        fluid[:, 1] = outer_downflow + inner_upflow + tube_suction
        fluid[:, 2] = angular_speed * tangent_z - inward_speed * radial_z
        return fluid

    def _collide_with_walls(self, pos: np.ndarray, vel: np.ndarray, active_indices: np.ndarray) -> None:
        g = self.geometry
        damping = -0.22

        low_x = pos[:, 0] < -g.half_width_m
        high_x = pos[:, 0] > g.half_width_m
        pos[low_x, 0] = -g.half_width_m
        pos[high_x, 0] = g.half_width_m
        vel[low_x | high_x, 0] *= damping

        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        in_overflow_tube = self._inside_overflow_tube[active_indices]
        low_y = pos[:, 1] < 0.0
        high_y = (pos[:, 1] >= g.height_m - 1e-6) & ~in_overflow_tube
        pos[low_y, 1] = 0.0
        pos[high_y, 1] = g.height_m - 1e-3
        vel[low_y | high_y, 1] *= damping

        low_z = pos[:, 2] < -g.half_depth_m
        high_z = pos[:, 2] > g.half_depth_m
        pos[low_z, 2] = -g.half_depth_m
        pos[high_z, 2] = g.half_depth_m - 1e-4
        vel[low_z, 2] *= damping
        vel[high_z, 2] = -np.maximum(np.abs(vel[high_z, 2]) * 0.35, 0.025)

        self._collide_with_cylinder(pos, vel)
        self._collide_with_overflow_tube(pos, vel, active_indices)
        self._collide_with_hopper(pos, vel)
        self._collide_with_trap_floor(pos, vel)

    def _update_overflow_tube_membership(
        self,
        active_indices: np.ndarray,
        previous_pos: np.ndarray,
        pos: np.ndarray,
    ) -> None:
        g = self.geometry
        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        crossed_mouth = (
            (previous_pos[:, 1] < g.overflow_tube_bottom_height_m)
            & (pos[:, 1] >= g.overflow_tube_bottom_height_m)
            & (radial <= g.overflow_tube_radius_m)
        )
        self._inside_overflow_tube[active_indices[crossed_mouth]] = True

    def _collide_with_overflow_tube(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        active_indices: np.ndarray,
    ) -> None:
        g = self.geometry
        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        tube_zone = pos[:, 1] >= g.overflow_tube_bottom_height_m
        inside_tube = self._inside_overflow_tube[active_indices]

        normal_x = np.divide(pos[:, 0], np.maximum(radial, 1e-6))
        normal_z = np.divide(pos[:, 2], np.maximum(radial, 1e-6))
        near_axis = radial < 1e-6
        normal_x[near_axis] = 1.0
        normal_z[near_axis] = 0.0

        blocked_by_outer_pipe_wall = tube_zone & ~inside_tube & (radial < g.overflow_tube_radius_m)

        outer_pipe_shell = (
            tube_zone
            & ~inside_tube
            & (radial >= g.overflow_tube_radius_m)
            & (radial < g.overflow_tube_radius_m * 1.65)
        )
        if np.any(outer_pipe_shell):
            vel[outer_pipe_shell, 1] = np.minimum(vel[outer_pipe_shell, 1], -0.025)

        if np.any(blocked_by_outer_pipe_wall):
            pos[blocked_by_outer_pipe_wall, 0] = normal_x[blocked_by_outer_pipe_wall] * (
                g.overflow_tube_radius_m + 1e-4
            )
            pos[blocked_by_outer_pipe_wall, 2] = normal_z[blocked_by_outer_pipe_wall] * (
                g.overflow_tube_radius_m + 1e-4
            )
            radial_velocity = (
                vel[blocked_by_outer_pipe_wall, 0] * normal_x[blocked_by_outer_pipe_wall]
                + vel[blocked_by_outer_pipe_wall, 2] * normal_z[blocked_by_outer_pipe_wall]
            )
            vel[blocked_by_outer_pipe_wall, 0] -= 1.4 * radial_velocity * normal_x[
                blocked_by_outer_pipe_wall
            ]
            vel[blocked_by_outer_pipe_wall, 2] -= 1.4 * radial_velocity * normal_z[
                blocked_by_outer_pipe_wall
            ]
            vel[blocked_by_outer_pipe_wall, 1] = np.minimum(
                vel[blocked_by_outer_pipe_wall, 1], -0.035
            )

        hit_inner_pipe_wall = tube_zone & inside_tube & (radial > g.overflow_tube_radius_m)
        if np.any(hit_inner_pipe_wall):
            pos[hit_inner_pipe_wall, 0] = normal_x[hit_inner_pipe_wall] * (
                g.overflow_tube_radius_m - 1e-4
            )
            pos[hit_inner_pipe_wall, 2] = normal_z[hit_inner_pipe_wall] * (
                g.overflow_tube_radius_m - 1e-4
            )
            radial_velocity = (
                vel[hit_inner_pipe_wall, 0] * normal_x[hit_inner_pipe_wall]
                + vel[hit_inner_pipe_wall, 2] * normal_z[hit_inner_pipe_wall]
            )
            vel[hit_inner_pipe_wall, 0] -= 1.2 * radial_velocity * normal_x[hit_inner_pipe_wall]
            vel[hit_inner_pipe_wall, 2] -= 1.2 * radial_velocity * normal_z[hit_inner_pipe_wall]

        inside_tube = self._inside_overflow_tube[active_indices]
        if np.any(inside_tube):
            tube_indices = active_indices[inside_tube]
            density = self._density_kg_m3[tube_indices]
            diameter = self._diameter_m[tube_indices]
            density_response = (WATER_DENSITY_KG_M3 / np.maximum(density, 1.0)) ** 0.50
            size_response = (150e-6 / np.maximum(diameter, 1e-9)) ** 0.35
            tube_response = np.clip(1.2 * density_response * size_response, 0.10, 1.0)
            tube_upflow = 4.0 * g.upward_velocity_m_s * tube_response
            vel[inside_tube, 1] = np.maximum(vel[inside_tube, 1], tube_upflow)

    def _collide_with_cylinder(self, pos: np.ndarray, vel: np.ndarray) -> None:
        g = self.geometry
        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        radius = _allowed_radius_for_y(pos[:, 1], g)
        outside = radial > radius
        if not np.any(outside):
            return

        normal_x = pos[outside, 0] / np.maximum(radial[outside], 1e-6)
        normal_z = pos[outside, 2] / np.maximum(radial[outside], 1e-6)
        pos[outside, 0] = normal_x * radius[outside]
        pos[outside, 2] = normal_z * radius[outside]

        normal_velocity = vel[outside, 0] * normal_x + vel[outside, 2] * normal_z
        vel[outside, 0] -= 1.25 * normal_velocity * normal_x
        vel[outside, 2] -= 1.25 * normal_velocity * normal_z

    def _collide_with_hopper(self, pos: np.ndarray, vel: np.ndarray) -> None:
        g = self.geometry
        cone_top_y = g.cone_top_height_m
        neck_half_depth = g.cone_neck_half_depth_m
        trap_bottom_half_depth = g.trap_bottom_half_depth_m

        in_cone = (pos[:, 1] < cone_top_y) & (pos[:, 1] >= g.trap_height_m)
        if np.any(in_cone):
            allowed = _allowed_radius_for_y(pos[in_cone, 1], g)
            radial = np.sqrt(pos[in_cone, 0] ** 2 + pos[in_cone, 2] ** 2)
            outside = radial > allowed
            cone_indices = np.flatnonzero(in_cone)
            hit_indices = cone_indices[outside]
            if hit_indices.size:
                hit_radial = np.maximum(radial[outside], 1e-6)
                normal_x = pos[hit_indices, 0] / hit_radial
                normal_z = pos[hit_indices, 2] / hit_radial
                pos[hit_indices, 0] = normal_x * allowed[outside]
                pos[hit_indices, 2] = normal_z * allowed[outside]
                radial_velocity = vel[hit_indices, 0] * normal_x + vel[hit_indices, 2] * normal_z
                vel[hit_indices, 0] -= 1.18 * radial_velocity * normal_x
                vel[hit_indices, 2] -= 1.18 * radial_velocity * normal_z
                vel[hit_indices, 1] = np.maximum(vel[hit_indices, 1], -0.015)

        in_trap_tube = pos[:, 1] < g.trap_height_m
        if np.any(in_trap_tube):
            allowed = _allowed_radius_for_y(pos[in_trap_tube, 1], g)
            radial = np.sqrt(pos[in_trap_tube, 0] ** 2 + pos[in_trap_tube, 2] ** 2)
            outside = radial > allowed
            tube_indices = np.flatnonzero(in_trap_tube)
            hit_indices = tube_indices[outside]
            if hit_indices.size:
                hit_radial = np.maximum(radial[outside], 1e-6)
                normal_x = pos[hit_indices, 0] / hit_radial
                normal_z = pos[hit_indices, 2] / hit_radial
                pos[hit_indices, 0] = normal_x * allowed[outside]
                pos[hit_indices, 2] = normal_z * allowed[outside]
                radial_velocity = vel[hit_indices, 0] * normal_x + vel[hit_indices, 2] * normal_z
                vel[hit_indices, 0] -= 1.12 * radial_velocity * normal_x
                vel[hit_indices, 2] -= 1.12 * radial_velocity * normal_z

    def _collide_with_trap_floor(self, pos: np.ndarray, vel: np.ndarray) -> None:
        g = self.geometry
        if abs(g.trap_floor_curve) < 1e-9:
            return
        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        in_trap_floor = (pos[:, 1] < g.trap_height_m) & (radial <= g.trap_bottom_half_depth_m)
        if not np.any(in_trap_floor):
            return
        local_radial = radial[in_trap_floor]
        floor = np.array(
            [g.trap_floor_height_at_radius(float(radius)) for radius in local_radial],
            dtype=np.float32,
        )
        below_floor = pos[in_trap_floor, 1] < floor
        if not np.any(below_floor):
            return
        trap_indices = np.flatnonzero(in_trap_floor)
        hit_indices = trap_indices[below_floor]
        pos[hit_indices, 1] = floor[below_floor] + 1e-4
        vel[hit_indices, 1] = np.maximum(np.abs(vel[hit_indices, 1]) * 0.18, 0.002)

    def _update_status(self, active_mask: np.ndarray) -> None:
        active_indices = np.flatnonzero(active_mask)
        pos = self.positions[active_indices]
        g = self.geometry

        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        central_underflow = radial <= g.trap_bottom_half_depth_m
        inside_overflow_tube = self._inside_overflow_tube[active_indices]
        floor = np.array([g.trap_floor_height_at_radius(float(radius)) for radius in radial], dtype=np.float32)
        trapped = (pos[:, 1] <= floor + 0.006) & central_underflow
        overflow = (pos[:, 1] >= g.height_m) & inside_overflow_tube

        self.status[active_indices[trapped]] = 1
        self.status[active_indices[overflow]] = 2

