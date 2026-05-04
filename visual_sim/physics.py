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
    dt_s: float = 0.003
    seed: int = 7
    max_particle_radius_m: float = 0.0024
    feed_duration_s: float = 5.0
    use_hindrance: bool = True
    feed_solid_volume_fraction: float = 0.01


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

        self._init_fluid_grid()

        self.reset()

    def _init_fluid_grid(self) -> None:
        g = self.geometry
        self._grid_r_size = 256
        self._grid_y_size = 256
        self._max_r = g.width_m / 2.0
        self._max_y = g.height_m

        r = np.linspace(0.0, self._max_r, self._grid_r_size, dtype=np.float32)
        y = np.linspace(0.0, self._max_y, self._grid_y_size, dtype=np.float32)
        RR, YY = np.meshgrid(r, y, indexing='ij')
        radial = RR.flatten()
        y_flat = YY.flatten()
        
        safe_radial = np.maximum(radial, 1e-6)
        local_radius = _allowed_radius_for_y(y_flat, g)
        
        y_norm = np.clip(y_flat / g.height_m, 0.0, 1.0)
        inlet_height_norm = np.clip(g.inlet_height_m / g.height_m, 0.0, 1.0)
        self._inlet_zone_grid = np.exp(-((y_flat - g.inlet_height_m) ** 2) / max(1e-6, (0.11 * g.height_m) ** 2)).reshape(self._grid_r_size, self._grid_y_size)

        viscous_core_radius = max(g.overflow_tube_radius_m, 1e-3)
        circulation_inlet_radius = max(0.6 * g.body_top_radius_m, g.overflow_tube_radius_m * 2.0)
        circulation = (
            2.0
            * np.pi
            * circulation_inlet_radius
            * g.inlet_velocity_m_s
            * np.clip(g.deflector_strength, 0.05, 1.5)
        )
        free_vortex = circulation / (2.0 * np.pi * safe_radial)
        viscous_factor = 1.0 - np.exp(-((safe_radial / viscous_core_radius) ** 2))
        tangential_speed = free_vortex * viscous_factor
        height_envelope = np.clip(
            np.minimum(y_norm / max(0.05, inlet_height_norm * 0.6), 1.0)
            * np.clip((1.0 - y_norm) / max(0.05, 1.0 - inlet_height_norm), 0.0, 1.0)
            + 0.15,
            0.0,
            1.0,
        )
        tangential_speed *= height_envelope
        tangential_speed += 0.45 * g.inlet_velocity_m_s * self._inlet_zone_grid.flatten()

        envelope_floor = g.overflow_tube_radius_m
        envelope_ceiling = 2.0 * g.overflow_tube_radius_m
        upflow_envelope_radius = np.clip(0.5 * local_radius, envelope_floor, envelope_ceiling)
        upflow_norm = radial / np.maximum(upflow_envelope_radius, 1e-6)
        upflow_weight = np.exp(-(upflow_norm**4))
        body_area = np.pi * max(g.body_top_radius_m, 1e-6) ** 2
        mean_envelope = max(envelope_floor, 0.5 * (envelope_floor + envelope_ceiling))
        core_integral_area = np.pi * mean_envelope ** 2
        annulus_area = max(body_area - core_integral_area, 1e-6)
        inlet_flow_rate = g.inlet_velocity_m_s * 0.05 * body_area
        
        real_upward_velocity = inlet_flow_rate / (np.pi * max(g.overflow_tube_radius_m, 1e-6)**2)
        self._real_upward_velocity = float(real_upward_velocity)
        
        downflow_speed = 2.5 * inlet_flow_rate / annulus_area
        reversal_zone = np.clip(
            0.18
            + 0.82
            * (y_flat - 0.25 * g.trap_height_m)
            / max(1e-6, g.overflow_tube_bottom_height_m - 0.25 * g.trap_height_m),
            0.0,
            1.0,
        )
        above_mouth = (y_flat >= g.overflow_tube_bottom_height_m).astype(np.float32)
        inside_bore_axial = (radial <= g.overflow_tube_radius_m).astype(np.float32)
        core_active = (1.0 - above_mouth) + above_mouth * inside_bore_axial
        inner_upflow = real_upward_velocity * upflow_weight * reversal_zone * core_active
        downflow_profile = (1.0 - above_mouth) * (1.0 - upflow_weight) + above_mouth * (1.0 - inside_bore_axial)
        
        # Efecto "Downcomer" (Corriente descendente de pared) descubierto en OpenFOAM
        # El agua choca con la pared exterior, pierde velocidad tangencial y es forzada hacia abajo violentamente.
        wall_proximity = np.clip((radial - 0.80 * local_radius) / np.maximum(0.20 * local_radius, 1e-6), 0.0, 1.0)
        wall_downcomer = -1.5 * g.inlet_velocity_m_s * (wall_proximity ** 2)
        
        outer_downflow = -downflow_speed * downflow_profile + wall_downcomer
        axial_velocity = outer_downflow + inner_upflow

        active_height = max(g.overflow_tube_bottom_height_m - 0.25 * g.trap_height_m, 1e-3)
        inflow_speed_at_radius = inlet_flow_rate / (2.0 * np.pi * np.maximum(radial, 1e-3) * active_height)
        below_mouth = y_flat < g.overflow_tube_bottom_height_m
        mouth_below = np.exp(
            -((y_flat - g.overflow_tube_bottom_height_m) ** 2)
            / max(1e-6, (0.10 * g.height_m) ** 2)
        )
        mouth_zone = np.where(below_mouth, mouth_below, 1.0)
        outside_tube_bore = radial >= g.overflow_tube_radius_m
        capture_kernel = np.exp(
            -((radial / max(1e-6, 0.55 * g.body_top_radius_m)) ** 2)
        ) * outside_tube_bore.astype(np.float32)
        mouth_boost = 0.6 * g.inlet_velocity_m_s * mouth_zone * capture_kernel
        inward_speed = inflow_speed_at_radius * (1.0 - upflow_weight) + mouth_boost

        self._grid_v_tangential = tangential_speed.reshape(self._grid_r_size, self._grid_y_size)
        self._grid_v_axial = axial_velocity.reshape(self._grid_r_size, self._grid_y_size)
        self._grid_v_inward = inward_speed.reshape(self._grid_r_size, self._grid_y_size)

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
        solid_fraction, granular_velocity = self._local_solid_effects(pos, diameter)
        settling = self._settling_velocity(density, diameter, shape, solid_fraction)
        target_velocity = fluid.copy()
        radial = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        radial_x = np.divide(pos[:, 0], np.maximum(radial, 1e-6))
        radial_z = np.divide(pos[:, 2], np.maximum(radial, 1e-6))
        tangential_speed = np.abs(
            fluid[:, 0] * (-radial_z) + fluid[:, 2] * radial_x
        )
        tangential_reynolds = (
            WATER_DENSITY_KG_M3
            * tangential_speed
            * np.maximum(diameter, 1e-9)
            / WATER_VISCOSITY_PA_S
        )
        drag_correction = 1.0 + 0.15 * np.maximum(tangential_reynolds, 1e-9) ** 0.687
        response_time = density * np.maximum(diameter, 1e-9) ** 2 / (
            18.0 * WATER_VISCOSITY_PA_S * drag_correction
        )
        centrifugal_acceleration = tangential_speed**2 / np.maximum(radial, 0.01 * self.geometry.cylinder_radius_m)
        density_factor = np.maximum(0.0, density - WATER_DENSITY_KG_M3) / np.maximum(density, 1.0)
        centrifugal_slip = np.clip(
            response_time * centrifugal_acceleration * density_factor,
            0.0,
            0.5 * tangential_speed,
        )
        target_velocity[:, 0] += centrifugal_slip * radial_x
        target_velocity[:, 2] += centrifugal_slip * radial_z
        
        # Numerical artifact correction: explicit Euler integration of circular
        # motion adds a spurious outward radial drift of approx 0.5 * v_theta^2 * dt / r.
        # We subtract this from the target velocity so particles don't artificially
        # resist the inward fluid suction.
        spurious_drift = 0.5 * tangential_speed**2 * self.config.dt_s / np.maximum(radial, 1e-3)
        target_velocity[:, 0] -= spurious_drift * radial_x
        target_velocity[:, 2] -= spurious_drift * radial_z

        target_velocity[:, 1] -= settling
        target_velocity += granular_velocity

        alpha = self._drag_alpha_schiller_naumann(vel, target_velocity, density, diameter)
        # Localised turbulence amplifier under the vortex finder, where the
        # contraction of streamlines into the overflow tube generates intense
        # eddies that physically break orbital trapping.
        mouth_dist = pos[:, 1] - self.geometry.overflow_tube_bottom_height_m
        mouth_kernel = np.exp(-(mouth_dist / 0.040) ** 2)
        turb_scale = self.geometry.turbulence * (1.0 + 4.0 * mouth_kernel)
        noise = self.rng.normal(0.0, 1.0, size=vel.shape).astype(np.float32) * turb_scale[:, None]
        # Extra radial component captures the strong cross-flow eddies typical
        # of cyclone short-circuit zones.
        noise[:, 0] += self.rng.normal(0.0, 1.0, size=vel.shape[0]).astype(np.float32) * (
            turb_scale * 3.0
        )
        vel += (target_velocity - vel) * alpha[:, None] + noise * np.sqrt(self.config.dt_s)
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
        solid_fraction: np.ndarray,
    ) -> np.ndarray:
        density_delta = np.maximum(0.0, density - WATER_DENSITY_KG_M3)
        v_stokes = density_delta * GRAVITY_M_S2 * diameter**2 / (18.0 * WATER_VISCOSITY_PA_S)
        reynolds = WATER_DENSITY_KG_M3 * v_stokes * np.maximum(diameter, 1e-9) / WATER_VISCOSITY_PA_S
        drag_correction = 1.0 + 0.15 * np.maximum(reynolds, 1e-9) ** 0.687
        v_terminal = v_stokes / drag_correction
        hindered = np.clip(1.0 - solid_fraction, 0.05, 1.0) ** 4.65
        return v_terminal * np.clip(shape, 0.15, 1.5) * hindered

    def _drag_alpha_schiller_naumann(
        self,
        particle_velocity: np.ndarray,
        target_velocity: np.ndarray,
        density: np.ndarray,
        diameter: np.ndarray,
    ) -> np.ndarray:
        relative_speed = np.linalg.norm(target_velocity - particle_velocity, axis=1)
        reynolds = (
            WATER_DENSITY_KG_M3
            * relative_speed
            * np.maximum(diameter, 1e-9)
            / WATER_VISCOSITY_PA_S
        )
        correction = np.where(
            reynolds < 1000.0,
            1.0 + 0.15 * np.maximum(reynolds, 1e-9) ** 0.687,
            0.0183333333 * np.maximum(reynolds, 1e-9),
        )
        response_time = density * np.maximum(diameter, 1e-9) ** 2 / (
            18.0 * WATER_VISCOSITY_PA_S * correction
        )
        return np.clip(self.config.dt_s / np.maximum(response_time, 1e-5), 0.0, 1.0)

    def _local_solid_effects(
        self,
        pos: np.ndarray,
        diameter: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.use_hindrance or pos.size == 0:
            return np.zeros(pos.shape[0], dtype=np.float32), np.zeros_like(pos)

        grid_n = 20
        g = self.geometry
        domain_min = np.array([-g.half_width_m, 0.0, -g.half_depth_m], dtype=np.float32)
        domain_size = np.array([g.width_m, g.height_m, g.depth_m], dtype=np.float32)
        normalized = np.clip((pos - domain_min) / np.maximum(domain_size, 1e-6), 0.0, 0.999999)
        cells = (normalized * grid_n).astype(np.int32)
        cells = np.clip(cells, 0, grid_n - 1)

        grid = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
        particle_volume = (np.pi / 6.0) * np.maximum(diameter, 1e-9) ** 3
        cell_volume = float(np.prod(domain_size / grid_n))
        domain_volume = float(np.prod(domain_size))
        target_solid_volume = self.config.feed_solid_volume_fraction * domain_volume
        simulated_solid_volume = float(particle_volume.sum())
        parcel_scale = float(np.clip(target_solid_volume / max(simulated_solid_volume, 1e-12), 1.0, 1e6))
        np.add.at(grid, (cells[:, 0], cells[:, 1], cells[:, 2]), particle_volume * parcel_scale)

        solid_fraction_grid = np.clip(grid / max(cell_volume, 1e-12), 0.0, 0.62)
        solid_fraction = solid_fraction_grid[cells[:, 0], cells[:, 1], cells[:, 2]]

        spacing = domain_size / grid_n
        grad_x, grad_y, grad_z = np.gradient(
            solid_fraction_grid,
            float(spacing[0]),
            float(spacing[1]),
            float(spacing[2]),
            edge_order=1,
        )
        gradient = np.column_stack(
            (
                grad_x[cells[:, 0], cells[:, 1], cells[:, 2]],
                grad_y[cells[:, 0], cells[:, 1], cells[:, 2]],
                grad_z[cells[:, 0], cells[:, 1], cells[:, 2]],
            )
        ).astype(np.float32)
        gradient_norm = np.linalg.norm(gradient, axis=1)
        overload = np.clip((solid_fraction - 0.45) / 0.17, 0.0, 1.0)
        granular_velocity = -gradient / np.maximum(gradient_norm[:, None], 1e-6)
        granular_velocity *= (0.018 * overload)[:, None]
        return solid_fraction.astype(np.float32), granular_velocity.astype(np.float32)

    def _fluid_velocity(self, pos: np.ndarray, swirl_bias: np.ndarray) -> np.ndarray:
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        radial = np.sqrt(x**2 + z**2)
        safe_radial = np.maximum(radial, 1e-6)
        tangent_x = -z / safe_radial
        tangent_z = x / safe_radial
        radial_x = x / safe_radial
        radial_z = z / safe_radial

        r_idx = np.clip(radial / self._max_r * (self._grid_r_size - 1), 0.0, self._grid_r_size - 1.001)
        y_idx = np.clip(y / self._max_y * (self._grid_y_size - 1), 0.0, self._grid_y_size - 1.001)

        r0 = r_idx.astype(np.int32)
        r1 = r0 + 1
        dr = r_idx - r0

        y0 = y_idx.astype(np.int32)
        y1 = y0 + 1
        dy = y_idx - y0

        w00 = (1.0 - dr) * (1.0 - dy)
        w10 = dr * (1.0 - dy)
        w01 = (1.0 - dr) * dy
        w11 = dr * dy

        tangential_speed = (
            self._grid_v_tangential[r0, y0] * w00
            + self._grid_v_tangential[r1, y0] * w10
            + self._grid_v_tangential[r0, y1] * w01
            + self._grid_v_tangential[r1, y1] * w11
        )
        axial_velocity = (
            self._grid_v_axial[r0, y0] * w00
            + self._grid_v_axial[r1, y0] * w10
            + self._grid_v_axial[r0, y1] * w01
            + self._grid_v_axial[r1, y1] * w11
        )
        inward_speed = (
            self._grid_v_inward[r0, y0] * w00
            + self._grid_v_inward[r1, y0] * w10
            + self._grid_v_inward[r0, y1] * w01
            + self._grid_v_inward[r1, y1] * w11
        )
        inlet_zone = (
            self._inlet_zone_grid[r0, y0] * w00
            + self._inlet_zone_grid[r1, y0] * w10
            + self._inlet_zone_grid[r0, y1] * w01
            + self._inlet_zone_grid[r1, y1] * w11
        )

        stream_spread = 0.010 * swirl_bias * inlet_zone

        fluid = np.zeros_like(pos)
        fluid[:, 0] = tangential_speed * tangent_x - inward_speed * radial_x + stream_spread
        fluid[:, 1] = axial_velocity
        fluid[:, 2] = tangential_speed * tangent_z - inward_speed * radial_z
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
        previous_radial = np.sqrt(previous_pos[:, 0] ** 2 + previous_pos[:, 2] ** 2)
        capture_radius = g.overflow_tube_radius_m
        # Capture band: the small region around the vortex finder mouth where
        # the suction can pull a particle laterally into the tube bore. Above
        # this band the tube wall is treated as solid.
        capture_band_height = 0.06 * g.height_m
        in_capture_band = (
            np.abs(pos[:, 1] - g.overflow_tube_bottom_height_m) < capture_band_height
        )
        # Membership is granted when the particle: (a) crosses the mouth plane
        # from below into the bore, (b) was already inside the bore last step
        # (sticky state), or (c) crosses the bore wall while located inside
        # the capture band.
        crossing_mouth = (
            (previous_pos[:, 1] < g.overflow_tube_bottom_height_m)
            & (pos[:, 1] >= g.overflow_tube_bottom_height_m)
            & (radial <= capture_radius)
        )
        already_inside_bore = (
            (previous_pos[:, 1] >= g.overflow_tube_bottom_height_m)
            & (previous_radial <= capture_radius)
            & (pos[:, 1] >= g.overflow_tube_bottom_height_m)
            & (radial <= capture_radius)
        )
        crossed_wall_in_band = (
            in_capture_band
            & (previous_radial > capture_radius)
            & (radial <= capture_radius)
        )
        captured = crossing_mouth | already_inside_bore | crossed_wall_in_band
        self._inside_overflow_tube[active_indices[captured]] = True

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

        # Particles that tunnelled through the outer pipe wall in the last
        # step end up with radius < tube_radius without being marked as
        # belonging to the overflow stream. Push them back outside the wall
        # and reflect their inward radial velocity so they cannot keep
        # crossing the wall on subsequent steps.
        wall_penetration = tube_zone & ~inside_tube & (radial < g.overflow_tube_radius_m)
        if np.any(wall_penetration):
            pos[wall_penetration, 0] = normal_x[wall_penetration] * (
                g.overflow_tube_radius_m + 5e-4
            )
            pos[wall_penetration, 2] = normal_z[wall_penetration] * (
                g.overflow_tube_radius_m + 5e-4
            )
            radial_velocity = (
                vel[wall_penetration, 0] * normal_x[wall_penetration]
                + vel[wall_penetration, 2] * normal_z[wall_penetration]
            )
            inward = radial_velocity < 0.0
            mask = np.where(wall_penetration)[0][inward]
            vel[mask, 0] -= 1.8 * radial_velocity[inward] * normal_x[mask]
            vel[mask, 2] -= 1.8 * radial_velocity[inward] * normal_z[mask]

        # Particles outside the tube wall, above the capture band: damp any
        # inward radial velocity so they slide along the wall down towards
        # the mouth instead of pushing through. Inside the capture band the
        # tube wall is permeable so the suction can pull particles in.
        capture_band_height = 0.06 * g.height_m
        above_capture_band = pos[:, 1] >= (
            g.overflow_tube_bottom_height_m + capture_band_height
        )
        outer_pipe_skin = (
            tube_zone
            & above_capture_band
            & ~inside_tube
            & (radial >= g.overflow_tube_radius_m)
            & (radial < g.overflow_tube_radius_m * 1.25)
        )
        if np.any(outer_pipe_skin):
            radial_velocity_skin = (
                vel[outer_pipe_skin, 0] * normal_x[outer_pipe_skin]
                + vel[outer_pipe_skin, 2] * normal_z[outer_pipe_skin]
            )
            inward = radial_velocity_skin < 0.0
            skin_indices = np.where(outer_pipe_skin)[0]
            inward_indices = skin_indices[inward]
            vel[inward_indices, 0] -= radial_velocity_skin[inward] * normal_x[inward_indices]
            vel[inward_indices, 2] -= radial_velocity_skin[inward] * normal_z[inward_indices]
            # Mild downward bias so particles slide off the tube exterior
            # rather than stagnating in front of the wall.
            vel[outer_pipe_skin, 1] = np.minimum(vel[outer_pipe_skin, 1], -0.100)

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
            tube_upflow = 1.5 * getattr(self, "_real_upward_velocity", g.upward_velocity_m_s)
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

