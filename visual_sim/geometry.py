from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierGeometry:
    """Geometria simplificada, pensada para iterar un producto fabricable."""

    width_m: float = 0.18
    depth_m: float = 0.12
    height_m: float = 0.35
    trap_height_m: float = 0.065
    outlet_height_m: float = 0.30
    inlet_height_m: float = 0.24
    inlet_velocity_m_s: float = 0.22
    upward_velocity_m_s: float = 0.065
    turbulence: float = 0.018
    deflector_strength: float = 0.65
    body_top_radius_ratio: float = 1.0
    body_bottom_radius_ratio: float = 0.76
    overflow_tube_radius_ratio: float = 0.10
    overflow_tube_bottom_height_ratio: float = 0.50
    cone_top_height_ratio: float = 0.34
    cone_neck_radius_ratio: float = 0.24
    trap_bottom_radius_ratio: float = 0.55
    body_curve: float = 0.0
    cone_curve: float = 0.0
    trap_curve: float = 0.0
    trap_floor_curve: float = 0.0

    @property
    def half_width_m(self) -> float:
        return 0.5 * self.width_m

    @property
    def half_depth_m(self) -> float:
        return 0.5 * self.depth_m

    @property
    def domain_min(self) -> tuple[float, float, float]:
        return (-self.half_width_m, 0.0, -self.half_depth_m)

    @property
    def domain_max(self) -> tuple[float, float, float]:
        return (self.half_width_m, self.height_m, self.half_depth_m)

    @property
    def trap_volume_m3(self) -> float:
        return self.width_m * self.depth_m * self.trap_height_m

    @property
    def cylinder_radius_m(self) -> float:
        return min(self.half_width_m, self.half_depth_m)

    @property
    def body_top_radius_m(self) -> float:
        return self.cylinder_radius_m * self.body_top_radius_ratio

    @property
    def body_bottom_radius_m(self) -> float:
        return self.cylinder_radius_m * self.body_bottom_radius_ratio

    @property
    def overflow_tube_radius_m(self) -> float:
        return self.cylinder_radius_m * self.overflow_tube_radius_ratio

    @property
    def overflow_tube_bottom_height_m(self) -> float:
        return self.height_m * self.overflow_tube_bottom_height_ratio

    @property
    def cone_top_height_m(self) -> float:
        return self.height_m * self.cone_top_height_ratio

    @property
    def cone_neck_half_depth_m(self) -> float:
        return self.cylinder_radius_m * self.cone_neck_radius_ratio

    @property
    def trap_bottom_half_depth_m(self) -> float:
        return self.cone_neck_half_depth_m * self.trap_bottom_radius_ratio

    def allowed_half_depth_at_height(self, height_m: float) -> float:
        return self.allowed_radius_at_height(height_m)

    def trap_floor_height_at_radius(self, radius_m: float) -> float:
        if self.trap_bottom_half_depth_m <= 0.0:
            return 0.0
        normalized_radius = max(0.0, min(1.0, radius_m / self.trap_bottom_half_depth_m))
        floor_height = 0.35 * abs(self.trap_floor_curve) * self.trap_height_m
        if self.trap_floor_curve >= 0.0:
            return floor_height * normalized_radius**2
        return floor_height * (1.0 - normalized_radius**2)

    def allowed_radius_at_height(self, height_m: float) -> float:
        if height_m < self.trap_height_m:
            t = max(0.0, min(1.0, height_m / self.trap_height_m))
            linear_radius = self.trap_bottom_half_depth_m + (
                self.cone_neck_half_depth_m - self.trap_bottom_half_depth_m
            ) * t
            curve_radius = self.trap_curve * self.cone_neck_half_depth_m * 4.0 * t * (1.0 - t)
            return max(1e-6, linear_radius + curve_radius)
        if height_m < self.cone_top_height_m:
            t = (height_m - self.trap_height_m) / max(
                1e-6, self.cone_top_height_m - self.trap_height_m
            )
            linear_radius = self.cone_neck_half_depth_m + (
                self.body_bottom_radius_m - self.cone_neck_half_depth_m
            ) * t
            curve_scale = max(self.body_bottom_radius_m, self.cone_neck_half_depth_m)
            curve_radius = self.cone_curve * curve_scale * 4.0 * t * (1.0 - t)
            return max(1e-6, linear_radius + curve_radius)
        t = (height_m - self.cone_top_height_m) / max(
            1e-6, self.height_m - self.cone_top_height_m
        )
        t = max(0.0, min(1.0, t))
        linear_radius = self.body_bottom_radius_m + (self.body_top_radius_m - self.body_bottom_radius_m) * t
        curve_radius = self.body_curve * self.cylinder_radius_m * 4.0 * t * (1.0 - t)
        return max(1e-6, linear_radius + curve_radius)

    def validate(self) -> None:
        if self.width_m <= 0.0 or self.depth_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("Las dimensiones de la camara deben ser positivas")
        if not 0.0 < self.trap_height_m < self.height_m:
            raise ValueError("trap_height_m debe estar dentro de la altura de camara")
        if not self.trap_height_m < self.inlet_height_m < self.height_m:
            raise ValueError("inlet_height_m debe estar sobre la trampa y dentro de la camara")
        if not self.inlet_height_m < self.outlet_height_m <= self.height_m:
            raise ValueError("outlet_height_m debe estar sobre la entrada y dentro de la camara")
        if self.inlet_velocity_m_s <= 0.0 or self.upward_velocity_m_s < 0.0:
            raise ValueError("Las velocidades de flujo deben ser no negativas")
        if not 0.05 <= self.body_top_radius_ratio <= 1.0:
            raise ValueError("body_top_radius_ratio debe estar entre 0.05 y 1.0")
        if not 0.005 <= self.body_bottom_radius_ratio <= 1.0:
            raise ValueError("body_bottom_radius_ratio debe estar entre 0.005 y 1.0")
        if not 0.005 <= self.overflow_tube_radius_ratio <= 0.22:
            raise ValueError("overflow_tube_radius_ratio debe estar entre 0.005 y 0.22")
        if not 0.30 <= self.overflow_tube_bottom_height_ratio <= 0.78:
            raise ValueError("overflow_tube_bottom_height_ratio debe estar entre 0.30 y 0.78")
        if not 0.22 <= self.cone_top_height_ratio <= 0.55:
            raise ValueError("cone_top_height_ratio debe estar entre 0.22 y 0.55")
        if not 0.03 <= self.cone_neck_radius_ratio <= 0.60:
            raise ValueError("cone_neck_radius_ratio debe estar entre 0.03 y 0.60")
        if not 0.05 <= self.trap_bottom_radius_ratio <= 8.00:
            raise ValueError("trap_bottom_radius_ratio debe estar entre 0.05 y 8.00")
        if not -0.45 <= self.body_curve <= 0.45:
            raise ValueError("body_curve debe estar entre -0.45 y 0.45")
        if not -0.45 <= self.cone_curve <= 0.45:
            raise ValueError("cone_curve debe estar entre -0.45 y 0.45")
        if not -0.45 <= self.trap_curve <= 0.45:
            raise ValueError("trap_curve debe estar entre -0.45 y 0.45")
        if not -0.45 <= self.trap_floor_curve <= 0.45:
            raise ValueError("trap_floor_curve debe estar entre -0.45 y 0.45")
        if not self.trap_height_m < self.cone_top_height_m < self.inlet_height_m:
            raise ValueError("El cono debe quedar sobre la trampa y bajo la entrada")
        if self.cone_neck_half_depth_m >= self.cylinder_radius_m:
            raise ValueError("La garganta del cono debe caber dentro del radio maximo")
        for idx in range(33):
            height = self.height_m * idx / 32.0
            radius = self.allowed_radius_at_height(height)
            if radius <= 0.0 or radius > self.cylinder_radius_m * 1.0001:
                raise ValueError("La curva de pared debe mantenerse dentro del radio maximo")

