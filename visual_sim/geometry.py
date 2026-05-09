from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierGeometry:
    """Geometria 'Clepsamia' (reloj de arena): dos lobulos curvos independientes
    unidos por un cuello angosto, con un inlet tangencial en el lobulo superior
    y un tubo central como unico outlet.

    El perfil de revolucion es C^1-continuo en todos los puntos de union
    (sin vertices ni esquinas) y cierra a un punto en el fondo y al radio del
    tubo central en el tope. No hay superficies planas en ningun lado.
    """

    height_m: float = 0.22
    neck_height_ratio: float = 0.50
    neck_radius_m: float = 0.012

    upper_max_radius_m: float = 0.060
    upper_max_position_ratio: float = 0.50

    lower_max_radius_m: float = 0.055
    lower_max_position_ratio: float = 0.55

    inlet_height_ratio: float = 0.55
    inlet_pitch_deg: float = 0.0
    inlet_angle_deg: float = -90.0
    inlet_yaw_deg: float = 70.0
    """Tangencialidad del chorro de inyeccion respecto al plano radial:
        0  -> chorro 100% radial (apunta directo al eje, sin swirl)
        90 -> chorro 100% tangente a la pared (maximo swirl, ideal para ciclon)
    Reemplaza los pesos hardcoded swirl/radial_weight previos."""

    central_tube_radius_m: float = 0.008
    central_tube_bottom_ratio: float = 0.55

    inlet_velocity_m_s: float = 0.50

    @property
    def neck_height_m(self) -> float:
        return self.height_m * self.neck_height_ratio

    @property
    def upper_lobe_height_m(self) -> float:
        return self.height_m - self.neck_height_m

    @property
    def lower_lobe_height_m(self) -> float:
        return self.neck_height_m

    @property
    def upper_max_height_m(self) -> float:
        return self.neck_height_m + self.upper_lobe_height_m * self.upper_max_position_ratio

    @property
    def lower_max_height_m(self) -> float:
        return self.lower_lobe_height_m * (1.0 - self.lower_max_position_ratio)

    @property
    def central_tube_bottom_height_m(self) -> float:
        return self.height_m * (1.0 - self.central_tube_bottom_ratio)

    @property
    def inlet_height_m(self) -> float:
        return self.neck_height_m + self.upper_lobe_height_m * self.inlet_height_ratio

    @property
    def upper_lobe_top_radius_m(self) -> float:
        """En el tope del lobulo superior, la pared cierra exactamente al radio
        del tubo central, asegurando una transicion suave (sin techo plano)."""
        return self.central_tube_radius_m

    @property
    def lower_lobe_bottom_radius_m(self) -> float:
        """El fondo del lobulo inferior cierra suavemente a un punto."""
        return 0.0

    @property
    def cylinder_radius_m(self) -> float:
        """Radio cilindrico maximo del dominio: muestreamos el perfil para
        garantizar que sea siempre el radio maximo real (incluyendo curvas)."""
        max_r = max(self.upper_max_radius_m, self.lower_max_radius_m)
        for idx in range(401):
            h = self.height_m * idx / 400.0
            r = self.allowed_radius_at_height(h)
            if r > max_r:
                max_r = r
        return max_r * 1.005  # margen de seguridad

    @property
    def half_width_m(self) -> float:
        return self.cylinder_radius_m

    @property
    def half_depth_m(self) -> float:
        return self.cylinder_radius_m

    @property
    def width_m(self) -> float:
        return 2.0 * self.cylinder_radius_m

    @property
    def depth_m(self) -> float:
        return 2.0 * self.cylinder_radius_m

    @property
    def domain_min(self) -> tuple[float, float, float]:
        return (-self.cylinder_radius_m, 0.0, -self.cylinder_radius_m)

    @property
    def domain_max(self) -> tuple[float, float, float]:
        return (self.cylinder_radius_m, self.height_m, self.cylinder_radius_m)

    @staticmethod
    def _flat_top_arc(t: float) -> float:
        """De 0 a 1 con derivada nula en t=0 y t=1.
        Usa sin^2(pi/2 * t). C^1-continuo (de hecho C-infinito).
        Ideal para conectar segmentos en el maximo (donde pendiente == 0).
        """
        t = max(0.0, min(1.0, t))
        return math.sin(math.pi * 0.5 * t) ** 2

    @staticmethod
    def _flat_bottom_arc(t: float) -> float:
        """De 0 a 1 con derivada nula en t=0 y t=1, pero crece despacio al inicio.
        Es 1 - cos(pi/2 * (1-t))^2 = sin^2(pi/2 * t)... mismo que arriba en realidad.
        Mantenido por claridad de intencion al ser usado al reves."""
        return ClassifierGeometry._flat_top_arc(t)

    @staticmethod
    def _round_close_arc(t: float) -> float:
        """De 0 a 1 con derivada infinita en t=0 y derivada nula en t=1.
        Es un cuarto de elipse: cierra suavemente a un punto sin esquinas.
        Util para los extremos (fondo y tope) donde la pared converge a un (cuasi-)punto.
        """
        t = max(0.0, min(1.0, t))
        return math.sqrt(max(0.0, 1.0 - (1.0 - t) ** 2))

    def allowed_radius_at_height(self, height_m: float) -> float:
        """Radio externo de la pared (curva de revolucion) a una altura dada.

        El perfil tiene cuatro segmentos suaves con C^1-continuidad en sus uniones:
          - Bottom -> lower_max:  cierra a punto en el fondo (cuarto de elipse).
          - lower_max -> Neck:    sin^2 invertido (slope=0 en ambos extremos).
          - Neck -> upper_max:    sin^2 (slope=0 en ambos extremos).
          - upper_max -> Top:     cuarto de elipse invertido al radio del tubo.

        En todos los maximos y en el cuello la pendiente es exactamente 0, lo que
        garantiza ausencia de vertices visibles. En los extremos (h=0 y h=H) la
        pared es vertical, lo que da un cierre tipo huevo/balon.
        """
        h = max(0.0, min(self.height_m, height_m))
        h_lmax = self.lower_max_height_m
        h_neck = self.neck_height_m
        h_umax = self.upper_max_height_m
        H = self.height_m

        if h <= h_lmax:
            # Fondo (h=0, r=0) -> lower_max (h=h_lmax, r=lower_max_r)
            # Cuarto de elipse: vertical en el fondo, horizontal en el max.
            denom = max(1e-9, h_lmax)
            t = h / denom
            return self.lower_max_radius_m * self._round_close_arc(t)

        if h <= h_neck:
            # lower_max (h=h_lmax, r=lower_max_r) -> Cuello (h=h_neck, r=neck_r)
            # sin^2 invertido: slope=0 en ambos extremos.
            denom = max(1e-9, h_neck - h_lmax)
            t = (h - h_lmax) / denom
            # de 1 a 0
            arc = 1.0 - self._flat_top_arc(t)
            return self.neck_radius_m + (self.lower_max_radius_m - self.neck_radius_m) * arc

        if h <= h_umax:
            # Cuello (h=h_neck, r=neck_r) -> upper_max (h=h_umax, r=upper_max_r)
            # sin^2: slope=0 en ambos extremos.
            denom = max(1e-9, h_umax - h_neck)
            t = (h - h_neck) / denom
            arc = self._flat_top_arc(t)
            return self.neck_radius_m + (self.upper_max_radius_m - self.neck_radius_m) * arc

        # upper_max (h=h_umax, r=upper_max_r) -> Tope (h=H, r=central_tube_r)
        # Cuarto de elipse invertido: horizontal en el max, vertical en el tope.
        denom = max(1e-9, H - h_umax)
        t = (h - h_umax) / denom
        # arco que va de 0 a 1, con slope=0 en t=0 y slope=infinito en t=1
        arc = 1.0 - self._round_close_arc(1.0 - t)
        return self.upper_max_radius_m + (self.upper_lobe_top_radius_m - self.upper_max_radius_m) * arc

    def central_tube_radius_at_height(self, height_m: float) -> float:
        """El tubo central es un cilindro recto de radio fijo,
        existente solo entre central_tube_bottom_height_m y height_m."""
        if height_m < self.central_tube_bottom_height_m or height_m > self.height_m:
            return 0.0
        return self.central_tube_radius_m

    def validate(self) -> None:
        if self.height_m <= 0.0:
            raise ValueError("height_m debe ser positiva")
        if not 0.10 <= self.neck_height_ratio <= 0.90:
            raise ValueError("neck_height_ratio debe estar entre 0.10 y 0.90")
        if self.neck_radius_m <= 0.0:
            raise ValueError("neck_radius_m debe ser positivo")
        if self.upper_max_radius_m <= self.neck_radius_m:
            raise ValueError("upper_max_radius_m debe ser mayor que neck_radius_m")
        if self.lower_max_radius_m <= self.neck_radius_m:
            raise ValueError("lower_max_radius_m debe ser mayor que neck_radius_m")
        if not 0.10 <= self.upper_max_position_ratio <= 0.90:
            raise ValueError("upper_max_position_ratio debe estar entre 0.10 y 0.90")
        if not 0.10 <= self.lower_max_position_ratio <= 0.90:
            raise ValueError("lower_max_position_ratio debe estar entre 0.10 y 0.90")
        if not 0.05 <= self.inlet_height_ratio <= 0.95:
            raise ValueError("inlet_height_ratio debe estar entre 0.05 y 0.95 (dentro del lobulo superior)")
        if not -89.5 <= self.inlet_pitch_deg <= 89.5:
            raise ValueError("inlet_pitch_deg debe estar entre -89.5 y 89.5")
        if not -180.0 <= self.inlet_angle_deg <= 180.0:
            raise ValueError("inlet_angle_deg debe estar entre -180.0 y 180.0")
        if not 0.0 <= self.inlet_yaw_deg <= 89.5:
            raise ValueError("inlet_yaw_deg debe estar entre 0.0 (radial) y 89.5 (tangencial)")
        if self.central_tube_radius_m <= 0.0:
            raise ValueError("central_tube_radius_m debe ser positivo")
        if not 0.01 <= self.central_tube_bottom_ratio <= 0.99:
            raise ValueError("central_tube_bottom_ratio debe estar entre 0.01 y 0.99")
        # Si el tubo desciende por debajo del cuello, debe caber dentro del cuello con margen.
        if self.central_tube_bottom_height_m < self.neck_height_m:
            if self.central_tube_radius_m >= self.neck_radius_m * 0.8:
                raise ValueError(
                    "Si el tubo central pasa por el cuello, su radio debe ser <80% del radio del cuello"
                )
        if self.inlet_velocity_m_s <= 0.0:
            raise ValueError("inlet_velocity_m_s debe ser positiva")
        # Verificar que la pared se mantiene dentro del radio cilindrico maximo
        for idx in range(33):
            h = self.height_m * idx / 32.0
            r = self.allowed_radius_at_height(h)
            if r < 0.0 or r > self.cylinder_radius_m * 1.0001:
                raise ValueError("La curva de pared excede el radio cilindrico maximo")
