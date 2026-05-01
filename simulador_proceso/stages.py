from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import ParticleClass, SplitResult, Stream


WATER_DENSITY_KG_M3 = 1_000.0
WATER_VISCOSITY_PA_S = 0.001
GRAVITY_M_S2 = 9.80665


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def terminal_velocity_m_s(particle: ParticleClass, g_multiplier: float = 1.0) -> float:
    """Velocidad terminal tipo Stokes; valida como aproximacion para finos."""

    diameter_m = particle.size_um * 1e-6
    density_delta = max(0.0, particle.density_kg_m3 - WATER_DENSITY_KG_M3)
    velocity = (
        density_delta
        * GRAVITY_M_S2
        * g_multiplier
        * diameter_m**2
        / (18.0 * WATER_VISCOSITY_PA_S)
    )
    return velocity * clamp(particle.shape_factor, 0.15, 1.5)


class Stage(ABC):
    name: str

    def split(self, feed: Stream) -> SplitResult:
        concentrate = []
        tailings = []

        for particle_class in feed.classes:
            probability = clamp(self.capture_probability(particle_class))
            concentrate.append(particle_class.with_mass(particle_class.mass_kg * probability))
            tailings.append(particle_class.with_mass(particle_class.mass_kg * (1.0 - probability)))

        return SplitResult(
            stage_name=self.name,
            feed=feed,
            concentrate=Stream.from_classes(f"{feed.name} -> {self.name} concentrado", concentrate),
            tailings=Stream.from_classes(f"{feed.name} -> {self.name} cola", tailings),
        )

    @abstractmethod
    def capture_probability(self, particle: ParticleClass) -> float:
        """Fraccion de una clase que pasa al producto concentrado de la etapa."""


@dataclass(frozen=True)
class Screen(Stage):
    name: str
    cut_size_um: float
    keep: str = "undersize"
    sharpness_um: float = 60.0

    def capture_probability(self, particle: ParticleClass) -> float:
        transition = sigmoid((self.cut_size_um - particle.size_um) / max(1.0, self.sharpness_um))
        if self.keep == "undersize":
            return transition
        if self.keep == "oversize":
            return 1.0 - transition
        raise ValueError("keep debe ser 'undersize' u 'oversize'")


@dataclass(frozen=True)
class HydraulicClassifier(Stage):
    name: str
    upward_water_velocity_m_s: float
    sharpness_m_s: float = 0.015

    def capture_probability(self, particle: ParticleClass) -> float:
        settling = terminal_velocity_m_s(particle)
        return sigmoid((settling - self.upward_water_velocity_m_s) / self.sharpness_m_s)


@dataclass(frozen=True)
class SluiceBox(Stage):
    name: str
    length_m: float
    slope_deg: float
    water_velocity_m_s: float
    riffle_factor: float = 1.0
    turbulence: float = 0.35

    def capture_probability(self, particle: ParticleClass) -> float:
        settling = terminal_velocity_m_s(particle)
        residence = self.length_m / max(0.05, self.water_velocity_m_s)
        slope_penalty = 1.0 + max(0.0, self.slope_deg - 4.0) * 0.14
        cutoff = self.water_velocity_m_s * (0.08 + 0.05 * self.turbulence) * slope_penalty
        density_bonus = sigmoid((particle.density_kg_m3 - 3_500.0) / 1_200.0)
        opportunity = 1.0 - math.exp(-0.42 * residence * max(0.1, self.riffle_factor))
        return (0.04 + 0.96 * sigmoid((settling - cutoff) / 0.025)) * (
            0.35 + 0.65 * density_bonus
        ) * opportunity


@dataclass(frozen=True)
class ShakingTable(Stage):
    name: str
    slope_deg: float
    stroke_hz: float
    wash_water_l_min: float

    def capture_probability(self, particle: ParticleClass) -> float:
        size_term = sigmoid((particle.size_um - 30.0) / 35.0)
        density_term = sigmoid((particle.density_kg_m3 - 4_000.0) / 1_000.0)
        water_penalty = sigmoid((18.0 - self.wash_water_l_min) / 8.0)
        slope_penalty = sigmoid((9.0 - self.slope_deg) / 2.0)
        stroke_term = math.exp(-((self.stroke_hz - 5.0) ** 2) / 18.0)
        return 0.08 + 0.90 * size_term * density_term * water_penalty * slope_penalty * stroke_term


@dataclass(frozen=True)
class CentrifugalConcentrator(Stage):
    name: str
    g_force: float
    fluidization_water_l_min: float
    retention_factor: float = 1.0

    def capture_probability(self, particle: ParticleClass) -> float:
        settling = terminal_velocity_m_s(particle, g_multiplier=max(1.0, self.g_force))
        water_cutoff = 0.35 + 0.018 * self.fluidization_water_l_min
        heavy_selectivity = sigmoid((particle.density_kg_m3 - 3_000.0) / 900.0)
        size_penalty = sigmoid((particle.size_um - 18.0) / 18.0)
        return (
            0.01
            + 0.97
            * sigmoid((settling - water_cutoff) / 0.35)
            * heavy_selectivity
            * size_penalty
            * clamp(self.retention_factor, 0.2, 1.4)
        )

