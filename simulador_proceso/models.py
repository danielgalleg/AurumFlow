from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


GOLD_TAGS = {"oro", "gold", "au"}


@dataclass(frozen=True)
class ParticleClass:
    """Clase granulometrica/quimica agregada, no una particula individual."""

    name: str
    mass_kg: float
    size_um: float
    density_kg_m3: float
    shape_factor: float = 1.0

    @property
    def is_gold(self) -> bool:
        normalized = self.name.lower()
        return any(tag in normalized for tag in GOLD_TAGS)

    def with_mass(self, mass_kg: float) -> "ParticleClass":
        return ParticleClass(
            name=self.name,
            mass_kg=max(0.0, mass_kg),
            size_um=self.size_um,
            density_kg_m3=self.density_kg_m3,
            shape_factor=self.shape_factor,
        )


@dataclass(frozen=True)
class Stream:
    name: str
    classes: tuple[ParticleClass, ...]

    @classmethod
    def from_classes(cls, name: str, classes: Iterable[ParticleClass]) -> "Stream":
        return cls(name=name, classes=tuple(c for c in classes if c.mass_kg > 0.0))

    @property
    def total_mass_kg(self) -> float:
        return sum(c.mass_kg for c in self.classes)

    @property
    def gold_mass_kg(self) -> float:
        return sum(c.mass_kg for c in self.classes if c.is_gold)

    @property
    def gold_grade_g_t(self) -> float:
        if self.total_mass_kg <= 0.0:
            return 0.0
        return 1_000_000.0 * self.gold_mass_kg / self.total_mass_kg

    def get_mass_kg(self, name_fragment: str) -> float:
        needle = name_fragment.lower()
        return sum(c.mass_kg for c in self.classes if needle in c.name.lower())


@dataclass(frozen=True)
class SplitResult:
    stage_name: str
    feed: Stream
    concentrate: Stream
    tailings: Stream

    @property
    def gold_recovery_pct(self) -> float:
        if self.feed.gold_mass_kg <= 0.0:
            return 0.0
        return 100.0 * self.concentrate.gold_mass_kg / self.feed.gold_mass_kg

    @property
    def mass_yield_pct(self) -> float:
        if self.feed.total_mass_kg <= 0.0:
            return 0.0
        return 100.0 * self.concentrate.total_mass_kg / self.feed.total_mass_kg


@dataclass(frozen=True)
class ProcessResult:
    feed: Stream
    concentrate: Stream
    tailings: Stream
    stage_results: tuple[SplitResult, ...]

    @property
    def gold_recovery_pct(self) -> float:
        if self.feed.gold_mass_kg <= 0.0:
            return 0.0
        return 100.0 * self.concentrate.gold_mass_kg / self.feed.gold_mass_kg

    @property
    def gold_loss_pct(self) -> float:
        return 100.0 - self.gold_recovery_pct

    @property
    def mass_yield_pct(self) -> float:
        if self.feed.total_mass_kg <= 0.0:
            return 0.0
        return 100.0 * self.concentrate.total_mass_kg / self.feed.total_mass_kg

    @property
    def upgrade_ratio(self) -> float:
        feed_grade = self.feed.gold_grade_g_t
        if feed_grade <= 0.0:
            return 0.0
        return self.concentrate.gold_grade_g_t / feed_grade

