from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialPreset:
    """Propiedades agregadas para paquetes de particulas visuales."""

    id: int
    name: str
    density_kg_m3: float
    diameter_m: float
    shape_factor: float
    color_rgb: tuple[float, float, float]
    feed_fraction: float
    is_target: bool = False

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m


def default_materials() -> tuple[MaterialPreset, ...]:
    """Mezcla ficticia tipo placer para explorar disenos sin datos de campo."""

    return (
        MaterialPreset(
            id=0,
            name="oro fino",
            density_kg_m3=19_300.0,
            diameter_m=120e-6,
            shape_factor=0.65,
            color_rgb=(1.0, 0.78, 0.05),
            feed_fraction=0.05,
            is_target=True,
        ),
        MaterialPreset(
            id=1,
            name="oro medio",
            density_kg_m3=19_300.0,
            diameter_m=350e-6,
            shape_factor=0.9,
            color_rgb=(1.0, 0.55, 0.02),
            feed_fraction=0.03,
            is_target=True,
        ),
        MaterialPreset(
            id=2,
            name="cuarzo arena",
            density_kg_m3=2_650.0,
            diameter_m=420e-6,
            shape_factor=0.9,
            color_rgb=(0.82, 0.76, 0.62),
            feed_fraction=0.62,
        ),
        MaterialPreset(
            id=3,
            name="magnetita arena negra",
            density_kg_m3=5_150.0,
            diameter_m=220e-6,
            shape_factor=0.75,
            color_rgb=(0.08, 0.08, 0.10),
            feed_fraction=0.20,
        ),
        MaterialPreset(
            id=4,
            name="limo arcilloso",
            density_kg_m3=2_400.0,
            diameter_m=45e-6,
            shape_factor=0.45,
            color_rgb=(0.50, 0.42, 0.32),
            feed_fraction=0.10,
        ),
    )


def normalize_fractions(materials: tuple[MaterialPreset, ...]) -> tuple[float, ...]:
    total = sum(max(0.0, material.feed_fraction) for material in materials)
    if total <= 0.0:
        raise ValueError("La suma de fracciones de alimentacion debe ser positiva")
    return tuple(max(0.0, material.feed_fraction) / total for material in materials)

