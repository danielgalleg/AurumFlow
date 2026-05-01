from __future__ import annotations

from .models import ParticleClass, Stream


def example_alluvial_feed() -> Stream:
    """Alimentacion ficticia para comparar alternativas antes de medir material real."""

    return Stream.from_classes(
        "alimentacion aluvial ejemplo",
        [
            ParticleClass("oro fino 30-75um", 0.018, 50.0, 19_300.0, 0.65),
            ParticleClass("oro medio 75-250um", 0.024, 150.0, 19_300.0, 0.8),
            ParticleClass("oro grueso >250um", 0.010, 450.0, 19_300.0, 0.95),
            ParticleClass("arena cuarzo fina", 42.0, 120.0, 2_650.0, 0.85),
            ParticleClass("arena cuarzo media", 81.0, 420.0, 2_650.0, 0.9),
            ParticleClass("limo arcilloso", 22.0, 25.0, 2_400.0, 0.45),
            ParticleClass("minerales pesados magnetita", 3.6, 180.0, 5_150.0, 0.75),
            ParticleClass("grava liviana", 11.0, 1_800.0, 2_700.0, 0.9),
        ],
    )

