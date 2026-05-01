from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import ParticleClass, ProcessResult, SplitResult, Stream
from .stages import Stage


def merge_streams(name: str, streams: Iterable[Stream]) -> Stream:
    masses: dict[tuple[str, float, float, float], float] = {}

    for stream in streams:
        for particle_class in stream.classes:
            key = (
                particle_class.name,
                particle_class.size_um,
                particle_class.density_kg_m3,
                particle_class.shape_factor,
            )
            masses[key] = masses.get(key, 0.0) + particle_class.mass_kg

    classes = [
        ParticleClass(
            name=key[0],
            size_um=key[1],
            density_kg_m3=key[2],
            shape_factor=key[3],
            mass_kg=mass,
        )
        for key, mass in masses.items()
    ]
    return Stream.from_classes(name, classes)


@dataclass(frozen=True)
class Process:
    name: str
    stages: tuple[Stage, ...]

    def run(self, feed: Stream) -> ProcessResult:
        current = feed
        rejected = []
        stage_results: list[SplitResult] = []

        for stage in self.stages:
            result = stage.split(current)
            stage_results.append(result)
            current = result.concentrate
            rejected.append(result.tailings)

        tailings = merge_streams(f"{self.name} cola acumulada", rejected)
        return ProcessResult(
            feed=feed,
            concentrate=current,
            tailings=tailings,
            stage_results=tuple(stage_results),
        )

