from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import MaterialPreset


@dataclass(frozen=True)
class SimulationMetrics:
    active_count: int
    pending_count: int
    trapped_count: int
    overflow_count: int
    unprocessed_count: int
    target_recovery_pct: float
    target_loss_pct: float
    non_target_rejection_pct: float
    trapped_contamination_pct: float
    trapped_black_sand_pct: float
    unprocessed_pct: float

    def summary(self) -> str:
        return (
            f"pendientes={self.pending_count}, activos={self.active_count}, atrapados={self.trapped_count}, "
            f"rebalse={self.overflow_count}, recuperacion_target_peso={self.target_recovery_pct:.2f}%, "
            f"perdida_target_peso={self.target_loss_pct:.2f}%, rechazo_no_target_peso={self.non_target_rejection_pct:.2f}%, "
            f"contaminacion_trampa_peso={self.trapped_contamination_pct:.2f}%, "
            f"arena_negra_trampa_peso={self.trapped_black_sand_pct:.2f}%, no_procesado_peso={self.unprocessed_pct:.2f}%"
        )


def compute_metrics(
    status: np.ndarray,
    material_ids: np.ndarray,
    materials: tuple[MaterialPreset, ...],
) -> SimulationMetrics:
    target_ids = {material.id for material in materials if material.is_target}
    black_sand_ids = {material.id for material in materials if "magnetita" in material.name.lower()}

    pending = status == -1
    active = status == 0
    trapped = status == 1
    overflow = status == 2
    target = np.array([int(mid) in target_ids for mid in material_ids], dtype=bool)
    black_sand = np.array([int(mid) in black_sand_ids for mid in material_ids], dtype=bool)

    masses = np.zeros_like(material_ids, dtype=np.float64)
    for mat in materials:
        mask = material_ids == mat.id
        volume = (4.0 / 3.0) * np.pi * (mat.diameter_m / 2.0)**3
        masses[mask] = volume * mat.density_kg_m3

    target_mass = float(np.sum(masses[target]))
    target_trapped_mass = float(np.sum(masses[trapped & target]))
    target_overflow_mass = float(np.sum(masses[overflow & target]))
    
    non_target = ~target
    non_target_mass = float(np.sum(masses[non_target]))
    non_target_overflow_mass = float(np.sum(masses[overflow & non_target]))
    
    trapped_mass = float(np.sum(masses[trapped]))
    trapped_non_target_mass = float(np.sum(masses[trapped & non_target]))
    trapped_black_sand_mass = float(np.sum(masses[trapped & black_sand]))
    
    unprocessed_mass = float(np.sum(masses[pending | active]))
    total_mass = float(np.sum(masses))

    target_recovery = 100.0 * target_trapped_mass / target_mass if target_mass else 0.0
    target_loss = 100.0 * target_overflow_mass / target_mass if target_mass else 0.0
    non_target_rejection = 100.0 * non_target_overflow_mass / non_target_mass if non_target_mass else 0.0
    contamination = 100.0 * trapped_non_target_mass / trapped_mass if trapped_mass else 0.0
    black_sand_pct = 100.0 * trapped_black_sand_mass / trapped_mass if trapped_mass else 0.0
    unprocessed_pct = 100.0 * unprocessed_mass / total_mass if total_mass else 0.0

    return SimulationMetrics(
        active_count=int(np.count_nonzero(active)),
        pending_count=int(np.count_nonzero(pending)),
        trapped_count=int(np.count_nonzero(trapped)),
        overflow_count=int(np.count_nonzero(overflow)),
        unprocessed_count=int(np.count_nonzero(pending | active)),
        target_recovery_pct=target_recovery,
        target_loss_pct=target_loss,
        non_target_rejection_pct=non_target_rejection,
        trapped_contamination_pct=contamination,
        trapped_black_sand_pct=black_sand_pct,
        unprocessed_pct=unprocessed_pct,
    )

