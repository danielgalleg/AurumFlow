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
            f"rebalse={self.overflow_count}, recuperacion_target={self.target_recovery_pct:.2f}%, "
            f"perdida_target={self.target_loss_pct:.2f}%, rechazo_no_target={self.non_target_rejection_pct:.2f}%, "
            f"contaminacion_trampa={self.trapped_contamination_pct:.2f}%, "
            f"arena_negra_trampa={self.trapped_black_sand_pct:.2f}%, no_procesado={self.unprocessed_pct:.2f}%"
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

    target_total = int(np.count_nonzero(target))
    target_trapped = int(np.count_nonzero(trapped & target))
    target_overflow = int(np.count_nonzero(overflow & target))
    non_target = ~target
    non_target_total = int(np.count_nonzero(non_target))
    non_target_overflow = int(np.count_nonzero(overflow & non_target))
    trapped_total = int(np.count_nonzero(trapped))
    trapped_non_target = int(np.count_nonzero(trapped & ~target))
    trapped_black_sand = int(np.count_nonzero(trapped & black_sand))
    unprocessed_total = int(np.count_nonzero(pending | active))
    particle_total = int(status.size)

    target_recovery = 100.0 * target_trapped / target_total if target_total else 0.0
    target_loss = 100.0 * target_overflow / target_total if target_total else 0.0
    non_target_rejection = 100.0 * non_target_overflow / non_target_total if non_target_total else 0.0
    contamination = 100.0 * trapped_non_target / trapped_total if trapped_total else 0.0
    black_sand_pct = 100.0 * trapped_black_sand / trapped_total if trapped_total else 0.0
    unprocessed_pct = 100.0 * unprocessed_total / particle_total if particle_total else 0.0

    return SimulationMetrics(
        active_count=int(np.count_nonzero(active)),
        pending_count=int(np.count_nonzero(pending)),
        trapped_count=trapped_total,
        overflow_count=int(np.count_nonzero(overflow)),
        unprocessed_count=unprocessed_total,
        target_recovery_pct=target_recovery,
        target_loss_pct=target_loss,
        non_target_rejection_pct=non_target_rejection,
        trapped_contamination_pct=contamination,
        trapped_black_sand_pct=black_sand_pct,
        unprocessed_pct=unprocessed_pct,
    )

