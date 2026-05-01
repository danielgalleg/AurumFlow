from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulador_proceso import Process, Screen, ShakingTable, SluiceBox, example_alluvial_feed


@dataclass(frozen=True)
class Candidate:
    score: float
    recovery_pct: float
    grade_g_t: float
    mass_yield_pct: float
    length_m: float
    slope_deg: float
    water_velocity_m_s: float
    riffle_factor: float


def evaluate(
    length_m: float,
    slope_deg: float,
    water_velocity_m_s: float,
    riffle_factor: float,
) -> Candidate:
    process = Process(
        name="optimizacion canaleta",
        stages=(
            Screen("retirar grava > 1mm", cut_size_um=1_000.0, keep="undersize"),
            SluiceBox(
                "canaleta candidata",
                length_m=length_m,
                slope_deg=slope_deg,
                water_velocity_m_s=water_velocity_m_s,
                riffle_factor=riffle_factor,
                turbulence=0.35,
            ),
            ShakingTable("mesa fija", slope_deg=4.0, stroke_hz=5.0, wash_water_l_min=10.0),
        ),
    )
    result = process.run(example_alluvial_feed())

    # Puntaje balanceado: la recuperacion manda; la ley ayuda, pero con retornos decrecientes.
    score = (
        0.78 * result.gold_recovery_pct
        + 18.0 * min(2.0, result.upgrade_ratio / 10.0)
        - 0.35 * result.mass_yield_pct
    )

    return Candidate(
        score=score,
        recovery_pct=result.gold_recovery_pct,
        grade_g_t=result.concentrate.gold_grade_g_t,
        mass_yield_pct=result.mass_yield_pct,
        length_m=length_m,
        slope_deg=slope_deg,
        water_velocity_m_s=water_velocity_m_s,
        riffle_factor=riffle_factor,
    )


def main() -> None:
    candidates = []
    for length_m in (1.2, 1.8, 2.4, 3.0):
        for slope_deg in (3.0, 4.0, 5.0, 6.0, 7.0):
            for water_velocity_m_s in (0.35, 0.50, 0.65, 0.80, 0.95):
                for riffle_factor in (0.7, 1.0, 1.3):
                    candidates.append(
                        evaluate(length_m, slope_deg, water_velocity_m_s, riffle_factor)
                    )

    best = sorted(candidates, key=lambda c: c.score, reverse=True)[:10]

    print("Top 10 configuraciones de canaleta")
    print("-" * 96)
    print(
        "score    rec_Au%  ley_g/t   masa%   largo_m  pendiente  velocidad_m/s  riffle"
    )
    for c in best:
        print(
            f"{c.score:7.2f}  {c.recovery_pct:7.2f}  {c.grade_g_t:8.1f}  "
            f"{c.mass_yield_pct:6.2f}  {c.length_m:7.2f}  {c.slope_deg:9.2f}  "
            f"{c.water_velocity_m_s:13.2f}  {c.riffle_factor:6.2f}"
        )


if __name__ == "__main__":
    main()

