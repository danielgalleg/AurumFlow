from __future__ import annotations

from .models import ProcessResult, Stream


def format_stream(stream: Stream) -> str:
    return (
        f"{stream.name}: masa={stream.total_mass_kg:.3f} kg, "
        f"Au={stream.gold_mass_kg * 1_000:.3f} g, "
        f"ley={stream.gold_grade_g_t:.1f} g/t"
    )


def format_result(result: ProcessResult) -> str:
    lines = [
        "Resumen del proceso",
        "-" * 20,
        format_stream(result.feed),
        format_stream(result.concentrate),
        format_stream(result.tailings),
        (
            f"recuperacion Au={result.gold_recovery_pct:.2f}%, "
            f"perdida Au={result.gold_loss_pct:.2f}%, "
            f"rendimiento masa={result.mass_yield_pct:.2f}%, "
            f"upgrade={result.upgrade_ratio:.1f}x"
        ),
        "",
        "Etapas",
        "-" * 20,
    ]

    for stage_result in result.stage_results:
        lines.append(
            f"{stage_result.stage_name}: captura Au={stage_result.gold_recovery_pct:.2f}%, "
            f"masa a concentrado={stage_result.mass_yield_pct:.2f}%"
        )

    return "\n".join(lines)

