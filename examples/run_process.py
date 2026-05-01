import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulador_proceso import Process, Screen, ShakingTable, SluiceBox, example_alluvial_feed
from simulador_proceso.reporting import format_result


def build_process() -> Process:
    return Process(
        name="cribado + canaleta + mesa",
        stages=(
            Screen("retirar grava > 1mm", cut_size_um=1_000.0, keep="undersize"),
            SluiceBox(
                "canaleta riffles moderada",
                length_m=2.4,
                slope_deg=5.0,
                water_velocity_m_s=0.65,
                riffle_factor=1.1,
                turbulence=0.35,
            ),
            ShakingTable(
                "mesa de limpieza",
                slope_deg=4.0,
                stroke_hz=5.0,
                wash_water_l_min=10.0,
            ),
        ),
    )


def main() -> None:
    feed = example_alluvial_feed()
    result = build_process().run(feed)
    print(format_result(result))


if __name__ == "__main__":
    main()

