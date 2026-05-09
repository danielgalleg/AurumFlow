import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_sim.rl_env import parameters_to_geometry
from visual_sim.geometry import ClassifierGeometry


def generate_2d_profile(geom: ClassifierGeometry):
    y_samples = 240
    y = np.linspace(0, geom.height_m, y_samples)
    r_outer = np.array([geom.allowed_radius_at_height(yi) for yi in y])

    y_tube = np.linspace(geom.central_tube_bottom_height_m, geom.height_m, max(40, y_samples // 4))
    r_tube = np.full_like(y_tube, geom.central_tube_radius_m)

    return y, r_outer, y_tube, r_tube


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="rl_runs/ga_openfoam_res38_v2")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "evaluations"

    if not eval_dir.exists():
        print(f"No se encontro {eval_dir}")
        return

    json_files = list(eval_dir.glob("*/geometry.json"))
    print(f"Cargadas {len(json_files)} geometrias en curso.")

    if not json_files:
        return

    frames_dir = run_dir / "frames_all"
    frames_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 9))

    print("Generando graficos individuales...")
    count = 0
    for jpath in json_files:
        cand_id = jpath.parent.name
        frame_path = frames_dir / f"{cand_id}.png"
        if frame_path.exists():
            continue

        try:
            payload = json.loads(jpath.read_text())
            params = payload['params']
            geom = parameters_to_geometry(params)
            geom.validate()
        except Exception as e:
            print(f"Error cargando {cand_id}: {e}")
            continue

        max_radius = max(0.1, geom.cylinder_radius_m)
        max_height = max(0.35, geom.height_m)

        ax.clear()
        ax.set_xlim(-max_radius * 1.1, max_radius * 1.1)
        ax.set_ylim(-0.02, max_height * 1.05)
        ax.set_aspect('equal')

        ax.set_xlabel('Radio (m)')
        ax.set_ylabel('Altura (m)')

        y, r_outer, y_tube, r_tube = generate_2d_profile(geom)

        # Polygono del fluido cerrado: bajamos por la pared exterior, conectamos al
        # tubo central, bajamos por dentro del tubo y volvemos al eje.
        outer_x = np.concatenate([
            r_outer,
            r_tube[::-1],
            [0.0],
        ])
        outer_y = np.concatenate([
            y,
            y_tube[::-1],
            [y_tube[0]],
        ])
        full_x = np.concatenate([outer_x, -outer_x[::-1]])
        full_y = np.concatenate([outer_y, outer_y[::-1]])
        ax.fill(full_x, full_y, color='cyan', alpha=0.25)

        # Pared exterior
        ax.plot(r_outer, y, color='blue', linewidth=2)
        ax.plot(-r_outer, y, color='blue', linewidth=2)

        # Tubo central
        ax.plot(r_tube, y_tube, color='red', linewidth=3)
        ax.plot(-r_tube, y_tube, color='red', linewidth=3)
        ax.plot([-r_tube[0], r_tube[0]], [y_tube[0], y_tube[0]],
                color='red', linewidth=2, linestyle='--', alpha=0.5)

        # Cuello (separacion entre lobulos)
        ax.axhline(y=geom.neck_height_m, color='gray', linestyle=':', alpha=0.4)
        ax.text(max_radius * 0.95, geom.neck_height_m, 'Cuello',
                color='gray', va='bottom', ha='right', fontsize=8)

        # Inlet
        inlet_y = geom.inlet_height_m
        inlet_x = geom.allowed_radius_at_height(inlet_y)
        inlet_pitch = params.get('inlet_pitch_deg', 0.0)

        ax.plot(inlet_x, inlet_y, marker='o', color='green', markersize=8, zorder=5)
        ax.plot(-inlet_x, inlet_y, marker='o', color='green', markersize=8, zorder=5)

        v_len = max_radius * 0.4
        dx = -v_len * np.cos(np.radians(inlet_pitch))
        dy = v_len * np.sin(np.radians(inlet_pitch))

        ax.annotate('', xy=(inlet_x + dx, inlet_y + dy), xytext=(inlet_x, inlet_y),
                    arrowprops=dict(facecolor='green', edgecolor='green', width=2, headwidth=8, shrink=0),
                    zorder=4)
        ax.annotate('', xy=(-inlet_x - dx, inlet_y + dy), xytext=(-inlet_x, inlet_y),
                    arrowprops=dict(facecolor='green', edgecolor='green', width=2, headwidth=8, shrink=0),
                    zorder=4)

        result_json = jpath.parent / f"{cand_id}_000" / "openfoam_result.json"
        title = f"{cand_id}\nVel. Inyeccion: {params.get('flow_velocity_m_s', 0):.2f} m/s | Pitch: {inlet_pitch:.0f}°"
        if result_json.exists():
            try:
                res = json.loads(result_json.read_text())
                rec = res.get('target_recovery_pct', 0.0)
                rej = res.get('non_target_rejection_pct', 0.0)
                title += f"\nOro retenido: {rec:.1f}% | Arena rechazada: {rej:.1f}%"
            except Exception:
                title += "\n(Evaluando CFD...)"
        else:
            title += "\n(Evaluando CFD...)"

        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.4)

        fig.savefig(frame_path, dpi=130, bbox_inches='tight')
        count += 1
        if count % 10 == 0:
            print(f"Generadas {count} imagenes...")

    print(f"Completado. Imagenes nuevas generadas: {count}. Guardadas en: {frames_dir}")


if __name__ == "__main__":
    main()
