from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import uuid
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_sim.rl_env import (
    ACTION_BOUNDS,
    ACTION_NAMES,
    action_to_parameters,
    parameters_to_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimiza geometria con algoritmo genetico/evolutivo usando OpenFOAM."
    )
    parser.add_argument("--generations", type=int, default=15, help="Generaciones evolutivas.")
    parser.add_argument("--population", type=int, default=16, help="Individuos por generacion.")
    parser.add_argument("--base-cells", type=int, default=16, help="Resolucion base baja de CFD.")
    parser.add_argument("--particle-end-time", type=float, default=2.0, help="Tiempo de sim de particulas.")
    parser.add_argument("--parcels-scale", type=float, default=0.03, help="Escala de inyeccion de particulas.")
    parser.add_argument("--cores-per-eval", type=int, default=2, help="Nucleos MPI por evaluacion.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Evaluaciones CFD en paralelo (n-jobs * cores_per_eval <= CPU threads).")
    parser.add_argument("--elite-fraction", type=float, default=0.20, help="Fraccion de elite conservada.")
    parser.add_argument("--mutation-prob", type=float, default=0.30, help="Probabilidad de mutar cada gen.")
    parser.add_argument("--mutation-sigma", type=float, default=0.25, help="Desviacion estandar de mutacion.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Individuos por torneo.")
    parser.add_argument("--warm-start-jsons", nargs="+", help="Archivos best_geometry.json previos para sembrar la poblacion.")
    parser.add_argument("--interactive", action="store_true", help="Pausa antes de cada generacion para aprobar/rechazar geometrias manualmente.")
    parser.add_argument("--output-dir", default="rl_runs/ga_openfoam", help="Directorio de salida.")
    return parser.parse_args()


def params_to_action(params: dict[str, float]) -> np.ndarray:
    params = dict(params)
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    for idx, name in enumerate(ACTION_NAMES):
        if name not in params:
            continue
        low, high = ACTION_BOUNDS[name]
        normalized = 2.0 * (float(params[name]) - low) / (high - low) - 1.0
        action[idx] = np.clip(normalized, -1.0, 1.0)
    return action


def cfd_reward(metrics_combined: dict[str, float]) -> float:
    """Funcion de recompensa MULTIPLICATIVA para Clepsamia.

    Multiplicamos los tres factores deseables en vez de sumarlos, para que el
    reward solo sea alto cuando los TRES son simultaneamente altos. Una metrica
    perfecta NO puede compensar una mediocre (a diferencia del esquema sumado
    anterior).

        reward = 4 * recovery * purity * (1 - loss)

    donde:
        - recovery (0..1): fraccion del oro inyectado retenido dentro del dispositivo.
        - purity   (0..1): pureza del concentrado retenido = 1 - contamination.
        - 1 - loss (0..1): 1 menos la fraccion de oro que escapo por el outlet.

    Maximo teorico = 4.0 (los tres en 1.0).

    Comparacion con el esquema anterior (sumado), para el campeon previo
    (recovery=1.00, purity=0.98, loss=0.00):
        sumado: 2*1.0 + 2*(1-contam) - 2*0 - 2*contam ~= 3.88
        mult:   4 * 1.00 * 0.98 * 1.00              = 3.92

    Para una geometria 99.5/99.66 (sacrifica 0.5% oro por 3% mas de pureza):
        mult:   4 * 0.995 * 0.9966 * 0.995          = 3.94  -> mejora real
        sumado: 2*0.995 + 2*0.9966 - 2*0.005 - 2*(1-0.9966) = 3.97

    El esquema multiplicativo penaliza mucho mas duro las geometrias que
    sacrifican mucho de un factor (ej. recovery=0.5 con purity=1.0 da 2.0,
    mientras que en sumado daba ~3.0). Eso fuerza al GA a buscar el balance.
    """
    gold_recovery = max(0.0, metrics_combined.get("target_recovery_pct", 0.0) / 100.0)
    gold_loss = max(0.0, metrics_combined.get("target_loss_pct", 100.0) / 100.0)
    contamination = max(0.0, metrics_combined.get("trapped_contamination_pct", 100.0) / 100.0)

    purity = max(0.0, 1.0 - contamination)
    no_loss = max(0.0, 1.0 - gold_loss)
    return float(4.0 * gold_recovery * purity * no_loss)


def evaluate_action_cfd(payload: tuple[int, int, list[float], str, argparse.Namespace]) -> dict[str, Any]:
    generation, individual, action_values, output_dir, args = payload
    action = np.asarray(action_values, dtype=np.float32)
    params = action_to_parameters(action)
    
    cand_id = f"gen{generation:03d}_ind{individual:03d}_{uuid.uuid4().hex[:6]}"
    cand_dir = Path(output_dir) / "evaluations" / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    
    invalid = False
    reward = -5.0
    
    try:
        geometry = parameters_to_geometry(params)
        geometry.validate()

        # Pre-filtro de cell-count: estimamos cuantas celdas tendria snappyHexMesh
        # antes de llamar a OpenFOAM y rechazamos las geometrias que generarian
        # mallas demasiado caras. Asumimos cellSize = 2R/base_cells en X y Z, y
        # nY = H/cellSize en altura. Factor 1.7 cubre el refinamiento de capa
        # cercana a pared. Threshold calibrado a ~330k cells (~3 min de mesh+CFD).
        bb_radius = geometry.cylinder_radius_m
        cell_size = (2.0 * bb_radius) / max(1, args.base_cells)
        raw_cells = args.base_cells * args.base_cells * (geometry.height_m / max(cell_size, 1e-6))
        predicted_cells = int(raw_cells * 1.7)
        if predicted_cells > 330_000:
            raise ValueError(f"Mesh estimado demasiado grande ({predicted_cells} cells > 330k)")

        geom_json = cand_dir / "geometry.json"
        geom_json.write_text(json.dumps({"params": params, "geometry": asdict(geometry)}, indent=2))
        
        cmd = [
            sys.executable,
            "examples/evaluate_geometry_openfoam.py",
            "--geometry-json", str(geom_json),
            "--output-root", str(cand_dir),
            "--name-prefix", cand_id,
            "--base-cells", str(args.base_cells),
            "--refinement-level", "1",
            "--particle-end-time", str(args.particle_end_time),
            "--parcels-scale", str(args.parcels_scale),
            "--cores", str(args.cores_per_eval),
            "--timeout-flow", "600",
            "--timeout-particles", "900",
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        
        subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=900  # 15 minutos maximo por evaluacion para evitar cuelgues infinitos
        )
        
        result_file = cand_dir / f"{cand_id}_000" / "openfoam_result.json"
        if not result_file.is_file():
            raise RuntimeError("OpenFOAM did not produce a result JSON")
            
        result = json.loads(result_file.read_text())
        if "target_recovery_pct" not in result:
            raise RuntimeError("OpenFOAM result missing combined metrics")
            
        reward = cfd_reward(result)
        metrics = {
            "target_recovery_pct": result.get("target_recovery_pct"),
            "target_loss_pct": result.get("target_loss_pct"),
            "non_target_rejection_pct": result.get("non_target_rejection_pct"),
            "trapped_contamination_pct": result.get("trapped_contamination_pct"),
        }
        invalid = False
        error = ""
        
        # Cleanup heavy OpenFOAM case data to save disk space
        shutil.rmtree(cand_dir / f"{cand_id}_000" / "case", ignore_errors=True)
        shutil.rmtree(cand_dir / f"{cand_id}_000" / "particles", ignore_errors=True)
        
    except Exception as exc:
        geometry = None
        metrics = None
        reward = -5.0
        error = str(exc)
        invalid = True

    finally:
        # Limpieza agresiva de disco:
        # Siempre borramos las carpetas pesadas de OpenFOAM incluso si hubo error
        shutil.rmtree(cand_dir / f"{cand_id}_000" / "case", ignore_errors=True)
        shutil.rmtree(cand_dir / f"{cand_id}_000" / "particles", ignore_errors=True)
        
        # Si el modelo falló o su puntaje es muy bajo, borramos toda su carpeta para liberar espacio
        if invalid or reward < 2.0:
            shutil.rmtree(cand_dir, ignore_errors=True)

    return {
        "generation": generation,
        "individual": individual,
        "reward": reward,
        "action": action.tolist(),
        "params": params,
        "geometry": asdict(geometry) if geometry else None,
        "metrics": metrics,
        "invalid": invalid,
        "error": error,
        "id": cand_id,
    }


def interactive_review(population: list[np.ndarray], generation: int, out_dir: Path, rng: np.random.Generator, args: argparse.Namespace) -> list[np.ndarray]:
    if not args.interactive:
        return population
        
    review_dir = out_dir / f"review_gen_{generation:03d}"
    review_dir.mkdir(parents=True, exist_ok=True)
    
    current_pop = list(population)
    
    def generate_2d_profile(geom):
        y_samples = 200
        y = np.linspace(0, geom.height_m, y_samples)
        r_outer = np.array([geom.allowed_radius_at_height(yi) for yi in y])
        # Tubo central: solo existe entre central_tube_bottom_height_m y height_m
        y_tube = np.linspace(geom.central_tube_bottom_height_m, geom.height_m, max(20, y_samples // 4))
        r_tube = np.full_like(y_tube, geom.central_tube_radius_m)
        return y, r_outer, y_tube, r_tube

    fig, ax = plt.subplots(figsize=(6, 8))
    
    while True:
        print(f"\n--- REVISION INTERACTIVA: Generacion {generation} ---")
        print(f"Generando graficos de {len(current_pop)} individuos en: {review_dir}")
        
        for i, action in enumerate(current_pop):
            img_path = review_dir / f"ind_{i:03d}.png"
            if img_path.exists():
                continue
                
            ax.clear()
            try:
                params = action_to_parameters(action)
                geom = parameters_to_geometry(params)
                geom.validate()
                
                max_radius = max(0.1, geom.cylinder_radius_m)
                max_height = max(0.35, geom.height_m)
                ax.set_xlim(-max_radius * 1.1, max_radius * 1.1)
                ax.set_ylim(-0.02, max_height * 1.05)
                ax.set_aspect('equal')
                
                y, r_outer, y_tube, r_tube = generate_2d_profile(geom)
                
                # Polygono cuerpo cerrado: lado derecho desde abajo hasta tope, luego tubo central bajando, despues volviendo al eje
                outer_x = np.concatenate([
                    [0.0],                        # eje en y=0 (fondo cerrado)
                    r_outer,                      # pared exterior subiendo
                    [r_tube[-1]],                 # tope: empalme con tubo central
                    r_tube[::-1],                 # pared interna del tubo bajando
                    [0.0],                        # boca inferior del tubo (centro)
                ])
                outer_y = np.concatenate([
                    [0.0],
                    y,
                    [geom.height_m],
                    y_tube[::-1],
                    [y_tube[0]],
                ])
                full_x = np.concatenate([outer_x, -outer_x[::-1]])
                full_y = np.concatenate([outer_y, outer_y[::-1]])
                ax.fill(full_x, full_y, color='cyan', alpha=0.3)
                
                # Pared exterior (curva azul) - desde y=0 hasta y=height empalmando con el tubo
                wall_x = np.concatenate([[0.0], r_outer, [r_tube[-1]]])
                wall_y = np.concatenate([[0.0], y, [geom.height_m]])
                ax.plot(wall_x, wall_y, color='blue', linewidth=2)
                ax.plot(-wall_x, wall_y, color='blue', linewidth=2)
                
                # Tubo central (rojo)
                ax.plot(r_tube, y_tube, color='red', linewidth=3)
                ax.plot(-r_tube, y_tube, color='red', linewidth=3)
                ax.plot([-r_tube[0], r_tube[0]], [y_tube[0], y_tube[0]], color='red', linewidth=2, linestyle='--', alpha=0.5)
                
                # Marcar cuello con linea punteada
                ax.axhline(y=geom.neck_height_m, color='gray', linestyle=':', alpha=0.4)
                ax.text(max_radius * 0.95, geom.neck_height_m, f'Cuello',
                        color='gray', va='bottom', ha='right', fontsize=8)
                
                # Inlet
                inlet_y = geom.inlet_height_m
                inlet_x = geom.allowed_radius_at_height(inlet_y)
                inlet_pitch = params.get('inlet_pitch_deg', 0.0)
                inlet_yaw = params.get('inlet_yaw_deg', 70.0)
                ax.plot(inlet_x, inlet_y, marker='o', color='green', markersize=8, zorder=5)
                ax.plot(-inlet_x, inlet_y, marker='o', color='green', markersize=8, zorder=5)

                # En el corte 2D solo se ve la componente RADIAL+VERTICAL del chorro.
                # La componente tangencial (perpendicular al corte) se indica como anillo
                # alrededor del punto de entrada: anillo grande = mas tangencial.
                v_len_full = max_radius * 0.4
                v_len_radial = v_len_full * np.cos(np.radians(inlet_yaw))  # se acorta con yaw alto
                dx = -v_len_radial * np.cos(np.radians(inlet_pitch))
                dy = v_len_radial * np.sin(np.radians(inlet_pitch))
                ax.annotate('', xy=(inlet_x + dx, inlet_y + dy), xytext=(inlet_x, inlet_y),
                            arrowprops=dict(facecolor='green', edgecolor='green', width=2, headwidth=8, shrink=0), zorder=4)
                ax.annotate('', xy=(-inlet_x - dx, inlet_y + dy), xytext=(-inlet_x, inlet_y),
                            arrowprops=dict(facecolor='green', edgecolor='green', width=2, headwidth=8, shrink=0), zorder=4)
                # Indicador visual de tangencialidad: circulo cuyo radio crece con sin(yaw)
                tang_r = max_radius * 0.10 * np.sin(np.radians(inlet_yaw))
                if tang_r > 1e-4:
                    circle1 = plt.Circle((inlet_x, inlet_y), tang_r, color='orange', fill=False, linewidth=2, zorder=4)
                    circle2 = plt.Circle((-inlet_x, inlet_y), tang_r, color='orange', fill=False, linewidth=2, zorder=4)
                    ax.add_patch(circle1)
                    ax.add_patch(circle2)

                ax.set_title(
                    f"Ind {i} | Pitch: {inlet_pitch:+.0f}° | Yaw: {inlet_yaw:.0f}° | "
                    f"Vel: {params.get('flow_velocity_m_s', 0):.2f} m/s"
                )
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.set_xlabel('Radio (m)')
                ax.set_ylabel('Altura (m)')
            except Exception as e:
                ax.text(0.5, 0.5, f"INVALIDO\n{e}", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Individuo {i} (INVALIDO)")
                
            fig.savefig(img_path, dpi=100)
            
        print("Graficos actualizados.")
        print(f"\n>>> ABRE LA CARPETA: {review_dir}")
        print(">>> BORRA las imagenes de las geometrias que NO quieres simular.")
        print(">>> Las imagenes que dejes en la carpeta seran las geometrias aceptadas.")
        print(">>> Cuando termines de borrar, presiona ENTER para continuar.")
        print(">>> O escribe 'regen' para regenerar reemplazos para las que borraste y volver a revisar.")
        print(">>> O escribe 'all' para rechazar todas y volver a generar.")
        
        try:
            resp = input("Accion: ").strip().lower()
        except EOFError:
            print("No hay terminal interactiva. Saltando revision.")
            break
            
        # Detectar que imagenes el usuario dejo en la carpeta (las que conserva)
        kept_indices = set()
        for f in review_dir.glob("ind_*.png"):
            try:
                idx = int(f.stem.split("_")[1])
                kept_indices.add(idx)
            except (ValueError, IndexError):
                continue
                
        all_indices = set(range(len(current_pop)))
        deleted_indices = all_indices - kept_indices
        
        if resp == 'all':
            print("Rechazando TODAS las geometrias y regenerando desde cero...")
            for img in review_dir.glob("ind_*.png"):
                img.unlink()
            for idx in range(len(current_pop)):
                current_pop[idx] = rng.uniform(-1.0, 1.0, size=len(ACTION_NAMES)).astype(np.float32)
            continue
            
        if resp == 'regen':
            if not deleted_indices:
                print("No borraste ninguna imagen. Nada que regenerar.")
                continue
            print(f"Regenerando {len(deleted_indices)} reemplazos para las imagenes borradas...")
            good_indices = sorted(kept_indices)
            for idx in deleted_indices:
                if good_indices and rng.random() < 0.7:
                    base_idx = int(rng.choice(good_indices))
                    base = current_pop[base_idx]
                    mutated = base + rng.normal(0.0, 0.25, size=base.shape)
                    current_pop[idx] = np.clip(mutated, -1.0, 1.0).astype(np.float32)
                else:
                    current_pop[idx] = rng.uniform(-1.0, 1.0, size=len(ACTION_NAMES)).astype(np.float32)
            continue
            
        # Default (Enter): aceptar solo las que quedaron
        if deleted_indices:
            print(f"Aceptando {len(kept_indices)} geometrias. Descartadas {len(deleted_indices)}.")
            current_pop = [current_pop[i] for i in sorted(kept_indices)]
        else:
            print(f"Aceptando todas las {len(current_pop)} geometrias.")
        break
                    
    plt.close(fig)
    return current_pop


def init_population(args: argparse.Namespace, rng: np.random.Generator) -> list[np.ndarray]:
    population: list[np.ndarray] = []
    
    if args.warm_start_jsons:
        for ws_json in args.warm_start_jsons:
            if Path(ws_json).is_file():
                try:
                    payload = json.loads(Path(ws_json).read_text(encoding="utf-8"))
                    if "params" in payload:
                        params = payload["params"]
                        action = params_to_action(params)
                        population.append(action)
                    print(f"Warm start inyectado desde {ws_json}")
                    
                    # Generar algunas mutaciones cercanas a esta semilla
                    for _ in range(max(0, min(8, args.population // (4 * len(args.warm_start_jsons))))):
                        mutated = action + rng.normal(0.0, 0.15, size=action.shape).astype(np.float32)
                        population.append(np.clip(mutated, -1.0, 1.0))
                except Exception as exc:
                    print(f"Error cargando warm start {ws_json}: {exc}")

    while len(population) < args.population:
        action = rng.uniform(-1.0, 1.0, size=len(ACTION_NAMES)).astype(np.float32)
        population.append(action)

    return population


def next_generation(
    population: list[np.ndarray],
    rewards: list[float],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    indices = np.argsort(rewards)[::-1]
    elite_count = max(1, int(args.population * args.elite_fraction))
    elites = [population[i].copy() for i in indices[:elite_count]]

    new_population = list(elites)

    def tournament() -> np.ndarray:
        t_size = min(args.tournament_size, len(population))
        participants = rng.choice(len(population), size=t_size, replace=False)
        best_idx = int(participants[np.argmax([rewards[i] for i in participants])])
        return population[best_idx]

    while len(new_population) < args.population:
        parent1 = tournament()
        parent2 = tournament()

        crossover_mask = rng.random(size=len(ACTION_NAMES)) < 0.5
        child = np.where(crossover_mask, parent1, parent2)

        mutation_mask = rng.random(size=len(ACTION_NAMES)) < args.mutation_prob
        mutations = rng.normal(0.0, args.mutation_sigma, size=len(ACTION_NAMES))
        child = np.where(mutation_mask, child + mutations, child)
        child = np.clip(child, -1.0, 1.0).astype(np.float32)

        new_population.append(child)

    return new_population


def save_best(path: Path, result: dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reward": float(result["reward"]),
        "action": result["action"],
        "params": result["params"],
        "geometry": result["geometry"],
        "metrics": result["metrics"],
        "cfd_config": {
            "base_cells": args.base_cells,
            "particle_end_time": args.particle_end_time,
            "parcels_scale": args.parcels_scale,
        },
        "optimizer": "genetic_openfoam",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(7)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = out_dir / "episodes.csv"
    summary_path = out_dir / "generations.csv"
    best_path = out_dir / "best_geometry.json"

    print(f"Optimizacion CFD (OpenFOAM) iniciada.")
    print(f"Poblacion: {args.population}, Generaciones: {args.generations}")
    print(f"Workers CFD paralelos: {args.n_jobs} (x {args.cores_per_eval} MPI cores = {args.n_jobs * args.cores_per_eval} hilos CPU)")

    population = init_population(args, rng)
    best_result: dict[str, Any] | None = None

    for generation in range(args.generations):
        population = interactive_review(population, generation, out_dir, rng, args)
        
        payloads = [
            (generation, i, pop.tolist(), str(out_dir), args)
            for i, pop in enumerate(population)
        ]

        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            for result in executor.map(evaluate_action_cfd, payloads):
                results.append(result)

        rewards = [res["reward"] for res in results]

        write_header = not episodes_path.exists()
        with episodes_path.open("a", newline="", encoding="utf-8") as handle:
            for result in results:
                row = {
                    "generation": result["generation"],
                    "individual": result["individual"],
                    "reward": result["reward"],
                    "invalid": result["invalid"],
                    "id": result["id"],
                    **(result["params"] or {}),
                    **(result["metrics"] or {}),
                }
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

        valid_results = [res for res in results if not res["invalid"]]
        if not valid_results:
            print(f"gen={generation:03d} FATAL: Ningun individuo valido.")
            population = init_population(args, rng)
            continue

        generation_best = max(valid_results, key=lambda r: float(r["reward"]))
        if best_result is None or float(generation_best["reward"]) > float(best_result["reward"]):
            best_result = generation_best
            save_best(best_path, best_result, args)

        valid_count = len(valid_results)
        summary_row = {
            "generation": generation,
            "best_reward": float(generation_best["reward"]),
            "global_best_reward": float(best_result["reward"]) if best_result else float("nan"),
            "mean_reward": float(np.mean(rewards)),
            "valid_count": valid_count,
        }
        write_header = not summary_path.exists()
        with summary_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_row))
            if write_header:
                writer.writeheader()
            writer.writerow(summary_row)

        print(
            f"gen={generation:03d} best={summary_row['best_reward']:.4f} "
            f"global={summary_row['global_best_reward']:.4f} mean={summary_row['mean_reward']:.4f} "
            f"valid={valid_count}/{args.population}"
        )
        population = next_generation(population, rewards, args, rng)

    print(f"Mejor geometria (CFD): {best_path}")

if __name__ == "__main__":
    main()