import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_sim.rl_env import parameters_to_geometry
from visual_sim.geometry import ClassifierGeometry

def load_best_per_generation(csv_path: Path):
    generations = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['invalid'].lower() == 'true':
                continue
            gen = int(row['generation'])
            reward = float(row['reward'])
            if gen not in generations or reward > generations[gen]['reward']:
                # The CSV contains the metrics too, let's extract them
                metrics = {
                    'target_recovery_pct': float(row.get('target_recovery_pct', 0)),
                    'non_target_rejection_pct': float(row.get('non_target_rejection_pct', 0))
                }
                
                # Extract params
                params = {}
                for k, v in row.items():
                    if k not in ['generation', 'individual', 'reward', 'invalid', 'id', 'target_recovery_pct', 'target_loss_pct', 'non_target_rejection_pct', 'trapped_contamination_pct'] and v:
                        params[k] = float(v)
                        
                generations[gen] = {
                    'reward': reward,
                    'params': params,
                    'metrics': metrics
                }
    
    # Sort by generation
    sorted_gens = sorted(generations.keys())
    return [generations[g] for g in sorted_gens]

def generate_2d_profile(geom: ClassifierGeometry):
    y_samples = 240
    y = np.linspace(0, geom.height_m, y_samples)
    r_outer = np.array([geom.allowed_radius_at_height(yi) for yi in y])

    y_tube = np.linspace(geom.central_tube_bottom_height_m, geom.height_m, max(40, y_samples // 4))
    r_tube = np.full_like(y_tube, geom.central_tube_radius_m)

    return y, r_outer, y_tube, r_tube

def main():
    csv_path = Path("rl_runs/ga_openfoam_res38/episodes.csv")
    if not csv_path.exists():
        print(f"No se encontro {csv_path}")
        return
        
    best_individuals = load_best_per_generation(csv_path)
    print(f"Cargadas {len(best_individuals)} generaciones.")
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    max_radius = 0
    max_height = 0
    for ind in best_individuals:
        geom = parameters_to_geometry(ind['params'])
        max_radius = max(max_radius, geom.cylinder_radius_m)
        max_height = max(max_height, geom.height_m)
        
    def update(frame):
        ax.clear()
        
        # Mantener escala fija (aspect ratio 1:1)
        ax.set_xlim(-max_radius * 1.1, max_radius * 1.1)
        ax.set_ylim(-0.02, max_height * 1.05)
        ax.set_aspect('equal')
        
        ax.set_xlabel('Radio (m)')
        ax.set_ylabel('Altura (m)')
        
        ind = best_individuals[frame]
        geom = parameters_to_geometry(ind['params'])
        
        y, r_outer, y_tube, r_tube = generate_2d_profile(geom)

        # Polygono cerrado del fluido (cuerpo)
        outer_x = np.concatenate([r_outer, r_tube[::-1], [0.0]])
        outer_y = np.concatenate([y, y_tube[::-1], [y_tube[0]]])
        full_x = np.concatenate([outer_x, -outer_x[::-1]])
        full_y = np.concatenate([outer_y, outer_y[::-1]])
        ax.fill(full_x, full_y, color='cyan', alpha=0.25, label='Cuerpo (Agua)')

        # Pared exterior
        ax.plot(-r_outer, y, color='blue', linewidth=2)
        ax.plot(r_outer, y, color='blue', linewidth=2)

        # Tubo central
        ax.plot(-r_tube, y_tube, color='red', linewidth=3)
        ax.plot(r_tube, y_tube, color='red', linewidth=3)
        ax.plot([-r_tube[0], r_tube[0]], [y_tube[0], y_tube[0]],
                color='red', linewidth=2, linestyle='--', alpha=0.5)

        # Cuello (separacion entre lobulos)
        ax.axhline(y=geom.neck_height_m, color='gray', linestyle=':', alpha=0.4)
        ax.text(max_radius * 0.95, geom.neck_height_m, 'Cuello',
                color='gray', va='bottom', ha='right', fontsize=8)
                
        # Dibujar entrada de agua exacta en la pared
        inlet_y = geom.inlet_height_m
        inlet_x = geom.allowed_radius_at_height(inlet_y)
        inlet_pitch = ind['params'].get('inlet_pitch_deg', 0.0)
        
        # Marcar el punto exacto del parche de entrada
        ax.plot(inlet_x, inlet_y, marker='o', color='green', markersize=8, zorder=5)
        ax.plot(-inlet_x, inlet_y, marker='o', color='green', markersize=8, zorder=5) # En el lado opuesto tambien
        
        # Dibujar vector que representa la direccion del chorro de agua
        v_len = max_radius * 0.4
        dx = -v_len * np.cos(np.radians(inlet_pitch))
        dy = v_len * np.sin(np.radians(inlet_pitch))
        
        # Flecha en el lado derecho apuntando hacia adentro
        ax.annotate('', xy=(inlet_x + dx, inlet_y + dy), xytext=(inlet_x, inlet_y),
                    arrowprops=dict(facecolor='green', edgecolor='green', width=3, headwidth=10, shrink=0),
                    zorder=4)
                    
        # Flecha en el lado izquierdo apuntando hacia adentro
        ax.annotate('', xy=(-inlet_x - dx, inlet_y + dy), xytext=(-inlet_x, inlet_y),
                    arrowprops=dict(facecolor='green', edgecolor='green', width=3, headwidth=10, shrink=0),
                    zorder=4)
                    
        ax.text(inlet_x + max_radius*0.15, inlet_y, f'Inlet\nPitch: {inlet_pitch:.1f}°', 
                color='green', va='center', ha='left', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Titulo
        ax.set_title(f"Generacion {frame}\nReward: {ind['reward']:.4f} | Vel. Inyeccion: {ind['params']['flow_velocity_m_s']:.2f} m/s\n"
                     f"Recuperacion (Oro): {ind['metrics']['target_recovery_pct']:.1f}% | "
                     f"Rechazo (Arena): {ind['metrics']['non_target_rejection_pct']:.1f}%")
        ax.grid(True, linestyle='--', alpha=0.5)
        
    print("Guardando imagenes individuales por generacion...")
    frames_dir = Path("rl_runs/ga_openfoam_res38/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    for frame in range(len(best_individuals)):
        update(frame)
        frame_path = frames_dir / f"generacion_{frame:03d}.png"
        fig.savefig(frame_path, dpi=150, bbox_inches='tight')
        
    print(f"Imagenes guardadas en: {frames_dir}")
    
    print("Generando animacion 2D (es rapido)...")
    anim = animation.FuncAnimation(fig, update, frames=len(best_individuals), interval=1000, repeat_delay=3000)
    
    output_path = Path("rl_runs/ga_openfoam_res38/evolution_2d.gif")
    anim.save(output_path, writer='pillow', fps=2, dpi=150)
    print(f"Animacion guardada en: {output_path}")

if __name__ == "__main__":
    main()