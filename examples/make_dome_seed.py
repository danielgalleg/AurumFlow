"""Crea una semilla 'reloj de arena clasico' bien proporcionada como warm-start
para el GA. Con el nuevo set de 13 parametros de la geometria Clepsamia, el
perfil ya es C^1-suave por construccion (no necesita curvas).
"""
import json
from pathlib import Path

dome_path = Path("rl_runs/ga_openfoam_clepsamia/seed_clepsamia.json")
dome_path.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "params": {
        "height_m": 0.22,
        "neck_height_ratio": 0.50,
        "neck_radius_m": 0.012,
        "upper_max_radius_m": 0.060,
        "upper_max_position_ratio": 0.55,
        "lower_max_radius_m": 0.055,
        "lower_max_position_ratio": 0.55,
        "inlet_height_ratio": 0.65,
        "inlet_pitch_deg": 30.0,
        "inlet_angle_deg": -90.0,
        "inlet_yaw_deg": 75.0,
        "central_tube_radius_m": 0.008,
        "central_tube_bottom_ratio": 0.45,
        "flow_velocity_m_s": 0.50,
    }
}

dome_path.write_text(json.dumps(payload, indent=2))
print(f"Semilla Clepsamia creada en: {dome_path}")
