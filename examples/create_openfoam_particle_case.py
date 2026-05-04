from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_sim import ClassifierGeometry


DEFAULT_MATERIALS = [
    {
        "name": "gold",
        "label": "Oro",
        "density_kg_m3": 19300.0,
        "diameter_m": 0.00035,
        "parcels": 500,
        "target": True,
    },
    {
        "name": "quartz_sand",
        "label": "Arena cuarzo",
        "density_kg_m3": 2650.0,
        "diameter_m": 0.00035,
        "parcels": 1200,
        "target": False,
    },
    {
        "name": "magnetite",
        "label": "Magnetita",
        "density_kg_m3": 5200.0,
        "diameter_m": 0.00030,
        "parcels": 600,
        "target": False,
    },
]


@dataclass(frozen=True)
class Material:
    name: str
    label: str
    density_kg_m3: float
    diameter_m: float
    parcels: int
    target: bool


def foam_header(class_name: str, object_name: str) -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM particle validation case generated from AurumFlow                  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea casos OpenFOAM lagrangianos por material sobre un campo CFD existente."
    )
    parser.add_argument("--base-case", required=True, help="Caso CFD con polyMesh y campo U ya resuelto.")
    parser.add_argument("--metadata", required=True, help="Metadata JSON del export CFD.")
    parser.add_argument("--output-root", required=True, help="Directorio donde crear los casos por material.")
    parser.add_argument("--materials-json", default=None, help="JSON opcional con lista de materiales.")
    parser.add_argument("--flow-time", default="latest", help="Tiempo CFD del campo U a copiar, o latest.")
    parser.add_argument("--end-time", type=float, default=2.0, help="Tiempo de tracking de particulas.")
    parser.add_argument("--delta-t", type=float, default=2.0e-4, help="Paso temporal del solver de particulas.")
    parser.add_argument("--write-interval", type=float, default=0.05, help="Intervalo de escritura.")
    parser.add_argument("--seed", type=int, default=17, help="Semilla para posiciones iniciales.")
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="Escala para U0 inicial de particulas respecto de la velocidad de entrada.",
    )
    parser.add_argument(
        "--collision-model",
        choices=("none", "pairCollision"),
        default="none",
        help="none para barridos rapidos; pairCollision para validacion densa mas costosa.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=14,
        help="Numero de nucleos para MPI de particulas.",
    )
    parser.add_argument(
        "--parcels-scale",
        type=float,
        default=1.0,
        help="Escala la cantidad de parcels de cada material para pruebas rapidas.",
    )
    return parser.parse_args()


def load_geometry_from_metadata(metadata: dict) -> ClassifierGeometry:
    geometry = ClassifierGeometry(**metadata["geometry"])
    geometry.validate()
    return geometry


def material_from_dict(payload: dict) -> Material:
    return Material(
        name=str(payload["name"]),
        label=str(payload.get("label", payload["name"])),
        density_kg_m3=float(payload["density_kg_m3"]),
        diameter_m=float(payload["diameter_m"]),
        parcels=int(payload["parcels"]),
        target=bool(payload.get("target", False)),
    )


def load_materials(path: str | None) -> list[Material]:
    if path is None:
        return [material_from_dict(item) for item in DEFAULT_MATERIALS]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("materials", [])
    return [material_from_dict(item) for item in payload]


def latest_time_dir(case_dir: Path) -> Path:
    candidates = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if (child / "U").is_file():
            candidates.append((value, child))
    if not candidates:
        raise FileNotFoundError(f"No encontre tiempos con campo U en {case_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def selected_flow_time(base_case: Path, flow_time: str) -> Path:
    if flow_time == "latest":
        return latest_time_dir(base_case)
    selected = base_case / flow_time
    if not (selected / "U").is_file():
        raise FileNotFoundError(f"No existe {selected / 'U'}")
    return selected


def inlet_velocity_vector(metadata: dict, scale: float) -> tuple[float, float, float]:
    velocity = float(metadata["geometry"]["inlet_velocity_m_s"]) * scale
    angle = math.radians(float(metadata["derived"].get("inlet_angle_deg", -90.0)))
    pitch = math.radians(float(metadata["derived"].get("inlet_pitch_deg", 0.0)))
    inward = (-math.cos(angle), 0.0, -math.sin(angle))
    tangent = (-math.sin(angle), 0.0, math.cos(angle))
    swirl_weight = 0.75
    radial_weight = 0.65
    
    # Calculate horizontal components
    vx = velocity * math.cos(pitch) * (radial_weight * inward[0] + swirl_weight * tangent[0])
    vz = velocity * math.cos(pitch) * (radial_weight * inward[2] + swirl_weight * tangent[2])
    # Calculate vertical component
    vy = velocity * math.sin(pitch)
    
    return (vx, vy, vz)


def generate_inlet_positions(
    geometry: ClassifierGeometry,
    metadata: dict,
    parcels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    angle = math.radians(float(metadata["derived"].get("inlet_angle_deg", -90.0)))
    width = math.radians(float(metadata["derived"].get("inlet_angular_width_deg", 28.0)))
    inlet_height = float(metadata["derived"].get("inlet_height_m", geometry.inlet_height_m))
    inlet_patch_height = float(metadata["derived"].get("inlet_patch_height_m", 0.035))
    theta = rng.uniform(angle - 0.35 * width, angle + 0.35 * width, size=parcels)
    y = rng.uniform(
        max(0.0, inlet_height - 0.35 * inlet_patch_height),
        min(geometry.height_m, inlet_height + 0.35 * inlet_patch_height),
        size=parcels,
    )
    radii = np.array([geometry.allowed_radius_at_height(float(y_i)) for y_i in y], dtype=float)
    radius = radii * rng.uniform(0.82, 0.94, size=parcels)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def write_positions(path: Path, positions: np.ndarray) -> None:
    lines = [
        foam_header("vectorField", "kinematicCloudPositions"),
        f"{len(positions)}\n",
        "(\n",
    ]
    for x, y, z in positions:
        lines.append(f"({x:.9g} {y:.9g} {z:.9g})\n")
    lines.append(")\n")
    path.write_text("".join(lines), encoding="utf-8")


def copy_u_field(source_u: Path, target_u: Path) -> None:
    text = source_u.read_text(encoding="utf-8")
    text = text.replace(f'location    "{source_u.parent.name}";', 'location    "0";')
    target_u.write_text(text, encoding="utf-8")


def write_g(case_dir: Path) -> None:
    (case_dir / "constant" / "g").write_text(
        foam_header("uniformDimensionedVectorField", "g")
        + """
dimensions      [0 1 -2 0 0 0 0];
value           (0 -9.81 0);
""",
        encoding="utf-8",
    )


def write_transport(case_dir: Path) -> None:
    (case_dir / "constant" / "transportProperties").write_text(
        foam_header("dictionary", "transportProperties")
        + """
rhoInf          1000;
transportModel  Newtonian;
nu              1e-06;
""",
        encoding="utf-8",
    )
    (case_dir / "constant" / "turbulenceProperties").write_text(
        foam_header("dictionary", "turbulenceProperties")
        + """
simulationType  laminar;
""",
        encoding="utf-8",
    )


def write_system(case_dir: Path, end_time: float, delta_t: float, write_interval: float, cores: int) -> None:
    if cores > 1:
        (case_dir / "system" / "decomposeParDict").write_text(
            foam_header("dictionary", "decomposeParDict")
            + f"""
numberOfSubdomains {cores};
method          scotch;
""",
            encoding="utf-8",
        )
    (case_dir / "system" / "controlDict").write_text(
        foam_header("dictionary", "controlDict")
        + f"""
application     icoUncoupledKinematicParcelFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.8g};
deltaT          {delta_t:.8g};
writeControl    runTime;
writeInterval   {write_interval:.8g};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;
""",
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSchemes").write_text(
        foam_header("dictionary", "fvSchemes")
        + """
ddtSchemes { default none; }
gradSchemes { default none; }
divSchemes { default none; }
laplacianSchemes { default none; }
interpolationSchemes { default linear; }
""",
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSolution").write_text(
        foam_header("dictionary", "fvSolution") + "\n",
        encoding="utf-8",
    )


def write_cloud_properties(
    case_dir: Path,
    material: Material,
    velocity: tuple[float, float, float],
    patches: dict[str, str],
    collision_model: str,
) -> None:
    max_interaction = max(2.5 * material.diameter_m, material.diameter_m + 1.0e-6)
    wall_entries = "\n".join(
        f"""            {patch}
            {{
                youngsModulus   1e9;
                poissonsRatio   0.23;
                alpha           0.10;
                b               1.5;
                mu              0.35;
                cohesionEnergyDensity 0;
            }}"""
        for patch in (
            patches["outer_wall"],
            patches["roof"],
            patches["overflow_tube_wall"],
            patches["trap_floor"],
        )
    )
    collision_block = "    collisionModel none;\n"
    if collision_model == "pairCollision":
        collision_block = f"""    collisionModel pairCollision;

    pairCollisionCoeffs
    {{
        maxInteractionDistance  {max_interaction:.8g};
        writeReferredParticleCloud no;

        pairModel pairSpringSliderDashpot;

        pairSpringSliderDashpotCoeffs
        {{
            useEquivalentSize   no;
            alpha               0.08;
            b                   1.5;
            mu                  0.35;
            cohesionEnergyDensity 0;
            collisionResolutionSteps 12;
        }};

        wallModel    wallLocalSpringSliderDashpot;

        wallLocalSpringSliderDashpotCoeffs
        {{
            useEquivalentSize no;
            collisionResolutionSteps 12;
{wall_entries}
        }};
    }}
"""
    (case_dir / "constant" / "kinematicCloudProperties").write_text(
        foam_header("dictionary", "kinematicCloudProperties")
        + f"""
solution
{{
    active          true;
    coupled         false;
    transient       yes;
    cellValueSourceCorrection off;
    maxCo           0.3;

    interpolationSchemes
    {{
        rho             cell;
        U               cellPoint;
        mu              cell;
    }}

    integrationSchemes
    {{
        U               Euler;
    }}
}}

constantProperties
{{
    rho0            {material.density_kg_m3:.8g};
    youngsModulus   1e9;
    poissonsRatio   0.35;
}}

subModels
{{
    particleForces
    {{
        sphereDrag;
        gravity;
    }}

    injectionModels
    {{
        model1
        {{
            type            manualInjection;
            massTotal       0;
            parcelBasisType fixed;
            nParticle       1;
            SOI             0;
            positionsFile   "kinematicCloudPositions";
            U0              ({velocity[0]:.8g} {velocity[1]:.8g} {velocity[2]:.8g});
            sizeDistribution
            {{
                type        fixedValue;
                fixedValueDistribution
                {{
                    value   {material.diameter_m:.8g};
                }}
            }}
        }}
    }}

    dispersionModel none;

    patchInteractionModel localInteraction;

    localInteractionCoeffs
    {{
        patches
        (
            {patches["overflow_mouth"]}
            {{
                type escape;
            }}
            {patches["inlet"]}
            {{
                type escape;
            }}
            {patches["trap_floor"]}
            {{
                type stick;
            }}
            {patches["outer_wall"]}
            {{
                type rebound;
                e    0.85;
                mu   0.35;
            }}
            {patches["roof"]}
            {{
                type rebound;
                e    0.85;
                mu   0.35;
            }}
            {patches["overflow_tube_wall"]}
            {{
                type rebound;
                e    0.85;
                mu   0.35;
            }}
        );
    }}

    surfaceFilmModel none;

    stochasticCollisionModel none;

{collision_block}
}}

cloudFunctions
{{
    ReynoldsNumber1
    {{
        type    ReynoldsNumber;
    }}
}}
""",
        encoding="utf-8",
    )


def mesh_patch_names(stl_prefix: str) -> dict[str, str]:
    return {
        "outer_wall": f"classifier_{stl_prefix}_outer_wall",
        "inlet": f"classifier_{stl_prefix}_inlet",
        "roof": f"classifier_{stl_prefix}_roof",
        "overflow_tube_wall": f"classifier_{stl_prefix}_overflow_tube_wall",
        "overflow_mouth": f"classifier_{stl_prefix}_overflow_mouth",
        "trap_floor": f"classifier_{stl_prefix}_trap_floor",
    }


def write_manifest(
    case_dir: Path,
    base_case: Path,
    flow_time: Path,
    material: Material,
    metadata: dict,
    positions: np.ndarray,
    collision_model: str,
) -> None:
    manifest = {
        "base_case": str(base_case),
        "flow_time": flow_time.name,
        "solver": "icoUncoupledKinematicParcelFoam",
        "collision_model": collision_model,
        "material": material.__dict__,
        "initial_count": int(len(positions)),
        "trap_height_m": float(metadata["geometry"]["trap_height_m"]),
        "trap_bottom_radius_m": float(metadata["derived"]["trap_bottom_radius_m"]),
        "overflow_tube_radius_m": float(metadata["derived"]["overflow_tube_radius_m"]),
        "overflow_tube_bottom_height_m": float(metadata["derived"]["overflow_tube_bottom_height_m"]),
        "height_m": float(metadata["derived"]["height_m"]),
        "notes": [
            "Caso por material: rho0 es unico por cloud en icoUncoupledKinematicParcelFoam.",
            "El campo U se copia desde el caso CFD y se usa como flujo congelado.",
            "Esta validacion incluye arrastre, gravedad y rebote/stick de paredes.",
            "Use --collision-model pairCollision para activar colision parcel-parcel simple con mayor costo.",
        ],
    }
    (case_dir / "particle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_allrun(case_dir: Path) -> None:
    (case_dir / "AllrunParticles").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

icoUncoupledKinematicParcelFoam | tee particleFoam.log
""",
        encoding="utf-8",
    )
    (case_dir / "AllcleanParticles").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

rm -rf [1-9]* 0.* postProcessing processor* particleFoam.log
""",
        encoding="utf-8",
    )
    (case_dir / "AllrunParticles").chmod(0o755)
    (case_dir / "AllcleanParticles").chmod(0o755)


def write_root_runner(output_root: Path, cases: Iterable[Path]) -> None:
    lines = ["#!/usr/bin/env bash\n", "set -euo pipefail\n\n"]
    for case in cases:
        lines.append(f'(cd "{case.name}" && ./AllcleanParticles && ./AllrunParticles)\n')
    (output_root / "AllrunParticles").write_text("".join(lines), encoding="utf-8")
    (output_root / "AllrunParticles").chmod(0o755)


def create_case(
    output_root: Path,
    base_case: Path,
    flow_time: Path,
    metadata: dict,
    geometry: ClassifierGeometry,
    stl_prefix: str,
    material: Material,
    args: argparse.Namespace,
    material_index: int,
) -> Path:
    case_dir = output_root / material.name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    for subdir in ("0", "constant", "system"):
        (case_dir / subdir).mkdir(parents=True, exist_ok=True)

    shutil.copytree(base_case / "constant" / "polyMesh", case_dir / "constant" / "polyMesh")
    tri_surface = base_case / "constant" / "triSurface"
    if tri_surface.is_dir():
        shutil.copytree(tri_surface, case_dir / "constant" / "triSurface")
    copy_u_field(flow_time / "U", case_dir / "0" / "U")

    parcel_count = max(1, int(round(material.parcels * max(0.001, args.parcels_scale))))
    run_material = Material(
        name=material.name,
        label=material.label,
        density_kg_m3=material.density_kg_m3,
        diameter_m=material.diameter_m,
        parcels=parcel_count,
        target=material.target,
    )
    rng = np.random.default_rng(args.seed + 1009 * material_index)
    positions = generate_inlet_positions(geometry, metadata, run_material.parcels, rng)
    write_positions(case_dir / "constant" / "kinematicCloudPositions", positions)

    patches = mesh_patch_names(stl_prefix)
    velocity = inlet_velocity_vector(metadata, args.velocity_scale)
    write_g(case_dir)
    write_transport(case_dir)
    write_system(case_dir, args.end_time, args.delta_t, args.write_interval, args.cores)
    write_cloud_properties(case_dir, run_material, velocity, patches, args.collision_model)
    write_manifest(case_dir, base_case, flow_time, run_material, metadata, positions, args.collision_model)
    write_allrun(case_dir)
    (case_dir / f"{case_dir.name}.foam").write_text("", encoding="utf-8")
    return case_dir


def main() -> None:
    args = parse_args()
    base_case = Path(args.base_case)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    flow_time = selected_flow_time(base_case, args.flow_time)
    materials = load_materials(args.materials_json)
    geometry = load_geometry_from_metadata(metadata)
    stl_prefix = Path(next((base_case / "constant" / "triSurface").glob("*_internal_volume.stl"))).stem.replace(
        "_internal_volume", ""
    )

    cases = [
        create_case(output_root, base_case, flow_time, metadata, geometry, stl_prefix, material, args, idx)
        for idx, material in enumerate(materials)
    ]
    write_root_runner(output_root, cases)
    print(f"Casos de particulas creados en: {output_root}")
    print(f"Campo U usado: {flow_time}")
    print("Dentro del contenedor:")
    print(f"  cd {output_root}")
    print("  ./AllrunParticles")


if __name__ == "__main__":
    main()
