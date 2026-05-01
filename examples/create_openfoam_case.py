from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea un caso OpenFOAM base desde una exportacion STL.")
    parser.add_argument("--export-dir", required=True, help="Directorio creado por export_geometry_cfd.py.")
    parser.add_argument("--stl-name", required=True, help="Nombre del STL dentro de export-dir.")
    parser.add_argument("--metadata", required=True, help="Metadata JSON creada por export_geometry_cfd.py.")
    parser.add_argument("--case-dir", required=True, help="Directorio del caso OpenFOAM a crear.")
    parser.add_argument("--base-cells", type=int, default=44, help="Resolucion base por eje horizontal.")
    parser.add_argument("--refinement-level", type=int, default=2, help="Nivel snappyHexMesh sobre el STL.")
    return parser.parse_args()


def foam_header(class_name: str, object_name: str) -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| OpenFOAM case generated from AurumFlow geometry export                      |
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


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inlet_velocity_vector(metadata: dict) -> tuple[float, float, float]:
    velocity = float(metadata["geometry"]["inlet_velocity_m_s"])
    angle = math.radians(float(metadata["derived"].get("inlet_angle_deg", -90.0)))
    inward = (-math.cos(angle), 0.0, -math.sin(angle))
    tangent = (-math.sin(angle), 0.0, math.cos(angle))
    swirl_weight = 0.75
    radial_weight = 0.65
    vector = (
        velocity * (radial_weight * inward[0] + swirl_weight * tangent[0]),
        0.0,
        velocity * (radial_weight * inward[2] + swirl_weight * tangent[2]),
    )
    return vector


def patch_names(stl_prefix: str) -> dict[str, str]:
    return {
        "outer_wall": f"{stl_prefix}_outer_wall",
        "inlet": f"{stl_prefix}_inlet",
        "roof": f"{stl_prefix}_roof",
        "overflow_tube_wall": f"{stl_prefix}_overflow_tube_wall",
        "overflow_mouth": f"{stl_prefix}_overflow_mouth",
        "trap_floor": f"{stl_prefix}_trap_floor",
    }


def mesh_patch_names(stl_prefix: str) -> dict[str, str]:
    return {key: f"classifier_{value}" for key, value in patch_names(stl_prefix).items()}


def write_block_mesh(case_dir: Path, metadata: dict, base_cells: int) -> None:
    radius = float(metadata["derived"]["max_domain_radius_m"]) * 1.18
    height = float(metadata["derived"]["height_m"]) * 1.08
    cells_xz = max(16, base_cells)
    cells_y = max(16, int(base_cells * height / (2.0 * radius)))
    text = foam_header("dictionary", "blockMeshDict") + f"""
scale 1;

vertices
(
    ({-radius:.8f} {-0.02 * height:.8f} {-radius:.8f})
    ({ radius:.8f} {-0.02 * height:.8f} {-radius:.8f})
    ({ radius:.8f} {-0.02 * height:.8f} { radius:.8f})
    ({-radius:.8f} {-0.02 * height:.8f} { radius:.8f})
    ({-radius:.8f} { height:.8f} {-radius:.8f})
    ({ radius:.8f} { height:.8f} {-radius:.8f})
    ({ radius:.8f} { height:.8f} { radius:.8f})
    ({-radius:.8f} { height:.8f} { radius:.8f})
);

blocks
(
    hex (0 3 2 1 4 7 6 5) ({cells_xz} {cells_y} {cells_xz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    background
    {{
        type patch;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (1 2 6 5)
            (2 3 7 6)
            (3 0 4 7)
        );
    }}
);

mergePatchPairs
(
);
"""
    (case_dir / "system" / "blockMeshDict").write_text(text, encoding="utf-8")


def write_surface_features(case_dir: Path, stl_name: str) -> None:
    text = foam_header("dictionary", "surfaceFeatureExtractDict") + f"""
{stl_name}
{{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs
    {{
        includedAngle   150;
    }}
    writeObj            yes;
}}
"""
    (case_dir / "system" / "surfaceFeatureExtractDict").write_text(text, encoding="utf-8")


def write_snappy(case_dir: Path, metadata: dict, stl_name: str, stl_prefix: str, refinement_level: int) -> None:
    height = float(metadata["derived"]["height_m"])
    max_radius = float(metadata["derived"]["max_domain_radius_m"])
    body_top_radius = float(metadata["derived"].get("body_top_radius_m", 0.45 * max_radius))
    overflow_radius = float(metadata["derived"]["overflow_tube_radius_m"])
    patches = patch_names(stl_prefix)
    location_y = 0.52 * height
    # Keep a point in the annular classifier body, away from the central overflow
    # tube/axis. A point at x=z=0 can make snappyHexMesh retain the exterior box.
    location_x = max(2.2 * overflow_radius, min(0.35 * max_radius, 0.75 * body_top_radius))
    text = foam_header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    {stl_name}
    {{
        type triSurfaceMesh;
        name classifier;
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 250000;
    maxGlobalCells 2500000;
    minRefinementCells 10;
    nCellsBetweenLevels 3;

    features
    (
        {{
            file "{Path(stl_name).with_suffix('.eMesh').name}";
            level 1;
        }}
    );

    refinementSurfaces
    {{
        classifier
        {{
            level ({refinement_level} {refinement_level});
            regions
            {{
                {patches["outer_wall"]} {{ level ({refinement_level} {refinement_level}); patchInfo {{ type wall; }} }}
                {patches["inlet"]} {{ level ({refinement_level + 1} {refinement_level + 1}); patchInfo {{ type patch; }} }}
                {patches["roof"]} {{ level ({refinement_level} {refinement_level}); patchInfo {{ type wall; }} }}
                {patches["overflow_tube_wall"]} {{ level ({refinement_level + 1} {refinement_level + 1}); patchInfo {{ type wall; }} }}
                {patches["overflow_mouth"]} {{ level ({refinement_level + 1} {refinement_level + 1}); patchInfo {{ type patch; }} }}
                {patches["trap_floor"]} {{ level ({refinement_level} {refinement_level}); patchInfo {{ type wall; }} }}
            }}
        }}
    }}

    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh ({location_x:.8f} {location_y:.8f} 0);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap true;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{}}
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 70;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
"""
    (case_dir / "system" / "snappyHexMeshDict").write_text(text, encoding="utf-8")


def write_case_controls(case_dir: Path) -> None:
    (case_dir / "system" / "controlDict").write_text(
        foam_header("dictionary", "controlDict")
        + """
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         300;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""",
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSchemes").write_text(
        foam_header("dictionary", "fvSchemes")
        + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""",
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSolution").write_text(
        foam_header("dictionary", "fvSolution")
        + """
solvers
{
    p
    {
        solver GAMG;
        tolerance 1e-7;
        relTol 0.01;
        smoother GaussSeidel;
    }
    U
    {
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-8;
        relTol 0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
}

relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; }
}
""",
        encoding="utf-8",
    )


def write_physical_properties(case_dir: Path) -> None:
    (case_dir / "constant" / "transportProperties").write_text(
        foam_header("dictionary", "transportProperties")
        + """
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1e-06;
""",
        encoding="utf-8",
    )
    (case_dir / "constant" / "momentumTransport").write_text(
        foam_header("dictionary", "momentumTransport")
        + """
simulationType laminar;
""",
        encoding="utf-8",
    )
    (case_dir / "constant" / "turbulenceProperties").write_text(
        foam_header("dictionary", "turbulenceProperties")
        + """
simulationType laminar;
""",
        encoding="utf-8",
    )


def write_initial_fields(case_dir: Path, metadata: dict, stl_prefix: str) -> None:
    patches = mesh_patch_names(stl_prefix)
    ux, uy, uz = inlet_velocity_vector(metadata)
    wall_patches = [
        patches["outer_wall"],
        patches["roof"],
        patches["overflow_tube_wall"],
        patches["trap_floor"],
    ]
    wall_u_entries = "\n".join(
        f"""    {patch}
    {{
        type noSlip;
    }}"""
        for patch in wall_patches
    )
    wall_p_entries = "\n".join(
        f"""    {patch}
    {{
        type zeroGradient;
    }}"""
        for patch in wall_patches
    )
    (case_dir / "0" / "U").write_text(
        foam_header("volVectorField", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    {patches["inlet"]}
    {{
        type fixedValue;
        value uniform ({ux:.8f} {uy:.8f} {uz:.8f});
    }}
    {patches["overflow_mouth"]}
    {{
        type pressureInletOutletVelocity;
        value uniform (0 0 0);
    }}
{wall_u_entries}
}}
""",
        encoding="utf-8",
    )
    (case_dir / "0" / "p").write_text(
        foam_header("volScalarField", "p")
        + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    {patches["inlet"]}
    {{
        type zeroGradient;
    }}
    {patches["overflow_mouth"]}
    {{
        type fixedValue;
        value uniform 0;
    }}
{wall_p_entries}
}}
""",
        encoding="utf-8",
    )


def write_allrun(case_dir: Path) -> None:
    (case_dir / "Allrun").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
checkMesh
""",
        encoding="utf-8",
    )
    (case_dir / "AllrunFlow").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

simpleFoam
""",
        encoding="utf-8",
    )
    (case_dir / "Allclean").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

rm -rf [1-9]* processor* postProcessing constant/polyMesh constant/extendedFeatureEdgeMesh
find . -name '*.log' -delete
""",
        encoding="utf-8",
    )
    (case_dir / "Allrun").chmod(0o755)
    (case_dir / "AllrunFlow").chmod(0o755)
    (case_dir / "Allclean").chmod(0o755)


def write_paraview_foam(case_dir: Path) -> None:
    """Crea un .foam vacio: ParaView identifica casos OpenFOAM por extension."""
    (case_dir / f"{case_dir.name}.foam").write_text("", encoding="utf-8")


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir)
    metadata = read_metadata(Path(args.metadata))
    case_dir = Path(args.case_dir)
    stl_source = export_dir / args.stl_name
    stl_prefix = Path(args.stl_name).stem.replace("_internal_volume", "")

    for subdir in ("0", "constant/triSurface", "system"):
        (case_dir / subdir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stl_source, case_dir / "constant" / "triSurface" / args.stl_name)

    write_block_mesh(case_dir, metadata, args.base_cells)
    write_surface_features(case_dir, args.stl_name)
    write_snappy(case_dir, metadata, args.stl_name, stl_prefix, args.refinement_level)
    write_case_controls(case_dir)
    write_physical_properties(case_dir)
    write_initial_fields(case_dir, metadata, stl_prefix)
    write_allrun(case_dir)
    write_paraview_foam(case_dir)

    print(f"Caso OpenFOAM creado en: {case_dir}")
    print("Dentro del contenedor:")
    print(f"  cd {case_dir}")
    print("  ./Allrun")


if __name__ == "__main__":
    main()
