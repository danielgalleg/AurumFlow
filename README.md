# AurumFlow

Experimental tools for designing and validating a compact hydraulic classifier for small-scale alluvial gold separation.

The project combines advanced parametric geometry exploration, evolutionary optimization, OpenFOAM CFD export, multi-material particle tracking, and ParaView visualization. It is intended as a research and prototyping aid for creating highly efficient, low-power separation devices, not as a certified mining, safety, or recovery guarantee.

## Project Status

This repository is experimental and under active development. The current focus has shifted from traditional hydrocyclones to **upflow elutriators** after the Genetic Algorithm (GA) discovered that slow, upward-directed flows are vastly superior for achieving >99% pure concentrates while operating on low-power 12V diaphragm pumps.

Current capabilities:

- **"Clepsamia" (Hourglass) Geometry**: A fully C¹-continuous 3D parameterization featuring two curved lobes, a narrow neck, a central outlet tube, and an actively positioned/angled inlet. There are no flat surfaces or vertices.
- **Multiplicative Reward Function**: The GA evaluates designs using a strict `4.0 * recovery * purity * (1 - loss)` formula, penalizing any geometry that sacrifices gold recovery for sediment rejection or vice versa.
- **OpenFOAM-Driven Genetic Algorithm**: Direct coupling of evolutionary optimization with high-fidelity OpenFOAM CFD evaluation (`train_geometry_ga_openfoam.py`).
- **Interactive File-Based Review**: The GA allows manual filtering of geometries before expensive CFD runs by simply deleting unwanted 2D profile images from a folder.
- **Multi-Material Particle Tracking**: Native OpenFOAM particle tracking evaluating Gold (target), Quartz Sand, and Magnetite simultaneously.
- **Automated STL/Mesh/Case Generation**: Zero-touch generation of `snappyHexMesh` and `simpleFoam` fluid flow cases.

Important limitation: while OpenFOAM is robust, the current particle solver (`icoUncoupledKinematicParcelFoam`) is uncoupled (particles don't affect fluid or each other). The next step for validating the dense concentrate bed is full CFD+DEM.

## Repository Layout

```text
examples/
  train_geometry_ga_openfoam.py     # Main hybrid GA/OpenFOAM optimizer with interactive review
  evaluate_geometry_openfoam.py     # Single geometry OpenFOAM evaluation script
  export_geometry_cfd.py            # Generates STL and metadata from Clepsamia geometry
  create_openfoam_case.py           # Configures simpleFoam fluid cases (inlet yaw/pitch)
  create_openfoam_particle_case.py  # Configures multi-material particle tracking
  analyze_openfoam_particles.py     # Calculates recovery/rejection/contamination metrics
  plot_all_evaluations.py           # 2D profile grid of all evaluated geometries
  animate_evolution.py              # Creates GIFs of the GA's progression
  make_dome_seed.py                 # Generates warm-start JSONs
  paraview_openfoam_view.py         # Scripted ParaView visualization

visual_sim/
  geometry.py                       # C¹-continuous Clepsamia math and parameterization
  rl_env.py                         # Action spaces, parameter bounds, and physics config
```

Generated outputs such as `rl_runs/`, `cfd_cases/`, `cfd_exports/`, images, and videos are intentionally ignored by Git.

## Installation

Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

OpenFOAM is currently used through Docker:

```bash
docker pull opencfd/openfoam-default:latest
```

ParaView is used from the host system:

```bash
paraview
pvpython --version
```

## Quick Start

The recommended workflow is to run the OpenFOAM-coupled Genetic Algorithm. This script automatically meshes, simulates fluid flow, tracks particles, and computes the multiplicative reward.

### 1. Run the Optimizer

You can warm-start the optimizer using an existing highly-performing geometry JSON to speed up convergence:

```bash
python3 examples/train_geometry_ga_openfoam.py \
  --generations 60 \
  --population 64 \
  --base-cells 44 \
  --particle-end-time 4.0 \
  --parcels-scale 0.04 \
  --cores-per-eval 2 \
  --n-jobs 7 \
  --output-dir rl_runs/ga_clepsamia_run \
  --warm-start-jsons rl_runs/ga_clepsamia_v5_max_bounds/best_geometry.json \
  --mutation-prob 0.40 \
  --mutation-sigma 0.30
```

### 2. Interactive Review

If the `--interactive` flag is set (default behavior), the script will pause before every generation. It generates 2D profiles of all candidate geometries in a `review_gen_XXX` folder.
- **To accept**: Leave the image in the folder.
- **To reject**: Delete the image file.
- Type `Enter` in the terminal to proceed, `regen` to replace deleted geometries with mutations of the survivors, or `all` to reject everything and try again.

### 3. Analyze Results

Once the run completes (or is paused), you can generate a grid showing all tested geometries:

```bash
python3 examples/plot_all_evaluations.py \
  --run-dir rl_runs/ga_clepsamia_run
```

To create an animation of the evolutionary progress:

```bash
python3 examples/animate_evolution.py \
  --run-dir rl_runs/ga_clepsamia_run
```

## Single Geometry Evaluation Workflow

If you want to manually test a specific JSON geometry without running the GA:

### 1. Export the STL

```bash
python3 examples/export_geometry_cfd.py \
  --geometry-json rl_runs/ga_clepsamia_run/best_geometry.json \
  --output-dir cfd_exports/manual_test \
  --name clepsamia
```

### 2. Evaluate with OpenFOAM

This script automates `create_openfoam_case.py`, `create_openfoam_particle_case.py`, running the Docker container, and `analyze_openfoam_particles.py`:

```bash
python3 examples/evaluate_geometry_openfoam.py \
  --geometry-json rl_runs/ga_clepsamia_run/best_geometry.json \
  --output-root cfd_cases/manual_test \
  --name-prefix test \
  --base-cells 44 \
  --particle-end-time 5.0
```

### 3. Visualize with ParaView

```bash
pvpython examples/paraview_openfoam_view.py \
  --foam cfd_cases/manual_test/test_000/case/case.foam \
  --particles-csv cfd_cases/manual_test/test_000/particles/particles_latest.csv
```

## CFD+DEM Direction

OpenFOAM native particle tracking is extremely fast for optimization but does not model inter-particle collisions or the displacement of fluid by a dense bed of trapped concentrate.

Recommended next fidelity levels:

- `MPPICFoam`: native OpenFOAM option for dense particle clouds with cell-averaged collision/packing behavior.
- `DPMFoam`: native coupled particle solver with parcel collision support.
- YADE + OpenFOAM: open-source DEM with Python scripting and MPI coupling.
- CFDEM + LIGGGHTS: established GPL CFD+DEM coupling options.

## Safety And Validation

This project is experimental. Do not use it as the sole basis for equipment design, safety decisions, recovery claims, or environmental decisions. Validate any promising geometry with:

- mesh and boundary-condition checks;
- higher-fidelity CFD/particle models;
- physical bench testing;
- safe operating procedures.

## Contributing

Contributions are welcome. Useful areas include:

- OpenFOAM/MPPICFoam case improvements;
- better particle/material presets;
- geometry constraints for manufacturability;
- validation against bench-test data;
- ParaView visualization improvements.

Before opening a pull request, please keep generated outputs out of Git and prefer small, reviewable changes.

## Development Transparency

AurumFlow was developed with AI-assisted coding and review. The project direction, engineering decisions, validation requirements, and final responsibility for the work remain with the maintainer. Generated planning notes, private AI context, CFD outputs, run logs, screenshots, and videos are intentionally kept out of the public repository.

## License

This project is licensed under the GNU General Public License v3.0 or later. See `LICENSE`.