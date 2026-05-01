# AurumFlow

Experimental tools for designing and validating a compact hydraulic classifier/cyclone for small-scale alluvial gold separation.

The project combines fast geometry exploration, evolutionary optimization, OpenFOAM CFD export, particle tracking, and ParaView visualization. It is intended as a research and prototyping aid, not as a certified mining, safety, or recovery guarantee.

## Project Status

This repository is experimental and under active development.

Current capabilities:

- A simple 0-D process simulator for rough mass-balance thinking.
- A local 3D particle visualizer for rapid geometry exploration.
- Reinforcement learning and genetic algorithm workflows for candidate geometry search.
- STL/metadata export for OpenFOAM.
- OpenFOAM case generation for internal water-flow validation.
- Native OpenFOAM particle tracking by material class.
- ParaView automation for screenshots, orbit videos, streamlines, and particle overlays.

Important limitation: the local visual simulator is not a full CFD/DEM solver. It is useful for generating hypotheses, but final design decisions should be checked with OpenFOAM, higher-fidelity CFD+DEM, and physical bench tests.

## Repository Layout

```text
examples/
  run_process.py                    # 0-D process example
  run_classifier_3d.py              # local visual classifier simulator
  train_geometry_ga.py              # genetic geometry optimizer
  train_geometry_rl.py              # RL geometry optimizer
  export_geometry_cfd.py            # geometry export for CFD/SPH
  create_openfoam_case.py           # OpenFOAM water-flow case generator
  create_openfoam_particle_case.py  # OpenFOAM particle cases by material
  analyze_openfoam_particles.py     # particle metrics post-processing
  evaluate_geometry_openfoam.py     # hybrid GA/OpenFOAM evaluator
  evaluate_geometry.py              # local batch geometry evaluation
  optimize_sluice.py                # 0-D sluice parameter sweep
  paraview_openfoam_view.py         # scripted ParaView visualization

visual_sim/                         # geometry, physics, metrics, viewer, RL environment
simulador_proceso/                  # rough 0-D process simulator
```

Generated outputs such as `rl_runs/`, `cfd_cases/`, `cfd_exports/`, `cfd_particle_cases/`, `cfd_sweeps/`, images, and videos are intentionally ignored by Git.

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

On Ubuntu, `pvpython` may need:

```bash
export PYTHONPATH=/usr/lib/python3/dist-packages
```

## Quick Start

Run the rough process model:

```bash
python3 examples/run_process.py
```

Run the local classifier visualizer:

```bash
python3 examples/run_classifier_3d.py
```

Headless local simulation:

```bash
python3 examples/run_classifier_3d.py --headless --frames 3000 --particles 5000
```

Run a genetic optimization:

```bash
python3 examples/train_geometry_ga.py \
  --generations 50 \
  --population 64 \
  --n-jobs 8 \
  --particles 1600 \
  --frames 3000 \
  --substeps 3 \
  --feed-duration 2.5 \
  --output-dir rl_runs/ga_geometry
```

## OpenFOAM Workflow

Export a candidate geometry:

```bash
python3 examples/export_geometry_cfd.py \
  --geometry-json rl_runs/ga_geometry/best_geometry.json \
  --output-dir cfd_exports/candidate_cfd \
  --axial-samples 120 \
  --radial-samples 24 \
  --angular-segments 72 \
  --name candidate
```

Create an OpenFOAM case:

```bash
python3 examples/create_openfoam_case.py \
  --export-dir cfd_exports/candidate_cfd \
  --stl-name candidate_internal_volume.stl \
  --metadata cfd_exports/candidate_cfd/candidate_metadata.json \
  --case-dir cfd_cases/candidate_simple \
  --base-cells 32 \
  --refinement-level 1
```

Run OpenFOAM:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/work opencfd/openfoam-default:latest \
  bash -lc 'cd /work/cfd_cases/candidate_simple && ./Allclean && ./Allrun && ./AllrunFlow'
```

Visualize with ParaView:

```bash
pvpython examples/paraview_openfoam_view.py \
  --foam cfd_cases/candidate_simple/candidate_simple.foam \
  --camera iso \
  --screenshot vista_3d.png
```

## Particle Validation

Create particle tracking cases:

```bash
python3 examples/create_openfoam_particle_case.py \
  --base-case cfd_cases/candidate_simple \
  --metadata cfd_exports/candidate_cfd/candidate_metadata.json \
  --output-root cfd_particle_cases/candidate_particles \
  --end-time 0.5 \
  --write-interval 0.05
```

Run all material cases:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/work opencfd/openfoam-default:latest \
  bash -lc 'cd /work/cfd_particle_cases/candidate_particles && ./AllrunParticles'
```

Analyze metrics:

```bash
python3 examples/analyze_openfoam_particles.py \
  --cases-root cfd_particle_cases/candidate_particles
```

Overlay particles in ParaView:

```bash
pvpython examples/paraview_openfoam_view.py \
  --foam cfd_cases/candidate_simple/candidate_simple.foam \
  --particles-csv cfd_particle_cases/candidate_particles/particles_latest.csv \
  --no-glyph \
  --slice none \
  --screenshot aurumflow_cfd.png
```

## Hybrid Optimization Strategy

The local simulator should not be treated as the final judge of physical performance. The recommended workflow is multi-fidelity:

```text
fast local GA/RL exploration
  -> diverse candidate geometries
  -> low-resolution OpenFOAM re-ranking
  -> particle validation
  -> MPPICFoam or CFD+DEM for finalists
  -> physical bench testing
```

Run an OpenFOAM re-ranking pass from a GA result:

```bash
python3 examples/evaluate_geometry_openfoam.py \
  --episodes-csv rl_runs/ga_geometry/episodes.csv \
  --top-n 12 \
  --output-root cfd_sweeps/ga_top12 \
  --base-cells 24 \
  --refinement-level 1 \
  --particle-end-time 0.1 \
  --parcels-scale 0.05
```

Use the best OpenFOAM-ranked geometry as a warm start:

```bash
python3 examples/train_geometry_ga.py \
  --warm-start-json cfd_sweeps/ga_top12/best_cfd_geometry.json \
  --output-dir rl_runs/ga_after_cfd
```

## CFD+DEM Direction

OpenFOAM native particle tracking is useful but not full DEM.

Recommended next fidelity levels:

- `MPPICFoam`: native OpenFOAM option for dense particle clouds with cell-averaged collision/packing behavior.
- `DPMFoam`: native coupled particle solver with parcel collision support, but can be costly.
- YADE + OpenFOAM: open-source DEM with Python scripting and MPI coupling; promising for final high-fidelity validation.
- CFDEM + LIGGGHTS or CFDEM-PFM: established GPL CFD+DEM coupling options, but installation/version compatibility can be harder.

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
- ParaView visualization improvements;
- documentation and reproducible examples.

Before opening a pull request, please keep generated outputs out of Git and prefer small, reviewable changes.

## Development Transparency

AurumFlow was developed with AI-assisted coding and review. The project direction, engineering decisions, validation requirements, and final responsibility for the work remain with the maintainer. Generated planning notes, private AI context, CFD outputs, run logs, screenshots, and videos are intentionally kept out of the public repository.

## License

This project is licensed under the GNU General Public License v3.0 or later. See `LICENSE`.
