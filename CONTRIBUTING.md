# Contributing

Thanks for your interest in contributing to AurumFlow.

This project is experimental software for exploring and validating classifier/cyclone geometries for small-scale alluvial gold separation. Contributions are welcome, especially around reproducibility, OpenFOAM workflows, particle validation, visualization, documentation, and comparison against bench-test data.

## Before You Start

Please keep in mind:

- This project is not a certified engineering tool.
- Simulation results should not be presented as guaranteed recovery, safety, or environmental performance.
- Generated CFD cases, videos, images, optimization runs, and other large outputs should not be committed.
- Small, focused pull requests are easier to review than large rewrites.

## Development Setup

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a quick process example:

```bash
python3 examples/run_process.py
```

Run a quick visual-simulator smoke test:

```bash
python3 examples/run_classifier_3d.py --headless --frames 100 --particles 200
```

Compile-check scripts:

```bash
python3 -m py_compile examples/*.py visual_sim/*.py simulador_proceso/*.py
```

## OpenFOAM Contributions

OpenFOAM output directories are ignored by Git. If you improve an OpenFOAM workflow, please commit the generator scripts and documentation, not generated case folders.

Useful areas:

- More robust boundary conditions.
- MPPICFoam support for dense sediment validation.
- Better particle/material presets.
- Automated comparison of multiple geometries.
- ParaView scripts for clearer visual diagnostics.

## Pull Request Guidelines

When opening a pull request:

1. Explain the problem or goal.
2. Summarize the change.
3. Include commands you ran to test it.
4. Mention limitations or assumptions.
5. Avoid committing generated outputs, videos, images, or local run directories.

## Reporting Issues

Please include:

- Operating system and Python version.
- Exact command run.
- Error output or log snippet.
- Whether Docker/OpenFOAM/ParaView is involved.
- A small reproducer if possible.

## License

By contributing, you agree that your contributions are licensed under the GNU General Public License v3.0 or later, consistent with this repository.
