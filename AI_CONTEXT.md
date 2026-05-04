# AurumFlow: optimizacion y validacion de un clasificador para oro aluvial

Este proyecto busca disenar un clasificador/ciclon compacto para separar oro de arena, magnetita y otros sedimentos antes de pasar a pruebas de banco o fabricacion 3D.

El estado actual combina tres niveles:

1. **Simulador local rapido**: visualiza particulas y permite optimizar geometria con RL/GA. Sirve para preseleccionar ideas, no como validacion fisica final.
2. **CFD con OpenFOAM**: exporta la geometria optimizada, genera malla interna, resuelve flujo de agua y visualiza velocidad/lineas de corriente con ParaView.
3. **Particulas OpenFOAM nativas**: valida arrastre, gravedad, escape y captura por material sobre el flujo CFD ya resuelto.

## Estado Actual

- La geometria parametrica avanzada ("Geometría 2.0") vive en `visual_sim/geometry.py`. Incluye miniaturización, límites para bombas de 12V, tubo de rebalse con curvatura independiente (toberas/campanas) y ángulo de entrada 3D.
- La evaluacion rapida para RL/GA vive en `visual_sim/rl_env.py`.
- El optimizador genetico principal acoplado a OpenFOAM está en `examples/train_geometry_ga_openfoam.py`.
- La exportacion CFD esta en `examples/export_geometry_cfd.py`.
- El caso OpenFOAM base se genera con `examples/create_openfoam_case.py`.
- La validacion de particulas OpenFOAM se genera con `examples/create_openfoam_particle_case.py`.
- El postproceso de particulas esta en `examples/analyze_openfoam_particles.py`.
- La visualizacion ParaView automatizada esta en `examples/paraview_openfoam_view.py`.

La optimización genética actual se está ejecutando directamente en OpenFOAM con resolución media (`base-cells 32`) para evitar el "overfitting" a mallas gruesas que se observó en intentos anteriores.

```text
rl_runs/ga_openfoam_med_res/
```

## Instalacion

Dependencias Python:

```bash
python3 -m pip install -r requirements.txt
```

OpenFOAM se usa via Docker:

```bash
docker pull opencfd/openfoam-default:latest
```

ParaView se usa desde Ubuntu:

```bash
paraview
pvpython --version
```

En Ubuntu puede ser necesario:

```bash
export PYTHONPATH=/usr/lib/python3/dist-packages
```

## Simulador Local Rapido

Ejecutar visualizador:

```bash
python3 examples/run_classifier_3d.py
```

Modo headless:

```bash
python3 examples/run_classifier_3d.py --headless --frames 3000 --particles 5000
```

Vista 3D:

```bash
python3 examples/run_classifier_3d.py --display 3d --view isometric
```

Si Taichi falla con CUDA:

```bash
python3 examples/run_classifier_3d.py --arch vulkan
python3 examples/run_classifier_3d.py --arch opengl
```

Importante: el simulador local usa un campo de velocidad analitico. Es util para explorar geometria y alimentar GA/RL, pero no resuelve el agua como CFD ni modela sedimento denso real.

## Optimizacion GA/RL

Entrenamiento genetico recomendado para preseleccion:

```bash
python3 examples/train_geometry_ga.py \
  --generations 50 \
  --population 64 \
  --n-jobs 8 \
  --particles 1600 \
  --frames 3000 \
  --substeps 3 \
  --feed-duration 2.5 \
  --warm-start-json rl_runs/angular_45_135_f3000/best_geometry.json \
  --output-dir rl_runs/ga_curved_f3000
```

El GA evalua geometria con el simulador local y guarda:

- `episodes.csv`: cada geometria evaluada.
- `generations.csv`: resumen por generacion.
- `best_geometry.json`: mejor geometria encontrada.

La funcion objetivo actual premia recuperacion de oro y rechazo de no-oro, y penaliza perdida de oro, contaminacion de trampa y material no procesado.

## Exportar Geometria A CFD

```bash
python3 examples/export_geometry_cfd.py \
  --geometry-json rl_runs/ga_curved_f3000/best_geometry.json \
  --output-dir cfd_exports/ga_curved_f3000_cfd \
  --axial-samples 120 \
  --radial-samples 24 \
  --angular-segments 72 \
  --name ga_curved_f3000
```

La exportacion genera:

- `*_profile.csv`: perfil axial de radios.
- `*_references.csv`: referencias de entrada, tubo de rebalse, garganta y trampa.
- `*_metadata.json`: parametros originales y derivados.
- `*_internal_volume.stl`: superficie interna cerrada para OpenFOAM.

## CFD De Flujo De Agua

Crear caso OpenFOAM:

```bash
python3 examples/create_openfoam_case.py \
  --export-dir cfd_exports/ga_curved_f3000_cfd \
  --stl-name ga_curved_f3000_internal_volume.stl \
  --metadata cfd_exports/ga_curved_f3000_cfd/ga_curved_f3000_metadata.json \
  --case-dir cfd_cases/ga_curved_f3000_simple \
  --base-cells 32 \
  --refinement-level 1
```

Correr malla y flujo:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/work opencfd/openfoam-default:latest \
  bash -lc 'cd /work/cfd_cases/ga_curved_f3000_simple && ./Allclean && ./Allrun && ./AllrunFlow'
```

Nota importante: `create_openfoam_case.py` usa `locationInMesh` fuera del eje central para conservar el volumen interno del ciclon, no la caja exterior.

## Visualizacion ParaView

Abrir manualmente:

```bash
paraview cfd_cases/ga_curved_f3000_simple/ga_curved_f3000_simple.foam
```

Generar imagen con contexto visual:

```bash
pvpython examples/paraview_openfoam_view.py \
  --foam cfd_cases/ga_curved_f3000_simple/ga_curved_f3000_simple.foam \
  --camera iso \
  --screenshot vista_3d.png
```

Generar orbita MP4 sin plano de corte:

```bash
pvpython examples/paraview_openfoam_view.py \
  --foam cfd_cases/ga_curved_f3000_simple/ga_curved_f3000_simple.foam \
  --slice none \
  --no-glyph \
  --orbit-mp4 orbita_cfd.mp4
```

## Particulas OpenFOAM Nativas

Crear casos por material sobre el campo CFD resuelto:

```bash
python3 examples/create_openfoam_particle_case.py \
  --base-case cfd_cases/ga_curved_f3000_simple \
  --metadata cfd_exports/ga_curved_f3000_cfd/ga_curved_f3000_metadata.json \
  --output-root cfd_particle_cases/ga_curved_f3000_particles \
  --end-time 0.5 \
  --write-interval 0.05
```

Correr todos los materiales:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/work opencfd/openfoam-default:latest \
  bash -lc 'cd /work/cfd_particle_cases/ga_curved_f3000_particles && ./AllrunParticles'
```

Analizar metricas:

```bash
python3 examples/analyze_openfoam_particles.py \
  --cases-root cfd_particle_cases/ga_curved_f3000_particles
```

Visualizar particulas sobre el flujo:

```bash
pvpython examples/paraview_openfoam_view.py \
  --foam cfd_cases/ga_curved_f3000_simple/ga_curved_f3000_simple.foam \
  --particles-csv cfd_particle_cases/ga_curved_f3000_particles/particles_latest.csv \
  --no-glyph \
  --slice none \
  --screenshot aurumflow_cfd.png
```

El flujo actual usa `icoUncoupledKinematicParcelFoam`: arrastre de esfera, gravedad y reglas de pared/salida sobre un flujo fijo. Es rapido y util para comparar geometrias. Para pruebas mas densas se puede activar:

```bash
--collision-model pairCollision
```

pero es mucho mas lento.

## Recomendacion CFD+DEM

OpenFOAM nativo no es DEM completo. Opciones:

- **`DPMFoam`**: particulas acopladas al fluido con colisiones parcel-parcel. Mas realista que tracking desacoplado, pero costoso.
- **`MPPICFoam`**: recomendado como siguiente paso para sedimento denso aproximado. Modela interaccion colectiva por celda, packing e isotropia sin resolver cada choque DEM individual. Ya viene en la imagen OpenFOAM.
- **CFDEM + LIGGGHTS**: CFD+DEM real, GPL, clasico, pero suele requerir OpenFOAM antiguo o compilacion cuidadosa.
- **CFDEM-PFM**: alternativa academica mas moderna, tambien GPL.
- **YADE + OpenFOAM**: opcion libre interesante para DEM real con interfaz Python y acoplamiento MPI. Es candidata fuerte si queremos automatizar geometria, particulas y metricas desde scripts.

Ruta recomendada:

1. Mantener el simulador local + GA para buscar muchas geometrias baratas.
2. Validar las mejores con OpenFOAM `simpleFoam` + `icoUncoupledKinematicParcelFoam`.
3. Implementar una rama `MPPICFoam` para sedimento denso aproximado.
4. Usar YADE+OpenFOAM o CFDEM solo en 1-3 geometrias finalistas.

## Proximos Pasos

Los pasos 1-3 originales ya estan completados:

- [x] Barrido CFD rapido automatizado (`examples/evaluate_geometry_openfoam.py`).
- [x] Re-ranking CFD de los mejores individuos del GA.
- [x] Warm-start del GA desde el mejor candidato CFD.
- [x] Repositorio publicado en GitHub como AurumFlow (GPLv3).
- [x] Cambio a métricas basadas en **masa** (peso) en lugar de conteo de partículas.
- [x] Integración directa del Algoritmo Genético con OpenFOAM (`train_geometry_ga_openfoam.py`) para evitar la física artificial del simulador local.
- [x] Implementación de la "Geometría 2.0": miniaturización, límites de bomba solar 12V, tubo de salida independiente y entrada 3D.

Pendientes:

1. Esperar los resultados de la optimización genética de fidelidad media (`ga_openfoam_med_res`).
2. Validar la mejor geometría resultante con una malla de alta resolución (`base-cells 44`) y 15 segundos de simulación.
3. Probar `MPPICFoam` con el mismo STL para sedimento denso aproximado.
4. Evaluar YADE+OpenFOAM o CFDEM solo en 1-3 geometrias finalistas.
5. Prueba de banco fisica (impresión 3D) con la mejor geometria validada.

## GA Hibrido Con Barrido OpenFOAM

No conviene meter OpenFOAM dentro de cada evaluacion del GA principal: seria demasiado lento. La arquitectura correcta es de dos etapas:

```text
GA rapido local
  -> episodes.csv con miles de geometrías
  -> seleccionar top N diverso
  -> exportar cada geometria a CFD
  -> correr OpenFOAM en baja resolucion
  -> correr particulas rapidas
  -> calcular score CFD
  -> reordenar finalistas
  -> reinyectar los mejores como warm-start del GA
```

Score CFD sugerido:

```text
score_cfd =
    1.8 * target_recovery_pct
  + 1.5 * non_target_rejection_pct
  - 3.0 * target_loss_pct
  - 2.4 * trapped_contamination_pct
  - 0.2 * runtime_penalty
```

El script operativo es `examples/evaluate_geometry_openfoam.py`. Puede evaluar una geometria puntual:

```bash
python3 examples/evaluate_geometry_openfoam.py \
  --geometry-json rl_runs/ga_curved_f3000/best_geometry.json \
  --output-root cfd_sweeps/ga_curved_best \
  --base-cells 24 \
  --refinement-level 1 \
  --particle-end-time 0.1 \
  --parcels-scale 0.05
```

O puede re-rankear finalistas desde `episodes.csv`:

```bash
python3 examples/evaluate_geometry_openfoam.py \
  --episodes-csv rl_runs/ga_curved_f3000/episodes.csv \
  --top-n 12 \
  --output-root cfd_sweeps/ga_curved_top12 \
  --base-cells 24 \
  --refinement-level 1 \
  --particle-end-time 0.1 \
  --parcels-scale 0.05
```

El evaluador ejecuta por candidato:

1. Convierte params/JSON a `ClassifierGeometry`.
2. Exporta STL y metadata CFD.
3. Crea el caso OpenFOAM.
4. Corre malla + `simpleFoam`.
5. Crea y corre casos de particulas por material.
6. Analiza metricas y calcula `cfd_score`.
7. Escribe `cfd_score.csv` y `best_cfd_geometry.json`.

Tomar el mejor CFD como warm-start:

```bash
python3 examples/train_geometry_ga.py \
  --warm-start-json cfd_sweeps/<run>/best_cfd_geometry.json \
  --output-dir rl_runs/ga_after_cfd
```

Este enfoque usa OpenFOAM como juez fisico caro, no como evaluador masivo. Asi el GA sigue explorando rapido, pero deja de confiar ciegamente en el simulador local.

## Supuestos Y Limitaciones

- El simulador local es heuristico y sirve para exploracion.
- `simpleFoam` actual es laminar y estacionario; puede subestimar efectos turbulentos.
- La entrada tangencial CFD es aproximada: aun no hay boquilla 3D detallada.
- `icoUncoupledKinematicParcelFoam` usa flujo congelado, no retroalimenta el fluido.
- `pairCollision` existe pero es lento.
- Para sedimento denso, el siguiente paso nativo razonable es `MPPICFoam`.
- Para DEM real, evaluar YADE+OpenFOAM o CFDEM en geometrias finalistas.

## Simulador De Proceso 0-D

El modelo de proceso original sigue disponible para balances gruesos:

```bash
python3 examples/run_process.py
python3 examples/optimize_sluice.py
```

Se mantiene como herramienta conceptual para comparar etapas de beneficio, no como validador del clasificador CFD.

## Nota Sobre Asistencia IA

AurumFlow fue desarrollado con asistencia de IA para codificacion y revision. La direccion del proyecto, decisiones de ingenieria, requisitos de validacion y responsabilidad final son del mantenedor. Este archivo (`AI_CONTEXT.md`) es contexto interno y no se publica en el repositorio.
