# AurumFlow: Contexto IA

Este archivo es la fuente principal de contexto para cualquier agente de IA que trabaje en este proyecto. Debe mantenerse actualizado con cada cambio de arquitectura o dirección del proyecto.

## Objetivo del Proyecto
Diseñar, simular y optimizar un dispositivo de separación hidráulica de oro aluvial para pequeña minería. El objetivo es lograr la máxima pureza (>99% de rechazo de sedimentos) y máxima recuperación (>99% de oro) usando una bomba solar económica de 12V (bajo caudal y baja presión).

## Dirección y Descubrimientos Recientes
- **De Hidrociclón a Elutriador**: El algoritmo genético descubrió que en lugar de un vórtice rápido (hidrociclón), la configuración óptima para estas bombas de baja potencia es un **elutriador de flujo ascendente** muy lento (~0.3 m/s). 
- **Geometría "Clepsamia" (Reloj de Arena)**: Se abandonó la parametrización cilíndrica/cónica clásica. La nueva parametrización (`visual_sim/geometry.py`) es 100% curva, continua en C¹, con dos lóbulos conectados por un cuello estrecho.
  - Elimina todas las esquinas y superficies planas.
  - Cierra suavemente en un punto en el fondo (sin "trampa" plana).
  - Empalma suavemente con el radio del tubo central en la parte superior.
  - El inlet es dirigible en 3D (pitch y yaw) para inyectar fluido tangencialmente.
- **Optimización Multi-Material**: Simulamos simultáneamente partículas de Oro (objetivo), Arena de Cuarzo y Magnetita. La magnetita forzó a la geometría a evolucionar hacia diseños más eficientes debido a su alta densidad diferencial.

## Arquitectura de Software Actual

La arquitectura "híbrida" (GA rápido local + re-ranking CFD) fue abandonada porque la física local no modelaba bien los vórtices complejos. **Ahora el GA está directamente acoplado a OpenFOAM.**

1. **GA + OpenFOAM (`examples/train_geometry_ga_openfoam.py`)**: 
   - Genera poblaciones de geometrías.
   - Pausa en cada generación para **Revisión Interactiva** (basada en borrar imágenes 2D de una carpeta para descartar geometrías no deseadas).
   - Exporta STL (`examples/export_geometry_cfd.py`).
   - Malla con `snappyHexMesh` y corre `simpleFoam` (`examples/create_openfoam_case.py`).
   - Corre tracking de partículas desacoplado para múltiples materiales (`examples/create_openfoam_particle_case.py`).
   - Calcula métricas y recompensa (`examples/analyze_openfoam_particles.py`).

2. **Recompensa Multiplicativa**:
   La recompensa ahora obliga al algoritmo a equilibrar todos los objetivos:
   `reward = 4.0 * target_recovery * (1.0 - trapped_contamination) * (1.0 - target_loss)`
   Esto penaliza fuertemente a las geometrías que recuperan todo el oro pero retienen mucha arena, o que rechazan toda la arena pero escupen el oro.

3. **Limitaciones del Action Space**:
   - `inlet_angle_deg` fue eliminado del espacio de búsqueda porque, al ser una geometría de revolución, rotar el inlet sobre el eje vertical no cambia el comportamiento físico (es un grado de libertad fantasma).
   - Las velocidades máximas de entrada están limitadas a 2.0 m/s para evitar regímenes turbulentos caóticos que `simpleFoam` laminar no resuelve bien.

## Próximos Pasos (Hoja de Ruta)

1. **Validación Exhaustiva**: Estamos corriendo el GA con la fórmula multiplicativa (`ga_clepsamia_v6_mult`). Debemos observar si el algoritmo escapa del mínimo local anterior (yaw 30°, pitch 85°, v=0.3 m/s) o confirma que ese es el techo físico (Pareto óptimo) del elutriador.
2. **Alta Fidelidad (CFD+DEM)**: El tracking actual de OpenFOAM (`icoUncoupledKinematicParcelFoam`) es rápido pero *desacoplado* (las partículas no chocan entre sí ni tapan el flujo). Una vez validada la mejor Clepsamia, el paso final de software es simular la acumulación de cama densa usando `MPPICFoam` o acoplamientos DEM externos.
3. **Manufacturabilidad**: Evaluar métodos de impresión 3D (SLA/FDM) para fabricar la forma interna y diseñar una carcasa exterior que se pueda seccionar y atornillar.

## Flujos de Trabajo Recomendados (Comandos)

**Correr la optimización GA-OpenFOAM (Recomendado):**
```bash
python3 examples/train_geometry_ga_openfoam.py \
  --generations 60 --population 64 --base-cells 44 \
  --particle-end-time 4.0 --parcels-scale 0.04 \
  --cores-per-eval 2 --n-jobs 7 \
  --output-dir rl_runs/ga_clepsamia_run \
  --warm-start-jsons rl_runs/ga_clepsamia_v5_max_bounds/best_geometry.json \
  --mutation-prob 0.40 --mutation-sigma 0.30
```

**Generar matriz 2D de resultados:**
```bash
python3 examples/plot_all_evaluations.py --run-dir rl_runs/ga_clepsamia_run
```

**Animar la evolución del GA:**
```bash
python3 examples/animate_evolution.py --run-dir rl_runs/ga_clepsamia_run
```

**Evaluar y visualizar un solo individuo en ParaView:**
```bash
python3 examples/evaluate_geometry_openfoam.py \
  --geometry-json rl_runs/ga_clepsamia_run/best_geometry.json \
  --output-root cfd_cases/manual_test --name-prefix test --base-cells 44
pvpython examples/paraview_openfoam_view.py \
  --foam cfd_cases/manual_test/test_000/case/case.foam \
  --particles-csv cfd_cases/manual_test/test_000/particles/particles_latest.csv
```

## Reglas para Agentes de IA
- NUNCA modificar el entorno local (`visual_sim/`) sin antes considerar su impacto en el pipeline de CFD. OpenFOAM es el "juez verdadero".
- NUNCA agregar grados de libertad redundantes (como `inlet_angle_deg`) a los optimizadores sin una razón matemática de peso.
- SIEMPRE usar el esquema Multiplicativo para recompensas en lugar de Sumado para problemas de pureza/recuperación en minería, o de lo contrario el optimizador hará trampa.
- Mantener este archivo y el `README.md` actualizados si se producen refactorizaciones profundas.
