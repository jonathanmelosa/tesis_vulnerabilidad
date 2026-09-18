"""
smoke_test_multiclase.py
===========================
Prueba barata de punta a punta de `modelo_multiclase_robusto_comparacion.py`
tras el fix del bug de COLS_NO_FEATURE (ver docs/decisions.md,
"2026-09-18: Bug critico de fuga de identificadores...") y los tres
agregados de esa misma entrada (red neuronal, guardado de probabilidades,
orden de ejecucion) -- pedido explicito del usuario antes de comprometer
la corrida completa (~50+ horas): confirmar que las 6 rutas de codigo
(XGBoost/LightGBM/HistGB/RandomForest/NN/Logistica, cada una en su
version "principal" con holdout y su version "CV" sin holdout) corren sin
errores, incluyendo el guardado de predicciones por hogar.

NO toca ningun archivo de produccion: parchea `RESULTADOS_DIR` de
`modelo_utils_multiclase` a una carpeta separada
(`.../multiclase_smoke_test/`) ANTES de importar el script principal, asi
que `REGISTRO_CSV`, `REGISTRO_XLSX` y `PREDICCIONES_DIR` (todos derivados
de `RESULTADOS_DIR` al momento de importar) quedan apuntando ahi. Ademas
recorta CV_FOLDS/N_ITER_BUSQUEDA a valores minimos y las listas de
especificaciones a una sola por pista, para que la busqueda de
hiperparametros sea casi instantanea -- esto NO valida la calidad de los
hiperparametros ganadores (ese no es el objetivo), solo que el pipeline
completo (datos -> tuning -> metricas -> registro -> predicciones ->
interpretabilidad) corre sin excepciones para los 6 algoritmos.

Al terminar, borrar manualmente `data/processed/benchmark_resultados/multiclase_smoke_test/`
(o dejarla, no interfiere con nada -- la corrida real escribe en
`.../multiclase/`, una carpeta distinta).

COMO CORRER
    cd src/05_model && python -u smoke_test_multiclase.py
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils_multiclase as mcu

SMOKE_DIR = mcu.PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "multiclase_smoke_test"
shutil.rmtree(SMOKE_DIR, ignore_errors=True)  # limpiar corridas de smoke test previas
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

mcu.RESULTADOS_DIR = SMOKE_DIR
mcu.REGISTRO_CSV = SMOKE_DIR / "registro_modelos_multiclase.csv"
mcu.REGISTRO_XLSX = SMOKE_DIR / "registro_modelos_multiclase.xlsx"
mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL = ["A4"]
mcu.ESPECIFICACIONES_4CLASES_CV = ["A4geo3"]

import modelo_multiclase_robusto_comparacion as mrc  # noqa: E402 (import tras el parche)

mrc.CV_FOLDS = 2
mrc.N_ITER_BUSQUEDA = 2

_INICIO = time.time()

if __name__ == "__main__":
    mrc.main()
    print(f"\nSMOKE TEST OK -- {(time.time() - _INICIO) / 60:.1f} min. Salida en: {SMOKE_DIR}")

    predicciones = sorted((SMOKE_DIR / "predicciones").glob("*.parquet"))
    print(f"\nParquets de predicciones generados ({len(predicciones)}):")
    for p in predicciones:
        print(f"  {p.name}")
