"""
modelo_multiclase_predicciones.py
=====================================

NO LANZAR TODAVIA -- preparado a pedido del usuario (2026-09-18) para
correrlo DESPUES de que termine `modelo_multiclase_robusto_comparacion.py`.

Motivacion
----------
Toda la suite multiclase (`modelo_multiclase_robusto_comparacion.py`,
`modelo_multiclase_descomponer_auc.py`) calcula probabilidades por hogar
para las 4 clases en cada semilla, pero las usa solo para derivar
metricas agregadas (AUC, recall, etc.) y las descarta -- nunca las
guarda. El usuario pidio poder usar los modelos para PREDECIR, hogar por
hogar, la probabilidad de cada uno de los 4 grupos (nunca_pobre/entra/
sale/siempre_pobre), no solo evaluar que tan bien discriminan en
agregado.

Que hace este script
---------------------
Para cada fila YA REGISTRADA en `registro_modelos_multiclase.csv`
(algoritmo x especificacion, con `balanceo_elegido` y `hiperparametros`
ya elegidos por la busqueda cara -- NO se repite esa busqueda aqui, ver
docstring de `modelo_multiclase_descomponer_auc.py` para el mismo
argumento de costo):
  1. Reconstruye el pipeline (mismas funciones `pipeline_*` de
     `modelo_multiclase_robusto_comparacion.py`) con los hiperparametros
     guardados.
  2. Reentrena UNA SOLA VEZ con semilla=RANDOM_STATE (42, no las 5
     semillas -- para un output de prediccion por hogar se usa el ajuste
     canonico, no una distribucion; si se necesita el rango entre
     semillas para algun hogar puntual, se puede extender despues).
  3. Calcula las probabilidades de las 4 clases:
       - Especificaciones PRINCIPALES (holdout): predict_proba sobre el
         conjunto de PRUEBA (2013->2016) -- hogares que el modelo nunca
         vio en entrenamiento, la evaluacion mas honesta de que tan bien
         "predice" en el sentido de uso real.
       - Especificaciones CV (*geo3, sin holdout): probabilidades
         OUT-OF-FOLD (cross_val_predict) sobre el 100% de la muestra
         2010->2013 -- cada hogar predicho por un modelo que no lo vio en
         entrenamiento, igual convencion que el resto de la suite.
  4. Guarda, por cada fila, un parquet con: consecutivo, algoritmo,
     especificacion, Y_grupo real, y las 4 probabilidades
     (prob_nunca_pobre, prob_entra, prob_sale, prob_siempre_pobre).

Todos los parquets se concatenan en un unico archivo de salida.

INPUTS
    data/processed/benchmark_resultados/multiclase/registro_modelos_multiclase.csv
    data/processed/benchmark_train_test/modelo_{espec}_*.parquet

OUTPUTS
    data/processed/benchmark_resultados/multiclase/predicciones_probabilidades.parquet
    columnas: consecutivo, algoritmo, especificacion, Y_grupo,
    prob_nunca_pobre, prob_entra, prob_sale, prob_siempre_pobre

COMO CORRER (cuando termine modelo_multiclase_robusto_comparacion.py)
-----------------------------------------------------------------------
    cd src/05_model && python -u modelo_multiclase_predicciones.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
import modelo_utils_multiclase as mcu
import modelo_multiclase_robusto_comparacion as mrc
from modelo_multiclase_descomponer_auc import ALGORITMOS_ARBOL_NATIVO, ALGORITMOS_PREPROCESADOR

OUTPUT_PARQUET = mcu.RESULTADOS_DIR / "predicciones_probabilidades.parquet"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def _reordenar_proba(proba: np.ndarray, clases_estimador: list) -> np.ndarray:
    """Reindexa columnas de `proba` al orden de CATEGORIAS_Y_GRUPO (0..3),
    igual convencion que `calcular_metricas_multiclase`."""
    orden = [clases_estimador.index(i) for i in range(len(mcu.CATEGORIAS_Y_GRUPO))]
    return proba[:, orden]


def _armar_salida(consecutivo: pd.Series, y_grupo: np.ndarray, proba: np.ndarray, algoritmo: str, espec: str) -> pd.DataFrame:
    salida = pd.DataFrame({
        "consecutivo": consecutivo.values,
        "algoritmo": algoritmo,
        "especificacion": espec,
        "Y_grupo": [mcu.CATEGORIAS_Y_GRUPO[k] for k in y_grupo],
    })
    for k, nombre in enumerate(mcu.CATEGORIAS_Y_GRUPO):
        salida[f"prob_{nombre}"] = proba[:, k]
    return salida


def _procesar_principal(algoritmo: str, espec: str, balanceo: str, params: dict) -> pd.DataFrame:
    train, test = mcu.cargar_datos_4clases(espec)

    if algoritmo in ALGORITMOS_ARBOL_NATIVO:
        pipeline_fn, fit_params_fn = ALGORITMOS_ARBOL_NATIVO[algoritmo]
        x_train, y_train, cat_cols = mcu.preparar_arboles_nativos_multiclase(train)
        x_test, y_test, _ = mcu.preparar_arboles_nativos_multiclase(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)
        pipe = pipeline_fn(balanceo, semilla=mcu.RANDOM_STATE)
        fit_params = fit_params_fn(balanceo, y_train) if fit_params_fn else {}
    else:
        pipeline_fn = ALGORITMOS_PREPROCESADOR[algoritmo]
        x_train, y_train = mcu.preparar_xy_crudo_multiclase(train)
        x_test, y_test = mcu.preparar_xy_crudo_multiclase(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)
        pipe = pipeline_fn(x_train, balanceo, semilla=mcu.RANDOM_STATE)
        fit_params = {}

    if params:
        pipe.set_params(**params)
    pipe.fit(x_train, y_train, **fit_params)
    proba = _reordenar_proba(pipe.predict_proba(x_test), list(pipe.classes_))
    return _armar_salida(test["consecutivo"], np.asarray(y_test), proba, algoritmo, espec)


def _procesar_cv(algoritmo: str, espec: str, balanceo: str, params: dict) -> pd.DataFrame:
    datos = mcu.cargar_datos_cv_4clases(espec)

    if algoritmo in ALGORITMOS_ARBOL_NATIVO:
        pipeline_fn, fit_params_fn = ALGORITMOS_ARBOL_NATIVO[algoritmo]
        x, y, _ = mcu.preparar_arboles_nativos_multiclase(datos)
        pipe = pipeline_fn(balanceo, semilla=mcu.RANDOM_STATE)
        fit_params = fit_params_fn(balanceo, y) if fit_params_fn else None
    else:
        pipeline_fn = ALGORITMOS_PREPROCESADOR[algoritmo]
        x, y = mcu.preparar_xy_crudo_multiclase(datos)
        pipe = pipeline_fn(x, balanceo, semilla=mcu.RANDOM_STATE)
        fit_params = None

    if params:
        pipe.set_params(**params)
    cv = StratifiedKFold(n_splits=mrc.CV_FOLDS, shuffle=True, random_state=mcu.RANDOM_STATE)
    proba_oof = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba", n_jobs=-1, params=fit_params)

    # classes_ del estimador: se necesita un ajuste (no lo da cross_val_predict)
    pipe_ref = pipeline_fn(balanceo, semilla=mcu.RANDOM_STATE) if algoritmo in ALGORITMOS_ARBOL_NATIVO else pipeline_fn(x, balanceo, semilla=mcu.RANDOM_STATE)
    if params:
        pipe_ref.set_params(**params)
    pipe_ref.fit(x, y, **(fit_params or {}))

    proba = _reordenar_proba(proba_oof, list(pipe_ref.classes_))
    return _armar_salida(datos["consecutivo"], np.asarray(y), proba, algoritmo, espec)


def main() -> None:
    if not mcu.REGISTRO_CSV.exists():
        print(f"ERROR: no existe {mcu.REGISTRO_CSV} -- correr primero modelo_multiclase_robusto_comparacion.py", file=sys.stderr)
        sys.exit(1)

    registro = pd.read_csv(mcu.REGISTRO_CSV)
    log(f"INICIO generacion de predicciones sobre {len(registro)} filas ya registradas")

    piezas = []
    for _, fila in registro.iterrows():
        algoritmo, espec = fila["algoritmo"], fila["especificacion"]
        balanceo = fila["balanceo_elegido"]
        params = json.loads(fila["hiperparametros"]) if pd.notna(fila["hiperparametros"]) else {}

        log(f"=== {algoritmo} -- {espec} -- INICIO ===")
        if espec in mcu.ESPECIFICACIONES_4CLASES_CV:
            salida = _procesar_cv(algoritmo, espec, balanceo, params)
        else:
            salida = _procesar_principal(algoritmo, espec, balanceo, params)
        piezas.append(salida)
        log(f"=== {algoritmo} -- {espec} -- FIN ({len(salida)} hogares) ===")

    resultado = pd.concat(piezas, ignore_index=True)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(OUTPUT_PARQUET, index=False)
    log(f"FIN. Guardado: {OUTPUT_PARQUET} ({len(resultado)} filas totales)")


if __name__ == "__main__":
    main()
