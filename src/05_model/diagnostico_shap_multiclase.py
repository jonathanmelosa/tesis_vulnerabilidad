"""
diagnostico_shap_multiclase.py
=====================================
SHAP para los modelos multiclase (4 grupos: nunca_pobre/entra/sale/
siempre_pobre) -- pedido del usuario (2026-09-18), tras notar que TODAS
las figuras de importancia de variables de la Seccion~\\ref{subsec:desempeno}
(Random Forest, coeficientes logisticos, SHAP) vienen exclusivamente de
los modelos binarios entra-vs-nunca, y preguntar si el "nucleo" de
variables importantes cambiaria con los modelos que predicen los 4
grupos.

Corre SOLO para los 4 algoritmos YA completos (XGBoost, LightGBM,
HistGradientBoosting, Random Forest) -- la logistica regularizada queda
pendiente porque aun le falta una especificacion (B4geo3) en
`modelo_multiclase_robusto_comparacion.py`; se agrega despues con un
script separado (mismo patron que `diagnostico_shap_ab.py` vs.
`diagnostico_shap.py` en el benchmark binario: no repetir cascadas de
espera, extender cuando el resto este listo).

Por que SHAP para la clase "entra" especificamente
--------------------------------------------------------------------------
Un modelo multiclase da un vector de SHAP values por observacion, uno
POR CLASE (4 aqui) -- no un solo numero. Para mantener la comparacion
directamente equiparable con las Figuras de la Seccion~\\ref{subsec:desempeno}
(que miden "que tanto empuja cada variable la probabilidad de ENTRAR en
pobreza"), se reporta unicamente el SHAP de la clase "entra" (indice fijo
GRUPO_A_ENTERO["entra"]=1 en `modelo_utils_multiclase.py` -- las clases
del estimador quedan en ese mismo orden porque `y` ya se codifico como
enteros 0..3 antes de entrenar, `np.unique` las ordena 0,1,2,3).

Metodologia (mismo patron de bajo costo que el resto de la suite
multiclase): NO se repite la busqueda de hiperparametros -- se reconstruye
el pipeline ganador (balanceo/hiperparametros ya en
`registro_modelos_multiclase.csv`) y se reentrena una unica vez sobre el
conjunto de entrenamiento (o sobre el 100% de la muestra para las
especificaciones *geo3, que no tienen holdout -- ver docstring de
`modelo_multiclase_robusto_comparacion.py`). SHAP se calcula sobre el
conjunto de PRUEBA (holdout) o sobre la misma muestra de entrenamiento
para *geo3 (en ese caso, interpretar con cautela: mide que tanto se
apoya el ajuste en cada variable, no su aporte fuera de muestra, mismo
matiz que ya aplica a la importancia por permutacion de HistGB en el
benchmark binario CV-only).

INPUTS
    data/processed/benchmark_resultados/multiclase/registro_modelos_multiclase.csv
    data/processed/benchmark_train_test/modelo_{espec}_*.parquet

OUTPUTS
    data/processed/benchmark_resultados/multiclase/diagnostico_shap_importancia_entra.csv
    (columnas: algoritmo, especificacion, variable, shap_abs_medio, rank)

COMO CORRER
    cd src/05_model && python -u diagnostico_shap_multiclase.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
import modelo_utils_multiclase as mcu
import modelo_multiclase_robusto_comparacion as mrc

ALGORITMOS_TREE = {
    "XGBoost": (mrc.pipeline_xgb, mrc.fit_params_xgb, True),
    "LightGBM": (mrc.pipeline_lgb, None, False),
    "HistGradientBoosting (sklearn)": (mrc.pipeline_hgb, None, True),
}
ALGORITMO_RF = "Random Forest"

IDX_ENTRA = mcu.GRUPO_A_ENTERO["entra"]
RUTA_SALIDA = mcu.RESULTADOS_DIR / "diagnostico_shap_importancia_entra.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def _shap_entra_arbol_nativo(modelo, x, cat_cols: list, necesita_codificar: bool) -> tuple[np.ndarray, list]:
    x_shap = x
    if necesita_codificar:
        x_shap = x.copy()
        for c in cat_cols:
            if c in x_shap.columns:
                x_shap[c] = x_shap[c].cat.codes.astype(float).replace(-1, np.nan)
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(x_shap)
    return _extraer_clase_entra(shap_values, modelo.classes_), list(x_shap.columns)


def _extraer_clase_entra(shap_values, clases) -> np.ndarray:
    idx = list(clases).index(IDX_ENTRA)
    if isinstance(shap_values, list):
        return shap_values[idx]
    if np.ndim(shap_values) == 3:
        # shap>=0.45 devuelve (n, features, clases) para TreeExplainer multiclase
        return shap_values[:, :, idx]
    raise ValueError(f"Forma inesperada de shap_values para modelo multiclase: {np.shape(shap_values)}")


def _procesar_arbol_nativo(algoritmo: str, espec: str, balanceo: str, params: dict) -> pd.DataFrame:
    pipeline_fn, fit_params_fn, necesita_codificar = ALGORITMOS_TREE[algoritmo]

    if espec in mcu.ESPECIFICACIONES_4CLASES_CV:
        datos = mcu.cargar_datos_cv_4clases(espec)
        x, y, cat_cols = mcu.preparar_arboles_nativos_multiclase(datos)
        x_eval = x
    else:
        train, test = mcu.cargar_datos_4clases(espec)
        x, y, cat_cols = mcu.preparar_arboles_nativos_multiclase(train)
        x_test, y_test, _ = mcu.preparar_arboles_nativos_multiclase(test)
        x, x_test = mu.alinear_columnas_categoricas(x, x_test, cat_cols)
        x_eval = x_test

    pipe = pipeline_fn(balanceo, semilla=mcu.RANDOM_STATE)
    if params:
        pipe.set_params(**params)
    fit_params = fit_params_fn(balanceo, y) if fit_params_fn else {}
    pipe.fit(x, y, **fit_params)

    modelo = pipe.named_steps["modelo"]
    shap_entra, nombres = _shap_entra_arbol_nativo(modelo, x_eval, cat_cols, necesita_codificar)
    return _armar_ranking(shap_entra, nombres, algoritmo, espec)


def _procesar_random_forest(espec: str, balanceo: str, params: dict) -> pd.DataFrame:
    if espec in mcu.ESPECIFICACIONES_4CLASES_CV:
        datos = mcu.cargar_datos_cv_4clases(espec)
        x, y = mcu.preparar_xy_crudo_multiclase(datos)
        x_eval = x
    else:
        train, test = mcu.cargar_datos_4clases(espec)
        x, y = mcu.preparar_xy_crudo_multiclase(train)
        x_test, y_test = mcu.preparar_xy_crudo_multiclase(test)
        x, x_test = x.align(x_test, join="inner", axis=1)
        x_eval = x_test

    pipe = mrc.pipeline_rf(x, balanceo, semilla=mcu.RANDOM_STATE)
    if params:
        pipe.set_params(**params)
    pipe.fit(x, y)

    prep = pipe.named_steps["prep"]
    modelo = pipe.named_steps["modelo"]
    x_eval_t = prep.transform(x_eval)
    if hasattr(x_eval_t, "toarray"):
        x_eval_t = x_eval_t.toarray()
    nombres = list(prep.get_feature_names_out())

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(x_eval_t)
    shap_entra = _extraer_clase_entra(shap_values, modelo.classes_)
    return _armar_ranking(shap_entra, nombres, ALGORITMO_RF, espec)


def _armar_ranking(shap_entra: np.ndarray, nombres: list, algoritmo: str, espec: str) -> pd.DataFrame:
    importancia = pd.Series(np.abs(shap_entra).mean(axis=0), index=nombres).sort_values(ascending=False)
    ranking = importancia.rank(ascending=False)
    return pd.DataFrame({
        "algoritmo": algoritmo,
        "especificacion": espec,
        "variable": importancia.index,
        "shap_abs_medio_entra": importancia.values.round(6),
        "rank": ranking.loc[importancia.index].astype(int).values,
    })


def main() -> None:
    registro = pd.read_csv(mcu.REGISTRO_CSV)
    algoritmos_a_correr = list(ALGORITMOS_TREE) + [ALGORITMO_RF]
    registro = registro[registro["algoritmo"].isin(algoritmos_a_correr)]
    log(f"INICIO SHAP multiclase (clase 'entra') sobre {len(registro)} filas "
        f"({len(algoritmos_a_correr)} algoritmos, logistica pendiente)")

    piezas = []
    for _, fila in registro.iterrows():
        algoritmo, espec = fila["algoritmo"], fila["especificacion"]
        balanceo = fila["balanceo_elegido"]
        params = json.loads(fila["hiperparametros"]) if pd.notna(fila["hiperparametros"]) else {}

        log(f"=== {algoritmo} -- {espec} -- INICIO ===")
        if algoritmo == ALGORITMO_RF:
            ranking = _procesar_random_forest(espec, balanceo, params)
        else:
            ranking = _procesar_arbol_nativo(algoritmo, espec, balanceo, params)
        piezas.append(ranking)
        log(f"  Top 5 (|SHAP entra| medio): {ranking.head(5)['variable'].tolist()}")
        log(f"=== {algoritmo} -- {espec} -- FIN ===")

    resultado = pd.concat(piezas, ignore_index=True)
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(RUTA_SALIDA, index=False)
    log(f"FIN. Guardado: {RUTA_SALIDA} ({len(resultado)} filas)")


if __name__ == "__main__":
    main()
