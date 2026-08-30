"""
Ejercicio exploratorio (Pipeline 2, ver `construir_pipeline_geo3_cv_ipm.py`)
con target IPM: las TRES fuentes geoespaciales (DMSP-OLS, ALOS PALSAR,
Landsat 5 TM) juntas, restringido a la transicion 2010->2013, evaluado
con validacion cruzada agrupada (NO holdout temporal -- ALOS PALSAR y
Landsat 5 TM no tienen datos reales en 2013). Espejo exacto del bloque
`ESPECIFICACIONES_CV_2010_2013` que ya corren `modelo_xgboost.py` /
`modelo_histgradientboosting.py` / `modelo_logistica_regularizada.py`
para pobreza monetaria, aplicado a `Aipmgeo3`/`Bipmgeo3`.

Misma config robusta que el resto de la comparacion IPM: F-beta beta=2,
CV_FOLDS=10, N_ITER_BUSQUEDA=30, RANDOM_STATE=42, SEMILLAS=[42,1,2,3,4].
Usa `evaluar_cv_semillas` (NO `evaluar_multiples_semillas`) porque no hay
holdout separado -- las metricas se calculan sobre probabilidades
OUT-OF-FOLD dentro de la misma muestra 2010->2013.

RIESGO DE TIEMPO CONOCIDO Y ACEPTADO: Logistica regularizada (`saga`) ya
tardo hasta ~17 horas en una combinacion de la comparacion IPM principal
-- el mismo riesgo aplica aqui (n=6.474, la transicion 2010->2013
completa se usa como muestra unica). Se corre con `caffeinate -i`.
Decision confirmada con el usuario (2026-08-30) de incluir Logistica pese
al riesgo.

Resultados NO comparables cifra a cifra contra Aipm/Bipm/AipmgeoDMSP/
BipmgeoDMSP (esquema de validacion distinto) ni contra
Ageo3/Bgeo3 (target distinto).

OUTPUTS

    data/processed/benchmark_resultados/registro_modelos_ipm_geo3.csv
    data/processed/benchmark_resultados/ipm_geo3/{xgboost,histgradientboosting,logistica_regularizada}/
        metricas_multiples_semillas_modelo_{espec}.csv

COMO CORRER

    cd src/05_model && caffeinate -i python -u modelo_ipm_geo3_cv.py
"""

import time
from datetime import datetime

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb
import modelo_logistica_regularizada as m_log
import modelo_xgboost as m_xgb

BETA = 2.0
CV_FOLDS = 10
N_ITER_BUSQUEDA = 30
ESPECIFICACIONES = ["Aipmgeo3", "Bipmgeo3"]

REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_ipm_geo3.csv"
REGISTRO_XLSX = mu.RESULTADOS_DIR / "registro_modelos_ipm_geo3.xlsx"
OUTPUT_ROOT = mu.RESULTADOS_DIR / "ipm_geo3"

OBSERVACIONES = (
    f"PIPELINE 2 (exploratorio), target IPM: DMSP-OLS + ALOS PALSAR + Landsat "
    f"5 TM juntas, restringido a la transicion 2010->2013 (ALOS/Landsat sin "
    f"datos reales en 2013). Sin holdout temporal -- metricas sobre "
    f"probabilidades OUT-OF-FOLD (cross_val_predict) dentro de la misma "
    f"muestra, NO comparables cifra a cifra contra Aipm/Bipm/AipmgeoDMSP/"
    f"BipmgeoDMSP (holdout) ni contra Ageo3/Bgeo3 (target monetario). "
    f"F-beta (beta={BETA}), CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, "
    f"balanceo/hiperparametros por AUC-ROC, RANDOM_STATE={mu.RANDOM_STATE}, "
    f"SEMILLAS={mu.SEMILLAS}."
)

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def correr_xgboost() -> None:
    out_dir = OUTPUT_ROOT / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES:
        datos = mu.cargar_datos_cv(espec)
        x, y, cat_cols = mu.preparar_arboles_nativos(datos)

        log(f"=== XGBoost -- {espec} -- INICIO (n={x.shape[0]}, columnas={x.shape[1]}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_xgb.construir_pipeline(x, y, b),
            param_distributions_fn=lambda b: m_xgb.PARAM_DIST,
            x_train=x, y_train=y,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: m_xgb.construir_pipeline(x, y, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y, beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC(OOF): {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "scale_pos_weight": "auto (balanced)" if resultado["balanceo_elegido"] == "balanced" else 1.0}
        mu.registrar_resultado(
            algoritmo="XGBoost", especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (enable_categorical)",
            balanceo_info=resultado, hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === XGBoost -- {espec} -- FIN ===")


def correr_histgradientboosting() -> None:
    out_dir = OUTPUT_ROOT / "histgradientboosting"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES:
        datos = mu.cargar_datos_cv(espec)
        x, y, cat_cols = mu.preparar_arboles_nativos(datos)

        log(f"=== HistGradientBoosting -- {espec} -- INICIO (n={x.shape[0]}, columnas={x.shape[1]}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=m_hgb.construir_pipeline,
            param_distributions_fn=lambda b: m_hgb.PARAM_DIST,
            x_train=x, y_train=y,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: m_hgb.construir_pipeline(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y, beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC(OOF): {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="HistGradientBoosting (sklearn)", especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (categorical_features='from_dtype')",
            balanceo_info=resultado, hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === HistGradientBoosting -- {espec} -- FIN ===")


def correr_logistica() -> None:
    out_dir = OUTPUT_ROOT / "logistica_regularizada"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in ESPECIFICACIONES:
        datos = mu.cargar_datos_cv(espec)
        x, y = mu.preparar_xy_crudo(datos)

        log(f"=== Logistica regularizada -- {espec} -- INICIO (n={x.shape[0]}, columnas={x.shape[1]}) ===")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_log.construir_pipeline(x, b),
            param_distributions_fn=lambda b: m_log.PARAM_DIST,
            x_train=x, y_train=y,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_cv_semillas(
            construir_pipeline_fn=lambda s: m_log.construir_pipeline(x, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y, beta=BETA, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC(OOF): {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Logistica regularizada (elastic net, benchmark)", especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="0 + indicador (numericas), 'Sin dato' + one-hot (categoricas)",
            balanceo_info=resultado, hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"  === Logistica regularizada -- {espec} -- FIN ===")


def main() -> None:
    mu.RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"INICIO geo3 CV bajo IPM: CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}, beta={BETA}. 6 combinaciones (3 algoritmos x 2 especificaciones).")
    correr_xgboost()
    correr_histgradientboosting()
    correr_logistica()
    log(f"FIN. Registro: {REGISTRO_CSV}")


if __name__ == "__main__":
    main()
