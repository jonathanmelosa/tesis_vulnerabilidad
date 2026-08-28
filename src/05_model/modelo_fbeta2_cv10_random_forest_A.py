"""
Extension puntual de `modelo_fbeta2_cv10_comparacion.py`: Random Forest,
SOLO Modelo A, mismo criterio (F-beta beta=2, CV_FOLDS=10,
N_ITER_BUSQUEDA=30) -- para poder comparar su AUC-ROC contra XGBoost/
HistGradientBoosting/Logistica en la MISMA configuracion (ver conversacion
2026-08-28: comparar el AUC de Random Forest bajo folds=3/iter=8 contra
los otros bajo folds=10/iter=30 no era una comparacion valida). No se
corren B/AgeoDMSP/BgeoDMSP -- alcance acotado a lo que se necesita para
resolver esa duda puntual antes de decidir si Random Forest entra al
ensamble.

Escribe en el MISMO registro que `modelo_fbeta2_cv10_comparacion.py`
(`registro_modelos_fbeta2_cv10.csv`) para que quede directamente
comparable ahi -- upsert por (algoritmo, especificacion), no pisa las
filas de los otros 3 algoritmos.

COMO CORRER

    cd src/05_model && python -u modelo_fbeta2_cv10_random_forest_A.py
"""

import time
from datetime import datetime

import modelo_utils as mu
import modelo_random_forest as m_rf

BETA = 2.0
CV_FOLDS = 10
N_ITER_BUSQUEDA = 30
ESPEC = "A"

REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv"
REGISTRO_XLSX = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.xlsx"
OUT_DIR = mu.RESULTADOS_DIR / "fbeta2_cv10" / "random_forest"

OBSERVACIONES = (
    f"Extension puntual de modelo_fbeta2_cv10_comparacion.py: Random Forest, "
    f"solo Modelo A, para comparar AUC-ROC contra XGBoost/HistGB/Logistica en "
    f"la misma config (F-beta={BETA}, CV_FOLDS={CV_FOLDS}, "
    f"N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}) -- la comparacion previa contra "
    f"registro_modelos.csv (folds=3/iter=8) no era valida. RANDOM_STATE="
    f"{mu.RANDOM_STATE}, SEMILLAS={mu.SEMILLAS} sin cambios."
)

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, test = mu.cargar_datos(ESPEC)
    x_train, y_train = mu.preparar_xy_crudo(train)
    x_test, y_test = mu.preparar_xy_crudo(test)
    x_train, x_test = x_train.align(x_test, join="inner", axis=1)

    log(f"=== Random Forest -- Modelo {ESPEC} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

    resultado = mu.comparar_balanceo_y_tunear(
        construir_pipeline_fn=lambda b: m_rf.construir_pipeline(x_train, b),
        param_distributions_fn=lambda b: m_rf.PARAM_DIST,
        x_train=x_train, y_train=y_train,
        cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
    )
    log(f"  Balanceo/hiperparametros listos -- elegido: {resultado['balanceo_elegido']} (AUC-CV: {resultado['auc_cv_por_balanceo']})")

    multi = mu.evaluar_multiples_semillas(
        construir_pipeline_fn=lambda s: m_rf.construir_pipeline(x_train, resultado["balanceo_elegido"], semilla=s),
        mejores_params=resultado["mejores_params"],
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        beta=BETA, cv_folds=CV_FOLDS,
    )
    multi["detalle"].to_csv(OUT_DIR / f"metricas_multiples_semillas_modelo_{ESPEC}.csv", index=False)

    r = multi["resumen"]
    log(f"  Umbral medio: {r['umbral_media']:.3f}  AUC-ROC: {r['auc_roc']['media']:.4f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

    hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
    mu.registrar_resultado(
        algoritmo="Random Forest",
        especificacion=ESPEC,
        x_train_shape=x_train.shape, x_test_shape=x_test.shape,
        n_covariables_originales=x_train.shape[1],
        y_train=y_train, y_test=y_test,
        multi_resultado=multi,
        estrategia_imputacion="0 + indicador (numericas), 'Sin dato' + one-hot (categoricas) -- ver docstring de modelo_utils.py",
        balanceo_info=resultado,
        hiperparametros=hiperparametros,
        observaciones=OBSERVACIONES,
        registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
    )
    log(f"=== Random Forest -- Modelo {ESPEC} -- FIN. Registro: {REGISTRO_CSV} ===")


if __name__ == "__main__":
    main()
