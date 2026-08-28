"""
Diagnostico: cuanto se solapan los errores (FN y FP) de XGBoost,
HistGradientBoosting, Random Forest y Logistica regularizada en el
conjunto de prueba, Modelo A -- pregunta del usuario (2026-08-28): antes
de invertir en re-entrenar un algoritmo nuevo bajo el criterio F-beta/
folds=10, ver si sus errores ya son suficientemente distintos entre si
(con lo que ya existe, config original F1/folds=3/iter=8) como para que
un ensamble tenga sentido. Primer resultado (solo XGBoost/HistGB/RF):
Random Forest resulto MUY correlacionado con XGBoost (corr=0.972,
Jaccard FN=0.835) -- bagging vs. boosting no aporto la diversidad
esperada. Se agrega Logistica regularizada (lineal, mecanismo distinto a
los tres basados en arboles) como candidata mas probable a diversificar.

Ningun script de la suite guarda las predicciones por hogar (solo metricas
agregadas por semilla) -- este script las regenera reentrenando con la
CONFIGURACION ORIGINAL exacta (CV_FOLDS=3, N_ITER_BUSQUEDA=8,
RANDOM_STATE=42 -- los defaults de modelo_utils.py, iguales a
registro_modelos.csv) para que sea rapido y reproduzca los hiperparametros
ya registrados (ver chequeo de reproducibilidad de conversaciones
anteriores). El umbral de clasificacion se re-elige por CV (F1, igual que
la suite original) sobre cada modelo ya tuneado.

QUE HACE

    1. Entrena XGBoost, HistGradientBoosting y Random Forest en Modelo A
       (semilla 42, balanceo/hiperparametros por AUC-CV -- deberian
       coincidir con lo ya registrado).
    2. Predice probabilidad en el conjunto de prueba (n=3.191 hogares) y
       aplica el umbral elegido por CV (F1) a cada uno.
    3. Compara, por par de algoritmos: correlacion de las probabilidades
       predichas, solapamiento (Jaccard) del conjunto de FALSOS NEGATIVOS,
       solapamiento del conjunto de FALSOS POSITIVOS, y cuantos FN/FP de
       cada algoritmo el OTRO SI acierta (candidatos a corregir via
       ensamble).

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_solapamiento_errores_modelo_A.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_solapamiento_errores.py
"""

import numpy as np
import pandas as pd

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb
import modelo_logistica_regularizada as m_log
import modelo_random_forest as m_rf
import modelo_xgboost as m_xgb

ESPEC = "A"


def entrenar_arboles_nativos(construir_pipeline_fn, param_dist):
    train, test = mu.cargar_datos(ESPEC)
    x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
    x_test, y_test, _ = mu.preparar_arboles_nativos(test)
    x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

    resultado = mu.comparar_balanceo_y_tunear(
        construir_pipeline_fn=construir_pipeline_fn(x_train, y_train),
        param_distributions_fn=lambda b: param_dist,
        x_train=x_train, y_train=y_train,
    )
    pipe = resultado["estimador"]
    umbral = mu.elegir_umbral_por_cv(pipe, x_train, y_train)
    proba_test = pipe.predict_proba(x_test)[:, 1]
    return proba_test, umbral, y_test.reset_index(drop=True), resultado["balanceo_elegido"]


def entrenar_rf():
    train, test = mu.cargar_datos(ESPEC)
    x_train, y_train = mu.preparar_xy_crudo(train)
    x_test, y_test = mu.preparar_xy_crudo(test)
    x_train, x_test = x_train.align(x_test, join="inner", axis=1)

    resultado = mu.comparar_balanceo_y_tunear(
        construir_pipeline_fn=lambda b: m_rf.construir_pipeline(x_train, b),
        param_distributions_fn=lambda b: m_rf.PARAM_DIST,
        x_train=x_train, y_train=y_train,
    )
    pipe = resultado["estimador"]
    umbral = mu.elegir_umbral_por_cv(pipe, x_train, y_train)
    proba_test = pipe.predict_proba(x_test)[:, 1]
    return proba_test, umbral, y_test.reset_index(drop=True), resultado["balanceo_elegido"]


def entrenar_logistica():
    train, test = mu.cargar_datos(ESPEC)
    x_train, y_train = mu.preparar_xy_crudo(train)
    x_test, y_test = mu.preparar_xy_crudo(test)
    x_train, x_test = x_train.align(x_test, join="inner", axis=1)

    resultado = mu.comparar_balanceo_y_tunear(
        construir_pipeline_fn=lambda b: m_log.construir_pipeline(x_train, b),
        param_distributions_fn=lambda b: m_log.PARAM_DIST,
        x_train=x_train, y_train=y_train,
    )
    pipe = resultado["estimador"]
    umbral = mu.elegir_umbral_por_cv(pipe, x_train, y_train)
    proba_test = pipe.predict_proba(x_test)[:, 1]
    return proba_test, umbral, y_test.reset_index(drop=True), resultado["balanceo_elegido"]


def main() -> None:
    print(f"=== Modelo {ESPEC} -- entrenando XGBoost, HistGradientBoosting, Random Forest, Logistica (config original: folds={mu.CV_FOLDS}, iter={mu.N_ITER_BUSQUEDA}) ===")

    proba_xgb, umbral_xgb, y_test, bal_xgb = entrenar_arboles_nativos(
        lambda x_tr, y_tr: (lambda b: m_xgb.construir_pipeline(x_tr, y_tr, b)), m_xgb.PARAM_DIST,
    )
    print(f"  XGBoost: balanceo={bal_xgb}  umbral={umbral_xgb:.3f}")

    proba_hgb, umbral_hgb, y_test2, bal_hgb = entrenar_arboles_nativos(
        lambda x_tr, y_tr: m_hgb.construir_pipeline, m_hgb.PARAM_DIST,
    )
    print(f"  HistGradientBoosting: balanceo={bal_hgb}  umbral={umbral_hgb:.3f}")

    proba_rf, umbral_rf, y_test3, bal_rf = entrenar_rf()
    print(f"  Random Forest: balanceo={bal_rf}  umbral={umbral_rf:.3f}")

    proba_log, umbral_log, y_test4, bal_log = entrenar_logistica()
    print(f"  Logistica regularizada: balanceo={bal_log}  umbral={umbral_log:.3f}")

    assert (y_test.values == y_test2.values).all() and (y_test.values == y_test3.values).all() and (y_test.values == y_test4.values).all(), "y_test debe ser identico entre algoritmos (mismo test set)"

    pred = {
        "XGBoost": (proba_xgb >= umbral_xgb).astype(int),
        "HistGradientBoosting": (proba_hgb >= umbral_hgb).astype(int),
        "RandomForest": (proba_rf >= umbral_rf).astype(int),
        "Logistica": (proba_log >= umbral_log).astype(int),
    }
    proba = {"XGBoost": proba_xgb, "HistGradientBoosting": proba_hgb, "RandomForest": proba_rf, "Logistica": proba_log}
    y = y_test.values

    fn_sets = {nombre: set(np.where((y == 1) & (p == 0))[0]) for nombre, p in pred.items()}
    fp_sets = {nombre: set(np.where((y == 0) & (p == 1))[0]) for nombre, p in pred.items()}

    print(f"\nn_test={len(y)}  positivos_reales={int(y.sum())}")
    for nombre in pred:
        print(f"  {nombre:22s}: FN={len(fn_sets[nombre]):3d}  FP={len(fp_sets[nombre]):3d}")

    filas = []
    nombres = list(pred.keys())
    print("\n=== Solapamiento de errores (pares) ===")
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            a, b = nombres[i], nombres[j]
            corr = float(np.corrcoef(proba[a], proba[b])[0, 1])

            fn_a, fn_b = fn_sets[a], fn_sets[b]
            jacc_fn = len(fn_a & fn_b) / len(fn_a | fn_b) if (fn_a | fn_b) else float("nan")
            fn_a_corregido_por_b = len(fn_a - fn_b)  # hogares que A pierde (FN) pero B SI acierta
            fn_b_corregido_por_a = len(fn_b - fn_a)

            fp_a, fp_b = fp_sets[a], fp_sets[b]
            jacc_fp = len(fp_a & fp_b) / len(fp_a | fp_b) if (fp_a | fp_b) else float("nan")

            print(f"  {a} vs {b}: corr(proba)={corr:.3f}  Jaccard(FN)={jacc_fn:.3f}  Jaccard(FP)={jacc_fp:.3f}")
            print(f"    FN de {a} que {b} SI acierta: {fn_a_corregido_por_b}/{len(fn_a)}   FN de {b} que {a} SI acierta: {fn_b_corregido_por_a}/{len(fn_b)}")

            filas.append({
                "algoritmo_a": a, "algoritmo_b": b, "corr_proba": round(corr, 4),
                "jaccard_fn": round(jacc_fn, 4), "jaccard_fp": round(jacc_fp, 4),
                "fn_a": len(fn_a), "fn_b": len(fn_b),
                "fn_a_corregido_por_b": fn_a_corregido_por_b, "fn_b_corregido_por_a": fn_b_corregido_por_a,
            })

    out = mu.RESULTADOS_DIR / f"diagnostico_solapamiento_errores_modelo_{ESPEC}.csv"
    pd.DataFrame(filas).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
