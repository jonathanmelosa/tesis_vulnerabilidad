"""
XGBoost para prediccion de entrada a la pobreza -- Modelo A (con
ingreso/gasto) vs. Modelo B (sin) -- sobre la particion temporal principal
(train = 2010->2013, test = 2013->2016). Parte de la suite de comparacion
de algoritmos (ver `modelo_utils.py` y docs/decisions.md).

Igual que HistGradientBoosting y LightGBM, XGBoost maneja NaN nativamente
(`missing=np.nan`, default) y categoricas nativamente
(`enable_categorical=True`, columnas en dtype `category`) -- NO se imputa
(missingness estructural, ver docstring de `modelo_utils.py`).

Balanceo de clases: XGBoost no tiene `class_weight` -- el equivalente es
`scale_pos_weight` (razon negativos/positivos). Para "balanced" se computa
desde y_train; para "ninguno" y "oversampling" se deja en 1 (sin ajuste,
o el ajuste ya viene del remuestreo).

Imputacion, balanceo e hiperparametros: ver docstring de `modelo_utils.py`
-- decisiones CONFIRMADAS con el usuario (comparacion de 3 estrategias de
balanceo por AUC-ROC en CV, RandomizedSearchCV para hiperparametros: no se
fijan valores a mano como en la version anterior de este script).
"""

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform

import modelo_utils as mu

OUTPUT_DIR = mu.RESULTADOS_DIR / "xgboost"

PARAM_DIST = {
    "modelo__n_estimators": randint(100, 500),
    "modelo__max_depth": randint(3, 8),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__reg_lambda": loguniform(0.1, 10),
    "modelo__subsample": uniform(0.6, 0.4),
    "modelo__colsample_bytree": uniform(0.6, 0.4),
}


def construir_pipeline(x_train, y_train, balanceo: str) -> ImbPipeline:
    if balanceo == "balanced":
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        scale_pos_weight = 1.0
    modelo = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, scale_pos_weight=scale_pos_weight,
        random_state=mu.RANDOM_STATE, n_jobs=-1, eval_metric="auc",
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=mu.RANDOM_STATE)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for espec in ["A", "B"]:
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        print(f"\n=== XGBoost -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x_train, y_train, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]
        proba_test = pipe.predict_proba(x_test)[:, 1]
        metricas = mu.calcular_metricas(y_test, proba_test)

        modelo_final = pipe.named_steps["modelo"]
        importancias = pd.DataFrame({
            "variable": x_train.columns,
            "importancia": modelo_final.feature_importances_,
        }).sort_values("importancia", ascending=False)
        importancias.to_csv(OUTPUT_DIR / f"importancia_variables_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        print(f"  Hiperparametros: {resultado['mejores_params']}")
        print(f"  AUC-ROC test: {metricas['auc_roc']:.3f}  Recall: {metricas['recall']:.3f}  F1: {metricas['f1']:.3f}")
        print(f"  Top 10 variables mas importantes:")
        print(importancias.head(10).to_string(index=False))

        hiperparametros = {**resultado["mejores_params"], "scale_pos_weight": "auto (balanced)" if resultado["balanceo_elegido"] == "balanced" else 1.0}
        mu.registrar_resultado(
            algoritmo="XGBoost",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            metricas=metricas,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (enable_categorical)",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones="Hiperparametros tuneados por RandomizedSearchCV (AUC-ROC, CV).",
        )

    print(f"\nGuardado en: {OUTPUT_DIR}")
    print(f"Registro actualizado: {mu.REGISTRO_XLSX}")


if __name__ == "__main__":
    main()
