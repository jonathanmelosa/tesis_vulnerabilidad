"""
HistGradientBoostingClassifier (sklearn) para prediccion de entrada a la
pobreza -- Modelo A (con ingreso/gasto) vs. Modelo B (sin) -- sobre la
particion temporal principal (train = 2010->2013, test = 2013->2016).
Parte de la suite de comparacion de algoritmos (ver `modelo_utils.py` y
docs/decisions.md).

NOTA: reemplaza a `entrenar_benchmark.py` (version anterior a la suite,
cuando HistGB era el unico modelo entrenado, con hiperparametros fijos y
sin comparacion de balanceo). Ahora usa la misma arquitectura que el resto
de la suite: comparacion de 3 estrategias de balanceo por AUC-ROC en CV +
RandomizedSearchCV para hiperparametros (ver docstring de
`modelo_utils.py`, decisiones CONFIRMADAS con el usuario).

Por que HistGradientBoosting en esta suite
--------------------------------------------------------------------------
Maneja NaN y categoricas nativamente (missingness estructural, ver
docstring de `modelo_utils.py`) -- mismo argumento que XGBoost/LightGBM,
las otras dos implementaciones de gradient boosting de la suite.
Columnas categoricas en dtype `category`, `categorical_features="from_dtype"`.

Ademas de las metricas estandar, reporta importancia de variables por
permutation importance (AUC en el conjunto de prueba, 10 repeticiones) --
mas costosa pero mas robusta que `feature_importances_`.
"""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

import modelo_utils as mu

OUTPUT_DIR = mu.RESULTADOS_DIR / "histgradientboosting"

PARAM_DIST = {
    "modelo__max_iter": randint(100, 400),
    "modelo__max_depth": randint(3, 8),
    "modelo__max_leaf_nodes": randint(15, 63),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__l2_regularization": loguniform(0.01, 10),
}


def construir_pipeline(balanceo: str) -> ImbPipeline:
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = HistGradientBoostingClassifier(
        categorical_features="from_dtype", class_weight=class_weight, random_state=mu.RANDOM_STATE,
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

        print(f"\n=== HistGradientBoosting -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=construir_pipeline,
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]
        umbral = mu.elegir_umbral_por_cv(pipe, x_train, y_train)
        proba_test = pipe.predict_proba(x_test)[:, 1]
        metricas = mu.calcular_metricas(y_test, proba_test, umbral=umbral)

        modelo_final = pipe.named_steps["modelo"]
        imp = permutation_importance(modelo_final, x_test, y_test, scoring="roc_auc", n_repeats=10, random_state=mu.RANDOM_STATE, n_jobs=-1)
        importancias = pd.DataFrame({
            "variable": x_test.columns,
            "importancia_media": imp.importances_mean,
            "importancia_std": imp.importances_std,
        }).sort_values("importancia_media", ascending=False)
        importancias.to_csv(OUTPUT_DIR / f"importancia_variables_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        print(f"  Hiperparametros: {resultado['mejores_params']}")
        print(f"  Umbral elegido por CV: {umbral:.2f}")
        print(f"  AUC-ROC test: {metricas['auc_roc']:.3f}  Recall: {metricas['recall']:.3f}  F1: {metricas['f1']:.3f}")
        print(f"  Top 10 variables mas importantes:")
        print(importancias.head(10).to_string(index=False))

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="HistGradientBoosting (sklearn)",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            metricas=metricas,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (categorical_features='from_dtype')",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones="Reemplaza a entrenar_benchmark.py. Ahora con comparacion de balanceo y RandomizedSearchCV (antes hiperparametros fijos). Importancia por permutation importance (AUC, 10 repeticiones). Umbral de clasificacion elegido por CV maximizando F1 (no fijo en 0.5).",
            umbral_clasificacion=umbral,
        )

    print(f"\nGuardado en: {OUTPUT_DIR}")
    print(f"Registro actualizado: {mu.REGISTRO_XLSX}")


if __name__ == "__main__":
    main()
