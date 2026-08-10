"""
BENCHMARK: regresion logistica con regularizacion elastic net para
prediccion de entrada a la pobreza -- Modelo A (con ingreso/gasto) vs.
Modelo B (sin) -- sobre la particion temporal principal (train =
2010->2013, test = 2013->2016).

Por que logistica regularizada como benchmark
--------------------------------------------------------------------------
Estandar de facto en la literatura de prediccion de pobreza/vulnerabilidad
(interpretable, coeficientes con signo e intuicion economica directa).
Con p ~ 165-169 covariables (mas tras one-hot) y n ~ 3.089 (train), sin
regularizar el modelo sobreajustaria -- de ahi la penalizacion. Se usa
`penalty="elasticnet"`, `solver="saga"`, con `C` y `l1_ratio` tuneados por
`RandomizedSearchCV` (ver `modelo_utils.py`) en vez de comprometerse a
priori con Lasso o Ridge puro.

Imputacion, balanceo de clases e hiperparametros: ver docstring de
`modelo_utils.py` -- decisiones CONFIRMADAS explicitamente con el usuario
(0+indicador para missings, comparacion de 3 estrategias de balanceo
seleccionando por AUC-ROC en CV, RandomizedSearchCV para hiperparametros).
Para la estrategia "oversampling", `class_weight` NO se fija (evita doble
ajuste combinando remuestreo + reponderacion).
"""

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, uniform
from sklearn.linear_model import LogisticRegression

import modelo_utils as mu

OUTPUT_DIR = mu.RESULTADOS_DIR / "logistica_regularizada"

PARAM_DIST = {
    "modelo__C": loguniform(1e-3, 1e2),
    "modelo__l1_ratio": uniform(0, 1),
}


def construir_pipeline(x_train, balanceo: str) -> ImbPipeline:
    preprocesador = mu.construir_preprocesador(x_train, escalar=True)
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = LogisticRegression(
        penalty="elasticnet", solver="saga", class_weight=class_weight,
        max_iter=5000, random_state=mu.RANDOM_STATE,
    )
    pasos = [("prep", preprocesador)]
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=mu.RANDOM_STATE)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for espec in ["A", "B"]:
        train, test = mu.cargar_datos(espec)
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)

        print(f"\n=== Logistica regularizada -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        pipe = resultado["estimador"]
        umbral = mu.elegir_umbral_por_cv(pipe, x_train, y_train)
        proba_test = pipe.predict_proba(x_test)[:, 1]
        metricas = mu.calcular_metricas(y_test, proba_test, umbral=umbral)

        modelo_final = pipe.named_steps["modelo"]
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        import pandas as pd
        coeficientes = pd.DataFrame({
            "variable": nombres_features,
            "coeficiente": modelo_final.coef_[0],
        }).sort_values("coeficiente", key=np.abs, ascending=False)
        coeficientes.to_csv(OUTPUT_DIR / f"coeficientes_modelo_{espec}.csv", index=False)

        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")
        print(f"  Hiperparametros: {resultado['mejores_params']}")
        print(f"  Umbral elegido por CV: {umbral:.2f}")
        print(f"  AUC-ROC test: {metricas['auc_roc']:.3f}  Recall: {metricas['recall']:.3f}  F1: {metricas['f1']:.3f}")
        print(f"  Top 10 |coeficiente| mayor:")
        print(coeficientes.head(10).to_string(index=False))

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Logistica regularizada (elastic net, benchmark)",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            metricas=metricas,
            estrategia_imputacion="0 + indicador de faltante (numericas), 'Sin dato' + one-hot drop-first (categoricas), estandarizacion",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones="Benchmark designado por el usuario. C y l1_ratio tuneados por RandomizedSearchCV (AUC-ROC, CV). Umbral de clasificacion elegido por CV maximizando F1 (no fijo en 0.5). Coeficientes en unidades estandarizadas, signo interpretable directamente.",
            umbral_clasificacion=umbral,
        )

    print(f"\nGuardado en: {OUTPUT_DIR}")
    print(f"Registro actualizado: {mu.REGISTRO_XLSX}")


if __name__ == "__main__":
    main()
