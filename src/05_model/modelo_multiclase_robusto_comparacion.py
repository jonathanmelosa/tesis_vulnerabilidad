"""
modelo_multiclase_robusto_comparacion.py
===========================================

Suite de comparacion de 5 algoritmos (XGBoost, LightGBM,
HistGradientBoosting, Random Forest, Logistica regularizada) sobre el
benchmark de 4 CLASES (nunca_pobre/entra/sale/siempre_pobre,
`build_benchmark_train_test_4clases.py`), en "version robusta"
(CV_FOLDS=10, N_ITER_BUSQUEDA=30, SEMILLAS=[42,1,2,3,4] -- identico a
`modelo_fbeta2_cv10_comparacion.py`, la version robusta ya usada y
confirmada por el usuario en el benchmark binario). Ver
docs/decisions.md, "2026-09-16: Extension a modelo multiclase", para la
motivacion, el diagnostico que la origina, y el diseno completo.

NO modifica ni sobrescribe nada del benchmark binario (`modelo_utils.py`,
`registro_modelos*.csv`, los scripts `modelo_*.py` originales) -- usa
`modelo_utils_multiclase.py` (nuevo) y escribe en un registro separado
(`registro_modelos_multiclase.csv`, bajo
`data/processed/benchmark_resultados/multiclase/`).

Alcance (decision explicita del usuario: incluir TODAS las fuentes
geoespaciales en esta corrida para no tener que repetirla despues):
  - Pista principal (holdout temporal 2010->2013/2013->2016): A4, B4,
    A4geoDMSP, B4geoDMSP -- los 5 algoritmos.
  - Pista exploratoria (CV dentro de 2010->2013, sin holdout, las 3
    fuentes geoespaciales completas -- DMSP-OLS+ALOS PALSAR+Landsat 5 TM):
    A4geo3, B4geo3 -- los 5 algoritmos.

Balanceo de clases: identico en estructura al binario (3 estrategias,
elegidas por AUC-ROC-OVR en CV) -- ver docstring de
`modelo_utils_multiclase.py` para el detalle de como XGBoost maneja
"balanced" (sample_weight en vez de class_weight, unico caso especial).

Salida por algoritmo (bajo
`data/processed/benchmark_resultados/multiclase/{algoritmo}/`):
  - `metricas_multiples_semillas_modelo_{espec}.csv` (accuracy,
    balanced_accuracy, f1_macro/weighted, precision/recall/f1 de "entra",
    auc_ovr_macro, auc_entra_vs_resto, auc_entra_vs_sale,
    precision_top10_entra -- por semilla)
  - `importancia_variables_modelo_{espec}.csv` (arboles) o
    `coeficientes_por_clase_modelo_{espec}.csv` (logistica -- un
    coeficiente por clase y por variable, mas la columna
    `entra_menos_sale`, el contraste que responde directamente la
    pregunta de la Seccion 5.1 sobre la frontera entra/sale)

COMO CORRER
-----------
    cd src/05_model && python -u modelo_multiclase_robusto_comparacion.py
    (lanzar en segundo plano, envuelto en `caffeinate -i`, ver
    docs/decisions.md -- corrida larga, no bloquea la sesion de trabajo)
"""

import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

import modelo_utils as mu
import modelo_utils_multiclase as mcu

CV_FOLDS = 10
N_ITER_BUSQUEDA = 30

REGISTRO_CSV = mcu.REGISTRO_CSV
REGISTRO_XLSX = mcu.REGISTRO_XLSX

OBSERVACIONES_PRINCIPAL = (
    "Benchmark de 4 clases (nunca_pobre/entra/sale/siempre_pobre), "
    "poblacion completa (NO filtrada a no-pobres en la ola base -- ver "
    "build_benchmark_train_test_4clases.py y docs/decisions.md, "
    "'2026-09-16: Extension a modelo multiclase'). Holdout temporal "
    "2010->2013 / 2013->2016, CV_FOLDS=10, N_ITER_BUSQUEDA=30, "
    "SEMILLAS=[42,1,2,3,4]. Decision explicita del usuario sobre como "
    "interpretar precision/recall de 'entra': si bajan respecto al "
    "benchmark binario, NO es 'el modelo empeoro' -- es que 'sale' "
    "(el grupo mas parecido a 'entra', Seccion 5.1) ahora esta en el "
    "'resto', haciendo la tarea mas dificil y mas fiel a la definicion "
    "de vulnerabilidad. NO comparable cifra a cifra contra "
    "registro_modelos.csv/registro_modelos_fbeta2_cv10.csv (target y "
    "poblacion distintos)."
)
OBSERVACIONES_CV = (
    "PISTA EXPLORATORIA (igual que Ageo3/Bgeo3 en el binario): ELCA + "
    "DMSP-OLS + ALOS PALSAR + Landsat 5 TM, restringido a la transicion "
    "2010->2013, benchmark de 4 clases. Sin holdout temporal -- metricas "
    "sobre probabilidades OUT-OF-FOLD dentro de la misma muestra. NO "
    "comparable cifra a cifra contra A4/B4/A4geoDMSP/B4geoDMSP (holdout) "
    "ni contra Ageo3/Bgeo3 del binario (target distinto)."
)

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


# ---------------------------------------------------------------------
# XGBoost (arboles nativos -- NaN/categoricas sin imputar)
# ---------------------------------------------------------------------

PARAM_DIST_XGB = {
    "modelo__n_estimators": randint(100, 500),
    "modelo__max_depth": randint(3, 8),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__reg_lambda": loguniform(0.1, 10),
    "modelo__subsample": uniform(0.6, 0.4),
    "modelo__colsample_bytree": uniform(0.6, 0.4),
}


def pipeline_xgb(balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    modelo = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True,
        objective="multi:softprob", num_class=len(mcu.CATEGORIAS_Y_GRUPO),
        random_state=semilla, n_jobs=-1, eval_metric="mlogloss",
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


def fit_params_xgb(balanceo: str, y_train) -> dict:
    if balanceo == "balanced":
        return {"modelo__sample_weight": mcu.sample_weight_balanced(y_train)}
    return {}


# ---------------------------------------------------------------------
# LightGBM (arboles nativos)
# ---------------------------------------------------------------------

PARAM_DIST_LGB = {
    "modelo__n_estimators": randint(100, 500),
    "modelo__max_depth": randint(3, 8),
    "modelo__num_leaves": randint(15, 63),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__reg_lambda": loguniform(0.1, 10),
    "modelo__min_child_samples": randint(5, 50),
}


def pipeline_lgb(balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(mcu.CATEGORIAS_Y_GRUPO),
        class_weight=class_weight, random_state=semilla, n_jobs=-1, verbosity=-1,
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# HistGradientBoosting (arboles nativos) -- multiclase automatico
# ---------------------------------------------------------------------

PARAM_DIST_HGB = {
    "modelo__max_iter": randint(100, 400),
    "modelo__max_depth": randint(3, 8),
    "modelo__max_leaf_nodes": randint(15, 63),
    "modelo__learning_rate": loguniform(0.01, 0.3),
    "modelo__l2_regularization": loguniform(0.01, 10),
}


def pipeline_hgb(balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = HistGradientBoostingClassifier(
        categorical_features="from_dtype", class_weight=class_weight, random_state=semilla,
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# Random Forest (preprocesador 0+indicador/one-hot, sin escalar)
# ---------------------------------------------------------------------

PARAM_DIST_RF = {
    "modelo__n_estimators": randint(200, 600),
    "modelo__max_depth": randint(3, 15),
    "modelo__min_samples_leaf": randint(2, 30),
    "modelo__max_features": uniform(0.2, 0.6),
}


def pipeline_rf(x_train: pd.DataFrame, balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    preprocesador = mu.construir_preprocesador(x_train, escalar=False)
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = RandomForestClassifier(class_weight=class_weight, random_state=semilla, n_jobs=-1)
    pasos = [("prep", preprocesador)]
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# Logistica regularizada (preprocesador + escalado; multinomial nativo
# de sklearn para >2 clases)
# ---------------------------------------------------------------------

PARAM_DIST_LOG = {
    "modelo__C": loguniform(1e-3, 1e2),
    "modelo__l1_ratio": uniform(0, 1),
}


def pipeline_log(x_train: pd.DataFrame, balanceo: str, semilla: int = mcu.RANDOM_STATE) -> ImbPipeline:
    preprocesador = mu.construir_preprocesador(x_train, escalar=True)
    class_weight = "balanced" if balanceo == "balanced" else None
    modelo = LogisticRegression(
        penalty="elasticnet", solver="saga", class_weight=class_weight,
        max_iter=5000, random_state=semilla,
    )
    pasos = [("prep", preprocesador)]
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ---------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------

def _log_resumen(r: dict) -> None:
    log(
        f"  accuracy: {r['accuracy']['media']:.3f}  f1_macro: {r['f1_macro']['media']:.3f}  "
        f"recall_entra: {r['recall_entra']['media']:.3f}  precision_entra: {r['precision_entra']['media']:.3f}  "
        f"auc_ovr_macro: {r['auc_ovr_macro']['media']:.3f}  auc_entra_vs_sale: {r['auc_entra_vs_sale']['media']:.3f}"
    )


def correr_arbol_principal(nombre_algoritmo: str, out_subdir: str, pipeline_fn, param_dist: dict, fit_params_fn=None) -> None:
    out_dir = mcu.RESULTADOS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL:
        train, test = mcu.cargar_datos_4clases(espec)
        x_train, y_train, cat_cols = mcu.preparar_arboles_nativos_multiclase(train)
        x_test, y_test, _ = mcu.preparar_arboles_nativos_multiclase(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        log(f"=== {nombre_algoritmo} -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        ffn = (lambda b: fit_params_fn(b, y_train)) if fit_params_fn else None
        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=lambda b: pipeline_fn(b),
            param_distributions_fn=lambda b: param_dist,
            x_train=x_train, y_train=y_train, fit_params_fn=ffn,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        modelo_final = pipe.named_steps["modelo"]
        if hasattr(modelo_final, "feature_importances_"):
            importancias = pd.DataFrame({
                "variable": x_train.columns, "importancia": modelo_final.feature_importances_,
            }).sort_values("importancia", ascending=False)
            importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)

        ffn_semilla = (lambda s: fit_params_fn(resultado["balanceo_elegido"], y_train)) if fit_params_fn else None
        multi = mcu.evaluar_multiples_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_fn(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            fit_params_fn=ffn_semilla,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo=nombre_algoritmo, especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_PRINCIPAL,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== {nombre_algoritmo} -- Modelo {espec} -- FIN ===")


def correr_arbol_cv(nombre_algoritmo: str, out_subdir: str, pipeline_fn, param_dist: dict, fit_params_fn=None) -> None:
    out_dir = mcu.RESULTADOS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mcu.ESPECIFICACIONES_4CLASES_CV:
        datos = mcu.cargar_datos_cv_4clases(espec)
        x, y, cat_cols = mcu.preparar_arboles_nativos_multiclase(datos)

        log(f"=== {nombre_algoritmo} -- Modelo {espec} (CV, sin holdout) -- INICIO (n={x.shape[0]}) ===")

        ffn = (lambda b: fit_params_fn(b, y)) if fit_params_fn else None
        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=lambda b: pipeline_fn(b),
            param_distributions_fn=lambda b: param_dist,
            x_train=x, y_train=y, fit_params_fn=ffn,
            cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        modelo_final = pipe.named_steps["modelo"]
        if hasattr(modelo_final, "feature_importances_"):
            importancias = pd.DataFrame({
                "variable": x.columns, "importancia": modelo_final.feature_importances_,
            }).sort_values("importancia", ascending=False)
            importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)

        ffn_semilla = (lambda s: fit_params_fn(resultado["balanceo_elegido"], y)) if fit_params_fn else None
        multi = mcu.evaluar_cv_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_fn(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x=x, y=y, fit_params_fn=ffn_semilla, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo=nombre_algoritmo, especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_CV,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== {nombre_algoritmo} -- Modelo {espec} (CV) -- FIN ===")


def correr_hgb() -> None:
    """HistGB no tiene feature_importances_ nativo -- permutation importance, igual que el binario."""
    out_dir = mcu.RESULTADOS_DIR / "histgradientboosting"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL:
        train, test = mcu.cargar_datos_4clases(espec)
        x_train, y_train, cat_cols = mcu.preparar_arboles_nativos_multiclase(train)
        x_test, y_test, _ = mcu.preparar_arboles_nativos_multiclase(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        log(f"=== HistGradientBoosting -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=pipeline_hgb, param_distributions_fn=lambda b: PARAM_DIST_HGB,
            x_train=x_train, y_train=y_train, cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        imp = permutation_importance(pipe.named_steps["modelo"], x_test, y_test, scoring="roc_auc_ovr", n_repeats=10, random_state=mcu.RANDOM_STATE, n_jobs=-1)
        importancias = pd.DataFrame({
            "variable": x_test.columns, "importancia_media": imp.importances_mean, "importancia_std": imp.importances_std,
        }).sort_values("importancia_media", ascending=False)
        importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)

        multi = mcu.evaluar_multiples_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_hgb(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo="HistGradientBoosting (sklearn)", especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_PRINCIPAL,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== HistGradientBoosting -- Modelo {espec} -- FIN ===")

    for espec in mcu.ESPECIFICACIONES_4CLASES_CV:
        datos = mcu.cargar_datos_cv_4clases(espec)
        x, y, cat_cols = mcu.preparar_arboles_nativos_multiclase(datos)

        log(f"=== HistGradientBoosting -- Modelo {espec} (CV, sin holdout) -- INICIO (n={x.shape[0]}) ===")

        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=pipeline_hgb, param_distributions_fn=lambda b: PARAM_DIST_HGB,
            x_train=x, y_train=y, cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        multi = mcu.evaluar_cv_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_hgb(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"], x=x, y=y, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo="HistGradientBoosting (sklearn)", especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas",
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_CV,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== HistGradientBoosting -- Modelo {espec} (CV) -- FIN ===")


def _correr_lineal(nombre_algoritmo: str, out_subdir: str, pipeline_fn_train) -> None:
    """Comun a Random Forest y Logistica (preprocesador dependiente de x_train)."""
    out_dir = mcu.RESULTADOS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    es_logistica = nombre_algoritmo.startswith("Logistica")
    param_dist = PARAM_DIST_LOG if es_logistica else PARAM_DIST_RF

    for espec in mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL:
        train, test = mcu.cargar_datos_4clases(espec)
        x_train, y_train = mcu.preparar_xy_crudo_multiclase(train)
        x_test, y_test = mcu.preparar_xy_crudo_multiclase(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)

        log(f"=== {nombre_algoritmo} -- Modelo {espec} -- INICIO (train {x_train.shape}, test {x_test.shape}) ===")

        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=lambda b: pipeline_fn_train(x_train, b),
            param_distributions_fn=lambda b: param_dist,
            x_train=x_train, y_train=y_train, cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        modelo_final = pipe.named_steps["modelo"]
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        if es_logistica:
            coefs = pd.DataFrame(modelo_final.coef_.T, columns=[f"coef_{mcu.CATEGORIAS_Y_GRUPO[k]}" for k in modelo_final.classes_])
            coefs.insert(0, "variable", nombres_features)
            coefs["entra_menos_sale"] = coefs[f"coef_entra"] - coefs[f"coef_sale"]
            coefs = coefs.sort_values("entra_menos_sale", key=np.abs, ascending=False)
            coefs.to_csv(out_dir / f"coeficientes_por_clase_modelo_{espec}.csv", index=False)
        else:
            importancias = pd.DataFrame({
                "variable": nombres_features, "importancia": modelo_final.feature_importances_,
            }).sort_values("importancia", ascending=False)
            importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)

        multi = mcu.evaluar_multiples_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_fn_train(x_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo=nombre_algoritmo, especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test, multi_resultado=multi,
            estrategia_imputacion="0 + indicador (numericas), 'Sin dato' + one-hot (categoricas)" + (", estandarizacion" if es_logistica else ""),
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_PRINCIPAL,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== {nombre_algoritmo} -- Modelo {espec} -- FIN ===")

    for espec in mcu.ESPECIFICACIONES_4CLASES_CV:
        datos = mcu.cargar_datos_cv_4clases(espec)
        x, y = mcu.preparar_xy_crudo_multiclase(datos)

        log(f"=== {nombre_algoritmo} -- Modelo {espec} (CV, sin holdout) -- INICIO (n={x.shape[0]}) ===")

        resultado = mcu.comparar_balanceo_y_tunear_multiclase(
            construir_pipeline_fn=lambda b: pipeline_fn_train(x, b),
            param_distributions_fn=lambda b: param_dist,
            x_train=x, y_train=y, cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA, verbose=1,
        )
        log(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-OVR-CV: {resultado['auc_cv_por_balanceo']})")

        pipe = resultado["estimador"]
        modelo_final = pipe.named_steps["modelo"]
        nombres_features = pipe.named_steps["prep"].get_feature_names_out()
        if es_logistica:
            coefs = pd.DataFrame(modelo_final.coef_.T, columns=[f"coef_{mcu.CATEGORIAS_Y_GRUPO[k]}" for k in modelo_final.classes_])
            coefs.insert(0, "variable", nombres_features)
            coefs["entra_menos_sale"] = coefs["coef_entra"] - coefs["coef_sale"]
            coefs = coefs.sort_values("entra_menos_sale", key=np.abs, ascending=False)
            coefs.to_csv(out_dir / f"coeficientes_por_clase_modelo_{espec}.csv", index=False)
        else:
            importancias = pd.DataFrame({
                "variable": nombres_features, "importancia": modelo_final.feature_importances_,
            }).sort_values("importancia", ascending=False)
            importancias.to_csv(out_dir / f"importancia_variables_modelo_{espec}.csv", index=False)

        multi = mcu.evaluar_cv_semillas_multiclase(
            construir_pipeline_fn=lambda s: pipeline_fn_train(x, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"], x=x, y=y, cv_folds=CV_FOLDS,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)
        _log_resumen(multi["resumen"])

        mcu.registrar_resultado_multiclase(
            algoritmo=nombre_algoritmo, especificacion=espec,
            x_train_shape=x.shape, x_test_shape=x.shape,
            n_covariables_originales=x.shape[1],
            y_train=y, y_test=y, multi_resultado=multi,
            estrategia_imputacion="0 + indicador (numericas), 'Sin dato' + one-hot (categoricas)" + (", estandarizacion" if es_logistica else ""),
            balanceo_info=resultado, hiperparametros=resultado["mejores_params"],
            observaciones=OBSERVACIONES_CV,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )
        log(f"=== {nombre_algoritmo} -- Modelo {espec} (CV) -- FIN ===")


def main() -> None:
    mcu.RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"INICIO suite multiclase robusta: CV_FOLDS={CV_FOLDS}, N_ITER_BUSQUEDA={N_ITER_BUSQUEDA}. "
        f"5 algoritmos x ({len(mcu.ESPECIFICACIONES_4CLASES_PRINCIPAL)} principal + {len(mcu.ESPECIFICACIONES_4CLASES_CV)} CV) especificaciones.")

    log("\n### XGBoost ###")
    correr_arbol_principal("XGBoost", "xgboost", pipeline_xgb, PARAM_DIST_XGB, fit_params_fn=fit_params_xgb)
    correr_arbol_cv("XGBoost", "xgboost", pipeline_xgb, PARAM_DIST_XGB, fit_params_fn=fit_params_xgb)

    log("\n### LightGBM ###")
    correr_arbol_principal("LightGBM", "lightgbm", pipeline_lgb, PARAM_DIST_LGB)
    correr_arbol_cv("LightGBM", "lightgbm", pipeline_lgb, PARAM_DIST_LGB)

    log("\n### HistGradientBoosting ###")
    correr_hgb()

    log("\n### Random Forest ###")
    _correr_lineal("Random Forest", "random_forest", pipeline_rf)

    log("\n### Logistica regularizada (elastic net) ###")
    _correr_lineal("Logistica regularizada (elastic net, benchmark)", "logistica_regularizada", pipeline_log)

    log(f"\nFIN. Registro: {REGISTRO_CSV}")


if __name__ == "__main__":
    main()
