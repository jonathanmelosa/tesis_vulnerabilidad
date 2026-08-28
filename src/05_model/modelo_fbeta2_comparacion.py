"""
Comparacion alternativa: mismo benchmark que produce la Tabla "Contribucion
marginal de las variables geoespaciales" de la tesis (`tab:marginal_dmsp`,
ver `tabla_marginal_dmsp.py`), pero con el umbral de clasificacion elegido
maximizando F-beta (beta=2, pesa el recall el doble que la precision) en
vez de F1 (beta=1) -- ver conversacion con el usuario (Jonathan Melo,
2026-08-27): el objetivo declarado es minimizar falsos negativos en primer
orden y falsos positivos en segundo, y F1 pesa ambos por igual.

QUE NO CAMBIA respecto a `modelo_xgboost.py` / `modelo_histgradientboosting.py`
/ `modelo_logistica_regularizada.py` (documentado para que la comparacion
sea legible): el criterio de seleccion de balanceo de clases e
hiperparametros sigue siendo AUC-ROC (`comparar_balanceo_y_tunear`,
`modelo_utils.SCORING = "roc_auc"`, sin cambios), con los mismos
`CV_FOLDS=3`, `N_ITER_BUSQUEDA=8`, `RANDOM_STATE=42` y `SEMILLAS=[42,1,2,3,4]`
de siempre (decision explicita del usuario: no se cambian folds/iteraciones
en esta corrida). Los espacios de busqueda de hiperparametros (`PARAM_DIST`)
se importan literalmente de los tres scripts originales -- no se
reimplementan -- para que el unico factor que varia frente al benchmark ya
publicado sea el criterio de umbral. Dado que la busqueda de balanceo/
hiperparametros no depende de `beta` y usa los mismos folds y semillas que
la corrida original, `comparar_balanceo_y_tunear` deberia reproducir aqui
el mismo `balanceo_elegido`/`mejores_params` ya guardados en
`registro_modelos.csv` para cada algoritmo/especificacion -- una forma de
verificar en los logs que el pipeline es reproducible antes de comparar
las metricas de umbral.

QUE SI CAMBIA: `elegir_umbral_por_cv` maximiza F-beta con `BETA=2.0` en vez
de F1 (`modelo_utils.elegir_umbral_por_cv(..., beta=2.0)`), y el resultado
se registra en un archivo NUEVO y separado (`registro_modelos_fbeta2.csv`/
`.xlsx`), sin tocar `registro_modelos.csv` ni los CSV de detalle por
semilla/importancia de variables de los scripts originales.

ALCANCE: solo las 4 especificaciones de la Tabla `tab:marginal_dmsp`
(A, B, AgeoDMSP, BgeoDMSP) x los 3 algoritmos alli comparados (XGBoost,
HistGradientBoosting, Logistica regularizada) -- no se corre sobre
ESPECIFICACIONES_ABLATION ni ESPECIFICACIONES_CV_2010_2013.

OUTPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2.csv
    data/processed/benchmark_resultados/registro_modelos_fbeta2.xlsx
    data/processed/benchmark_resultados/fbeta2/{xgboost,histgradientboosting,logistica_regularizada}/
        metricas_multiples_semillas_modelo_{espec}.csv

COMO CORRER

    cd src/05_model && python modelo_fbeta2_comparacion.py
"""

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb
import modelo_logistica_regularizada as m_log
import modelo_xgboost as m_xgb

BETA = 2.0

REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_fbeta2.csv"
REGISTRO_XLSX = mu.RESULTADOS_DIR / "registro_modelos_fbeta2.xlsx"
OUTPUT_ROOT = mu.RESULTADOS_DIR / "fbeta2"

OBSERVACIONES = (
    f"Identico a la corrida principal (balanceo/hiperparametros por AUC-ROC, "
    f"CV_FOLDS={mu.CV_FOLDS}, N_ITER_BUSQUEDA={mu.N_ITER_BUSQUEDA}, "
    f"RANDOM_STATE={mu.RANDOM_STATE}, SEMILLAS={mu.SEMILLAS}), salvo el "
    f"umbral de clasificacion: elegido por CV maximizando F-beta (beta="
    f"{BETA}, pesa recall el doble que precision) en vez de F1. Comparar "
    f"contra registro_modelos.csv (mismo algoritmo/especificacion) para "
    f"aislar el efecto del cambio de criterio de umbral."
)


def correr_xgboost() -> None:
    out_dir = OUTPUT_ROOT / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mu.ESPECIFICACIONES_PRINCIPAL:
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        print(f"\n=== [F-beta={BETA}] XGBoost -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_xgb.construir_pipeline(x_train, y_train, b),
            param_distributions_fn=lambda b: m_xgb.PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: m_xgb.construir_pipeline(x_train, y_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            beta=BETA,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        print(f"  Umbral medio (5 semillas): {multi['resumen']['umbral_media']:.3f}")
        print(f"  AUC-ROC: {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "scale_pos_weight": "auto (balanced)" if resultado["balanceo_elegido"] == "balanced" else 1.0}
        mu.registrar_resultado(
            algoritmo="XGBoost",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (enable_categorical)",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )


def correr_histgradientboosting() -> None:
    out_dir = OUTPUT_ROOT / "histgradientboosting"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mu.ESPECIFICACIONES_PRINCIPAL:
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

        print(f"\n=== [F-beta={BETA}] HistGradientBoosting -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=m_hgb.construir_pipeline,
            param_distributions_fn=lambda b: m_hgb.PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: m_hgb.construir_pipeline(resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            beta=BETA,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        print(f"  Umbral medio (5 semillas): {multi['resumen']['umbral_media']:.3f}")
        print(f"  AUC-ROC: {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="HistGradientBoosting (sklearn)",
            especificacion=espec,
            x_train_shape=x_train.shape, x_test_shape=x_test.shape,
            n_covariables_originales=x_train.shape[1],
            y_train=y_train, y_test=y_test,
            multi_resultado=multi,
            estrategia_imputacion="Ninguna -- soporte nativo de NaN y categoricas (categorical_features='from_dtype')",
            balanceo_info=resultado,
            hiperparametros=hiperparametros,
            observaciones=OBSERVACIONES,
            registro_csv=REGISTRO_CSV, registro_xlsx=REGISTRO_XLSX,
        )


def correr_logistica() -> None:
    out_dir = OUTPUT_ROOT / "logistica_regularizada"
    out_dir.mkdir(parents=True, exist_ok=True)

    for espec in mu.ESPECIFICACIONES_PRINCIPAL:
        train, test = mu.cargar_datos(espec)
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)

        print(f"\n=== [F-beta={BETA}] Logistica regularizada -- Modelo {espec} ===")
        print(f"  train: {x_train.shape}, test: {x_test.shape}")

        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=lambda b: m_log.construir_pipeline(x_train, b),
            param_distributions_fn=lambda b: m_log.PARAM_DIST,
            x_train=x_train, y_train=y_train,
        )
        print(f"  Balanceo elegido: {resultado['balanceo_elegido']} (AUC-CV por balanceo: {resultado['auc_cv_por_balanceo']})")

        multi = mu.evaluar_multiples_semillas(
            construir_pipeline_fn=lambda s: m_log.construir_pipeline(x_train, resultado["balanceo_elegido"], semilla=s),
            mejores_params=resultado["mejores_params"],
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            beta=BETA,
        )
        multi["detalle"].to_csv(out_dir / f"metricas_multiples_semillas_modelo_{espec}.csv", index=False)

        r = multi["resumen"]
        print(f"  Umbral medio (5 semillas): {multi['resumen']['umbral_media']:.3f}")
        print(f"  AUC-ROC: {r['auc_roc']['media']:.3f}  Recall: {r['recall']['media']:.3f}  Precision: {r['precision']['media']:.3f}  F1: {r['f1']['media']:.3f}")

        hiperparametros = {**resultado["mejores_params"], "class_weight": "balanced" if resultado["balanceo_elegido"] == "balanced" else None}
        mu.registrar_resultado(
            algoritmo="Logistica regularizada (elastic net, benchmark)",
            especificacion=espec,
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


def main() -> None:
    mu.RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    correr_xgboost()
    correr_histgradientboosting()
    correr_logistica()
    print(f"\nRegistro nuevo (no toca registro_modelos.csv): {REGISTRO_CSV}")


if __name__ == "__main__":
    main()
