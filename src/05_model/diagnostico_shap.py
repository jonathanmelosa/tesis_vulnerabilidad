"""
Analisis SHAP (SHapley Additive exPlanations) para XGBoost y
HistGradientBoosting, en las especificaciones CON DMSP-OLS de ambos
targets (AgeoDMSP -- pobreza monetaria; AipmgeoDMSP -- IPM) -- pedido por
el usuario (2026-08-30) para profundizar en la pregunta abierta de los
informes: ¿por que XGBoost capta un efecto de AUC significativo de
DMSP-OLS bajo IPM y HistGB no, si la comparacion de importancia de
variables (feature_importances_/permutation importance) no mostraba una
diferencia grande entre ambos?

SHAP tiene una ventaja sobre feature_importances_/permutation importance
para esta pregunta especifica: satisface propiedades de aditividad
(la suma de los SHAP values de un hogar reconstruye exactamente la
prediccion) y consistencia (si un modelo depende MAS de una variable,
su SHAP value nunca puede bajar) -- una comparacion mas rigurosa de
"cuanto pesa cada variable en cada prediccion individual" que las
alternativas ya usadas.

NO reentrena con RandomizedSearchCV -- reutiliza balanceo_elegido y
mejores_params ya encontrados (mismo patron que los scripts de
bootstrap), un solo fit por combinacion.

QUE HACE

    Para cada uno de los 4 pares (algoritmo x target):
    1. Reconstruye el modelo ganador y lo entrena sobre x_train.
    2. Calcula SHAP values sobre el conjunto de PRUEBA (shap.TreeExplainer,
       soporta XGBoost y HistGradientBoostingClassifier de forma nativa).
    3. Reporta: |SHAP| medio por variable (ranking global), el ranking
       especifico de las 23 variables DMSP-OLS dentro de ese ranking, y
       la suma total de |SHAP| de DMSP-OLS como % del total de todas las
       variables (mide cuanto "peso" total tienen las 23 variables juntas
       en las predicciones, mas alla del ranking de la mejor variable
       individual).

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_importancia.csv
    (ranking completo de variables por combinacion)
    data/processed/benchmark_resultados/diagnostico_shap_resumen_dmsp.csv
    (resumen: peso total y ranking de DMSP-OLS por combinacion)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap.py
"""

import json

import numpy as np
import pandas as pd
import shap

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb
import modelo_xgboost as m_xgb

# shap.TreeExplainer no soporta columnas dtype 'category' de pandas para
# HistGradientBoostingClassifier (intenta `X.astype(model.input_dtype)`
# sobre las categorias como texto -- ValueError). XGBoost SI las soporta
# nativamente via shap. Fix: para HistGB, codificar las categoricas a sus
# codigos numericos (`.cat.codes`, -1->NaN) antes de llamar a SHAP --
# verificado que reproduce el mismo split interno que usa el modelo.
NECESITA_CODIFICAR_CATEGORICAS = {"XGBoost": False, "HistGradientBoosting (sklearn)": True}

COMBINACIONES = [
    ("XGBoost", "AgeoDMSP", "Monetaria"),
    ("XGBoost", "AipmgeoDMSP", "IPM"),
    ("HistGradientBoosting (sklearn)", "AgeoDMSP", "Monetaria"),
    ("HistGradientBoosting (sklearn)", "AipmgeoDMSP", "IPM"),
]

REGISTROS = {
    "Monetaria": mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv",
    "IPM": mu.RESULTADOS_DIR / "registro_modelos_ipm.csv",
}


def filtrar_params_modelo(hiperparametros_json: str) -> dict:
    d = json.loads(hiperparametros_json)
    return {k: v for k, v in d.items() if k.startswith("modelo__")}


def entrenar(algoritmo_raw: str, espec: str, registro: pd.DataFrame):
    fila = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)].iloc[0]
    balanceo = fila["balanceo_elegido"]
    params = filtrar_params_modelo(fila["hiperparametros"])

    train, test = mu.cargar_datos(espec)
    x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
    x_test, y_test, _ = mu.preparar_arboles_nativos(test)
    x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

    if algoritmo_raw == "XGBoost":
        pipe = m_xgb.construir_pipeline(x_train, y_train, balanceo)
    else:
        pipe = m_hgb.construir_pipeline(balanceo)

    if params:
        pipe.set_params(**params)
    pipe.fit(x_train, y_train)

    if NECESITA_CODIFICAR_CATEGORICAS[algoritmo_raw]:
        x_test = x_test.copy()
        for c in cat_cols:
            if c in x_test.columns:
                x_test[c] = x_test[c].cat.codes.astype(float).replace(-1, np.nan)

    return pipe.named_steps["modelo"], x_test, y_test


def main() -> None:
    filas_ranking = []
    filas_resumen = []

    for algoritmo_raw, espec, target in COMBINACIONES:
        nombre_algo = "XGBoost" if algoritmo_raw == "XGBoost" else "HistGradientBoosting"
        print(f"\n=== {nombre_algo} -- {target} ({espec}) ===")

        registro = pd.read_csv(REGISTROS[target])
        modelo, x_test, y_test = entrenar(algoritmo_raw, espec, registro)

        print(f"  Calculando SHAP sobre {x_test.shape[0]} hogares de test, {x_test.shape[1]} variables...")
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(x_test)
        if isinstance(shap_values, list):  # algunos wrappers devuelven [neg, pos]
            shap_values = shap_values[1]
        if shap_values.ndim == 3:  # (n, features, clases) -- clase positiva
            shap_values = shap_values[:, :, 1]

        importancia_media = pd.Series(np.abs(shap_values).mean(axis=0), index=x_test.columns).sort_values(ascending=False)
        ranking = importancia_media.rank(ascending=False)

        for var, val in importancia_media.items():
            filas_ranking.append({
                "algoritmo": nombre_algo, "target": target, "variable": var,
                "shap_abs_medio": round(val, 6), "rank": int(ranking[var]),
            })

        dmsp_vars = [c for c in x_test.columns if c.startswith("dmsp_")]
        peso_total = importancia_media.sum()
        peso_dmsp = importancia_media[dmsp_vars].sum()
        pct_dmsp = 100 * peso_dmsp / peso_total if peso_total > 0 else float("nan")
        mejor_rank_dmsp = int(ranking[dmsp_vars].min())
        mediana_rank_dmsp = float(ranking[dmsp_vars].median())

        print(f"  Peso total |SHAP| de las {len(dmsp_vars)} variables DMSP-OLS: {pct_dmsp:.2f}% del total")
        print(f"  Mejor rank DMSP-OLS: {mejor_rank_dmsp}/{len(importancia_media)}  Mediana rank: {mediana_rank_dmsp:.0f}")
        print(f"  Top 5 variables (|SHAP| medio): {importancia_media.head(5).index.tolist()}")

        filas_resumen.append({
            "algoritmo": nombre_algo, "target": target, "especificacion": espec,
            "n_variables_totales": len(importancia_media), "n_variables_dmsp": len(dmsp_vars),
            "pct_shap_dmsp": round(pct_dmsp, 2),
            "mejor_rank_dmsp": mejor_rank_dmsp, "mediana_rank_dmsp": mediana_rank_dmsp,
        })

    ranking_df = pd.DataFrame(filas_ranking)
    resumen_df = pd.DataFrame(filas_resumen)

    ruta_ranking = mu.RESULTADOS_DIR / "diagnostico_shap_importancia.csv"
    ruta_resumen = mu.RESULTADOS_DIR / "diagnostico_shap_resumen_dmsp.csv"
    ranking_df.to_csv(ruta_ranking, index=False)
    resumen_df.to_csv(ruta_resumen, index=False)

    print("\n\n=== Resumen final ===")
    print(resumen_df.to_string(index=False))
    print(f"\nCSV ranking completo: {ruta_ranking}")
    print(f"CSV resumen: {ruta_resumen}")


if __name__ == "__main__":
    main()
