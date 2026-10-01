"""
diagnostico_shap_fp_vs_vn.py
=====================================
Version SHAP del analisis de errores de inclusion de
`diagnostico_fp_vs_vn.py` -- pedido explicito del usuario (2026-09-21)
tras notar que comparar el PROMEDIO CRUDO de cada variable entre FP y VN
no puede distinguir cual variable especifica empujo la decision del
modelo de aquellas que solo estan correlacionadas de fondo (el nucleo de
variables esta fuertemente correlacionado entre si -- riqueza, bienes
durables, educacion, estrato se mueven juntos).

Diferencia con `diagnostico_fp_vs_vn.py`: en vez de comparar
promedio(variable_cruda | FP) vs. promedio(variable_cruda | VN), compara
promedio(SHAP_con_signo(variable) | FP) vs. promedio(SHAP_con_signo(variable) | VN).
El SHAP value CON SIGNO de una variable para un hogar mide cuanto
empujo esa variable la prediccion hacia "riesgo" (positivo) o "sin
riesgo" (negativo) PARA ESE HOGAR especifico, ya con todas las
interacciones/no linealidades que el modelo aprendio -- si una variable
tiene SHAP promedio alto en FP pero bajo o negativo en VN, esa es la
variable que el modelo realmente uso para sonar la alarma en los FP,
no solo una que esta de fondo correlacionada con otra.

METODOLOGIA
    Reutiliza `entrenar()`/`clasificar_celda()` (mismo pipeline ganador,
    mismo umbral, sin reentrenar nada) y `calcular_shap()` de
    `diagnostico_shap.py` (mismo calculo ya usado en toda la
    Seccion 5.2.2 -- TreeExplainer sobre HistGradientBoosting/XGBoost,
    Modelo A). Para cada variable, promedio del SHAP CON SIGNO
    (`shap_values`, sin valor absoluto) separado en el grupo FP y el
    grupo VN, y su diferencia -- ordenado por |diferencia| para ver que
    variables distinguen mas la alarma del modelo entre los dos grupos.

    Cinco algoritmos (2026-10-01): para Random Forest y Logistica el
    preprocesador expande cada variable en varias columnas (valor +
    indicador de faltante, dummies one-hot). El SHAP con signo es aditivo,
    asi que se suma por hogar sobre todas las columnas de una misma
    variable original (mismo mapeo `variable_base` que usa
    `diagnostico_shap_nucleo_perfil.py`) antes de promediar por grupo --
    asi las cinco filas de una variable son comparables entre familias.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_train_test/modelo_A_2010_2013.parquet
    data/processed/benchmark_train_test/modelo_A_2013_2016.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_fp_vs_vn_modelo_a.csv
    data/processed/benchmark_resultados/diagnostico_shap_fp_vs_vn_resumen_modelo_a.csv
    (resumen entre algoritmos citado en el Hallazgo 4, ver `resumir()`)
    data/processed/benchmark_resultados/diagnostico_shap_fp_vs_vn_puestos_modelo_a.csv
    (puesto de cada variable por algoritmo, ver `puestos()`)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_fp_vs_vn.py
    cd src/05_model && python -u diagnostico_shap_fp_vs_vn.py --solo-resumen
    (solo rehace el resumen desde el CSV ya guardado, sin reentrenar)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
from diagnostico_fp_vs_vn import ALGORITMOS, ESPEC, REGISTRO, clasificar_celda
from diagnostico_shap import calcular_shap, entrenar
from diagnostico_shap_nucleo_perfil import _construir_mapa_variable_base, _variables_categoricas_originales

RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_shap_fp_vs_vn_modelo_a.csv"
RUTA_RESUMEN = mu.RESULTADOS_DIR / "diagnostico_shap_fp_vs_vn_resumen_modelo_a.csv"
RUTA_PUESTOS = mu.RESULTADOS_DIR / "diagnostico_shap_fp_vs_vn_puestos_modelo_a.csv"
TOP_N_REPORTADO = 15
TOP_K_RESUMEN = 6


def resumir(tabla: pd.DataFrame) -> pd.DataFrame:
    """Resumen entre algoritmos (Hallazgo 4, Seccion 5.1.2): para las
    TOP_K_RESUMEN variables con mayor |SHAP(FP) - SHAP(VN)| de cada
    algoritmo, en cuantos algoritmos aparece cada variable, en cuantos la
    brecha viene sobre todo del lado VN (|SHAP_VN| > |SHAP_FP|: la
    variable baja mucho el riesgo de los VN y poco o nada el de los FP),
    y en cuantos sube el riesgo de los FP (SHAP_FP > 0)."""
    t = tabla.copy()
    t["rank"] = t.groupby("algoritmo")["diferencia_fp_menos_vn"].transform(lambda x: x.abs().rank(ascending=False))
    t = t[t["rank"] <= TOP_K_RESUMEN].copy()
    t["brecha_por_lado_vn"] = t["shap_medio_vn"].abs() > t["shap_medio_fp"].abs()
    t["sube_riesgo_fp"] = t["shap_medio_fp"] > 0
    resumen = (
        t.groupby("variable")
        .agg(n_algoritmos_top=("algoritmo", "size"),
             n_brecha_por_lado_vn=("brecha_por_lado_vn", "sum"),
             n_sube_riesgo_fp=("sube_riesgo_fp", "sum"))
        .sort_values("n_algoritmos_top", ascending=False)
        .reset_index()
    )
    # HistGB casi no usa el ingreso y la Logistica casi no usa la riqueza
    # (Spearman ingreso-riqueza = 0.34, no son sustitutos cercanos): se
    # cuenta en cuantos algoritmos al menos UNA de las dos esta en el top.
    par = ["ingreso_percapita_hogar_real", "riqueza_pca_hogar"]
    n_par = t[t["variable"].isin(par)]["algoritmo"].nunique()
    fila_par = pd.DataFrame([{"variable": "ingreso_percapita_hogar_real O riqueza_pca_hogar",
                              "n_algoritmos_top": n_par,
                              "n_brecha_por_lado_vn": None, "n_sube_riesgo_fp": None}])
    total = pd.DataFrame([{
        "variable": f"TOTAL (top {TOP_K_RESUMEN} x {t['algoritmo'].nunique()} algoritmos)",
        "n_algoritmos_top": len(t),
        "n_brecha_por_lado_vn": int(t["brecha_por_lado_vn"].sum()),
        "n_sube_riesgo_fp": int(t["sube_riesgo_fp"].sum()),
    }])
    return pd.concat([resumen, fila_par, total], ignore_index=True)


def puestos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Puesto de cada variable por |SHAP(FP) - SHAP(VN)| en cada algoritmo,
    solo para las variables que entran al top TOP_K_RESUMEN en alguno
    (ej. educacion maxima: septima en Random Forest)."""
    t = tabla.copy()
    t["puesto"] = t.groupby("algoritmo")["diferencia_fp_menos_vn"].transform(
        lambda x: x.abs().rank(ascending=False, method="first")).astype(int)
    en_top = t.loc[t["puesto"] <= TOP_K_RESUMEN, "variable"].unique()
    return t[t["variable"].isin(en_top)].pivot(index="variable", columns="algoritmo", values="puesto")


def guardar_resumenes(tabla: pd.DataFrame) -> None:
    resumen = resumir(tabla)
    resumen.to_csv(RUTA_RESUMEN, index=False)
    print(f"\n=== Resumen entre algoritmos (top {TOP_K_RESUMEN}) ===")
    print(resumen.to_string(index=False))
    tabla_puestos = puestos(tabla)
    tabla_puestos.to_csv(RUTA_PUESTOS)
    print("\n=== Puesto por algoritmo ===")
    print(tabla_puestos.to_string())
    print(f"\nGuardado: {RUTA_RESUMEN}\nGuardado: {RUTA_PUESTOS}")


def main() -> None:
    if "--solo-resumen" in sys.argv:
        guardar_resumenes(pd.read_csv(RUTA_SALIDA))
        return

    registro = pd.read_csv(REGISTRO)
    variable_base = _construir_mapa_variable_base(_variables_categoricas_originales())

    filas = []
    for algoritmo_raw in ALGORITMOS:
        umbral = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == ESPEC)].iloc[0]["umbral_clasificacion_media"]
        pipe, x_train, x_test, y_test, cat_cols = entrenar(algoritmo_raw, ESPEC, registro)

        proba = pipe.predict_proba(x_test)[:, 1]
        y_pred = (proba >= umbral).astype(int)
        celda = clasificar_celda(np.asarray(y_test), y_pred)

        print(f"=== {algoritmo_raw}: calculando SHAP sobre {x_test.shape[0]} hogares del holdout ===")
        shap_values, nombres, _ = calcular_shap(algoritmo_raw, pipe, x_train, x_test, cat_cols)

        mask_fp = celda == "FP"
        mask_vn = celda == "VN"
        print(f"  FP={mask_fp.sum()}  VN={mask_vn.sum()}")

        shap_por_variable = (
            pd.DataFrame(shap_values, columns=[variable_base(n) for n in nombres])
            .T.groupby(level=0).sum().T
        )
        shap_fp_medio = shap_por_variable[mask_fp].mean(axis=0)
        shap_vn_medio = shap_por_variable[mask_vn].mean(axis=0)

        for var, s_fp, s_vn in zip(shap_por_variable.columns, shap_fp_medio, shap_vn_medio):
            filas.append({
                "algoritmo": algoritmo_raw, "variable": var,
                "shap_medio_fp": round(float(s_fp), 5), "shap_medio_vn": round(float(s_vn), 5),
                "diferencia_fp_menos_vn": round(float(s_fp - s_vn), 5),
            })

    tabla = pd.DataFrame(filas)
    tabla["abs_diferencia"] = tabla["diferencia_fp_menos_vn"].abs()
    tabla = tabla.sort_values(["algoritmo", "abs_diferencia"], ascending=[True, False])

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.drop(columns="abs_diferencia").to_csv(RUTA_SALIDA, index=False)

    for algoritmo_raw in ALGORITMOS:
        print(f"\n=== {algoritmo_raw}: top {TOP_N_REPORTADO} variables por |SHAP(FP) - SHAP(VN)| ===")
        sub = tabla[tabla["algoritmo"] == algoritmo_raw].head(TOP_N_REPORTADO)
        print(sub.drop(columns="abs_diferencia").to_string(index=False))

    print(f"\nGuardado: {RUTA_SALIDA}")
    guardar_resumenes(tabla.drop(columns="abs_diferencia"))


if __name__ == "__main__":
    main()
