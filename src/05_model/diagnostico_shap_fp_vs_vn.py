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

INPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_train_test/modelo_A_2010_2013.parquet
    data/processed/benchmark_train_test/modelo_A_2013_2016.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_fp_vs_vn_modelo_a.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_fp_vs_vn.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
from diagnostico_fp_vs_vn import ALGORITMOS, ESPEC, REGISTRO, clasificar_celda
from diagnostico_shap import calcular_shap, entrenar

RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_shap_fp_vs_vn_modelo_a.csv"
TOP_N_REPORTADO = 15


def main() -> None:
    registro = pd.read_csv(REGISTRO)

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

        shap_fp_medio = shap_values[mask_fp].mean(axis=0)
        shap_vn_medio = shap_values[mask_vn].mean(axis=0)

        for var, s_fp, s_vn in zip(nombres, shap_fp_medio, shap_vn_medio):
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


if __name__ == "__main__":
    main()
