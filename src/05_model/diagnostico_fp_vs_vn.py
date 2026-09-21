"""
diagnostico_fp_vs_vn.py
=====================================
Caracteriza a los errores de inclusion (falsos positivos, FP) del
benchmark comparandolos contra los verdaderos negativos (VN) -- pedido
explicito del usuario (2026-09-21), extension del Hallazgo de la
Seccion~5.1.2 (analisis de errores de inclusion y exclusion) que hasta
ahora solo reportaba CUANTOS son, no A QUIENES se esta incluyendo por
error.

Por que FP vs. VN y no FP vs. VP (ver docstring del hallazgo en
main.tex, Seccion~5.1.2): tanto FP como VN son hogares que en la
realidad NO cayeron en pobreza -- la unica diferencia entre ellos es que
el modelo SI marco a los FP como en riesgo. Comparando estos dos grupos
se aisla que caracteristicas "disparan" la alarma del modelo, sin la
variable adicional de "le toco un choque distinto" que contamina la
comparacion contra VP (que si cayeron).

METODOLOGIA
    Reconstruye el pipeline ganador de HistGradientBoosting y XGBoost
    (Modelo A, monetaria -- los dos ya nombrados en la Seccion 5.1.2 por
    su contraste de tasa de verdaderos negativos, 40% vs. 26%) desde
    `registro_modelos_fbeta2_cv10.csv`, UN solo fit sobre el conjunto de
    entrenamiento (sin repetir RandomizedSearchCV, mismo patron que
    `diagnostico_shap_ab.py`). Predice sobre el holdout de prueba
    (2013->2016, semilla 42) y aplica el umbral de clasificacion YA
    elegido por CV (`umbral_clasificacion_media`, la misma Tabla 5).
    Compara FP vs. VN en las 22 variables del nucleo de importancia
    multivariada ya validado (`diagnostico_shap_nucleo_perfil.csv`,
    Seccion~5.2.2) -- variables numericas: promedio simple (el conjunto
    de prueba del benchmark no usa ponderador muestral, a diferencia del
    perfil univariado de la Seccion 5.1); variables categoricas
    (`estado_civil_jefe`, `material_pisos_hogar`): categoria mas
    frecuente y su participacion en cada grupo.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_resultados/diagnostico_shap_nucleo_perfil.csv
    data/processed/benchmark_train_test/modelo_A_2010_2013.parquet
    data/processed/benchmark_train_test/modelo_A_2013_2016.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_fp_vs_vn_modelo_a.csv
    (una fila por variable numerica x algoritmo, con el promedio en FP,
    en VN, y la diferencia)
    data/processed/benchmark_resultados/diagnostico_fp_vs_vn_categoricas_modelo_a.csv
    (una fila por variable categorica x algoritmo x categoria)

COMO CORRER

    cd src/05_model && python -u diagnostico_fp_vs_vn.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
from diagnostico_shap import entrenar

REGISTRO = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv"
NUCLEO_PATH = mu.RESULTADOS_DIR / "diagnostico_shap_nucleo_perfil.csv"
ESPEC = "A"
ALGORITMOS = ["HistGradientBoosting (sklearn)", "XGBoost"]

RUTA_SALIDA_NUM = mu.RESULTADOS_DIR / "diagnostico_fp_vs_vn_modelo_a.csv"
RUTA_SALIDA_CAT = mu.RESULTADOS_DIR / "diagnostico_fp_vs_vn_categoricas_modelo_a.csv"


def clasificar_celda(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    celda = np.full(len(y_real), "", dtype=object)
    celda[(y_pred == 1) & (y_real == 0)] = "FP"
    celda[(y_pred == 0) & (y_real == 0)] = "VN"
    celda[(y_pred == 1) & (y_real == 1)] = "VP"
    celda[(y_pred == 0) & (y_real == 1)] = "FN"
    return celda


def main() -> None:
    registro = pd.read_csv(REGISTRO)
    nucleo = pd.read_csv(NUCLEO_PATH)
    test_crudo = pd.read_parquet(mu.DATA_DIR / f"modelo_{ESPEC}_2013_2016.parquet")

    variables_categoricas = [v for v in nucleo["variable"] if test_crudo[v].dtype == object]
    variables_numericas = [v for v in nucleo["variable"] if v not in variables_categoricas]
    print(f"Variables numéricas del núcleo: {len(variables_numericas)} | categóricas: {len(variables_categoricas)}")

    filas_num, filas_cat = [], []
    for algoritmo_raw in ALGORITMOS:
        umbral = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == ESPEC)].iloc[0]["umbral_clasificacion_media"]
        pipe, x_train, x_test, y_test, cat_cols = entrenar(algoritmo_raw, ESPEC, registro)

        proba = pipe.predict_proba(x_test)[:, 1]
        y_pred = (proba >= umbral).astype(int)
        celda = clasificar_celda(np.asarray(y_test), y_pred)

        # x_test conserva el mismo indice que test_crudo (preparar_arboles_nativos/
        # preparar_xy_crudo no reindexan) -- se usa para recuperar las 22
        # variables del nucleo EN CRUDO, no en la version imputada/transformada
        # que ve el modelo.
        sub = test_crudo.loc[x_test.index].copy()
        sub["celda"] = celda

        n_fp, n_vn = (celda == "FP").sum(), (celda == "VN").sum()
        print(f"{algoritmo_raw}: FP={n_fp}  VN={n_vn}  (umbral={umbral})")

        fp = sub[sub["celda"] == "FP"]
        vn = sub[sub["celda"] == "VN"]

        for var in variables_numericas:
            media_fp = fp[var].mean()
            media_vn = vn[var].mean()
            filas_num.append({
                "algoritmo": algoritmo_raw, "variable": var,
                "media_fp": round(float(media_fp), 4), "media_vn": round(float(media_vn), 4),
                "diferencia_fp_menos_vn": round(float(media_fp - media_vn), 4),
                "n_fp": int(n_fp), "n_vn": int(n_vn),
            })

        for var in variables_categoricas:
            for grupo_nombre, grupo_df in [("FP", fp), ("VN", vn)]:
                dist = grupo_df[var].value_counts(normalize=True, dropna=True)
                for categoria, participacion in dist.items():
                    filas_cat.append({
                        "algoritmo": algoritmo_raw, "variable": var, "grupo": grupo_nombre,
                        "categoria": categoria, "participacion": round(float(participacion), 4),
                    })

    tabla_num = pd.DataFrame(filas_num)
    tabla_num["abs_diferencia"] = tabla_num["diferencia_fp_menos_vn"].abs()
    tabla_num = tabla_num.sort_values(["algoritmo", "abs_diferencia"], ascending=[True, False]).drop(columns="abs_diferencia")
    tabla_cat = pd.DataFrame(filas_cat)

    tabla_num.to_csv(RUTA_SALIDA_NUM, index=False)
    tabla_cat.to_csv(RUTA_SALIDA_CAT, index=False)

    print("\n=== Variables numéricas, mayor |diferencia FP - VN| primero ===")
    print(tabla_num.to_string(index=False))
    print(f"\nGuardado: {RUTA_SALIDA_NUM}")
    print(f"Guardado: {RUTA_SALIDA_CAT}")


if __name__ == "__main__":
    main()
