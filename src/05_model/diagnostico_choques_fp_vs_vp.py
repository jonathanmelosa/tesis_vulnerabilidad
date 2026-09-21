"""
diagnostico_choques_fp_vs_vp.py
=====================================
Pregunta del usuario (2026-09-21), separada a proposito del analisis
principal de errores de inclusion (`diagnostico_fp_vs_vn.py`): ¿los
verdaderos positivos (VP, hogares que el modelo marco como en riesgo y
SI cayeron) vivieron mas choques que los falsos positivos (FP, marcados
en riesgo pero que NO cayeron) durante la ventana que se esta
prediciendo? Es la prueba directa de la idea de que "les toco un choque"
distingue a quien realiza el riesgo de quien no -- una intuicion que se
uso sin evidencia en la discusion de `diagnostico_fp_vs_vn.py` y que este
script SI pone a prueba.

Por que esto NO es lo mismo que comparar choques de la ola BASE (ya
hecho en `diagnostico_fp_vs_vn.py`, sin diferencia real entre FP/VN en
`tuvo_choque_economico_hogar`, 19.8-20.7% vs. 19.3-20.7%): esos choques
son ANTERIORES o contemporaneos a la prediccion, y el modelo SI los usa
como covariable. Los choques de este script son los reportados en la ola
de DESTINO (2016, ola 3) -- lo que le paso al hogar DURANTE la ventana
que se esta prediciendo (2013->2016) -- que el modelo NO puede ver (no
son covariables validas, son posteriores al momento de prediccion), pero
que SI se pueden usar para describir, despues del hecho, si la
realizacion del riesgo coincidio con mas exposicion a choques. Mismo
principio metodologico que `eda_choques_ola3_split_trayectoria.py`
(choques posteriores para explicar quien queda atrapado vs. quien se
recupera), aplicado aqui a la pregunta de FP vs. VP en vez de transitorio
vs. persistente.

METODOLOGIA
    Reutiliza literalmente `entrenar()` y `clasificar_celda()` de
    `diagnostico_fp_vs_vn.py` (mismo pipeline ganador de
    HistGradientBoosting/XGBoost, Modelo A, mismo umbral de
    `registro_modelos_fbeta2_cv10.csv` -- no se reentrena nada nuevo).
    Identifica los hogares FP y VP del holdout de prueba (2013->2016),
    les pega los choques de la ola 3 (2016,
    `choques_hogar_elca_longitudinal.parquet`, la MISMA fuente que
    `eda_choques_ola3_split_trayectoria.py`) por `consecutivo`, y compara
    promedio/tasa entre los dos grupos en las 17 variables de choques.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_train_test/modelo_A_2010_2013.parquet
    data/processed/benchmark_train_test/modelo_A_2013_2016.parquet
    data/processed/choques_hogar_elca_longitudinal.parquet

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_choques_fp_vs_vp_modelo_a.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_choques_fp_vs_vp.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo_utils as mu
from diagnostico_fp_vs_vn import ALGORITMOS, ESPEC, REGISTRO, clasificar_celda
from diagnostico_shap import entrenar

CHOQUES_PATH = mu.PROJECT_ROOT / "data" / "processed" / "choques_hogar_elca_longitudinal.parquet"
OLA_DESTINO = 3  # 2016 -- lo vivido durante la ventana que se predice (2013->2016)
COLUMNAS_EXCLUIR_CHOQUES = ["consecutivo", "llave", "llave_n16", "ola", "zona"]

RUTA_SALIDA = mu.RESULTADOS_DIR / "diagnostico_choques_fp_vs_vp_modelo_a.csv"


def main() -> None:
    registro = pd.read_csv(REGISTRO)
    test_crudo = pd.read_parquet(mu.DATA_DIR / f"modelo_{ESPEC}_2013_2016.parquet")

    choques = pd.read_parquet(CHOQUES_PATH)
    choques = choques[choques["ola"] == OLA_DESTINO].copy()
    cols_choque = [c for c in choques.columns if c not in COLUMNAS_EXCLUIR_CHOQUES]

    filas = []
    for algoritmo_raw in ALGORITMOS:
        umbral = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == ESPEC)].iloc[0]["umbral_clasificacion_media"]
        pipe, x_train, x_test, y_test, cat_cols = entrenar(algoritmo_raw, ESPEC, registro)

        proba = pipe.predict_proba(x_test)[:, 1]
        y_pred = (proba >= umbral).astype(int)
        celda = clasificar_celda(y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test, y_pred)

        sub = test_crudo.loc[x_test.index, ["consecutivo"]].copy()
        sub["celda"] = celda
        sub = sub.merge(choques[["consecutivo"] + cols_choque], on="consecutivo", how="left")

        fp = sub[sub["celda"] == "FP"]
        vp = sub[sub["celda"] == "VP"]
        n_fp_con_dato = fp[cols_choque[0]].notna().sum()
        n_vp_con_dato = vp[cols_choque[0]].notna().sum()
        print(f"{algoritmo_raw}: FP={len(fp)} (con dato ola 3: {n_fp_con_dato})  VP={len(vp)} (con dato ola 3: {n_vp_con_dato})")

        for var in cols_choque:
            media_fp = fp[var].mean()
            media_vp = vp[var].mean()
            filas.append({
                "algoritmo": algoritmo_raw, "variable": var,
                "media_fp": round(float(media_fp), 4), "media_vp": round(float(media_vp), 4),
                "diferencia_vp_menos_fp": round(float(media_vp - media_fp), 4),
                "n_fp_con_dato": int(n_fp_con_dato), "n_vp_con_dato": int(n_vp_con_dato),
            })

    tabla = pd.DataFrame(filas)
    tabla["abs_diferencia"] = tabla["diferencia_vp_menos_fp"].abs()
    tabla = tabla.sort_values(["algoritmo", "abs_diferencia"], ascending=[True, False]).drop(columns="abs_diferencia")

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_SALIDA, index=False)

    print("\n=== Choques ola 3 (2016), VP vs. FP, mayor |diferencia| primero ===")
    print(tabla.to_string(index=False))
    print(f"\nGuardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
