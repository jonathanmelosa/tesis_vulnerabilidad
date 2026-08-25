"""
heterogeneidad_cerca_umbral.py
=================================

Pregunta del economista de pobreza (ver Seccion 5.3 "Contribucion
marginal de las variables geoespaciales"): el resultado nulo de DMSP-OLS
es un PROMEDIO sobre todos los hogares -- ¿podria estar ayudando mas,
especificamente, para los hogares CERCA del umbral de pobreza, donde el
ingreso/gasto reportado es mas ruidoso y menos discriminante, aunque el
efecto promedio sea cero?

QUE HACE
--------
    1. Toma los hiperparametros y estrategia de balanceo YA elegidos por
       XGBoost para las especificaciones A y AgeoDMSP (leidos de
       registro_modelos.csv -- no se vuelve a tunear, para que la
       comparacion aisle unicamente el efecto de agregar DMSP-OLS, no
       de una busqueda de hiperparametros distinta).
    2. Reentrena XGBoost (semilla 42) sobre el train de cada
       especificacion y predice probabilidades sobre el mismo test
       (2013->2016, 3,191 hogares -- el mismo universo en A y AgeoDMSP,
       XGBoost maneja NaN nativamente asi que ningun hogar se pierde por
       no tener DMSP-OLS real).
    3. Define la distancia al umbral de pobreza con |brecha_lp_ingreso|
       en la ola base (2013, columna de Modelo A) y separa el test en
       terciles: 'cerca del umbral' (tercil inferior de distancia),
       'medio', 'lejos del umbral' (tercil superior).
    4. Para cada tercil, calcula AUC-ROC y precision_top10 con A vs.
       AgeoDMSP, y el cambio (AgeoDMSP - A) -- si DMSP-OLS ayuda
       especificamente cerca del umbral, ese cambio deberia ser mayor ahi
       que en el tercil lejano.

INPUTS
------
    data/processed/benchmark_resultados/registro_modelos.csv
    data/processed/benchmark_train_test/modelo_{A,AgeoDMSP}_{2010_2013,2013_2016}.parquet

OUTPUTS
-------
    data/processed/benchmark_resultados/heterogeneidad_cerca_umbral.csv

CORRER
------
    python heterogeneidad_cerca_umbral.py
"""

import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import modelo_utils as mu

ESPECIFICACIONES = ["A", "AgeoDMSP"]
ALGORITMO_REGISTRO = "XGBoost"
N_TERCILES = 3
ETIQUETAS_TERCILES = ["cerca del umbral", "medio", "lejos del umbral"]


def leer_config_ya_elegida(especificacion: str) -> tuple[str, dict]:
    """Lee balanceo_elegido e hiperparametros ya guardados en
    registro_modelos.csv para XGBoost + `especificacion` -- no se vuelve
    a tunear."""
    registro = pd.read_csv(mu.REGISTRO_CSV)
    fila = registro[(registro["algoritmo"] == ALGORITMO_REGISTRO) & (registro["especificacion"] == especificacion)]
    if fila.empty:
        print(f"ERROR: no hay fila en registro_modelos.csv para {ALGORITMO_REGISTRO}/{especificacion} -- corre primero modelo_xgboost.py", file=sys.stderr)
        sys.exit(1)
    fila = fila.iloc[0]
    hiperparametros = json.loads(fila["hiperparametros"])
    hiperparametros.pop("scale_pos_weight", None)  # se recalcula segun balanceo, no se fija literal
    return fila["balanceo_elegido"], hiperparametros


def construir_pipeline(balanceo: str, hiperparametros: dict, y_train, semilla: int = mu.RANDOM_STATE) -> ImbPipeline:
    if balanceo == "balanced":
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        scale_pos_weight = 1.0
    modelo = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, scale_pos_weight=scale_pos_weight,
        random_state=semilla, n_jobs=-1, eval_metric="auc",
        **{k.replace("modelo__", ""): v for k, v in hiperparametros.items()},
    )
    pasos = []
    if balanceo == "oversampling":
        pasos.append(("muestreo", RandomOverSampler(random_state=semilla)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


def obtener_proba_test(especificacion: str) -> tuple[np.ndarray, pd.Series]:
    """Reentrena XGBoost con la config ya elegida y devuelve probabilidades sobre test."""
    balanceo, hiperparametros = leer_config_ya_elegida(especificacion)
    train, test = mu.cargar_datos(especificacion)
    x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
    x_test, y_test, _ = mu.preparar_arboles_nativos(test)
    x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)

    pipe = construir_pipeline(balanceo, hiperparametros, y_train)
    pipe.fit(x_train, y_train)
    proba_test = pipe.predict_proba(x_test)[:, 1]
    return proba_test, y_test


def cargar_brecha_lp(especificacion_base: str = "A") -> pd.Series:
    """Carga brecha_lp_ingreso de la ola base del TEST (2013) -- columna
    de Modelo A, usada solo para DEFINIR los terciles, no como covariable."""
    test = pd.read_parquet(mu.DATA_DIR / f"modelo_{especificacion_base}_2013_2016.parquet")
    if "brecha_lp_ingreso" not in test.columns:
        print("ERROR: brecha_lp_ingreso no esta en el archivo de Modelo A -- revisar build_benchmark_train_test.py", file=sys.stderr)
        sys.exit(1)
    return test["brecha_lp_ingreso"]


def calcular_metricas_por_tercil(y_test, proba_a, proba_geo, brecha) -> pd.DataFrame:
    distancia = brecha.abs()
    terciles = pd.qcut(distancia, N_TERCILES, labels=ETIQUETAS_TERCILES)

    filas = []
    for etiqueta in ETIQUETAS_TERCILES:
        mask = (terciles == etiqueta).values
        n = int(mask.sum())
        y_sub = y_test[mask]
        met_a = mu.calcular_metricas(y_sub, proba_a[mask])
        met_geo = mu.calcular_metricas(y_sub, proba_geo[mask])
        filas.append({
            "tercil": etiqueta, "n": n, "tasa_entrada": round(float(y_sub.mean()), 4),
            "auc_A": round(met_a["auc_roc"], 4), "auc_AgeoDMSP": round(met_geo["auc_roc"], 4),
            "delta_auc": round(met_geo["auc_roc"] - met_a["auc_roc"], 4),
            "precision_top10_A": round(met_a["precision_top10"], 4), "precision_top10_AgeoDMSP": round(met_geo["precision_top10"], 4),
            "delta_precision_top10": round(met_geo["precision_top10"] - met_a["precision_top10"], 4),
        })
    return pd.DataFrame(filas)


def main() -> None:
    print("=== heterogeneidad_cerca_umbral.py ===\n")
    print("Reentrenando XGBoost con config ya elegida (sin re-tunear)...")
    proba_a, y_test_a = obtener_proba_test("A")
    proba_geo, y_test_geo = obtener_proba_test("AgeoDMSP")
    assert (y_test_a.reset_index(drop=True) == y_test_geo.reset_index(drop=True)).all(), "y_test difiere entre A y AgeoDMSP -- mismos hogares esperados"

    brecha = cargar_brecha_lp("A")
    assert len(brecha) == len(y_test_a), "brecha_lp_ingreso y test no tienen el mismo numero de filas"

    resultado = calcular_metricas_por_tercil(y_test_a.reset_index(drop=True), proba_a, proba_geo, brecha.reset_index(drop=True))

    out_path = mu.RESULTADOS_DIR / "heterogeneidad_cerca_umbral.csv"
    resultado.to_csv(out_path, index=False)

    print("\n" + resultado.to_string(index=False))
    print(f"\nGuardado en: {out_path}")

    if resultado.loc[resultado["tercil"] == "cerca del umbral", "delta_auc"].iloc[0] > resultado.loc[resultado["tercil"] == "lejos del umbral", "delta_auc"].iloc[0]:
        print("\nDMSP-OLS ayuda MAS cerca del umbral que lejos -- consistente con la hipotesis de heterogeneidad.")
    else:
        print("\nDMSP-OLS NO ayuda mas cerca del umbral que lejos -- la hipotesis de heterogeneidad no se sostiene aqui.")


if __name__ == "__main__":
    main()
