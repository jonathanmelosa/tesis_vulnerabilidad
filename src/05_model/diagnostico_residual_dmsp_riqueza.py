"""
Reproduce la correlacion residualizada de dmsp_stable_lights contra el
outcome (entrada a pobreza monetaria) citada en el parrafo "Por que
DMSP-OLS no aporta: redundancia, no ausencia de senal" de la tesis
(paper/main.tex, seccion 5.3) -- ese parrafo no citaba un script/output
fuente (a diferencia de los parrafos vecinos), asi que no habia forma de
verificar los numeros contra un archivo reproducible. Este script llena
ese vacio.

Pregunta: la correlacion cruda de dmsp_stable_lights con el outcome,
¿sigue existiendo una vez que se controla por las dos variables de la
ELCA que, segun la tesis, la hacen redundante (n_servicios_publicos_hogar
y n_bienes_durables_hogar)?

Metodo: correlacion parcial via residuos --
    1. Regresion lineal de dmsp_stable_lights ~ n_servicios_publicos_hogar
       + n_bienes_durables_hogar -> residuos_dmsp.
    2. Regresion lineal de Y ~ n_servicios_publicos_hogar +
       n_bienes_durables_hogar -> residuos_Y.
    3. Correlacion de Pearson entre residuos_dmsp y residuos_Y =
       correlacion parcial de dmsp_stable_lights con Y, controlando por
       esas dos variables.
Se calcula con dos implementaciones independientes (formula cerrada de
minimos cuadrados con numpy, y sklearn.LinearRegression) para verificar
que coinciden, tal como afirma la tesis.

Muestra: la muestra de entrenamiento (transicion 2010->2013, Modelo A +
DMSP-OLS), la misma que usan las tablas principales de la tesis --
data/processed/benchmark_train_test/modelo_AgeoDMSP_2010_2013.parquet.
Se excluyen filas con NaN en cualquiera de las 3 variables involucradas
(dmsp_stable_lights tiene 74 NaN en esta muestra; el resto no tiene).

Output
------
    data/processed/benchmark_resultados/diagnostico_residual_dmsp_riqueza.csv

Como correr
------------
    cd src/05_model && python diagnostico_residual_dmsp_riqueza.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_AgeoDMSP_2010_2013.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_residual_dmsp_riqueza.csv"

VAR_GEO = "dmsp_stable_lights"
VARS_RIQUEZA = ["n_servicios_publicos_hogar", "n_bienes_durables_hogar"]
VAR_OUTCOME = "Y"

VARS_COMPARACION = {
    "ingreso_percapita_hogar_real": "Ingreso per capita real (Modelo A)",
    "estrato_verificado_hogar": "Estrato verificado del hogar",
}


def correlacion_manual(x: np.ndarray, y: np.ndarray) -> float:
    """Correlacion de Pearson calculada a mano (sin scipy), para
    verificacion cruzada independiente de scipy.stats.pearsonr."""
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))


def residuos_manual(objetivo: np.ndarray, predictores: np.ndarray) -> np.ndarray:
    """Residuos de una regresion lineal OLS via ecuacion normal
    (minimos cuadrados manual: beta = (X'X)^-1 X'y), para verificacion
    cruzada independiente de sklearn.LinearRegression."""
    X = np.column_stack([np.ones(len(predictores)), predictores])
    beta = np.linalg.solve(X.T @ X, X.T @ objetivo)
    return objetivo - X @ beta


def residuos_sklearn(objetivo: np.ndarray, predictores: np.ndarray) -> np.ndarray:
    modelo = LinearRegression().fit(predictores, objetivo)
    return objetivo - modelo.predict(predictores)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    # Muestra principal: solo exige no-NaN en las 3 variables centrales del
    # analisis de redundancia (dmsp, las 2 de riqueza, outcome). NO se filtra
    # por las variables de referencia (ingreso/estrato), que tienen su propio
    # patron de faltantes y solo se usan como punto de comparacion informal.
    cols_nucleo = [VAR_GEO, *VARS_RIQUEZA, VAR_OUTCOME]
    n_antes = len(df)
    muestra = df[cols_nucleo].dropna()
    n_dsp = len(muestra)
    print(f"Muestra principal (dmsp+riqueza+outcome): {n_dsp:,} hogares de "
          f"{n_antes:,} (se excluyen {n_antes - n_dsp:,} con NaN en "
          f"{VAR_GEO} -- las otras 3 columnas no tienen NaN en este parquet).")

    y = muestra[VAR_OUTCOME].to_numpy(dtype=float)
    dmsp = muestra[VAR_GEO].to_numpy(dtype=float)
    riqueza = muestra[VARS_RIQUEZA].to_numpy(dtype=float)

    filas = []

    # --- Correlaciones crudas de referencia (para comparar con el texto
    # de la tesis: dmsp r=-0.214, ingreso r=-0.211, estrato r=-0.215) ---
    r_dmsp_y, p_dmsp_y = stats.pearsonr(dmsp, y)
    filas.append({
        "comparacion": "cruda", "variable": VAR_GEO, "contra": VAR_OUTCOME,
        "r": r_dmsp_y, "p": p_dmsp_y, "n": n_dsp,
    })
    for var, etiqueta in VARS_COMPARACION.items():
        ref = df[[var, VAR_OUTCOME]].dropna()
        r, p = stats.pearsonr(ref[var].to_numpy(dtype=float), ref[VAR_OUTCOME].to_numpy(dtype=float))
        filas.append({
            "comparacion": "cruda (referencia, muestra propia)", "variable": var, "contra": VAR_OUTCOME,
            "r": r, "p": p, "n": len(ref),
        })

    # --- Correlacion de dmsp con cada variable de riqueza (para
    # comparar con el texto: servicios r=0.865, bienes r=0.477) ---
    for var in VARS_RIQUEZA:
        r, p = stats.pearsonr(muestra[var].to_numpy(dtype=float), dmsp)
        filas.append({
            "comparacion": "cruda (dmsp vs riqueza)", "variable": var, "contra": VAR_GEO,
            "r": r, "p": p, "n": n_dsp,
        })
        r_riq_y, p_riq_y = stats.pearsonr(muestra[var].to_numpy(dtype=float), y)
        filas.append({
            "comparacion": "cruda (riqueza vs outcome)", "variable": var, "contra": VAR_OUTCOME,
            "r": r_riq_y, "p": p_riq_y, "n": n_dsp,
        })

    # --- Correlacion residualizada (parcial), dos implementaciones ---
    resid_dmsp_manual = residuos_manual(dmsp, riqueza)
    resid_y_manual = residuos_manual(y, riqueza)
    r_parcial_manual = correlacion_manual(resid_dmsp_manual, resid_y_manual)

    resid_dmsp_sk = residuos_sklearn(dmsp, riqueza)
    resid_y_sk = residuos_sklearn(y, riqueza)
    r_parcial_sk, p_parcial_sk = stats.pearsonr(resid_dmsp_sk, resid_y_sk)

    filas.append({
        "comparacion": "parcial (manual, minimos cuadrados)",
        "variable": VAR_GEO, "contra": VAR_OUTCOME,
        "r": r_parcial_manual, "p": np.nan, "n": n_dsp,
    })
    filas.append({
        "comparacion": "parcial (sklearn.LinearRegression)",
        "variable": VAR_GEO, "contra": VAR_OUTCOME,
        "r": r_parcial_sk, "p": p_parcial_sk, "n": n_dsp,
    })

    tabla = pd.DataFrame(filas)
    print("\n" + tabla.to_string(index=False))

    diferencia_implementaciones = abs(r_parcial_manual - r_parcial_sk)
    print(f"\nDiferencia entre las dos implementaciones de la correlacion "
          f"parcial: {diferencia_implementaciones:.2e} "
          f"({'coinciden' if diferencia_implementaciones < 1e-9 else 'NO COINCIDEN -- revisar'})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUT_PATH, index=False)
    print(f"\nGuardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
