"""
Resumen de robustez y estabilidad temporal por MODULO tematico (Personas,
Hogar/vivienda/activos, Comunidades, Choques, Monetario/pobreza, Ninos,
Geoespacial) -- pedido del usuario (2026-09-11) tras encontrar que
Comunidades e IPM tenian un patron de inestabilidad marcado en la cola de
bajo efecto (ver `comparacion_periodos_*.csv`), para revisar si otros
modulos tienen el mismo problema.

Dos metricas por modulo, separadas (no deben confundirse):
  1. % de variables candidatas que pasan el filtro de robustez del ranking
     de 4 grupos (umbral + bootstrap CI) -- ver `ranking_covariables_*.csv`.
     Mide "cuanta senal tiene este modulo, en una sola transicion".
  2. % de variables SELECCIONADAS (en al menos 1 de las 2 transiciones,
     union) que aparecen en AMBAS -- ver `comparacion_periodos_*.csv`,
     EXCLUYENDO VARIABLES_OBLIGATORIAS (aparecen en ambas por
     construccion, no por hallazgo empirico). Mide "si el modulo tiene
     senal, esa senal se repite en el tiempo".

INPUTS
    outputs/tables/eda_transicion_covariables/ranking_covariables_{monetaria,ipm}_{2010_2013,2013_2016}.csv
    outputs/tables/eda_transicion_covariables/comparacion_periodos_{monetaria,ipm}.csv
    outputs/tables/eda_variables_modelo/01_inventario_variables.csv (modulo por variable)

OUTPUTS
    outputs/tables/eda_transicion_covariables/estabilidad_por_modulo_{monetaria,ipm}.csv

COMO CORRER
    cd src/02_build && python eda_estabilidad_por_modulo.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_transicion_covariables import TABLES_DIR, VARIABLES_OBLIGATORIAS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTARIO_PATH = PROJECT_ROOT / "outputs" / "tables" / "eda_variables_modelo" / "01_inventario_variables.csv"


def main() -> None:
    inv = pd.read_csv(INVENTARIO_PATH)
    tipos_modulo = dict(zip(inv["variable"], inv["modulo"]))
    tipos_modulo["dmsp_stable_lights"] = "Geoespacial"
    tipos_modulo.setdefault("zona", "Vivienda")

    for nombre in ("monetaria", "ipm"):
        # --- Metrica 1: robustez dentro de cada transicion (por separado) ---
        filas_robustez = []
        for sufijo in ("2010_2013", "2013_2016"):
            ranking = pd.read_csv(TABLES_DIR / f"ranking_covariables_{nombre}_{sufijo}.csv")
            ranking["modulo"] = ranking["variable"].map(tipos_modulo)
            resumen = ranking.groupby("modulo").agg(
                n_candidatas=("variable", "size"), n_robustas=("robusto", "sum")
            )
            resumen["pct_robustas"] = (resumen["n_robustas"] / resumen["n_candidatas"] * 100).round(1)
            resumen["transicion"] = sufijo
            filas_robustez.append(resumen.reset_index())
        robustez = pd.concat(filas_robustez, ignore_index=True)

        # --- Metrica 2: estabilidad entre periodos (variables NO forzadas) ---
        comp = pd.read_csv(TABLES_DIR / f"comparacion_periodos_{nombre}.csv")
        comp = comp[~comp["variable"].isin(VARIABLES_OBLIGATORIAS)].copy()
        comp["modulo"] = comp["variable"].map(tipos_modulo)
        estabilidad = comp.groupby("modulo").agg(
            n_seleccionadas_union=("variable", "size"), n_en_ambas=("en_ambas", "sum")
        )
        estabilidad["pct_estable"] = (estabilidad["n_en_ambas"] / estabilidad["n_seleccionadas_union"] * 100).round(1)

        print(f"\n=== {nombre.upper()} -- % candidatas robustas por modulo y transicion ===")
        pivote = robustez.pivot(index="modulo", columns="transicion", values="pct_robustas")
        print(pivote.to_string())

        print(f"\n=== {nombre.upper()} -- estabilidad entre periodos por modulo (variables no forzadas) ===")
        print(estabilidad.to_string())

        pivote.join(estabilidad, how="outer").to_csv(TABLES_DIR / f"estabilidad_por_modulo_{nombre}.csv")

    print(f"\nGuardado en: {TABLES_DIR}")


if __name__ == "__main__":
    main()
