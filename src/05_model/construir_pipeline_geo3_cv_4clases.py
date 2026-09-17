"""
construir_pipeline_geo3_cv_4clases.py
========================================

Identico a `construir_pipeline_geo3_cv.py` (que NO se modifica), pero
sobre las especificaciones A4/B4 del benchmark de 4 clases -- ver
docs/decisions.md, "2026-09-16: Extension a modelo multiclase". Pega las
TRES fuentes geoespaciales (DMSP-OLS, ALOS PALSAR, Landsat 5 TM) a la
transicion 2010->2013 completa, SIN holdout temporal (ALOS/Landsat solo
tienen datos reales en 2010 -- ver docstring de construir_pipeline_geo3_cv.py
para la justificacion completa de por que este ejercicio usa CV en vez de
holdout).

QUE HACE / INPUTS / OUTPUTS: identico a construir_pipeline_geo3_cv.py,
sustituyendo modelo_{A,B}_2010_2013 por modelo_{A4,B4}_2010_2013 ->
modelo_{A4,B4}geo3_2010_2013.

COMO CORRER
-----------
    cd src/05_model && python construir_pipeline_geo3_cv_4clases.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"
GEOESPACIAL_PATH = PROJECT_ROOT / "data" / "processed" / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"

ESPECIFICACIONES_BASE = ["A4", "B4"]
OLA_OBJETIVO = 2010
PREFIJOS_GEO = ("dmsp_", "alos_", "l5_")

COLS_CAMBIO_ENTRE_OLAS = [
    "dmsp_crecimiento_entre_olas", "dmsp_cambio_nivel_migracion",
    "dmsp_hogar_se_movio", "dmsp_distancia_movimiento_m",
    "l5_hogar_se_movio", "l5_distancia_movimiento_m",
]


def cargar_geoespacial_ola2010() -> pd.DataFrame:
    if not GEOESPACIAL_PATH.exists():
        print(f"ERROR: no se encontro {GEOESPACIAL_PATH}", file=sys.stderr)
        sys.exit(1)
    geo = pd.read_parquet(GEOESPACIAL_PATH)
    geo = geo[(geo["ola"] == OLA_OBJETIVO) & (geo["es_split"] == 0)].copy()

    cols_geo = [c for c in geo.columns if c.startswith(PREFIJOS_GEO) and c not in COLS_CAMBIO_ENTRE_OLAS]
    geo = geo[["consecutivo"] + cols_geo]

    n_dup = geo.duplicated(subset=["consecutivo"]).sum()
    if n_dup > 0:
        print(f"ERROR: {n_dup} filas duplicadas en consecutivo tras excluir es_split -- revisar.", file=sys.stderr)
        sys.exit(1)

    VARIABLE_INSIGNIA = {"dmsp": "dmsp_stable_lights", "alos": "alos_hh_db", "l5": "l5_ndvi"}
    n_por_fuente = {
        fuente: int(geo[col].notna().sum()) if col in geo.columns else None
        for fuente, col in VARIABLE_INSIGNIA.items()
    }
    print(f"Geoespacial ola {OLA_OBJETIVO} cargado: {len(geo):,} filas x {len(cols_geo)} columnas -- cobertura por fuente (variable insignia): {n_por_fuente}")
    return geo


def agregar_geo_a_benchmark(geo: pd.DataFrame) -> None:
    for espec in ESPECIFICACIONES_BASE:
        ruta_in = DATA_DIR / f"modelo_{espec}_2010_2013.parquet"
        if not ruta_in.exists():
            print(f"ERROR: no se encontro {ruta_in}", file=sys.stderr)
            sys.exit(1)
        base = pd.read_parquet(ruta_in)

        antes = len(base)
        resultado = base.merge(geo, on="consecutivo", how="left", validate="one_to_one")
        assert len(resultado) == antes, f"El merge cambio el numero de filas en {ruta_in.name}"

        espec_geo = f"{espec}geo3"
        ruta_out = DATA_DIR / f"modelo_{espec_geo}_2010_2013.parquet"
        resultado.to_parquet(ruta_out, index=False)

        cols_geo = [c for c in geo.columns if c != "consecutivo"]
        n_con_geo = int(resultado[cols_geo].notna().any(axis=1).sum())
        print(f"  modelo_{espec_geo}_2010_2013.parquet: {resultado.shape[0]:,} filas x {resultado.shape[1]:,} columnas -- {n_con_geo:,}/{antes:,} hogares con al menos una variable geoespacial")


def main() -> None:
    print("=== construir_pipeline_geo3_cv_4clases.py ===")
    geo = cargar_geoespacial_ola2010()
    print("\nAgregando las 3 fuentes geoespaciales a la transicion 2010->2013 de 4 clases (sin holdout):")
    agregar_geo_a_benchmark(geo)
    print("\nListo. Especificaciones nuevas disponibles: A4geo3, B4geo3")
    print("(agregadas a modelo_utils_multiclase.ESPECIFICACIONES_4CLASES_CV -- evaluar con evaluar_cv_semillas_multiclase)")


if __name__ == "__main__":
    main()
