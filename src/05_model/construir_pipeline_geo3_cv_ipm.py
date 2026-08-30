"""
construir_pipeline_geo3_cv_ipm.py
====================================

Version IPM de `construir_pipeline_geo3_cv.py` (Pipeline 2 exploratorio):
agrega las TRES fuentes geoespaciales (DMSP-OLS, ALOS PALSAR, Landsat 5
TM) a las covariables de la transicion 2010->2013 con target IPM en vez
de pobreza monetaria -- pedido por el usuario (2026-08-30), espejo exacto
del ejercicio ya hecho para pobreza monetaria.

MISMA RAZON para el esquema de validacion distinto (ver docstring
original): ALOS PALSAR y Landsat 5 TM solo tienen datos reales en 2010
(0% en 2013) -- si se usaran en el benchmark principal (holdout
2010->2013/2013->2016), esas columnas quedarian 100% vacias en el
conjunto de prueba. Por eso este ejercicio se evalua con validacion
cruzada agrupada por hogar DENTRO de la transicion 2010->2013, no con el
holdout temporal. Los resultados de este ejercicio NO son comparables
cifra a cifra contra Aipm/Bipm/AipmgeoDMSP/BipmgeoDMSP (esquemas de
validacion distintos) -- usar `modelo_utils.evaluar_cv_semillas`, NO
`evaluar_multiples_semillas`.

QUE HACE

    1. Carga variables_geoespaciales_unificadas.parquet, filtra a ola ==
       2010, se queda con dmsp_*/alos_*/l5_* (excluyendo las columnas de
       "cambio entre olas", 100% NaN en 2010 por construccion) +
       consecutivo. Excluye hogares divididos (es_split == 1).
    2. Para cada especificacion (Aipm, Bipm): carga
       modelo_{espec}_2010_2013.parquet (construido por
       build_benchmark_train_test_ipm.py -- la transicion IPM completa,
       usada aqui como muestra unica del ejercicio, no como train de un
       holdout), pega las columnas geoespaciales por consecutivo, LEFT
       JOIN.
    3. Exporta modelo_{espec}geo3_2010_2013.parquet -- especificacion
       nueva ("Aipmgeo3"/"Bipmgeo3"). Los archivos modelo_{Aipm,Bipm}_
       2010_2013.parquet NO se tocan.

INPUTS

    data/processed/benchmark_train_test/modelo_{Aipm,Bipm}_2010_2013.parquet
    data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet

OUTPUTS

    data/processed/benchmark_train_test/modelo_{Aipm,Bipm}geo3_2010_2013.parquet

COMO CORRER

    cd src/05_model && python construir_pipeline_geo3_cv_ipm.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"
GEOESPACIAL_PATH = PROJECT_ROOT / "data" / "processed" / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"

ESPECIFICACIONES_BASE = ["Aipm", "Bipm"]
OLA_OBJETIVO = 2010
PREFIJOS_GEO = ("dmsp_", "alos_", "l5_")

# Mismas exclusiones que construir_pipeline_geo3_cv.py -- columnas de
# "cambio entre olas" 100% NaN en 2010 por construccion (no hay ola
# anterior con la que comparar).
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
            print(f"ERROR: no se encontro {ruta_in} -- correr primero build_benchmark_train_test_ipm.py", file=sys.stderr)
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
    print("=== construir_pipeline_geo3_cv_ipm.py ===")
    geo = cargar_geoespacial_ola2010()
    print("\nAgregando las 3 fuentes geoespaciales a la transicion IPM 2010->2013 (sin holdout):")
    agregar_geo_a_benchmark(geo)
    print("\nListo. Especificaciones nuevas disponibles: Aipmgeo3, Bipmgeo3")
    print("(evaluar con modelo_utils.evaluar_cv_semillas, NO con evaluar_multiples_semillas -- sin holdout temporal)")


if __name__ == "__main__":
    main()
