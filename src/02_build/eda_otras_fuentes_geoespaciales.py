"""
Perfil por grupo de transicion (2010->2013, pobreza monetaria) de las
otras 2 fuentes geoespaciales exploradas en la tesis ademas de DMSP-OLS:
ALOS PALSAR (radar, `alos_hh_db`) y Landsat 5 TM (optico, `l5_ndvi`) --
pedido del usuario (2026-09-11) para poder comparar visualmente su perfil
con el de iluminacion nocturna, ya mostrado en `eda_nucleo_estable.py` /
`eda_perfil_completo.py`.

A diferencia de DMSP, main.tex ya documenta (Seccion "ALOS PALSAR y
Landsat 5 TM: mismo patron, por un mecanismo distinto") que estas 2
fuentes se evaluaron con un diseno EXPLORATORIO distinto (validacion
cruzada agrupada por hogar dentro de 2010->2013, no holdout a 2013->2016,
porque no tienen dato real en la ola de prueba) -- aqui solo se calcula
el perfil descriptivo (media ponderada por grupo), no se repite el
ejercicio de modelado.

INPUTS
    Reutiliza las funciones y fuentes de datos de eda_transicion_covariables.py
    data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet (ALOS, Landsat, ola 2010)

OUTPUTS
    outputs/tables/eda_transicion_covariables/otras_fuentes_geoespaciales_monetaria_2010_2013.csv

COMO CORRER
    cd src/02_build && python eda_otras_fuentes_geoespaciales.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import _llave_compuesta, cargar_pesos_muestrales, construir_matriz_transicion  # noqa: E402
from eda_transicion_covariables import CATEGORIAS_ORDEN, GEO_PATH, POBREZA_PATH, TABLES_DIR  # noqa: E402

VARIABLES = {
    "alos_hh_db": "ALOS PALSAR (radar, HH dB)",
    "l5_ndvi": "Landsat 5 TM (NDVI, vegetación)",
}


def cargar_geo_por_consecutivo(anio: int, variables: list) -> pd.DataFrame:
    geo = pd.read_parquet(GEO_PATH, columns=["consecutivo", "ola"] + variables)
    geo_anio = geo[geo["ola"] == anio]
    assert not geo_anio["consecutivo"].duplicated().any(), f"consecutivo debe ser unico en ola {anio}"
    return geo_anio.set_index("consecutivo")[variables]


def main() -> None:
    variables = list(VARIABLES)
    geo = cargar_geo_por_consecutivo(2010, variables)

    pobreza = pd.read_parquet(POBREZA_PATH)
    llave_pobreza = _llave_compuesta(pobreza)
    cargar_pesos_muestrales(pobreza, llave_pobreza)
    resultado = construir_matriz_transicion(
        pobreza, 1, 2, col_pobre="pobre_ingreso", peso_col="peso_longitudinal"
    )
    panel = resultado["panel_categorias"].copy()
    panel["categoria"] = pd.Categorical(panel["categoria"], categories=CATEGORIAS_ORDEN, ordered=True)

    df = panel.merge(geo, left_on="consecutivo", right_index=True, how="left")

    filas = []
    for var, etiqueta in VARIABLES.items():
        fila = {"variable": var, "etiqueta": etiqueta}
        for cat in CATEGORIAS_ORDEN:
            sub = df[(df["categoria"] == cat) & df[var].notna() & df["peso_longitudinal"].notna()]
            fila[cat] = np.average(sub[var], weights=sub["peso_longitudinal"]) if len(sub) else float("nan")
            fila[f"n_{cat}"] = len(sub)
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))

    out_path = TABLES_DIR / "otras_fuentes_geoespaciales_monetaria_2010_2013.csv"
    tabla.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
