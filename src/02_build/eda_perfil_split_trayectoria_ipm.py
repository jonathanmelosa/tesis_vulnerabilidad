"""
eda_perfil_split_trayectoria_ipm.py
=======================================
Equivalente de `eda_perfil_split_trayectoria.py` (que divide "Entra en
pobreza" en transitorio/persistente para pobreza monetaria) pero bajo
IPM -- pedido del usuario (2026-09-16), simetrico al Hallazgo 3 de la
Seccion 5.1 (hecho solo para monetaria hasta ahora).

De los hogares "Entra en pobreza" bajo IPM (transicion 2010->2013,
perfil_completo_ipm_2010_2013.csv), los que tienen trayectoria de 3 olas
ya calculada en `trayectorias_3olas_ipm_hogares.csv` (445 transitorio,
214 persistente, ver ese archivo) se separan en dos subgrupos ANTES de
calcular los promedios ponderados -- misma metodologia exacta que la
version monetaria, solo cambia la fuente de pobreza (IPM en vez de
ingreso) y el archivo de trayectorias.

Selecccion de variables: reusa las 35 variables YA seleccionadas en
`perfil_completo_ipm_2010_2013.csv` (robusto + efecto>0.10 sobre
2010->2013) -- no se vuelve a filtrar, mismo criterio que la version
monetaria (comparar las MISMAS variables ya publicadas, no buscar
nuevas).

INPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_ipm_2010_2013.csv
    outputs/tables/eda_transicion_covariables/trayectorias_3olas_ipm_hogares.csv

OUTPUTS
    outputs/tables/eda_transicion_covariables/perfil_split_transitorio_persistente_ipm.csv

COMO CORRER
    cd src/02_build && python eda_perfil_split_trayectoria_ipm.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import construir_matriz_transicion  # noqa: E402
from eda_transicion_covariables import (  # noqa: E402
    IPM_PATH,
    TABLES_DIR,
    cargar_covariables_ola_base,
    cargar_dmsp_por_consecutivo,
    cargar_peso_longitudinal_por_consecutivo,
    cargar_tipos_variables,
    construir_tabla_comparativa,
)
import eda_transicion_covariables as etc  # noqa: E402

MAPA_TRAYECTORIA = {
    "Entra y sale (transitorio)": "Entra (transitorio)",
    "Entra y se queda (persistente)": "Entra (persistente)",
}
CATEGORIAS_ORDEN_SPLIT = [
    "Siempre pobre", "Sale de la pobreza",
    "Entra (transitorio)", "Entra (persistente)", "Nunca pobre",
]


def construir_panel_split() -> pd.DataFrame:
    ipm = pd.read_parquet(IPM_PATH)
    cargar_peso_longitudinal_por_consecutivo(ipm, ola_fin=2)
    resultado = construir_matriz_transicion(ipm, 1, 2, col_pobre="pobre_ipm", peso_col="peso_longitudinal")
    panel = resultado["panel_categorias"].copy()

    tray = pd.read_csv(TABLES_DIR / "trayectorias_3olas_ipm_hogares.csv")
    tray_map = tray.set_index("consecutivo")["trayectoria"]

    mask_entra = panel["categoria"] == "Entra en pobreza"
    sub_tray = panel.loc[mask_entra, "consecutivo"].map(tray_map).map(MAPA_TRAYECTORIA)

    nueva_cat = panel["categoria"].astype(str).copy()
    nueva_cat[mask_entra] = sub_tray.fillna("Entra (sin dato 2016)")
    panel["categoria"] = nueva_cat
    return panel


def main() -> None:
    panel = construir_panel_split()
    n_transitorio = int((panel["categoria"] == "Entra (transitorio)").sum())
    n_persistente = int((panel["categoria"] == "Entra (persistente)").sum())
    print(f"Entra (transitorio): {n_transitorio} hogares | Entra (persistente): {n_persistente} hogares")

    etc.CATEGORIAS_ORDEN = CATEGORIAS_ORDEN_SPLIT

    dmsp = cargar_dmsp_por_consecutivo(2010)
    covariables = cargar_covariables_ola_base(dmsp, 1)
    tipos = cargar_tipos_variables()

    perfil_ipm = pd.read_csv(TABLES_DIR / "perfil_completo_ipm_2010_2013.csv")
    seleccion = perfil_ipm["variable"].tolist()
    ranking_ref = pd.read_csv(TABLES_DIR / "ranking_covariables_ipm_2010_2013.csv")

    tabla = construir_tabla_comparativa(covariables, tipos, panel, seleccion, ranking_ref)

    out_path = TABLES_DIR / "perfil_split_transitorio_persistente_ipm.csv"
    tabla.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(tabla)} variables)")


if __name__ == "__main__":
    main()
