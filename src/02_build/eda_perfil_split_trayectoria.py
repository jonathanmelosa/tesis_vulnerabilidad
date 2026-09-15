"""
eda_perfil_split_trayectoria.py
=====================================
Divide el grupo "Entra en pobreza" (transicion monetaria 2010->2013) en dos
subgrupos segun su trayectoria en la tercera ola (2016) -- pedido del
usuario (2026-09-15), a partir de un hallazgo del panel de revision de la
Seccion 5.1: "el hogar vulnerable" no es homogeneo, el 58% vuelve a salir
de la pobreza para 2016 ("transitorio") y el 42% se queda pobre
("persistente") (`trayectorias_3olas_monetaria_hogares.csv`, ya calculado
por `eda_trayectorias_3olas.py`).

De los 723 hogares "Entra en pobreza" (panel 4-grupos, 2010->2013), 610
tienen dato de la ola 3 (113 se pierden por atricion, igual que en el
resto del panel): 355 transitorio, 255 persistente.

Pregunta que responde: el hallazgo central de la Seccion 5.1 ("el hogar
vulnerable se parece mas a quien sale de la pobreza que a quien nunca fue
pobre") es un patron agregado que oculta este mix -- necesitamos saber si
el patron sobrevive por separado en cada subgrupo, o si es un artefacto de
promediar dos poblaciones distintas.

METODOLOGIA
    Reusa exactamente `construir_tabla_comparativa` (misma funcion que
    genera la tabla oficial de 53 variables) sobre el mismo panel de
    2010->2013, pero con la categoria "Entra en pobreza" separada en dos
    ("Entra (transitorio)" / "Entra (persistente)") ANTES de calcular los
    promedios ponderados -- no se recalcula el ranking ni el filtro de
    variables, solo se desagrega el grupo ya seleccionado.

RESULTADO (ver docstring de main() para el detalle numerico)
    El patron agregado SI sobrevive en ambos subgrupos (39/53 variables
    para transitorio, 44/53 para persistente quedan mas cerca de "sale"
    que de "nunca cae" -- el subgrupo persistente incluso mas), pero hay
    un gradiente real hacia el extremo cronico: persistente esta mas cerca
    de "siempre pobre" que de "sale" en 11/53 variables (transitorio en
    solo 2/53). El hallazgo mas fuerte esta en variables puntuales que el
    promedio agregado diluye por completo, sobre todo deuda informal
    (transitorio 17.3%, mas bajo que "sale" 22.7%; persistente 41.9%, mas
    alto que TODOS los demas grupos, incluido "siempre pobre" 31.6%).

INPUTS
    Reutiliza las funciones y fuentes de datos de eda_transicion_covariables.py
    outputs/tables/eda_transicion_covariables/trayectorias_3olas_monetaria_hogares.csv

OUTPUTS
    outputs/tables/eda_transicion_covariables/perfil_split_transitorio_persistente_monetaria.csv

COMO CORRER
    cd src/02_build && python eda_perfil_split_trayectoria.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import _llave_compuesta, cargar_pesos_muestrales, construir_matriz_transicion  # noqa: E402
from eda_transicion_covariables import (  # noqa: E402
    POBREZA_PATH,
    TABLES_DIR,
    cargar_covariables_ola_base,
    cargar_dmsp_por_consecutivo,
    cargar_tipos_variables,
    construir_tabla_comparativa,
)
import eda_transicion_covariables as etc  # noqa: E402
from eda_perfil_completo import calcular_seleccion_2010_2013  # noqa: E402

MAPA_TRAYECTORIA = {
    "Entra y sale (transitorio)": "Entra (transitorio)",
    "Entra y se queda (persistente)": "Entra (persistente)",
}
CATEGORIAS_ORDEN_SPLIT = [
    "Siempre pobre", "Sale de la pobreza",
    "Entra (transitorio)", "Entra (persistente)", "Nunca pobre",
]


def construir_panel_split() -> pd.DataFrame:
    pobreza = pd.read_parquet(POBREZA_PATH)
    llave = _llave_compuesta(pobreza)
    cargar_pesos_muestrales(pobreza, llave)
    resultado = construir_matriz_transicion(pobreza, 1, 2, col_pobre="pobre_ingreso", peso_col="peso_longitudinal")
    panel = resultado["panel_categorias"].copy()

    tray = pd.read_csv(TABLES_DIR / "trayectorias_3olas_monetaria_hogares.csv")
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
    seleccion = calcular_seleccion_2010_2013()
    ranking_ref = pd.read_csv(TABLES_DIR / "ranking_covariables_monetaria_2010_2013.csv")

    tabla = construir_tabla_comparativa(covariables, tipos, panel, seleccion, ranking_ref)

    out_path = TABLES_DIR / "perfil_split_transitorio_persistente_monetaria.csv"
    tabla.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(tabla)} variables)")


if __name__ == "__main__":
    main()
