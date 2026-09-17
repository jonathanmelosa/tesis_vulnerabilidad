"""
Equivalente de `eda_perfil_completo_ipm.py` (transicion 2010->2013) pero
para la transicion 2013->2016 -- pedido del usuario (2026-09-16) para
poder verificar si el patron de la Seccion 5.1 bajo IPM ("entra en
pobreza" (vulnerable) parece cronico solo en variables de infraestructura
de vivienda) se sostiene en la segunda ola, igual que ya se hizo para
monetaria (Hallazgo 1, Hallazgo 2, DMSP).

Mismo criterio de inclusion que la version 2010->2013: robusto=True Y
efecto>0.10, sin deduplicar por correlacion, sin el chequeo extendido de
13 variables adicionales que si tiene la version monetaria (fuera de
alcance aqui -- este archivo es para el chequeo de consistencia entre
olas, no para reemplazar el anexo principal de IPM).

INPUTS
    outputs/tables/eda_transicion_covariables/ranking_covariables_ipm_2013_2016.csv
    (ya existe -- el ranking de robustez para esta ola ya se habia
    calculado, ver docs/decisions.md)

OUTPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_ipm_2013_2016.csv

COMO CORRER
    cd src/02_build && python eda_perfil_completo_ipm_2013_2016.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import construir_matriz_transicion  # noqa: E402
from eda_perfil_completo_ipm import CATEGORIA  # noqa: E402
from eda_transicion_covariables import (  # noqa: E402
    IPM_PATH,
    TABLES_DIR,
    cargar_covariables_ola_base,
    cargar_dmsp_por_consecutivo,
    cargar_peso_longitudinal_por_consecutivo,
    cargar_tipos_variables,
    construir_tabla_comparativa,
)

UMBRAL_EFECTO = 0.10
DUPLICADOS_EXACTOS = ["ingreso_percapita_hogar", "gasto_percapita_hogar"]


def main() -> None:
    dmsp = cargar_dmsp_por_consecutivo(2013)
    covariables = cargar_covariables_ola_base(dmsp, 2)
    tipos = cargar_tipos_variables()

    ipm = pd.read_parquet(IPM_PATH)
    cargar_peso_longitudinal_por_consecutivo(ipm, ola_fin=3)
    resultado = construir_matriz_transicion(
        ipm, 2, 3, col_pobre="pobre_ipm", peso_col="peso_longitudinal"
    )
    panel = resultado["panel_categorias"]

    ranking = pd.read_csv(TABLES_DIR / "ranking_covariables_ipm_2013_2016.csv")
    filtradas = ranking[(ranking["robusto"]) & (ranking["efecto"] > UMBRAL_EFECTO)]
    seleccion = [v for v in filtradas["variable"] if v not in DUPLICADOS_EXACTOS]

    faltantes = set(seleccion) - set(CATEGORIA)
    if faltantes:
        raise ValueError(f"Variables sin categoria asignada en CATEGORIA: {faltantes}")

    tabla = construir_tabla_comparativa(covariables, tipos, panel, seleccion, ranking)
    tabla["categoria"] = tabla["variable"].map(lambda v: CATEGORIA[v][0])
    tabla["subpanel_unidad"] = tabla["variable"].map(lambda v: CATEGORIA[v][1])

    print(f"Perfil completo IPM 2013->2016: {len(tabla)} variables (umbral robusto + efecto>{UMBRAL_EFECTO})")
    print(tabla["categoria"].value_counts())

    out_path = TABLES_DIR / "perfil_completo_ipm_2013_2016.csv"
    tabla.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
