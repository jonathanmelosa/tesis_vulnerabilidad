"""
Equivalente de `eda_perfil_completo.py` para pobreza MULTIDIMENSIONAL
(IPM), transicion 2010->2013 -- pedido del usuario (2026-09-11) para
anexar el perfil completo de IPM en la tesis (mientras el cuerpo
principal de la Seccion 5.2 se centra en pobreza monetaria) y poder
comparar cuales variables son comunes a ambas definiciones de pobreza.

Mismo criterio de inclusion que la version monetaria: robusto=True Y
efecto>0.10 (38 variables), sin deduplicar por correlacion. Reutiliza el
diccionario CATEGORIA de `eda_perfil_completo.py` para las 27 variables
que resultan comunes a ambas definiciones (verificado por interseccion de
conjuntos, ver docs/decisions.md) y agrega categoria solo para las
variables exclusivas de IPM.

INPUTS
    Reutiliza las funciones y fuentes de datos de eda_transicion_covariables.py

OUTPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_ipm_2010_2013.csv

COMO CORRER
    cd src/02_build && python eda_perfil_completo_ipm.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import construir_matriz_transicion  # noqa: E402
from eda_perfil_completo import CATEGORIA as CATEGORIA_MONETARIA  # noqa: E402
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

# Variables exclusivas de IPM (no estaban en la lista monetaria) -- las
# 27 comunes ya tienen categoria asignada en CATEGORIA_MONETARIA.
CATEGORIA_IPM_EXTRA = {
    "lp": ("Ingreso y gasto", "Miles $ por mes (línea de pobreza)"),
    "li": ("Ingreso y gasto", "Miles $ por mes (línea de indigencia)"),
    "pct_adultos_alfabetizados": ("Educación, empleo y seguridad social", "% del grupo"),
    "pct_ninos_cuidado_terceros_hogar": ("Composición del hogar", "% del grupo"),
    "pobre_extremo_ingreso": ("Ingreso y gasto", "% del grupo (pobre extremo por ingreso, 2010)"),
    "pobre_ingreso": ("Ingreso y gasto", "% del grupo (pobre por ingreso, 2010)"),
    "pobre_gasto": ("Ingreso y gasto", "% del grupo (pobre por gasto, 2010)"),
    "pobre_extremo_gasto": ("Ingreso y gasto", "% del grupo (pobre extremo por gasto, 2010)"),
    "sexo_jefe": ("Composición del hogar", "% del grupo"),
}
CATEGORIA = {**CATEGORIA_MONETARIA, **CATEGORIA_IPM_EXTRA}
DUPLICADOS_EXACTOS = ["ingreso_percapita_hogar", "gasto_percapita_hogar"]


def main() -> None:
    dmsp = cargar_dmsp_por_consecutivo(2010)
    covariables = cargar_covariables_ola_base(dmsp, 1)
    tipos = cargar_tipos_variables()

    ipm = pd.read_parquet(IPM_PATH)
    cargar_peso_longitudinal_por_consecutivo(ipm, ola_fin=2)
    resultado = construir_matriz_transicion(
        ipm, 1, 2, col_pobre="pobre_ipm", peso_col="peso_longitudinal"
    )
    panel = resultado["panel_categorias"]

    ranking = pd.read_csv(TABLES_DIR / "ranking_covariables_ipm_2010_2013.csv")
    filtradas = ranking[(ranking["robusto"]) & (ranking["efecto"] > UMBRAL_EFECTO)]
    seleccion = [v for v in filtradas["variable"] if v not in DUPLICADOS_EXACTOS]

    faltantes = set(seleccion) - set(CATEGORIA)
    if faltantes:
        raise ValueError(f"Variables sin categoria asignada en CATEGORIA: {faltantes}")

    tabla = construir_tabla_comparativa(covariables, tipos, panel, seleccion, ranking)
    tabla["categoria"] = tabla["variable"].map(lambda v: CATEGORIA[v][0])
    tabla["subpanel_unidad"] = tabla["variable"].map(lambda v: CATEGORIA[v][1])

    print(f"Perfil completo IPM: {len(tabla)} variables (umbral robusto + efecto>{UMBRAL_EFECTO})")
    print(tabla["categoria"].value_counts())

    out_path = TABLES_DIR / "perfil_completo_ipm_2010_2013.csv"
    tabla.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
