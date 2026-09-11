"""
Perfil completo (sin deduplicar por correlacion) de las variables que
distinguen a los 4 grupos de transicion de pobreza monetaria 2010->2013
-- pedido del usuario (2026-09-11): mostrar TODAS las variables que
discriminan bien, incluidas las redundantes entre si (a diferencia de
`eda_nucleo_estable.py`, que colapsa clusters de correlacion y aplica
cupo por modulo).

Criterio de inclusion (decision del usuario, mismo dia): `robusto=True`
(el mismo umbral ya usado en todo el pipeline: eta^2/V minimo + bootstrap
CI) Y efecto > 0.10 -- un piso uniforme entre categorias, no un "top N"
arbitrario por modulo. Se excluyen 2 duplicados exactos
(`ingreso_percapita_hogar` == `ingreso_percapita_hogar_real`,
`gasto_percapita_hogar` == `gasto_percapita_hogar_real`, mismo valor en
las 4 categorias -- se conserva la version "_real", deflactada, la misma
que usan los modelos de ML).

Reutiliza `construir_tabla_comparativa` (misma funcion que genera la
tabla oficial de la tesis) para los valores reales por grupo -- no
recalcula nada nuevo, solo aplica un `seleccion` mas amplio.

Extension 2026-09-11 (pedido del usuario): la MISMA lista de 40 variables
(definida solo a partir del ranking 2010->2013) se recalcula tambien
sobre el panel y covariables de la transicion 2013->2016, para poder
comparar como se ven las mismas variables en el otro periodo -- la
seleccion NO se vuelve a filtrar con el ranking de 2013->2016 (esa
pregunta, "cuales son robustas en los dos periodos", ya la responde
`eda_nucleo_estable.py`; aqui el objetivo es una comparacion visual
directa de las mismas 40 variables).

INPUTS
    Reutiliza las funciones y fuentes de datos de eda_transicion_covariables.py

OUTPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_monetaria_2010_2013.csv
    outputs/tables/eda_transicion_covariables/perfil_completo_monetaria_2013_2016.csv

COMO CORRER
    cd src/02_build && python eda_perfil_completo.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTARIO_PATH = PROJECT_ROOT / "outputs" / "tables" / "eda_variables_modelo" / "01_inventario_variables.csv"

UMBRAL_EFECTO = 0.10
DUPLICADOS_EXACTOS = ["ingreso_percapita_hogar", "gasto_percapita_hogar"]  # se prefiere la version "_real"

# Categoria tematica y sub-panel por unidad -- asignado a mano a partir
# del `modulo` del inventario y de la unidad real de cada variable
# (revisado con `construir_tabla_comparativa`: "(media)" vs "(media, %)"
# vs categorica -- ver columna nivel_mostrado del output).
CATEGORIA = {
    "dmsp_stable_lights": ("Geoespacial", "Iluminación nocturna (0-63)"),
    "zona": ("Zona de residencia", "% del grupo"),
    "brecha_lp_ingreso": ("Ingreso y gasto", "Veces la línea de pobreza"),
    "brecha_lp_gasto": ("Ingreso y gasto", "Veces la línea de pobreza"),
    "ingreso_percapita_hogar_real": ("Ingreso y gasto", "Miles $ por mes"),
    "gasto_percapita_hogar_real": ("Ingreso y gasto", "Miles $ por mes"),
    "material_pisos_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "energia_cocinan_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "servicio_sanitario_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "eliminan_basura_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "obtencion_agua_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "tipo_vivienda_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "material_paredes_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "tenencia_vivienda_hogar": ("Vivienda: materiales y servicios", "% del grupo"),
    "personas_por_cuarto_hogar": ("Vivienda: hacinamiento", "Personas por cuarto/dormitorio"),
    "personas_por_dormitorio_hogar": ("Vivienda: hacinamiento", "Personas por cuarto/dormitorio"),
    "valor_arriendo_pagado_hogar": ("Vivienda: hacinamiento", "Miles $ de arriendo/mes"),
    "n_bienes_durables_hogar": ("Activos del hogar", "Número (conteo)"),
    "n_servicios_publicos_hogar": ("Activos del hogar", "Número (conteo)"),
    "n_activos_financieros_hogar": ("Activos del hogar", "Número (conteo)"),
    "estrato_hogar": ("Activos del hogar", "Estrato (1-6)"),
    "estrato_verificado_hogar": ("Activos del hogar", "Estrato (1-6)"),
    "riqueza_pca_hogar": ("Activos del hogar", "Índice de riqueza (PCA)"),
    "tiene_internet_hogar": ("Activos del hogar", "% del grupo"),
    "n_programas_sociales_hogar": ("Programas sociales y deuda", "Número (conteo)"),
    "beneficiario_familias_accion_hogar": ("Programas sociales y deuda", "% del grupo"),
    "beneficiario_algun_programa_hogar": ("Programas sociales y deuda", "% del grupo"),
    "nivel_educ_jefe": ("Educación y empleo del jefe", "% del grupo"),
    "categoria_ocupacional_jefe": ("Educación y empleo del jefe", "% del grupo"),
    "medio_consiguio_jefe": ("Educación y empleo del jefe", "% del grupo"),
    "registro_mercantil_jefe": ("Educación y empleo del jefe", "% del grupo"),
    "n_empleados_jefe": ("Educación y empleo del jefe", "% del grupo"),
    "etnia_jefe": ("Educación y empleo del jefe", "% del grupo"),
    "estado_civil_jefe": ("Educación y empleo del jefe", "% del grupo"),
    "nivel_educ_max_hogar": ("Educación y empleo del jefe", "Años/nivel (escala propia)"),
    "nivel_educ_ordinal_jefe": ("Educación y empleo del jefe", "Años/nivel (escala propia)"),
    "tasa_cotizacion_pension_hogar": ("Educación y empleo del jefe", "% del grupo"),
    "cotiza_pension_jefe": ("Educación y empleo del jefe", "% del grupo"),
    "n_ninos_12": ("Composición del hogar", "Número (conteo)"),
    "razon_dependencia_demografica": ("Composición del hogar", "Razón de dependencia"),
}


def calcular_seleccion_2010_2013() -> list:
    """La seleccion de 40 variables SIEMPRE se define a partir del ranking
    de 2010->2013 (robusto + efecto>UMBRAL_EFECTO) -- la transicion
    2013->2016 reusa exactamente la MISMA lista de variables (no vuelve a
    filtrar por su propio ranking), porque el objetivo (pedido del
    usuario 2026-09-11) es comparar como se ven las MISMAS variables en
    el otro periodo, no encontrar cuales son robustas ahi tambien (eso ya
    se hizo en `eda_nucleo_estable.py`)."""
    ranking = pd.read_csv(TABLES_DIR / "ranking_covariables_monetaria_2010_2013.csv")
    filtradas = ranking[(ranking["robusto"]) & (ranking["efecto"] > UMBRAL_EFECTO)]
    seleccion = [v for v in filtradas["variable"] if v not in DUPLICADOS_EXACTOS]
    faltantes = set(seleccion) - set(CATEGORIA)
    if faltantes:
        raise ValueError(f"Variables sin categoria asignada en CATEGORIA: {faltantes}")
    return seleccion


def construir_perfil(seleccion: list, ola_ini: int, ola_fin: int, anio_dmsp: int, sufijo: str) -> pd.DataFrame:
    dmsp = cargar_dmsp_por_consecutivo(anio_dmsp)
    covariables = cargar_covariables_ola_base(dmsp, ola_ini)
    tipos = cargar_tipos_variables()

    pobreza = pd.read_parquet(POBREZA_PATH)
    llave_pobreza = _llave_compuesta(pobreza)
    cargar_pesos_muestrales(pobreza, llave_pobreza)
    resultado = construir_matriz_transicion(
        pobreza, ola_ini, ola_fin, col_pobre="pobre_ingreso", peso_col="peso_longitudinal"
    )
    panel = resultado["panel_categorias"]

    # El ranking de referencia para "efecto" en la tabla es siempre el de
    # 2010->2013 (asi el orden/color de las variables es el mismo en
    # ambos periodos); los VALORES por grupo si se calculan sobre el
    # panel y covariables de esta transicion especifica.
    ranking_referencia = pd.read_csv(TABLES_DIR / "ranking_covariables_monetaria_2010_2013.csv")
    tabla = construir_tabla_comparativa(covariables, tipos, panel, seleccion, ranking_referencia)
    tabla["categoria"] = tabla["variable"].map(lambda v: CATEGORIA[v][0])
    tabla["subpanel_unidad"] = tabla["variable"].map(lambda v: CATEGORIA[v][1])
    tabla.to_csv(TABLES_DIR / f"perfil_completo_monetaria_{sufijo}.csv", index=False)
    print(f"Guardado: perfil_completo_monetaria_{sufijo}.csv ({len(tabla)} variables)")
    return tabla


def main() -> None:
    seleccion = calcular_seleccion_2010_2013()
    construir_perfil(seleccion, 1, 2, 2010, "2010_2013")
    construir_perfil(seleccion, 2, 3, 2013, "2013_2016")


if __name__ == "__main__":
    main()
