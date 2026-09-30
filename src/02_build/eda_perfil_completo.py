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

EXTENSION 2026-09-15 (pedido explicito del usuario, tras revisar con el
asesor de tesis por que la seleccion quedo en 40 y no en mas variables):
se agregan 13 variables adicionales que NO pasan el umbral de 0.10, pero
que superan un criterio mas estricto que "robusto" a secas -- estas 13
son "robusto=True" Y efecto>0.05 EN LAS DOS TRANSICIONES POR SEPARADO
(2010->2013 Y 2013->2016, verificado contra
`ranking_covariables_monetaria_2013_2016.csv`), es decir, su señal no es
un artefacto de un solo periodo. De las 18 variables candidatas que
pasaban efecto>0.05 solo en 2010->2013, se descartaron 5 cuyo efecto cae
por debajo de 0.05 (o deja de ser robusto) en 2013->2016: `tiene_deuda_hogar`,
`tvip_puntaje_directo_hogar`, `tasa_ahorro_hogar`,
`pct_ninos_control_pediatrico`, y sobre todo
`pct_ninos_no_estudia_razon_economica` (efecto 0.076 -> 0.011, deja de
ser robusto en el segundo periodo -- la señal del primer periodo era,
aparentemente, ruido de esa muestra). Se verifico ademas que ninguna de
las 13 correlaciona por encima de |rho|=0.70 (Spearman) con ninguna de
las 40 variables ya seleccionadas (la maxima observada es 0.43, entre
`n_espacios_publicos_comunidad` y `dmsp_stable_lights`) -- no son
redundantes con el nucleo existente. Resultado: 53 variables en total
(40 + 13) para la version final de la tesis.

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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etiquetas_variables import CATEGORIA  # noqa: E402
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

# 13 variables adicionales (ver docstring, "EXTENSION 2026-09-15"): robustas
# y con efecto>0.05 en AMBAS transiciones, no redundantes (|rho|<0.70) con
# ninguna de las 40 ya seleccionadas por UMBRAL_EFECTO.
VARIABLES_ESTABLES_AMPLIADAS = [
    "tasa_afiliacion_pension_hogar", "tasa_afiliacion_salud_laboral_hogar",
    "afiliado_pension_jefe", "afiliado_salud_laboral_jefe",
    "deuda_formal_hogar", "deuda_informal_hogar",
    "tiene_escritura_vivienda_hogar", "financio_credito_formal_vivienda_hogar",
    "tiene_vehiculo_hogar",
    "pct_ninos_cuidado_terceros_hogar", "pct_ninos_apoyo_alimentario_escolar",
    "n_espacios_publicos_comunidad", "tiene_transporte_publico_comunidad",
]

# CATEGORIA vive en src/etiquetas_variables.py (movida 2026-09-30).


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

    # Extensión 2026-09-15: agrega las 13 variables de VARIABLES_ESTABLES_AMPLIADAS
    # (robustas y efecto>0.05 en AMBAS transiciones, no redundantes con las 40
    # anteriores -- ver docstring del modulo). Se verifica aqui, en tiempo de
    # ejecucion, que efectivamente cumplen ese criterio contra el ranking
    # ACTUAL de las dos transiciones -- si algun dia se recalculan los rankings
    # con datos nuevos y una de estas 13 deja de cumplirlo, el script falla
    # ruidosamente en vez de incluirla en silencio.
    ranking_2013_2016 = pd.read_csv(TABLES_DIR / "ranking_covariables_monetaria_2013_2016.csv")
    r1 = ranking.set_index("variable")
    r2 = ranking_2013_2016.set_index("variable")
    for var in VARIABLES_ESTABLES_AMPLIADAS:
        ok_1 = var in r1.index and bool(r1.loc[var, "robusto"]) and r1.loc[var, "efecto"] > 0.05
        ok_2 = var in r2.index and bool(r2.loc[var, "robusto"]) and r2.loc[var, "efecto"] > 0.05
        if not (ok_1 and ok_2):
            raise ValueError(
                f"{var!r} está en VARIABLES_ESTABLES_AMPLIADAS pero ya no cumple "
                f"robusto+efecto>0.05 en ambas transiciones (revisar rankings actuales)."
            )
        if var not in seleccion:
            seleccion.append(var)

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
