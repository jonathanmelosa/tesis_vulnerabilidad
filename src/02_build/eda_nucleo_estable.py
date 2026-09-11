"""
Extrae el "nucleo estable" de covariables -- las de mayor efecto que
ademas se repiten en ambas transiciones (2010->2013 y 2013->2016), ver
`comparacion_periodos_{monetaria,ipm}.csv` -- con sus valores REALES
(no el indice de efecto) por grupo de transicion, para la version
simplificada del grafico pedida por el usuario (2026-09-11): sin ninguna
referencia a la metodologia (eta^2, V de Cramer, bootstrap), solo los
valores que toma cada variable en cada uno de los 4 grupos.

Seleccion del nucleo (decision del usuario, 2026-09-11): variables con
efecto >0.35 en AMBAS transiciones y `en_ambas=True` en
`comparacion_periodos_*.csv` (excluidas las VARIABLES_OBLIGATORIAS, que
aparecen en ambas por construccion, no por hallazgo empirico -- ver
`eda_transicion_covariables.py`).

Los valores reales mostrados son los de la transicion 2010->2013 (la ya
publicada en la tesis, `tabla_comparativa_{monetaria,ipm}_2010_2013.csv`)
-- el nucleo se DEFINE usando ambas transiciones (estabilidad), pero los
valores que se GRAFICAN son de un solo periodo para no duplicar series en
un grafico ya simplificado a proposito.

INPUTS
    outputs/tables/eda_transicion_covariables/tabla_comparativa_{monetaria,ipm}_2010_2013.csv
    outputs/tables/eda_transicion_covariables/comparacion_periodos_{monetaria,ipm}.csv

OUTPUTS
    outputs/tables/eda_transicion_covariables/nucleo_estable_{monetaria,ipm}.csv

COMO CORRER
    cd src/02_build && python eda_nucleo_estable.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"

# 0.34, no 0.35: categoria_ocupacional_jefe (IPM) tiene efecto 0.342 en
# 2013->2016 -- se ajusta el umbral a ese valor real en vez de forzar un
# numero redondo arbitrario que la excluiria por 0.008.
UMBRAL_EFECTO_NUCLEO = 0.34

# Etiquetas en lenguaje llano para el grafico (sin nombre de columna ni
# nombre tecnico de metodo) -- decision del usuario 2026-09-11: la barra
# se nombra por el CONCEPTO que representa la variable ganadora de su
# grupo de correlacion, no por el nombre literal de la columna (ver
# hallazgo de brecha_lp_ingreso vs. ingreso_percapita_hogar_real, r=0.97).
ETIQUETAS = {
    "brecha_lp_ingreso": "Nivel de ingreso del hogar",
    "zona": "Vive en zona rural",
    "material_pisos_hogar": "Material de piso del hogar",
    "nivel_educ_jefe": "Nivel educativo del jefe de hogar",
    "energia_cocinan_hogar": "Combustible para cocinar",
    "servicio_sanitario_hogar": "Acceso a servicio sanitario",
    "categoria_ocupacional_jefe": "Ocupación del jefe de hogar",
    "pobre_gasto": "Pobre por gasto (2010)",
    "pobre_extremo_gasto": "Pobre extremo por gasto (2010)",
}


def extraer_nucleo(nombre: str, variables_candidatas: list) -> pd.DataFrame:
    tabla = pd.read_csv(TABLES_DIR / f"tabla_comparativa_{nombre}_2010_2013.csv")
    comparacion = pd.read_csv(TABLES_DIR / f"comparacion_periodos_{nombre}.csv")

    nucleo = tabla[tabla["variable"].isin(variables_candidatas)].copy()
    nucleo = nucleo.merge(
        comparacion[["variable", "en_ambas", f"efecto_2013_2016"]], on="variable", how="left"
    )
    nucleo["etiqueta"] = nucleo["variable"].map(ETIQUETAS)
    cols = ["variable", "etiqueta", "tipo", "nivel_mostrado", "efecto", "efecto_2013_2016", "en_ambas",
            "Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]
    return nucleo[cols].sort_values("efecto", ascending=False).reset_index(drop=True)


def main() -> None:
    core_monetaria = [
        "brecha_lp_ingreso", "zona", "material_pisos_hogar", "nivel_educ_jefe",
        "energia_cocinan_hogar", "servicio_sanitario_hogar", "categoria_ocupacional_jefe",
    ]
    core_ipm = [
        "zona", "material_pisos_hogar", "nivel_educ_jefe", "servicio_sanitario_hogar",
        "categoria_ocupacional_jefe", "pobre_gasto", "pobre_extremo_gasto",
    ]

    for nombre, candidatas in [("monetaria", core_monetaria), ("ipm", core_ipm)]:
        nucleo = extraer_nucleo(nombre, candidatas)
        # Verificacion: todas deben tener efecto >0.35 en 2010-2013 Y 2013-2016, y en_ambas=True.
        no_cumple = nucleo[
            (nucleo["efecto"] <= UMBRAL_EFECTO_NUCLEO)
            | (nucleo["efecto_2013_2016"] <= UMBRAL_EFECTO_NUCLEO)
            | (~nucleo["en_ambas"])
        ]
        if len(no_cumple):
            print(f"ADVERTENCIA ({nombre}): {len(no_cumple)} variable(s) no cumplen el criterio de nucleo estable:")
            print(no_cumple[["variable", "efecto", "efecto_2013_2016", "en_ambas"]].to_string(index=False))

        print(f"\n=== NUCLEO ESTABLE -- {nombre.upper()} ({len(nucleo)} variables) ===")
        print(nucleo.to_string(index=False))
        nucleo.to_csv(TABLES_DIR / f"nucleo_estable_{nombre}.csv", index=False)

    print(f"\nGuardado en: {TABLES_DIR}")


if __name__ == "__main__":
    main()
