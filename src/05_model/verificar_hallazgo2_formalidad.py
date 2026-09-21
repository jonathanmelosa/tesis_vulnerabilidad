"""
verificar_hallazgo2_formalidad.py
=====================================
Verifica si el Segundo hallazgo de la Seccion 5.2.2 (la formalidad
laboral del jefe -- categoria_ocupacional_jefe -- es fuerte en el perfil
univariado pero cae a importancia marginal/nula en multivariado, mientras
tasa_cotizacion_pension_hogar sobrevive con fuerza) se sostiene tambien
bajo IPM y bajo la ventana 2013->2016 propia, no solo en la combinacion
original (monetaria, 2010->2013 evaluado en test 2013->2016).

NO recalcula SHAP: solo re-agrega (misma logica one-hot->variable de
`diagnostico_shap_nucleo_perfil.py`) y reporta el rango de posiciones de
las variables relevantes en cada una de las 4 combinaciones ya
calculadas.

COMO CORRER

    cd src/05_model && python verificar_hallazgo2_formalidad.py
"""

import re

import pandas as pd

import modelo_utils as mu

VARIABLES_INTERES = [
    "categoria_ocupacional_jefe",
    "tasa_cotizacion_pension_hogar",
    "pct_ninos_cuidado_terceros_hogar",
    "tiene_transporte_publico_comunidad",
    "deuda_informal_hogar",
]

FUENTES = {
    "Monetaria_2010-13(test)": {
        "csv": mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ab.csv",
        "parquet": mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_A_2010_2013.parquet",
    },
    "Monetaria_2013-16(propia)": {
        "csv": mu.RESULTADOS_DIR / "diagnostico_shap_importancia_2013_2016.csv",
        "parquet": mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_A_2013_2016.parquet",
    },
    "IPM_2010-13(test)": {
        "csv": mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ipm.csv",
        "parquet": mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_Aipm_2010_2013.parquet",
    },
    "IPM_2013-16(propia)": {
        "csv": mu.RESULTADOS_DIR / "diagnostico_shap_importancia_ipm_2013_2016.csv",
        "parquet": mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test" / "modelo_Aipm_2013_2016.parquet",
    },
}


def _variable_base_fn(parquet_path):
    df = pd.read_parquet(parquet_path)
    categoricas = df.select_dtypes(include="object").columns.tolist()
    categoricas_por_largo = sorted(categoricas, key=len, reverse=True)

    def variable_base(columna: str) -> str:
        columna = re.sub(r"^(num__|cat__)", "", columna)
        columna = re.sub(r"^missingindicator_", "", columna)
        for c in categoricas_por_largo:
            if columna == c or columna.startswith(c + "_"):
                return c
        return columna

    return variable_base


def main() -> None:
    for nombre, cfg in FUENTES.items():
        print(f"\n{'='*70}\n{nombre}\n{'='*70}")
        shap_raw = pd.read_csv(cfg["csv"])
        variable_base = _variable_base_fn(cfg["parquet"])
        shap_raw = shap_raw.copy()
        shap_raw["base"] = shap_raw["variable"].map(variable_base)

        for (algoritmo, espec), g in shap_raw.groupby(["algoritmo", "especificacion"]):
            agregado = g.groupby("base")["shap_abs_medio"].sum().sort_values(ascending=False)
            ranking = agregado.rank(ascending=False, method="min")
            n_total = len(agregado)

            fila_rangos = []
            for var in VARIABLES_INTERES:
                if var in ranking.index:
                    fila_rangos.append(f"{var}={int(ranking[var])}/{n_total}")
                else:
                    fila_rangos.append(f"{var}=ausente")
            print(f"  {algoritmo:35s} {espec:15s} " + "  ".join(fila_rangos))


if __name__ == "__main__":
    main()
