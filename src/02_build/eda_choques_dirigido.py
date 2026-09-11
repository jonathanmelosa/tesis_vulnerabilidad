"""
Investigacion a fondo de por que las variables de Choques (y como se
afronta un choque) no pasan el umbral de robustez en el ranking de 4
grupos de `eda_transicion_covariables.py` -- pedido explicito del
usuario (2026-09-10) tras encontrar sospechoso un resultado tan
uniformemente nulo (0 de 17 variables robustas, en las 2 definiciones de
pobreza).

Tres diagnosticos, en las 2 TRANSICIONES (2010->2013 y 2013->2016) y las
2 definiciones de pobreza:

  1. Prueba dirigida "Entra en pobreza" vs "Nunca pobre" (eta^2 de 2
     grupos en vez de 4) -- prueba la hipotesis de que el eta^2 de 4
     grupos diluye una senal que solo existe entre estos 2 grupos.
     RESULTADO EN 2010->2013 (ya verificado): la hipotesis NO se
     sostiene, el efecto de 2 grupos es MENOR que el de 4 grupos en las
     17 variables -- no es un problema de dilucion.

  2. Medias ponderadas reales por grupo (no solo el eta^2 resumen) --
     revela que el patron NO es plano: `total_choques_hogar` SI muestra
     una brecha sustantiva entre "entra" (0.63) y "nunca pobre" (0.49) en
     2010->2013, pero las variables de AFRONTAMIENTO (erosivo, redujo
     alimentos) muestran a "entra" como el grupo con el valor MAS BAJO de
     los 4 -- lo opuesto a lo esperado teoricamente -- con una celda
     "entra" muy chica (n~250-260, porque afrontamiento solo aplica a
     quien tuvo choque, ~35% de la muestra). Esto sugiere que el
     eta^2 de rangos puede estar subestimando un efecto real que existe
     en niveles pero se pierde en el ordenamiento, y que la celda "entra"
     de afrontamiento es lo bastante chica para que el patron sea
     ruidoso.

  3. Cobertura/n_efectivo: `afrontamiento_*` y `redujo_alimentos_*` solo
     tienen dato para hogares que tuvieron ALGUN choque (~35% de la
     muestra en 2010, correcto por diseno -- NaN para quien no tuvo
     choque, ver docs/decisions.md 2026-08-09), asi que su n_efectivo de
     Kish (~1,183) es menos de la mitad que el de las demas variables
     (~2,758) -- el bootstrap tiene menos poder para confirmar robustez
     aunque el punto estimado supere el umbral.

INPUTS
    Reutiliza las funciones y fuentes de datos de eda_transicion_covariables.py

OUTPUTS
    outputs/tables/eda_transicion_covariables/choques_dirigido_{monetaria,ipm}_{sufijo}.csv
    outputs/tables/eda_transicion_covariables/choques_medias_grupo_{monetaria,ipm}_{sufijo}.csv

COMO CORRER
    cd src/02_build && python eda_choques_dirigido.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import (  # noqa: E402
    _llave_compuesta,
    cargar_pesos_muestrales,
    construir_matriz_transicion,
)
from eda_transicion_covariables import (  # noqa: E402
    CATEGORIAS_ORDEN,
    IPM_PATH,
    POBREZA_PATH,
    TABLES_DIR,
    TRANSICIONES,
    bootstrap_ci,
    cargar_covariables_ola_base,
    cargar_dmsp_por_consecutivo,
    cargar_peso_longitudinal_por_consecutivo,
    weighted_rank_eta2,
)

UMBRAL_ETA2 = 0.01
UMBRAL_CI_ETA2 = 0.005

VARIABLES_CHOQUES = [
    "total_choques_hogar", "tuvo_algun_choque_hogar", "n_tipos_choque_hogar",
    "tuvo_choque_salud_hogar", "tuvo_choque_economico_hogar", "tuvo_choque_patrimonial_hogar",
    "tuvo_choque_agropecuario_hogar", "tuvo_choque_familiar_hogar", "tuvo_choque_severo_hogar",
    "afrontamiento_erosivo_hogar", "afrontamiento_protector_hogar", "intensifico_trabajo_hogar",
    "retiro_hijos_colegio_choque_hogar", "redujo_alimentos_choque_hogar",
    "se_endeudo_formal_choque_hogar", "se_endeudo_informal_choque_hogar", "no_ajusto_choque_hogar",
]

CATEGORIAS_COMPARADAS = ["Entra en pobreza", "Nunca pobre"]


def rankear_2_grupos(covariables: pd.DataFrame, panel: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Igual a `rankear_covariables` pero restringido a 2 categorias y a
    una lista de variables dada."""
    df = panel.merge(covariables, left_on="consecutivo", right_index=True, how="left")
    df = df[df["categoria"].isin(CATEGORIAS_COMPARADAS)]
    grupo, peso = df["categoria"], df["peso_longitudinal"]

    filas = []
    for var in variables:
        valores = df[var]
        if not pd.api.types.is_numeric_dtype(valores):
            continue
        efecto, n_eff = weighted_rank_eta2(valores, grupo, peso)
        if pd.isna(efecto):
            continue
        pasa_umbral = efecto >= UMBRAL_ETA2
        ci_low, ci_high = (float("nan"), float("nan"))
        if pasa_umbral:
            ci_low, ci_high = bootstrap_ci(valores, grupo, peso, "Numerica")
        filas.append({
            "variable": var, "efecto_2grupos": efecto, "n_efectivo": n_eff,
            "pasa_umbral": pasa_umbral, "ci_low": ci_low, "ci_high": ci_high,
            "robusto_2grupos": pasa_umbral and (not pd.isna(ci_low)) and ci_low >= UMBRAL_CI_ETA2,
        })
    return pd.DataFrame(filas).sort_values("efecto_2grupos", ascending=False).reset_index(drop=True)


def medias_por_grupo(covariables: pd.DataFrame, panel: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Media ponderada REAL de cada variable de Choques por categoria (las
    4), con n no-nulo por celda -- para inspeccionar el patron sustantivo
    directamente, sin pasar por el eta^2 de rangos (que puede comprimir
    diferencias reales de nivel, sobre todo con celdas chicas)."""
    df = panel.merge(covariables, left_on="consecutivo", right_index=True, how="left")
    df["categoria"] = pd.Categorical(df["categoria"], categories=CATEGORIAS_ORDEN, ordered=True)

    filas = []
    for var in variables:
        fila = {"variable": var}
        for cat in CATEGORIAS_ORDEN:
            sub = df[(df["categoria"] == cat) & df[var].notna() & df["peso_longitudinal"].notna()]
            media = np.average(sub[var], weights=sub["peso_longitudinal"]) if len(sub) else float("nan")
            fila[f"media_{cat}"] = media
            fila[f"n_{cat}"] = len(sub)
        fila["brecha_entra_vs_nunca"] = fila["media_Entra en pobreza"] - fila["media_Nunca pobre"]
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("brecha_entra_vs_nunca", key=abs, ascending=False).reset_index(drop=True)


def main() -> None:
    for transicion in TRANSICIONES:
        ola_ini, ola_fin = transicion["ola_ini"], transicion["ola_fin"]
        anio_dmsp, sufijo = transicion["anio_dmsp"], transicion["sufijo"]

        dmsp = cargar_dmsp_por_consecutivo(anio_dmsp)
        covariables = cargar_covariables_ola_base(dmsp, ola_ini)

        pobreza = pd.read_parquet(POBREZA_PATH)
        llave_pobreza = _llave_compuesta(pobreza)
        cargar_pesos_muestrales(pobreza, llave_pobreza)
        resultado_monetaria = construir_matriz_transicion(
            pobreza, ola_ini, ola_fin, col_pobre="pobre_ingreso", peso_col="peso_longitudinal"
        )

        ipm = pd.read_parquet(IPM_PATH)
        cargar_peso_longitudinal_por_consecutivo(ipm, ola_fin)
        resultado_ipm = construir_matriz_transicion(
            ipm, ola_ini, ola_fin, col_pobre="pobre_ipm", peso_col="peso_longitudinal"
        )

        ranking_4grupos = {
            nombre: pd.read_csv(TABLES_DIR / f"ranking_covariables_{nombre}_{sufijo}.csv")
            for nombre in ("monetaria", "ipm")
        }

        for nombre, resultado in [("monetaria", resultado_monetaria), ("ipm", resultado_ipm)]:
            panel = resultado["panel_categorias"]
            panel = panel.assign(categoria=pd.Categorical(panel["categoria"]))

            dirigido = rankear_2_grupos(covariables, panel, VARIABLES_CHOQUES)
            efecto_4g = dict(zip(ranking_4grupos[nombre]["variable"], ranking_4grupos[nombre]["efecto"]))
            dirigido["efecto_4grupos"] = dirigido["variable"].map(efecto_4g)
            dirigido["diferencia"] = dirigido["efecto_2grupos"] - dirigido["efecto_4grupos"]

            medias = medias_por_grupo(covariables, panel, VARIABLES_CHOQUES)

            print(f"\n=== CHOQUES -- {nombre.upper()} {sufijo} ===")
            print(f"Robustas en 2 grupos (entra vs nunca pobre): {dirigido['robusto_2grupos'].sum()} de {len(dirigido)}")
            print("\nMedias ponderadas por grupo (ordenado por |brecha entra-nunca|):")
            cols_mostrar = ["variable", "media_Siempre pobre", "media_Sale de la pobreza",
                             "media_Entra en pobreza", "media_Nunca pobre", "brecha_entra_vs_nunca", "n_Entra en pobreza"]
            with pd.option_context("display.width", 160, "display.precision", 3):
                print(medias[cols_mostrar].to_string(index=False))

            dirigido.to_csv(TABLES_DIR / f"choques_dirigido_{nombre}_{sufijo}.csv", index=False)
            medias.to_csv(TABLES_DIR / f"choques_medias_grupo_{nombre}_{sufijo}.csv", index=False)

    print(f"\nGuardado en: {TABLES_DIR}")


if __name__ == "__main__":
    main()
