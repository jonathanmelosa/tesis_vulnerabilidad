"""
eda_choques_ola3_split_trayectoria.py
=====================================
Pregunta del usuario (2026-09-21): todo el analisis de choques hecho hasta
ahora usa los choques reportados en la ola base (2010, contemporaneos con
el paso a la pobreza 2010->2013). Este script pregunta algo distinto: de
los 610 hogares "Entra en pobreza" ya separados en transitorio (355,
vuelve a salir para 2016) y persistente (255, sigue pobre en 2016) por
`eda_perfil_split_trayectoria.py`, ¿los choques que vivieron DESPUES de
caer en pobreza -- entre 2013 y 2016, reportados en la ola 3 (2016) de
`choques_hogar_elca_longitudinal.parquet`, que ya existe con las 3 olas--
ayudan a explicar quien se queda atrapado y quien se recupera?

Distincion clave con el resto del perfil (Hallazgo 3, Seccion 5.2.1): esas
comparaciones (deuda informal, cotizacion a pension) usan covariables de
la OLA BASE (2010) -- lo que el hogar YA tenia antes/al momento de caer.
Este script usa covariables de la OLA 3 (2016) -- lo que le paso al hogar
DESPUES de caer, mientras ya era pobre. Es una prueba mas directa de la
hipotesis de "exposicion a riesgo hacia adelante" (vs. "acervo de activos
hacia atras") que se discutio para explicar el techo de AUC-ROC del
benchmark (Seccion 5.1.1/5.3).

METODOLOGIA
    Reusa `construir_panel_split()` de `eda_perfil_split_trayectoria.py`
    (NO se recalcula la clasificacion transitorio/persistente, ya
    validada). Pega las 17 variables de choques de
    `choques_hogar_elca_longitudinal.parquet` filtradas a ola==3 (2016)
    por `consecutivo`. Reporta promedio/tasa ponderada por
    `peso_longitudinal` en cada uno de los 5 grupos de la matriz de
    transicion (con "Entra" ya separado en transitorio/persistente),
    igual que el resto del perfil de la Seccion 5.1 -- pero SOLO
    descriptivo (sin el bootstrap de robustez de eta^2/Cramer's V que usa
    la seleccion de las 53 variables oficiales, consistente con que
    `eda_perfil_split_trayectoria.py` tampoco lo repite ahi, y con
    muestras de 255-355 hogares que ya son chicas para un ejercicio de
    ese tipo).

INPUTS
    outputs/tables/eda_transicion_covariables/trayectorias_3olas_monetaria_hogares.csv
    data/processed/pobreza_monetaria_elca_longitudinal.parquet
    data/processed/choques_hogar_elca_longitudinal.parquet

OUTPUTS
    outputs/tables/eda_transicion_covariables/choques_ola3_split_trayectoria.csv

COMO CORRER
    cd src/02_build && python eda_choques_ola3_split_trayectoria.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_perfil_split_trayectoria import CATEGORIAS_ORDEN_SPLIT, construir_panel_split  # noqa: E402
from eda_transicion_covariables import PROJECT_ROOT, TABLES_DIR  # noqa: E402

CHOQUES_PATH = PROJECT_ROOT / "data" / "processed" / "choques_hogar_elca_longitudinal.parquet"
OLA_CHOQUES_POSTERIOR = 3  # 2016 -- lo vivido ENTRE 2013 y 2016, ya siendo pobre (para "Entra")

COLUMNAS_EXCLUIR = ["consecutivo", "llave", "llave_n16", "ola", "zona"]


def cargar_choques_ola(ola: int) -> pd.DataFrame:
    choques = pd.read_parquet(CHOQUES_PATH)
    choques = choques[choques["ola"] == ola].copy()
    cols_choque = [c for c in choques.columns if c not in COLUMNAS_EXCLUIR]
    return choques[["consecutivo"] + cols_choque], cols_choque


def promedio_ponderado(valores: pd.Series, pesos: pd.Series) -> float:
    mask = valores.notna() & pesos.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(valores[mask], weights=pesos[mask]))


def main() -> None:
    panel = construir_panel_split()
    choques, cols_choque = cargar_choques_ola(OLA_CHOQUES_POSTERIOR)

    datos = panel.merge(choques, on="consecutivo", how="left")
    n_con_dato = datos.groupby("categoria")[cols_choque[0]].apply(lambda s: s.notna().sum())
    print(f"Hogares con dato de choques en ola {OLA_CHOQUES_POSTERIOR} (2016), por grupo:")
    print(n_con_dato.reindex(CATEGORIAS_ORDEN_SPLIT))
    print()

    filas = []
    for variable in cols_choque:
        fila = {"variable": variable}
        for categoria in CATEGORIAS_ORDEN_SPLIT:
            sub = datos[datos["categoria"] == categoria]
            fila[categoria] = round(promedio_ponderado(sub[variable], sub["peso_longitudinal"]), 4)
        fila["dif_persistente_menos_transitorio"] = round(
            fila["Entra (persistente)"] - fila["Entra (transitorio)"], 4
        )
        filas.append(fila)

    tabla = pd.DataFrame(filas).sort_values("dif_persistente_menos_transitorio", key=abs, ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "choques_ola3_split_trayectoria.csv"
    tabla.to_csv(out_path, index=False)

    print("Choques en ola 3 (2016), ordenados por |diferencia persistente - transitorio|:")
    print(tabla.to_string(index=False))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
