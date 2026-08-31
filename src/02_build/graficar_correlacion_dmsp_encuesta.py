"""
graficar_correlacion_dmsp_encuesta.py
=====================================
Version legible, enfocada solo en iluminacion nocturna (DMSP-OLS), de la
matriz de correlacion general de `eda_variables_modelo.py`
(04_matriz_correlacion.png) -- pedido por el usuario (2026-08-31) porque
esa matriz incluye TODAS las variables numericas del inventario (100+),
con etiquetas en fuente tamano 5, lo que hace muy dificil ubicar alli las
2 variables DMSP-OLS (dmsp_stable_lights, dmsp_stable_lights_acum_tendencia)
y leer sus correlaciones con el resto de variables de encuesta.

NO MODIFICA nada de lo ya hecho: reutiliza (importa, no duplica)
`cargar_consolidado_con_geo`, `clasificar_tipo`, `ID_COLS`, `MIN_N_CORR`
y `GEO_INSIGNIA_2010` de `eda_variables_modelo.py`, y escribe unicamente
archivos NUEVOS (no sobreescribe 04_matriz_correlacion.png ni
04_correlaciones_altas.csv).

QUE HACE

    1. Carga el mismo consolidado + geoespacial que usa la matriz general.
    2. Calcula la correlacion de Pearson de cada una de las 2 variables
       DMSP-OLS contra TODAS las demas variables numericas de encuesta
       (excluye ALOS/Landsat, que no son "de encuesta"), con el mismo
       filtro de robustez que la matriz general (n conjunto >= 500).
    3. Genera dos vistas nuevas:
       a. Barras horizontales, una figura por variable DMSP, ordenadas
          por |r| descendente (top 25) -- ranking claro.
       b. Franja de heatmap (2 filas x variables de encuesta con
          |r| >= 0.2, para legibilidad), mismo estilo/colormap que la
          matriz original pero con fuente legible.
    4. Exporta la tabla completa de correlaciones (todas las variables,
       no solo el top 25) para consulta.

INPUTS

    data/processed/benchmark_consolidado_elca_longitudinal.parquet
    data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet

OUTPUTS (todos NUEVOS, no pisan nada existente)

    outputs/figures/eda_variables_modelo/04b_correlacion_dmsp_barras.png
    outputs/figures/eda_variables_modelo/04c_correlacion_dmsp_franja.png
    outputs/tables/eda_variables_modelo/04b_correlaciones_dmsp_encuesta.csv

COMO CORRER

    cd src/02_build && python graficar_correlacion_dmsp_encuesta.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_variables_modelo import (  # noqa: E402
    GEO_INSIGNIA_2010,
    ID_COLS,
    MIN_N_CORR,
    cargar_consolidado_con_geo,
    clasificar_tipo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_variables_modelo"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda_variables_modelo"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VARS_DMSP = ["dmsp_stable_lights", "dmsp_stable_lights_acum_tendencia"]
UMBRAL_FRANJA = 0.2  # |r| minimo para entrar a la franja de heatmap (legibilidad)
TOP_N_BARRAS = 25

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def calcular_correlaciones(df: pd.DataFrame) -> pd.DataFrame:
    content_cols = [c for c in df.columns if c not in ID_COLS]
    numericas = [c for c in content_cols if clasificar_tipo(df[c]) == "Numerica"]
    # variables "de encuesta": todas las numericas del consolidado menos las
    # geoespaciales (DMSP mismas, y ALOS/Landsat, que no son de encuesta).
    vars_encuesta = [c for c in numericas if c not in GEO_INSIGNIA_2010]

    presente = df[VARS_DMSP + vars_encuesta].notna().astype(int)
    n_conjunto = presente[VARS_DMSP].T.dot(presente[vars_encuesta])

    filas = []
    for var_dmsp in VARS_DMSP:
        for var_enc in vars_encuesta:
            n_ij = int(n_conjunto.loc[var_dmsp, var_enc])
            if n_ij < MIN_N_CORR:
                continue
            r = df[var_dmsp].corr(df[var_enc])
            if pd.isna(r):
                continue
            filas.append({"variable_dmsp": var_dmsp, "variable_encuesta": var_enc, "r": r, "n_conjunto": n_ij})

    tabla = pd.DataFrame(filas).sort_values(["variable_dmsp", "r"], key=lambda s: s.abs() if s.name == "r" else s, ascending=[True, False])
    return tabla


def graficar_barras(tabla: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(VARS_DMSP), figsize=(16, 9))
    for ax, var_dmsp in zip(axes, VARS_DMSP):
        sub = tabla[tabla["variable_dmsp"] == var_dmsp].reindex(
            tabla[tabla["variable_dmsp"] == var_dmsp]["r"].abs().sort_values(ascending=False).index
        ).head(TOP_N_BARRAS)
        sub = sub.iloc[::-1]  # para que el mayor quede arriba en barh
        colores = ["#C44E52" if r < 0 else "#4C72B0" for r in sub["r"]]
        ax.barh(sub["variable_encuesta"], sub["r"], color=colores)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Correlacion de Pearson (r)")
        ax.set_title(f"Top {TOP_N_BARRAS} variables de encuesta\ncorrelacionadas con {var_dmsp}", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Correlacion de iluminacion nocturna (DMSP-OLS) con variables de encuesta", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04b_correlacion_dmsp_barras.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {FIGURES_DIR / '04b_correlacion_dmsp_barras.png'}")


def graficar_franja(tabla: pd.DataFrame) -> None:
    vars_relevantes = sorted(
        tabla.loc[tabla["r"].abs() >= UMBRAL_FRANJA, "variable_encuesta"].unique().tolist(),
        key=lambda v: -tabla.loc[tabla["variable_encuesta"] == v, "r"].abs().max(),
    )
    if not vars_relevantes:
        print(f"Ninguna variable de encuesta con |r| >= {UMBRAL_FRANJA} -- no se genera la franja.")
        return

    matriz = tabla.pivot(index="variable_dmsp", columns="variable_encuesta", values="r").reindex(columns=vars_relevantes)

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(vars_relevantes)), 3.5))
    im = ax.imshow(matriz.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(vars_relevantes)))
    ax.set_xticklabels(vars_relevantes, fontsize=8, rotation=90)
    ax.set_yticks(range(len(VARS_DMSP)))
    ax.set_yticklabels(matriz.index.tolist(), fontsize=9)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, f"{matriz.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlacion de Pearson", pad=0.01)
    ax.set_title(f"Correlacion DMSP-OLS vs. variables de encuesta con |r| >= {UMBRAL_FRANJA}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04c_correlacion_dmsp_franja.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {FIGURES_DIR / '04c_correlacion_dmsp_franja.png'} ({len(vars_relevantes)} variables)")


def main() -> None:
    df = cargar_consolidado_con_geo()
    tabla = calcular_correlaciones(df)
    ruta_csv = TABLES_DIR / "04b_correlaciones_dmsp_encuesta.csv"
    tabla.to_csv(ruta_csv, index=False)
    print(f"Guardado: {ruta_csv} ({len(tabla)} pares)")

    graficar_barras(tabla)
    graficar_franja(tabla)

    print("\nTop 5 correlaciones (|r|) por variable DMSP:")
    for var_dmsp in VARS_DMSP:
        sub = tabla[tabla["variable_dmsp"] == var_dmsp]
        top5 = sub.reindex(sub["r"].abs().sort_values(ascending=False).index).head(5)
        print(f"\n{var_dmsp}:")
        print(top5[["variable_encuesta", "r", "n_conjunto"]].to_string(index=False))


if __name__ == "__main__":
    main()
