"""
graf_otras_fuentes_geoespaciales.py
=====================================
Figura para ALOS PALSAR (radar) y Landsat 5 TM (NDVI) por grupo de
transicion, pobreza monetaria 2010->2013 -- pedido del usuario
(2026-09-11) para incluirlas en la subseccion geoespacial de la tesis,
junto a DMSP-OLS. A DIFERENCIA de DMSP, estas 2 variables NO pasaron por
el mismo filtro de robustez (eta^2 + bootstrap) que arma el universo de
`eda_perfil_completo.py` -- viven en `GEO_PATH` pero nunca se agregaron
al consolidado de covariables candidatas, asi que esta figura es
puramente descriptiva (media ponderada por grupo, sin prueba de
significancia), consistente con el tratamiento exploratorio que ya les
da la tesis en la seccion de contribucion marginal (sin holdout real en
la ola de prueba).

INPUTS
    outputs/tables/eda_transicion_covariables/otras_fuentes_geoespaciales_monetaria_2010_2013.csv
    (generado por eda_otras_fuentes_geoespaciales.py)

OUTPUTS
    outputs/figures/eda_transicion_covariables/perfil_otras_fuentes_geoespaciales_monetaria.png

COMO CORRER
    python src/graf_otras_fuentes_geoespaciales.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
FIG_DIR = REPO_ROOT / "outputs" / "figures" / "eda_transicion_covariables"

GRUPOS = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]
GRUPOS_ETIQUETA = ["Siempre\npobre", "Sale de la\npobreza", "Entra en\npobreza\n(VULNERABLE)", "Nunca\npobre"]


def graficar_panel(ax, valores, titulo, color, decimales):
    x = [0, 1, 2, 3]
    ax.axvspan(1.6, 2.4, color="#eb6834", alpha=0.07, zorder=0)
    ax.plot(x, valores, "-o", color=color, linewidth=2, markersize=7)
    for xi, v in zip(x, valores):
        ax.annotate(f"{v:.{decimales}f}", (xi, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=color, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(GRUPOS_ETIQUETA, fontsize=8)
    for label in ax.get_xticklabels():
        if "VULNERABLE" in label.get_text():
            label.set_color("#c1531a")
            label.set_fontweight("bold")
    ax.set_title(titulo, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.3, 3.3)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.15 or 1
    ax.set_ylim(ymin - pad * 0.1, ymax + pad)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tabla = pd.read_csv(TABLES_DIR / "otras_fuentes_geoespaciales_monetaria_2010_2013.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))
    especificaciones = [
        ("alos_hh_db", "Radar ALOS PALSAR (HH, dB)", "#2a78d6", 2),
        ("l5_ndvi", "Vegetación Landsat 5 TM (NDVI)", "#1baf7a", 2),
    ]
    for ax, (var, titulo, color, dec) in zip(axes, especificaciones):
        fila = tabla[tabla["variable"] == var].iloc[0]
        valores = [fila[g] for g in GRUPOS]
        graficar_panel(ax, valores, titulo, color, dec)

    fig.tight_layout()
    out_path = FIG_DIR / "perfil_otras_fuentes_geoespaciales_monetaria.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
