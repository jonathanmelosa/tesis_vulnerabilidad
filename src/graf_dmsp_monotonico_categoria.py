"""
graf_dmsp_monotonico_categoria.py
===================================
Grafica el HALLAZGO PRINCIPAL de la covariable DMSP (iluminacion nocturna)
sobre los 4 grupos de transicion de pobreza: relacion MONOTONICA -- Siempre
pobre tiene la iluminacion mas baja, Nunca pobre la mas alta, con Sale de
la pobreza y Entra en pobreza en el orden intermedio esperado, tanto para
pobreza monetaria como IPM (2010->2013).

POR QUE UN GRAFICO DE BARRAS Y NO UN MAPA (decision del usuario, 2026-09-12)
    Se probaron varias variantes de mapa territorial (tramas, hexagonos,
    glifos por region, small multiples) para mostrar los 4 grupos sobre el
    fondo de brillo DMSP. El usuario señalo que ninguna deja ver con
    claridad el hallazgo real que quiere comunicar: la relacion monotonica
    categoria->iluminacion. Un mapa dispersa la atencion en la geografia;
    el hallazgo es estadistico (una relacion ORDENADA entre 4 categorias y
    una variable continua), y un grafico de barras ordenado por valor lo
    muestra sin ambiguedad, sin competir con ningun fondo.

DATOS (reales, ya calculados por eda_transicion_covariables.py)
    Siempre pobre < Sale de la pobreza < Entra en pobreza < Nunca pobre,
    tanto en media ponderada como en mediana, en monetaria E IPM (4 de 4
    combinaciones monotonicas) -- ver
    outputs/tables/eda_transicion_covariables/dmsp_por_categoria_{monetaria,ipm}_2010_2013.csv

INPUTS
    outputs/tables/eda_transicion_covariables/dmsp_por_categoria_monetaria_2010_2013.csv
    outputs/tables/eda_transicion_covariables/dmsp_por_categoria_ipm_2010_2013.csv

OUTPUTS
    outputs/figures/eda_transicion_covariables/dmsp_monotonico_categoria.png

COMO CORRER
    python src/graf_dmsp_monotonico_categoria.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda_transicion_covariables"

# Misma paleta categorica que el resto de la tesis (build_pobreza_desagregaciones.py,
# mapa_transicion_regiones.py) -- consistencia de color por grupo en todo el documento.
COLOR_CATEGORIA = {
    "Nunca pobre": "#2a78d6",
    "Sale de la pobreza": "#1baf7a",
    "Entra en pobreza": "#eb6834",
    "Siempre pobre": "#e34948",
}
INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECUNDARIO, "text.color": INK_PRIMARIO,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "font.family": "sans-serif",
    "font.size": 10.5, "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def graficar_panel(ax, tabla: pd.DataFrame, titulo: str) -> None:
    tabla = tabla.sort_values("dmsp_media_ponderada")
    colores = [COLOR_CATEGORIA[c] for c in tabla["categoria"]]
    barras = ax.barh(tabla["categoria"], tabla["dmsp_media_ponderada"], color=colores, height=0.62)
    ax.bar_label(barras, fmt="%.1f", padding=6, fontsize=10.5, color=INK_PRIMARIO, fontweight="medium")

    # mediana como marcador secundario, para dejar ver que el patron no
    # depende de valores atipicos (misma relacion monotonica en ambas metricas)
    ax.scatter(tabla["dmsp_mediana"], tabla["categoria"], marker="|", s=380,
               color=INK_PRIMARIO, linewidths=2.2, zorder=5, label="Mediana")

    ax.set_xlim(0, tabla["dmsp_media_ponderada"].max() * 1.28)
    ax.set_title(titulo, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("DMSP-OLS -- iluminación nocturna (0-63)")
    ax.tick_params(axis="y", labelsize=10.5)


def main() -> None:
    monetaria = pd.read_csv(TABLES_DIR / "dmsp_por_categoria_monetaria_2010_2013.csv")
    ipm = pd.read_csv(TABLES_DIR / "dmsp_por_categoria_ipm_2010_2013.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    graficar_panel(axes[0], monetaria, "Pobreza monetaria")
    graficar_panel(axes[1], ipm, "Pobreza multidimensional (IPM)")

    handles = [plt.Line2D([0], [0], marker="|", linestyle="", color=INK_PRIMARIO,
                           markersize=14, markeredgewidth=2.2, label="Mediana")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=9.5, ncol=1)

    fig.suptitle(
        "La iluminación nocturna aumenta de forma monotónica entre los 4 grupos de transición\n"
        "(barra = media ponderada; línea = mediana), 2010 → 2013",
        fontsize=12.5, y=1.06,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "dmsp_monotonico_categoria.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
