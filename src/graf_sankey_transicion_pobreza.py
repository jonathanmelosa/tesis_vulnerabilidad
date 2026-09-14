"""
graf_sankey_transicion_pobreza.py
===================================
Diagrama aluvial (Sankey de 2 columnas) de la transicion de pobreza
monetaria por ingreso entre 2010 y 2013: permite "hacer zoom" sobre el
universo de No pobres 2010 (cuantos se quedan No pobres vs. cuantos entran
en pobreza) y sobre el universo de Pobres 2010 (cuantos salen vs. cuantos
se quedan pobres), con n y % rotulados directamente sobre cada franja.

Es una vista alternativa a la barra apilada de
`build_pobreza_desagregaciones.py::graf_transiciones` (misma fuente de
datos y misma paleta categorica) -- la barra apilada muestra bien el
tamano relativo de las 4 categorias en el total, pero no deja ver de forma
inmediata que "Sale de la pobreza" y "Siempre pobre" son ambos subconjuntos
del universo Pobre-2010, ni que "Nunca pobre" y "Entra en pobreza" son
subconjuntos de No pobre-2010. El diagrama aluvial hace ese origen/destino
explicito.

DATOS (reales, panel emparejado 1 a 1, excluye hogares divididos entre
olas -- misma fuente que la Tabla "Matrices de transicion de pobreza
monetaria por ingreso" del documento). Genera AMBOS periodos:
    outputs/tables/pobreza/transicion_conteo_ola1_a_2.csv  (2010 -> 2013)
    outputs/tables/pobreza/transicion_pct_ola1_a_2.csv
    outputs/tables/pobreza/transicion_conteo_ola2_a_3.csv  (2013 -> 2016)
    outputs/tables/pobreza/transicion_pct_ola2_a_3.csv

OUTPUTS
    outputs/figures/pobreza/sankey_transicion_pobreza_2010_2013.png
    outputs/figures/pobreza/sankey_transicion_pobreza_2013_2016.png

COMO CORRER
    python src/graf_sankey_transicion_pobreza.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "pobreza"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures" / "pobreza"

# Misma paleta categorica que build_pobreza_desagregaciones.py / graf_dmsp_monotonico_categoria.py
COLOR_CATEGORIA = {
    "Nunca pobre": "#2a78d6",
    "Sale de la pobreza": "#1baf7a",
    "Entra en pobreza": "#eb6834",
    "Siempre pobre": "#e34948",
}
COLOR_NODO = {"No pobre": "#8a97a8", "Pobre": "#a87d7c"}
INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.size": 10,
    "text.color": INK_PRIMARIO,
    "axes.edgecolor": INK_SECUNDARIO,
})


def cargar_matriz(sufijo_ola: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conteo = pd.read_csv(TABLES_DIR / f"transicion_conteo_{sufijo_ola}.csv", index_col=0)
    pct = pd.read_csv(TABLES_DIR / f"transicion_pct_{sufijo_ola}.csv", index_col=0)
    return conteo, pct


def ribbon_path(x0: float, x1: float, y0_top: float, y0_bot: float,
                 y1_top: float, y1_bot: float) -> MplPath:
    """Franja curva (bezier cubica) entre un tramo vertical [y0_bot,y0_top]
    en x0 y un tramo vertical [y1_bot,y1_top] en x1."""
    xm = (x0 + x1) / 2
    verts = [
        (x0, y0_top), (xm, y0_top), (xm, y1_top), (x1, y1_top),
        (x1, y1_bot), (xm, y1_bot), (xm, y0_bot), (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def graf_sankey(conteo: pd.DataFrame, pct: pd.DataFrame, ano_inicial: int, ano_final: int,
                nombre_archivo: str) -> None:
    n_np_np = int(conteo.loc["No pobre", "No pobre"])   # Nunca pobre
    n_np_p = int(conteo.loc["No pobre", "Pobre"])         # Entra en pobreza
    n_p_np = int(conteo.loc["Pobre", "No pobre"])         # Sale de la pobreza
    n_p_p = int(conteo.loc["Pobre", "Pobre"])             # Siempre pobre

    pct_np_np = pct.loc["No pobre", "No pobre"]
    pct_np_p = pct.loc["No pobre", "Pobre"]
    pct_p_np = pct.loc["Pobre", "No pobre"]
    pct_p_p = pct.loc["Pobre", "Pobre"]

    n_total = n_np_np + n_np_p + n_p_np + n_p_p
    n_np_ini = n_np_np + n_np_p
    n_p_ini = n_p_np + n_p_p
    n_np_fin = n_np_np + n_p_np
    n_p_fin = n_np_p + n_p_p

    # --- Coordenadas verticales (columna izquierda = ano_inicial, derecha = ano_final) ---
    # Izquierda: bloque "No pobre" arriba, "Pobre" abajo.
    y_np_ini_top, y_np_ini_bot = n_total, n_total - n_np_ini
    y_p_ini_top, y_p_ini_bot = y_np_ini_bot, 0.0
    # Dentro de "No pobre inicial": se-queda arriba, entra-en-pobreza abajo.
    y_nunca_top, y_nunca_bot = y_np_ini_top, y_np_ini_top - n_np_np
    y_entra_top, y_entra_bot = y_nunca_bot, y_np_ini_bot
    # Dentro de "Pobre inicial": sale arriba, siempre-pobre abajo.
    y_sale_top, y_sale_bot = y_p_ini_top, y_p_ini_top - n_p_np
    y_siempre_top, y_siempre_bot = y_sale_bot, y_p_ini_bot

    # Derecha: bloque "No pobre final" arriba, "Pobre final" abajo.
    y_np_fin_top, y_np_fin_bot = n_total, n_total - n_np_fin
    y_p_fin_top, y_p_fin_bot = y_np_fin_bot, 0.0
    # Dentro de "No pobre final": llega-de-nunca-pobre arriba, llega-de-sale abajo.
    y_nunca_r_top, y_nunca_r_bot = y_np_fin_top, y_np_fin_top - n_np_np
    y_sale_r_top, y_sale_r_bot = y_nunca_r_bot, y_np_fin_bot
    # Dentro de "Pobre final": llega-de-entra arriba, llega-de-siempre abajo.
    y_entra_r_top, y_entra_r_bot = y_p_fin_top, y_p_fin_top - n_np_p
    y_siempre_r_top, y_siempre_r_bot = y_entra_r_bot, y_p_fin_bot

    x0, x1 = 0.12, 0.88
    fig, ax = plt.subplots(figsize=(9, 6.4))

    # t_label: posicion relativa (0=lado 2010, 1=lado 2013) donde se centra el
    # rotulo de cada franja. Las dos franjas que SE CRUZAN (Entra/Sale) se
    # rotulan cerca de su extremo de origen y destino, respectivamente, para
    # que los dos textos no queden superpuestos en el centro del cruce.
    # El % de cada franja es SIEMPRE respecto de su grupo de origen en 2010
    # (fila de la matriz de transicion) -- p.ej. "23.4%" de Entra en pobreza
    # es el 23.4% de los No pobres 2010, NO el 23.4% del total ni de los
    # Pobres 2013. Se deja explicito en el rotulo para que no se confunda.
    etq_np_ini = f"de No pobres {ano_inicial}"
    etq_p_ini = f"de Pobres {ano_inicial}"
    flujos = [
        ("Nunca pobre", n_np_np, pct_np_np, etq_np_ini, y_nunca_top, y_nunca_bot, y_nunca_r_top, y_nunca_r_bot, 0.5),
        ("Entra en pobreza", n_np_p, pct_np_p, etq_np_ini, y_entra_top, y_entra_bot, y_entra_r_top, y_entra_r_bot, 0.28),
        ("Sale de la pobreza", n_p_np, pct_p_np, etq_p_ini, y_sale_top, y_sale_bot, y_sale_r_top, y_sale_r_bot, 0.72),
        ("Siempre pobre", n_p_p, pct_p_p, etq_p_ini, y_siempre_top, y_siempre_bot, y_siempre_r_top, y_siempre_r_bot, 0.5),
    ]

    for nombre, n_val, pct_val, base_pct, yt0, yb0, yt1, yb1, t_label in flujos:
        color = COLOR_CATEGORIA[nombre]
        path = ribbon_path(x0, x1, yt0, yb0, yt1, yb1)
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=0.82, zorder=2))
        xm = x0 + t_label * (x1 - x0)
        mid0 = (yt0 + yb0) / 2
        mid1 = (yt1 + yb1) / 2
        ym = mid0 + t_label * (mid1 - mid0)
        ax.text(xm, ym, f"{nombre}\n{n_val:,} hogares\n({pct_val:.1f}% {base_pct})",
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold", zorder=3)

    # --- Nodos (barras) ---
    ancho_nodo = 0.035
    nodos = [
        (x0 - ancho_nodo, y_np_ini_bot, y_np_ini_top, "No pobre", n_np_ini, "left"),
        (x0 - ancho_nodo, y_p_ini_bot, y_p_ini_top, "Pobre", n_p_ini, "left"),
        (x1, y_np_fin_bot, y_np_fin_top, "No pobre", n_np_fin, "right"),
        (x1, y_p_fin_bot, y_p_fin_top, "Pobre", n_p_fin, "right"),
    ]
    for x, ybot, ytop, etiqueta, n_val, lado in nodos:
        color = COLOR_NODO[etiqueta]
        ax.add_patch(Rectangle((x, ybot), ancho_nodo, ytop - ybot,
                                facecolor=color, edgecolor=SURFACE, linewidth=1.5, zorder=4))
        x_texto = x - 0.01 if lado == "left" else x + ancho_nodo + 0.01
        ha = "right" if lado == "left" else "left"
        ax.text(x_texto, (ybot + ytop) / 2, f"{etiqueta}\n{n_val:,}",
                ha=ha, va="center", fontsize=10, color=INK_PRIMARIO, fontweight="bold")

    ax.text(x0 - ancho_nodo / 2, n_total * 1.04, str(ano_inicial), ha="center", fontsize=13,
            color=INK_PRIMARIO, fontweight="bold")
    ax.text(x1 + ancho_nodo / 2, n_total * 1.04, str(ano_final), ha="center", fontsize=13,
            color=INK_PRIMARIO, fontweight="bold")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-n_total * 0.02, n_total * 1.10)
    ax.axis("off")
    ax.set_title(
        f"Transicion de pobreza monetaria por ingreso, {ano_inicial} → {ano_final}\n"
        f"Metodologia Lopez-Calva y Ortiz-Juarez (2014) (n = {n_total:,} hogares)",
        fontsize=12, pad=14,
    )

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / nombre_archivo
    fig.savefig(ruta, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Guardado: {ruta}")


def main() -> None:
    conteo_1_2, pct_1_2 = cargar_matriz("ola1_a_2")
    graf_sankey(conteo_1_2, pct_1_2, 2010, 2013, "sankey_transicion_pobreza_2010_2013.png")

    conteo_2_3, pct_2_3 = cargar_matriz("ola2_a_3")
    graf_sankey(conteo_2_3, pct_2_3, 2013, 2016, "sankey_transicion_pobreza_2013_2016.png")


if __name__ == "__main__":
    main()
