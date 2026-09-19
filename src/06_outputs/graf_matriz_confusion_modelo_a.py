"""
graf_matriz_confusion_modelo_a.py
=====================================
Version en imagen (PNG) de la matriz de confusion combinada (Modelo A,
monetaria + IPM) que en el paper se ve como el Cuadro~6
(`paper/tables/tab_matriz_confusion_modelo_a.tex`) -- pedido explicito del
usuario (2026-09-18): la tabla debe poder existir tambien como imagen,
pero generada por script a partir del mismo dato ya guardado, no
recortada a mano del PDF compilado (eso no seria reproducible si la tabla
cambia de pagina o de formato).

Lee `matriz_confusion_modelo_a.csv` -- generado por
`src/tabla_matriz_confusion_modelo_a.py`, que a su vez es la unica fuente
de este dato (no se recalcula nada aqui) -- y dibuja una tabla con
matplotlib replicando la estructura del Cuadro 6: una fila por algoritmo,
columnas VP/FP/FN/VN agrupadas por Monetaria e IPM. El encabezado usa una
sola fila con etiquetas "Mon./IPM + VP/FP/FN/VN" en vez de un encabezado
de dos niveles fusionado -- mas simple y sin coordenadas ajustadas a mano
que se desalinearian si cambia el numero de filas.

INPUTS
    data/processed/benchmark_resultados/matriz_confusion_modelo_a.csv

OUTPUTS
    outputs/figures/modelos/tab_matriz_confusion_modelo_a.png

COMO CORRER
    cd src/06_outputs && python graf_matriz_confusion_modelo_a.py
    (correr DESPUES de `python src/tabla_matriz_confusion_modelo_a.py`,
    que es quien genera el CSV de entrada)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUTA_CSV = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "matriz_confusion_modelo_a.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "modelos"

INK_PRIMARIO = "#0b0b0b"
GRIS_BORDE = "#c9c9c9"
GRIS_ENCABEZADO = "#e9e9e9"

COLUMNAS = [
    ("vp_monetaria", "Mon.\nVP"), ("fp_monetaria", "Mon.\nFP"),
    ("fn_monetaria", "Mon.\nFN"), ("vn_monetaria", "Mon.\nVN"),
    ("vp_ipm", "IPM\nVP"), ("fp_ipm", "IPM\nFP"),
    ("fn_ipm", "IPM\nFN"), ("vn_ipm", "IPM\nVN"),
]


def graf_matriz_confusion(matriz: pd.DataFrame) -> None:
    n_filas = len(matriz)

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * n_filas + 1.1))
    ax.axis("off")

    celdas = [["Algoritmo"] + [etq for _, etq in COLUMNAS]]
    for _, fila in matriz.iterrows():
        celdas.append([fila["algoritmo"]] + [str(int(fila[col])) for col, _ in COLUMNAS])

    tabla = ax.table(cellText=celdas, cellLoc="center", bbox=[0, 0, 1, 1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10.5)

    ancho_algoritmo = 0.22
    ancho_dato = (1 - ancho_algoritmo) / len(COLUMNAS)
    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor(GRIS_BORDE)
        celda.set_width(ancho_algoritmo if col == 0 else ancho_dato)
        if fila == 0:
            celda.set_facecolor(GRIS_ENCABEZADO)
            celda.set_text_props(weight="bold", color=INK_PRIMARIO)
        else:
            celda.set_text_props(color=INK_PRIMARIO)
            celda.set_facecolor("#f7f7f7" if fila % 2 == 0 else "white")

    ax.set_title(
        "Matriz de confusión, Modelo A, holdout temporal (test 2013→2016, semilla 42)\n"
        "Pobreza monetaria (n=3,191) e IPM (n=5,571)",
        fontsize=11, color=INK_PRIMARIO, pad=10,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / "tab_matriz_confusion_modelo_a.png"
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {ruta}")


def main() -> None:
    if not RUTA_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_CSV} -- correr primero "
            f"src/tabla_matriz_confusion_modelo_a.py"
        )
    matriz = pd.read_csv(RUTA_CSV)
    graf_matriz_confusion(matriz)


if __name__ == "__main__":
    main()
