"""
graf_matriz_confusion_modelo_a_pct.py
=====================================
Version en imagen (PNG) de la matriz de confusion en porcentajes (Modelo A,
monetaria + IPM, % de fila) que en el paper se ve como
`paper/tables/tab_matriz_confusion_modelo_a_pct.tex` -- misma estructura que
`graf_matriz_confusion_modelo_a.py` (la version en conteos), pero leyendo el
CSV de porcentajes y redondeando cada celda a numero entero (sin decimales),
igual que en la tabla LaTeX.

Lee `matriz_confusion_modelo_a_pct.csv` -- generado por
`src/tabla_matriz_confusion_modelo_a_pct.py`, que a su vez es la unica
fuente de este dato (no se recalcula nada aqui).

INPUTS
    data/processed/benchmark_resultados/matriz_confusion_modelo_a_pct.csv

OUTPUTS
    outputs/figures/modelos/tab_matriz_confusion_modelo_a_pct.png

COMO CORRER
    cd src/06_outputs && python graf_matriz_confusion_modelo_a_pct.py
    (correr DESPUES de `python src/tabla_matriz_confusion_modelo_a_pct.py`,
    que es quien genera el CSV de entrada)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUTA_CSV = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "matriz_confusion_modelo_a_pct.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "modelos"

INK_PRIMARIO = "#0b0b0b"
GRIS_BORDE = "#c9c9c9"
GRIS_ENCABEZADO = "#e9e9e9"

COLUMNAS = [
    ("vp_monetaria_pct", "Mon.\nVP"), ("fp_monetaria_pct", "Mon.\nFP"),
    ("fn_monetaria_pct", "Mon.\nFN"), ("vn_monetaria_pct", "Mon.\nVN"),
    ("vp_ipm_pct", "IPM\nVP"), ("fp_ipm_pct", "IPM\nFP"),
    ("fn_ipm_pct", "IPM\nFN"), ("vn_ipm_pct", "IPM\nVN"),
]


def graf_matriz_confusion_pct(matriz: pd.DataFrame) -> None:
    n_filas = len(matriz)

    fig, ax = plt.subplots(figsize=(9.5, 0.55 * n_filas + 1.1))
    ax.axis("off")

    celdas = [["Algoritmo"] + [etq for _, etq in COLUMNAS]]
    for _, fila in matriz.iterrows():
        celdas.append([fila["algoritmo"]] + [f"{round(fila[col])}%" for col, _ in COLUMNAS])

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
        "Pobreza monetaria (n=3,191) e IPM (n=5,571) — % de fila",
        fontsize=11, color=INK_PRIMARIO, pad=10,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / "tab_matriz_confusion_modelo_a_pct.png"
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {ruta}")


def main() -> None:
    if not RUTA_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró {RUTA_CSV} -- correr primero "
            f"src/tabla_matriz_confusion_modelo_a_pct.py"
        )
    matriz = pd.read_csv(RUTA_CSV)
    graf_matriz_confusion_pct(matriz)


if __name__ == "__main__":
    main()
