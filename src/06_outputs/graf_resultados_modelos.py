"""
Graficas de resultados de la suite de comparacion de algoritmos del
benchmark (logistica regularizada, Random Forest, XGBoost, LightGBM,
HistGradientBoosting) para el paper (ver docs/decisions.md, "Suite de
comparacion de algoritmos", y `src/05_model/`).

Lee `data/processed/benchmark_resultados/registro_modelos.csv` (registro
transversal, una fila por algoritmo x especificacion) y la carpeta de
importancia de variables del modelo con mayor AUC-ROC en la especificacion
A para graficar sus predictores principales. Paleta y estilo replican los
usados en `src/04_features/build_pobreza_desagregaciones.py` (dataviz
skill del proyecto).

Output: outputs/figures/modelos/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRO_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos.csv"
RESULTADOS_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "modelos"

NOMBRES_CORTOS = {
    "Logistica regularizada (elastic net, benchmark)": "Logística\n(benchmark)",
    "Random Forest": "Random\nForest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "HistGradientBoosting (sklearn)": "HistGradient\nBoosting",
}
ORDEN_ALGORITMOS = [
    "Logistica regularizada (elastic net, benchmark)", "Random Forest",
    "XGBoost", "LightGBM", "HistGradientBoosting (sklearn)",
]

PALETA = {
    "azul": "#2a78d6",
    "naranja": "#eb6834",
    "aguamarina": "#1baf7a",
    "amarillo": "#eda100",
    "magenta": "#e87ba4",
    "verde": "#008300",
    "violeta": "#4a3aa7",
    "rojo": "#e34948",
}
INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECUNDARIO,
    "text.color": INK_PRIMARIO,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _guardar(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    ruta = FIGURES_DIR / nombre
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"Guardado: {ruta}")


def graf_auc_por_algoritmo(registro: pd.DataFrame) -> None:
    """AUC-ROC de cada algoritmo, agrupado por especificacion (A/B)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(ORDEN_ALGORITMOS))
    ancho = 0.35
    colores = {"A": PALETA["azul"], "B": PALETA["naranja"]}
    etiquetas = {"A": "Modelo A (con ingreso/gasto)", "B": "Modelo B (sin ingreso/gasto)"}

    for i, espec in enumerate(["A", "B"]):
        sub = registro[registro["especificacion"] == espec].set_index("algoritmo").reindex(ORDEN_ALGORITMOS)
        offset = (i - 0.5) * ancho
        barras = ax.bar(x + offset, sub["auc_roc"], width=ancho, color=colores[espec], label=etiquetas[espec])
        for rect, valor in zip(barras, sub["auc_roc"]):
            ax.annotate(f"{valor:.3f}", (rect.get_x() + rect.get_width() / 2, valor),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=INK_SECUNDARIO)

    ax.set_xticks(x)
    ax.set_xticklabels([NOMBRES_CORTOS[a] for a in ORDEN_ALGORITMOS], fontsize=9)
    ax.set_ylim(0.6, 0.82)
    ax.set_ylabel("AUC-ROC (test, 2013→2016)")
    ax.set_title("Desempeño de los 5 algoritmos por especificación")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    _guardar(fig, "01_auc_roc_por_algoritmo.png")


def graf_metricas_umbral(registro: pd.DataFrame) -> None:
    """Recall y precision (al umbral elegido por CV) por algoritmo, especificacion A."""
    sub = registro[registro["especificacion"] == "A"].set_index("algoritmo").reindex(ORDEN_ALGORITMOS)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(ORDEN_ALGORITMOS))
    ancho = 0.35
    ax.bar(x - ancho / 2, sub["recall"], width=ancho, color=PALETA["aguamarina"], label="Recall")
    ax.bar(x + ancho / 2, sub["precision"], width=ancho, color=PALETA["violeta"], label="Precision")
    for i, (r, p) in enumerate(zip(sub["recall"], sub["precision"])):
        ax.annotate(f"{r:.2f}", (x[i] - ancho / 2, r), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=INK_SECUNDARIO)
        ax.annotate(f"{p:.2f}", (x[i] + ancho / 2, p), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=INK_SECUNDARIO)
    ax.set_xticks(x)
    ax.set_xticklabels([NOMBRES_CORTOS[a] for a in ORDEN_ALGORITMOS], fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Proporción")
    ax.set_title("Recall y precision al umbral elegido por validación cruzada (Modelo A)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    _guardar(fig, "02_recall_precision_umbral.png")


def graf_importancia_variables(carpeta: str, archivo: str, columna_valor: str, titulo: str, nombre_salida: str, n_top: int = 12) -> None:
    """Top-N variables mas importantes de un modelo (importancia o |coeficiente|)."""
    ruta = RESULTADOS_DIR / carpeta / archivo
    df = pd.read_csv(ruta)
    df = df.reindex(df[columna_valor].abs().sort_values(ascending=False).index).head(n_top)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    colores = [PALETA["rojo"] if v < 0 else PALETA["azul"] for v in df[columna_valor]]
    ax.barh(df["variable"], df[columna_valor], color=colores)
    ax.set_xlabel(columna_valor.replace("_", " ").capitalize())
    ax.set_title(titulo, fontsize=10.5)
    _guardar(fig, nombre_salida)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    registro = pd.read_csv(REGISTRO_PATH)

    graf_auc_por_algoritmo(registro)
    graf_metricas_umbral(registro)

    graf_importancia_variables(
        carpeta="random_forest", archivo="importancia_variables_modelo_A.csv",
        columna_valor="importancia",
        titulo="Random Forest (Modelo A): variables más importantes",
        nombre_salida="03_importancia_random_forest_A.png",
    )
    graf_importancia_variables(
        carpeta="logistica_regularizada", archivo="coeficientes_modelo_A.csv",
        columna_valor="coeficiente",
        titulo="Logística regularizada (Modelo A): mayores coeficientes (estandarizados)",
        nombre_salida="04_coeficientes_logistica_A.png",
    )

    print(f"\nTodas las graficas guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
