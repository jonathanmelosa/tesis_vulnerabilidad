"""
graficar_shap.py
=====================================
Genera las graficas del analisis SHAP (beeswarm + barras de |SHAP| medio)
para las 4 combinaciones algoritmo x target ya analizadas en
`diagnostico_shap.py` -- pedido por el usuario (2026-08-31) porque ese
script solo habia exportado los CSV de importancia, no graficas.

Reutiliza literalmente `entrenar()` de `diagnostico_shap.py` (IMPORTADA,
no duplicada) para reconstruir cada modelo ganador con un solo fit
(mismos hiperparametros/balanceo ya encontrados, sin repetir la busqueda)
y recalcula los SHAP values sobre el mismo conjunto de prueba -- el
script original no los guardo en disco.

QUE HACE

    Para cada uno de los 4 pares (algoritmo x target):
    1. Reentrena el modelo ganador (via diagnostico_shap.entrenar).
    2. Recalcula SHAP values sobre x_test.
    3. Exporta un beeswarm (shap.summary_plot, dot) y un bar plot
       (shap.summary_plot, bar) de las top 20 variables.

INPUTS

    (los mismos que diagnostico_shap.py: registro_modelos_fbeta2_cv10.csv,
    registro_modelos_ipm.csv, benchmark_train_test/modelo_*.parquet)

OUTPUTS

    outputs/figures/modelos/shap_beeswarm_{algoritmo}_{target}.png
    outputs/figures/modelos/shap_barras_{algoritmo}_{target}.png

COMO CORRER

    cd src/05_model && python -u graficar_shap.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from diagnostico_shap import COMBINACIONES, REGISTROS, entrenar

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURAS_DIR = PROJECT_ROOT / "outputs" / "figures" / "modelos"

NOMBRE_ARCHIVO_ALGO = {"XGBoost": "xgboost", "HistGradientBoosting (sklearn)": "histgb"}
NOMBRE_ARCHIVO_TARGET = {"Monetaria": "monetaria", "IPM": "ipm"}


def graficar_combinacion(algoritmo_raw: str, espec: str, target: str) -> None:
    nombre_algo = "XGBoost" if algoritmo_raw == "XGBoost" else "HistGradientBoosting"
    print(f"\n=== {nombre_algo} -- {target} ({espec}) ===")

    registro = pd.read_csv(REGISTROS[target])
    modelo, x_test, y_test = entrenar(algoritmo_raw, espec, registro)

    print(f"  Calculando SHAP sobre {x_test.shape[0]} hogares de test, {x_test.shape[1]} variables...")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(x_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    sufijo = f"{NOMBRE_ARCHIVO_ALGO[algoritmo_raw]}_{NOMBRE_ARCHIVO_TARGET[target]}"

    plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type="dot", max_display=20, show=False)
    plt.title(f"SHAP (beeswarm) -- {nombre_algo}, target {target} ({espec})")
    plt.tight_layout()
    ruta_beeswarm = FIGURAS_DIR / f"shap_beeswarm_{sufijo}.png"
    plt.savefig(ruta_beeswarm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta_beeswarm}")

    plt.figure()
    shap.summary_plot(shap_values, x_test, plot_type="bar", max_display=20, show=False)
    plt.title(f"SHAP (|valor medio|) -- {nombre_algo}, target {target} ({espec})")
    plt.tight_layout()
    ruta_barras = FIGURAS_DIR / f"shap_barras_{sufijo}.png"
    plt.savefig(ruta_barras, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta_barras}")


def main() -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    for algoritmo_raw, espec, target in COMBINACIONES:
        graficar_combinacion(algoritmo_raw, espec, target)
    print(f"\nListo. Graficas en: {FIGURAS_DIR}")


if __name__ == "__main__":
    main()
