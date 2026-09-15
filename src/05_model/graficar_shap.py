"""
graficar_shap.py
=====================================
Genera las graficas del analisis SHAP (beeswarm + barras de |SHAP| medio)
para TODAS las combinaciones algoritmo x especificacion ya analizadas en
`diagnostico_shap.py` (especificaciones CON DMSP-OLS, `AgeoDMSP`/
`AipmgeoDMSP`, targets Monetaria/IPM) y en `diagnostico_shap_ab.py`
(especificaciones PURAS `A`/`B`, target Monetaria).

ARREGLADO (fecha de esta sesion): la version anterior de este script
importaba `COMBINACIONES`/`REGISTROS` de `diagnostico_shap.py` y asumia
que `entrenar()` devolvia `(modelo, x_test, y_test)` -- ninguna de las
dos cosas es cierta desde que `diagnostico_shap.py` se amplio a 5
algoritmos (`AMPLIADO`, ver su docstring): ahora expone `TARGETS` (no
`COMBINACIONES`/`REGISTROS`) y `entrenar()` devuelve
`(pipe, x_train, x_test, y_test, cat_cols)` (5 valores). El script
quedaba roto (`ImportError`) y las 8 figuras `shap_*_{xgboost,histgb}_
{monetaria,ipm}.png` que habia en `outputs/figures/modelos/` eran de una
corrida con una version anterior del codigo -- se regeneraron desde cero
con este script arreglado.

Reutiliza literalmente `entrenar()`/`calcular_shap()` de
`diagnostico_shap.py` (IMPORTADAS, no duplicadas) para reconstruir cada
modelo ganador con un solo fit (mismos hiperparametros/balanceo ya
encontrados, sin repetir la busqueda) y recalcular los SHAP values sobre
el mismo conjunto de prueba -- ninguno de los dos scripts de diagnostico
los guarda en disco. `calcular_shap()` ahora devuelve tambien la matriz
`x_test_shap` (la que el explainer usa de verdad: nativa para arbol_
nativo, o la salida del preprocesador para preprocesador_clasico) --
antes este script recalculaba eso por su cuenta con `shap.TreeExplainer`
directo sobre `x_test` crudo, lo cual habria fallado silenciosamente
para Random Forest/Logistica (necesitan pasar por el preprocesador) si
alguna vez se hubiera ampliado sin arreglar tambien esta parte.

QUE HACE

    1. Para cada (algoritmo, especificacion, target) de `diagnostico_
       shap.py` (CON DMSP-OLS -- hasta 5 algoritmos x 2 targets = 10
       combinaciones) y de `diagnostico_shap_ab.py` (SIN DMSP-OLS,
       Modelos A/B puros -- hasta 5 algoritmos x 2 especificaciones = 10
       combinaciones):
       a. Reentrena el modelo ganador (via `diagnostico_shap.entrenar`).
       b. Recalcula SHAP values + X transformado (via `diagnostico_shap.
          calcular_shap`).
       c. Exporta un beeswarm (`shap.summary_plot`, dot) y un bar plot
          (`shap.summary_plot`, bar) de las top 20 variables.

INPUTS

    (los mismos que diagnostico_shap.py / diagnostico_shap_ab.py:
    registro_modelos_fbeta2_cv10.csv, registro_modelos_ipm.csv,
    benchmark_train_test/modelo_*.parquet)

OUTPUTS

    outputs/figures/modelos/shap_beeswarm_{algoritmo}_{sufijo}.png
    outputs/figures/modelos/shap_barras_{algoritmo}_{sufijo}.png
    (sufijo = "monetaria"/"ipm" para las combinaciones CON DMSP-OLS,
    "A"/"B" para los Modelos puros)

COMO CORRER

    cd src/05_model && python -u graficar_shap.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from algoritmos_suite import algoritmos_presentes_en_registro, resolver_algoritmo
from diagnostico_shap import TARGETS, calcular_shap, entrenar
from diagnostico_shap_ab import ESPECIFICACIONES as ESPECIFICACIONES_AB
from diagnostico_shap_ab import REGISTRO as REGISTRO_AB

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURAS_DIR = PROJECT_ROOT / "outputs" / "figures" / "modelos"

# Slug de archivo por algoritmo -- deriva de resolver_algoritmo(...)
# ["nombre_bonito"], no se mantiene una lista aparte que se pueda
# desincronizar si se agrega un algoritmo nuevo a la suite.
def _slug(nombre_bonito: str) -> str:
    return nombre_bonito.lower().replace(" ", "")


def graficar_combinacion(algoritmo_raw: str, espec: str, registro: pd.DataFrame, sufijo: str) -> None:
    nombre_algo = resolver_algoritmo(algoritmo_raw)["nombre_bonito"]
    print(f"\n=== {nombre_algo} -- {sufijo} ({espec}) ===")

    pipe, x_train, x_test, y_test, cat_cols = entrenar(algoritmo_raw, espec, registro)
    print(f"  Calculando SHAP sobre {x_test.shape[0]} hogares de test, {x_test.shape[1]} variables...")
    shap_values, nombres, x_test_shap = calcular_shap(algoritmo_raw, pipe, x_train, x_test, cat_cols)

    slug = f"{_slug(nombre_algo)}_{sufijo}"

    plt.figure()
    shap.summary_plot(shap_values, x_test_shap, feature_names=list(nombres), plot_type="dot", max_display=20, show=False)
    plt.title(f"SHAP (beeswarm) -- {nombre_algo} ({espec})")
    plt.tight_layout()
    ruta_beeswarm = FIGURAS_DIR / f"shap_beeswarm_{slug}.png"
    plt.savefig(ruta_beeswarm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta_beeswarm}")

    plt.figure()
    shap.summary_plot(shap_values, x_test_shap, feature_names=list(nombres), plot_type="bar", max_display=20, show=False)
    plt.title(f"SHAP (|valor medio|) -- {nombre_algo} ({espec})")
    plt.tight_layout()
    ruta_barras = FIGURAS_DIR / f"shap_barras_{slug}.png"
    plt.savefig(ruta_barras, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {ruta_barras}")


def main() -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    sufijo_target = {"Monetaria": "monetaria", "IPM": "ipm"}
    for target, cfg in TARGETS.items():
        registro = pd.read_csv(cfg["registro"])
        espec = cfg["espec_con_dmsp"]
        for algoritmo_raw in algoritmos_presentes_en_registro(cfg["registro"]):
            existe = ((registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)).any()
            if not existe:
                print(f"\n=== {algoritmo_raw} -- {target} ({espec}) === OMITIDO: sin fila en el registro")
                continue
            graficar_combinacion(algoritmo_raw, espec, registro, sufijo_target[target])

    registro_ab = pd.read_csv(REGISTRO_AB)
    for espec in ESPECIFICACIONES_AB:
        for algoritmo_raw in algoritmos_presentes_en_registro(REGISTRO_AB):
            existe = ((registro_ab.algoritmo == algoritmo_raw) & (registro_ab.especificacion == espec)).any()
            if not existe:
                print(f"\n=== {algoritmo_raw} -- Modelo {espec} === OMITIDO: sin fila en el registro")
                continue
            graficar_combinacion(algoritmo_raw, espec, registro_ab, espec)

    print(f"\nListo. Graficas en: {FIGURAS_DIR}")


if __name__ == "__main__":
    main()
