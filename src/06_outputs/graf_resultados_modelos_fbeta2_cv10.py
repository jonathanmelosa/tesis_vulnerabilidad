"""
Version de `graf_resultados_modelos.py` que lee
`registro_modelos_fbeta2_cv10.csv` (umbral por F-beta beta=2, CV_FOLDS=10,
N_ITER_BUSQUEDA=30 -- ver `modelo_fbeta2_cv10_comparacion.py`) en vez de
`registro_modelos.csv`, para las cuatro graficas de la Seccion "Desempeno
comparativo de modelos":

    01_auc_roc_por_algoritmo.png       (registro transversal)
    02_recall_precision_umbral.png     (registro transversal)
    03_importancia_random_forest_A.png (fbeta2_cv10/random_forest/importancia_variables_modelo_A.csv)
    04_coeficientes_logistica_A.png    (fbeta2_cv10/logistica_regularizada/coeficientes_modelo_A.csv)

03/04 requieren haber corrido antes `modelo_fbeta2_cv10_importancia.py`
(genera esos CSV reconstruyendo el pipeline ganador ya registrado, sin
volver a buscar hiperparametros -- ver su docstring para el detalle de
reproducibilidad).

Output: outputs/figures/modelos/ (sobreescribe 01, 02, 03 y 04)
"""

from pathlib import Path

import pandas as pd

import graf_resultados_modelos as base

REGISTRO_PATH_FBETA2_CV10 = (
    base.PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_fbeta2_cv10.csv"
)


def main() -> None:
    base.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    registro = pd.read_csv(REGISTRO_PATH_FBETA2_CV10)

    base.graf_auc_por_algoritmo(registro)
    base.graf_metricas_umbral(registro, ylim_max=1.02)

    base.graf_importancia_variables(
        carpeta="fbeta2_cv10/random_forest", archivo="importancia_variables_modelo_A.csv",
        columna_valor="importancia",
        titulo="Random Forest (Modelo A): variables más importantes",
        nombre_salida="03_importancia_random_forest_A.png",
    )
    base.graf_importancia_variables(
        carpeta="fbeta2_cv10/logistica_regularizada", archivo="coeficientes_modelo_A.csv",
        columna_valor="coeficiente",
        titulo="Logística regularizada (Modelo A): mayores coeficientes (estandarizados)",
        nombre_salida="04_coeficientes_logistica_A.png",
    )

    print(f"\nFiguras 01-04 regeneradas (F-beta=2, CV10) en: {base.FIGURES_DIR}")


if __name__ == "__main__":
    main()
