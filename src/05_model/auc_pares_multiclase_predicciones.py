"""
auc_pares_multiclase_predicciones.py
=====================================
AUC de los 6 pares de grupos de la matriz de transicion (nunca pobre,
entra, sale, siempre pobre) a partir de las probabilidades por hogar ya
guardadas por `modelo_multiclase_predicciones.py`, sin reentrenar nada.

Motivo (Seccion 5.2 de `main.tex`, Tercer hallazgo): el registro multiclase
solo reporta `auc_entra_vs_sale` (~0.53-0.60 en el Modelo B, casi azar).
Ese numero solo dice algo si se compara con el resto de pares: si TODOS
los pares dieran cerca de 0.55, el modelo simplemente no distinguiria
grupos; si los demas pares dan mucho mas, entonces la frontera entra/sale
es especificamente la que las covariables de la ola base no separan --
apoyo multivariado (independiente del perfil univariado) de que "entra" y
"sale" se parecen.

Definicion (misma que `calcular_metricas_multiclase`,
`modelo_utils_multiclase.py`): para el par (a, b), se restringe a los
hogares de a o b; score = P(a); AUC con a como clase positiva. Es un AUC
por par; no depende de que a sea "mas pobre" que b.

Limitaciones: (i) una sola semilla (42, el ajuste canonico de
`modelo_multiclase_predicciones.py`), sin IC; los IC de 5 semillas para
`entra_vs_sale` estan en el registro. (ii) Las especificaciones de holdout
(A4, B4, ..., geoDMSP) son evaluadas sobre 2013->2016 (hogares que el
modelo no vio); las *geo3 son out-of-fold sobre 2010->2013. (iii) Modelo A
incluye el ingreso de la ola base, que define entra frente a sale, asi
que sus AUC de ese par (~1.0) son una tautologia, no una señal.

INPUTS

    data/processed/benchmark_resultados/multiclase/predicciones/*.parquet

OUTPUT

    data/processed/benchmark_resultados/multiclase/auc_pares_multiclase.csv

COMO CORRER

    cd src/05_model && python auc_pares_multiclase_predicciones.py
"""

import itertools

import pandas as pd
from sklearn.metrics import roc_auc_score

import modelo_utils as mu

DIR_PRED = mu.RESULTADOS_DIR / "multiclase" / "predicciones"
RUTA_SALIDA = mu.RESULTADOS_DIR / "multiclase" / "auc_pares_multiclase.csv"
GRUPOS = ["nunca_pobre", "entra", "sale", "siempre_pobre"]
PROB = {"nunca_pobre": "prob_nunca_pobre", "entra": "prob_entra", "sale": "prob_sale", "siempre_pobre": "prob_siempre_pobre"}


def auc_par(df: pd.DataFrame, a: str, b: str) -> float:
    sub = df[df["Y_grupo"].isin([a, b])]
    if sub["Y_grupo"].nunique() < 2:
        return float("nan")
    return roc_auc_score((sub["Y_grupo"] == a).astype(int), sub[PROB[a]])


def main() -> None:
    filas = []
    for ruta in sorted(DIR_PRED.glob("*.parquet")):
        df = pd.read_parquet(ruta)
        fila = {"algoritmo": df["algoritmo"].iloc[0], "especificacion": df["especificacion"].iloc[0], "n": len(df)}
        for a, b in itertools.combinations(GRUPOS, 2):
            fila[f"auc_{a}_vs_{b}"] = round(auc_par(df, a, b), 4)
        filas.append(fila)
    res = pd.DataFrame(filas)
    res.to_csv(RUTA_SALIDA, index=False)

    columnas = [c for c in res.columns if c.startswith("auc_")]
    print(res[res["especificacion"].isin(["B4", "B4geoDMSP"])].round(3).to_string(index=False))
    print("\nMediana entre algoritmos, por especificacion:")
    print(res.groupby("especificacion")[columnas].median().round(3).T.to_string())
    print(f"\nGuardado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
