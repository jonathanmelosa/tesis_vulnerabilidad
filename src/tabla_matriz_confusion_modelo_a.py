"""
tabla_matriz_confusion_modelo_a.py
=====================================
Tabla de matriz de confusión (Modelo A, pobreza monetaria, holdout
2013->2016) para los cinco algoritmos de la Tabla~\\ref{tab:desempeno_modelos}
-- pedido del usuario (2026-09-17): la Seccion 4 ya promete "la matriz de
confusion completa" (Seccion~\\ref{subsec:validacion}), pero esa matriz
nunca aparecia en ningun lado del documento. Usa las celdas ya calculadas
(semilla 42, misma corrida robusta) en
`registro_modelos_fbeta2_cv10.csv` -- no se recalcula nada.

Ademas de VN/FP/FN/VP, calcula dos cifras derivadas que hacen el
trade-off recall/precision tangible en terminos de un programa de
focalizacion real:
  - `pct_muestra_marcada`: (FP+VP)/n_test -- que fraccion de TODA la
    muestra de prueba el modelo marcaria como "en riesgo".
  - `pct_seguros_bien_identificados`: VN/(VN+FP) -- que fraccion de los
    hogares que en realidad NO caen en pobreza el modelo excluye
    correctamente.

INPUTS
    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv

OUTPUTS
    paper/tables/tab_matriz_confusion_modelo_a.tex

COMO CORRER
    python src/tabla_matriz_confusion_modelo_a.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRO = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_fbeta2_cv10.csv"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

NOMBRES_ALGORITMO = {
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "LightGBM": "LightGBM",
    "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
}


def main() -> None:
    df = pd.read_csv(REGISTRO)
    df = df[df["especificacion"] == "A"].copy()
    df["algoritmo"] = df["algoritmo"].map(NOMBRES_ALGORITMO).fillna(df["algoritmo"])
    df["pct_muestra_marcada"] = (df["fp_semilla42"] + df["tp_semilla42"]) / df["n_test"]
    df["pct_seguros_identificados"] = df["tn_semilla42"] / (df["tn_semilla42"] + df["fp_semilla42"])
    df = df.sort_values("auc_roc_media", ascending=False)

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Matriz de confusión, Modelo A, holdout temporal (test",
        r"  2013$\to$2016, $n=3{,}191$, semilla 42). VN/FP/FN/VP: hogares",
        r"  verdadero negativo, falso positivo, falso negativo y verdadero",
        r"  positivo al umbral de clasificación de la Tabla~\ref{tab:desempeno_modelos}.}",
        r"  \label{tab:matriz_confusion_a}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lcccccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{VN} & \textbf{FP} & \textbf{FN} & \textbf{VP} & "
        r"\textbf{\% muestra} & \textbf{\% seguros} \\",
        r"     & & & & & \textbf{marcada} & \textbf{identificados} \\",
        r"    \midrule",
    ]
    for _, fila in df.iterrows():
        lineas.append(
            f"    {fila['algoritmo']:<24s} & {int(fila['tn_semilla42'])} & {int(fila['fp_semilla42'])} & "
            f"{int(fila['fn_semilla42'])} & {int(fila['tp_semilla42'])} & "
            f"{fila['pct_muestra_marcada']*100:.1f}\\% & {fila['pct_seguros_identificados']*100:.1f}\\% \\\\"
        )
    lineas += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tab_matriz_confusion_modelo_a.tex"
    out_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado: {out_path}")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
