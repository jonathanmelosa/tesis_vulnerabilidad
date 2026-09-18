"""
tabla_matriz_confusion_modelo_a.py
=====================================
Tabla de matriz de confusión (Modelo A) para los cinco algoritmos,
comparando pobreza monetaria (Tabla~\\ref{tab:desempeno_modelos}) e IPM
(Tabla~\\ref{tab:desempeno_modelos_ipm}, Anexo~\\ref{apx:desempeno_ipm})
lado a lado -- pedido del usuario (2026-09-18): una fila por algoritmo,
columnas divididas por definición de pobreza. Cumple además la promesa
ya hecha en la Sección 4 ("se reporta... la matriz de confusión
completa", Sección~\\ref{subsec:validacion}), que antes no se cumplía en
ningún lado del documento. Usa las celdas ya calculadas (semilla 42,
misma corrida robusta) en `registro_modelos_fbeta2_cv10.csv` (monetaria)
y `registro_modelos_ipm.csv` (IPM) -- no se recalcula nada.

Solo VN/FP/FN/VP (sin los porcentajes derivados que tenía la versión
anterior de esta tabla -- esos números ya se citan en prosa en la
Sección~\\ref{subsec:desempeno} donde hacen falta, y aquí sobrecargaban
la tabla).

INPUTS
    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
    data/processed/benchmark_resultados/registro_modelos_ipm.csv

OUTPUTS
    paper/tables/tab_matriz_confusion_modelo_a.tex

COMO CORRER
    python src/tabla_matriz_confusion_modelo_a.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRO_MONETARIA = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_fbeta2_cv10.csv"
REGISTRO_IPM = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_ipm.csv"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

NOMBRES_ALGORITMO = {
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "LightGBM": "LightGBM",
    "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
}

CELDAS = ["tn_semilla42", "fp_semilla42", "fn_semilla42", "tp_semilla42"]


def _cargar(ruta: Path, especificacion: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df = df[df["especificacion"] == especificacion].copy()
    df["algoritmo"] = df["algoritmo"].map(NOMBRES_ALGORITMO).fillna(df["algoritmo"])
    return df.set_index("algoritmo")[CELDAS + ["auc_roc_media"]]


def main() -> None:
    monetaria = _cargar(REGISTRO_MONETARIA, "A")
    ipm = _cargar(REGISTRO_IPM, "Aipm")

    orden_algoritmos = monetaria.sort_values("auc_roc_media", ascending=False).index.tolist()

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Matriz de confusión, Modelo A, holdout temporal (test",
        r"  2013$\to$2016, semilla 42), pobreza monetaria ($n=3{,}191$) e",
        r"  IPM ($n=5{,}571$). VN/FP/FN/VP: hogares verdadero negativo, falso",
        r"  positivo, falso negativo y verdadero positivo, al umbral de",
        r"  clasificación de las Tablas~\ref{tab:desempeno_modelos} y",
        r"  \ref{tab:desempeno_modelos_ipm} respectivamente.}",
        r"  \label{tab:matriz_confusion_a}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lcccccccc}",
        r"    \toprule",
        r"     & \multicolumn{4}{c}{\textbf{Monetaria}} & \multicolumn{4}{c}{\textbf{IPM}} \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"    \textbf{Algoritmo} & \textbf{VP} & \textbf{FP} & \textbf{FN} & \textbf{VN} & "
        r"\textbf{VP} & \textbf{FP} & \textbf{FN} & \textbf{VN} \\",
        r"    \midrule",
    ]
    for algoritmo in orden_algoritmos:
        m, i = monetaria.loc[algoritmo], ipm.loc[algoritmo]
        lineas.append(
            f"    {algoritmo:<24s} & {int(m['tp_semilla42'])} & {int(m['fp_semilla42'])} & "
            f"{int(m['fn_semilla42'])} & {int(m['tn_semilla42'])} & "
            f"{int(i['tp_semilla42'])} & {int(i['fp_semilla42'])} & "
            f"{int(i['fn_semilla42'])} & {int(i['tn_semilla42'])} \\\\"
        )
    lineas += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tab_matriz_confusion_modelo_a.tex"
    out_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado: {out_path}")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
