"""
tabla_desempeno_modelos_ipm.py
=====================================
Version de `tabla_desempeno_modelos_fbeta2_cv10.py` (que NO se modifica)
para el benchmark bajo pobreza multidimensional (IPM) -- pedido del
usuario (2026-09-17), para sostener con una tabla propia (en el anexo,
citada desde la Sección~\\ref{subsec:desempeno}) los numeros de IPM ya
citados en el texto principal (rango de AUC-ROC, brecha Aipm/Bipm).

Misma configuracion robusta que el benchmark monetario (CV_FOLDS=10,
N_ITER_BUSQUEDA=30, 5 semillas) -- confirmado en las `observaciones` del
propio registro, comparabilidad directa entre targets por diseno (ver
docstring de `modelo_utils.py` / `build_benchmark_train_test.py`).

INPUTS
    data/processed/benchmark_resultados/registro_modelos_ipm.csv

OUTPUTS
    paper/tables/tab_desempeno_modelos_ipm.tex

COMO CORRER
    python src/tabla_desempeno_modelos_ipm.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRO = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_ipm.csv"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

ESPECIFICACIONES = ["Aipm", "Bipm"]
NOMBRES_ALGORITMO = {
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "LightGBM": "LightGBM",
    "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
}


def formatear_fila(fila: pd.Series) -> str:
    return (
        f"    {fila['algoritmo']:<24s} & {fila['especificacion'].replace('ipm', '')} & "
        f"{fila['umbral_clasificacion_media']:.2f} & {fila['auc_roc_media']:.3f} & "
        f"{fila['recall_media']:.3f} & {fila['precision_media']:.3f} & "
        f"{fila['precision_top10_media']:.3f} & {fila['f1_media']:.3f} \\\\"
    )


def main() -> None:
    df = pd.read_csv(REGISTRO)
    df = df[df["especificacion"].isin(ESPECIFICACIONES)].copy()
    df["algoritmo"] = df["algoritmo"].map(NOMBRES_ALGORITMO).fillna(df["algoritmo"])

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Desempeño de los cinco algoritmos en el conjunto de prueba,",
        r"  pobreza multidimensional (IPM), 2013$\to$2016.}",
        r"  \label{tab:desempeno_modelos_ipm}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llcccccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Espec.} & \textbf{Umbral} & \textbf{AUC-ROC} & \textbf{\emph{Recall}} & \textbf{Precisión} & \textbf{Prec.-top10} & \textbf{F1} \\",
        r"    \midrule",
    ]
    for i, espec in enumerate(ESPECIFICACIONES):
        bloque = df[df["especificacion"] == espec].sort_values("auc_roc_media", ascending=False)
        for _, fila in bloque.iterrows():
            lineas.append(formatear_fila(fila))
        if i < len(ESPECIFICACIONES) - 1:
            lineas.append(r"    \addlinespace")
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Promedio de 5 semillas, entrenamiento en",
        r"    2010$\to$2013. Misma metodología y covariables que la",
        r"    Tabla~\ref{tab:desempeno_modelos}; solo cambia la variable de",
        r"    resultado. Espec.: especificación (A con ingreso y gasto del hogar,",
        r"    B sin ellos). Prec.-top10: precisión en el 10\% de hogares con mayor",
        r"    riesgo predicho. " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tab_desempeno_modelos_ipm.tex"
    out_path.write_text(aplicar_signo_menos("\n".join(lineas)) + "\n", encoding="utf-8")
    print(f"Guardado: {out_path}")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
