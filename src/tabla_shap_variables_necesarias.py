"""
tabla_shap_variables_necesarias.py
=====================================
Tabla para la Sección de importancia de variables (Sección~5.3 de
`main.tex`): de las ~90 variables originales del benchmark, cuántas bastan
para capturar la mayor parte de la señal predictiva medida por SHAP, por
algoritmo y especificación (A/B). Lee el CSV ya agregado por
`src/05_model/diagnostico_shap_variables_necesarias.py` (que resume las
columnas one-hot/missingindicator de vuelta a su variable original antes
de calcular la masa acumulada) -- este script solo formatea, no calcula.

INPUTS

    data/processed/benchmark_resultados/diagnostico_shap_variables_necesarias.csv

OUTPUTS

    paper/tables/tab_shap_variables_necesarias.tex

COMO CORRER

    python src/tabla_shap_variables_necesarias.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "entrada": REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_shap_variables_necesarias.csv",
    "nombres_algoritmo": {
        "Random Forest": "Random Forest",
        "XGBoost": "XGBoost",
        "HistGradientBoosting": "HistGradientBoosting",
        "LightGBM": "LightGBM",
        "Logistica": "Logística regularizada",
    },
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar(cfg: dict) -> pd.DataFrame:
    ruta = Path(cfg["entrada"])
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta} -- correr primero "
              f"src/05_model/diagnostico_shap_variables_necesarias.py", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ruta)
    df["algoritmo"] = df["algoritmo"].map(cfg["nombres_algoritmo"]).fillna(df["algoritmo"])
    return df


def formatear_fila(fila: pd.Series) -> str:
    return (
        f"    {fila['algoritmo']:<24s} & {fila['especificacion']} & "
        f"{fila['n_variables_originales']:d} & "
        f"{fila['pct_shap_top10']*100:.1f}\\% & "
        f"{fila['n_variables_para_80pct']:d} \\\\"
    )


def generar_tex(df: pd.DataFrame) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Concentración de la señal predictiva (SHAP), variables",
        r"  originales agregadas (dummies \emph{one-hot} e indicadores de",
        r"  faltante sumados a su variable de origen antes de calcular la",
        r"  masa acumulada).}",
        r"  \label{tab:shap_variables_necesarias}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Espec.} & \textbf{Variables (total)} & "
        r"\textbf{\% SHAP en top-10} & \textbf{Variables para 80\% del SHAP} \\",
        r"    \midrule",
    ]
    especificaciones = sorted(df["especificacion"].unique())
    for i, espec in enumerate(especificaciones):
        bloque = df[df["especificacion"] == espec].sort_values("n_variables_para_80pct")
        for _, fila in bloque.iterrows():
            lineas.append(formatear_fila(fila))
        if i < len(especificaciones) - 1:
            lineas.append(r"    \addlinespace")
    lineas += [r"    \bottomrule", r"  \end{tabular}", r"  }"]
    return "\n".join(lineas)


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = cargar(cfg)
    tex = generar_tex(df)
    ruta_tex = out_dir / "tab_shap_variables_necesarias.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada (cuerpo de tabular; encabezado table/caption/nota siguen a mano en main.tex): {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
