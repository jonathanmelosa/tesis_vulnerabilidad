"""
tabla_matriz_confusion_modelo_a_pct.py
=====================================
Versión en porcentajes (agregados por fila) de la Tabla~\\ref{tab:matriz_confusion_a}
(tabla_matriz_confusion_modelo_a.py) -- pedido del usuario (2026-09-22): mismo
layout, pero cada celda VP/FP/FN/VN expresada como % del total de esa
definición de pobreza (monetaria o IPM) para ese algoritmo, en vez de conteos
absolutos. No recalcula nada: lee el CSV ya generado por el script de
conteos.

INPUTS
    data/processed/benchmark_resultados/matriz_confusion_modelo_a.csv

OUTPUTS
    data/processed/benchmark_resultados/matriz_confusion_modelo_a_pct.csv
    data/processed/benchmark_resultados/matriz_confusion_modelo_a_pct.xlsx
    paper/tables/tab_matriz_confusion_modelo_a_pct.tex

COMO CORRER
    python src/tabla_matriz_confusion_modelo_a_pct.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTEOS_CSV = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "matriz_confusion_modelo_a.csv"
RESULTADOS_DIR = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

CELDAS = ["vp", "fp", "fn", "vn"]


def construir_matriz_pct() -> pd.DataFrame:
    conteos = pd.read_csv(CONTEOS_CSV)
    filas = []
    for _, fila in conteos.iterrows():
        registro = {"algoritmo": fila["algoritmo"]}
        for definicion in ["monetaria", "ipm"]:
            total = sum(fila[f"{c}_{definicion}"] for c in CELDAS)
            for c in CELDAS:
                registro[f"{c}_{definicion}_pct"] = 100 * fila[f"{c}_{definicion}"] / total
        filas.append(registro)
    return pd.DataFrame(filas)


def main() -> None:
    matriz = construir_matriz_pct()

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_csv = RESULTADOS_DIR / "matriz_confusion_modelo_a_pct.csv"
    ruta_xlsx = RESULTADOS_DIR / "matriz_confusion_modelo_a_pct.xlsx"
    matriz.to_csv(ruta_csv, index=False)
    matriz.to_excel(ruta_xlsx, index=False, sheet_name="Matriz confusión Modelo A (%)")
    print(f"Guardado: {ruta_csv}")
    print(f"Guardado: {ruta_xlsx}")

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Matriz de confusión, Modelo A, holdout temporal (test",
        r"  2013$\to$2016, semilla 42), pobreza monetaria ($n=3{,}191$) e",
        r"  IPM ($n=5{,}571$), en \% de fila (VN+FP+FN+VP suma 100\% dentro de",
        r"  cada definición de pobreza). VN/FP/FN/VP: hogares verdadero",
        r"  negativo, falso positivo, falso negativo y verdadero positivo, al",
        r"  umbral de clasificación de las Tablas~\ref{tab:desempeno_modelos} y",
        r"  \ref{tab:desempeno_modelos_ipm} respectivamente.}",
        r"  \label{tab:matriz_confusion_a_pct}",
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
    for _, fila in matriz.iterrows():
        lineas.append(
            f"    {fila['algoritmo']:<24s} & {fila['vp_monetaria_pct']:.1f}\\% & "
            f"{fila['fp_monetaria_pct']:.1f}\\% & {fila['fn_monetaria_pct']:.1f}\\% & "
            f"{fila['vn_monetaria_pct']:.1f}\\% & {fila['vp_ipm_pct']:.1f}\\% & "
            f"{fila['fp_ipm_pct']:.1f}\\% & {fila['fn_ipm_pct']:.1f}\\% & "
            f"{fila['vn_ipm_pct']:.1f}\\% \\\\"
        )
    lineas += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tab_matriz_confusion_modelo_a_pct.tex"
    out_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado: {out_path}")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
