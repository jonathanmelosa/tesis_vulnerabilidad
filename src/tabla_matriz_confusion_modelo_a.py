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
    data/processed/benchmark_resultados/matriz_confusion_modelo_a.csv
    data/processed/benchmark_resultados/matriz_confusion_modelo_a.xlsx
        (el resultado combinado en sí, como dato independiente del LaTeX --
        agregado 2026-09-18 a pedido del usuario: el .tex por si solo no
        bastaba para que el resultado combinado fuera reproducible como
        DATO, solo como texto ya formateado)
    paper/tables/tab_matriz_confusion_modelo_a.tex (formatea el CSV de
        arriba, no vuelve a leer los registros originales)

COMO CORRER
    python src/tabla_matriz_confusion_modelo_a.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRO_MONETARIA = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_fbeta2_cv10.csv"
REGISTRO_IPM = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_ipm.csv"
RESULTADOS_DIR = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
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


def construir_matriz_combinada() -> pd.DataFrame:
    """El resultado combinado en sí (una fila por algoritmo, columnas
    VP/FP/FN/VN de monetaria e IPM) -- separado de la generación del
    LaTeX para poder guardarlo como CSV/Excel, no solo como texto ya
    formateado."""
    monetaria = _cargar(REGISTRO_MONETARIA, "A")
    ipm = _cargar(REGISTRO_IPM, "Aipm")
    orden_algoritmos = monetaria.sort_values("auc_roc_media", ascending=False).index.tolist()

    filas = []
    for algoritmo in orden_algoritmos:
        m, i = monetaria.loc[algoritmo], ipm.loc[algoritmo]
        filas.append({
            "algoritmo": algoritmo,
            "vp_monetaria": int(m["tp_semilla42"]), "fp_monetaria": int(m["fp_semilla42"]),
            "fn_monetaria": int(m["fn_semilla42"]), "vn_monetaria": int(m["tn_semilla42"]),
            "vp_ipm": int(i["tp_semilla42"]), "fp_ipm": int(i["fp_semilla42"]),
            "fn_ipm": int(i["fn_semilla42"]), "vn_ipm": int(i["tn_semilla42"]),
        })
    return pd.DataFrame(filas)


def main() -> None:
    matriz = construir_matriz_combinada()

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_csv = RESULTADOS_DIR / "matriz_confusion_modelo_a.csv"
    ruta_xlsx = RESULTADOS_DIR / "matriz_confusion_modelo_a.xlsx"
    matriz.to_csv(ruta_csv, index=False)
    matriz.to_excel(ruta_xlsx, index=False, sheet_name="Matriz de confusión Modelo A")
    print(f"Guardado: {ruta_csv}")
    print(f"Guardado: {ruta_xlsx}")

    orden_algoritmos = matriz["algoritmo"].tolist()

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
    for _, fila in matriz.iterrows():
        lineas.append(
            f"    {fila['algoritmo']:<24s} & {fila['vp_monetaria']} & {fila['fp_monetaria']} & "
            f"{fila['fn_monetaria']} & {fila['vn_monetaria']} & "
            f"{fila['vp_ipm']} & {fila['fp_ipm']} & "
            f"{fila['fn_ipm']} & {fila['vn_ipm']} \\\\"
        )
    lineas += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tab_matriz_confusion_modelo_a.tex"
    out_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado: {out_path}")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
