"""
tabla_matriz_confusion_modelo_a.py
=====================================
Tablas de matriz de confusión (Modelo A en el texto; Modelo B en un
anexo, agregado 2026-09-25) para los cinco algoritmos,
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
    data/processed/benchmark_resultados/matriz_confusion_modelo_b.{csv,xlsx}
    paper/tables/tab_matriz_confusion_modelo_b.tex

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


# Cambio 2026-09-25 (pedido del usuario): se genera tambien la matriz del
# Modelo B, en el mismo formato, para un anexo. Las dos notas remiten al
# umbral y los hiperparametros del Anexo de hiperparametros (la semilla ya
# no se menciona en el documento).
ESPECIFICACIONES = {
    # clave: (espec. monetaria, espec. IPM, etiqueta en el titulo)
    "a": ("A", "Aipm", "A"),
    "b": ("B", "Bipm", "B"),
}


def construir_matriz_combinada(espec_mon: str = "A", espec_ipm: str = "Aipm") -> pd.DataFrame:
    """El resultado combinado en sí (una fila por algoritmo, columnas
    VP/FP/FN/VN de monetaria e IPM) -- separado de la generación del
    LaTeX para poder guardarlo como CSV/Excel, no solo como texto ya
    formateado."""
    monetaria = _cargar(REGISTRO_MONETARIA, espec_mon)
    ipm = _cargar(REGISTRO_IPM, espec_ipm)
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


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def generar_tex(matriz: pd.DataFrame, clave: str, etiqueta: str) -> str:
    n_mon = int(matriz[["vp_monetaria", "fp_monetaria", "fn_monetaria", "vn_monetaria"]].iloc[0].sum())
    n_ipm = int(matriz[["vp_ipm", "fp_ipm", "fn_ipm", "vn_ipm"]].iloc[0].sum())

    def celda(fila, col: str, total: int) -> str:
        return f"{fila[col]} ({100 * fila[col] / total:.0f}\\%)"

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        f"  \\caption{{Matriz de confusión del Modelo {etiqueta} en el conjunto de prueba",
        r"  (2013$\to$2016), pobreza monetaria e IPM.}",
        f"  \\label{{tab:matriz_confusion_{clave}}}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lcccccccc}",
        r"    \toprule",
        r"     & \multicolumn{4}{c}{\textbf{Monetaria}} & \multicolumn{4}{c}{\textbf{IPM}} \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"    \textbf{Algoritmo} & \textbf{VP} & \textbf{FP} & \textbf{FN} & \textbf{VN} & "
        r"\textbf{VP} & \textbf{FP} & \textbf{FN} & \textbf{VN} \\",
        r"    \midrule",
    ]
    for _, fila in matriz.iterrows():
        mon = " & ".join(celda(fila, f"{c}_monetaria", n_mon) for c in ("vp", "fp", "fn", "vn"))
        ipm = " & ".join(celda(fila, f"{c}_ipm", n_ipm) for c in ("vp", "fp", "fn", "vn"))
        lineas.append(f"    {fila['algoritmo']:<24s} & {mon} & {ipm} \\\\")
    remision_b = (
        r" La matriz del Modelo B se reporta en la Tabla~\ref{tab:matriz_confusion_b}"
        r" del Anexo~\ref{apx:confusion_b}."
        if clave == "a" else ""
    )
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} VP, FP, FN y VN: hogares verdadero",
        r"    positivo, falso positivo, falso negativo y verdadero negativo. Entre",
        r"    paréntesis, porcentaje sobre el total de la fila para cada definición",
        f"    de pobreza (monetaria, $n={_miles(n_mon)}$; IPM, $n={_miles(n_ipm)}$). Modelos",
        r"    clasificados con los hiperparámetros y el umbral reportados en la",
        r"    Tabla~\ref{tab:hiperparametros} del Anexo~\ref{apx:hiperparametros}." + remision_b,
        r"    Fuente: cálculos propios.",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lineas) + "\n"


def main() -> None:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for clave, (espec_mon, espec_ipm, etiqueta) in ESPECIFICACIONES.items():
        matriz = construir_matriz_combinada(espec_mon, espec_ipm)
        ruta_csv = RESULTADOS_DIR / f"matriz_confusion_modelo_{clave}.csv"
        matriz.to_csv(ruta_csv, index=False)
        matriz.to_excel(RESULTADOS_DIR / f"matriz_confusion_modelo_{clave}.xlsx", index=False,
                        sheet_name=f"Matriz de confusión Modelo {etiqueta}")
        out_path = OUTPUT_DIR / f"tab_matriz_confusion_modelo_{clave}.tex"
        out_path.write_text(generar_tex(matriz, clave, etiqueta), encoding="utf-8")
        print(f"Guardado: {ruta_csv}\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
