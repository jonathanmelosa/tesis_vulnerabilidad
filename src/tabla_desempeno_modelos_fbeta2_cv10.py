"""
tabla_desempeno_modelos_fbeta2_cv10.py
=====================================
Version de `tabla_desempeno_modelos.py` que lee el registro generado por
`src/05_model/modelo_fbeta2_cv10_comparacion.py`
(`registro_modelos_fbeta2_cv10.csv`) -- mismo contenido (cinco algoritmos x
Modelos A/B, holdout temporal), pero con el criterio de umbral robusto
adoptado para la version final de la tesis: F-beta (beta=2, en vez de F1)
elegido por CV con CV_FOLDS=10 y N_ITER_BUSQUEDA=30 (en vez de 3/8), ver
docstring de `src/05_model/modelo_fbeta2_cv10_comparacion.py`.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv

OUTPUTS

    paper/tables/tab_desempeno_modelos_fbeta2_cv10.tex

COMO CORRER

    python src/tabla_desempeno_modelos_fbeta2_cv10.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estilo_tablas import FUENTE, aplicar_signo_menos  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FRACCION_TOP_K = 0.10  # igual que modelo_utils.FRACCION_TOP_K

CONFIG = {
    "registro_modelos": REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_fbeta2_cv10.csv",
    "especificaciones_benchmark": ["A", "B"],
    "nombres_algoritmo": {
        "Random Forest": "Random Forest",
        "XGBoost": "XGBoost",
        "HistGradientBoosting (sklearn)": "HistGradientBoosting",
        "LightGBM": "LightGBM",
        "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
    },
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar_registro(cfg: dict) -> pd.DataFrame:
    ruta = Path(cfg["registro_modelos"])
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta} -- correr primero src/05_model/modelo_fbeta2_cv10_comparacion.py", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ruta)
    df = df[df["especificacion"].isin(cfg["especificaciones_benchmark"])].copy()
    df["algoritmo"] = df["algoritmo"].map(cfg["nombres_algoritmo"]).fillna(df["algoritmo"])
    return df


def agregar_recall_top10(df: pd.DataFrame) -> pd.DataFrame:
    """Recall en el 10% de mayor riesgo, derivado sin reentrenar (agregado
    2026-09-28, pedido del usuario): de los hogares que entran en pobreza,
    que fraccion queda dentro del decil atendido. El decil tiene
    ceil(0.1 * n_test) hogares (misma regla que `precision_top_k` en
    `modelo_utils.py`) y el conjunto de prueba es fijo entre semillas, asi
    que recall_top10 = precision_top10 * n_top / positivos es exacto
    tambien sobre el promedio de semillas."""
    df = df.copy()
    n_top = np.ceil(df["n_test"] * FRACCION_TOP_K)
    positivos = df["tp_semilla42"] + df["fn_semilla42"]
    df["recall_top10_media"] = df["precision_top10_media"] * n_top / positivos
    return df


def formatear_fila(fila: pd.Series) -> str:
    return (
        f"    {fila['algoritmo']:<26s} & {fila['especificacion']} & "
        f"{fila['umbral_clasificacion_media']:.2f} & {fila['auc_roc_media']:.3f} & "
        f"{fila['recall_media']:.3f} & {fila['precision_media']:.3f} & "
        f"{fila['precision_top10_media']:.3f} & {fila['recall_top10_media']:.3f} & "
        f"{fila['f1_media']:.3f} \\\\"
    )


def generar_tex(df: pd.DataFrame, cfg: dict) -> str:
    lineas = [
        r"\begin{table}[!htbp]",  # [H] dejaba espacio en blanco al final de pagina (2026-09-28)
        r"  \centering",
        # Titulo corto (2026-09-25, pedido del usuario): holdout, umbral F2 y
        # parametros de CV ya se explican en el texto (Secciones 4.4 y 4.5)
        # y en el parrafo que introduce la tabla.
        r"  \caption{Desempeño de los cinco algoritmos en el conjunto de prueba,",
        r"  pobreza monetaria, 2013$\to$2016.}",
        r"  \label{tab:desempeno_modelos}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        # Con la columna Rec.-top10 (2026-09-28) la tabla excede el ancho de
        # texto; se escala igual que la matriz de confusion.
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llccccccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Espec.} & \textbf{Umbral} & \textbf{AUC-ROC} & \textbf{\emph{Recall}} & \textbf{Precisión} & \textbf{Prec.-top10} & \textbf{Rec.-top10} & \textbf{F1} \\",
        r"    \midrule",
    ]
    especificaciones = cfg["especificaciones_benchmark"]
    for i, espec in enumerate(especificaciones):
        bloque = df[df["especificacion"] == espec].sort_values("auc_roc_media", ascending=False)
        for _, fila in bloque.iterrows():
            lineas.append(formatear_fila(fila))
        if i < len(especificaciones) - 1:
            lineas.append(r"    \addlinespace")
    # Nota y cierre del entorno dentro del .tex (2026-10-01): antes el
    # \end{table} vivia en main.tex y las definiciones iban en el titulo.
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} Promedio de 5 semillas. Espec.: especificación",
        r"    (A con ingreso y gasto del hogar, B sin ellos). Umbral: elegido por",
        r"    validación cruzada maximizando F$_2$. Prec.-top10: precisión en el",
        r"    10\% de hogares con mayor riesgo predicho; Rec.-top10: fracción de",
        r"    los hogares que entran en pobreza que queda dentro de ese 10\%. " + FUENTE,
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return aplicar_signo_menos("\n".join(lineas))


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = agregar_recall_top10(cargar_registro(cfg))
    tex = generar_tex(df, cfg)
    ruta_tex = out_dir / "tab_desempeno_modelos_fbeta2_cv10.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
