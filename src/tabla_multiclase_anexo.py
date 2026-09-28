"""
tabla_multiclase_anexo.py
=====================================
Tablas del anexo del modelo multiclase (pedido del usuario, 2026-09-28):
la Seccion 5.2 (Tercer hallazgo) cita sus AUC por par de grupos, pero
ningun anexo los reportaba. Solo la especificacion que usa el texto:
Modelo B (sin ingreso ni gasto), holdout temporal (entrenamiento
2010->2013, prueba 2013->2016), sin variables geoespaciales. El Modelo A
se omite porque el ingreso de la ola base define "entra" frente a "sale"
y su AUC de ese par (~1.0) es una tautologia. Los hiperparametros de
ambos modelos ya estan en `tab_hiperparametros.tex`; no se repiten aqui.

INPUTS

    data/processed/benchmark_resultados/multiclase/registro_modelos_multiclase.csv
        (metricas: promedio de 5 semillas)
    data/processed/benchmark_resultados/multiclase/auc_pares_multiclase.csv
        (AUC por par, semilla 42; lo genera
        `src/05_model/auc_pares_multiclase_predicciones.py`)

OUTPUTS

    paper/tables/tab_multiclase_metricas.tex
    paper/tables/tab_multiclase_auc_pares.tex

COMO CORRER

    python src/tabla_multiclase_anexo.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MULTICLASE = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "multiclase"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"
ESPECIFICACION = "B4"

ALGORITMOS = {
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "Random Forest": "Random Forest",
    "Red neuronal (MLP)": "Red neuronal (MLP)",
    "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
}

METRICAS = [
    ("accuracy_media", r"\shortstack{Exactitud}"),
    ("balanced_accuracy_media", r"\shortstack{Exactitud\\balanceada}"),
    ("f1_macro_media", r"\shortstack{F1\\macro}"),
    ("auc_ovr_macro_media", r"\shortstack{AUC-ROC\\macro}"),
    ("auc_entra_vs_resto_media", r"\shortstack{AUC entra\\vs.\ resto}"),
    ("recall_entra_media", r"\shortstack{Recall\\entra}"),
    ("precision_top10_entra_media", r"\shortstack{Prec.-top10\\entra}"),
]

PARES = [
    ("auc_nunca_pobre_vs_siempre_pobre", r"\shortstack{Nunca vs.\\siempre}"),
    ("auc_nunca_pobre_vs_entra", r"\shortstack{Nunca vs.\\entra}"),
    ("auc_nunca_pobre_vs_sale", r"\shortstack{Nunca vs.\\sale}"),
    ("auc_entra_vs_siempre_pobre", r"\shortstack{Entra vs.\\siempre}"),
    ("auc_sale_vs_siempre_pobre", r"\shortstack{Sale vs.\\siempre}"),
    ("auc_entra_vs_sale", r"\shortstack{\textbf{Entra vs.}\\\textbf{sale}}"),
]


def _cargar(nombre: str, especificacion: str = ESPECIFICACION) -> pd.DataFrame:
    df = pd.read_csv(MULTICLASE / nombre)
    df = df[df["especificacion"] == especificacion].copy()
    df["algoritmo"] = df["algoritmo"].map(ALGORITMOS)
    orden = list(ALGORITMOS.values())
    return df.sort_values("algoritmo", key=lambda s: s.map(orden.index))


def _tabla(caption: str, label: str, columnas: list, filas: list, nota: str) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        "  \\begin{tabular}{l" + "c" * len(columnas) + "}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & " + " & ".join(columnas) + r" \\",
        r"    \midrule",
        *filas,
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        f"    \\footnotesize \\textit{{Nota:}} {nota} Fuente: cálculos propios.",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lineas) + "\n"


def tabla_metricas(registro: pd.DataFrame) -> str:
    columnas = [r"\textbf{Balanceo}"] + [f"\\textbf{{{t}}}" for _, t in METRICAS]
    filas = [
        f"    {f['algoritmo']} & {f['balanceo_elegido']} & "
        + " & ".join(f"{f[c]:.3f}" for c, _ in METRICAS) + r" \\"
        for _, f in registro.iterrows()
    ]
    n_test = f"{int(registro['n_test'].iloc[0]):,}".replace(",", "{,}")
    nota = (
        "Modelo B (sin ingreso ni gasto), entrenado en 2010$\\to$2013 y "
        f"evaluado en 2013$\\to$2016 ($n={n_test}$ hogares de los cuatro "
        "grupos), promedio de 5 semillas. El modelo "
        "asigna a cada hogar el grupo más probable, sin umbral ajustable, de "
        "modo que casi nunca asigna ``entra'', el grupo más pequeño "
        "(\\emph{recall} cercano a cero); por eso se usa como diagnóstico y "
        "no como modelo principal. Prec.-top10 entra: fracción de hogares "
        "que efectivamente entran en pobreza entre el 10\\% con mayor "
        "probabilidad de ``entra''. Hiperparámetros en la "
        "Tabla~\\ref{tab:hiperparametros}."
    )
    return _tabla(
        "Desempeño del modelo multiclase (cuatro grupos) en el conjunto de prueba.",
        "tab:multiclase_metricas", columnas, filas, nota,
    )


def tabla_auc_pares(pares: pd.DataFrame, pares_dmsp: pd.DataFrame) -> str:
    """Panel superior: AUC por par del Modelo B. Panel inferior (agregado
    2026-09-28, pedido del usuario): cambio de ese AUC al agregar DMSP-OLS
    (especificacion B4geoDMSP menos B4), que la Seccion 5.3 usa para mostrar
    que la luz nocturna no ayuda a separar ningun par de grupos."""
    columnas = [f"\\textbf{{{t}}}" if "textbf" not in t else t for _, t in PARES]
    n_col = len(PARES) + 1
    filas = [f"    \\multicolumn{{{n_col}}}{{l}}{{\\textit{{AUC-ROC, Modelo B}}}} \\\\"]
    filas += [
        f"    {f['algoritmo']} & " + " & ".join(f"{f[c]:.3f}" for c, _ in PARES) + r" \\"
        for _, f in pares.iterrows()
    ]
    delta = pares_dmsp.set_index("algoritmo")[[c for c, _ in PARES]] - pares.set_index("algoritmo")[[c for c, _ in PARES]]
    filas += [r"    \addlinespace", f"    \\multicolumn{{{n_col}}}{{l}}{{\\textit{{Cambio al agregar DMSP-OLS}}}} \\\\"]
    filas += [
        f"    {alg} & " + " & ".join(f"{fila[c]:+.3f}" for c, _ in PARES) + r" \\"
        for alg, fila in delta.loc[pares["algoritmo"]].iterrows()
    ]
    nota = (
        "AUC-ROC de cada par de grupos: se restringe a los hogares de los dos "
        "grupos y se usa como puntaje la probabilidad que el modelo asigna al "
        "primero. Modelo B, conjunto de prueba 2013$\\to$2016, una sola "
        "corrida (semilla 42). El modelo separa bien los extremos (nunca "
        "frente a siempre pobre), pero casi no separa a quienes entran de "
        "quienes salen de la pobreza. El panel inferior es la diferencia entre "
        "el mismo modelo con las 23 variables de DMSP-OLS y sin ellas."
    )
    return _tabla(
        "AUC-ROC del modelo multiclase por par de grupos de la matriz de transición.",
        "tab:multiclase_auc_pares", columnas, filas, nota,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    salidas = {
        "tab_multiclase_metricas.tex": tabla_metricas(_cargar("registro_modelos_multiclase.csv")),
        "tab_multiclase_auc_pares.tex": tabla_auc_pares(
            _cargar("auc_pares_multiclase.csv"), _cargar("auc_pares_multiclase.csv", "B4geoDMSP")),
    }
    for nombre, tex in salidas.items():
        (OUTPUT_DIR / nombre).write_text(tex, encoding="utf-8")
        print(f"Tabla exportada: {OUTPUT_DIR / nombre}")


if __name__ == "__main__":
    main()
