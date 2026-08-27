"""
tabla_marginal_dmsp.py
=====================================
Genera la tabla de contribución marginal de DMSP-OLS (AUC-ROC,
precision_top10, y las métricas de referencia al umbral de clasificación
elegido -- recall, precision, F1 --, con y sin DMSP-OLS, para los tres
algoritmos con mejor desempeño del benchmark; Sección "Contribución
marginal de las variables geoespaciales", Tabla~\\ref{tab:marginal_dmsp}
de la tesis) a partir del registro de resultados de todos los modelos
entrenados -- no reentrena nada, solo filtra y formatea filas ya calculadas por
`src/05_model/modelo_{xgboost,histgradientboosting,logistica_regularizada}.py`
sobre los datasets `modelo_{A,B,AgeoDMSP,BgeoDMSP}_2010_2013/2013_2016.parquet`
(construidos por `src/05_model/construir_pipeline_geo_dmsp.py`).

QUÉ HACE

    1. Carga `registro_modelos.csv`.
    2. Filtra a los 3 algoritmos comparados (XGBoost, HistGradientBoosting,
       Logística regularizada) y a las 4 especificaciones relevantes (A,
       AgeoDMSP, B, BgeoDMSP).
    3. Para cada algoritmo, empareja A con A+DMSP-OLS y B con B+DMSP-OLS
       en filas consecutivas, en el orden ya usado en la tesis (XGBoost,
       HistGradientBoosting, Logística regularizada; A antes que B).
    4. Exporta una tabla .tex lista para incorporar al documento.

INPUTS

    data/processed/benchmark_resultados/registro_modelos.csv

OUTPUTS

    paper/tables/tab_marginal_dmsp.tex

CÓMO CORRER

    python src/tabla_marginal_dmsp.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "registro_modelos": REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos.csv",
    # Orden exacto de la tesis: algoritmo (en el orden de mejor a peor
    # desempeño en el benchmark principal) x especificacion base (A, B).
    "algoritmos_orden": ["XGBoost", "HistGradientBoosting (sklearn)", "Logistica regularizada (elastic net, benchmark)"],
    "nombres_algoritmo": {
        "XGBoost": "XGBoost",
        "HistGradientBoosting (sklearn)": "HistGradientBoosting",
        "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
    },
    "pares_especificacion": [("A", "AgeoDMSP"), ("B", "BgeoDMSP")],
    "etiqueta_especificacion": {"A": "A", "AgeoDMSP": "A + DMSP-OLS", "B": "B", "BgeoDMSP": "B + DMSP-OLS"},
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar_registro(cfg: dict) -> pd.DataFrame:
    ruta = Path(cfg["registro_modelos"])
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(ruta)


def fila_tex(df: pd.DataFrame, algoritmo_raw: str, espec: str, cfg: dict) -> str:
    fila = df[(df["algoritmo"] == algoritmo_raw) & (df["especificacion"] == espec)]
    if fila.empty:
        print(f"ERROR: no hay fila para algoritmo={algoritmo_raw!r}, especificacion={espec!r}", file=sys.stderr)
        sys.exit(1)
    fila = fila.iloc[0]
    nombre = cfg["nombres_algoritmo"][algoritmo_raw]
    etiqueta = cfg["etiqueta_especificacion"][espec]
    return (
        f"    {nombre:<24s} & {etiqueta:<15s} & {fila['auc_roc_media']:.3f} & "
        f"[{fila['auc_roc_ci95_low']:.3f}, {fila['auc_roc_ci95_high']:.3f}] & "
        f"{fila['precision_top10_media']:.3f} & "
        f"[{fila['precision_top10_ci95_low']:.3f}, {fila['precision_top10_ci95_high']:.3f}] & "
        f"{fila['recall_media']:.3f} & {fila['precision_media']:.3f} & {fila['f1_media']:.3f} \\\\"
    )


def generar_tex(df: pd.DataFrame, cfg: dict) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{AUC-ROC y precisión en el decil de mayor riesgo, con y sin",
        r"  DMSP-OLS, holdout temporal (train 2010$\to$2013, test 2013$\to$2016).}",
        r"  \label{tab:marginal_dmsp}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llccccccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Especificación} & \textbf{AUC-ROC} & \textbf{IC95\%} & \textbf{Precision top-10\%} & \textbf{IC95\%} & \textbf{Recall} & \textbf{Precision} & \textbf{F1} \\",
        r"    \midrule",
    ]
    # Orden exacto de la tesis: agrupado primero por especificacion base
    # (todas las filas "A" de los 3 algoritmos, luego todas las "B"), no
    # por algoritmo -- ver Tabla~\ref{tab:marginal_dmsp} en main.tex.
    for i, (espec_base, espec_geo) in enumerate(cfg["pares_especificacion"]):
        for j, algoritmo_raw in enumerate(cfg["algoritmos_orden"]):
            lineas.append(fila_tex(df, algoritmo_raw, espec_base, cfg))
            lineas.append(fila_tex(df, algoritmo_raw, espec_geo, cfg))
            if j < len(cfg["algoritmos_orden"]) - 1:
                lineas.append(r"    \addlinespace")
        if i < len(cfg["pares_especificacion"]) - 1:
            lineas.append(r"    \addlinespace")
    lineas += [r"    \bottomrule", r"  \end{tabular}%", r"  }"]
    return "\n".join(lineas)


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = cargar_registro(cfg)
    tex = generar_tex(df, cfg)
    ruta_tex = out_dir / "tab_marginal_dmsp.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada (cuerpo de tabular; encabezado table/caption/nota siguen a mano en main.tex): {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
