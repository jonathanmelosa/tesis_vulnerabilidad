"""
tabla_marginal_dmsp_fbeta2_cv10.py
=====================================
Version de `tabla_marginal_dmsp_fbeta2.py` que lee el registro generado por
`src/05_model/modelo_fbeta2_cv10_comparacion.py`
(`registro_modelos_fbeta2_cv10.csv`) -- mismo criterio de umbral (F-beta,
beta=2) que `tabla_marginal_dmsp_fbeta2.py`, pero con CV_FOLDS=10 y
N_ITER_BUSQUEDA=30 en vez de 3/8. Para comparar las tres tablas lado a lado
(F1/folds=3 -- tab_marginal_dmsp.tex; F-beta=2/folds=3 --
tab_marginal_dmsp_fbeta2.tex; F-beta=2/folds=10 -- este script) sin
modificar ninguna.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv

OUTPUTS

    paper/tables/tab_marginal_dmsp_fbeta2_cv10.tex

CÓMO CORRER

    python src/tabla_marginal_dmsp_fbeta2_cv10.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "registro_modelos": REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos_fbeta2_cv10.csv",
    "algoritmos_orden": [
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "HistGradientBoosting (sklearn)",
        "Logistica regularizada (elastic net, benchmark)",
    ],
    "nombres_algoritmo": {
        "Random Forest": "Random Forest",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
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
        print(f"ERROR: no se encontró {ruta} -- correr primero src/05_model/modelo_fbeta2_cv10_comparacion.py", file=sys.stderr)
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
        f"{fila['umbral_clasificacion_media']:.3f} & "
        f"{fila['recall_media']:.3f} & {fila['precision_media']:.3f} & {fila['f1_media']:.3f} \\\\"
    )


def generar_tex(df: pd.DataFrame, cfg: dict) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{AUC-ROC y precisión en el decil de mayor riesgo, con y sin",
        r"  DMSP-OLS, holdout temporal (train 2010$\to$2013, test 2013$\to$2016).",
        r"  Umbral elegido por CV maximizando F-beta ($\beta=2$), con",
        r"  CV\_FOLDS=10 y N\_ITER\_BUSQUEDA=30 -- comparar con",
        r"  Tabla~\ref{tab:marginal_dmsp} (F1, folds=3) y",
        r"  Tabla~\ref{tab:marginal_dmsp_fbeta2} (F-beta=2, folds=3).}",
        r"  \label{tab:marginal_dmsp_fbeta2_cv10}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llcccccccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Especificación} & \textbf{AUC-ROC} & \textbf{IC95\%} & \textbf{Precision top-10\%} & \textbf{IC95\%} & \textbf{Umbral} & \textbf{Recall} & \textbf{Precision} & \textbf{F1} \\",
        r"    \midrule",
    ]
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
    ruta_tex = out_dir / "tab_marginal_dmsp_fbeta2_cv10.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
