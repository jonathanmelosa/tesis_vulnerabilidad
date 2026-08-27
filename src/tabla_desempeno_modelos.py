"""
tabla_desempeno_modelos.py
=====================================
Genera la tabla de desempeño comparativo de los cinco algoritmos del
benchmark (Modelos A y B, holdout temporal train 2010->2013, test
2013->2016; Sección "Desempeño comparativo de modelos",
Tabla~\\ref{tab:desempeno_modelos} de la tesis) a partir del registro de
resultados de todos los modelos entrenados -- no reentrena nada, solo
filtra y formatea filas ya calculadas por los scripts de
`src/05_model/modelo_*.py`.

QUÉ HACE

    1. Carga `registro_modelos.csv` (una fila por algoritmo x especificación
       x corrida, con media/IC95% de AUC-ROC/recall/f1 sobre 5 semillas).
    2. Filtra a las especificaciones principales del benchmark: "A" (con
       ingreso/gasto) y "B" (sin ellos) -- excluye las variantes de
       ablation/geoespaciales (Ageo3, AgeoDMSP, Anoriq, etc., usadas en
       otras tablas de la tesis, ver `tabla_marginal_dmsp.py`).
    3. Ordena cada bloque (A, B) por AUC-ROC descendente, igual que en la
       tesis.
    4. Exporta una tabla .tex lista para incorporar al documento.

INPUTS

    data/processed/benchmark_resultados/registro_modelos.csv

OUTPUTS

    paper/tables/tab_desempeno_modelos.tex

CÓMO CORRER

    python src/tabla_desempeno_modelos.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "registro_modelos": REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "registro_modelos.csv",
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
        print(f"ERROR: no se encontró {ruta}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ruta)
    df = df[df["especificacion"].isin(cfg["especificaciones_benchmark"])].copy()
    df["algoritmo"] = df["algoritmo"].map(cfg["nombres_algoritmo"]).fillna(df["algoritmo"])
    return df


def formatear_fila(fila: pd.Series) -> str:
    return (
        f"    {fila['algoritmo']:<26s} & {fila['especificacion']} & {fila['balanceo_elegido']:<9s} & "
        f"{fila['umbral_clasificacion_media']:.2f} & {fila['auc_roc_media']:.3f} & "
        f"[{fila['auc_roc_ci95_low']:.3f}, {fila['auc_roc_ci95_high']:.3f}] & "
        f"{fila['recall_media']:.3f} & {fila['f1_media']:.3f} \\\\"
    )


def generar_tex(df: pd.DataFrame, cfg: dict) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Desempeño de los cinco algoritmos, holdout temporal",
        r"  (train 2010$\to$2013, test 2013$\to$2016). Media $\pm$ intervalo de",
        r"  confianza al 95\% sobre 5 semillas del ajuste final}",
        r"  \label{tab:desempeno_modelos}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llcccccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Espec.} & \textbf{Balanceo} & \textbf{Umbral} & \textbf{AUC-ROC} & \textbf{IC95\%} & \textbf{Recall} & \textbf{F1} \\",
        r"    \midrule",
    ]
    especificaciones = cfg["especificaciones_benchmark"]
    for i, espec in enumerate(especificaciones):
        bloque = df[df["especificacion"] == espec].sort_values("auc_roc_media", ascending=False)
        for _, fila in bloque.iterrows():
            lineas.append(formatear_fila(fila))
        if i < len(especificaciones) - 1:
            lineas.append(r"    \addlinespace")
    lineas += [r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lineas)


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = cargar_registro(cfg)
    tex = generar_tex(df, cfg)
    ruta_tex = out_dir / "tab_desempeno_modelos.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada (cuerpo de tabular; encabezado table/caption/nota siguen a mano en main.tex): {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
