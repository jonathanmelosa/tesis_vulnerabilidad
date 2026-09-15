"""
tabla_trayectorias_3olas.py
=====================================
Formatea los CSV ya calculados por `src/02_build/eda_trayectorias_3olas.py`
(`trayectorias_3olas_{monetaria,ipm}.csv`) en UNA tabla LaTeX compacta que
pone monetaria e IPM lado a lado -- para la Sección "Dinámica de
transiciones a la pobreza", que necesita mostrar ambas definiciones sin
usar dos tablas separadas (restricción de espacio).

INPUTS

    outputs/tables/eda_transicion_covariables/trayectorias_3olas_monetaria.csv
    outputs/tables/eda_transicion_covariables/trayectorias_3olas_ipm.csv

OUTPUTS

    paper/tables/tab_trayectorias_3olas.tex

CÓMO CORRER

    python src/tabla_trayectorias_3olas.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "monetaria": REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables" / "trayectorias_3olas_monetaria.csv",
    "ipm": REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables" / "trayectorias_3olas_ipm.csv",
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}

ORDEN_TRAYECTORIAS = [
    "Nunca cae",
    "Cae tarde (solo en 2016)",
    "Entra y sale (transitorio)",
    "Entra y se queda (persistente)",
]


def cargar(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta} -- correr primero src/02_build/eda_trayectorias_3olas.py", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ruta).set_index("trayectoria").reindex(ORDEN_TRAYECTORIAS)
    return df


def generar_tex(monetaria: pd.DataFrame, ipm: pd.DataFrame) -> str:
    n_monetaria = int(monetaria["n_hogares"].sum())
    n_ipm = int(ipm["n_hogares"].sum())
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Trayectorias a 3 olas (2010$\to$2013$\to$2016) de hogares",
        r"  NO pobres en 2010, panel emparejado 1 a 1, monetaria vs. IPM.}",
        r"  \label{tab:trayectorias_3olas}",
        r"  \footnotesize",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{\textbf{Monetaria}} & \multicolumn{2}{c}{\textbf{IPM}} \\",
        r"    \textbf{Trayectoria} & \textbf{n} & \textbf{\%} & \textbf{n} & \textbf{\%} \\",
        r"    \midrule",
    ]
    for trayectoria in ORDEN_TRAYECTORIAS:
        fm, fi = monetaria.loc[trayectoria], ipm.loc[trayectoria]
        lineas.append(
            f"    {trayectoria} & {int(fm['n_hogares']):,} & {fm['pct_n']:.1f}\\% & "
            f"{int(fi['n_hogares']):,} & {fi['pct_n']:.1f}\\% \\\\".replace(",", "{,}").replace("&", "&", 0)
        )
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.85\textwidth}",
        r"    \vspace{4pt}",
        rf"    \footnotesize \textit{{Nota:}} $n$={n_monetaria:,} hogares no pobres en 2010 bajo".replace(",", "{,}"),
        rf"    pobreza monetaria; $n$={n_ipm:,} bajo IPM (universos distintos porque".replace(",", "{,}"),
        r"    no-pobre-monetaria y no-pobre-IPM en 2010 no son el mismo conjunto de",
        r"    hogares). \% sobre ese universo, sin ponderar (ver también",
        r"    \texttt{pct\_ponderado} en los CSV fuente). Fuente: cálculos propios,",
        r"    \texttt{outputs/tables/eda\_transicion\_covariables/trayectorias\_3olas\_\{monetaria,ipm\}.csv}",
        r"    (\texttt{src/02\_build/eda\_trayectorias\_3olas.py}), generados por",
        r"    \texttt{src/tabla\_trayectorias\_3olas.py}.",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lineas)


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    monetaria = cargar(cfg["monetaria"])
    ipm = cargar(cfg["ipm"])
    tex = generar_tex(monetaria, ipm)
    ruta_tex = out_dir / "tab_trayectorias_3olas.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
