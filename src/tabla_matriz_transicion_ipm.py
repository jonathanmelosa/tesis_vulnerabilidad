"""
tabla_matriz_transicion_ipm.py
=====================================
Version IPM de `tabla_matriz_transicion.py` -- formatea los CSV ya
calculados por `src/04_features/build_matriz_transicion_ipm.py`
(`transicion_{pct,conteo}_ipm_ola*.csv`) en una tabla LaTeX lista para
incorporar al documento, espejo exacto de la version monetaria.

INPUTS

    outputs/tables/pobreza/transicion_pct_ipm_ola1_a_2.csv
    outputs/tables/pobreza/transicion_pct_ipm_ola2_a_3.csv
    outputs/tables/pobreza/transicion_conteo_ipm_ola1_a_2.csv
    outputs/tables/pobreza/transicion_conteo_ipm_ola2_a_3.csv

OUTPUTS

    paper/tables/tab_matriz_transicion_ipm.tex

CÓMO CORRER

    python src/tabla_matriz_transicion_ipm.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "pct_ola1_a_2": REPO_ROOT / "outputs" / "tables" / "pobreza" / "transicion_pct_ipm_ola1_a_2.csv",
    "pct_ola2_a_3": REPO_ROOT / "outputs" / "tables" / "pobreza" / "transicion_pct_ipm_ola2_a_3.csv",
    "conteo_ola1_a_2": REPO_ROOT / "outputs" / "tables" / "pobreza" / "transicion_conteo_ipm_ola1_a_2.csv",
    "conteo_ola2_a_3": REPO_ROOT / "outputs" / "tables" / "pobreza" / "transicion_conteo_ipm_ola2_a_3.csv",
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar_matriz(ruta_pct: Path, ruta_conteo: Path) -> tuple[pd.DataFrame, int]:
    if not ruta_pct.exists() or not ruta_conteo.exists():
        print(f"ERROR: no se encontró {ruta_pct} o {ruta_conteo} -- correr primero build_matriz_transicion_ipm.py", file=sys.stderr)
        sys.exit(1)
    pct = pd.read_csv(ruta_pct, index_col=0)
    conteo = pd.read_csv(ruta_conteo, index_col=0)
    n_total = int(conteo.to_numpy().sum())
    return pct, n_total


def formatear_fila(pct: pd.DataFrame, fila: str) -> str:
    return f"{pct.loc[fila, 'No pobre']:.1f}\\% & {pct.loc[fila, 'Pobre']:.1f}\\%"


def generar_tex(pct_1_2: pd.DataFrame, n_1_2: int, pct_2_3: pd.DataFrame, n_2_3: int) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Matrices de transición de pobreza multidimensional (IPM)",
        r"  (\% de fila, panel emparejado 1 a 1)}",
        r"  \label{tab:matriz_transicion_ipm}",
        r"  \begin{tabular}{llcc}",
        r"    \toprule",
        r"    \textbf{Periodo} & \textbf{Estado inicial} & \textbf{No pobre (final)} & \textbf{Pobre (final)} \\",
        r"    \midrule",
        f"    \\multirow{{2}}{{*}}{{2010 $\\rightarrow$ 2013 ($n={n_1_2:,}$)}}".replace(",", "{,}"),
        f"      & No pobre & {formatear_fila(pct_1_2, 'No pobre')} \\\\",
        f"      & Pobre    & {formatear_fila(pct_1_2, 'Pobre')} \\\\",
        r"    \addlinespace",
        f"    \\multirow{{2}}{{*}}{{2013 $\\rightarrow$ 2016 ($n={n_2_3:,}$)}}".replace(",", "{,}"),
        f"      & No pobre & {formatear_fila(pct_2_3, 'No pobre')} \\\\",
        f"      & Pobre    & {formatear_fila(pct_2_3, 'Pobre')} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.85\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} $n$ = hogares con emparejamiento 1 a 1;",
        r"    excluye 511 (2010--2013) y 1{,}190 (2013--2016) casos de hogares que se",
        r"    dividieron entre olas -- mismo criterio de emparejamiento que la matriz",
        r"    de pobreza monetaria (Tabla~\ref{tab:matriz_transicion}), aquí aplicado",
        r"    a la clasificación IPM (\texttt{pobre\_ipm}, ver",
        r"    \texttt{build\_ipm\_multidimensional.py}). Fuente: cálculos propios con",
        r"    base en ELCA 2010, 2013, 2016, generados por",
        r"    \texttt{src/tabla\_matriz\_transicion\_ipm.py} sobre",
        r"    \texttt{outputs/tables/pobreza/transicion\_\{pct,conteo\}\_ipm\_ola*.csv}",
        r"    (\texttt{src/04\_features/build\_matriz\_transicion\_ipm.py}).",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lineas)


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pct_1_2, n_1_2 = cargar_matriz(cfg["pct_ola1_a_2"], cfg["conteo_ola1_a_2"])
    pct_2_3, n_2_3 = cargar_matriz(cfg["pct_ola2_a_3"], cfg["conteo_ola2_a_3"])

    tex = generar_tex(pct_1_2, n_1_2, pct_2_3, n_2_3)
    ruta_tex = out_dir / "tab_matriz_transicion_ipm.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
