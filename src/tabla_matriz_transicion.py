"""
tabla_matriz_transicion.py
=====================================
Genera la tabla de matrices de transición de pobreza monetaria por ingreso
(Cuadro "Matrices de transición de pobreza monetaria por ingreso", Sección
"Dinámica de transiciones a la pobreza en la ELCO", Tabla~\\ref{tab:matriz_transicion}
de la tesis) a partir de los conteos/porcentajes ya calculados por el
pipeline de pobreza -- este script NO recalcula la matriz desde cero, solo
formatea los CSV ya producidos por `build_pobreza_desagregaciones.py` en
la tabla LaTeX que hoy está tecleada a mano en `paper/main.tex`.

QUÉ HACE

    1. Carga los porcentajes por fila (`transicion_pct_ola{1_a_2,2_a_3}.csv`)
       y los conteos absolutos (`transicion_conteo_ola{1_a_2,2_a_3}.csv`) de
       `outputs/tables/pobreza/`.
    2. Calcula n total (hogares con emparejamiento 1 a 1) de cada periodo
       sumando los 4 conteos de la matriz 2x2.
    3. Exporta una tabla .tex lista para incorporar al documento.

INPUTS

    outputs/tables/pobreza/transicion_pct_ola1_a_2.csv
    outputs/tables/pobreza/transicion_pct_ola2_a_3.csv
    outputs/tables/pobreza/transicion_conteo_ola1_a_2.csv
    outputs/tables/pobreza/transicion_conteo_ola2_a_3.csv

OUTPUTS

    paper/tables/tab_matriz_transicion.tex

CÓMO CORRER

    python src/tabla_matriz_transicion.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "pct_ola1_a_2": REPO_ROOT / "outputs" / "tables" / "pobreza" / "transicion_pct_ola1_a_2.csv",
    "pct_ola2_a_3": REPO_ROOT / "outputs" / "tables" / "pobreza" / "transicion_pct_ola2_a_3.csv",
    "conteo_ola1_a_2": REPO_ROOT / "outputs" / "tables" / "pobreza" / "transicion_conteo_ola1_a_2.csv",
    "conteo_ola2_a_3": REPO_ROOT / "outputs" / "tables" / "pobreza" / "transicion_conteo_ola2_a_3.csv",
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar_matriz(ruta_pct: Path, ruta_conteo: Path) -> tuple[pd.DataFrame, int]:
    if not ruta_pct.exists() or not ruta_conteo.exists():
        print(f"ERROR: no se encontró {ruta_pct} o {ruta_conteo}", file=sys.stderr)
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
        r"  \caption{Matrices de transición de pobreza monetaria por ingreso",
        r"  (\% de fila, panel emparejado 1 a 1)}",
        r"  \label{tab:matriz_transicion}",
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
        r"    dividieron entre olas. Fuente: cálculos propios con base en ELCA",
        r"    2010, 2013, 2016, generados por \texttt{src/tabla\_matriz\_transicion.py}",
        r"    sobre \texttt{outputs/tables/pobreza/transicion\_\{pct,conteo\}\_ola*.csv}",
        r"    (\texttt{src/04\_features/build\_pobreza\_desagregaciones.py}).",
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
    ruta_tex = out_dir / "tab_matriz_transicion.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
