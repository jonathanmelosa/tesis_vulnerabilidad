"""
tabla_matriz_transicion_combinada.py
=====================================
Version COMPACTA de las matrices de transición monetaria (`tabla_matriz_
transicion.py`) e IPM (`tabla_matriz_transicion_ipm.py`): en vez de dos
tablas separadas, pone ambas definiciones de pobreza lado a lado en una
sola tabla -- para la Sección "Dinámica de transiciones a la pobreza",
que tiene un límite de espacio de 2 páginas y no puede permitirse dos
tablas completas más sus notas al pie por separado.

No recalcula nada: reusa los mismos CSV ya producidos por
`build_pobreza_desagregaciones.py` (monetaria) y
`build_matriz_transicion_ipm.py` (IPM).

INPUTS

    outputs/tables/pobreza/transicion_{pct,conteo}_ola{1_a_2,2_a_3}.csv
    outputs/tables/pobreza/transicion_{pct,conteo}_ipm_ola{1_a_2,2_a_3}.csv

OUTPUTS

    paper/tables/tab_matriz_transicion_combinada.tex

CÓMO CORRER

    python src/tabla_matriz_transicion_combinada.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
POBREZA_DIR = REPO_ROOT / "outputs" / "tables" / "pobreza"

CONFIG = {
    "monetaria": {
        "pct_1_2": POBREZA_DIR / "transicion_pct_ola1_a_2.csv",
        "pct_2_3": POBREZA_DIR / "transicion_pct_ola2_a_3.csv",
        "conteo_1_2": POBREZA_DIR / "transicion_conteo_ola1_a_2.csv",
        "conteo_2_3": POBREZA_DIR / "transicion_conteo_ola2_a_3.csv",
    },
    "ipm": {
        "pct_1_2": POBREZA_DIR / "transicion_pct_ipm_ola1_a_2.csv",
        "pct_2_3": POBREZA_DIR / "transicion_pct_ipm_ola2_a_3.csv",
        "conteo_1_2": POBREZA_DIR / "transicion_conteo_ipm_ola1_a_2.csv",
        "conteo_2_3": POBREZA_DIR / "transicion_conteo_ipm_ola2_a_3.csv",
    },
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


def celda(pct: pd.DataFrame, fila: str, columna: str) -> str:
    return f"{pct.loc[fila, columna]:.1f}\\%"


def generar_tex(datos: dict) -> str:
    m12, n_m12 = cargar_matriz(CONFIG["monetaria"]["pct_1_2"], CONFIG["monetaria"]["conteo_1_2"])
    m23, n_m23 = cargar_matriz(CONFIG["monetaria"]["pct_2_3"], CONFIG["monetaria"]["conteo_2_3"])
    i12, n_i12 = cargar_matriz(CONFIG["ipm"]["pct_1_2"], CONFIG["ipm"]["conteo_1_2"])
    i23, n_i23 = cargar_matriz(CONFIG["ipm"]["pct_2_3"], CONFIG["ipm"]["conteo_2_3"])

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Matrices de transición de pobreza, monetaria por ingreso vs.",
        r"  IPM (\% de fila, panel emparejado 1 a 1). Estado inicial en filas,",
        r"  estado final en columnas.}",
        r"  \label{tab:matriz_transicion}",
        r"  \footnotesize",
        r"  \begin{tabular}{ll cc cc}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{\textbf{Monetaria}} & \multicolumn{2}{c}{\textbf{IPM}} \\",
        r"    \textbf{Periodo} & \textbf{Estado inicial} & \textbf{No pobre} & \textbf{Pobre} & \textbf{No pobre} & \textbf{Pobre} \\",
        r"    \midrule",
        rf"    \multirow{{2}}{{*}}{{2010 $\rightarrow$ 2013}} & No pobre & {celda(m12,'No pobre','No pobre')} & {celda(m12,'No pobre','Pobre')} & {celda(i12,'No pobre','No pobre')} & {celda(i12,'No pobre','Pobre')} \\",
        rf"     & Pobre & {celda(m12,'Pobre','No pobre')} & {celda(m12,'Pobre','Pobre')} & {celda(i12,'Pobre','No pobre')} & {celda(i12,'Pobre','Pobre')} \\",
        r"    \addlinespace",
        rf"    \multirow{{2}}{{*}}{{2013 $\rightarrow$ 2016}} & No pobre & {celda(m23,'No pobre','No pobre')} & {celda(m23,'No pobre','Pobre')} & {celda(i23,'No pobre','No pobre')} & {celda(i23,'No pobre','Pobre')} \\",
        rf"     & Pobre & {celda(m23,'Pobre','No pobre')} & {celda(m23,'Pobre','Pobre')} & {celda(i23,'Pobre','No pobre')} & {celda(i23,'Pobre','Pobre')} \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        rf"    \footnotesize \textit{{Nota:}} $n={n_m12:,}$ (2010$\to$2013) y $n={n_m23:,}$".replace(",", "{,}"),
        r"    (2013$\to$2016) hogares con emparejamiento 1 a 1 -- idéntico universo para",
        r"    monetaria e IPM (mismo panel, solo cambia la clasificación de pobreza);",
        r"    excluye 511 (2010--2013) y 1{,}190 (2013--2016) casos de hogares que se",
        r"    dividieron entre olas. Fuente: cálculos propios con base en ELCA 2010,",
        r"    2013, 2016, generados por \texttt{src/tabla\_matriz\_transicion\_combinada.py}",
        r"    sobre \texttt{outputs/tables/pobreza/transicion\_\{pct,conteo\}\_\{,ipm\_\}ola*.csv}",
        r"    (\texttt{src/04\_features/build\_pobreza\_desagregaciones.py},",
        r"    \texttt{build\_matriz\_transicion\_ipm.py}).",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    assert n_m12 == n_i12, f"universo monetaria ({n_m12}) != IPM ({n_i12}) en 2010->2013 -- revisar supuesto de mismo panel"
    assert n_m23 == n_i23, f"universo monetaria ({n_m23}) != IPM ({n_i23}) en 2013->2016 -- revisar supuesto de mismo panel"
    return "\n".join(lineas)


def main() -> None:
    out_dir = Path(CONFIG["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tex = generar_tex(CONFIG)
    ruta_tex = out_dir / "tab_matriz_transicion_combinada.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
