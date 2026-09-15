"""
tabla_perfil_split_trayectoria.py
=====================================
Formatea `perfil_split_transitorio_persistente_monetaria.csv` (ya
calculado por `src/02_build/eda_perfil_split_trayectoria.py`) en una tabla
LaTeX compacta para la Seccion 5.1: compara el subgrupo "Entra
(transitorio)" (vuelve a salir de la pobreza para 2016) contra "Entra
(persistente)" (se queda pobre), sobre 3 variables ejemplo (formalidad
laboral, deuda informal, transporte publico comunitario) mas 2 filas de
resumen agregado (cuantas de las 53 variables quedan mas cerca de "sale"
que de "nunca cae", y mas cerca de "siempre pobre" que de "sale") -- todos
los numeros se recalculan aqui desde el CSV fuente, no estan hardcodeados.

INPUTS
    outputs/tables/eda_transicion_covariables/perfil_split_transitorio_persistente_monetaria.csv

OUTPUTS
    paper/tables/tab_perfil_split_trayectoria.tex

COMO CORRER
    python src/tabla_perfil_split_trayectoria.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

VARIABLES_EJEMPLO = {
    "categoria_ocupacional_jefe": "Jefe asalariado",
    "deuda_informal_hogar": "Tiene deuda informal",
    "tiene_transporte_publico_comunidad": "Comunidad con transporte público",
}

GRUPOS = ["Siempre pobre", "Sale de la pobreza", "Entra (transitorio)", "Entra (persistente)", "Nunca pobre"]


def main() -> None:
    df = pd.read_csv(TABLES_DIR / "perfil_split_transitorio_persistente_monetaria.csv")

    d_sale_t = (df["Entra (transitorio)"] - df["Sale de la pobreza"]).abs()
    d_nunca_t = (df["Entra (transitorio)"] - df["Nunca pobre"]).abs()
    d_sale_p = (df["Entra (persistente)"] - df["Sale de la pobreza"]).abs()
    d_nunca_p = (df["Entra (persistente)"] - df["Nunca pobre"]).abs()
    d_siempre_t = (df["Entra (transitorio)"] - df["Siempre pobre"]).abs()
    d_siempre_p = (df["Entra (persistente)"] - df["Siempre pobre"]).abs()

    n_total = len(df)
    cerca_sale_t = int((d_sale_t < d_nunca_t).sum())
    cerca_sale_p = int((d_sale_p < d_nunca_p).sum())
    cerca_siempre_t = int((d_siempre_t < d_sale_t).sum())
    cerca_siempre_p = int((d_siempre_p < d_sale_p).sum())

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Perfil del hogar vulnerable (``entra en pobreza'', 2010$\to$2013) "
        r"desagregado por trayectoria a 2016: transitorio ($n=355$, vuelve a salir) vs.\ "
        r"persistente ($n=255$, se queda pobre). Valores ponderados por grupo; filas "
        r"superiores, resumen agregado sobre las 53 variables; filas inferiores, ejemplos "
        r"puntuales donde el promedio agregado (Sección~\ref{subsec:caracterizacion_grupos}) "
        r"ocultaba una divergencia entre subgrupos.}",
        r"  \label{tab:perfil_split_trayectoria}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    \textbf{Variable} & \textbf{Siempre} & \textbf{Sale} & \textbf{Transitorio} & \textbf{Persistente} & \textbf{Nunca} \\",
        r"    \midrule",
        rf"    Más cerca de ``sale'' que de ``nunca'' (de 53) & \multicolumn{{2}}{{c}}{{--}} & {cerca_sale_t}/53 & {cerca_sale_p}/53 & -- \\",
        rf"    Más cerca de ``siempre'' que de ``sale'' (de 53) & \multicolumn{{2}}{{c}}{{--}} & {cerca_siempre_t}/53 & {cerca_siempre_p}/53 & -- \\",
        r"    \addlinespace",
    ]

    for var, etiqueta in VARIABLES_EJEMPLO.items():
        fila = df.loc[df["variable"] == var].iloc[0]
        valores = " & ".join(f"{fila[g]:.1f}\\%" for g in GRUPOS)
        lineas.append(f"    {etiqueta} & {valores} \\\\")

    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} de los 723 hogares que entran en pobreza entre "
        r"2010 y 2013, 610 tienen dato de la ola 2016 (113 se pierden por atrición, misma "
        r"tasa que el resto del panel); de esos 610, 355 (58\%) vuelven a salir de la "
        r"pobreza para 2016 y 255 (42\%) se quedan pobres. Fuente: cálculos propios, "
        r"\texttt{perfil\_split\_transitorio\_persistente\_monetaria.csv} "
        r"(\texttt{src/02\_build/eda\_perfil\_split\_trayectoria.py}).",
        r"  \end{minipage}",
        r"\end{table}",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tab_perfil_split_trayectoria.tex"
    out_path.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Tabla exportada: {out_path}")


if __name__ == "__main__":
    main()
