"""
tabla_shap_signo_nucleo.py
=====================================
Tabla de DIRECCIÓN de las 22 variables del núcleo SHAP (Sección~5.2.2 de
`main.tex`): para cada una, en cada una de las 4 combinaciones
definición de pobreza x ventana temporal, en cuántas de las 10
combinaciones algoritmo/especificación el modelo la usa en cada sentido.
El ranking de importancia (`tab_shap_nucleo_perfil`) dice cuáles variables
importan; esta tabla dice en qué sentido, sin apoyarse en el análisis
univariado. Une en una sola tabla el núcleo (combinaciones, de 10) y la
dirección (por fuente y clasificación). Este script solo formatea: el cálculo del signo está en
`src/05_model/diagnostico_shap_signo.py`.

Definiciones

    Signo de una combinación   signo del Spearman entre el valor de la
                               variable y su SHAP (`corr_spearman`); se
                               ignoran las correlaciones con
                               |rho| < UMBRAL_RHO (sin dirección clara).
                               $-$: valores altos REDUCEN el riesgo de
                               entrar en pobreza; $+$: lo AUMENTAN.
    Celda                      signo mayoritario, "n/m": n combinaciones
                               con ese signo de las m que tienen
                               dirección clara (las otras 10-m casi no
                               usan la variable: eso es importancia, ya
                               medida en `tab_shap_nucleo_perfil`, no
                               dirección).
    Dirección                  clasificación según cuántas de las 12
                               celdas de la malla de sensibilidad
                               (`sensibilidad_shap_nucleo.py`: |rho| x
                               proporción mínima de signo) declaran a la
                               variable consistente en las 4 fuentes:
                               Estable (12 de 12), Moderada (>= 8) e
                               Inestable (< 8).

Las variables categóricas (material de piso, estado civil) no tienen un
signo a nivel de variable (el orden de las categorías es arbitrario) y se
marcan "cat.": su lectura por nivel está en
`diagnostico_shap_signo.csv` (filas tipo `nivel`).

INPUTS

    data/processed/benchmark_resultados/diagnostico_shap_signo.csv
    data/processed/benchmark_resultados/diagnostico_shap_nucleo_perfil.csv
    data/processed/benchmark_resultados/sensibilidad_signo_nucleo_por_variable.csv

OUTPUTS

    paper/tables/tab_shap_signo_nucleo.tex
    data/processed/benchmark_resultados/shap_signo_nucleo_resumen.csv
    (resumen numérico detrás de la tabla, una fila por variable x fuente)

COMO CORRER

    python src/tabla_shap_signo_nucleo.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabla_shap_nucleo_perfil import ORDEN_CATEGORIAS, etiqueta

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = REPO_ROOT / "data" / "processed" / "benchmark_resultados"
RUTA_SIGNO = RESULTADOS / "diagnostico_shap_signo.csv"
RUTA_NUCLEO = RESULTADOS / "diagnostico_shap_nucleo_perfil.csv"
RUTA_SENSIBILIDAD = RESULTADOS / "sensibilidad_signo_nucleo_por_variable.csv"
RUTA_RESUMEN = RESULTADOS / "shap_signo_nucleo_resumen.csv"
RUTA_TEX = REPO_ROOT / "paper" / "tables" / "tab_shap_signo_nucleo.tex"

UMBRAL_RHO = 0.10
CELDAS_ESTABLE = 12  # de 12 celdas de la malla de sensibilidad
CELDAS_MODERADA = 8
COMBINACIONES = 10
FUENTES = [
    ("Monetaria_2010-2013", "\\shortstack{\\textbf{Monetaria}\\\\\\textbf{2010$\\to$13}}"),
    ("Monetaria_2013-2016", "\\shortstack{\\textbf{Monetaria}\\\\\\textbf{2013$\\to$16}}"),
    ("IPM_2010-2013", "\\shortstack{\\textbf{IPM}\\\\\\textbf{2010$\\to$13}}"),
    ("IPM_2013-2016", "\\shortstack{\\textbf{IPM}\\\\\\textbf{2013$\\to$16}}"),
]
CATEGORICAS = {"material_pisos_hogar", "estado_civil_jefe"}


def resumir(signo: pd.DataFrame, variables: list) -> pd.DataFrame:
    columnas = signo[signo["tipo"] == "columna"].copy()
    columnas["v"] = columnas["variable"].str.replace("^num__", "", regex=True)
    filas = []
    for v in variables:
        for fuente, _ in FUENTES:
            g = columnas[(columnas["v"] == v) & (columnas["fuente"] == fuente)]
            if v in CATEGORICAS:
                filas.append({"variable": v, "fuente": fuente, "n_comb": 0, "n_neg": None, "n_pos": None})
                continue
            assert len(g) == COMBINACIONES, f"{v} / {fuente}: {len(g)} combinaciones, se esperaban {COMBINACIONES}"
            filas.append({
                "variable": v, "fuente": fuente, "n_comb": len(g),
                "n_neg": int((g["corr_spearman"] <= -UMBRAL_RHO).sum()),
                "n_pos": int((g["corr_spearman"] >= UMBRAL_RHO).sum()),
            })
    return pd.DataFrame(filas)


def celda(fila: pd.Series) -> str:
    if fila["n_neg"] is None or pd.isna(fila["n_neg"]):
        return "cat."
    n_neg, n_pos = int(fila["n_neg"]), int(fila["n_pos"])
    if n_neg == n_pos == 0:
        return "--"
    signo, n = ("$-$", n_neg) if n_neg >= n_pos else ("$+$", n_pos)
    return f"{signo}\\,{n}/{n_neg + n_pos}"


def clasificar(variable: str, sensibilidad: pd.DataFrame) -> str:
    if variable in CATEGORICAS:
        return "--"
    n = int(sensibilidad.loc[sensibilidad["variable"] == variable, "celdas_consistente"].iloc[0])
    return "Estable" if n >= CELDAS_ESTABLE else "Moderada" if n >= CELDAS_MODERADA else "Inestable"


def generar_tex(nucleo: pd.DataFrame, resumen: pd.DataFrame, sensibilidad: pd.DataFrame) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Núcleo de variables del análisis SHAP y su dirección.",
        r"  \emph{Comb.}: combinaciones algoritmo/especificación (de 10) en que",
        r"  la variable está dentro del 80\% de la masa SHAP acumulada",
        r"  (Sección~\ref{subsec:importancia_resultados}). Cada celda de signo",
        r"  indica el signo mayoritario y en cuántas ($n$) de las $m$",
        r"  combinaciones con dirección clara ($|\rho|\geq 0.10$) aparece: $-$",
        r"  significa que valores altos de la variable \emph{reducen} el riesgo",
        r"  de entrar en pobreza; $+$, que lo \emph{aumentan}.",
        r"  \emph{Dirección}: Estable = mismo signo en las cuatro fuentes bajo",
        r"  las 12 combinaciones de umbrales evaluadas; Moderada = en al menos 8;",
        r"  Inestable = en menos. ``cat.'': variable categórica, sin signo a nivel",
        r"  de variable. $^{\dagger}$variable fuera del perfil univariado de 53",
        r"  variables robustas de la Sección~\ref{subsec:caracterizacion_grupos}.}",
        r"  \label{tab:shap_signo_nucleo}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{lccccccc}",
        r"    \toprule",
        "    \\textbf{Variable} & \\textbf{Comb.} & " + " & ".join(t for _, t in FUENTES) + r" & \textbf{Dirección} \\",
        r"    \midrule",
    ]
    for categoria in ORDEN_CATEGORIAS:
        bloque = nucleo[nucleo["categoria"] == categoria].sort_values("n_combinaciones", ascending=False)
        if bloque.empty:
            continue
        lineas.append(f"    \\textbf{{{categoria}}} & & & & & & \\\\")
        for _, v in bloque.iterrows():
            g = resumen[resumen["variable"] == v["variable"]].set_index("fuente").loc[[f for f, _ in FUENTES]].reset_index()
            marca = "" if v["en_perfil_53"] else "$^{\\dagger}$"
            celdas = " & ".join(celda(f) for _, f in g.iterrows())
            lineas.append(f"    \\quad {etiqueta(v['variable'])}{marca} & {v['n_combinaciones']} & {celdas} & {clasificar(v['variable'], sensibilidad)} \\\\")
        lineas.append(r"    \addlinespace")
    if lineas[-1] == r"    \addlinespace":
        lineas.pop()
    lineas += [r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lineas)


def main() -> None:
    signo = pd.read_csv(RUTA_SIGNO)
    nucleo = pd.read_csv(RUTA_NUCLEO)
    resumen = resumir(signo, nucleo["variable"].tolist())
    resumen.to_csv(RUTA_RESUMEN, index=False)

    sensibilidad = pd.read_csv(RUTA_SENSIBILIDAD)
    tex = generar_tex(nucleo, resumen, sensibilidad)
    RUTA_TEX.parent.mkdir(parents=True, exist_ok=True)
    RUTA_TEX.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada (cuerpo de tabular; el cierre \\end{{table}} sigue a mano en main.tex): {RUTA_TEX}")
    print(f"Resumen numérico: {RUTA_RESUMEN}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
