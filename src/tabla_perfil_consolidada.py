"""
tabla_perfil_consolidada.py
=====================================
Version CONSOLIDADA de `tabla_perfil_completo.py`: en vez de 1 tabla por
categoria + 6 figuras "sparse" (`graf_perfil_categorias_sparse.py`), junta
las 53 variables de pobreza monetaria en UNA sola tabla larga para caber
en el presupuesto de 5 paginas de la Seccion 5.1 -- pedido del usuario
(2026-09-15): "incorpora todo y dejalo en 5 paginas".

Cambio 2026-09-15 (segundo intento, tras medir el resultado del primero):
la primera version partia las 53 variables en 2 tablas `table`+`resizebox`
de ~27 filas via [tp] -- funcionaba, pero cada una que no cabia junto al
texto quedaba en una "pagina exclusiva de floats" (regla del nucleo de
LaTeX: paginas asi se centran verticalmente con `\@fptop`/`\@fpbot`,
dejando ~40% de la pagina en blanco, sin que el texto siguiente pueda
fluir alrededor). Se cambia a `longtable` con anchos de columna FIJOS
(`p{0.40\textwidth}` + 4 columnas de `0.135\textwidth}`, sin resizebox)
-- longtable pagina de forma continua como texto normal, sin ese
problema. Distinto del intento de `tabla_perfil_completo.py` (2026-09-11,
abandonado por desbordarse a la derecha): aqui el ancho total de columnas
se fija explicitamente por diseno (0.40 + 4*0.135 = 0.94\textwidth) en vez
de dejar que el contenido determine el ancho natural, que es lo que
causaba el desborde esa vez.

Las 4 variables "excepcion" (el hogar vulnerable queda MAS cerca de
"siempre pobre" que de "sale", o incluso supera a "nunca pobre" -- ver
parrafo de sintesis en `main.tex`) se marcan con `\dagger` en la etiqueta:
`categoria_ocupacional_jefe` (Asalariado), `deuda_informal_hogar`,
`pct_ninos_cuidado_terceros_hogar`, `tiene_transporte_publico_comunidad`.

INPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_monetaria_2010_2013.csv

OUTPUTS
    paper/tables/tab_perfil_consolidada.tex (longtable, las 53 variables)

COMO CORRER
    python src/tabla_perfil_consolidada.py
"""

from pathlib import Path

import pandas as pd

from tabla_perfil_completo import (
    CATEGORIAS_ORDEN_GRUPO,
    etiqueta_fila,
    formatear_valor,
    inferir_formato,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

# Orden tematico completo (reemplaza GRUPO_A/GRUPO_B de la version anterior).
ORDEN_CATEGORIAS = [
    "Educación y empleo del jefe", "Vivienda: materiales y servicios",
    "Zona de residencia", "Geoespacial", "Activos del hogar",
    "Vivienda: hacinamiento", "Programas sociales y deuda",
    "Composición del hogar", "Comunidad", "Ingreso y gasto",
]

EXCEPCIONES = {
    "categoria_ocupacional_jefe",
    "deuda_informal_hogar",
    "pct_ninos_cuidado_terceros_hogar",
    "tiene_transporte_publico_comunidad",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla = pd.read_csv(TABLES_DIR / "perfil_completo_monetaria_2010_2013.csv")

    assert set(tabla["categoria"]) == set(ORDEN_CATEGORIAS), "categorías del CSV no coinciden con ORDEN_CATEGORIAS"
    orden = {c: i for i, c in enumerate(ORDEN_CATEGORIAS)}
    tabla["_orden"] = tabla["categoria"].map(orden)
    tabla = tabla.sort_values(["_orden", "efecto"], ascending=[True, False])

    lineas = [
        r"{\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{p{0.40\textwidth}rrrr}",
        r"  \caption{Perfil por grupo de transición, pobreza monetaria "
        r"(2010$\to$2013), las 53 variables robustas. $^{\dagger}$variable "
        r"donde el hogar vulnerable no queda entre ``sale'' y ``siempre "
        r"pobre''/``nunca pobre'' como el resto (ver texto).}"
        r"\label{tab:perfil_consolidada}\\",
        r"    \toprule",
        r"    \textbf{Variable} & \textbf{Siempre} & \textbf{Sale} & \textbf{Entra} & \textbf{Nunca} \\",
        r"    \midrule",
        r"    \endfirsthead",
        r"    \multicolumn{5}{l}{\footnotesize \textit{(Cuadro~\ref{tab:perfil_consolidada}, continuación)}} \\",
        r"    \toprule",
        r"    \textbf{Variable} & \textbf{Siempre} & \textbf{Sale} & \textbf{Entra} & \textbf{Nunca} \\",
        r"    \midrule",
        r"    \endhead",
        r"    \bottomrule",
        r"    \multicolumn{5}{r}{\footnotesize \textit{continúa en la página siguiente}} \\",
        r"    \endfoot",
        r"    \bottomrule",
        r"    \endlastfoot",
    ]

    categoria_actual = None
    for _, fila in tabla.iterrows():
        if fila["categoria"] != categoria_actual:
            categoria_actual = fila["categoria"]
            lineas.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{categoria_actual}}}}} \\\\")
        formato = inferir_formato(fila["tipo"], fila["nivel_mostrado"])
        etiqueta = etiqueta_fila(fila["variable"], fila["nivel_mostrado"])
        if fila["variable"] in EXCEPCIONES:
            etiqueta += r"$^{\dagger}$"
        valores = " & ".join(formatear_valor(fila[cat], formato) for cat in CATEGORIAS_ORDEN_GRUPO)
        lineas.append(f"    \\quad {etiqueta} & {valores} \\\\")

    lineas.append(r"\end{longtable}}")

    out_path = OUTPUT_DIR / "tab_perfil_consolidada.tex"
    out_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado: {out_path.name} ({len(tabla)} filas)")

    # limpieza: elimina los .tex de la version anterior (2 tablas [tp]) si existen
    for nombre in ("tab_perfil_consolidada_a.tex", "tab_perfil_consolidada_b.tex"):
        ruta_vieja = OUTPUT_DIR / nombre
        if ruta_vieja.exists():
            ruta_vieja.unlink()
            print(f"Eliminado (obsoleto): {nombre}")


if __name__ == "__main__":
    main()
