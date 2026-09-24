"""
tabla_perfil_nucleo_comun.py
================================
Tabla del "núcleo del perfil" (Hallazgo 4, Sección 5.1): las variables
robustas comunes a AMBAS definiciones de pobreza (monetaria, 53
variables, y IPM, 36 variables) -- pedido del usuario (2026-09-16), tras
notar que el párrafo citaba "28 comunes" sin que el lector pudiera verlas.

La interseccion de variables (por nombre, no por posicion) se calcula
AQUI, dinamicamente, contra los dos CSV fuente -- no esta hardcodeada,
para que si algun dia cambia la seleccion de 53 o de 36 variables, el
numero y la lista se actualicen solos (y el script falla ruidosamente si
el conteo ya no da 28, en vez de reportar en silencio un numero
desactualizado).

Muestra los valores bajo pobreza MONETARIA (mismos valores que ya estan
en la Tabla~\\ref{tab:perfil_consolidada}, aqui aislados como el
subconjunto que ademas se sostiene bajo IPM) -- no se listan tambien los
valores de IPM para evitar el problema de que el nivel "mas discriminante"
de una variable categorica (ej. zona: Rural/Urbano) puede diferir entre el
ranking monetario y el de IPM, lo que haria los dos conjuntos de columnas
no directamente comparables sin trabajo adicional de alineacion.

Layout de 2 columnas lado a lado, balanceadas por ALTO renderizado
estimado (no por cantidad de filas) -- mismo enfoque, mismos bugs ya
encontrados y corregidos, que `tabla_perfil_consolidada.py` (28 variables
+ 8 categorias no caben en una sola columna a una pagina, se probo y
desbordo -- ver docs/decisions.md). No se reusa el codigo de ese script
directamente porque marca con $^\\dagger$ las 4 variables "excepcion" del
Hallazgo 2 (2 de las cuales caen dentro de estas 28) sin que esta tabla
explique que significa esa marca -- mas simple mantener este script
independiente sin esa logica.

INPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_monetaria_2010_2013.csv
    outputs/tables/eda_transicion_covariables/perfil_completo_ipm_2010_2013.csv

OUTPUTS
    paper/tables/tab_perfil_nucleo_comun.tex

COMO CORRER
    python src/tabla_perfil_nucleo_comun.py
"""

import re
from pathlib import Path

import pandas as pd

from tabla_perfil_completo import CATEGORIAS_ORDEN_GRUPO, etiqueta_fila, formatear_valor, inferir_formato

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

N_ESPERADO = 28

# Mismo esquema que tabla_perfil_consolidada.py (ver ese script para la
# justificacion), pero CHARS_POR_LINEA recalibrado el 2026-09-24 para el
# ancho de columna de ESTA tabla (p{0.3038\textwidth}, distinto al de la
# consolidada): en el PDF compilado "Estado civil del jefe: En unión
# libre" (37 caracteres) cabe en una linea; con 34 se sobreestimaban los
# renglones de la mitad izquierda y el relleno caia en la mitad que ya
# era mas larga.
CHARS_POR_LINEA = 37
PESO_CATEGORIA = 1.5


def _peso_lineas(etiqueta_latex: str) -> float:
    texto = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", etiqueta_latex)
    texto = re.sub(r"\\[a-zA-Z]+", "", texto)
    return max(1, -(-len(texto) // CHARS_POR_LINEA))


def _construir_entradas(tabla: pd.DataFrame) -> list[tuple[str, str, float, str]]:
    entradas = []
    categoria_actual = None
    for _, fila in tabla.iterrows():
        if fila["categoria"] != categoria_actual:
            categoria_actual = fila["categoria"]
            entradas.append(("categoria", categoria_actual, PESO_CATEGORIA, categoria_actual))
        formato = inferir_formato(fila["tipo"], fila["nivel_mostrado"])
        etiqueta = etiqueta_fila(fila["variable"], fila["nivel_mostrado"])
        valores = " & ".join(formatear_valor(fila[cat], formato) for cat in CATEGORIAS_ORDEN_GRUPO)
        entradas.append(("fila", f"    \\quad {etiqueta} & {valores} \\\\", _peso_lineas(etiqueta), categoria_actual))
    return entradas


def _punto_de_corte(entradas: list[tuple[str, str, float, str]]) -> int:
    peso_total = sum(peso for _, _, peso, _ in entradas)
    mitad = peso_total / 2
    acumulado = 0.0
    mejor_idx, mejor_dist = len(entradas) // 2, peso_total
    for i, (tipo, _, peso, _) in enumerate(entradas):
        if i != 0 and entradas[i - 1][0] != "categoria":
            dist = abs(acumulado - mitad)
            if dist < mejor_dist:
                mejor_idx, mejor_dist = i, dist
        acumulado += peso
    return mejor_idx


def _tabular(entradas: list[tuple[str, str, float, str]]) -> str:
    lineas = [
        # [t]: la linea base de la caja es la primera fila, para que las
        # dos mitades queden alineadas arriba aunque tengan altos distintos.
        r"\begin{tabular}[t]{p{0.3038\textwidth}rrrr}",
        r"  \toprule",
        r"    \textbf{Variable} & \textbf{Siempre} & \textbf{Sale} & \textbf{Entra} & \textbf{Nunca} \\",
        r"  \midrule",
    ]
    for tipo, contenido, _, _ in entradas:
        if tipo == "categoria":
            lineas.append(f"    \\addlinespace\\multicolumn{{5}}{{l}}{{\\textbf{{{contenido}}}}} \\\\")
        else:
            lineas.append(contenido)
    lineas.append(r"  \bottomrule")
    lineas.append(r"\end{tabular}")
    return "\n".join(lineas)


def main() -> None:
    monetaria = pd.read_csv(TABLES_DIR / "perfil_completo_monetaria_2010_2013.csv")
    ipm = pd.read_csv(TABLES_DIR / "perfil_completo_ipm_2010_2013.csv")

    comunes = set(monetaria["variable"]) & set(ipm["variable"])
    if len(comunes) != N_ESPERADO:
        raise ValueError(
            f"Se esperaban {N_ESPERADO} variables comunes (citado en el texto de la tesis), "
            f"pero la interseccion actual da {len(comunes)} -- revisar el párrafo del Hallazgo 4 "
            f"en main.tex antes de regenerar esta tabla."
        )

    tabla = monetaria[monetaria["variable"].isin(comunes)].sort_values(["categoria", "efecto"], ascending=[True, False])

    entradas = _construir_entradas(tabla)
    corte = _punto_de_corte(entradas)
    izquierda, derecha = entradas[:corte], entradas[corte:]

    if izquierda and derecha and derecha[0][3] == izquierda[-1][3]:
        cat = derecha[0][3]
        derecha = [("categoria", f"{cat} (cont.)", PESO_CATEGORIA, cat)] + derecha

    peso_izq = sum(p for _, _, p, _ in izquierda)
    peso_der = sum(p for _, _, p, _ in derecha)
    relleno = round(abs(peso_izq - peso_der))
    fila_vacia = ("fila", "    \\quad & & & & \\\\", 1, None)
    if relleno > 0:
        if peso_izq > peso_der:
            derecha = derecha + [fila_vacia] * relleno
        else:
            izquierda = izquierda + [fila_vacia] * relleno

    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Núcleo del perfil de vulnerabilidad: las 28 variables robustas "
        r"comunes a pobreza monetaria (53 variables, Tabla~\ref{tab:perfil_consolidada}) "
        r"y a pobreza multidimensional --IPM-- (36 variables, Anexo~\ref{apx:perfil_ipm}). "
        r"Valores bajo pobreza monetaria, transición 2010$\to$2013.}",
        r"  \label{tab:perfil_nucleo_comun}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \newsavebox{\nucleoBoxA}",
        r"  \newsavebox{\nucleoBoxB}",
        r"  \sbox{\nucleoBoxA}{%",
        "    " + _tabular(izquierda).replace("\n", "\n    ") + "}",
        r"  \sbox{\nucleoBoxB}{%",
        "    " + _tabular(derecha).replace("\n", "\n    ") + "}",
        # Ambas mitades se escalan por el MISMO factor (regla de tres sobre
        # el ancho natural mayor de las dos), en vez de dos \resizebox
        # independientes -- que, al normalizar cada una a 0.49\textwidth
        # por separado, producian factores de escala ligeramente distintos
        # (una de las dos quedaba perceptiblemente mas grande). Este
        # esquema estaba editado a mano en el .tex (commit 5cb242f) y no en
        # este script; se incorpora aqui el 2026-09-24 para que regenerar
        # la tabla no lo pierda.
        r"  \newlength{\nucleoTarget}",
        r"  \setlength{\nucleoTarget}{0.49\textwidth}",
        r"  \newlength{\nucleoMax}",
        r"  \ifdim\wd\nucleoBoxA>\wd\nucleoBoxB",
        r"    \setlength{\nucleoMax}{\wd\nucleoBoxA}",
        r"  \else",
        r"    \setlength{\nucleoMax}{\wd\nucleoBoxB}",
        r"  \fi",
        "",
        r"  \begin{minipage}[t]{0.49\textwidth}",
        r"    \resizebox{\dimexpr\nucleoTarget*\wd\nucleoBoxA/\nucleoMax\relax}{!}{\usebox{\nucleoBoxA}}",
        r"  \end{minipage}\hfill",
        r"  \begin{minipage}[t]{0.49\textwidth}",
        r"    \resizebox{\dimexpr\nucleoTarget*\wd\nucleoBoxB/\nucleoMax\relax}{!}{\usebox{\nucleoBoxB}}",
        r"  \end{minipage}",
        r"\end{table}",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "tab_perfil_nucleo_comun.tex"
    out_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado: {out_path} ({len(tabla)} variables, corte en {corte}/{len(entradas)})")


if __name__ == "__main__":
    main()
