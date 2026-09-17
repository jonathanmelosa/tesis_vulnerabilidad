"""
tabla_perfil_consolidada.py
=====================================
Version CONSOLIDADA de `tabla_perfil_completo.py`: en vez de 1 tabla por
categoria + 6 figuras "sparse" (`graf_perfil_categorias_sparse.py`), junta
las 53 variables de pobreza monetaria en UNA sola tabla larga para caber
en el presupuesto de 5 paginas de la Seccion 5.1 -- pedido del usuario
(2026-09-15): "incorpora todo y dejalo en 5 paginas".

Cambio 2026-09-16 (tercer intento, pedido del usuario): la version anterior
(`longtable`, ver historial en git) resolvia el problema de paginas de
floats con ~40% en blanco, pero dejaba la tabla repartida en varias
paginas seguidas del texto -- el usuario pidio que quedara consolidada en
UNA sola hoja, y que las cabeceras de categoria (`Educacion y empleo del
jefe`, etc.) resaltaran mas (se perdian entre las filas). Se cambia a un
UNICO `table[H]` (posicion FIJA justo donde se inserta -- `H` de
`usepackage{float}`, ya cargado en main.tex y usado por el resto de
tablas del documento; NO `[p]`: un intento intermedio con `[p]` dejo que
LaTeX "flotara" la tabla libremente y la mando a la pagina 65, ~40
paginas despues de donde se referencia en el texto -- `[H]` fija la
tabla exactamente en su lugar, igual que `tab_perfil_split_trayectoria.tex`)
con DOS `tabular` en `minipage`s lado a lado (cada una a ~0.48\textwidth,
`scriptsize`, cada `tabular` envuelto en `\resizebox{\linewidth}{!}{...}`
-- sin esto, las columnas `r` sin ancho fijo desbordaban mas alla de
0.49\textwidth y las dos mitades quedaban superpuestas, otro bug
encontrado y corregido en el mismo intento). El punto de corte entre las
dos columnas se calcula dinamicamente: la frontera de categoria mas
cercana a la mitad del total de lineas (cabeceras + variables), para no
partir una categoria a la mitad y quedar automaticamente balanceado si
el CSV fuente cambia. Cabeceras de categoria ahora en `\textbf{}` (antes
`\textit{}`, pedido explicito del usuario: "las categorías se pierden,
creo que mejor poner negrilla").

Las 4 variables "excepcion" (el hogar vulnerable queda MAS cerca de
"siempre pobre" que de "sale", o incluso supera a "nunca pobre" -- ver
parrafo de sintesis en `main.tex`) se marcan con `\dagger` en la etiqueta:
`categoria_ocupacional_jefe` (Asalariado), `deuda_informal_hogar`,
`pct_ninos_cuidado_terceros_hogar`, `tiene_transporte_publico_comunidad`.

INPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_monetaria_2010_2013.csv

OUTPUTS
    paper/tables/tab_perfil_consolidada.tex (1 tabla, 2 columnas lado a lado, 1 pagina)

COMO CORRER
    python src/tabla_perfil_consolidada.py
"""

import re
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

# Caracteres por linea que caben en la columna de variable (columna
# p{0.62\linewidth} de cada minipage a ~0.49\textwidth, scriptsize,
# despues del \resizebox) -- calibrado a mano comparando contra el PDF ya
# compilado: "Obtencion de agua: Acueducto publico" (37 caracteres) ya
# envuelve a 2 lineas, "Tipo de vivienda: Apartamento" (30) no. Se usa
# para estimar cuantas lineas ocupa cada fila al renderizarse y balancear
# las dos columnas por ALTO real, no por numero de filas -- pedido
# explicito del usuario: "que las dos tablas queden del mismo largo para
# que se vean como si fueran una sola" (antes se partia por cantidad de
# filas, pero las etiquetas largas se concentraban mas en una columna que
# en la otra, dejando una visiblemente mas alta).
CHARS_POR_LINEA = 34

# Peso de una cabecera de categoria: 1 linea de texto + el espacio extra
# de \addlinespace (~medio renglon) que NO tiene una fila normal -- sin
# este ajuste, una columna con muchas categorias chicas (varias con 1-2
# variables, ej. "Zona de residencia", "Geoespacial") queda mas alta que
# otra con pocas categorias grandes aunque el total de LINEAS DE TEXTO
# de las dos columnas cuadre exacto (bug encontrado comparando el PDF: 41
# lineas de texto en cada columna, pero la de mas categorias quedaba mas
# larga igual).
PESO_CATEGORIA = 1.5


def _peso_lineas(etiqueta_latex: str) -> float:
    """Estima cuantas lineas ocupa una etiqueta al renderizarse: limpia
    comandos LaTeX simples (\\comando{contenido} -> contenido, \\comando ->
    nada) para aproximar el largo visual real, y divide por
    CHARS_POR_LINEA."""
    texto = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", etiqueta_latex)
    texto = re.sub(r"\\[a-zA-Z]+", "", texto)
    return max(1, -(-len(texto) // CHARS_POR_LINEA))  # ceil sin importar math.ceil

ENCABEZADO_TABULAR = (
    r"    \textbf{Variable} & \textbf{Siempre} & \textbf{Sale} & \textbf{Entra} & \textbf{Nunca} \\"
)


def _construir_entradas(tabla: pd.DataFrame) -> list[tuple[str, str, float, str]]:
    """Una entrada por fila de la tabla final, en orden: ('categoria', nombre,
    peso, categoria) o ('fila', linea_latex_ya_formateada, peso, categoria).
    `peso` = lineas renderizadas estimadas (ver `_peso_lineas`), usado para
    balancear las dos columnas por ALTO real en vez de por cantidad de
    filas. `categoria` (4to campo) permite detectar si el corte cae A
    MITAD de una categoria (ver `_punto_de_corte`)."""
    entradas = []
    categoria_actual = None
    for _, fila in tabla.iterrows():
        if fila["categoria"] != categoria_actual:
            categoria_actual = fila["categoria"]
            entradas.append(("categoria", categoria_actual, PESO_CATEGORIA, categoria_actual))
        formato = inferir_formato(fila["tipo"], fila["nivel_mostrado"])
        etiqueta = etiqueta_fila(fila["variable"], fila["nivel_mostrado"])
        if fila["variable"] in EXCEPCIONES:
            etiqueta += r"$^{\dagger}$"
        valores = " & ".join(formatear_valor(fila[cat], formato) for cat in CATEGORIAS_ORDEN_GRUPO)
        entradas.append(("fila", f"    \\quad {etiqueta} & {valores} \\\\", _peso_lineas(etiqueta), categoria_actual))
    return entradas


def _punto_de_corte(entradas: list[tuple[str, str, float, str]]) -> int:
    """Indice (en `entradas`) donde parte la columna izquierda de la
    derecha -- el mas cercano a la mitad del PESO total (lineas
    renderizadas estimadas, ver `_peso_lineas`), para que ambas columnas
    queden del mismo alto. A diferencia de una version anterior, NO se
    restringe a fronteras de categoria (con solo 10 categorias, esa
    granularidad dejaba una diferencia visible de ~1 linea entre columnas,
    ver docs/decisions.md) -- se permite cortar A MITAD de una categoria;
    `main()` repite el nombre de la categoria con "(cont.)" en la columna
    derecha si eso pasa. Unica restriccion: no cortar justo despues de una
    cabecera de categoria sin ninguna fila de datos debajo en esa misma
    columna (se veria una categoria vacia al final de la izquierda)."""
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
    # Ancho NO se deja a las columnas `r` (natural, sin limite) -- ese fue
    # el bug que hizo que las dos tablas del minipage se superpusieran
    # (el tabular se desbordaba mas alla de 0.49\linewidth). En vez de
    # ajustar anchos a mano, se envuelve en \resizebox (mismo patron ya
    # usado en tab_covariables_transicion_monetaria.tex) para forzar el
    # tabular a caber EXACTO en su mitad de pagina sin importar el ancho
    # natural del contenido.
    lineas = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{p{0.62\linewidth}rrrr}",
        r"  \toprule",
        ENCABEZADO_TABULAR,
        r"  \midrule",
    ]
    for tipo, contenido, _, _ in entradas:
        if tipo == "categoria":
            lineas.append(f"    \\addlinespace\\multicolumn{{5}}{{l}}{{\\textbf{{{contenido}}}}} \\\\")
        else:
            lineas.append(contenido)
    lineas.append(r"  \bottomrule")
    lineas.append(r"\end{tabular}}")
    return "\n".join(lineas)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla = pd.read_csv(TABLES_DIR / "perfil_completo_monetaria_2010_2013.csv")

    assert set(tabla["categoria"]) == set(ORDEN_CATEGORIAS), "categorías del CSV no coinciden con ORDEN_CATEGORIAS"
    orden = {c: i for i, c in enumerate(ORDEN_CATEGORIAS)}
    tabla["_orden"] = tabla["categoria"].map(orden)
    tabla = tabla.sort_values(["_orden", "efecto"], ascending=[True, False])

    entradas = _construir_entradas(tabla)
    corte = _punto_de_corte(entradas)
    izquierda, derecha = entradas[:corte], entradas[corte:]

    # Si el corte cae A MITAD de una categoria (la fila justo antes del
    # corte y la primera fila de la derecha comparten categoria), se
    # repite el nombre de esa categoria al inicio de la derecha, marcado
    # "(cont.)" -- para que quede claro que sigue siendo la misma
    # categoria y no una nueva. Ver docstring de `_punto_de_corte`.
    if izquierda and derecha and derecha[0][3] == izquierda[-1][3]:
        cat = derecha[0][3]
        derecha = [("categoria", f"{cat} (cont.)", PESO_CATEGORIA, cat)] + derecha

    # Relleno final: con pesos en unidades discretas (1 o 2 lineas por
    # fila, ver `_peso_lineas`), el mejor corte disponible casi nunca cae
    # EXACTO a la mitad -- queda un residuo de 1-2 lineas entre columnas
    # (encontrado comparando el PDF compilado, ver docs/decisions.md).
    # Se cierra ese residuo agregando filas en blanco (mismo alto que una
    # fila normal) al final de la columna mas corta, en vez de perseguir
    # un ajuste de pesos que nunca da exacto por la cuantizacion.
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
        r"  \caption{Perfil por grupo de transición, pobreza monetaria "
        r"(2010$\to$2013), las 53 variables robustas. $^{\dagger}$variable "
        r"donde el hogar vulnerable no queda entre ``sale'' y ``siempre "
        r"pobre''/``nunca pobre'' como el resto (ver texto).}",
        r"  \label{tab:perfil_consolidada}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{minipage}[t]{0.49\textwidth}",
        "    " + _tabular(izquierda).replace("\n", "\n    "),
        r"  \end{minipage}\hfill",
        r"  \begin{minipage}[t]{0.49\textwidth}",
        "    " + _tabular(derecha).replace("\n", "\n    "),
        r"  \end{minipage}",
        r"\end{table}",
    ]

    out_path = OUTPUT_DIR / "tab_perfil_consolidada.tex"
    out_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Guardado: {out_path.name} ({len(tabla)} filas, corte en linea {corte}/{len(entradas)})")

    # limpieza: elimina los .tex de versiones anteriores si existen
    for nombre in ("tab_perfil_consolidada_a.tex", "tab_perfil_consolidada_b.tex"):
        ruta_vieja = OUTPUT_DIR / nombre
        if ruta_vieja.exists():
            ruta_vieja.unlink()
            print(f"Eliminado (obsoleto): {nombre}")


if __name__ == "__main__":
    main()
