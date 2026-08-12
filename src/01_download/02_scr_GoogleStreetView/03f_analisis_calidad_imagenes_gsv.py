"""
Análisis de calidad, cobertura y consistencia de las imágenes de Google Street View
====================================================================================

Este script NO vuelve a consultar ni descargar nada. Es un análisis descriptivo
de segunda mano sobre los archivos "finales" entregados por el pipeline GSV:

    data/processed/embeddings/consultas_final.csv
        Una fila por (hogar × ola × radio) consultado contra la GSV Metadata API.
    data/processed/embeddings/descargas_final.csv
        Una fila por imagen efectivamente descargada (exito == True).
    data/processed/embeddings/embeddings_unidos_anonimizado.parquet
        Una fila por imagen con embedding, después de anonimizar identificadores
        de foto (pano_id, heading, nombre_archivo) y reemplazarlos por
        photo_id_uuid (ver union_parquets.py y uuid.py en esta misma carpeta).

Estos tres archivos son el único rastro disponible localmente del proceso de
consulta/descarga: no existen localmente inventario_panos.csv, registro_descargas.csv
ni las imágenes .jpg (se generaron en otra máquina, ver rutas de red en
union_parquets.py / uuid.py). El análisis se basa exclusivamente en lo entregado.

IMPORTANTE — hallazgos de la inspección previa que este script cuantifica y que
NO deben perderse al leer los resultados:

1. Los reportes de texto previos (reporte_cobertura_gsv.txt, reporte_descarga.txt)
   corresponden a una corrida anterior del pipeline y NO coinciden con
   consultas_final.csv / descargas_final.csv (radios distintos, conteos distintos).
   Este script no los usa como fuente y genera un reporte independiente.
2. El código actual de 01_analisis_cobertura_gsv.py define ventanas temporales
   estrictas por ola (VENTANAS_OLA) para decidir qué foto es "elegible" para cada
   ola, pero los datos de descargas_final.csv muestran que esa restricción NO se
   aplicó en la corrida que produjo los datos entregados: la mayoría de las fotos
   tienen fecha de captura muy posterior a la ola que dicen representar. Esto se
   cuantifica explícitamente en la Sección F.
3. La cobertura desigual entre olas dentro del panel balanceado se explica
   enteramente por disponibilidad real de panorama (status == "OK"): se verificó
   una correspondencia empírica 1 a 1 entre "hay panorama OK en algún radio" y
   "se descargó una imagen" (Sección C). No es un artefacto de una regla de
   deduplicación entre olas.
4. Existe duplicación real de imágenes: la misma fotografía física (mismo
   pano_id + heading) puede quedar asignada a más de un hogar cuando ambos
   comparten el panorama más cercano. Esto se cuantifica en la Sección D.

Outputs
-------
- Reporte de texto:              paper/tables/reporte_analisis_imagenes_gsv.txt
- Tablas LaTeX (\\input-eables):  paper/tables/tab_gsv_*.tex
- Figuras (PNG, 300 dpi):        paper/figures/fig_gsv_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# NOTA DE CALIDAD DE PIPELINE: este mismo directorio contiene un script del
# estudiante llamado uuid.py (usado para anonimizar photo_id). Python inserta
# automáticamente el directorio del script en ejecución al inicio de sys.path,
# así que cualquier "import uuid" —incluido el que hace matplotlib
# internamente— resuelve a ese archivo local en vez de al módulo estándar de
# la librería, y falla. Se retira aquí explícitamente para poder ejecutar este
# análisis; ver Sección H del reporte para la recomendación de renombrarlo.
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[3]           # .../tesis_vulnerabilidad
DATA_DIR = REPO_ROOT / "data" / "processed" / "embeddings"
FIG_DIR = REPO_ROOT / "paper" / "figures"
TAB_DIR = REPO_ROOT / "paper" / "tables"

CONSULTAS_PATH = DATA_DIR / "consultas_final.csv"
DESCARGAS_PATH = DATA_DIR / "descargas_final.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings_unidos_anonimizado.parquet"

REPORTE_PATH = TAB_DIR / "reporte_analisis_imagenes_gsv.txt"

# Ventanas temporales tal como están definidas en el script 01 actual
# (01_analisis_cobertura_gsv.py, VENTANAS_OLA). Se usan aquí SOLO para
# contrastar contra los datos entregados, no para filtrar nada.
VENTANAS_OLA = {2010: (None, 2010), 2013: (2011, 2013), 2016: (2014, 2016)}
OLAS = [2010, 2013, 2016]

# ──────────────────────────────────────────────────────────────────────────────
# ESTILO DE FIGURAS — paleta categórica fija (nunca ciclada) + colores de estado
# reservados para semántica de calidad (OK / problema). Fuente: skill dataviz.
# ──────────────────────────────────────────────────────────────────────────────

COLOR_OLA = {2010: "#2a78d6", 2013: "#eb6834", 2016: "#1baf7a"}   # blue/orange/aqua
COLOR_STATUS = {
    "OK": "#0ca30c",              # good
    "ZERO_RESULTS": "#fab219",    # warning
    "SIN_COORD_VALIDA": "#8a8a86",  # muted gray — no es un fallo de la API
    "NOT_FOUND": "#ec835a",       # serious
    "UNKNOWN_ERROR": "#d03b3b",   # critical
}
COLOR_MUTED = "#8a8a86"
COLOR_TEXT = "#2b2b28"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.edgecolor": "#c9c8c0",
    "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e4dd",
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
})


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

def cargar_datos():
    for p in (CONSULTAS_PATH, DESCARGAS_PATH, EMBEDDINGS_PATH):
        if not p.exists():
            sys.exit(f"ERROR: no se encontró {p}. Revisa data/processed/embeddings/.")

    consultas = pd.read_csv(CONSULTAS_PATH, dtype={"consecutivo": str})

    # descargas_final.csv tiene filas corruptas: 'mensaje_error' contiene comas
    # sin comillar (ej. "Contenido inválido: 8,872 bytes, inicio=ffd8ffe0"), lo
    # que rompe el parser C de pandas en 7 líneas. Esas 7 filas corresponden
    # exactamente a las 7 descargas fallidas (columna exito=False) — no afectan
    # ninguna imagen usada en los embeddings, pero se documentan como hallazgo
    # de calidad de datos en la Sección H en vez de descartarse en silencio.
    n_lineas_archivo = sum(1 for _ in open(DESCARGAS_PATH, encoding="utf-8", errors="replace")) - 1
    descargas = pd.read_csv(
        DESCARGAS_PATH, dtype={"consecutivo": str}, engine="python", on_bad_lines="skip"
    )
    n_filas_corruptas = n_lineas_archivo - len(descargas)

    embeddings = pd.read_parquet(
        EMBEDDINGS_PATH,
        columns=["consecutivo", "ola", "llave", "llave_n16", "n_repeticiones", "photo_id_uuid"],
    )

    return consultas, descargas, embeddings, n_filas_corruptas


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────

def fmt(n, decimales=0):
    if pd.isna(n):
        return "NA"
    return f"{n:,.{decimales}f}"


def pct(num, den):
    if den == 0:
        return float("nan")
    return 100 * num / den


class Reporte:
    """Acumula texto para el .txt final y lo va imprimiendo por consola."""

    def __init__(self):
        self.lineas = []

    def sec(self, titulo):
        bloque = ["", "=" * 78, titulo, "=" * 78]
        self.lineas.extend(bloque)
        print("\n".join(bloque))

    def sub(self, titulo):
        bloque = ["", f"── {titulo} " + "─" * max(0, 74 - len(titulo))]
        self.lineas.extend(bloque)
        print("\n".join(bloque))

    def p(self, texto=""):
        self.lineas.append(texto)
        print(texto)

    def guardar(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lineas) + "\n", encoding="utf-8")


def guardar_tabla_tex(df: pd.DataFrame, nombre: str, caption: str, label: str,
                       col_format: str | None = None, index=False):
    """Guarda un DataFrame como tabla LaTeX legible: enteros con separador de
    miles, floats con 1 decimal, sin escapar manualmente los nombres de columna
    (deja que to_latex(escape=True) escape '%' una sola vez)."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    path = TAB_DIR / f"{nombre}.tex"

    df_fmt = df.copy()
    for col in df_fmt.columns:
        if str(col).strip().lower() == "ola":
            df_fmt[col] = df_fmt[col].astype(int).astype(str)
        elif pd.api.types.is_float_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(lambda v: "--" if pd.isna(v) else f"{v:,.1f}")
        elif pd.api.types.is_integer_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(lambda v: f"{v:,}")

    latex = df_fmt.to_latex(
        index=index,
        escape=True,
        caption=caption,
        label=label,
        column_format=col_format,
        na_rep="--",
    )
    path.write_text(latex, encoding="utf-8")
    return path


def guardar_fig(fig, nombre: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{nombre}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN A — RESUMEN GENERAL
# ──────────────────────────────────────────────────────────────────────────────

def seccion_a(rep: Reporte, consultas, descargas, embeddings, n_filas_corruptas):
    rep.sec("SECCIÓN A — RESUMEN GENERAL")

    obs_hogar_ola = consultas[["consecutivo", "ola"]].drop_duplicates()
    n_obs = len(obs_hogar_ola)
    n_consultas_filas = len(consultas)
    n_radios = consultas["radio_m"].nunique()

    rep.p(f"Observaciones hogar × ola en el archivo de consultas: {fmt(n_obs)}")
    rep.p(f"Filas totales en consultas_final.csv (hogar×ola×radio): {fmt(n_consultas_filas)}")
    rep.p(f"Radios de búsqueda consultados: {sorted(consultas['radio_m'].dropna().unique())} m")
    rep.p("")
    rep.p("Distribución de 'status' de la Metadata API (a nivel hogar×ola×radio):")
    status_counts = consultas["status"].value_counts(dropna=False)
    for k, v in status_counts.items():
        rep.p(f"  {str(k):20s} {fmt(v):>10s}  ({pct(v, n_consultas_filas):5.1f}%)")

    rep.p("")
    n_descargadas = len(descargas)
    rep.p(f"Imágenes efectivamente descargadas (exito=True): {fmt(n_descargadas)}")
    rep.p(f"Filas corruptas en descargas_final.csv (comas sin comillar en "
          f"mensaje_error, todas de descargas fallidas): {n_filas_corruptas}")
    rep.p(f"Imágenes con embedding extraído (las 4 arquitecturas): {fmt(len(embeddings))}")
    if n_descargadas == len(embeddings):
        rep.p("  → Coincide exactamente con las descargas exitosas: no se perdió "
              "ninguna imagen entre la descarga y la extracción de embeddings.")
    else:
        rep.p(f"  ⚠ DIFERENCIA de {n_descargadas - len(embeddings)} filas entre "
              "descargas exitosas y embeddings — revisar antes de usar los datos.")

    return {
        "n_obs_hogar_ola": n_obs,
        "n_consultas_filas": n_consultas_filas,
        "status_counts": status_counts,
        "n_descargadas": n_descargadas,
        "n_embeddings": len(embeddings),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN B — COBERTURA POR OLA
# ──────────────────────────────────────────────────────────────────────────────

def seccion_b(rep: Reporte, consultas, descargas):
    rep.sec("SECCIÓN B — COBERTURA POR OLA")

    filas = []
    for ola in OLAS:
        c_ola = consultas[consultas["ola"] == ola]
        hogares_consultados = c_ola["consecutivo"].nunique()
        hogares_validos = c_ola.loc[c_ola["coord_valida_ola"] == 1, "consecutivo"].nunique()
        con_pano = (
            c_ola[c_ola["status"] == "OK"]["consecutivo"].nunique()
        )
        con_imagen = descargas.loc[descargas["ola"] == ola, "consecutivo"].nunique()
        filas.append({
            "ola": ola,
            "hogares_consultados": hogares_consultados,
            "hogares_coord_valida": hogares_validos,
            "hogares_con_panorama_ok": con_pano,
            "hogares_con_imagen_descargada": con_imagen,
            "cobertura_%_sobre_validos": pct(con_imagen, hogares_validos),
        })
    tabla = pd.DataFrame(filas)

    rep.p("Hogares (consecutivo únicos) por etapa del embudo, por ola:")
    rep.p(tabla.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    guardar_tabla_tex(
        tabla.rename(columns={
            "ola": "Ola",
            "hogares_consultados": "Consultados",
            "hogares_coord_valida": "Coord. válida",
            "hogares_con_panorama_ok": "Con panorama",
            "hogares_con_imagen_descargada": "Con imagen",
            "cobertura_%_sobre_validos": "Cobertura (%)",
        }),
        "tab_gsv_cobertura_ola",
        "Cobertura de Google Street View por ola de la ELCA: hogares consultados, con "
        "coordenada válida, con panorama disponible (status OK en algún radio) y con "
        "imagen efectivamente descargada.",
        "tab:gsv_cobertura_ola",
    )

    # Figura: embudo por ola
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    etapas = ["hogares_consultados", "hogares_coord_valida", "hogares_con_panorama_ok",
              "hogares_con_imagen_descargada"]
    etiquetas = ["Consultados", "Coord.\nválida", "Con\npanorama", "Con imagen\ndescargada"]
    x = np.arange(len(etapas))
    width = 0.25
    for i, ola in enumerate(OLAS):
        vals = tabla.loc[tabla["ola"] == ola, etapas].values.flatten()
        ax.bar(x + (i - 1) * width, vals, width=width, label=str(ola), color=COLOR_OLA[ola])
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Hogares (consecutivo único)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.suptitle("Embudo de cobertura de Google Street View por ola", y=0.99, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Ola", frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(0.55, 0.94))
    fig.subplots_adjust(top=0.78)
    guardar_fig(fig, "fig_gsv_embudo_cobertura_ola")

    # Figura: status por ola (barras apiladas, a nivel hogar×ola×radio→ usamos
    # nivel hogar×ola con el mejor status observado en cualquier radio)
    mejor_status = (
        consultas.sort_values("radio_m")
        .groupby(["consecutivo", "ola"])["status"]
        .apply(lambda s: "OK" if (s == "OK").any() else s.iloc[0])
        .reset_index()
    )
    orden_status = ["OK", "ZERO_RESULTS", "SIN_COORD_VALIDA", "NOT_FOUND", "UNKNOWN_ERROR"]
    tabla_status = (
        mejor_status.groupby(["ola", "status"]).size().unstack(fill_value=0)
        .reindex(columns=orden_status, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bottom = np.zeros(len(tabla_status))
    for status in orden_status:
        vals = tabla_status[status].values
        ax.bar(tabla_status.index.astype(str), vals, bottom=bottom, label=status,
               color=COLOR_STATUS.get(status, COLOR_MUTED))
        bottom += vals
    ax.set_ylabel("Hogares (consecutivo único)")
    ax.set_xlabel("Ola")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.suptitle("Resultado de la consulta a la Metadata API por hogar, por ola\n"
                  "(mejor status entre los radios consultados)", y=0.99, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.55, 0.86))
    fig.subplots_adjust(top=0.7)
    guardar_fig(fig, "fig_gsv_status_por_ola")

    return tabla


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN C — PANEL BALANCEADO Y PATRÓN DE COBERTURA ENTRE OLAS
# ──────────────────────────────────────────────────────────────────────────────

def seccion_c(rep: Reporte, consultas, descargas):
    rep.sec("SECCIÓN C — HOGARES CUBIERTOS Y NO CUBIERTOS (PANEL DE 3 OLAS)")

    hh_ola = consultas.groupby("consecutivo")["ola"].nunique()
    hh_3olas = set(hh_ola[hh_ola == 3].index)
    rep.p(f"Hogares (consecutivo) presentes en las tres olas en consultas_final.csv: "
          f"{fmt(len(hh_3olas))}")

    c3 = consultas[consultas["consecutivo"].isin(hh_3olas)]
    status_ok_hogar_ola = (
        c3.groupby(["consecutivo", "ola"])["status"].apply(lambda s: (s == "OK").any())
        .reset_index(name="status_ok_algun_radio")
    )

    tiene_foto = (
        descargas.groupby(["consecutivo", "ola"]).size().reset_index(name="n_fotos")
    )
    tiene_foto["tiene_foto"] = True

    verificacion = status_ok_hogar_ola.merge(
        tiene_foto[["consecutivo", "ola", "tiene_foto"]],
        on=["consecutivo", "ola"], how="left"
    )
    verificacion["tiene_foto"] = verificacion["tiene_foto"].where(
        verificacion["tiene_foto"].notna(), False
    ).astype(bool)

    cruce = pd.crosstab(verificacion["status_ok_algun_radio"], verificacion["tiene_foto"])
    rep.sub("Verificación: ¿'panorama OK' implica 'imagen descargada'?")
    rep.p("Cruce (hogar×ola, solo hogares presentes en las 3 olas):")
    rep.p(cruce.to_string())
    discordancias = int(cruce.values.sum() - cruce.values.diagonal().sum())
    if discordancias == 0:
        rep.p("→ Correspondencia perfecta 1 a 1: la brecha de cobertura entre olas dentro "
              "del panel de 3 olas se explica enteramente por disponibilidad real de "
              "panorama (status=OK en algún radio), NO por una regla adicional de "
              "deduplicación entre olas del pipeline.")
    else:
        rep.p(f"→ {discordancias} discordancias encontradas entre status OK y tener imagen "
              "descargada: requiere revisión adicional del script 02.")

    # Patrón de cobertura fotográfica: 0,1,2,3 olas con imagen, y qué combinación
    hh_fotos_olas = (
        descargas.groupby("consecutivo")["ola"].apply(lambda s: frozenset(s.unique()))
    )
    hh_fotos_olas = hh_fotos_olas.reindex(sorted(hh_3olas)).apply(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )
    combinaciones = hh_fotos_olas.value_counts()
    etiquetas_legibles = {
        frozenset(): "Ninguna ola",
        frozenset({2010}): "Solo 2010",
        frozenset({2013}): "Solo 2013",
        frozenset({2016}): "Solo 2016",
        frozenset({2010, 2013}): "2010 + 2013",
        frozenset({2010, 2016}): "2010 + 2016",
        frozenset({2013, 2016}): "2013 + 2016",
        frozenset({2010, 2013, 2016}): "2010 + 2013 + 2016",
    }
    tabla_patron = pd.DataFrame({
        "combinación": [etiquetas_legibles.get(k, str(set(k))) for k in combinaciones.index],
        "n_hogares": combinaciones.values,
        "%": [pct(v, len(hh_3olas)) for v in combinaciones.values],
    }).sort_values("n_hogares", ascending=False)

    rep.sub("Patrón de cobertura fotográfica dentro del panel de 3 olas")
    rep.p(tabla_patron.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    guardar_tabla_tex(
        tabla_patron.rename(columns={"combinación": "Combinación de olas con imagen",
                                      "n_hogares": "Hogares", "%": "%"}),
        "tab_gsv_patron_cobertura",
        "Patrón de cobertura fotográfica dentro del panel de hogares presentes en las "
        "tres olas de la ELCA (2010, 2013, 2016). Un hogar puede tener coordenada válida "
        "en las tres olas y aun así carecer de imagen en una o más de ellas por ausencia "
        "de panorama de Google Street View cercano.",
        "tab:gsv_patron_cobertura",
    )

    orden_plot = ["2010 + 2013 + 2016", "2013 + 2016", "2010 + 2013", "2010 + 2016",
                  "Solo 2016", "Solo 2013", "Solo 2010", "Ninguna ola"]
    tp = tabla_patron.set_index("combinación").reindex(orden_plot).dropna()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colores = ["#1baf7a" if c == "2010 + 2013 + 2016" else ("#8a8a86" if c == "Ninguna ola" else "#2a78d6")
               for c in tp.index]
    ax.barh(tp.index, tp["n_hogares"], color=colores)
    ax.set_xlabel("Hogares")
    ax.set_title("¿En cuántas y cuáles olas tiene imagen cada hogar?\n"
                  "(panel de hogares presentes en las tres olas)")
    for y, v in enumerate(tp["n_hogares"]):
        ax.text(v, y, f"  {int(v):,}", va="center", fontsize=8)
    guardar_fig(fig, "fig_gsv_patron_cobertura")

    n_completo = int(tabla_patron.loc[
        tabla_patron["combinación"] == "2010 + 2013 + 2016", "n_hogares"
    ].sum())
    n_con_alguna = int(tabla_patron.loc[
        tabla_patron["combinación"] != "Ninguna ola", "n_hogares"
    ].sum())

    return {
        "n_hogares_3olas": len(hh_3olas),
        "n_completo_3olas": n_completo,
        "n_con_alguna_imagen": n_con_alguna,
        "tabla_patron": tabla_patron,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN D — IMÁGENES ÚNICAS Y DUPLICACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def seccion_d(rep: Reporte, embeddings):
    rep.sec("SECCIÓN D — IMÁGENES ÚNICAS Y DUPLICACIÓN ENTRE HOGARES")

    n_filas = len(embeddings)
    n_unicas = embeddings["photo_id_uuid"].nunique()
    n_compartidas = int((embeddings["n_repeticiones"] > 1).sum())

    rep.p(f"Filas (hogar × ola con imagen asignada): {fmt(n_filas)}")
    rep.p(f"Fotografías físicamente distintas (photo_id_uuid único, construido a partir "
          f"de pano_id + heading antes de anonimizar): {fmt(n_unicas)} "
          f"({pct(n_unicas, n_filas):.1f}% del total de filas)")
    rep.p(f"Filas cuya fotografía es compartida con al menos otro hogar×ola "
          f"(n_repeticiones > 1): {fmt(n_compartidas)} ({pct(n_compartidas, n_filas):.1f}%)")
    rep.p("")
    rep.p("Esto ocurre cuando dos o más hogares (o el mismo hogar en distintas sub-divisiones) "
          "comparten el panorama de Street View más cercano: reciben literalmente la misma "
          "fotografía y, por lo tanto, el mismo embedding — no hay dos observaciones "
          "visualmente independientes en esos casos, aunque cuenten como dos filas del panel.")

    rep.sub("Distribución de n_repeticiones (nº de hogares×ola que comparten la misma foto)")
    dist = embeddings.drop_duplicates("photo_id_uuid")["n_repeticiones"].value_counts().sort_index()
    rep.p(dist.to_string())
    rep.p(f"Máximo de hogares×ola distintos compartiendo una sola fotografía: "
          f"{int(embeddings['n_repeticiones'].max())}")

    tabla_resumen = pd.DataFrame({
        "Concepto": [
            "Filas (hogar × ola con imagen)",
            "Fotografías físicamente distintas",
            "Filas con foto compartida (repetida)",
        ],
        "N": [n_filas, n_unicas, n_compartidas],
        "%": [100.0, pct(n_unicas, n_filas), pct(n_compartidas, n_filas)],
    })
    guardar_tabla_tex(
        tabla_resumen,
        "tab_gsv_duplicacion",
        "Duplicación de fotografías entre hogares: número de filas hogar×ola frente al "
        "número de fotografías físicamente distintas (mismo \\texttt{pano\\_id} y "
        "\\texttt{heading}).",
        "tab:gsv_duplicacion",
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    conteo_rep = embeddings.drop_duplicates("photo_id_uuid")["n_repeticiones"]
    bins = np.arange(1, conteo_rep.max() + 2) - 0.5
    ax.hist(conteo_rep, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Nº de hogares × ola que comparten la misma fotografía")
    ax.set_ylabel("Nº de fotografías distintas")
    ax.set_yscale("log")
    ax.set_title("Reutilización de una misma fotografía entre hogares distintos")
    guardar_fig(fig, "fig_gsv_duplicacion_fotos")

    return {"n_filas": n_filas, "n_unicas": n_unicas, "n_compartidas": n_compartidas}


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN E — IMÁGENES POR HOGAR
# ──────────────────────────────────────────────────────────────────────────────

def seccion_e(rep: Reporte, descargas):
    rep.sec("SECCIÓN E — IMÁGENES POR HOGAR (CONSECUTIVO) Y POR OLA")

    por_hogar_ola = descargas.groupby(["consecutivo", "ola"]).size()
    rep.p("Distribución del nº de imágenes descargadas por (hogar, ola):")
    rep.p(por_hogar_ola.value_counts().sort_index().to_string())
    rep.p("")
    rep.p("Un mismo 'consecutivo' puede tener más de una imagen en la misma ola cuando el "
          "hogar se dividió en sub-hogares (splits) con coordenadas propias: cada sub-hogar "
          "puede generar su propia consulta y, si tiene panorama distinto, su propia imagen. "
          "No es duplicación: son sub-hogares distintos comprimidos bajo el mismo 'consecutivo'.")

    por_hogar_total = descargas.groupby("consecutivo").size()
    rep.sub("Total de imágenes por hogar (sumando las tres olas)")
    rep.p(por_hogar_total.describe().to_string())

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bins = np.arange(1, por_hogar_total.max() + 2) - 0.5
    ax.hist(por_hogar_total, bins=bins, color="#eb6834", edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Nº total de imágenes descargadas por hogar (consecutivo)")
    ax.set_ylabel("Nº de hogares")
    ax.set_title("Imágenes disponibles por hogar (todas las olas)")
    guardar_fig(fig, "fig_gsv_imagenes_por_hogar")

    return por_hogar_total


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN F — FECHAS Y DISTRIBUCIÓN TEMPORAL
# ──────────────────────────────────────────────────────────────────────────────

def seccion_f(rep: Reporte, descargas):
    rep.sec("SECCIÓN F — FECHAS DE LAS IMÁGENES Y DESFASE TEMPORAL RESPECTO A LA OLA")

    d = descargas.copy()
    d["anio_pano"] = d["fecha_pano"].str[:4].astype(int)

    rep.p("Rango de años de captura de los panoramas efectivamente descargados:")
    rep.p(f"  Mínimo: {d['anio_pano'].min()}   Máximo: {d['anio_pano'].max()}   "
          f"Mediana: {int(d['anio_pano'].median())}")

    rep.sub("Año de captura del panorama, por ola de la encuesta")
    resumen_anio = d.groupby("ola")["anio_pano"].describe()[["min", "25%", "50%", "75%", "max"]]
    rep.p(resumen_anio.to_string())

    def elegible(row):
        lo, hi = VENTANAS_OLA[row["ola"]]
        if lo is not None and row["anio_pano"] < lo:
            return False
        if hi is not None and row["anio_pano"] > hi:
            return False
        return True

    d["dentro_ventana_ola"] = d.apply(elegible, axis=1)
    rep.sub("Contraste contra la ventana temporal 'elegible' definida en el código actual "
            "del script 01 (VENTANAS_OLA: ≤2010 / 2011–2013 / 2014–2016)")
    tabla_ventana = (
        d.groupby("ola")["dentro_ventana_ola"]
        .agg(n_total="size", n_dentro_ventana="sum")
        .assign(pct_dentro_ventana=lambda t: 100 * t["n_dentro_ventana"] / t["n_total"])
        .reset_index()
    )
    rep.p(tabla_ventana.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    pct_fuera_total = pct((~d["dentro_ventana_ola"]).sum(), len(d))
    rep.p("")
    rep.p(f"→ {pct_fuera_total:.1f}% de las imágenes descargadas tiene una fecha de captura "
          "FUERA de la ventana temporal que el pipeline define como 'elegible' para su "
          "propia ola. Para la ola 2010 la proporción dentro de ventana es prácticamente "
          "nula, porque la cobertura histórica de Street View en Colombia con fecha "
          "verificable es casi inexistente antes de 2012. Esto indica que los datos "
          "entregados NO se generaron aplicando ese filtro de elegibilidad temporal "
          "(está en el código actual del script 01 pero no se refleja en descargas_final.csv), "
          "y de forma independiente confirma que la mayoría de las imágenes usadas para "
          "caracterizar el entorno de un hogar en una ola dada fueron tomadas años — en "
          "muchos casos más de una década — después de esa ola.")

    guardar_tabla_tex(
        tabla_ventana.rename(columns={
            "ola": "Ola", "n_total": "Imágenes", "n_dentro_ventana": "Dentro de ventana",
            "pct_dentro_ventana": "% dentro de ventana",
        }),
        "tab_gsv_ventana_temporal",
        "Contraste entre la fecha real de captura del panorama descargado y la ventana "
        "temporal que el script de cobertura define como elegible para cada ola "
        "(2010: $\\leq$2010; 2013: 2011--2013; 2016: 2014--2016).",
        "tab:gsv_ventana_temporal",
    )

    # Figura: histograma de año de captura por ola, con línea vertical en el año de la ola
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    bins = np.arange(2011, 2028) - 0.5
    for ax, ola in zip(axes, OLAS):
        vals = d.loc[d["ola"] == ola, "anio_pano"]
        ax.hist(vals, bins=bins, color=COLOR_OLA[ola], edgecolor="white", linewidth=0.4)
        ax.axvline(ola, color="#d03b3b", linestyle="--", linewidth=1.3)
        ax.set_title(f"Ola {ola}")
        ax.set_xlabel("Año de captura de la foto")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Nº de imágenes")
    fig.suptitle("Año de captura de las fotos descargadas vs. año de la ola encuestada "
                  "(línea roja)", y=1.04)
    guardar_fig(fig, "fig_gsv_desfase_temporal")

    return {"tabla_ventana": tabla_ventana, "pct_fuera_ventana": pct_fuera_total}


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN G — DISTANCIA AL PANORAMA
# ──────────────────────────────────────────────────────────────────────────────

def seccion_g(rep: Reporte, descargas):
    rep.sec("SECCIÓN G — DISTANCIA ENTRE EL HOGAR Y EL PANORAMA UTILIZADO")

    desc = descargas["distancia_m"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
    rep.p(desc.to_string())
    rep.p("")
    rep.p("Distancias grandes implican que la fotografía puede no corresponder al frente "
          "del hogar sino a un segmento de calle distinto (p. ej. una vía principal cercana "
          "en vez de la calle interna donde vive el hogar).")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(descargas["distancia_m"].clip(upper=500), bins=50, color="#4a3aa7",
             edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Distancia al panorama (m, recortado en 500 m)")
    ax.set_ylabel("Nº de imágenes")
    ax.set_title("Distancia entre la coordenada del hogar y el panorama utilizado")
    guardar_fig(fig, "fig_gsv_distancia_panorama")


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN H — CONSISTENCIA Y LIMITACIONES ADICIONALES
# ──────────────────────────────────────────────────────────────────────────────

def seccion_h(rep: Reporte, consultas, descargas, embeddings, n_filas_corruptas):
    rep.sec("SECCIÓN H — CONSISTENCIA DE LOS DATOS Y LIMITACIONES ADICIONALES")

    rep.sub("H.1 — Filas corruptas en descargas_final.csv")
    rep.p(f"{n_filas_corruptas} filas del archivo tienen comas sin comillar dentro de "
          "'mensaje_error' (ej. 'Contenido inválido: 8,872 bytes, inicio=ffd8ffe0'), lo que "
          "rompe el parser por defecto de pandas/Stata/R. Las 7 filas corresponden todas a "
          "descargas fallidas (exito=False) y no afectan ninguna imagen usada en los "
          "embeddings, pero el archivo debería regenerarse con comillado CSV correcto para "
          "evitar que un análisis futuro las descarte en silencio o falle al leer el archivo.")

    rep.sub("H.2 — Mensajes de error también muestran mojibake de codificación")
    rep.p("El texto de mensaje_error contiene secuencias como 'invÃ¡lido' en vez de "
          "'inválido', indicando una doble codificación UTF-8/Latin-1 en algún punto del "
          "pipeline de descarga o de guardado del CSV. No afecta variables numéricas, pero "
          "conviene corregirlo si el texto de error se usa para clasificar fallos.")

    rep.sub("H.3 — Identificador 'llave' faltante en algunas filas de ola 2016")
    n_2016 = (embeddings["ola"] == 2016).sum()
    n_falta_llave_2016 = ((embeddings["ola"] == 2016) & (embeddings["llave"].isna())).sum()
    rep.p(f"{n_falta_llave_2016} de {n_2016} filas de ola 2016 ({pct(n_falta_llave_2016, n_2016):.1f}%) "
          "no tienen 'llave' (identificador de panel 2013) a pesar de no ser ola 2010. "
          "Puede deberse a sub-hogares 2016 sin contraparte directa en 2013 (nuevas "
          "divisiones); se documenta para que la unión con el panel ELCA lo tenga en cuenta "
          "y no se asuma como error de construcción sin verificar antes con las tablas ELCA.")

    rep.sub("H.4 — Radios de búsqueda: código actual vs. reportes previos")
    radios_actuales = sorted(int(r) for r in consultas["radio_m"].dropna().unique())
    rep.p(f"consultas_final.csv usa los radios {radios_actuales} metros. El código actual de "
          "01_analisis_cobertura_gsv.py (CONFIG['radios_m']) usa [50, 100, 200, 400], y el "
          "reporte de texto reporte_cobertura_gsv.txt (de una corrida anterior) documenta "
          "[50, 100, 200, 500]. Los tres no coinciden entre sí. Esto confirma que los "
          "reportes .txt existentes en data/processed/embeddings/ no reflejan ni el código "
          "actual ni los datos finales entregados, y no deberían citarse como fuente en la "
          "tesis: la fuente autoritativa es consultas_final.csv / descargas_final.csv.")

    rep.sub("H.5 — Carpeta de datos mixta con un pipeline no relacionado")
    rep.p("data/processed/embeddings/ contiene, además de los archivos de Street View, "
          "reportes de un pipeline distinto basado en imágenes satelitales Sentinel-1 "
          "(reporte_coleccion_s1.txt, reporte_composiciones_temporales.txt, "
          "reporte_conexion_gee.txt, reporte_extraccion_buffer.txt, "
          "reporte_metricas_textura.txt). Quedan fuera del alcance de este análisis, que se "
          "limita a Google Street View, pero conviene separarlos en carpetas distintas para "
          "evitar confusión sobre qué reporte corresponde a qué fuente de datos.")

    rep.sub("H.6 — 'uuid.py' local sombrea la librería estándar")
    rep.p("El script uuid.py que el estudiante dejó en esta misma carpeta (usado para "
          "generar photo_id_uuid al anonimizar) tiene el mismo nombre que el módulo "
          "estándar de Python 'uuid'. Python antepone el directorio del script en "
          "ejecución a sys.path, así que cualquier código —incluyendo matplotlib, que "
          "depende de 'uuid' internamente— que se ejecute desde esta carpeta importará "
          "ese archivo en vez de la librería estándar y fallará. Este mismo script tuvo que "
          "neutralizarlo explícitamente para poder generar las figuras. Se recomienda "
          "renombrar uuid.py (p. ej. a anonimizar_photo_id.py) antes de que rompa otro "
          "script del pipeline.")

    rep.sub("H.7 — Cobertura urbano/rural")
    if "zona_archivo" in consultas.columns:
        base = consultas[["consecutivo", "ola", "zona_archivo"]].drop_duplicates()
        ok = (
            consultas[consultas["status"] == "OK"][["consecutivo", "ola"]]
            .drop_duplicates()
        )
        ok["con_pano"] = True
        base = base.merge(ok, on=["consecutivo", "ola"], how="left")
        base["con_pano"] = base["con_pano"].where(base["con_pano"].notna(), False).astype(bool)
        resumen_zona = base.groupby("zona_archivo")["con_pano"].agg(["sum", "count"])
        resumen_zona["cobertura_%"] = 100 * resumen_zona["sum"] / resumen_zona["count"]
        rep.p(resumen_zona.to_string())
        rep.p("")
        rep.p("La cobertura de Street View es sistemáticamente mayor en zonas urbanas que "
              "rurales (esperable dado el patrón de cobertura vial de Google), lo que "
              "introduce un sesgo de selección espacial en qué hogares terminan teniendo "
              "variables de entorno visual disponibles.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    rep = Reporte()
    rep.p("=" * 78)
    rep.p("ANÁLISIS DE CALIDAD Y COBERTURA — IMÁGENES GOOGLE STREET VIEW (ELCA)")
    rep.p("Fuente: data/processed/embeddings/{consultas_final.csv, descargas_final.csv, "
          "embeddings_unidos_anonimizado.parquet}")
    rep.p("=" * 78)

    consultas, descargas, embeddings, n_filas_corruptas = cargar_datos()

    seccion_a(rep, consultas, descargas, embeddings, n_filas_corruptas)
    seccion_b(rep, consultas, descargas)
    seccion_c(rep, consultas, descargas)
    seccion_d(rep, embeddings)
    seccion_e(rep, descargas)
    seccion_f(rep, descargas)
    seccion_g(rep, descargas)
    seccion_h(rep, consultas, descargas, embeddings, n_filas_corruptas)

    rep.guardar(REPORTE_PATH)
    print(f"\nReporte guardado en: {REPORTE_PATH}")
    print(f"Figuras guardadas en: {FIG_DIR}")
    print(f"Tablas LaTeX guardadas en: {TAB_DIR}")


if __name__ == "__main__":
    main()
