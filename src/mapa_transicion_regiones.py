"""
mapa_transicion_regiones.py
=====================================
Panel esquematico de la distribucion espacial de los 4 grupos de
transicion de pobreza (2010->2013) por `region` de ELCA, con iluminacion
nocturna (DMSP) superpuesta -- Panel A de las paginas de resultados
(pobreza monetaria / IPM) de la seccion de covariables por grupo de
transicion. Este script NO recalcula nada, solo formatea/grafica los CSV
ya producidos por `eda_transicion_covariables.py`.

POR QUE "ESQUEMATICO" Y NO UN COROPLETICO REAL (decision del usuario,
2026-09-04, ver hilo de la conversacion)
----------------------------------------------------------------------
1. `id_dpto`/`id_mpio` de ELCA son "identificador falso" segun el propio
   diccionario de la encuesta (docs/decisions.md linea 739,
   elca_2010_unido.pdf HR4/HR5) -- no hay departamento/municipio real
   verificable a nivel de hogar.
2. `region` (la unica variable geografica valida) tampoco es un poligono
   administrativo: segun la ficha metodologica publica de CEDE-Uniandes
   (encuestalongitudinal.uniandes.edu.co / datoscede.uniandes.edu.co), las
   4 "microrregiones" rurales (Atlántica Media, Cundi-Boyacense, Eje
   Cafetero, Centro-Oriente) son colecciones de MUNICIPIOS PUNTUALES
   dispersos dentro de 2-3 departamentos cada una (ej. Cundi-Boyacense
   incluye Saboyá en Boyacá, Simijaca/Susa en Cundinamarca y Puente
   Nacional en Santander -- no los departamentos completos). Pintar un
   poligono relleno por region implicaria cobertura geografica que la
   encuesta no tiene.

Por eso este mapa es un DIAGRAMA (burbujas en posicion aproximada sobre
el contorno nacional real de Colombia, NO un choropleth de poligonos por
region) -- da orientacion espacial sin fingir precision cartografica que
los datos no respaldan. Las coordenadas de cada burbuja en
`COORDENADAS_REGION` son aproximaciones a ojo de la ubicacion general de
cada region (no un centroide calculado), documentadas como tal en el
titulo de la figura.

INPUTS
    outputs/tables/eda_transicion_covariables/region_x_categoria_{nombre}.csv
    outputs/tables/eda_transicion_covariables/region_x_categoria_n... (ver nota)
    outputs/tables/eda_transicion_covariables/dmsp_por_region_{nombre}.csv
    data/interim/geo_referencia/gadm41_COL_0.shp (contorno nacional, GADM 4.1,
        descargado de geodata.ucdavis.edu -- solo el nivel 0/pais, sin
        subdivisiones, para dar contexto visual)

OUTPUTS
    outputs/figures/eda_transicion_covariables/mapa_transicion_regiones_{monetaria,ipm}.png

COMO CORRER
    python src/mapa_transicion_regiones.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import ConnectionPatch, Wedge
from matplotlib.transforms import Affine2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda_transicion_covariables"
CONTORNO_PATH = PROJECT_ROOT / "data" / "interim" / "geo_referencia" / "gadm41_COL_0.shp"

CATEGORIAS_ORDEN = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]
# Misma paleta y mismo mapeo categoria->color que graf_transiciones_generico
# (build_pobreza_desagregaciones.py) -- para que el color de cada grupo sea
# IDENTICO en todas las figuras de la tesis.
COLOR_CATEGORIA = {
    "Nunca pobre": "#2a78d6",       # PALETA["azul"]
    "Sale de la pobreza": "#1baf7a",  # PALETA["aguamarina"]
    "Entra en pobreza": "#eb6834",    # PALETA["naranja"]
    "Siempre pobre": "#e34948",       # PALETA["rojo"]
}
INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
AMBAR = "#c4820a"  # color reservado para DMSP (distinto de la paleta categorica)
AMBAR_TENUE = "#f2e6c9"
DMSP_MAX_ESCALA = 70  # tope de la barra "medidor de brillo" (DMSP observado va ~0-63)

# Posicion APROXIMADA (lon, lat) de cada region -- ver docstring: no son
# centroides calculados, son ubicaciones de referencia a partir de la
# composicion departamental documentada por CEDE-Uniandes (rural) o la
# region DANE estandar homonima (urbano).
COORDENADAS_REGION = {
    "Atlántica": (-74.8, 10.6),
    "Atlántica Media": (-75.0, 9.0),
    "Oriental": (-73.1, 7.1),
    "Central": (-75.6, 6.3),
    "Pacífica": (-76.5, 3.5),
    "Bogotá": (-74.08, 4.65),
    "Cundi-Boyacense": (-73.5, 5.6),
    "Eje Cafetero": (-75.75, 4.6),
    "Centro-Oriente": (-73.0, 6.6),
}

# `region` de ELCA junta DOS taxonomias distintas bajo el mismo nombre de
# columna: "Atlántica"/"Pacífica"/"Oriental"/"Central"/"Bogotá" son 100%
# URBANAS (cabeceras), y "Atlántica Media"/"Cundi-Boyacense"/"Eje
# Cafetero"/"Centro-Oriente" son 100% RURALES (microrregiones campesinas
# especificas) -- verificado por crosstab region x zona (ningun cruce
# mixto). CRITICO para la lectura: "Pacífica" no es la región Pacífica
# de Colombia, es solo su muestra urbana (ej. Cali), y el diseño rural de
# ELCA no incluye ni Chocó rural ni La Guajira en absoluto -- las zonas de
# mayor pobreza del país no estan representadas en estos datos. Se
# etiqueta explicitamente "(urbano)"/"(rural)" para que esto sea visible
# en la figura misma, no solo en el texto (decision del usuario,
# 2026-09-04, tras notar que "Pacífica" salía con poca pobreza y DMSP
# alto -- contraintuitivo si se lee como "la región Pacífica del país").
ETIQUETA_REGION = {
    "Atlántica": "Atlántica (urbano)",
    "Oriental": "Oriental (urbano)",
    "Central": "Central (urbano)",
    "Pacífica": "Pacífica (urbano)",
    "Bogotá": "Bogotá (urbano)",
    "Atlántica Media": "Atlántica Media (rural)",
    "Cundi-Boyacense": "Cundi-Boyacense (rural)",
    "Eje Cafetero": "Eje Cafetero (rural)",
    "Centro-Oriente": "Centro-Oriente (rural)",
}


def cargar_datos_region(nombre: str) -> pd.DataFrame:
    pct = pd.read_csv(TABLES_DIR / f"region_x_categoria_{nombre}.csv", index_col=0)
    # n_hogares por region no se guardo en un CSV aparte; se toma de
    # dmsp_por_region_{nombre}.csv, que ya trae n_hogares junto al DMSP
    # agregado (ver eda_transicion_covariables.py::perfilar_transicion).
    dmsp = pd.read_csv(TABLES_DIR / f"dmsp_por_region_{nombre}.csv", index_col=0)

    datos = pct.copy()
    datos["grupo_dominante"] = datos[CATEGORIAS_ORDEN].idxmax(axis=1)
    datos["pct_dominante"] = datos[CATEGORIAS_ORDEN].max(axis=1)
    datos["n_hogares"] = dmsp["n_hogares"]
    datos["dmsp_media"] = dmsp["dmsp_media_ponderada"]
    return datos


def _factor_circular(fig, ax) -> float:
    """Cuanto hay que escalar en y (en unidades de datos) para que un
    circulo dibujado en `ax` se vea circular en la figura ya renderizada
    -- ver docstring de `_dibujar_dona_composicion`."""
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    xr0, xr1 = ax.get_xlim()
    yr0, yr1 = ax.get_ylim()
    # abs(): `ax.invert_yaxis()` hace que ylim salga en orden (grande,
    # chico), asi que (yr1-yr0) da negativo -- sin el abs(), el factor
    # sale negativo (espeja la dona en vez de solo redimensionarla) y
    # rompe el calculo de donde queda su borde inferior real, usado para
    # ubicar las etiquetas de % debajo.
    escala_x = bbox.width / abs(xr1 - xr0)
    escala_y = bbox.height / abs(yr1 - yr0)
    return escala_x / escala_y


def _dibujar_dona_composicion(
    ax, x0: float, x1: float, y_centro: float, fila: pd.Series, factor_circular: float = 1.0
) -> None:
    """Dona de composicion (4 grupos) centrada en la mitad izquierda de la
    ficha (entre x0 y x1), con los 4 porcentajes -- uno por grupo, no solo
    el dominante -- en una fila debajo (decision del usuario, 2026-09-04:
    prefiere dona/pie sobre barra, pero mantiene el reparto simetrico
    izquierda/derecha que ya se habia corregido).

    `factor_circular` corrige la dona para que sea un circulo perfecto
    aunque el eje x y el eje y de `ax` no tengan la misma escala fisica
    (px por unidad de dato) -- que es justamente el caso aqui: cada
    columna cubre 10 unidades en x pero solo `N_FILAS_FICHA` en y, sobre
    una caja que no es exactamente 2:1. En vez de ajustar el tamaño de
    la figura a ojo (se probo, distorsiona el resto del layout), se mide
    la escala real de `ax` con el renderer y se aplica esa correccion
    SOLO a la dona, vía una transformacion afin que la escala en el eje y
    antes de pasar por `ax.transData` -- ver `_factor_circular` mas abajo."""
    cx = (x0 + x1) / 2
    radio, radio_interno = 0.40, 0.20
    transform = (
        Affine2D().translate(-cx, -y_centro).scale(1, factor_circular).translate(cx, y_centro)
        + ax.transData
    )
    angulo = 90.0
    for cat in CATEGORIAS_ORDEN:
        frac = fila[cat] / 100.0
        ang_fin = angulo - frac * 360.0
        cuna = Wedge(
            (cx, y_centro), radio, ang_fin, angulo, width=radio - radio_interno,
            facecolor=COLOR_CATEGORIA[cat], edgecolor=SURFACE, linewidth=0.8, zorder=3,
        )
        cuna.set_transform(transform)
        ax.add_patch(cuna)
        angulo = ang_fin

    # Los 4 porcentajes, uno por grupo, en una fila compacta debajo de la
    # dona -- misma logica de etiquetado que se uso antes para la barra.
    # El borde inferior REAL de la dona (en coordenadas de datos normales,
    # que es lo que usa el texto) queda en radio*factor_circular, no radio.
    y_etq = y_centro + radio * factor_circular + 0.12
    ancho_total = x1 - x0
    n = len(CATEGORIAS_ORDEN)
    for j, cat in enumerate(CATEGORIAS_ORDEN):
        xj = x0 + ancho_total * (j + 0.5) / n
        ax.plot(xj - 0.16, y_etq, marker="o", markersize=4.0, color=COLOR_CATEGORIA[cat],
                markeredgecolor="none", clip_on=False, zorder=4)
        ax.text(xj - 0.05, y_etq, f"{fila[cat]:.0f}%", fontsize=7.4, color=INK_SECUNDARIO,
                va="center", ha="left")


UMBRAL_LON_OESTE_ESTE = -74.4  # separa las regiones al oeste vs al este de Colombia (ver COORDENADAS_REGION)
N_FILAS_FICHA = 5  # = max(len(oeste), len(este)); ver nota en _dibujar_columna_fichas


def _dibujar_columna_fichas(fig, ax_fichas, ax_mapa, orden_regiones: list, datos: pd.DataFrame, borde_mapa: float) -> None:
    """Dibuja una columna de fichas (dona + 4% + nombre/n/DMSP) y sus
    lineas guia hacia `ax_mapa`. `borde_mapa` es el borde (0.0 o 10.0) de
    `ax_fichas` mas cercano al mapa -- las columnas al oeste de Colombia
    conectan por su borde derecho (10.0), las del este por su izquierdo
    (0.0), para que la linea guia salga del lado que efectivamente mira
    hacia el mapa (decision del usuario, 2026-09-04: las fichas deben
    quedar geograficamente agrupadas al lado del mapa que les corresponde,
    no todas apiladas a la derecha)."""
    # N_FILAS_FICHA fijo (no len(orden_regiones)): AMBAS columnas usan el
    # mismo rango de datos y el mismo width_ratio en el gridspec, para que
    # su caja fisica sea identica y una dona dibujada con `Wedge` (en
    # unidades de datos) salga circular sin depender de aspect tricks que
    # además encogen la caja (se probo y distorsionaba todo el layout). La
    # columna con menos regiones simplemente deja filas finales en blanco.
    ax_fichas.set_xlim(0, 10)
    ax_fichas.set_ylim(0, N_FILAS_FICHA)
    ax_fichas.set_axis_off()
    ax_fichas.set_facecolor(SURFACE)
    ax_fichas.invert_yaxis()  # fila 0 arriba (norte), ultima fila abajo (sur)
    factor_circular = _factor_circular(fig, ax_fichas)

    barra_x0, barra_x1 = 0.55, 4.75
    texto_x = 5.15
    gauge_x0, gauge_x1 = 5.15, 9.6

    for i, region in enumerate(orden_regiones):
        if region not in datos.index or pd.isna(datos.loc[region, "grupo_dominante"]):
            continue
        fila = datos.loc[region]
        lon, lat = COORDENADAS_REGION[region]
        y_centro = i + 0.5
        color_dominante = COLOR_CATEGORIA[fila["grupo_dominante"]]

        # punto real en el mapa (tamaño = n hogares, color = grupo dominante)
        radio_pt = 4 + 5 * (fila["n_hogares"] / datos["n_hogares"].max())
        ax_mapa.plot(lon, lat, marker="o", markersize=radio_pt, color=color_dominante,
                     markeredgecolor="white", markeredgewidth=1.0, zorder=3)

        # linea guia: del punto en el mapa al borde de la ficha mas cercano
        con = ConnectionPatch(
            xyA=(lon, lat), coordsA=ax_mapa.transData,
            xyB=(borde_mapa, y_centro), coordsB=ax_fichas.transData,
            color=GRIDLINE, linewidth=0.8, zorder=1,
        )
        fig.add_artist(con)
        ax_fichas.plot(borde_mapa, y_centro, marker="o", markersize=4, color=color_dominante,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=3, clip_on=False)

        # dona de composicion (mitad izquierda de la ficha) + los 4 % debajo
        _dibujar_dona_composicion(ax_fichas, barra_x0, barra_x1, y_centro - 0.06, fila, factor_circular)
        if i == 0:
            ax_fichas.text((barra_x0 + barra_x1) / 2, y_centro - 0.06 - 0.40 * factor_circular - 0.14,
                            "Composición (4 grupos)", fontsize=6.3, color=INK_MUTED,
                            ha="center", va="bottom")

        # nombre + n hogares (mitad derecha de la ficha)
        ax_fichas.text(texto_x, y_centro - 0.32, ETIQUETA_REGION[region], fontsize=8.6, color=INK_PRIMARIO, va="center")
        ax_fichas.text(texto_x, y_centro - 0.15, f"n={int(fila['n_hogares']):,}".replace(",", "."),
                        fontsize=6.8, color=INK_MUTED, va="center")

        # medidor de brillo DMSP (barra ambar)
        y_gauge = y_centro + 0.16
        ax_fichas.add_patch(plt.Rectangle((gauge_x0, y_gauge), gauge_x1 - gauge_x0, 0.16,
                                           facecolor=AMBAR_TENUE, edgecolor="none", zorder=2))
        ancho_dmsp = (gauge_x1 - gauge_x0) * min(fila["dmsp_media"] / DMSP_MAX_ESCALA, 1.0)
        ax_fichas.add_patch(plt.Rectangle((gauge_x0, y_gauge), max(ancho_dmsp, 0.05), 0.16,
                                           facecolor=AMBAR, edgecolor="none", zorder=3))
        ax_fichas.text(gauge_x1 + 0.12, y_gauge + 0.08, f"{fila['dmsp_media']:.0f}",
                        fontsize=7.5, color=INK_SECUNDARIO, va="center", ha="left")
        if i == 0:
            ax_fichas.text(gauge_x0, y_gauge - 0.06, "DMSP (luz nocturna)", fontsize=6.3,
                            color=INK_MUTED, va="bottom")


def graficar_mapa(datos: pd.DataFrame, titulo: str, nombre_archivo: str) -> None:
    """Version final: mapa AL CENTRO, con dos columnas de fichas (dona de
    composicion + medidor de brillo DMSP) flanqueandolo -- las regiones al
    OESTE de Colombia (Pacífica, Central, Eje Cafetero, Atlántica,
    Atlántica Media) a la izquierda, las del ESTE (Bogotá, Oriental,
    Cundi-Boyacense, Centro-Oriente) a la derecha, cada una unida a su
    punto real en el mapa por una linea guia -- tecnica cartografica
    estandar para clusters densos de etiquetas (ver docstring del modulo:
    6 de las 9 regiones caen muy juntas en el centro andino; intentos
    anteriores -- texto sobre el mapa, o una sola columna de fichas a la
    derecha -- quedaron ilegibles o geograficamente arbitrarios, decision
    del usuario 2026-09-04 de agrupar cada ficha del lado del mapa que le
    corresponde)."""
    contorno = gpd.read_file(CONTORNO_PATH)
    oeste = sorted(
        (r for r in COORDENADAS_REGION if COORDENADAS_REGION[r][0] <= UMBRAL_LON_OESTE_ESTE),
        key=lambda r: -COORDENADAS_REGION[r][1],
    )
    este = sorted(
        (r for r in COORDENADAS_REGION if COORDENADAS_REGION[r][0] > UMBRAL_LON_OESTE_ESTE),
        key=lambda r: -COORDENADAS_REGION[r][1],
    )

    fig, (ax_oeste, ax_mapa, ax_este) = plt.subplots(
        1, 3, figsize=(16.5, 7.4), gridspec_kw={"width_ratios": [1, 1.1, 1]}
    )
    contorno.plot(ax=ax_mapa, color="#eeece4", edgecolor="#c9c6bc", linewidth=0.8, zorder=1)
    ax_mapa.set_axis_off()
    ax_mapa.set_facecolor(SURFACE)

    _dibujar_columna_fichas(fig, ax_oeste, ax_mapa, oeste, datos, borde_mapa=10.0)
    _dibujar_columna_fichas(fig, ax_este, ax_mapa, este, datos, borde_mapa=0.0)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=COLOR_CATEGORIA[cat], markersize=8, label=cat)
        for cat in CATEGORIAS_ORDEN
    ]
    fig.legend(
        handles=handles, title="Grupo dominante (mayor % en la región)",
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=8.5, title_fontsize=8.5,
    )
    fig.suptitle(titulo, fontsize=12, color=INK_PRIMARIO, y=0.98)
    fig.patch.set_facecolor(SURFACE)
    fig.text(
        0.5, -0.02,
        "Posición del punto en el mapa: aproximada, no un centroide calculado ni un polígono administrativo "
        "(las microrregiones rurales de ELCA son municipios dispersos, no departamentos completos). "
        "Tamaño del punto = n.º de hogares del panel.",
        ha="center", fontsize=6.5, color=INK_MUTED, style="italic",
    )
    fig.text(
        0.5, -0.05,
        "Por anonimización de la encuesta (identificador falso de departamento/municipio, ver diccionario ELCA), "
        "no se dispone de un nivel de desagregación geográfica mayor al de región.",
        ha="center", fontsize=6.5, color=INK_MUTED, style="italic",
    )
    fig.text(
        0.5, -0.08,
        "\"(urbano)\"/\"(rural)\" indica el tipo de muestra de cada región: ELCA NO tiene un componente rural "
        "en Atlántica/Oriental/Central/Pacífica/Bogotá, y su muestra rural no incluye Chocó ni La Guajira -- "
        "estas cifras no representan las zonas de mayor pobreza del país.",
        ha="center", fontsize=6.5, color=INK_MUTED, style="italic",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / nombre_archivo, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    for nombre, etiqueta in [("monetaria", "pobreza monetaria"), ("ipm", "pobreza multidimensional (IPM)")]:
        datos = cargar_datos_region(nombre)
        print(f"\n=== {nombre} ===")
        print(datos[["grupo_dominante", "pct_dominante", "n_hogares", "dmsp_media"]])
        graficar_mapa(
            datos,
            titulo=f"Grupo de transición dominante por región e iluminación nocturna (DMSP)\n{etiqueta}, 2010→2013",
            nombre_archivo=f"mapa_transicion_regiones_{nombre}.png",
        )
    print(f"\nGuardado en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
