"""
mapa_territorial_transicion_divipola.py
========================================
Mapa territorial REAL (a diferencia de mapa_transicion_regiones.py, que es
un diagrama esquematico por region -- ver docstring de ese script) de los
4 grupos de transicion de pobreza (2010->2013), ubicados por COORDENADAS
REALES (latitud/longitud) de cada hogar, sobre un fondo que simula una foto
satelital de luces nocturnas (DMSP-OLS 2010) de Colombia.

RESPONSABILIDAD EXCLUSIVA
    Generar exactamente 2 mapas (pobreza monetaria e IPM, ambos 2010->2013)
    ubicando los 4 grupos de transicion por la posicion real de cada hogar.
    Este script NO descarga nada, NO depura nada en caliente: esta disenado
    para correrse UNA VEZ, sin acceso a internet ni intervencion, en la
    sala de computo de la universidad, donde se puede acceder a la base de
    ELCA con coordenadas reales de hogar (a diferencia de la copia de este
    proyecto, donde esa columna existe pero viene vacia -- verificado: 0%
    de cobertura en panel_coordenadas.csv, dato sensible que solo esta
    disponible en el entorno restringido).

POR QUE COORDENADAS DIRECTAS Y NO EL CODIGO DIVIPOLA (decision del usuario,
2026-09-11, version anterior de este script)
    La primera version de este script cruzaba por `id_mpio` (codigo
    DIVIPOLA) contra una tabla oficial de coordenadas por municipio. El
    usuario reporto que los codigos DIVIPOLA de la base de la universidad
    "no son consistentes" pero que SI tiene coordenadas (lat/lon) por hogar
    en esa misma base -- lo cual es estrictamente mejor: coordenadas reales
    ubican al hogar en su posicion exacta, no solo en el centroide de su
    municipio, y evitan por completo el problema de formato/consistencia
    del codigo. Los nombres de columna (`lat_decimal`/`lon_decimal`) siguen
    la misma convencion ya usada en este proyecto para coordenadas de hogar
    (ver src/01_download/02_scr_GoogleStreetView/00_construir_panel_coordenadas.py),
    son CONFIGURABLES en CONFIG por si la base de la universidad usa otro
    nombre.

POR QUE ESTE SCRIPT ES AUTOCONTENIDO (decision del usuario, 2026-09-11)
    Todo el calculo de la matriz de transicion (identico a
    `construir_matriz_transicion` de `build_pobreza_desagregaciones.py`) y
    la exclusion de hogares divididos (identica a
    `_excluir_hogares_divididos` de `eda_transicion_covariables.py`) estan
    COPIADOS aqui en vez de importados -- pedido explicito del usuario:
    "todo en un archivo único", porque en la sala no se puede resolver un
    error de import de otro modulo del proyecto si algo no coincide.

POR QUE EL FONDO DE BRILLO VIENE PRECOMPUTADO, NO DE UN RASTER EN VIVO
    Ver `eda_dmsp_fondo_mapa.py` (corrido de antemano, CON internet, para
    generar `fondo_dmsp_colombia_2010.npz`, ~5.5MB). Ese raster original
    pesa 369MB y su procesamiento depende de rasterio/scipy -- una
    superficie de fallo innecesaria en la sala si lo unico que hace falta
    ese dia es el resultado visual ya listo. Decision del usuario tras
    comparar las dos opciones explicitamente.

ESTILO VISUAL (validado por el usuario en el artifact de prueba
8ae3a4d2-a548-4df6-9b3d-bf960c45bd8a, iteraciones v1 a v8, 2026-09-11)
    - Fondo oscuro con el brillo DMSP como imagen CONTINUA (no puntos
      discretos por municipio -- eso se probo primero y el usuario lo
      rechazo: "me refiero a algo mucho más continuo... no hay luz y eso
      no es cierto").
    - Colores de los 4 grupos elegidos por teoria del color: verde (142°),
      azul (199°) para "nunca pobre", rojo para "siempre pobre" y naranja
      para "entra en pobreza" (pedido explicito del usuario) -- separados
      en la rueda de color para no confundirse entre si NI con la paleta
      "inferno" del fondo (que tambien pasa por rojo-naranja-amarillo).
    - Los 4 marcadores tienen el MISMO tamaño de circulo (pedido explicito
      final del usuario). El grupo "entra en pobreza" (vulnerable) se sigue
      distinguiendo del fondo brillante mediante un halo blanco translucido
      DETRAS del circulo (el halo no agranda el circulo en si).

CARPETA PORTABLE (pedido del usuario, 2026-09-11: no se puede llevar el
repositorio completo a la sala)
    Este script asume que existe una carpeta `datos/` UBICADA JUNTO A EL
    (mismo directorio, no la estructura data/processed/ / data/interim/
    del repo) con los archivos de INPUTS de abajo. El paquete completo
    (este script + esa carpeta datos/) pesa unos 15MB y es lo UNICO que hay
    que copiar a la sala -- no el proyecto entero.

INPUTS (dentro de datos/, junto a este script)
    pobreza_monetaria_elca_longitudinal.parquet
    ipm_multidimensional_elca_longitudinal.parquet
    hogar_elca_longitudinal_clean.parquet
        (con columnas de latitud/longitud REALES por hogar -- obtenidas en
        la sala; nombres configurables en CONFIG["col_latitud"/"col_longitud"])
    fondo_dmsp_colombia_2010.npz (fondo de brillo precomputado, ver eda_dmsp_fondo_mapa.py)
    gadm41_COL_0.{shp,shx,dbf,cpg,prj} (contorno pais)
    gadm41_COL_1.{shp,shx,dbf,cpg,prj} (limites departamento)

OUTPUTS (se crean junto al script, dentro de salidas/)
    salidas/figuras/mapa_territorial_monetaria_2010_2013.png
    salidas/figuras/mapa_territorial_ipm_2010_2013.png
    salidas/tablas/resumen_coordenadas_{monetaria,ipm}.csv
        (diagnostico: n hogares del panel, n con coordenadas validas,
        n efectivamente graficados, por categoria)

COMO CORRER
    cd carpeta_donde_esta_este_script/
    python mapa_territorial_transicion_divipola.py

SI ALGO FALLA EN LA SALA
    Este script valida cada insumo ANTES de graficar y termina con un
    mensaje que dice exactamente que archivo o columna revisar (ver
    seccion VALIDACIONES). No requiere internet en ningun punto.
"""

import sys
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

# Este script es AUTOCONTENIDO tambien en sus rutas: espera encontrar sus
# datos en una carpeta "datos/" UBICADA JUNTO A EL (no la estructura del
# repo completo data/processed/, data/interim/...). Esto permite llevar
# solo esta carpeta (script + datos/) a la sala de computo, sin copiar el
# resto del proyecto (fotos de Google Street View, notebooks, etc.).
CARPETA_DATOS = Path(__file__).resolve().parent / "datos"

CONFIG = {
    # ── Inputs de pobreza (ver INPUTS en el docstring) ────────────────────────
    "pobreza_monetaria_path": CARPETA_DATOS / "pobreza_monetaria_elca_longitudinal.parquet",
    "ipm_path": CARPETA_DATOS / "ipm_multidimensional_elca_longitudinal.parquet",
    "hogar_path": CARPETA_DATOS / "hogar_elca_longitudinal_clean.parquet",

    # ── Nombres de columna de coordenadas en hogar_path ───────────────────────
    # AJUSTAR AQUI si la base de la universidad usa otro nombre -- es el
    # UNICO cambio que deberia hacer falta si el nombre de columna difiere.
    "col_latitud": "lat_decimal",
    "col_longitud": "lon_decimal",

    # ── Inputs geograficos (ya preparados, no requieren internet) ─────────────
    "fondo_dmsp_path": CARPETA_DATOS / "fondo_dmsp_colombia_2010.npz",
    "contorno_pais_path": CARPETA_DATOS / "gadm41_COL_0.shp",
    "limites_depto_path": CARPETA_DATOS / "gadm41_COL_1.shp",

    # ── Ola de la transicion (fijo: 2010->2013, unica transicion pedida) ──────
    "ola_inicial": 1,
    "ola_final": 2,

    # ── Bounding box de Colombia, para validar coordenadas (mismo rango que
    # usa 00_construir_panel_coordenadas.py del pipeline de Google Street View) ──
    "lat_min": -4.2,
    "lat_max": 12.5,
    "lon_min": -79.0,
    "lon_max": -66.8,

    # ── Umbral de validacion de coordenadas ────────────────────────────────
    # Si menos de este % de hogares del panel tiene coordenadas validas
    # (no nulas y dentro del territorio colombiano), el script se detiene
    # -- ver VALIDACIONES.
    "umbral_pct_coordenadas_validas": 50.0,

    # ── Salidas (se crean junto al script, dentro de esta misma carpeta) ──────
    "figuras_dir": Path(__file__).resolve().parent / "salidas" / "figuras",
    "tablas_dir": Path(__file__).resolve().parent / "salidas" / "tablas",

    # ── Estilo visual (validado por el usuario, ver docstring) ────────────────
    "color_categoria": {
        "Sale de la pobreza": "#22c55e",
        "Nunca pobre": "#38bdf8",
        "Siempre pobre": "#e6242f",
        "Entra en pobreza": "#ff8c1a",
    },
    "tamano_marcador": 40,       # mismo tamaño para los 4 grupos
    "factor_halo": 3.2,          # halo del vulnerable = tamano_marcador * este factor
    "alpha_halo": 0.35,
    "borde_marcador_normal": 0.6,
    "borde_marcador_vulnerable": 1.4,
    "extent_mapa": (-82, -66.5, -4.5, 13.8),  # xmin, xmax, ymin, ymax
}


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)


CATEGORIAS_ORDEN = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]


# ──────────────────────────────────────────────────────────────────────────────
# VALIDACIONES (correr ANTES de cualquier calculo -- fallar rapido y claro)
# ──────────────────────────────────────────────────────────────────────────────

def validar_archivos(cfg: dict) -> None:
    archivos_requeridos = [
        ("pobreza_monetaria_path", "pobreza_monetaria_elca_longitudinal.parquet"),
        ("ipm_path", "ipm_multidimensional_elca_longitudinal.parquet"),
        ("hogar_path", "hogar_elca_longitudinal_clean.parquet"),
        ("fondo_dmsp_path", "fondo_dmsp_colombia_2010.npz"),
        ("contorno_pais_path", "gadm41_COL_0.shp"),
        ("limites_depto_path", "gadm41_COL_1.shp"),
    ]
    faltantes = [(clave, nombre) for clave, nombre in archivos_requeridos if not Path(cfg[clave]).exists()]
    if faltantes:
        log.error("-" * 62)
        log.error("ERROR: No se encontraron los siguientes archivos de entrada:")
        for clave, nombre in faltantes:
            log.error(f"  - CONFIG['{clave}'] -> {cfg[clave]}")
        log.error("SOLUCION: verifica que la carpeta 'datos/' este junto a este")
        log.error("script (mismo nivel, no en otra ubicacion) y que contenga")
        log.error("todos los archivos listados arriba.")
        log.error("-" * 62)
        sys.exit(1)

    hogar_cols = set(pd.read_parquet(cfg["hogar_path"]).columns)
    columnas_esperadas = {"consecutivo", "ola", cfg["col_latitud"], cfg["col_longitud"]}
    faltantes_cols = columnas_esperadas - hogar_cols
    if faltantes_cols:
        log.error("-" * 62)
        log.error(f"ERROR: faltan columnas en hogar_elca_longitudinal_clean.parquet: {faltantes_cols}")
        log.error(f"Columnas presentes: {sorted(hogar_cols)}")
        log.error("SOLUCION: si la base de la universidad usa otro nombre para")
        log.error("latitud/longitud, cambia CONFIG['col_latitud'] / CONFIG['col_longitud']")
        log.error("al inicio de este script para que coincidan.")
        log.error("-" * 62)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: MATRIZ DE TRANSICIÓN (copiado de build_pobreza_desagregaciones.py
# ::construir_matriz_transicion -- ver docstring del modulo, "todo en un
# archivo único")
# ──────────────────────────────────────────────────────────────────────────────

def construir_matriz_transicion(pobreza: pd.DataFrame, ola_ini: int, ola_fin: int, col_pobre: str) -> pd.DataFrame:
    """
    Devuelve panel_categorias: 1 fila = 1 hogar (consecutivo, ola inicial)
    con match 1 a 1 entre ola_ini y ola_fin, clasificado en las 4 categorias
    de transicion. Hogares divididos (consecutivo duplicado en cualquiera de
    las 2 olas) se excluyen -- misma regla que el resto del proyecto.
    """
    ini = pobreza[pobreza["ola"] == ola_ini][["consecutivo", col_pobre]].dropna(subset=[col_pobre])
    fin = pobreza[pobreza["ola"] == ola_fin][["consecutivo", col_pobre]].dropna(subset=[col_pobre])

    ini_unicos = ini[~ini["consecutivo"].duplicated(keep=False)]
    fin_unicos = fin[~fin["consecutivo"].duplicated(keep=False)]

    panel = ini_unicos.merge(fin_unicos, on="consecutivo", suffixes=(f"_{ola_ini}", f"_{ola_fin}"))
    col_ini, col_fin = f"{col_pobre}_{ola_ini}", f"{col_pobre}_{ola_fin}"
    panel[col_ini] = panel[col_ini].map({True: "Pobre", False: "No pobre"})
    panel[col_fin] = panel[col_fin].map({True: "Pobre", False: "No pobre"})

    def clasificar(fila):
        if fila[col_ini] == "No pobre" and fila[col_fin] == "No pobre":
            return "Nunca pobre"
        if fila[col_ini] == "Pobre" and fila[col_fin] == "Pobre":
            return "Siempre pobre"
        if fila[col_ini] == "Pobre" and fila[col_fin] == "No pobre":
            return "Sale de la pobreza"
        return "Entra en pobreza"

    panel["categoria"] = panel.apply(clasificar, axis=1)
    log.info(f"  Panel de transicion ({col_pobre}): {len(panel):,} hogares con match 1 a 1")
    for cat in CATEGORIAS_ORDEN:
        n = (panel["categoria"] == cat).sum()
        log.info(f"    {cat}: {n:,} ({100 * n / len(panel):.1f}%)")
    return panel[["consecutivo", "categoria"]]


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: COORDENADAS REALES POR CONSECUTIVO (misma regla de exclusion de
# hogares divididos que el resto del proyecto -- ver
# eda_transicion_covariables.py ::_excluir_hogares_divididos)
# ──────────────────────────────────────────────────────────────────────────────

def cargar_coordenadas_por_consecutivo(cfg: dict, ola_base: int) -> pd.DataFrame:
    """Latitud/longitud reales de la ola BASE de la transicion, indexadas
    por consecutivo. Hogares divididos excluidos (misma regla exacta que
    el resto del proyecto)."""
    col_lat, col_lon = cfg["col_latitud"], cfg["col_longitud"]
    hogar = pd.read_parquet(cfg["hogar_path"], columns=["consecutivo", "ola", col_lat, col_lon])
    hogar_ola = hogar[hogar["ola"] == ola_base]
    hogar_ola = hogar_ola[~hogar_ola["consecutivo"].duplicated(keep=False)]
    assert not hogar_ola["consecutivo"].duplicated().any(), f"consecutivo debe ser unico en ola {ola_base}"
    return hogar_ola.set_index("consecutivo")[[col_lat, col_lon]].rename(
        columns={col_lat: "latitud", col_lon: "longitud"}
    )


def agregar_coordenadas(cfg: dict, panel: pd.DataFrame, coords: pd.DataFrame, nombre_definicion: str) -> pd.DataFrame:
    """
    Agrega latitud/longitud a `panel` y valida que esten dentro del
    territorio colombiano. Se detiene con un mensaje accionable si menos
    del umbral configurado (CONFIG['umbral_pct_coordenadas_validas']) tiene
    coordenadas utilizables -- la causa mas probable es un nombre de
    columna equivocado (valores todos nulos) o unidades/formato distinto
    (ej. grados-minutos-segundos en vez de decimal).
    """
    panel = panel.merge(coords, on="consecutivo", how="left")

    n_total = len(panel)
    n_no_nulas = panel["latitud"].notna() & panel["longitud"].notna()
    dentro_col = (
        panel["latitud"].between(cfg["lat_min"], cfg["lat_max"])
        & panel["longitud"].between(cfg["lon_min"], cfg["lon_max"])
    )
    panel["coord_valida"] = n_no_nulas & dentro_col
    n_validas = panel["coord_valida"].sum()
    pct_validas = 100 * n_validas / n_total if n_total else 0.0

    log.info(f"  [{nombre_definicion}] Hogares del panel: {n_total:,}")
    log.info(f"  [{nombre_definicion}] Con coordenadas validas dentro de Colombia: {n_validas:,} ({pct_validas:.1f}%)")

    if pct_validas < cfg["umbral_pct_coordenadas_validas"]:
        n_nulas = (~n_no_nulas).sum()
        n_fuera_rango = (n_no_nulas & ~dentro_col).sum()
        muestra_fuera_rango = panel.loc[n_no_nulas & ~dentro_col, ["latitud", "longitud"]].head(5).to_dict("records")
        log.error("-" * 62)
        log.error(f"ERROR: solo {pct_validas:.1f}% de los hogares tienen coordenadas")
        log.error(f"validas (umbral minimo: {cfg['umbral_pct_coordenadas_validas']}%).")
        log.error(f"  Coordenadas nulas: {n_nulas:,}")
        log.error(f"  Coordenadas no nulas pero FUERA del rango de Colombia: {n_fuera_rango:,}")
        if muestra_fuera_rango:
            log.error(f"  Ejemplos fuera de rango: {muestra_fuera_rango}")
        log.error("CAUSA PROBABLE:")
        log.error("  - Si la mayoria son nulas: el nombre de columna en CONFIG")
        log.error(f"    ('{cfg['col_latitud']}'/'{cfg['col_longitud']}') no es el correcto.")
        log.error("    Revisa las columnas reales del archivo de la sala y ajustalas.")
        log.error("  - Si estan fuera de rango: puede que vengan en formato")
        log.error("    grados-minutos-segundos en vez de decimal, o con el signo")
        log.error("    de la longitud invertido (Colombia siempre es longitud negativa).")
        log.error("-" * 62)
        sys.exit(1)

    return panel


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: GRAFICADO (estilo validado por el usuario, ver docstring)
# ──────────────────────────────────────────────────────────────────────────────

def graficar_mapa(cfg: dict, panel_geo: pd.DataFrame, titulo: str, nombre_archivo: Path) -> None:
    fondo = np.load(cfg["fondo_dmsp_path"])
    compuesto_log = fondo["compuesto_log"]
    xmin, xmax, ymin, ymax = fondo["extent"]

    pais = gpd.read_file(cfg["contorno_pais_path"])
    deptos = gpd.read_file(cfg["limites_depto_path"])

    fig, ax = plt.subplots(figsize=(9, 11))
    fig.patch.set_facecolor("#05070c")
    ax.set_facecolor("#05070c")

    ax.imshow(compuesto_log, cmap="inferno", extent=[xmin, xmax, ymin, ymax],
              origin="upper", zorder=1, vmin=0, vmax=np.percentile(compuesto_log, 99.7))

    pais.boundary.plot(ax=ax, color="#5a6270", linewidth=1.0, zorder=2, alpha=0.8)
    deptos.boundary.plot(ax=ax, color="#3a4250", linewidth=0.35, zorder=2, alpha=0.7)

    ubicados = panel_geo[panel_geo["coord_valida"]]
    tam = cfg["tamano_marcador"]
    for categoria in CATEGORIAS_ORDEN:
        sub = ubicados[ubicados["categoria"] == categoria]
        if sub.empty:
            continue
        color = cfg["color_categoria"][categoria]
        es_vulnerable = categoria == "Entra en pobreza"
        if es_vulnerable:
            ax.scatter(sub["longitud"], sub["latitud"], s=tam * cfg["factor_halo"], c="white",
                       alpha=cfg["alpha_halo"], linewidths=0, zorder=3.5)
            ax.scatter(sub["longitud"], sub["latitud"], s=tam, c=color, edgecolors="white",
                       linewidths=cfg["borde_marcador_vulnerable"], alpha=1.0, zorder=4, label=categoria)
        else:
            ax.scatter(sub["longitud"], sub["latitud"], s=tam, c=color, edgecolors="white",
                       linewidths=cfg["borde_marcador_normal"], alpha=0.9, zorder=3, label=categoria)

    xmin_c, xmax_c, ymin_c, ymax_c = cfg["extent_mapa"]
    ax.set_xlim(xmin_c, xmax_c)
    ax.set_ylim(ymin_c, ymax_c)
    ax.set_axis_off()
    ax.set_title(titulo, color="white", fontsize=12, loc="left")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color("white")

    n_sin_ubicar = len(panel_geo) - len(ubicados)
    if n_sin_ubicar:
        fig.text(0.5, 0.01, f"{n_sin_ubicar:,} hogares sin coordenadas validas, no graficados.",
                  ha="center", fontsize=8, color="#8b93a1")

    fig.tight_layout()
    nombre_archivo.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(nombre_archivo, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Guardado: {nombre_archivo}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def procesar_definicion(cfg: dict, nombre: str, etiqueta: str, poverty_path: Path, col_pobre: str) -> None:
    log.info(f"\n=== {etiqueta} ({cfg['ola_inicial']} -> {cfg['ola_final']}) ===")
    pobreza = pd.read_parquet(poverty_path)
    panel = construir_matriz_transicion(pobreza, cfg["ola_inicial"], cfg["ola_final"], col_pobre)

    coords = cargar_coordenadas_por_consecutivo(cfg, cfg["ola_inicial"])
    panel_geo = agregar_coordenadas(cfg, panel, coords, nombre)

    cfg["tablas_dir"].mkdir(parents=True, exist_ok=True)
    resumen = (
        panel_geo.groupby("categoria")["coord_valida"]
        .agg(n_total="size", n_validas="sum")
        .reindex(CATEGORIAS_ORDEN)
    )
    resumen.to_csv(cfg["tablas_dir"] / f"resumen_coordenadas_{nombre}.csv")

    graficar_mapa(
        cfg, panel_geo,
        titulo=f"Grupos de transición de {etiqueta} por ubicación real del hogar\n2010 → 2013",
        nombre_archivo=cfg["figuras_dir"] / f"mapa_territorial_{nombre}_2010_2013.png",
    )


def main() -> None:
    log.info("Validando archivos de entrada...")
    validar_archivos(CONFIG)
    log.info("OK: todos los archivos de entrada estan presentes.\n")

    procesar_definicion(CONFIG, "monetaria", "pobreza monetaria", CONFIG["pobreza_monetaria_path"], "pobre_ingreso")
    procesar_definicion(CONFIG, "ipm", "pobreza multidimensional (IPM)", CONFIG["ipm_path"], "pobre_ipm")

    log.info(f"\nListo. Mapas guardados en: {CONFIG['figuras_dir']}")
    log.info(f"Tablas de diagnostico guardadas en: {CONFIG['tablas_dir']}")


if __name__ == "__main__":
    main()
