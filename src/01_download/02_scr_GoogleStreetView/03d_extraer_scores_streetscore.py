"""
03d_extraer_scores_streetscore.py
==================================
Calcula puntajes de percepción urbana (Place Pulse 2.0 / StreetScore) para
cada foto de Google Street View descargada por 02_descarga_fotos_GSV.py,
usando los modelos ViT preentrenados del paquete ZenSVI.

RESPONSABILIDAD EXCLUSIVA
    Transformar un directorio de imágenes JPEG en una tabla de puntajes de
    percepción urbana (una fila por imagen, una columna por dimensión).

    Este script NO realiza:
    · Descarga de imágenes (responsabilidad de script 02).
    · Extracción de embeddings visuales (responsabilidad de 03 y 03b —
      pipeline separado porque aquí no generamos un embedding propio: la
      librería ZenSVI ya expone directamente un score interpretable por
      dimensión, entrenado por regresión sobre comparaciones pareadas
      humanas de Place Pulse 2.0, no por un backbone de clasificación).
    · Entrenamiento, fine-tuning ni clasificación propia.
    · Selección de variables ni análisis estadístico.

MODELO
    ZenSVI (https://github.com/koito19960406/ZenSVI), clasificador ViT
    (Ouyang 2023) entrenado sobre Place Pulse 2.0 (110,988 imágenes de 56
    ciudades, comparaciones pareadas de percepción humana). Pesos
    descargados automáticamente desde Hugging Face Hub
    (repo "Jiani11/human-perception-place-pulse") la primera vez que se
    instancia cada dimensión; quedan en caché dentro del propio paquete
    zensvi instalado.

    Seis dimensiones perceptuales (Place Pulse 2.0), score continuo 0-10:
        "safer"            → score_safety
        "livelier"         → score_lively
        "wealthy"          → score_wealthy
        "more beautiful"   → score_beautiful
        "more boring"      → score_boring
        "more depressing"  → score_depressing

    IMPORTANTE — naturaleza exploratoria: estos scores provienen de
    comparaciones hechas por voluntarios de internet mayormente de fuera de
    Colombia. La correlación esperada con vulnerabilidad a pobreza (ej.
    score_wealthy bajo ~ mayor vulnerabilidad) es una hipótesis a validar
    contra el índice de activos/SISBEN de la ELCA, no un supuesto aceptado.

INPUT
    gsv/registro_descargas.csv  → log de script 02 (imágenes exitosas)
    gsv/inventario_panos.csv    → identificadores del hogar (llave, etc.)
    gsv/fotos/                  → carpeta raíz con las imágenes descargadas
                                   (ZenSVI las busca recursivamente aquí)

OUTPUTS
    scores_streetscore/scores_streetscore.parquet → puntajes (Output 1)
    scores_streetscore/reporte_scores_streetscore.txt → informe (Output 2)

FORMATO DE SALIDA
    Una fila por imagen (la intersección entre el registro de descargas y
    lo que ZenSVI logró clasificar). Columnas principales:
        image_id, consecutivo, ola, pano_id, heading, nombre_archivo,
        llave (si disponible), llave_n16 (si disponible),
        score_safety, score_lively, score_wealthy, score_beautiful,
        score_boring, score_depressing.

CÓMO CORRER
    python 03d_extraer_scores_streetscore.py

    Instalar dependencia primero:  pip install zensvi
    Para procesar en GPU: modificar CONFIG["device"] = "cuda" o "mps".
"""

import sys
import time
import hashlib
import logging
from pathlib import Path

try:
    import pandas as pd
    import pyarrow as _pyarrow; del _pyarrow  # guard: falla aquí si falta, antes de clasificar
    from zensvi.cv import ClassifierPerceptionViT
except ImportError as _err:
    # Si ves este error, el entorno virtual no está activo o falta zensvi.
    # Solución paso a paso:
    #   1. Activa el entorno virtual:
    #        Mac/Linux : source .venv/bin/activate
    #        Windows   : .venv\Scripts\activate
    #   2. Instala zensvi (trae torch, torchvision, huggingface_hub, etc.):
    #        pip install zensvi
    print("=" * 62)
    print(f"ERROR: librería no encontrada → {_err}")
    print("Revisa los comentarios junto a los imports en este script.")
    print("=" * 62)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent   # carpeta GoogleStreetView/

CONFIG = {
    # ── Inputs ─────────────────────────────────────────────────────────────────
    # Registro de descargas producido por 02_descarga_fotos_GSV.py.
    "input_registro": _HERE / "gsv" / "registro_descargas.csv",

    # Inventario producido por 01_analisis_cobertura_gsv.py (llave, llave_n16).
    "input_inventario": _HERE / "gsv" / "inventario_panos.csv",

    # Carpeta raíz donde viven las imágenes descargadas por script 02.
    # ZenSVI la escanea recursivamente (rglob), no hace falta listar subcarpetas
    # por ola a mano.
    "carpeta_fotos": _HERE / "gsv" / "fotos",

    # ── Dimensiones de Place Pulse 2.0 ─────────────────────────────────────────
    # Clave = valor exacto que espera ClassifierPerceptionViT(perception_study=...).
    # Valor = nombre de columna en el output ("score_" + valor).
    "dimensiones": {
        "safer":            "safety",
        "livelier":         "lively",
        "wealthy":          "wealthy",
        "more beautiful":   "beautiful",
        "more boring":      "boring",
        "more depressing":  "depressing",
    },

    # ── Inferencia ─────────────────────────────────────────────────────────────
    "batch_size": 32,          # imágenes por lote; reducir si hay OOM en GPU
    "device":     "auto",      # "auto" | "cpu" | "cuda" | "mps"

    # ── Salida ─────────────────────────────────────────────────────────────────
    "output_dir": _HERE / "scores_streetscore",

    # Si True, conserva también el results.csv/json crudo que genera ZenSVI
    # por cada dimensión (útil para depurar), en output_dir/_raw_zensvi/<dim>/.
    "guardar_resultados_crudos": False,
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


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 1: CARGA DEL REGISTRO DE IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────
# Idéntico al Módulo 1 de 03_extraer_embeddings.py y 03b: el insumo (registro +
# inventario) no depende del modelo usado para puntuar las imágenes.

def cargar_imagenes(cfg: dict) -> pd.DataFrame:
    """
    Lee el registro de descargas y retorna las imágenes disponibles para
    puntuar.

    Filtros aplicados:
    · exito == True  → solo imágenes descargadas con éxito por script 02.
    · ruta_archivo no nula.

    Si el inventario está disponible, incorpora identificadores adicionales
    del hogar (llave, llave_n16) mediante un join por (consecutivo, ola, pano_id).

    Retorna:
        DataFrame con una fila por imagen, con columnas de identificadores,
        la ruta absoluta del archivo en disco, y "filename_key" (el nombre
        de archivo sin extensión, usado para unir con los resultados de
        ZenSVI, que identifica cada imagen por Path(archivo).stem).
    """
    path_reg = Path(cfg["input_registro"])
    if not path_reg.exists():
        log.error("-" * 62)
        log.error("ERROR: Registro de descargas no encontrado.")
        log.error(f"Buscado en: {path_reg.resolve()}")
        log.error("SOLUCION: Ejecuta primero el script anterior:")
        log.error("  python 02_descarga_fotos_GSV.py")
        log.error("-" * 62)
        sys.exit(1)

    reg = pd.read_csv(
        path_reg,
        dtype={"consecutivo": str, "pano_id": str},
        low_memory=False,
    )
    log.info(f"Registro cargado: {len(reg):,} filas ({path_reg.name})")

    # NOTA: 'exito' se lee como string "True"/"False", no como booleano Python.
    df_ok = reg[reg["exito"].astype(str).str.strip() == "True"].copy()
    df_ok = df_ok[df_ok["ruta_archivo"].notna()].copy()
    log.info(f"  Imágenes con exito=True y ruta válida: {len(df_ok):,}")

    if df_ok.empty:
        log.error("ERROR: No hay imagenes con exito=True en el registro.")
        sys.exit(0)

    path_inv = Path(cfg["input_inventario"])
    if path_inv.exists():
        inv = pd.read_csv(
            path_inv,
            dtype={"consecutivo": str, "pano_id": str, "llave": str, "llave_n16": str},
            low_memory=False,
        )
        cols_ids = ["consecutivo", "ola", "pano_id"]
        for _c in ["llave", "llave_n16"]:
            if _c in inv.columns:
                cols_ids.append(_c)
        ids_extra = inv[cols_ids].drop_duplicates(subset=["consecutivo", "ola", "pano_id"])
        df_ok = df_ok.merge(ids_extra, on=["consecutivo", "ola", "pano_id"], how="left")
        log.info("  Identificadores llave y llave_n16 incorporados desde el inventario.")
    else:
        df_ok["llave"]     = pd.NA
        df_ok["llave_n16"] = pd.NA
        log.warning(f"Inventario no encontrado ({path_inv.name}). "
                    "llave y llave_n16 quedarán vacíos.")

    # Clave de unión con los resultados de ZenSVI: nombre de archivo sin extensión.
    df_ok["filename_key"] = df_ok["nombre_archivo"].apply(lambda f: Path(f).stem)

    return df_ok.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: CLASIFICACIÓN POR DIMENSIÓN
# ──────────────────────────────────────────────────────────────────────────────

def _resolver_device(device_cfg: str):
    """
    Traduce CONFIG["device"] al valor que espera ClassifierPerceptionViT.
    "auto" se traduce a None: la librería detecta CUDA → MPS → CPU internamente.
    """
    return None if device_cfg == "auto" else device_cfg


def ejecutar_clasificacion(cfg: dict) -> dict:
    """
    Corre, una a una, las 6 dimensiones de Place Pulse 2.0 sobre todas las
    imágenes en cfg["carpeta_fotos"].

    ZenSVI escanea la carpeta de forma recursiva y clasifica TODAS las
    imágenes que encuentre ahí — no filtra por el registro de descargas.
    Por eso el resultado de este módulo se une con df_ok (Módulo 1) recién
    en el Módulo 3, para quedarnos solo con las imágenes que también están
    marcadas como exito=True.

    Cada dimensión carga su propio modelo (checkpoint distinto por
    perception_study); se libera de memoria antes de pasar a la siguiente
    para no acumular 6 modelos ViT cargados a la vez.

    Retorna:
        {nombre_columna: DataFrame con columnas ["filename_key", nombre_columna]}
    """
    device = _resolver_device(cfg["device"])
    resultados = {}

    for perception_study, nombre_columna in cfg["dimensiones"].items():
        log.info(f"Clasificando dimensión '{perception_study}' → score_{nombre_columna} …")

        dir_salida_cruda = None
        if cfg["guardar_resultados_crudos"]:
            dir_salida_cruda = Path(cfg["output_dir"]) / "_raw_zensvi" / nombre_columna

        clasificador = ClassifierPerceptionViT(perception_study=perception_study, device=device)
        registros = clasificador.classify(
            cfg["carpeta_fotos"],
            dir_summary_output=dir_salida_cruda,
            batch_size=cfg["batch_size"],
        )
        del clasificador   # libera el modelo ViT antes de cargar la siguiente dimensión

        df_dim = pd.DataFrame(registros)   # columnas: filename_key, <perception_study>
        df_dim = df_dim.rename(columns={perception_study: f"score_{nombre_columna}"})
        resultados[nombre_columna] = df_dim

        log.info(f"  {len(df_dim):,} imágenes puntuadas para '{perception_study}'.")

    return resultados


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: COMBINACIÓN CON EL REGISTRO DE IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────

def combinar_scores(df_ok: pd.DataFrame, resultados: dict) -> tuple:
    """
    Une cada tabla de scores (una por dimensión) con df_ok por "filename_key",
    y consolida las 6 columnas score_* en un solo DataFrame de una fila por
    imagen.

    Retorna:
        (df_combinado, n_sin_match)
        · df_combinado : df_ok + columnas score_* (solo filas con match en
          TODAS las dimensiones — si ZenSVI no logró procesar una imagen en
          alguna dimensión, esa imagen queda fuera del output final).
        · n_sin_match   : cantidad de imágenes de df_ok sin score en al
          menos una dimensión.
    """
    df_combinado = df_ok.copy()
    for nombre_columna, df_dim in resultados.items():
        df_combinado = df_combinado.merge(df_dim, on="filename_key", how="left")

    cols_score = [f"score_{c}" for c in resultados.keys()]
    mascara_completa = df_combinado[cols_score].notna().all(axis=1)
    n_sin_match = int((~mascara_completa).sum())

    if n_sin_match > 0:
        log.warning(f"  {n_sin_match:,} imágenes sin score en al menos una dimensión "
                     "(excluidas del output).")

    return df_combinado[mascara_completa].reset_index(drop=True), n_sin_match


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: CONSTRUCCIÓN DEL OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def _generar_image_id(nombre_archivo: str) -> str:
    """
    Genera un identificador único para la imagen basado en el nombre del
    archivo. Mismo criterio que 03_extraer_embeddings.py y 03b (SHA-256 del
    nombre de archivo, 16 caracteres hex) para que image_id sea comparable
    entre los tres pipelines.
    """
    return hashlib.sha256(nombre_archivo.encode()).hexdigest()[:16]


def construir_output(df_combinado: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Construye el DataFrame final en el orden canónico de columnas.
    """
    df_out = df_combinado.copy()
    df_out["image_id"] = df_out["nombre_archivo"].apply(_generar_image_id)

    cols_base = [
        "image_id", "consecutivo", "ola", "pano_id", "heading", "nombre_archivo",
    ]
    cols_extra  = [c for c in ["llave", "llave_n16"] if c in df_out.columns]
    cols_scores = [f"score_{c}" for c in cfg["dimensiones"].values()]

    idx_ola  = cols_base.index("ola") + 1
    cols_out = cols_base[:idx_ola] + cols_extra + cols_base[idx_ola:] + cols_scores

    return df_out[[c for c in cols_out if c in df_out.columns]]


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5: EXPORTACIÓN A PARQUET
# ──────────────────────────────────────────────────────────────────────────────

def exportar_parquet(df_scores: pd.DataFrame, cfg: dict) -> Path:
    """
    Exporta el DataFrame de scores a formato Parquet.
    """
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta_parquet = out_dir / "scores_streetscore.parquet"
    df_scores.to_parquet(ruta_parquet, engine="pyarrow", index=False)
    log.info(f"Scores exportados: {ruta_parquet}")
    return ruta_parquet


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(
    df_scores: pd.DataFrame,
    n_total:    int,
    n_sin_match: int,
    t_inicio:   float,
    cfg:        dict,
) -> str:
    """
    Genera un reporte de texto plano con estadísticas descriptivas por
    dimensión y cobertura por ola.
    """
    t_total = time.time() - t_inicio
    n_proc  = len(df_scores)

    lineas = [
        "=" * 72,
        "REPORTE DE SCORES DE PERCEPCIÓN URBANA (PLACE PULSE 2.0 / STREETSCORE)",
        "=" * 72,
        "",
        "── 1. Parámetros de ejecución ─────────────────────────────────────────",
        f"  Modelo             : ZenSVI ClassifierPerceptionViT (Place Pulse 2.0)",
        f"  Dimensiones        : {', '.join(cfg['dimensiones'].values())}",
        f"  Batch size         : {cfg['batch_size']}",
        f"  Device             : {cfg['device']}",
        "",
        "── 2. Resumen de imágenes procesadas ──────────────────────────────────",
        f"  Total programadas       : {n_total:>8,}",
        f"  Con score completo      : {n_proc:>8,}",
        f"  Sin match en alguna dim.: {n_sin_match:>8,}",
        f"  Tasa de éxito           : {n_proc / n_total * 100:.1f}%" if n_total > 0 else "",
        "",
        "── 3. Tiempos y rendimiento ────────────────────────────────────────────",
        f"  Tiempo total       : {t_total / 60:.1f} minutos",
        "",
    ]

    cols_scores = [c for c in df_scores.columns if c.startswith("score_")]
    if cols_scores and n_proc > 0:
        lineas += ["── 4. Estadísticas por dimensión ───────────────────────────────────────"]
        for c in cols_scores:
            lineas.append(
                f"  {c:<20} media={df_scores[c].mean():.2f}  std={df_scores[c].std():.2f}  "
                f"min={df_scores[c].min():.2f}  max={df_scores[c].max():.2f}"
            )
        lineas.append("")

    if "ola" in df_scores.columns and n_proc > 0:
        lineas += ["── 5. Cobertura por ola ─────────────────────────────────────────────────"]
        for ola, grp in df_scores.groupby("ola"):
            n_hogares = grp["consecutivo"].nunique()
            lineas.append(
                f"  Ola {ola}: {len(grp):>6,} imágenes  |  {n_hogares:>5,} hogares únicos"
            )
        lineas.append("")

    lineas += [
        "── 6. Notas metodológicas ──────────────────────────────────────────────",
        "  · Scores continuos 0-10, un modelo ViT distinto por dimensión,",
        "    entrenado por ZenSVI sobre comparaciones pareadas de Place Pulse 2.0.",
        "  · Estos scores reflejan percepción de voluntarios mayormente fuera de",
        "    Colombia — su relación con vulnerabilidad a pobreza en este contexto",
        "    es una hipótesis a validar, no un supuesto aceptado.",
        "  · Una imagen queda fuera del output si falta el score en cualquiera",
        "    de las 6 dimensiones (ver sección 2).",
        "",
        "=" * 72,
    ]

    return "\n".join(lineas)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_inicio = time.time()
    log.info("=" * 62)
    log.info("  03d_extraer_scores_streetscore.py")
    log.info(f"  Dimensiones: {list(CONFIG['dimensiones'].values())}")
    log.info("  Si ves ModuleNotFoundError, activa el entorno virtual e instala:")
    log.info("    pip install zensvi")
    log.info("=" * 62)

    # ── 1. Carga las imágenes disponibles ─────────────────────────────────────
    df_ok = cargar_imagenes(CONFIG)
    n_total = len(df_ok)

    # ── 2. Clasifica cada dimensión sobre toda la carpeta de fotos ────────────
    resultados = ejecutar_clasificacion(CONFIG)

    # ── 3. Une los scores con el registro de imágenes exitosas ───────────────
    df_combinado, n_sin_match = combinar_scores(df_ok, resultados)

    # ── 4. Construye el DataFrame de output ───────────────────────────────────
    df_scores = construir_output(df_combinado, CONFIG)

    # ── 5. Exporta a Parquet ──────────────────────────────────────────────────
    ruta_parquet = exportar_parquet(df_scores, CONFIG)

    # ── 6. Genera y guarda el reporte ─────────────────────────────────────────
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    reporte = generar_reporte(
        df_scores,
        n_total=n_total,
        n_sin_match=n_sin_match,
        t_inicio=t_inicio,
        cfg=CONFIG,
    )

    ruta_reporte = out_dir / "reporte_scores_streetscore.txt"
    ruta_reporte.write_text(reporte, encoding="utf-8")

    log.info(f"Reporte guardado: {ruta_reporte}")
    log.info(f"Proceso completado en {(time.time() - t_inicio) / 60:.1f} minutos.")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
