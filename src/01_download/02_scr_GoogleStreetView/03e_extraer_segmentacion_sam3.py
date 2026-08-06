"""
03e_extraer_segmentacion_sam3.py
=================================
Segmenta conceptos visuales específicos (vía pavimentada, huecos, vehículos,
vegetación, etc.) en cada foto de Google Street View descargada por
02_descarga_fotos_GSV.py, usando SAM 3 (Meta, nov. 2025) con prompts de texto.

RESPONSABILIDAD EXCLUSIVA
    Transformar un directorio de imágenes JPEG en una tabla de features
    geométricas por concepto (conteo de instancias, % de área cubierta,
    score de confianza promedio) — una fila por imagen.

    Este script NO realiza:
    · Descarga de imágenes (responsabilidad de script 02).
    · Embeddings globales de imagen (scripts 03 y 03b) ni scores de
      percepción agregada (script 03d) — SAM3 es un pipeline aparte porque
      no da un vector/score por imagen completa, sino máscaras de
      segmentación por objeto/concepto detectado. Es la única fuente de
      "cuántos carros hay" o "qué % de la imagen es vía sin pavimentar",
      en vez de "qué tan similar es la imagen a X en general".
    · Entrenamiento, fine-tuning ni tracking en video (SAM3 también soporta
      video, no usado aquí — las fotos GSV son estáticas).

MODELO
    SAM 3 (facebookresearch/sam3, https://github.com/facebookresearch/sam3),
    "Segment Anything with Concepts": detección + segmentación promptable
    por texto libre (frases cortas tipo "car", "pothole", no captions
    largos como CLIP).

    REQUISITOS (a diferencia de los otros 3 pipelines de este proyecto, que
    corren en CPU/MPS de Mac sin problema):
    · GPU con CUDA 12.6+ es lo único soportado/probado oficialmente por
      Meta. CPU (CONFIG["device"]="cpu") funciona a nivel de código —
      build_sam3_image_model() cae a CPU automáticamente si no hay CUDA, y
      flash-attn (la única dependencia estrictamente CUDA-only) es
      opcional, no obligatoria — pero NO está soportado ni probado por
      Meta, y es un modelo de 848M parámetros con 13 prompts por imagen:
      espera segundos a minutos por imagen. Usar CPU solo para un piloto
      chico que valide el pipeline, nunca para el dataset completo.
      No hay ruta MPS (Apple Silicon) documentada ni probada.
    · Python 3.12+, PyTorch 2.7+.
    · Checkpoint con acceso restringido en Hugging Face: hay que solicitar
      acceso en https://huggingface.co/facebook/sam3 y luego autenticarse
      localmente con `huggingface-cli login` (o variable de entorno
      HF_TOKEN) ANTES de correr este script — a diferencia de
      open_clip/zensvi, no se descarga automáticamente sin autenticación.
      Nota: la solicitud de acceso y la generación del token se pueden
      hacer desde cualquier máquina con navegador, no hace falta estar ya
      en el servidor/entorno con GPU — ver README del proyecto para el
      detalle de estos pasos.
    · Instalación: clonar facebookresearch/sam3 e instalar en modo editable
      (`pip install -e .`) — no está en PyPI como paquete instalable directo
      al momento de escribir este script.

    ESTADO: pensado para correr en un servidor/cloud con GPU NVIDIA (Colab,
    cluster universitario, etc.) como corrida principal; CPU disponible
    solo como piloto de validación en el Mac local. Revisar CONFIG["device"]
    antes de ejecutar.

INPUT
    gsv/registro_descargas.csv  → log de script 02 (imágenes exitosas)
    gsv/inventario_panos.csv    → identificadores del hogar (llave, etc.)

OUTPUTS
    segmentacion_sam3/segmentacion_sam3.parquet      → features (Output 1)
    segmentacion_sam3/reporte_segmentacion_sam3.txt  → informe (Output 2)

FORMATO DE SALIDA
    Una fila por imagen. Por cada concepto en CONFIG["conceptos"] se generan
    tres columnas:
        sam3_n_<concepto>      : número de instancias detectadas (int)
        sam3_area_<concepto>   : fracción de la imagen cubierta por la unión
                                  de las máscaras del concepto (0.0-1.0)
        sam3_score_<concepto>  : score de confianza promedio de las
                                  instancias detectadas (NaN si n=0)

CÓMO CORRER
    python 03e_extraer_segmentacion_sam3.py

    Instalación (en la máquina con GPU):
        git clone https://github.com/facebookresearch/sam3.git
        cd sam3 && pip install -e .

    Autenticación con Hugging Face (se puede preparar desde CUALQUIER
    máquina, sin GPU, con anticipación — no hace falta estar ya en el
    servidor/sala con GPU):
        1. Solicitar acceso en https://huggingface.co/facebook/sam3
        2. Generar un token (scope "Read") en
           https://huggingface.co/settings/tokens
        3. Guardar ESE token en el archivo local (NUNCA se commitea a git,
           ver .gitignore):
               src/01_download/02_scr_GoogleStreetView/.hf_token
           El archivo debe contener únicamente el token, sin comillas ni
           saltos de línea extra. Este script lo lee automáticamente al
           arrancar (ver _cargar_token_hf() más abajo) y ya no hace falta
           correr `huggingface-cli login` a mano en el servidor.
"""

import os
import sys
import time
import hashlib
import logging
import contextlib
from pathlib import Path

# ── Token de Hugging Face (checkpoint gated de SAM3) ───────────────────────────
# Lee el token desde un archivo local NO versionado (ver .gitignore) en vez de
# hardcodearlo en este script: el script sí está trackeado en git con remoto
# en GitHub, y un token pegado literalmente aquí quedaría en el historial del
# repo aunque se borre después. Este archivo local logra el mismo resultado
# práctico (no hay que exportar nada a mano cada sesión) sin ese riesgo.
_RUTA_TOKEN_HF = Path(__file__).resolve().parent / ".hf_token"


def _cargar_token_hf() -> None:
    """
    Si existe .hf_token junto a este script, lo carga en la variable de
    entorno HF_TOKEN — huggingface_hub la detecta automáticamente en
    build_sam3_image_model() sin necesidad de huggingface-cli login.
    Si no existe, no hace nada (se asume login ya hecho o HF_TOKEN ya
    exportado por fuera).
    """
    if _RUTA_TOKEN_HF.exists():
        token = _RUTA_TOKEN_HF.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token


_cargar_token_hf()

try:
    import numpy as np
    import pandas as pd
    import pyarrow as _pyarrow; del _pyarrow  # guard: falla aquí si falta, antes de segmentar
    import torch
    from PIL import Image
    from tqdm import tqdm

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as _err:
    # Si ves este error, o falta el entorno virtual, o (más probable con SAM3)
    # el paquete no está instalado porque no está en PyPI.
    # Solución paso a paso:
    #   1. Activa el entorno virtual:
    #        Mac/Linux : source .venv/bin/activate
    #   2. Instala SAM3 desde el repo (requiere GPU CUDA en la máquina):
    #        git clone https://github.com/facebookresearch/sam3.git
    #        cd sam3 && pip install -e .
    #   3. Solicita acceso al checkpoint y autentícate:
    #        https://huggingface.co/facebook  (buscar el repo de SAM 3)
    #        huggingface-cli login
    print("=" * 62)
    print(f"ERROR: librería no encontrada → {_err}")
    print("Revisa los comentarios junto a los imports en este script.")
    print("SAM3 requiere GPU CUDA y no está en PyPI — ver docstring del script.")
    print("=" * 62)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent   # carpeta GoogleStreetView/

CONFIG = {
    # ── Inputs ─────────────────────────────────────────────────────────────────
    "input_registro":   _HERE / "gsv" / "registro_descargas.csv",
    "input_inventario": _HERE / "gsv" / "inventario_panos.csv",

    # ── Modelo ─────────────────────────────────────────────────────────────────
    # "cuda" | "cpu"  — no hay soporte para "mps" (Apple Silicon): el código
    # de sam3 solo tiene rama explícita para "cuda" (_setup_device_and_mode),
    # cualquier otro valor deja el modelo donde nn.Module lo inicializa por
    # defecto (CPU), pero no hay ruta MPS documentada ni probada.
    #
    # "cpu" NO está soportado oficialmente por Meta (el README lo omite y el
    # notebook de referencia asume CUDA), pero el código base sí lo permite:
    # build_sam3_image_model() cae a "cpu" automáticamente si no hay CUDA
    # disponible, y flash-attn (lo único explícitamente CUDA-only) es una
    # dependencia opcional, no obligatoria. Usar "cpu" solo para un PILOTO
    # chico (unas pocas decenas de imágenes) que valide el pipeline antes de
    # correr el dataset completo en GPU — es un modelo de 848M parámetros
    # corriendo 13 prompts por imagen; en CPU esto puede tomar de segundos a
    # minutos por imagen, inviable para el corpus completo de la ELCA.
    "device": "cuda",

    # Umbral de confianza para conservar una instancia detectada (Sam3Processor
    # ya filtra internamente por esto antes de devolver masks/boxes/scores).
    "confidence_threshold": 0.5,

    # ── Conceptos a segmentar ──────────────────────────────────────────────────
    # Prompts de texto CORTOS (frases nominales, no captions tipo CLIP) — así
    # es como SAM3 espera los prompts de concepto según los ejemplos oficiales
    # ("shoe", "person", "face"). En inglés por la misma razón que en CLIP:
    # mejor cobertura del vocabulario visual con el que se entrenó el modelo.
    "conceptos": {

        # ── Infraestructura vial ──────────────────────────────────────────────
        "via_pavimentada":     "paved road",
        "via_sin_pavimentar":  "unpaved dirt road",
        "anden":                "sidewalk",
        "hueco_via":            "pothole",

        # ── Vivienda y construcción ───────────────────────────────────────────
        "fachada_vivienda":    "house facade",
        "techo":                "roof",
        "pared_sin_terminar":  "exposed unfinished brick wall",

        # ── Activos / vehículos y comercio ────────────────────────────────────
        "carro":                "car",
        "moto":                 "motorcycle",
        "comercio_local":      "storefront",
        "puesto_informal":     "street vendor stall",

        # ── Entorno y espacio público ─────────────────────────────────────────
        "basura":                "garbage",
        "vegetacion":            "tree",
        "grafiti":               "graffiti",
    },

    # ── Salida ─────────────────────────────────────────────────────────────────
    "output_dir": _HERE / "segmentacion_sam3",
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
# Idéntico al Módulo 1 de 03, 03b y 03d: el insumo no depende del modelo
# usado para procesar las imágenes.

def cargar_imagenes(cfg: dict) -> pd.DataFrame:
    """
    Lee el registro de descargas y retorna las imágenes disponibles para
    segmentar.

    Filtros aplicados:
    · exito == True  → solo imágenes descargadas con éxito por script 02.
    · ruta_archivo no nula.

    Si el inventario está disponible, incorpora identificadores adicionales
    del hogar (llave, llave_n16) mediante un join por (consecutivo, ola, pano_id).
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

    return df_ok.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: CARGA DEL MODELO SAM3
# ──────────────────────────────────────────────────────────────────────────────

def _contexto_inferencia(device: str):
    """
    Contexto de autocast usado durante la inferencia.

    El notebook oficial de SAM3 activa TF32 (Ampere+) y autocast a bfloat16
    incondicionalmente asumiendo CUDA — ambas son optimizaciones específicas
    de GPU sin beneficio (y con riesgo de romper) en CPU. Aquí solo se
    activan si device == "cuda"; en CPU se usa un contexto nulo y precisión
    float32 normal.
    """
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def cargar_modelo(cfg: dict) -> Sam3Processor:
    """
    Carga SAM3 y retorna el Sam3Processor listo para segmentar por texto.

    El vocabulario BPE (bpe_simple_vocab_16e6.txt.gz) se busca dentro del
    propio paquete instalado — mismo patrón que el notebook oficial
    examples/sam3_image_predictor_example.ipynb.
    """
    import os

    if cfg["device"] == "cpu":
        log.warning("-" * 62)
        log.warning("AVISO: corriendo SAM3 en CPU — no soportado oficialmente por Meta.")
        log.warning("Es un modelo de 848M parámetros con 13 prompts por imagen;")
        log.warning("espera varios segundos a minutos POR IMAGEN. Usar solo para un")
        log.warning("piloto chico, no para el dataset completo de la ELCA.")
        log.warning("-" * 62)

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path  = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    log.info("Cargando SAM3 (requiere checkpoint autenticado de Hugging Face) …")
    model = build_sam3_image_model(bpe_path=bpe_path, device=cfg["device"])

    processor = Sam3Processor(
        model,
        device=cfg["device"],
        confidence_threshold=cfg["confidence_threshold"],
    )
    log.info(f"SAM3 cargado. Device: {cfg['device']}, "
             f"confidence_threshold: {cfg['confidence_threshold']}")
    return processor


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: SEGMENTACIÓN POR IMAGEN
# ──────────────────────────────────────────────────────────────────────────────

def _resumir_deteccion(state: dict) -> tuple:
    """
    Reduce el resultado crudo de un prompt de texto (state["masks"],
    state["boxes"], state["scores"], ver Sam3Processor._forward_grounding en
    el código fuente de sam3) a tres estadísticos por concepto:

    · n_instancias : cantidad de instancias detectadas sobre confidence_threshold.
    · area_frac    : fracción de píxeles de la imagen cubiertos por la UNIÓN
                      de todas las máscaras del concepto (evita doble conteo
                      si dos instancias del mismo concepto se solapan).
    · score_medio  : promedio de los scores de confianza de las instancias
                      detectadas. NaN si no hubo ninguna detección.

    state["masks"] tiene forma (N, 1, H, W) booleana (el canal extra viene
    de cómo SAM3 interpola las máscaras internamente) — se aplana a (N, H*W)
    antes de calcular la unión, sin asumir que el canal siempre sea 1.
    """
    masks  = state.get("masks")
    scores = state.get("scores")

    n_instancias = 0 if masks is None else int(masks.shape[0])

    if n_instancias == 0:
        return 0, 0.0, float("nan")

    masks_flat = masks.reshape(n_instancias, -1)          # (N, H*W)
    union      = masks_flat.any(dim=0)                     # (H*W,)
    area_frac  = union.float().mean().item()
    score_medio = scores.mean().item()

    return n_instancias, area_frac, score_medio


def segmentar_imagen(processor: Sam3Processor, ruta_imagen: str, conceptos: dict) -> dict:
    """
    Corre los N conceptos de CONFIG["conceptos"] sobre una sola imagen.

    Reutiliza el mismo `inference_state` (backbone de imagen ya calculado
    una vez con set_image) para todos los conceptos — solo se recalcula el
    encoder de texto y el grounding en cada set_text_prompt, evitando
    recomputar el backbone visual N veces por imagen.

    Retorna:
        {f"sam3_n_<nombre>": int, f"sam3_area_<nombre>": float,
         f"sam3_score_<nombre>": float, ...}  — 3 columnas por concepto.
    """
    image = Image.open(ruta_imagen).convert("RGB")
    state = processor.set_image(image)

    fila = {}
    for nombre, prompt in conceptos.items():
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

        n_instancias, area_frac, score_medio = _resumir_deteccion(state)
        fila[f"sam3_n_{nombre}"]     = n_instancias
        fila[f"sam3_area_{nombre}"]  = area_frac
        fila[f"sam3_score_{nombre}"] = score_medio

    return fila


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: PROCESAMIENTO DEL LOTE DE IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────

def procesar_imagenes(processor: Sam3Processor, df_ok: pd.DataFrame, cfg: dict) -> tuple:
    """
    Ejecuta segmentar_imagen() para cada imagen exitosa del registro.

    A diferencia de 03/03b (que procesan por lotes con DataLoader) y de 03d
    (que delega el batching a ZenSVI), aquí se procesa una imagen a la vez:
    Sam3Processor está pensado para uso interactivo/por imagen (set_image +
    múltiples prompts), no para batches de imágenes distintas en un solo
    forward — batchear imágenes distintas requeriría set_image_batch() y
    coordinar prompts de texto distintos por imagen, que no es el caso aquí
    (mismos N conceptos para todas las imágenes).

    Retorna:
        (filas, indices_error)
        · filas         : lista de dicts, una por imagen procesada con éxito
        · indices_error : posiciones en df_ok que fallaron al abrir/procesar
    """
    filas         = []
    indices_error = []

    log.info(f"Segmentando {len(df_ok):,} imágenes × {len(cfg['conceptos'])} conceptos …")

    with _contexto_inferencia(cfg["device"]):
        for idx, ruta in enumerate(tqdm(df_ok["ruta_archivo"].tolist(), desc="Imágenes", unit="img")):
            try:
                fila = segmentar_imagen(processor, ruta, cfg["conceptos"])
                fila["_idx_df_ok"] = idx
                filas.append(fila)
            except Exception as e:
                log.warning(f"  No se pudo procesar imagen [{idx}]: {ruta} — {e}")
                indices_error.append(idx)

    if not filas:
        log.error("-" * 62)
        log.error("ERROR: Ninguna imagen pudo segmentarse.")
        log.error(f"Total de imagenes intentadas: {len(df_ok):,}")
        log.error(f"Errores: {len(indices_error):,}")
        log.error("-" * 62)
        sys.exit(1)

    log.info(f"Segmentación completa: {len(filas):,} exitosas, {len(indices_error):,} errores.")
    return filas, indices_error


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5: CONSTRUCCIÓN DEL OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def _generar_image_id(nombre_archivo: str) -> str:
    """
    Genera un identificador único para la imagen basado en el nombre del
    archivo. Mismo criterio (SHA-256, 16 caracteres hex) que 03, 03b y 03d
    para que image_id sea comparable entre los cuatro pipelines.
    """
    return hashlib.sha256(nombre_archivo.encode()).hexdigest()[:16]


def construir_output(df_ok: pd.DataFrame, filas: list) -> pd.DataFrame:
    """
    Une las filas de segmentación (una por imagen procesada) con los
    identificadores de df_ok, usando "_idx_df_ok" como llave de posición.
    """
    df_seg = pd.DataFrame(filas)
    indices_ok = df_seg.pop("_idx_df_ok").tolist()

    df_proc = df_ok.iloc[indices_ok].copy().reset_index(drop=True)
    df_proc["image_id"] = df_proc["nombre_archivo"].apply(_generar_image_id)

    df_out = pd.concat([df_proc.reset_index(drop=True), df_seg.reset_index(drop=True)], axis=1)

    cols_base = [
        "image_id", "consecutivo", "ola", "pano_id", "heading", "nombre_archivo",
    ]
    cols_extra = [c for c in ["llave", "llave_n16"] if c in df_out.columns]
    cols_sam3  = [c for c in df_out.columns if c.startswith("sam3_")]

    idx_ola  = cols_base.index("ola") + 1
    cols_out = cols_base[:idx_ola] + cols_extra + cols_base[idx_ola:] + cols_sam3

    return df_out[[c for c in cols_out if c in df_out.columns]]


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: EXPORTACIÓN A PARQUET
# ──────────────────────────────────────────────────────────────────────────────

def exportar_parquet(df_seg: pd.DataFrame, cfg: dict) -> Path:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta_parquet = out_dir / "segmentacion_sam3.parquet"
    df_seg.to_parquet(ruta_parquet, engine="pyarrow", index=False)
    log.info(f"Segmentación exportada: {ruta_parquet}")
    return ruta_parquet


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 7: REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(
    df_seg:      pd.DataFrame,
    n_total:     int,
    n_errores:   int,
    t_inicio:    float,
    cfg:         dict,
) -> str:
    t_total = time.time() - t_inicio
    n_proc  = len(df_seg)

    lineas = [
        "=" * 72,
        "REPORTE DE SEGMENTACIÓN SAM3 (PROMPTS DE CONCEPTO)",
        "=" * 72,
        "",
        "── 1. Parámetros de ejecución ─────────────────────────────────────────",
        f"  Modelo               : SAM3 (facebookresearch/sam3)",
        f"  Conceptos            : {len(cfg['conceptos'])}",
        f"  Confidence threshold : {cfg['confidence_threshold']}",
        f"  Device               : {cfg['device']}",
        "",
        "── 2. Resumen de imágenes procesadas ──────────────────────────────────",
        f"  Total programadas  : {n_total:>8,}",
        f"  Exitosas           : {n_proc:>8,}",
        f"  Con error          : {n_errores:>8,}",
        f"  Tasa de éxito      : {n_proc / n_total * 100:.1f}%" if n_total > 0 else "",
        "",
        "── 3. Tiempos y rendimiento ────────────────────────────────────────────",
        f"  Tiempo total       : {t_total / 60:.1f} minutos",
        f"  Tiempo por imagen  : {t_total / n_proc:.2f} segundos" if n_proc > 0 else "",
        "",
    ]

    if n_proc > 0:
        lineas += ["── 4. Detecciones por concepto ─────────────────────────────────────────"]
        for nombre in cfg["conceptos"]:
            col_n    = f"sam3_n_{nombre}"
            col_area = f"sam3_area_{nombre}"
            if col_n not in df_seg.columns:
                continue
            pct_presente = (df_seg[col_n] > 0).mean() * 100
            area_media   = df_seg.loc[df_seg[col_n] > 0, col_area].mean()
            lineas.append(
                f"  {nombre:<20} presente en {pct_presente:5.1f}% de imágenes  |  "
                f"área media (cuando presente): {area_media:.3f}"
                if pd.notna(area_media) else
                f"  {nombre:<20} presente en {pct_presente:5.1f}% de imágenes"
            )
        lineas.append("")

    if "ola" in df_seg.columns and n_proc > 0:
        lineas += ["── 5. Cobertura por ola ─────────────────────────────────────────────────"]
        for ola, grp in df_seg.groupby("ola"):
            n_hogares = grp["consecutivo"].nunique()
            lineas.append(
                f"  Ola {ola}: {len(grp):>6,} imágenes  |  {n_hogares:>5,} hogares únicos"
            )
        lineas.append("")

    lineas += [
        "── 6. Notas metodológicas ──────────────────────────────────────────────",
        "  · sam3_n_<concepto>: número de instancias detectadas con confianza",
        "    por encima de confidence_threshold.",
        "  · sam3_area_<concepto>: fracción de la imagen cubierta por la unión",
        "    de las máscaras del concepto (0.0 si no se detectó nada).",
        "  · sam3_score_<concepto>: confianza promedio de las instancias",
        "    detectadas; NaN si sam3_n_<concepto> == 0.",
        "  · A diferencia de los scores de CLIP (similitud global) o Place",
        "    Pulse (percepción agregada), estas son features geométricas",
        "    directas — conteos y áreas — de objetos/superficies concretas.",
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
    log.info("  03e_extraer_segmentacion_sam3.py")
    log.info(f"  Conceptos configurados: {len(CONFIG['conceptos'])}")
    log.info(f"  Device: {CONFIG['device']} — SAM3 requiere GPU CUDA.")
    log.info("  Si ves ModuleNotFoundError, revisa la instalación de SAM3")
    log.info("  y la autenticación de Hugging Face en el docstring del script.")
    log.info("=" * 62)

    # ── 1. Carga las imágenes disponibles ─────────────────────────────────────
    df_ok = cargar_imagenes(CONFIG)
    n_total = len(df_ok)

    # ── 2. Carga SAM3 ──────────────────────────────────────────────────────────
    processor = cargar_modelo(CONFIG)

    # ── 3. Segmenta cada imagen sobre todos los conceptos ─────────────────────
    filas, indices_error = procesar_imagenes(processor, df_ok, CONFIG)

    # ── 4. Construye el DataFrame de output ───────────────────────────────────
    df_seg = construir_output(df_ok, filas)

    # ── 5. Exporta a Parquet ──────────────────────────────────────────────────
    ruta_parquet = exportar_parquet(df_seg, CONFIG)

    # ── 6. Genera y guarda el reporte ─────────────────────────────────────────
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    reporte = generar_reporte(
        df_seg,
        n_total=n_total,
        n_errores=len(indices_error),
        t_inicio=t_inicio,
        cfg=CONFIG,
    )

    ruta_reporte = out_dir / "reporte_segmentacion_sam3.txt"
    ruta_reporte.write_text(reporte, encoding="utf-8")

    log.info(f"Reporte guardado: {ruta_reporte}")
    log.info(f"Proceso completado en {(time.time() - t_inicio) / 60:.1f} minutos.")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
