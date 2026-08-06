"""
03b_extraer_embeddings_clip.py
===============================
Extrae embeddings visuales CLIP a partir de las imágenes de Google Street View
descargadas por 02_descarga_fotos_GSV.py, y calcula puntajes de similitud
zero-shot contra un conjunto de prompts de texto definidos en CONFIG.

RESPONSABILIDAD EXCLUSIVA
    Transformar un directorio de imágenes JPEG en:
    (a) una matriz de embeddings CLIP (espacio compartido imagen-texto), y
    (b) puntajes de similitud coseno contra prompts de texto (zero-shot scoring).

    Este script NO realiza:
    · Descarga de imágenes (responsabilidad de script 02).
    · Extracción de embeddings con VGG19/ResNet50/Places365 (responsabilidad
      de script 03_extraer_embeddings.py — pipeline separado por naturaleza
      distinta del modelo: CLIP no comparte preprocesamiento ImageNet y
      añade un codificador de texto).
    · Entrenamiento, fine-tuning ni clasificación.
    · Selección de variables ni análisis estadístico.

INPUT
    gsv/registro_descargas.csv  → log de script 02 (imágenes exitosas)
    gsv/inventario_panos.csv    → identificadores del hogar (llave, etc.)

OUTPUTS
    embeddings/embeddings_clip.parquet      → embeddings + clip_score_* (Output 1)
    embeddings/reporte_embeddings_clip.txt  → informe (Output 2)

FORMATO DE SALIDA
    Una fila por imagen. Columnas principales:
        image_id, consecutivo, ola, pano_id, heading, nombre_archivo,
        llave (si disponible), llave_n16 (si disponible),
        modelo, embedding_dim, embedding (lista de float32),
        clip_score_<prompt_1>, clip_score_<prompt_2>, ... (una por prompt en CONFIG)

CÓMO CORRER
    python 03b_extraer_embeddings_clip.py

    Para cambiar la variante de CLIP: modificar CONFIG["variante_clip"].
    Para cambiar los prompts de scoring: modificar CONFIG["prompts"].
    Para procesar en GPU: modificar CONFIG["device"] = "cuda" o "mps".

ESTADO
    Esqueleto. Pendiente: carga del modelo CLIP (Módulo 2), definición de
    prompts (CONFIG["prompts"]) y codificación de texto (Módulo 5).
"""

import sys
import time
import hashlib
import logging
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import pyarrow as _pyarrow; del _pyarrow  # guard: falla aquí si falta, antes de extraer embeddings
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    # TODO: agregar import de open_clip cuando se implemente el Módulo 2.
    # import open_clip
except ImportError as _err:
    # Si ves este error, el entorno virtual no está activo o las dependencias
    # no están instaladas.
    # Solución paso a paso:
    #   1. Activa el entorno virtual:
    #        Mac/Linux : source .venv/bin/activate
    #        Windows   : .venv\Scripts\activate
    #   2. Instala las dependencias generales:
    #        pip install -r requirements.txt
    #   3. Si falta torch o open_clip específicamente:
    #        CPU / Mac  : pip install torch open_clip_torch
    #        GPU CUDA   : pip install torch --index-url https://download.pytorch.org/whl/cu118  open_clip_torch
    #        Apple MPS  : pip install torch open_clip_torch   (ya incluye MPS en macOS ≥ 12)
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
    # ── Modelo ─────────────────────────────────────────────────────────────────
    # Variante de CLIP (open_clip). Ejemplos: "ViT-B-32", "ViT-L-14", "RN50".
    "variante_clip": "ViT-B-32",

    # Checkpoint preentrenado a usar con open_clip (ver open_clip.list_pretrained()).
    # Ejemplo: "openai" (pesos originales de OpenAI) o "laion2b_s34b_b79k".
    "pretrained_clip": "openai",

    # ── Inputs ─────────────────────────────────────────────────────────────────
    # Registro de descargas producido por 02_descarga_fotos_GSV.py.
    # Contiene las rutas de archivo y el estado de cada imagen (exito=True/False).
    "input_registro": _HERE / "gsv" / "registro_descargas.csv",

    # Inventario producido por 01_analisis_cobertura_gsv.py.
    # Se usa para incorporar identificadores adicionales del hogar (llave, llave_n16).
    # Si no existe, el output incluirá solo consecutivo y ola.
    "input_inventario": _HERE / "gsv" / "inventario_panos.csv",

    # ── Prompts de scoring zero-shot ───────────────────────────────────────────
    # Cada entrada define un concepto visual candidato a asociarse con
    # vulnerabilidad a pobreza monetaria. Genera una columna
    # "clip_score_<nombre>" en el output (promedio de similitud coseno entre
    # el embedding de imagen y el embedding de cada template, normalizados L2 —
    # ver Módulo 5, ensemble de 2-3 variaciones de redacción por concepto).
    #
    # IMPORTANTE — naturaleza exploratoria de la lista:
    # Esta lista fue discutida entre perfiles de economía de la pobreza, ML,
    # trabajo social, antropología y sociología de la pobreza/violencia en
    # Colombia, y una persona con experiencia vivida de pobreza. Ninguno de
    # los signos de correlación esperados está confirmado — el campo
    # "hipotesis" documenta el argumento a favor de cada concepto y el campo
    # "riesgo" documenta la objeción/confusor que el mismo equipo levantó.
    # Ambos deben leerse como preguntas a validar empíricamente (correlación
    # de clip_score_<nombre> contra el índice de activos / SISBEN de la ELCA
    # y, para las olas repetidas, contra transiciones reales a pobreza), no
    # como supuestos ya aceptados para construir features de un modelo.
    "prompts": {

        # ── Vivienda: material y estado de construcción ──────────────────────
        "vivienda_sin_terminar": {
            "templates": [
                "a photo of a house with unfinished or bare brick walls",
                "a picture of a house with exposed brick or block walls, no plaster",
                "a street view image of a house under unfinished construction",
            ],
            "hipotesis": "Construcción inconclusa como proxy de restricción de "
                          "liquidez para terminar la vivienda (economista).",
            "riesgo":    "Puede reflejar fase normal de autoconstrucción "
                          "progresiva (común y no estigmatizante en Colombia), "
                          "no necesariamente pobreza actual (antropólogo).",
        },
        "vivienda_dañada": {
            "templates": [
                "a photo of a house with visible structural damage or cracks",
                "a picture of a deteriorated house with cracked walls",
                "a street view image of a damaged residential building",
            ],
            "hipotesis": "Daño estructural visible como proxy directo de "
                          "privación material del hogar.",
            "riesgo":    "Confundido con antigüedad de la vivienda o "
                          "exposición a eventos (sismos, inundaciones) "
                          "no relacionados con el ingreso del hogar.",
        },
        "techo_ligero": {
            "templates": [
                "a photo of a house with a corrugated metal or zinc roof",
                "a picture of a house with a tin roof",
                "a street view image of a house with a sheet metal roof",
            ],
            "hipotesis": "Material de techo de bajo costo asociado a menor "
                          "ingreso del hogar.",
            "riesgo":    "En zonas rurales y cálidas de Colombia el zinc es "
                          "material funcional estándar independientemente del "
                          "ingreso — riesgo de confundirse con clima/región, "
                          "no con pobreza (antropólogo).",
        },
        "vivienda_terminada": {
            "templates": [
                "a photo of a house with a finished, painted facade",
                "a picture of a well-finished residential building",
                "a street view image of a house with a complete painted exterior",
            ],
            "hipotesis": "Contraparte de vivienda_sin_terminar: se espera "
                          "correlación inversa con vulnerabilidad.",
            "riesgo":    "Mismo riesgo que vivienda_sin_terminar en dirección "
                          "opuesta — validar que no sea solo el negativo "
                          "especular del mismo concepto (redundancia).",
        },

        # ── Infraestructura vial ──────────────────────────────────────────────
        "via_sin_pavimentar": {
            "templates": [
                "a photo of an unpaved dirt road",
                "a picture of a street without pavement",
                "a street view image of a dirt road in a neighborhood",
            ],
            "hipotesis": "Falta de pavimentación como proxy de baja inversión "
                          "pública / vulnerabilidad de infraestructura.",
            "riesgo":    "CONFUNDIDO CON RURALIDAD: casi ninguna vía rural "
                          "colombiana está pavimentada sin importar el ingreso "
                          "del hogar. No usar como score global sin controlar "
                          "por el indicador urbano/rural ya existente en el "
                          "panel — riesgo de medir '¿es rural?' en vez de "
                          "'¿es vulnerable?' (economista).",
        },
        "via_pavimentada": {
            "templates": [
                "a photo of a paved street with sidewalks",
                "a picture of a paved urban street",
                "a street view image of a paved road with curbs",
            ],
            "hipotesis": "Contraparte de via_sin_pavimentar.",
            "riesgo":    "Mismo confusor rural/urbano que via_sin_pavimentar, "
                          "en dirección opuesta.",
        },
        "via_deteriorada": {
            "templates": [
                "a photo of a paved street full of potholes",
                "a picture of a damaged road with potholes",
                "a street view image of a deteriorated paved street",
            ],
            "hipotesis": "Mal mantenimiento vial como proxy de vulnerabilidad "
                          "del entorno del hogar.",
            "riesgo":    "Mide calidad de mantenimiento MUNICIPAL/barrial, no "
                          "del hogar específico — es una variable de contexto "
                          "de vecindario, no de privación del hogar fotografiado "
                          "(trabajador social).",
        },

        # ── Servicios básicos ──────────────────────────────────────────────────
        "cableado_informal": {
            "templates": [
                "a photo of exposed or informally connected electrical wiring "
                "on a building",
                "a picture of a house with tangled exposed electrical cables",
                "a street view image showing informal electrical connections",
            ],
            "hipotesis": "Conexión eléctrica informal/expuesta como proxy de "
                          "falta de acceso a servicios formales — privación "
                          "material más directa que indicadores estéticos "
                          "(trabajador social).",
            "riesgo":    "CLIP puede tener baja sensibilidad para detectar "
                          "cableado específico en imágenes de calle a "
                          "distancia — validar tasa de detección antes de "
                          "confiar en el score.",
        },

        # ── Espacio público / entorno ──────────────────────────────────────────
        "basura_acumulada": {
            "templates": [
                "a photo of uncollected garbage on a public street",
                "a picture of a street with trash and litter",
                "a street view image of accumulated waste on a sidewalk",
            ],
            "hipotesis": "Acumulación de basura como proxy de vulnerabilidad "
                          "del entorno.",
            "riesgo":    "Mide servicio de recolección MUNICIPAL, no privación "
                          "del hogar — variable de contexto barrial, no de "
                          "vulnerabilidad del hogar (trabajador social).",
        },
        "entorno_cuidado": {
            "templates": [
                "a photo of a clean, well-maintained street",
                "a picture of a tidy residential street",
                "a street view image of a well-kept neighborhood street",
            ],
            "hipotesis": "Contraparte de basura_acumulada: se esperaría "
                          "correlación inversa con vulnerabilidad.",
            "riesgo":    "OBJECIÓN FUERTE (persona con experiencia de "
                          "pobreza): mantener el espacio limpio con recursos "
                          "escasos es común en hogares de bajos ingresos; "
                          "'limpieza' correlaciona con esfuerzo/cultura, no "
                          "con ingreso. Es plausible que este score NO "
                          "correlacione con pobreza, o incluso vaya en "
                          "dirección contraria a la asumida — tratar como "
                          "hipótesis abierta, no como proxy validado.",
        },
        "vegetacion": {
            "templates": [
                "a photo of a street with trees and green vegetation",
                "a picture of a leafy green residential street",
                "a street view image of a street lined with trees",
            ],
            "hipotesis": "Exploratorio — presencia de zonas verdes podría "
                          "asociarse a mayor inversión pública/privada.",
            "riesgo":    "Dirección ambigua: puede reflejar inversión urbana "
                          "(zonas verdes gestionadas) o simplemente baja "
                          "densidad rural (vegetación espontánea) — dos "
                          "fenómenos opuestos en términos de vulnerabilidad. "
                          "No asumir signo, solo explorar.",
        },
        "grafiti": {
            "templates": [
                "a photo of a wall with graffiti or vandalism",
                "a picture of a street wall covered in graffiti",
                "a street view image of a building with graffiti",
            ],
            "hipotesis": "Exploratorio — posible marcador de descuido/"
                          "informalidad urbana.",
            "riesgo":    "Ambiguo: el grafiti también es común en zonas "
                          "urbanas de ingresos medios/altos con arte urbano "
                          "'legítimo' — no discrimina bien vulnerabilidad sin "
                          "contexto adicional. Solo exploratorio.",
        },

        # ── Densidad / tipología de asentamiento ────────────────────────────
        "asentamiento_informal": {
            "templates": [
                "a photo of an informal settlement",
                "a picture of an informal urban settlement area",
                "a street view image of an informal housing settlement",
            ],
            "hipotesis": "Marcador directo de informalidad urbana asociada a "
                          "vulnerabilidad.",
            "riesgo":    "OBJECIÓN (antropólogo): 'informal settlement' es "
                          "una categoría cargada en el imaginario con el que "
                          "CLIP fue entrenado (fotos de stock etiquetadas "
                          "'slum'/'favela' desde una mirada occidental) — el "
                          "score puede capturar un sesgo de representación "
                          "visual heredado del entrenamiento, no informalidad "
                          "real. Interpretar con cautela; idealmente "
                          "contrastar contra si mide señales concretas "
                          "(densidad, material) o solo 'estilo visual "
                          "asociado a pobreza' en el imaginario de CLIP.",
        },
        "zona_rural": {
            "templates": [
                "a photo of a rural area with sparse housing",
                "a picture of a countryside road with few houses",
                "a street view image of a rural landscape",
            ],
            "hipotesis": "No es hipótesis de vulnerabilidad — es un chequeo "
                          "de consistencia: ¿el score de CLIP coincide con el "
                          "indicador rural/urbano ya conocido por coordenadas "
                          "GPS en el panel?",
            "riesgo":    "Redundante con dato ya existente (economista): no "
                          "usar como feature independiente de vulnerabilidad, "
                          "solo como validación de que CLIP 've' correctamente "
                          "el tipo de zona (riesgo de multicolinealidad si se "
                          "usa como predictor).",
        },
        "zona_urbana_densa": {
            "templates": [
                "a photo of a dense urban neighborhood",
                "a picture of a crowded city block",
                "a street view image of a dense urban street",
            ],
            "hipotesis": "Mismo chequeo de consistencia que zona_rural, "
                          "polo opuesto.",
            "riesgo":    "Mismo riesgo de redundancia que zona_rural.",
        },

        # ── Marcadores de activos visibles ──────────────────────────────────
        "vehiculos_particulares": {
            "templates": [
                "a photo of a street with parked cars",
                "a picture of cars parked along a residential street",
                "a street view image showing parked automobiles",
            ],
            "hipotesis": "Presencia de carros como marcador de mayor ingreso.",
            "riesgo":    "En el rango de ingresos donde se mueve la ELCA, el "
                          "carro es un activo de clase media-alta poco común; "
                          "riesgo de baja varianza (casi siempre 'ausente') y "
                          "por tanto poco informativo para discriminar dentro "
                          "de la población objetivo (economista) — comparar "
                          "varianza empírica contra moto_estacionada.",
        },
        "moto_estacionada": {
            "templates": [
                "a photo of a street with parked motorcycles",
                "a picture of motorcycles parked outside houses",
                "a street view image showing parked motorcycles",
            ],
            "hipotesis": "Activo más relevante que el carro para el rango de "
                          "ingresos de la ELCA en Colombia — mayor varianza "
                          "esperada y mejor poder discriminante (economista).",
            "riesgo":    "También usada como herramienta de trabajo (mototaxi, "
                          "domicilios) y no solo como activo de consumo — "
                          "posible confusión entre 'activo' y 'medio de "
                          "subsistencia informal'.",
        },
        "rejas_seguridad": {
            "templates": [
                "a photo of a house with security bars on windows",
                "a picture of a house with a metal security fence",
                "a street view image of a house with barred windows",
            ],
            "hipotesis": "Rejas como inversión en protección de activos, "
                          "asociada a mayor ingreso.",
            "riesgo":    "OBJECIÓN (sociólogo, pobreza y violencia en "
                          "Colombia): dirección AMBIGUA — también se instalan "
                          "por miedo/inseguridad en zonas con presencia de "
                          "violencia, independientemente del ingreso, o "
                          "incluso más en zonas vulnerables expuestas a "
                          "violencia. No asumir signo ex ante; el sociólogo "
                          "recomienda no construir hipótesis direccional "
                          "hasta ver el dato.",
        },
        "comercio_local": {
            "templates": [
                "a photo of a small business or informal commercial storefront",
                "a picture of a small local shop front",
                "a street view image of an informal street vendor stall",
            ],
            "hipotesis": "Exploratorio — presencia de comercio como proxy de "
                          "actividad económica del entorno.",
            "riesgo":    "Ambiguo (trabajador social): puede reflejar "
                          "dinamismo económico positivo O informalidad/"
                          "precariedad laboral (subsistencia sin acceso a "
                          "empleo formal) — mismo score, interpretaciones "
                          "opuestas. Requiere lectura cualitativa adicional, "
                          "no solo el número.",
        },
    },

    # ── Inferencia ─────────────────────────────────────────────────────────────
    "batch_size": 32,          # imágenes por lote; reducir a 8 si hay OOM en GPU
    "device":     "auto",      # "auto" | "cpu" | "cuda" | "mps"

    # ── Salida ─────────────────────────────────────────────────────────────────
    "output_dir": _HERE / "embeddings",
}


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

# Dimensión del embedding por variante de CLIP.
# TODO: completar según las variantes que se vayan a usar.
_EMBEDDING_DIM = {
    "ViT-B-32": 512,
    "ViT-L-14": 768,
    "RN50":     1024,
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
# Idéntico al Módulo 1 de 03_extraer_embeddings.py: el insumo (registro +
# inventario) no depende del modelo usado para extraer embeddings.

def cargar_imagenes(cfg: dict) -> pd.DataFrame:
    """
    Lee el registro de descargas y retorna las imágenes disponibles para
    extraer embeddings.

    Filtros aplicados:
    · exito == True  → solo imágenes descargadas con éxito por script 02.
    · ruta_archivo no nula.

    Si el inventario está disponible, incorpora identificadores adicionales del
    hogar (llave, llave_n16) mediante un join por (consecutivo, ola, pano_id).

    Retorna:
        DataFrame con una fila por imagen, con columnas de identificadores
        y la ruta absoluta del archivo en disco.
    """
    path_reg = Path(cfg["input_registro"])
    if not path_reg.exists():
        log.error("-" * 62)
        log.error("ERROR: Registro de descargas no encontrado.")
        log.error(f"Buscado en: {path_reg.resolve()}")
        log.error("SOLUCION: Ejecuta primero el script anterior:")
        log.error("  python 02_descarga_fotos_GSV.py")
        log.error("Ese script genera: gsv/registro_descargas.csv")
        log.error("-" * 62)
        sys.exit(1)

    reg = pd.read_csv(
        path_reg,
        dtype={"consecutivo": str, "pano_id": str},
        low_memory=False,
    )
    log.info(f"Registro cargado: {len(reg):,} filas ({path_reg.name})")

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
# MÓDULO 2: CARGA DEL MODELO CLIP
# ──────────────────────────────────────────────────────────────────────────────

def _resolver_device(device_cfg: str) -> torch.device:
    """
    Resuelve el dispositivo de cómputo a usar.
    "auto" selecciona CUDA si está disponible, luego MPS (Apple Silicon), luego CPU.
    """
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():   # Apple Silicon
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def cargar_modelo_clip(cfg: dict) -> tuple:
    """
    Carga el modelo CLIP (encoder de imagen + encoder de texto) y su
    preprocesamiento asociado vía open_clip.

    TODO: implementar usando open_clip.create_model_and_transforms(
        cfg["variante_clip"], pretrained=cfg["pretrained_clip"]
    ) y open_clip.get_tokenizer(cfg["variante_clip"]).

    Retorna: (model, tokenizer, preprocess, embedding_dim, device)
    """
    raise NotImplementedError(
        "Módulo 2 pendiente: carga de CLIP vía open_clip. "
        "Ver TODO en cargar_modelo_clip()."
    )


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 3: DATASET DE IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────
# Genérico: igual que en 03_extraer_embeddings.py, recibe cualquier `transform`
# (aquí, el preprocess que retorna open_clip.create_model_and_transforms).

class ImageDataset(Dataset):
    """
    Dataset de PyTorch que carga imágenes JPEG desde el disco y aplica
    la transformación del modelo.

    Si una imagen no puede cargarse (archivo corrompido, permisos, etc.),
    __getitem__ retorna None en lugar de lanzar una excepción, para que
    el proceso continúe con las imágenes restantes.
    """

    def __init__(self, rutas: list, transform):
        self.rutas     = rutas            # lista de rutas absolutas (strings)
        self.transform = transform        # preprocesamiento del modelo

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, idx: int):
        ruta = self.rutas[idx]
        try:
            img = Image.open(ruta).convert("RGB")   # fuerza 3 canales (descarta alpha)
            tensor = self.transform(img)             # aplica preprocesamiento
            return tensor, idx                       # retorna tensor + índice de origen
        except Exception as e:
            log.warning(f"  No se pudo cargar imagen [{idx}]: {ruta} — {e}")
            return None, idx


def _collate_filtrar_nulos(batch: list) -> tuple:
    """
    Función de collation personalizada: separa los ejemplos válidos (tensor cargado)
    de los inválidos (tensor es None, imagen corrompida o ilegible).

    Retorna: (tensores_validos, indices_ok, indices_error)
    """
    validos   = [(tensor, idx) for tensor, idx in batch if tensor is not None]
    invalidos = [idx           for tensor, idx in batch if tensor is None]

    if not validos:
        return torch.tensor([]), [], invalidos

    tensores, indices = zip(*validos)
    return torch.stack(tensores), list(indices), invalidos


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 4: EXTRACCIÓN DE EMBEDDINGS DE IMAGEN
# ──────────────────────────────────────────────────────────────────────────────

def extraer_embeddings(
    model,
    transform,
    df_ok:     pd.DataFrame,
    cfg:       dict,
) -> tuple:
    """
    Ejecuta el encoder de imagen de CLIP sobre todas las imágenes exitosas
    en lotes (batches). Los embeddings resultantes se normalizan L2, siguiendo
    la convención estándar de CLIP para similitud coseno.

    TODO: dentro del forward, usar model.encode_image(tensores) en vez de
    model(tensores) — CLIP expone encoders separados para imagen y texto.

    Retorna:
        (embeddings, indices_ok, indices_error)
        · embeddings   : np.ndarray (N_ok, D) de float32, normalizado L2
        · indices_ok   : lista de posiciones en df_ok con embedding exitoso
        · indices_error: lista de posiciones en df_ok con error de carga
    """
    rutas   = df_ok["ruta_archivo"].tolist()
    dataset = ImageDataset(rutas, transform)

    loader  = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_filtrar_nulos,
    )

    dev           = next(model.parameters()).device
    todos_emb     = []
    indices_ok    = []
    indices_error = []

    log.info(f"Extrayendo embeddings CLIP de {len(df_ok):,} imágenes "
             f"(batch_size={cfg['batch_size']}, device={dev}) …")

    with torch.no_grad():
        for tensores, idxs_ok, idxs_err in tqdm(loader, desc="Lotes procesados", unit="lote"):
            indices_error.extend(idxs_err)

            if tensores.numel() == 0:
                continue

            tensores = tensores.to(dev)

            # TODO: reemplazar por model.encode_image(tensores) + normalización L2.
            salida = model(tensores)

            emb_np = salida.cpu().float().numpy()

            todos_emb.append(emb_np)
            indices_ok.extend(idxs_ok)

    if not todos_emb:
        log.error("ERROR: Ningun embedding pudo extraerse.")
        sys.exit(1)

    embeddings = np.vstack(todos_emb)
    log.info(f"Embeddings extraídos: {len(indices_ok):,} exitosos, "
             f"{len(indices_error):,} errores.")
    return embeddings, indices_ok, indices_error


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 5: SCORING ZERO-SHOT CONTRA PROMPTS DE TEXTO
# ──────────────────────────────────────────────────────────────────────────────
# TODO: implementar cuando se defina CONFIG["prompts"].
#   1. _codificar_prompts(model, tokenizer, prompts, device) → vectores de texto
#      normalizados L2, uno por prompt.
#   2. _similitud_prompts(embeddings_img, vectores_texto) → DataFrame con una
#      columna "clip_score_<nombre>" por prompt (producto punto, ya que ambos
#      lados están normalizados L2).

def _codificar_prompts(model, tokenizer, prompts: dict, device) -> dict:
    """
    Codifica cada concepto de CONFIG["prompts"] a un vector CLIP normalizado L2.

    Cada concepto trae varias variaciones de redacción en prompts[nombre]["templates"]
    (ensemble de 2-3 templates, práctica recomendada en el paper de CLIP para
    reducir sensibilidad a la redacción exacta). Se codifica cada template por
    separado, se promedian sus vectores (antes de normalizar) y el promedio se
    normaliza L2 una sola vez — así el score final es un solo número por concepto,
    no uno por template.

    TODO implementación:
        for nombre, spec in prompts.items():
            tokens = tokenizer(spec["templates"]).to(device)
            emb = model.encode_text(tokens)                  # (n_templates, D)
            emb = emb.mean(dim=0)                             # promedio simple
            emb = emb / emb.norm()                            # normaliza L2
            vectores_texto[nombre] = emb.cpu().numpy()

    Retorna: {nombre_concepto: vector_texto (D,) np.float32}
    """
    raise NotImplementedError("Módulo 5 pendiente: implementar codificación + ensemble.")


def _similitud_prompts(embeddings_img: np.ndarray, vectores_texto: dict) -> pd.DataFrame:
    """
    Calcula la similitud coseno (producto punto, embeddings ya normalizados)
    entre cada imagen y cada prompt de texto.

    Retorna: DataFrame (N, n_prompts) con columnas "clip_score_<nombre>".
    """
    raise NotImplementedError("Módulo 5 pendiente: definir prompts en CONFIG primero.")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 6: CONSTRUCCIÓN DEL OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def _generar_image_id(nombre_archivo: str) -> str:
    """
    Genera un identificador único para la imagen basado en el nombre del archivo.
    SHA-256 del nombre de archivo, truncado a 16 caracteres hex (~64 bits).
    Determinístico: el mismo archivo siempre produce el mismo image_id.
    """
    return hashlib.sha256(nombre_archivo.encode()).hexdigest()[:16]


def construir_output(
    df_ok:         pd.DataFrame,
    embeddings:    np.ndarray,
    indices_ok:    list,
    embedding_dim: int,
    df_scores:     pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Construye el DataFrame de output con una fila por imagen exitosa.

    Columnas: image_id, consecutivo, ola, pano_id, heading, nombre_archivo,
    llave, llave_n16 (si disponibles), modelo, embedding_dim, embedding,
    y clip_score_* (si df_scores fue provisto).
    """
    df_proc = df_ok.iloc[indices_ok].copy().reset_index(drop=True)

    df_proc["image_id"] = df_proc["nombre_archivo"].apply(_generar_image_id)
    df_proc["modelo"]        = "clip"
    df_proc["embedding_dim"] = embedding_dim
    df_proc["embedding"] = [row.astype(np.float32).tolist() for row in embeddings]

    if df_scores is not None:
        df_proc = pd.concat(
            [df_proc.reset_index(drop=True), df_scores.reset_index(drop=True)],
            axis=1,
        )

    cols_base = [
        "image_id", "consecutivo", "ola", "pano_id", "heading",
        "nombre_archivo", "modelo", "embedding_dim", "embedding",
    ]
    cols_extra = [c for c in ["llave", "llave_n16"] if c in df_proc.columns]
    cols_scores = [c for c in df_proc.columns if c.startswith("clip_score_")]

    idx_ola  = cols_base.index("ola") + 1
    cols_out = cols_base[:idx_ola] + cols_extra + cols_base[idx_ola:] + cols_scores

    return df_proc[[c for c in cols_out if c in df_proc.columns]]


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 7: EXPORTACIÓN A PARQUET
# ──────────────────────────────────────────────────────────────────────────────

def exportar_parquet(df_emb: pd.DataFrame, cfg: dict) -> Path:
    """
    Exporta el DataFrame de embeddings + scores a formato Parquet.
    Archivo fijo: embeddings_clip.parquet (un solo modelo en este pipeline).
    """
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ruta_parquet = out_dir / "embeddings_clip.parquet"
    df_emb.to_parquet(ruta_parquet, engine="pyarrow", index=False)
    log.info(f"Embeddings exportados: {ruta_parquet}")
    return ruta_parquet


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 8: REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def generar_reporte(
    df_emb:       pd.DataFrame,
    n_total:      int,
    n_errores:    int,
    t_inicio:     float,
    cfg:          dict,
) -> str:
    """
    Genera un reporte de texto plano con estadísticas del proceso de extracción
    y, si hay prompts configurados, un resumen descriptivo de los clip_score_*.
    """
    t_total  = time.time() - t_inicio
    n_proc   = len(df_emb)
    t_prom   = t_total / n_proc if n_proc > 0 else 0

    lineas = [
        "=" * 72,
        "REPORTE DE EXTRACCIÓN DE EMBEDDINGS CLIP",
        "=" * 72,
        "",
        "── 1. Parámetros de ejecución ─────────────────────────────────────────",
        f"  Variante CLIP      : {cfg['variante_clip']}",
        f"  Pretrained         : {cfg['pretrained_clip']}",
        f"  Dimensión embedding: {df_emb['embedding_dim'].iloc[0] if n_proc > 0 else 'N/A'}",
        f"  Batch size         : {cfg['batch_size']}",
        f"  Device             : {cfg['device']}",
        f"  Prompts configurados: {len(cfg['prompts'])}",
        "",
        "── 2. Resumen de imágenes procesadas ──────────────────────────────────",
        f"  Total programadas  : {n_total:>8,}",
        f"  Exitosas           : {n_proc:>8,}",
        f"  Con error de carga : {n_errores:>8,}",
        f"  Tasa de éxito      : {n_proc / n_total * 100:.1f}%" if n_total > 0 else "",
        "",
        "── 3. Tiempos y rendimiento ────────────────────────────────────────────",
        f"  Tiempo total       : {t_total / 60:.1f} minutos",
        f"  Tiempo por imagen  : {t_prom:.3f} segundos",
        "",
    ]

    cols_scores = [c for c in df_emb.columns if c.startswith("clip_score_")]
    if cols_scores and n_proc > 0:
        lineas += ["── 4. Estadísticas de clip_score_* ────────────────────────────────────"]
        for c in cols_scores:
            lineas.append(
                f"  {c:<40} media={df_emb[c].mean():.3f}  std={df_emb[c].std():.3f}"
            )
        lineas.append("")

    lineas += [
        "── 5. Notas metodológicas ──────────────────────────────────────────────",
        "  · Los embeddings fueron extraídos del encoder de imagen de CLIP,",
        "    sin entrenamiento ni fine-tuning adicional.",
        "  · Los embeddings están normalizados L2 (convención estándar de CLIP).",
        "  · Los clip_score_* son similitud coseno entre el embedding de la",
        "    imagen y el embedding de texto del prompt correspondiente (zero-shot).",
        "  · Para convertir embedding a numpy: np.array(df['embedding'].iloc[0]).",
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
    log.info("  03b_extraer_embeddings_clip.py")
    log.info(f"  Variante CLIP: {CONFIG['variante_clip']} ({CONFIG['pretrained_clip']})")
    log.info("  ESTADO: esqueleto — Módulos 2 y 5 pendientes de implementar.")
    log.info("=" * 62)

    # ── 1. Carga las imágenes disponibles ─────────────────────────────────────
    df_ok = cargar_imagenes(CONFIG)
    n_total = len(df_ok)

    # ── 2. Carga el modelo CLIP ────────────────────────────────────────────────
    model, tokenizer, transform, embedding_dim, device = cargar_modelo_clip(CONFIG)

    # ── 3. Extrae embeddings de imagen en lotes ───────────────────────────────
    embeddings, indices_ok, indices_error = extraer_embeddings(
        model, transform, df_ok, CONFIG
    )

    # ── 4. Scoring zero-shot contra prompts (si hay prompts configurados) ────
    df_scores = None
    if CONFIG["prompts"]:
        vectores_texto = _codificar_prompts(model, tokenizer, CONFIG["prompts"], device)
        df_scores = _similitud_prompts(embeddings, vectores_texto)

    # ── 5. Construye el DataFrame de output ───────────────────────────────────
    df_emb = construir_output(
        df_ok, embeddings, indices_ok,
        embedding_dim=embedding_dim,
        df_scores=df_scores,
    )

    # ── 6. Exporta a Parquet ──────────────────────────────────────────────────
    ruta_parquet = exportar_parquet(df_emb, CONFIG)

    # ── 7. Genera y guarda el reporte ─────────────────────────────────────────
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    reporte = generar_reporte(
        df_emb,
        n_total=n_total,
        n_errores=len(indices_error),
        t_inicio=t_inicio,
        cfg=CONFIG,
    )

    ruta_reporte = out_dir / "reporte_embeddings_clip.txt"
    ruta_reporte.write_text(reporte, encoding="utf-8")

    log.info(f"Reporte guardado: {ruta_reporte}")
    log.info(f"Proceso completado en {(time.time() - t_inicio) / 60:.1f} minutos.")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
