"""
03d_extraer_scores_streetscore.py
==================================
Calcula puntajes de percepción urbana (Place Pulse 2.0 / StreetScore) para
cada foto de Google Street View descargada por 02_descarga_fotos_GSV.py,
usando el clasificador ViT preentrenado publicado en Hugging Face Hub.

RESPONSABILIDAD EXCLUSIVA
    Transformar un directorio de imágenes JPEG en una tabla de puntajes de
    percepción urbana (una fila por imagen, una columna por dimensión).

    Este script NO realiza:
    · Descarga de imágenes (responsabilidad de script 02).
    · Extracción de embeddings visuales (responsabilidad de 03 y 03b —
      pipeline separado porque aquí no generamos un embedding propio: el
      clasificador ya expone directamente un score interpretable por
      dimensión, entrenado por regresión sobre comparaciones pareadas
      humanas de Place Pulse 2.0, no por un backbone de clasificación).
    · Entrenamiento, fine-tuning ni clasificación propia.
    · Selección de variables ni análisis estadístico.

MODELO
    Clasificador ViT-B-16 (Ouyang 2023, "human-perception-place-pulse")
    entrenado sobre Place Pulse 2.0 (110,988 imágenes de 56 ciudades,
    comparaciones pareadas de percepción humana). Pesos descargados
    automáticamente desde Hugging Face Hub
    (repo "Jiani11/human-perception-place-pulse") la primera vez que se
    corre cada dimensión; quedan en caché local en CONFIG["carpeta_modelos"].

    NOTA DE IMPLEMENTACIÓN — por qué este script NO depende del paquete
    zensvi (aunque el modelo es el mismo que usa zensvi.cv.ClassifierPerceptionViT):
    importar zensvi arrastra ~35 dependencias no relacionadas con este
    clasificador (open3d, faiss-cpu, geopandas, osmnx, rasterio,
    groundingdino-py, etc., usadas por otras clases del paquete como
    segmentación o detección de objetos). Varias de ellas solo tienen
    wheels precompiladas para versiones recientes de macOS/Linux y no
    instalan en Windows sin compilar desde código fuente — causa típica de
    que "pip install zensvi" falle o se cuelgue en un computador de sala.
    Este script reimplementa directamente las ~40 líneas que realmente se
    usan (arquitectura ViT-B-16 + descarga de checkpoint vía
    huggingface_hub) usando solo torch, torchvision, huggingface_hub, PIL
    y pandas — dependencias mucho más portables, ya requeridas por 03b.

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
                                   (se escanea recursivamente)

OUTPUTS
    scores_streetscore/scores_streetscore.parquet → puntajes (Output 1)
    scores_streetscore/reporte_scores_streetscore.txt → informe (Output 2)

FORMATO DE SALIDA
    Una fila por imagen (la intersección entre el registro de descargas y
    lo que el clasificador logró procesar). Columnas principales:
        image_id, consecutivo, ola, pano_id, heading, nombre_archivo,
        llave (si disponible), llave_n16 (si disponible),
        score_safety, score_lively, score_wealthy, score_beautiful,
        score_boring, score_depressing.

CÓMO CORRER
    python 03d_extraer_scores_streetscore.py

    No requiere instalar zensvi. Dependencias: torch, torchvision,
    huggingface_hub, pandas, pyarrow, Pillow (ver requirements.txt).
    Para procesar en GPU: modificar CONFIG["device"] = "cuda" o "mps".
    La primera corrida necesita internet para descargar los 6 checkpoints
    (uno por dimensión, ~350MB cada uno) desde Hugging Face Hub; después
    quedan cacheados en CONFIG["carpeta_modelos"] y ya no hace falta
    conexión.
"""

import os
import sys
import types
import time
import hashlib
import logging
from pathlib import Path

# Lee el token de Hugging Face desde un archivo local NO versionado (ver
# .gitignore) en vez de hardcodearlo en este script: el script sí está
# trackeado en git, y un token pegado literalmente aquí quedaría en el
# historial del repo aunque se borre después. Mismo patrón que 03e_extraer_
# segmentacion_sam3.py — reutiliza el mismo archivo .hf_token si ya existe.
_RUTA_TOKEN_HF = Path(__file__).resolve().parent / ".hf_token"


def _cargar_token_hf() -> None:
    """
    Si existe .hf_token junto a este script, lo carga en la variable de
    entorno HF_TOKEN — huggingface_hub la detecta automáticamente al
    descargar los checkpoints, sin necesidad de huggingface-cli login.
    Si no existe, no hace nada (las descargas siguen funcionando sin
    autenticar, solo con límites de tasa más bajos).
    """
    if _RUTA_TOKEN_HF.exists():
        token = _RUTA_TOKEN_HF.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token


_cargar_token_hf()

try:
    import pandas as pd
    import pyarrow as _pyarrow; del _pyarrow  # guard: falla aquí si falta, antes de clasificar
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
except ImportError as _err:
    # Si ves este error, el entorno virtual no está activo o falta una
    # dependencia. Solución paso a paso:
    #   1. Activa el entorno virtual:
    #        Mac/Linux : source .venv/bin/activate
    #        Windows   : .venv\Scripts\activate
    #   2. Instala las dependencias (torch, torchvision, huggingface_hub, etc.):
    #        pip install -r requirements.txt
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
    # Se escanea recursivamente (rglob), no hace falta listar subcarpetas
    # por ola a mano.
    "carpeta_fotos": _HERE / "gsv" / "fotos",

    # ── Dimensiones de Place Pulse 2.0 ─────────────────────────────────────────
    # Clave = valor exacto que espera el repo de Hugging Face como nombre de
    # estudio de percepción. Valor = nombre de columna en el output
    # ("score_" + valor).
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

    # ── Modelo ─────────────────────────────────────────────────────────────────
    # Carpeta donde se cachean los checkpoints descargados de Hugging Face
    # (uno por dimensión, ~350MB cada uno). Persiste entre corridas: si ya
    # están aquí, no se vuelven a descargar y el script funciona sin internet.
    "carpeta_modelos": _HERE / "modelos_streetscore",
    "hf_repo_id": "Jiani11/human-perception-place-pulse",

    # ── Salida ─────────────────────────────────────────────────────────────────
    "output_dir": _HERE / "scores_streetscore",

    # Si True, conserva también un results.csv/json crudo por dimensión
    # (útil para depurar), en output_dir/_raw/<dim>/.
    "guardar_resultados_crudos": False,
}

# Nombre del archivo de checkpoint en el repo de HF, por dimensión.
_ARCHIVOS_CHECKPOINT = {
    "safer":            "safety.pth",
    "livelier":         "lively.pth",
    "wealthy":           "wealthy.pth",
    "more beautiful":   "beautiful.pth",
    "more boring":      "boring.pth",
    "more depressing":  "depressing.pth",
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
        de archivo sin extensión, usado para unir con los resultados del
        clasificador, que identifica cada imagen por Path(archivo).stem).
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

    # Clave de unión con los resultados del clasificador: nombre sin extensión.
    df_ok["filename_key"] = df_ok["nombre_archivo"].apply(lambda f: Path(f).stem)

    return df_ok.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO 2: CLASIFICACIÓN POR DIMENSIÓN
# ──────────────────────────────────────────────────────────────────────────────
# Reimplementación directa (sin el paquete zensvi) del clasificador ViT-B-16
# de Ouyang (2023) — ver nota de implementación en el docstring del módulo.
# Arquitectura y checkpoints son EXACTAMENTE los mismos que usa
# zensvi.cv.ClassifierPerceptionViT (mismo repo de Hugging Face), solo cambia
# cómo se cargan.

class _NetPerceptionViT(nn.Module):
    """
    Arquitectura ViT-B-16 con cabeza de clasificación custom, usada para
    entrenar los checkpoints de "Jiani11/human-perception-place-pulse".

    Esta clase existe únicamente como "molde" para que torch.load() pueda
    reconstruir el objeto — cada checkpoint .pth fue guardado con
    torch.save(modelo_completo), no con un state_dict, así que al cargarlo
    se restauran directamente los pesos entrenados dentro de esta estructura
    (nunca se ejecuta __init__ durante la carga: no hace falta descargar los
    pesos base de ImageNet).
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        num_fc = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_fc, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return self._calcular_score(x)

    def _calcular_score(self, logits):
        probs = nn.Softmax(dim=1)(logits)[:, 1]
        return (probs * 10).round(decimals=2)


def _registrar_modulo_model_01() -> None:
    """
    Los checkpoints en Hugging Face fueron serializados con pickle apuntando
    a una clase "Model_01.Net" (nombre del script del repo original de
    Ouyang, https://github.com/strawmelon11/human-perception-place-pulse).
    torch.load(weights_only=False) necesita poder importar ese módulo para
    deserializar el objeto — como no instalamos ese paquete, registramos un
    módulo falso "Model_01" en sys.modules que expone nuestra propia clase
    (misma arquitectura, ver _NetPerceptionViT).
    """
    if "Model_01" in sys.modules:
        return
    modulo_falso = types.ModuleType("Model_01")
    modulo_falso.Net = _NetPerceptionViT
    sys.modules["Model_01"] = modulo_falso


class _ImagenesPerceptionDataset(Dataset):
    """
    Dataset de PyTorch que carga imágenes JPEG y aplica el preprocesamiento
    que espera el ViT-B-16 (resize a 384x384 + normalización ImageNet).
    """

    _TRANSFORM = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, rutas: list):
        self.rutas = rutas

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, idx: int):
        ruta = self.rutas[idx]
        img = Image.open(ruta).convert("RGB")
        return str(ruta), self._TRANSFORM(img)

    @staticmethod
    def collate_fn(batch):
        rutas, tensores = zip(*batch)
        return list(rutas), torch.stack(tensores)


def _resolver_device(device_cfg: str) -> torch.device:
    """
    Resuelve el dispositivo de cómputo a usar.
    "auto" selecciona CUDA si está disponible, luego MPS (Apple Silicon), luego CPU.
    """
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def _descargar_checkpoint(perception_study: str, cfg: dict) -> Path:
    """
    Descarga (o reutiliza del caché local) el checkpoint de una dimensión
    desde Hugging Face Hub. No requiere internet si ya está descargado.
    """
    archivo = _ARCHIVOS_CHECKPOINT[perception_study]
    carpeta_modelos = Path(cfg["carpeta_modelos"])
    snapshot_download(
        repo_id=cfg["hf_repo_id"],
        allow_patterns=[archivo, "README.md"],
        local_dir=carpeta_modelos,
    )
    return carpeta_modelos / archivo


def _cargar_clasificador(perception_study: str, cfg: dict, device: torch.device) -> nn.Module:
    """
    Descarga (si hace falta) y carga en memoria el modelo ViT de una
    dimensión de Place Pulse 2.0.
    """
    _registrar_modulo_model_01()
    checkpoint_path = _descargar_checkpoint(perception_study, cfg)
    modelo = torch.load(checkpoint_path, map_location=device, weights_only=False)
    modelo.eval()
    modelo.to(device)
    return modelo


def _clasificar_dimension(
    perception_study: str,
    nombre_columna:   str,
    rutas_fotos:      list,
    cfg:              dict,
    device:           torch.device,
) -> pd.DataFrame:
    """
    Corre el clasificador de una dimensión sobre todas las imágenes de
    rutas_fotos, en lotes.

    Retorna: DataFrame con columnas ["filename_key", "score_<nombre_columna>"].
    """
    modelo  = _cargar_clasificador(perception_study, cfg, device)
    dataset = _ImagenesPerceptionDataset(rutas_fotos)
    loader  = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=_ImagenesPerceptionDataset.collate_fn,
    )

    registros = []
    with torch.no_grad():
        for rutas_lote, tensores in tqdm(
            loader, desc=f"'{perception_study}' → score_{nombre_columna}", unit="lote"
        ):
            tensores = tensores.to(device, dtype=torch.float32)
            scores = modelo(tensores)
            for ruta, score in zip(rutas_lote, scores):
                registros.append({
                    "filename_key": Path(ruta).stem,
                    f"score_{nombre_columna}": score.item(),
                })

    del modelo   # libera el modelo ViT antes de cargar la siguiente dimensión

    if cfg["guardar_resultados_crudos"]:
        dir_crudo = Path(cfg["output_dir"]) / "_raw" / nombre_columna
        dir_crudo.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(registros).to_csv(dir_crudo / "results.csv", index=False)

    return pd.DataFrame(registros)


def ejecutar_clasificacion(cfg: dict) -> dict:
    """
    Corre, una a una, las 6 dimensiones de Place Pulse 2.0 sobre todas las
    imágenes en cfg["carpeta_fotos"].

    Se escanea la carpeta de forma recursiva y se clasifican TODAS las
    imágenes que se encuentren ahí — no se filtra por el registro de
    descargas. Por eso el resultado de este módulo se une con df_ok
    (Módulo 1) recién en el Módulo 3, para quedarnos solo con las imágenes
    que también están marcadas como exito=True.

    Cada dimensión carga su propio modelo (checkpoint distinto por
    perception_study); se libera de memoria antes de pasar a la siguiente
    para no acumular 6 modelos ViT cargados a la vez.

    Retorna:
        {nombre_columna: DataFrame con columnas ["filename_key", nombre_columna]}
    """
    device = _resolver_device(cfg["device"])
    log.info(f"Device de inferencia: {device}")

    carpeta_fotos = Path(cfg["carpeta_fotos"])
    rutas_fotos = [
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for p in carpeta_fotos.rglob(ext)
    ]
    log.info(f"Imágenes encontradas en {carpeta_fotos}: {len(rutas_fotos):,}")

    resultados = {}
    for perception_study, nombre_columna in cfg["dimensiones"].items():
        log.info(f"Clasificando dimensión '{perception_study}' → score_{nombre_columna} …")

        df_dim = _clasificar_dimension(
            perception_study, nombre_columna, rutas_fotos, cfg, device
        )
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
          TODAS las dimensiones — si el clasificador no logró procesar una
          imagen en alguna dimensión, esa imagen queda fuera del output final).
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
        f"  Modelo             : ViT-B-16 (Ouyang 2023, Place Pulse 2.0)",
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
        "    entrenado sobre comparaciones pareadas de Place Pulse 2.0.",
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
    log.info("    pip install -r requirements.txt")
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
