# Pipeline GSV–ELCA

Pipeline para medir cobertura de Google Street View (GSV) a nivel de hogar en las tres olas de la **Encuesta Longitudinal Colombiana de la Universidad de los Andes (ELCA)** — 2010, 2013 y 2016 — y extraer representaciones visuales del entorno mediante modelos de visión por computadora preentrenados.

---

## Tabla de contenidos

1. [Contexto](#1-contexto)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Requisitos e instalación](#3-requisitos-e-instalación)
4. [Flujo del pipeline](#4-flujo-del-pipeline)
5. [Scripts en detalle](#5-scripts-en-detalle)
   - [00 — Panel de coordenadas](#00--panel-de-coordenadas)
   - [01 — Análisis de cobertura GSV](#01--análisis-de-cobertura-gsv)
   - [02 — Descarga de imágenes](#02--descarga-de-imágenes)
   - [03 — Extracción de embeddings](#03--extracción-de-embeddings)
   - [04 — Variables visuales (pendiente)](#04--variables-visuales-pendiente)
   - [05 — Modelos econométricos (pendiente)](#05--modelos-econométricos-pendiente)
6. [Estructura de directorios de datos](#6-estructura-de-directorios-de-datos)
7. [Esquema de outputs](#7-esquema-de-outputs)
8. [Notas metodológicas](#8-notas-metodológicas)
9. [Solución de problemas frecuentes](#9-solución-de-problemas-frecuentes)

---

## 1. Contexto

### La ELCA

La ELCA es un panel longitudinal de hogares colombianos levantado por la Universidad de los Andes en tres olas: **2010, 2013 y 2016**. El diseño del panel permite seguir a los mismos hogares a lo largo del tiempo, incluyendo los que se dividieron entre olas (hogares *split*).

Cada hogar está identificado por una jerarquía de llaves:

| Ola | Identificador | Estructura |
|-----|--------------|------------|
| 2010 | `consecutivo` | 6 dígitos, único por hogar base |
| 2013 | `llave` | `consecutivo` + `hogar` (zero-pad 2 dígitos) |
| 2016 | `llave_n16` | `llave` + 2 dígitos adicionales |

Un hogar con `hogar = 1` en 2013 no se dividió desde 2010. Un hogar con `hogar > 1` es un sub-hogar producto de una división: antes de separarse, todos sus miembros vivían en la dirección del hogar padre, por lo que las coordenadas del padre son la referencia geográfica válida para la ola anterior.

### Google Street View

GSV provee imágenes esféricas a nivel de calle en la mayoría del territorio colombiano urbano y en algunos corredores rurales. Este pipeline consulta la **GSV Metadata API** (gratuita, no descarga imágenes) para verificar si existe un panorama cercano a cada hogar, y luego la **GSV Static API** (de pago) para descargar las imágenes confirmadas.

### Objetivo

Construir variables de entorno visual a nivel hogar × ola que puedan ser utilizadas como regresores en modelos econométricos sobre vulnerabilidad.

---

## 2. Estructura del proyecto

```
tesis_vulnerabilidad/
│
├── data/
│   ├── raw/
│   │   ├── elca_2010/          → RHogar-csv.tab, UHogar-csv.tab
│   │   ├── elca_2013/          → RHogar-csv.tab, UHogar-csv.tab
│   │   └── elca_2016/          → RHogar-csv.tab, UHogar-csv.tab
│   │
│   └── processed/
│       ├── coordenadas/        → outputs del script 00
│       ├── gsv/                → outputs de los scripts 01 y 02
│       │   └── fotos/
│       │       ├── ola_2010/
│       │       ├── ola_2013/
│       │       └── ola_2016/
│       ├── embeddings/         → outputs del script 03
│       └── variables_visuales/ → outputs del script 04 (pendiente)
│
└── src/
    └── 01_download/
        └── GoogleStreetView/   ← estás aquí
            ├── 00_construir_panel_coordenadas.py
            ├── 01_analisis_cobertura_gsv.py
            ├── 02_descarga_fotos_GSV.py
            ├── 03_extraer_embeddings.py
            ├── 04_construir_variables_visuales.py  (pendiente)
            ├── 05_modelos_econometricos.py          (pendiente)
            └── requirements.txt
```

---

## 3. Requisitos e instalación

### Python

Python 3.9 o superior. Verificar con:

```bash
python3 --version
```

### Dependencias

```bash
pip install -r requirements.txt
```

Dependencias por script:

| Paquete | Scripts que lo usan | Propósito |
|---------|-------------------|-----------|
| `pandas` | 00, 01, 02, 03 | Manipulación de DataFrames |
| `numpy` | 00, 03 | Operaciones numéricas y arrays |
| `requests` | 01 | Consultas HTTP a la GSV Metadata API |
| `tqdm` | 01, 02, 03 | Barras de progreso |
| `torch` | 03 | Inferencia con modelos de visión |
| `torchvision` | 03 | VGG19, ResNet50 preentrenados |
| `Pillow` | 03 | Carga de imágenes JPEG |
| `pyarrow` | 03 | Exportación a formato Parquet |

Para el script 03 se recomienda instalar PyTorch siguiendo las instrucciones oficiales en [pytorch.org](https://pytorch.org/get-started/locally/) según el sistema operativo y si se dispone de GPU.

### API key de Google Street View

Los scripts 01 y 02 requieren una clave de la **Google Maps Platform**. Para obtenerla:

1. Ir a [console.cloud.google.com](https://console.cloud.google.com)
2. Crear un proyecto y habilitar **Street View Static API**
3. Crear una clave de API en _Credenciales_

Configurar la clave como variable de entorno antes de correr los scripts (nunca escribirla directamente en el código ni subirla al repositorio):

```bash
export GSV_API_KEY="tu_clave_aqui"
```

Para que persista entre sesiones, agregar esa línea a `~/.zshrc` o `~/.bash_profile`.

---

## 4. Flujo del pipeline

Los scripts deben ejecutarse en orden. Cada uno consume el output del anterior y produce archivos que el siguiente espera encontrar.

```
datos ELCA (.tab)
        │
        ▼
┌─────────────────────────┐
│  00_construir_panel     │  panel_coordenadas.csv
│  _coordenadas.py        │──────────────────────────────────────┐
└─────────────────────────┘                                      │
        │                                                        │
        ▼                                                        │
┌─────────────────────────┐                                      │
│  01_analisis_cobertura  │  inventario_panos.csv                │
│  _gsv.py                │  panel_enriquecido_gsv.csv           │
│  [GSV Metadata API]     │──────────────────────────────────────┤
└─────────────────────────┘                                      │
        │                                                        │
        ▼                                                        │
┌─────────────────────────┐                                      │
│  02_descarga_fotos      │  fotos/ola_{ola}/*.jpg               │
│  _GSV.py                │  registro_descargas.csv              │
│  [GSV Static API]       │  resumen_panel_gsv.csv               │
└─────────────────────────┘                                      │
        │                                                        │
        ▼                                                        │
┌─────────────────────────┐                                      │
│  03_extraer_embeddings  │  embeddings_{modelo}.parquet         │
│  .py                    │                                      │
│  [VGG19/ResNet/Places]  │                                      │
└─────────────────────────┘                                      │
        │                                                        │
        ▼                                                        │
┌─────────────────────────┐                                      │
│  04_construir_variables │  variables_visuales_{modelo}.csv ◄───┘
│  _visuales.py           │  (hogar × ola, listo para modelos)
│  [pendiente]            │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  05_modelos             │  tablas de resultados
│  _econometricos.py      │
│  [pendiente]            │
└─────────────────────────┘
```

**Reproducibilidad:** cada etapa produce archivos persistentes. Si se modifica solo el script 03 (por ejemplo, para cambiar el modelo), no es necesario volver a correr los scripts 00, 01 y 02. Cada script verifica si sus inputs existen antes de comenzar y emite un mensaje de error claro si no.

---

## 5. Scripts en detalle

---

### 00 — Panel de coordenadas

**Archivo:** `00_construir_panel_coordenadas.py`
**Comando:** `python 00_construir_panel_coordenadas.py`

#### Qué hace

1. Carga los archivos `.tab` de hogares (rural + urbano) para cada ola.
2. Combina rural y urbano en un solo DataFrame por ola.
3. Extrae latitud y longitud decimales de las variables consolidadas de la ELCA.
4. Valida la calidad de las coordenadas:
   - Olas 2010 y 2013: usa la variable `coordenadas_obs` ya revisada por el equipo ELCA (0 = municipio correcto → válida).
   - Ola 2016: verifica pertenencia al bounding box de Colombia.
5. Construye el panel en formato largo (una fila por hogar × ola).
6. Calcula `cambio_residencia_ola`: indica si el hogar cambió de dirección entre olas, usando la distancia de Haversine entre coordenadas consecutivas.
7. Exporta el panel como CSV y genera un reporte de calidad en texto plano.

#### Inputs

| Archivo | Descripción |
|---------|-------------|
| `data/raw/elca_2010/RHogar-csv.tab` | Hogares rurales ola 2010 |
| `data/raw/elca_2010/UHogar-csv.tab` | Hogares urbanos ola 2010 |
| `data/raw/elca_2013/RHogar-csv.tab` | Hogares rurales ola 2013 |
| `data/raw/elca_2013/UHogar-csv.tab` | Hogares urbanos ola 2013 |
| `data/raw/elca_2016/RHogar-csv.tab` | Hogares rurales ola 2016 |
| `data/raw/elca_2016/UHogar-csv.tab` | Hogares urbanos ola 2016 |

#### Outputs

| Archivo | Descripción |
|---------|-------------|
| `data/processed/coordenadas/panel_coordenadas.csv` | Panel longitudinal (una fila por hogar × ola) |
| `data/processed/coordenadas/reporte_calidad_coordenadas.txt` | Resumen de validación por ola |

#### Parámetros configurables (`CONFIG`)

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `candidatos_lat` | ver CONFIG | Nombre de la variable de latitud decimal por ola |
| `candidatos_lon` | ver CONFIG | Nombre de la variable de longitud decimal por ola |
| `colombia_bbox` | lat [-4.2, 12.5], lon [-79.0, -66.8] | Bounding box para validar coordenadas 2016 |
| `tolerancia_residencia_m` | 50 m | Umbral para considerar que un hogar cambió de dirección |

> **Importante:** Para la ola 2016, los nombres de las variables de coordenadas (`coor_latitud` / `coor_longitud`) están asumidos por analogía con la ola 2013. Verificar los nombres reales en los archivos antes de correr el script y actualizar `CONFIG["candidatos_lat"][2016]` y `CONFIG["candidatos_lon"][2016]` si son distintos.

#### Columnas principales del output

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `consecutivo` | str | ID del hogar base (6 dígitos) |
| `hogar` | int | Número de sub-hogar dentro del consecutivo |
| `llave` | str | ID único en 2013 (`consecutivo` + `hogar`) |
| `llave_n16` | str | ID único en 2016 (`llave` + 2 dígitos) |
| `ola` | int | Año de la ola (2010, 2013, 2016) |
| `lat_decimal` | float | Latitud en grados decimales (positivo = norte) |
| `lon_decimal` | float | Longitud en grados decimales (negativo = oeste) |
| `coord_valida_ola` | int | 1 si la coordenada pasó validación, 0 si no |
| `cambio_residencia_ola` | float | Distancia en metros respecto a la ola anterior (NaN si es la primera ola o no hay coord válida) |
| `es_split` | int | 1 si el hogar es un sub-hogar producto de una división |

---

### 01 — Análisis de cobertura GSV

**Archivo:** `01_analisis_cobertura_gsv.py`
**Comando:** `export GSV_API_KEY="..."` y luego `python 01_analisis_cobertura_gsv.py`

#### Qué hace

1. Carga el panel de coordenadas y separa los hogares con coordenada válida (`coord_valida_ola == 1`) de los inválidos.
2. Deduplica por coordenada exacta: hogares distintos en la misma ubicación generan una sola consulta a la API.
3. Consulta la GSV Metadata API para cada combinación única de (lat, lon, radio). Los radios por defecto son 50 m, 100 m y 200 m.
4. Guarda los resultados en un archivo de caché (`resultados_api_cache.csv`) y puede reanudar si se interrumpe.
5. Construye el inventario de panoramas (hogar × ola × radio).
6. Enriquece el panel con variables resumen de cobertura GSV.
7. Genera un reporte descriptivo de cobertura.

#### Inputs

| Archivo | Descripción |
|---------|-------------|
| `data/processed/coordenadas/panel_coordenadas.csv` | Output del script 00 |

#### Outputs

| Archivo | Descripción |
|---------|-------------|
| `data/processed/gsv/inventario_panos.csv` | Una fila por hogar × ola × radio |
| `data/processed/gsv/panel_enriquecido_gsv.csv` | Panel con variables GSV agregadas |
| `data/processed/gsv/reporte_cobertura_gsv.txt` | Estadísticas descriptivas de cobertura |
| `data/processed/gsv/resultados_api_cache.csv` | Caché interno de respuestas de la API |

#### Parámetros configurables (`CONFIG`)

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `radios_m` | [50, 100, 200] | Radios de búsqueda en metros |
| `source` | `"outdoor"` | Fuente de panoramas (`"outdoor"` excluye interiores) |
| `pausa_s` | 0.05 | Segundos de pausa entre requests a la API |

#### Variables GSV añadidas al panel enriquecido

| Variable | Descripción |
|----------|-------------|
| `gsv_n_panos_50m` | 1 si existe un panorama dentro de 50 m, 0 si no |
| `gsv_n_panos_100m` | 1 si existe un panorama dentro de 100 m, 0 si no |
| `gsv_n_panos_200m` | 1 si existe un panorama dentro de 200 m, 0 si no |
| `gsv_tiene_pano` | 1 si existe un panorama en cualquiera de los radios |
| `gsv_dist_cercano_m` | Distancia en metros al panorama más cercano encontrado |
| `gsv_pano_id_cercano` | ID del panorama más cercano |
| `gsv_fecha_cercano` | Fecha del panorama más cercano |
| `gsv_heading` | Ángulo azimutal del panorama hacia el hogar (usado en script 02) |
| `gsv_status` | `"OK"`, `"ZERO_RESULTS"`, `"ERROR_*"` o `"SIN_COORD_VALIDA"` |

> **Limitación de la API:** La Metadata API retorna **un solo panorama** por consulta — el más cercano dentro del radio dado. Las variables `gsv_n_panos_{R}m` son **indicadores de presencia (0/1)**, no recuentos reales de panoramas únicos. Para obtener recuentos reales se requeriría la Maps JavaScript API o muestreo en grilla, fuera del alcance de este pipeline.

#### Costos de la API

La GSV Metadata API es **gratuita** (no cobra por consultas de metadatos). Solo la descarga de imágenes (script 02) genera costo. Ver [precios de la plataforma](https://developers.google.com/maps/documentation/streetview/usage-and-billing).

---

### 02 — Descarga de imágenes

**Archivo:** `02_descarga_fotos_GSV.py`
**Comando:** `export GSV_API_KEY="..."` y luego `python 02_descarga_fotos_GSV.py`

#### Qué hace

1. Lee el inventario de panoramas construido por el script 01.
2. Filtra los panoramas con `status == "OK"` y deduplica por (consecutivo, ola, pano_id).
3. Construye un plan de descarga: **2 imágenes por panorama**, una en dirección al hogar (`heading`) y otra en dirección opuesta (`heading + 180°`).
4. Verifica cuáles imágenes ya existen en disco (son JPEG válidos ≥ 10 KB) y las omite.
5. Descarga en paralelo con `ThreadPoolExecutor`.
6. Guarda el registro de cada intento en `registro_descargas.csv`.
7. Genera un resumen por hogar × ola y un reporte de texto.

#### Inputs

| Archivo | Descripción |
|---------|-------------|
| `data/processed/gsv/inventario_panos.csv` | Output del script 01 |

#### Outputs

| Archivo/Directorio | Descripción |
|-------------------|-------------|
| `data/processed/gsv/fotos/ola_{ola}/` | Imágenes descargadas (JPEG 640×640) |
| `data/processed/gsv/registro_descargas.csv` | Log de cada intento de descarga |
| `data/processed/gsv/resumen_panel_gsv.csv` | Resumen por hogar × ola |
| `data/processed/gsv/reporte_descarga.txt` | Estadísticas del proceso |

#### Nomenclatura de archivos

Cada imagen se nombra:

```
{consecutivo}_{ola}_{pano_id}_{heading:03d}.jpg
```

Por ejemplo: `123456_2013_AbCdEfGh_047.jpg`

#### Parámetros configurables (`CONFIG`)

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `img_size` | `"640x640"` | Resolución de la imagen (máximo sin surcharge) |
| `fov` | 90 | Campo de visión en grados (50–120) |
| `pitch` | 0 | Ángulo vertical (0 = horizontal) |
| `max_workers` | 10 | Descargas simultáneas |
| `timeout_s` | 30 | Segundos máximos por imagen |
| `max_reintentos` | 3 | Intentos antes de registrar fallo |
| `reintentar_fallidas` | `True` | Si `False`, conserva fallos previos sin reintentar |

#### Columnas del registro de descargas

| Columna | Descripción |
|---------|-------------|
| `consecutivo` | ID del hogar |
| `ola` | Año de la ola |
| `pano_id` | ID del panorama |
| `heading` | Ángulo de la imagen en grados |
| `nombre_archivo` | Nombre del archivo JPEG |
| `ruta_archivo` | Ruta absoluta en disco |
| `exito` | `True` si la descarga fue exitosa |
| `ya_descargada` | `True` si el archivo ya existía antes de este run |
| `codigo_http` | Código de respuesta HTTP |
| `bytes` | Tamaño del archivo descargado |
| `mensaje_error` | Descripción del error, si aplica |

---

### 03 — Extracción de embeddings

**Archivo:** `03_extraer_embeddings.py`
**Comando:** `python 03_extraer_embeddings.py`

#### Qué hace

1. Lee el registro de descargas y filtra las imágenes con `exito == True`.
2. Incorpora identificadores adicionales del hogar (`llave`, `llave_n16`) desde el inventario.
3. Carga el modelo preentrenado seleccionado en `CONFIG["modelo"]`.
4. Extrae los embeddings en lotes mediante inferencia sin gradiente (`torch.no_grad()`).
5. Exporta los resultados en formato Parquet (una fila por imagen).

**Se corre una vez por modelo.** Produce un archivo Parquet independiente para cada uno:
- `embeddings_vgg19.parquet`
- `embeddings_resnet50.parquet`
- `embeddings_places365.parquet`

#### Inputs

| Archivo | Descripción |
|---------|-------------|
| `data/processed/gsv/registro_descargas.csv` | Output del script 02 |
| `data/processed/gsv/inventario_panos.csv` | Para obtener `llave` y `llave_n16` |

#### Outputs

| Archivo | Descripción |
|---------|-------------|
| `data/processed/embeddings/embeddings_{modelo}.parquet` | Embeddings (una fila por imagen) |
| `data/processed/embeddings/reporte_embeddings_{modelo}.txt` | Estadísticas del proceso |

#### Modelos disponibles

| Modelo | Parámetro | Embedding | Entrenado en |
|--------|-----------|-----------|-------------|
| VGG-19 | `"vgg19"` | 4 096 dim | ImageNet (1.2M imágenes, 1 000 clases) |
| ResNet-50 | `"resnet50"` | 2 048 dim | ImageNet (1.2M imágenes, 1 000 clases) |
| ResNet-50 Places365 | `"places365"` | 2 048 dim | Places365 (1.8M imágenes, 365 categorías de escenas) |

Places365 es particularmente relevante para el análisis de entorno urbano porque fue entrenado sobre fotografías de escenas (calles, plazas, parques, etc.) en lugar de objetos. Sus pesos se descargan automáticamente desde el MIT si no están en caché.

El embedding se extrae de la **penúltima capa** del modelo (antes de la capa de clasificación final), sin ningún entrenamiento ni fine-tuning adicional.

#### Parámetros configurables (`CONFIG`)

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `modelo` | `"vgg19"` | Modelo a usar: `"vgg19"`, `"resnet50"` o `"places365"` |
| `batch_size` | 32 | Imágenes por lote; reducir a 8 si hay errores de memoria |
| `device` | `"auto"` | Dispositivo: `"auto"`, `"cpu"`, `"cuda"` (NVIDIA) o `"mps"` (Apple Silicon) |

#### Esquema del Parquet de salida

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `image_id` | str | Identificador único de la imagen (SHA-256 del nombre de archivo, primeros 16 chars) |
| `consecutivo` | str | ID del hogar base |
| `ola` | int | Año de la ola |
| `llave` | str | ID de panel 2013/2016 (NaN para ola 2010) |
| `llave_n16` | str | ID de panel 2016 (NaN para olas 2010 y 2013) |
| `pano_id` | str | ID del panorama (trazabilidad hacia la API) |
| `heading` | float | Ángulo de la imagen en grados |
| `nombre_archivo` | str | Nombre del archivo JPEG |
| `modelo` | str | Nombre del modelo usado |
| `embedding_dim` | int | Dimensión del vector (4096 o 2048) |
| `embedding` | list[float32] | Vector de embedding completo |

Para convertir el embedding a numpy al leer el Parquet:

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("embeddings_vgg19.parquet")
vec = np.array(df["embedding"].iloc[0])   # shape: (4096,)
```

---

### 04 — Variables visuales (pendiente)

**Archivo:** `04_construir_variables_visuales.py`

Procesará los embeddings extraídos por el script 03 y construirá variables visuales agregadas a nivel hogar × ola. El output será una tabla con una fila por hogar × ola que pueda unirse directamente con el panel ELCA para la etapa de modelos.

---

### 05 — Modelos econométricos (pendiente)

**Archivo:** `05_modelos_econometricos.py`

Estimará los modelos econométricos que relacionan las variables visuales del entorno con los outcomes de interés (indicadores de vulnerabilidad del hogar). Incluirá comparación de especificaciones y evaluación predictiva.

---

## 6. Estructura de directorios de datos

Después de correr los scripts 00–03, la carpeta `data/` tendrá esta estructura:

```
data/
├── raw/
│   ├── elca_2010/
│   │   ├── RHogar-csv.tab
│   │   └── UHogar-csv.tab
│   ├── elca_2013/
│   │   ├── RHogar-csv.tab
│   │   └── UHogar-csv.tab
│   └── elca_2016/
│       ├── RHogar-csv.tab
│       └── UHogar-csv.tab
│
└── processed/
    ├── coordenadas/
    │   ├── panel_coordenadas.csv          ← script 00
    │   └── reporte_calidad_coordenadas.txt
    │
    ├── gsv/
    │   ├── inventario_panos.csv           ← script 01
    │   ├── panel_enriquecido_gsv.csv      ← script 01
    │   ├── reporte_cobertura_gsv.txt      ← script 01
    │   ├── resultados_api_cache.csv       ← script 01 (caché interna)
    │   ├── registro_descargas.csv         ← script 02
    │   ├── resumen_panel_gsv.csv          ← script 02
    │   ├── reporte_descarga.txt           ← script 02
    │   └── fotos/
    │       ├── ola_2010/
    │       │   └── {consecutivo}_2010_{pano_id}_{heading}.jpg
    │       ├── ola_2013/
    │       │   └── {consecutivo}_2013_{pano_id}_{heading}.jpg
    │       └── ola_2016/
    │           └── {consecutivo}_2016_{pano_id}_{heading}.jpg
    │
    └── embeddings/
        ├── embeddings_vgg19.parquet       ← script 03
        ├── embeddings_resnet50.parquet    ← script 03
        ├── embeddings_places365.parquet   ← script 03
        ├── reporte_embeddings_vgg19.txt
        ├── reporte_embeddings_resnet50.txt
        └── reporte_embeddings_places365.txt
```

---

## 7. Esquema de outputs

### `panel_coordenadas.csv` (script 00)

Una fila por hogar × ola. Columnas completas:

```
consecutivo, hogar, llave, llave_n16, ola,
zona, zona_2010, zona_2016, region, RegionLb, region_2016, dpto, mpio,
t_hogar, t_personas,
lat_decimal, lon_decimal,
coord_valida_ola, cambio_residencia_ola, es_split
```

### `inventario_panos.csv` (script 01)

Una fila por hogar × ola × radio. Columnas principales:

```
consecutivo, llave, llave_n16, ola, radio_m,
lat_decimal, lon_decimal,
pano_id, lat_pano, lon_pano, fecha, heading, distancia_m,
status
```

### `registro_descargas.csv` (script 02)

Una fila por intento de descarga (imagen):

```
consecutivo, ola, pano_id, radio_m, distancia_m, fecha_pano,
lat_pano, lon_pano, heading, pitch, fov, size,
nombre_archivo, ruta_archivo,
exito, ya_descargada, omitida, timestamp, codigo_http, bytes, mensaje_error
```

### `embeddings_{modelo}.parquet` (script 03)

Una fila por imagen procesada:

```
image_id, consecutivo, ola, llave, llave_n16,
pano_id, heading, nombre_archivo,
modelo, embedding_dim, embedding
```

---

## 8. Notas metodológicas

### Convención de signos en coordenadas

Colombia se ubica principalmente al **norte del ecuador** (latitudes positivas) y al **oeste del meridiano de Greenwich** (longitudes negativas). Las variables consolidadas de la ELCA ya incluyen el signo correcto; el pipeline las usa directamente sin transformación.

Bounding box de validación para la ola 2016:
- Latitud: −4.2° a 12.5° (norte)
- Longitud: −79.0° a −66.8° (oeste)

### Hogares divididos (*split households*)

Cuando un hogar se divide entre olas, los sub-hogares resultantes comparten el mismo `consecutivo` pero tienen distinto `hogar` (y por ende distinta `llave`). Antes de la separación, todos los miembros vivían en la misma dirección que el hogar padre. Por esto, el script 00 usa las coordenadas del padre en la ola anterior como referencia para calcular `cambio_residencia_ola` de los sub-hogares, lo cual es metodológicamente correcto.

### Limitación de la GSV Metadata API

La Metadata API devuelve **exactamente un panorama** por consulta — el más cercano a la coordenada dada dentro del radio especificado. Esto implica:

- Las variables `gsv_n_panos_{R}m` son **indicadores de presencia** (0 o 1), no recuentos.
- Un hogar puede aparecer con el mismo `pano_id` para radios de 50 m, 100 m y 200 m si ese es el panorama más cercano en todos los casos.
- La ausencia de panorama dentro de un radio no garantiza que no existan panoramas a mayor distancia.

### Deduplicación en el inventario

Si varios hogares tienen exactamente las mismas coordenadas decimales, se realiza una sola consulta a la API y el resultado se replica para todos esos hogares. Esto reduce costos sin perder información.

### Transfer learning para embeddings

Los modelos se cargan con sus pesos preentrenados y se usa únicamente la etapa de extracción de características (sin clasificador final). No se realiza ningún entrenamiento, fine-tuning ni ajuste de parámetros. Los embeddings capturan representaciones de alto nivel aprendidas por el modelo durante su entrenamiento original:

- **VGG19 / ResNet50 (ImageNet):** representaciones de objetos y texturas generales.
- **ResNet50 (Places365):** representaciones de escenas y entornos, más apropiadas para análisis de entorno urbano.

---

## 9. Solución de problemas frecuentes

### `FileNotFoundError: panel_coordenadas.csv`

El script 01 no encuentra el output del script 00. Correr primero `python 00_construir_panel_coordenadas.py`.

### `KeyError: 'GSV_API_KEY'` o clave vacía

La variable de entorno no está configurada. Ejecutar:

```bash
export GSV_API_KEY="tu_clave_aqui"
```

y verificar con `echo $GSV_API_KEY`.

### HTTP 403 en descarga de imágenes

La clave de API no tiene habilitada la **Street View Static API** en Google Cloud Console, o la clave tiene restricciones de IP/referrer. Verificar en el panel de la API.

### Imágenes de error (imagen gris de Google)

GSV retorna una imagen gris placeholder cuando el `pano_id` es inválido o ya no existe. El script 02 valida que el archivo sea JPEG real (≥ 10 KB y magic bytes `\xff\xd8`); los placeholders suelen pesar menos y son descartados como fallo.

### `OutOfMemoryError` en script 03

Reducir `CONFIG["batch_size"]` a 8 o 4. En CPU sin GPU disponible, el procesamiento es más lento pero no hay restricción de memoria de GPU.

### El script 03 no encuentra `torch` o `torchvision`

Instalar PyTorch siguiendo las instrucciones oficiales para el sistema operativo:

```bash
# CPU solamente
pip install torch torchvision

# macOS con Apple Silicon (MPS)
pip install torch torchvision

# NVIDIA GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Reanudación después de interrupción

- **Script 01:** guarda el caché de la API en `resultados_api_cache.csv`. Al reiniciar, detecta qué consultas ya están en el caché y omite las duplicadas.
- **Script 02:** verifica si cada imagen ya existe en disco antes de descargar. Si existe y es JPEG válido, la marca como `ya_descargada = True` y continúa.
- **Script 03:** no tiene estado persistente entre corridas; vuelve a procesar todas las imágenes. En caso de interrupción, simplemente volver a correr.
