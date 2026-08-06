# Prompt maestro: Pipeline de extracción de variables satelitales Sentinel-1

## Rol y contexto

Actúa como un investigador experto en teledetección, Google Earth Engine (GEE), Python, SIG, econometría espacial y aprendizaje automático, con experiencia en el desarrollo de software científico reproducible.

Estoy desarrollando mi tesis de maestría sobre predicción de vulnerabilidad a la pobreza monetaria usando machine learning. Las fuentes de datos son la ELCA (Encuesta Longitudinal Colombiana de la Universidad de los Andes, rondas 2010, 2013, 2016) y la ELCO (rondas 2010, 2013, 2016, 2019, 2022), complementadas con imágenes públicas georreferenciadas. Ya tengo un pipeline funcional para Google Street View; ahora necesito construir uno equivalente para Sentinel-1.

## Restricción crítica: el código se ejecuta sin acceso a IA

El código se ejecutará en la sala de cómputo de la universidad, donde **no tengo acceso a IA ni a asistentes de código**. Por tanto:

1. **Cada script debe ser completamente autocontenido y funcional** — no puedo pedir correcciones en tiempo real.
2. **La documentación debe ser exhaustiva**: docstrings detallados, comentarios en línea que expliquen el *porqué* (no el *qué*), y mensajes de log informativos que me permitan diagnosticar problemas sin ayuda.
3. **Los mensajes de error deben ser diagnósticos**: cuando algo falle, el mensaje debe decirme exactamente qué ocurrió, por qué pudo ocurrir, y cómo solucionarlo — incluyendo comandos concretos.
4. **Cada CONFIG debe estar perfectamente documentado** con comentarios que me permitan modificar valores sin necesidad de entender todo el código.

## Patrón de código obligatorio

Todo el código debe seguir exactamente el patrón de mis scripts existentes de Google Street View. A continuación defino las convenciones que debes replicar:

### Estructura de cada script

```python
"""
NN_nombre_del_script.py
========================
Descripción de una línea.

QUÉ HACE
    Lista numerada de lo que hace el script.

    Este script NO realiza:
    · tarea X (responsabilidad de script NN).
    · tarea Y (responsabilidad de script MM).

INPUTS
    ruta/al/input.csv
    (output de NN_script_anterior.py)

OUTPUTS
    ruta/al/output1.csv    → descripción breve
    ruta/al/output2.txt    → descripción breve

CÓMO CORRER
    python NN_nombre_del_script.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import ...

# ── Librerías externas ─────────────────────────────────────────────────────────
import ...

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Único bloque que debe modificarse para adaptar el script.
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent

CONFIG = {
    # ── Sección 1 ──────────────────────────────────────────────────────────────
    # Comentario explicativo sobre esta sección.
    "clave": valor,  # comentario en línea sobre esta clave específica
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
# MÓDULO N: NOMBRE DEL MÓDULO
# ──────────────────────────────────────────────────────────────────────────────

def nombre_funcion(param: tipo, cfg: dict) -> tipo_retorno:
    """
    Docstring detallado.
    """
    ...

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Orquesta todos los módulos en orden y exporta los resultados."""
    log.info("=" * 62)
    log.info("  NN_nombre_del_script.py")
    log.info("  Si ves ModuleNotFoundError, el entorno virtual no esta")
    log.info("  activado o las dependencias no estan instaladas.")
    log.info("  Solucion:")
    log.info("    Mac/Linux : source .venv/bin/activate")
    log.info("    Windows   : .venv\\Scripts\\activate")
    log.info("    Luego     : pip install -r requirements.txt")
    log.info("=" * 62)
    ...

if __name__ == "__main__":
    main()
```

### Convenciones obligatorias

- **Separadores visuales**: usar `# ──────...` con ancho de 78 caracteres para separar secciones.
- **Comentarios en línea**: después de cada línea de código no trivial, explicar *por qué* se hace eso, no *qué* hace.
- **CONFIG centralizado**: todas las rutas, parámetros y opciones van en un diccionario CONFIG al inicio, con comentarios detallados.
- **Rutas con pathlib**: nunca usar strings para rutas. `_HERE = Path(__file__).resolve().parent` como base.
- **Logging, no print**: usar `log.info()`, `log.warning()`, `log.error()`. Solo `print()` para el reporte final.
- **Mensajes de error diagnósticos**: cuando un archivo no existe o un paso falla, el mensaje debe incluir: qué se buscaba, dónde se buscó, qué pudo salir mal, y cómo solucionarlo con comandos concretos.
- **Tipado de funciones**: todas las funciones deben tener type hints.
- **Sin notebooks**: todo son scripts .py ejecutables desde VS Code o terminal.
- **Idioma**: docstrings y comentarios en español. Nombres de variables y funciones en español (con underscore, sin tildes ni eñes en nombres de variables).
- **Reporte de calidad**: cada script genera un archivo .txt con estadísticas descriptivas del proceso, con secciones numeradas, explicación de la lógica y la interpretación.
- **Formato de reporte**: secciones con `── N. TÍTULO ──────...`, subsecciones con explicaciones QUÉ MIDE, LÓGICA, INTERPRETACIÓN.

## Especificaciones del pipeline Sentinel-1

### Objetivo general

Construir un pipeline que:
1. Lea una base de datos de hogares ELCA/ELCO con coordenadas geográficas (latitud, longitud).
2. Se conecte a Google Earth Engine mediante cuenta de servicio (JSON).
3. Descargue y procese imágenes Sentinel-1 GRD para cada hogar.
4. Genere variables derivadas de Sentinel-1 alineadas temporalmente con cada ronda de encuesta.
5. Exporte una base final en formato panel (una fila por hogar × ronda) en Parquet, lista para ML.

El pipeline debe ser completamente modular para permitir posteriormente incorporar otros satélites (Sentinel-2, Landsat, VIIRS, MODIS, etc.) sin modificar la arquitectura principal.

### Input

Los archivos de la ELCA contienen variables de latitud y longitud por hogar y ola. El pipeline debe:
- Leer directamente los archivos crudos de la ELCA (.tab o .csv) que contienen las variables de coordenadas.
- Detectar automáticamente las variables de coordenadas (mismo enfoque del script 00 de GSV).
- Menos de 10,000 hogares con coordenadas válidas en total.

### Estructura del output final: panel hogar × ronda

El output final es un **panel longitudinal**: una fila por hogar × ronda de encuesta. Si un hogar aparece en las 5 rondas, tendrá 5 filas.

```
consecutivo | ola  | lat    | lon     | s1_vv_vh_ratio_media | s1_cv_temporal_vv | s1_contraste_glcm | ...
H001        | 2010 | 4.123  | -74.456 | 1.82                 | 0.15              | 34.2              | ...
H001        | 2013 | 4.123  | -74.456 | 1.82                 | 0.15              | 34.2              | ...
H001        | 2016 | 4.125  | -74.460 | 1.79                 | 0.18              | 35.1              | ...
H001        | 2019 | 4.125  | -74.460 | 2.11                 | 0.12              | 28.7              | ...
H001        | 2022 | 4.125  | -74.460 | 2.34                 | 0.09              | 22.3              | ...
```

**Convención de nombres de columnas**: todas las variables Sentinel-1 llevan prefijo `s1_`. Esto facilita identificarlas al unir con variables de encuesta y de Google Street View.

**Lo que varía entre filas del mismo hogar**:
- Las coordenadas (si el hogar se mudó entre rondas).
- La ventana temporal de Sentinel-1 (cada ronda consulta imágenes de un período diferente).
- Ambos factores hacen que las variables cambien legítimamente entre rondas.

**Merge con el panel de la encuesta**:
```python
panel_final = panel_encuesta.merge(variables_sentinel1, on=["consecutivo", "ola"], how="left")
```
Hogares sin coordenadas válidas o sin cobertura Sentinel-1 quedan con NaN en las columnas satelitales.

### Ventanas temporales por ronda

Sentinel-1 está disponible desde octubre de 2014. El pipeline debe definir ventanas temporales de captura alineadas con cada ronda de encuesta. Configurar en CONFIG:

```python
"ventanas_temporales": {
    # ── LIMITACIÓN IMPORTANTE ──────────────────────────────────────────────
    # Sentinel-1 no existía antes de octubre 2014.
    # Las rondas 2010 y 2013 usan la misma ventana proxy: el primer año
    # completo de operación de S1. Esto significa que para hogares que no
    # se mudaron, las filas de 2010 y 2013 tendrán valores IDÉNTICOS de
    # Sentinel-1. No es un error — es una limitación del dato.
    # Documentar en la tesis y hacer análisis de sensibilidad excluyendo
    # 2010/2013 para verificar que los resultados no dependen de los proxies.
    2010: ("2014-10-01", "2015-09-30"),  # proxy: primer año completo de S1
    2013: ("2014-10-01", "2015-09-30"),  # proxy: primer año completo de S1
    2016: ("2015-10-01", "2016-09-30"),  # año previo a la ronda
    2019: ("2018-10-01", "2019-09-30"),  # año previo a la ronda
    2022: ("2021-10-01", "2022-09-30"),  # año previo a la ronda
},
```

El pipeline debe generar las 5 filas por hogar (cuando aplique) para mantener la estructura uniforme del panel. La decisión de excluir rondas proxy se toma en la etapa de modelado, no en la extracción.

### Autenticación GEE

Usar cuenta de servicio:
```python
"gee_service_account": "nombre@proyecto.iam.gserviceaccount.com",
"gee_key_file": _HERE / "credenciales" / "gee_service_account.json",
```

El script debe verificar que el archivo JSON existe y dar instrucciones claras si no.

### Variables a construir

#### Bandas base
- VV (polarización vertical-vertical, en dB)
- VH (polarización vertical-horizontal, en dB)

#### Variables derivadas de las bandas
- VV − VH (diferencia)
- VV / VH (razón, ratio)
- VH / VV (razón inversa)
- (VV − VH) / (VV + VH) (índice normalizado, análogo a NDVI)
- log(VV / VH) (razón logarítmica)

#### Estadísticos temporales
Para cada banda y variable derivada, sobre todas las imágenes disponibles en la ventana temporal:
- Media, mediana, mínimo, máximo
- Desviación estándar
- Rango (max − min)
- Coeficiente de variación (std / mean)
- Percentiles 10, 25, 75, 90
- Número de observaciones disponibles

#### Extracción espacial: buffer único de 250 metros

Usar un único buffer de 250 metros alrededor de cada hogar. Justificación:
- Las coordenadas de encuestas de hogares tienen errores de 50-200m; un buffer de 250m absorbe esa imprecisión.
- Sentinel-1 SAR tiene ruido de speckle a nivel de píxel individual (10m); promediar sobre 250m lo suaviza.
- El modelo predice vulnerabilidad a la pobreza, que opera a escala de vecindario, no de píxel.
- Con <10,000 hogares, minimizar el número de features reduce riesgo de sobreajuste.

Configurar en CONFIG como valor único modificable:
```python
"radio_buffer_m": 250,  # radio del buffer en metros; 250m recomendado
```

Todas las estadísticas temporales se calculan sobre el promedio espacial dentro del buffer.

#### Variables de textura (GLCM)
Calcular métricas GLCM sobre composiciones temporales (media temporal de VV y VH):
- Contraste, homogeneidad, energía, ASM
- Correlación, entropía, disimilitud
- Varianza

Usar `ee.Image.glcmTexture()` en GEE. Calcular sobre la composición temporal (media), no sobre imágenes individuales.

### Interpretabilidad de las variables

Las variables generadas tienen distintos niveles de interpretabilidad para la tesis. Documentar esto en los docstrings de cada variable:

#### Variables con interpretación directa (para la narrativa de la tesis)

| Variable | Interpretación | Justificación física |
|---|---|---|
| `s1_vv_vh_ratio_media` | **Grado de urbanización/densificación** del entorno | VV responde a estructuras verticales (edificios); VH a vegetación. La razón discrimina entre zona construida y vegetada. Validado en la literatura como proxy de built-up area. |
| `s1_vh_media` | **Densidad de vegetación** del entorno | VH responde al volumen de biomasa por dispersión cruzada en copas y ramas. Alto = vegetación densa; bajo = suelo desnudo, agua o concreto. |
| `s1_cv_temporal_vv` | **Estabilidad del entorno construido** | Infraestructura consolidada produce retrodispersión VV estable. CV alto = entorno cambiante (construcción nueva, demoliciones, inundaciones, asentamientos informales en expansión). |
| `s1_cv_temporal_vh` | **Estacionalidad agrícola** | VH varía con ciclos de siembra-cosecha. CV alto = zona agrícola activa con rotación. CV bajo = vegetación permanente o zona urbana. |
| `s1_n_observaciones` | **Accesibilidad/cobertura del territorio** | Zonas con pocas observaciones suelen ser áreas montañosas con sombra de radar o zonas muy remotas. Correlaciona con aislamiento geográfico. |

#### Variables con interpretación indirecta (argumentable con literatura)

| Variable | Interpretación |
|---|---|
| `s1_contraste_glcm_vv` | **Heterogeneidad del entorno construido.** Contraste alto = mezcla desordenada (informalidad urbana, bordes periurbanos). Bajo = zona uniforme. |
| `s1_homogeneidad_glcm_vv` | **Regularidad del paisaje.** Inverso del contraste. Alta = barrio consolidado o campo abierto. |
| `s1_rango_temporal_vv` | **Exposición a perturbaciones.** Rango alto = evento extremo durante el año (inundación, construcción abrupta, deforestación). |

#### Variables de apoyo (poder predictivo para ML, interpretación limitada)

Percentiles (P10, P25, P75, P90), entropía GLCM, energía/ASM GLCM, log(VV/VH), índice normalizado, varianza GLCM, correlación GLCM, disimilitud GLCM. Incluirlas como features; no intentar interpretarlas individualmente en la tesis.

### Arquitectura del pipeline

```
sentinel1_pipeline/
├── credenciales/              ← JSON de service account (NO commitear)
├── config/
│   └── config.py              ← toda la configuración centralizada
├── data/
│   ├── raw/                   ← archivos ELCA/ELCO crudos
│   └── processed/             ← outputs intermedios y finales (Parquet)
├── logs/                      ← archivos de log
├── scripts/
│   ├── 00_preparar_coordenadas.py
│   ├── 01_inicializar_gee.py
│   ├── 02_construir_coleccion_s1.py
│   ├── 03_composiciones_temporales.py
│   ├── 04_extraccion_buffer.py
│   ├── 05_metricas_textura.py
│   ├── 06_unir_variables.py
│   └── 07_control_calidad.py
├── utils/
│   ├── gee_utils.py           ← funciones reutilizables de GEE
│   ├── io_utils.py            ← funciones de lectura/escritura
│   └── stats_utils.py         ← funciones de cálculo estadístico
├── requirements.txt
└── README.md
```

Nota: se eliminaron los scripts separados de extracción por píxel y por múltiples buffers. Con un solo buffer de 250m, la extracción se simplifica a un único script (04_extraccion_buffer.py).

### Procesamiento por lotes

Aunque son <10,000 hogares, GEE tiene límites de tamaño por request. Implementar batch processing:
- Procesar en lotes de N hogares (configurable, por defecto 500).
- Cada lote genera un archivo intermedio.
- Si se interrumpe, al reiniciar retoma desde el último lote completado (idempotencia).
- Usar `ee.data.computeFeatures()` o `ee.FeatureCollection.getInfo()` según sea más eficiente.
- Priorizar procesamiento server-side en GEE; minimizar datos descargados.

### Escalabilidad y eficiencia

- Todo el cálculo pesado debe hacerse server-side en GEE (composiciones, estadísticas, GLCM).
- Solo descargar las tablas de resultados finales, nunca imágenes raster completas.
- Usar Parquet como formato de salida intermedio y final.
- Implementar checkpointing: guardar resultados intermedios para no reprocesar.
- Logging de progreso: `log.info(f"  Lote {i}/{n_lotes}: {n_hogares:,} hogares procesados")`.

## Flujo de desarrollo

**No generar todos los scripts simultáneamente.**

Trabajar módulo por módulo. Antes de escribir cada script:
1. Explicar el objetivo del módulo.
2. Justificar las decisiones metodológicas clave.
3. Describir las entradas y salidas exactas.
4. Explicar cómo interactúa con el resto del pipeline.
5. Señalar qué parámetros de CONFIG afectan este módulo.

Después de la explicación, escribir el código completo y funcional.

Cada script debe ser completamente funcional y probado conceptualmente antes de continuar con el siguiente.

## Requisitos de calidad del código

- Código limpio (Clean Code): funciones pequeñas, nombres descriptivos, sin duplicación.
- Principios SOLID cuando sean aplicables.
- Manejo robusto de errores con try/except específicos y mensajes diagnósticos.
- Sin variables globales excepto CONFIG y log.
- Toda función reutilizable en utils/.
- Cada script genera outputs independientes reutilizables sin re-ejecutar el pipeline completo.
- Compatible con Windows (sala de cómputo) y macOS (mi equipo personal).

## Nivel esperado

El código debe ser equivalente al desarrollado por un investigador especializado en teledetección aplicada a ciencias sociales computacionales. Priorizar: robustez > claridad > eficiencia > extensibilidad.

El objetivo final es obtener un pipeline que pueda convertirse en un paquete de investigación reutilizable y ampliable para incorporar nuevas fuentes satelitales sin modificar la arquitectura general.
