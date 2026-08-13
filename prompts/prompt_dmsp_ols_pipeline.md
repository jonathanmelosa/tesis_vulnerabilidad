# Prompt maestro: Pipeline de extracción de variables satelitales DMSP-OLS

## Rol y contexto

Actúa como un investigador experto en teledetección, Google Earth Engine
(GEE), Python y economía del desarrollo con datos de luces nocturnas.

Ya existen pipelines funcionales para Google Street View, Sentinel-1,
Sentinel-2 y VIIRS DNB. VIIRS DNB solo tiene cobertura real desde abril de
2012, así que las rondas 2010 y 2013 de la ELCA usan una ventana proxy
compartida (primer año calendario completo, 2013) — ver
`prompts/prompt_viirs_pipeline.md`, sección "Ventanas temporales por
ronda". Este pipeline, **DMSP-OLS** (la generación de sensor de luces
nocturnas anterior a VIIRS, operada 1992-2014), existe específicamente
para resolver ese hueco con datos **reales**, no un proxy: DMSP-OLS sí
cubre 2010 y 2013 con composites anuales reales y distintos entre sí.

## Restricción crítica y patrón de código obligatorio

Idénticos a `prompts/prompt_sentinel1_pipeline.md` y
`prompts/prompt_viirs_pipeline.md` — no se repiten aquí. El código se
ejecuta sin acceso a IA en la sala de cómputo de la universidad: cada
script autocontenido, documentación exhaustiva, mensajes de error
diagnósticos, CONFIG centralizado.

## Alcance: por qué solo 2010 y 2013

`NOAA/DMSP-OLS/NIGHTTIME_LIGHTS` en el catálogo de Earth Engine cubre
**1992-2014** (verificado contra la documentación oficial del catálogo,
resolución nativa 927.67 m, bandas `avg_vis`/`stable_lights`/`cf_cvg`). No
tiene datos para 2016 en adelante. Este pipeline **no intenta cubrir**
2016/2019/2022 — `viirs_pipeline/` ya los cubre con datos reales
(ventanas ex-ante genuinas, sin proxy). Agregar DMSP-OLS ahí sería
redundante y, de hecho, imposible.

A diferencia de Sentinel-1/VIIRS, **2010 y 2013 NO comparten ventana**:
cada ronda usa su propio año calendario real (`anios_por_ronda = {2010:
2010, 2013: 2013}`). Esto significa que el crecimiento 2010→2013 SÍ es
información temporal real (a diferencia del mismo cálculo en
`viirs_pipeline/`, que excluye ese par por compartir ventana proxy) — es
el principal valor agregado de este pipeline.

## Fundamento teórico y limitaciones (deben quedar en la tesis)

Mismo mecanismo causal luces-nocturnas-como-proxy-de-actividad-económica
que VIIRS (Henderson, Storeygard & Weil 2012; ver
`prompts/prompt_viirs_pipeline.md`, sección "Fundamento teórico"), con
limitaciones ADICIONALES propias de la generación DMSP-OLS frente a
VIIRS:

1. **Cuantización de 6 bits (0-63) y saturación.** A diferencia de VIIRS
   (14 bits, sin saturación reportada), DMSP-OLS satura sobre núcleos
   urbanos densos: todos los píxeles muy brillantes quedan en 63,
   perdiendo capacidad discriminante justo donde más importaría (ver
   `dmsp_utils.py::marcar_saturacion`). Se reporta explícitamente el % de
   hogares saturados por ronda en el control de calidad final.
2. **Resolución más gruesa** (927.67 m vs. ~463-500 m de VIIRS) —
   refuerza la recomendación de NO usar esta variable como predictor
   fino a nivel de hogar individual, sino como característica de entorno
   más amplio (ver discusión de escala en el capítulo de fuentes
   geoespaciales de la tesis).
3. **Sin corrección on-board de calibración entre satélites** (a
   diferencia de VIIRS): distintos satélites (F10-F18) pueden tener
   sensibilidades ligeramente distintas. Este pipeline usa exclusivamente
   los años 2010 y 2013, ambos cubiertos por el satélite F18 según la
   documentación pública del producto — no se mezclan satélites distintos
   entre las dos rondas usadas, evitando el problema de raíz sin
   necesidad de una calibración cruzada explícita.
4. Blooming, insensibilidad a pobreza rural dispersa, fuentes de luz no
   económicas (flaring, incendios): mismas limitaciones ya documentadas
   para VIIRS, aplicables aquí sin cambios.

## Escala espacial

Buffer circular único de radio ≈ resolución nativa (927.67 m) — mismo
criterio ya usado en `viirs_pipeline/` (buffer ≈ resolución del sensor).
No se calcula anillo/brillo relativo (a diferencia de VIIRS): con una
resolución aún más gruesa, la justificación de comparar núcleo vs.
periferia en un radio corto es más débil, y añadir una segunda geometría
no aporta suficiente valor frente al costo de mantenerla — se documenta
esta asimetría frente a VIIRS explícitamente en el código (ver
`02_extraccion_buffer.py`).

## Adenda (ago-2026): estado vs. ventana acumulada

Rediseño posterior al primer borrador de este documento: cada ronda
(2010, 2013) ya no extrae un solo año, sino un año de **estado** (el año
exacto de la ola) más una **ventana acumulada** de 3 años que termina en
ese mismo año (2008-2010 para la ronda 2010; 2011-2013 para la ronda
2013). Motivación: un solo año describe el entorno en un instante, pero
no si ese entorno se venía iluminando o apagando en los años previos a la
ola — información potencialmente relevante para un modelo de *transición*
a la pobreza, que por definición mira un proceso, no solo un corte
transversal. `02_extraccion_buffer.py` ahora extrae los 3 años de la
ventana de cada ronda (el año de estado es simplemente el último);
`03_series_temporales.py` (nuevo) separa ambos tipos de variable;
`04_indicadores_derivados.py` opera sobre el estado (saturación, log,
crecimiento ENTRE olas). Ver el desglose completo, con ejemplos numéricos,
en la tesis, Sección 3.3.7. La arquitectura de 7 scripts (00-06) y las
convenciones de código no cambian.

## Variables construidas

`dmsp_avg_vis`, `dmsp_stable_lights` (variable de nivel recomendada),
`dmsp_cf_cvg` (calidad), `dmsp_saturado` (bandera), `dmsp_log_stable_lights`
(transformación estándar), `dmsp_crecimiento_2010_2013` /
`dmsp_cambio_nivel_migracion` / `dmsp_hogar_se_movio` /
`dmsp_distancia_movimiento_m` (mismo diseño no-se-movió-vs-se-movió que
VIIRS, aplicado al único par de rondas real que este pipeline cubre).

## Arquitectura

```
dmsp_ols_pipeline/
├── credenciales/
├── data/{raw,processed/checkpoints}/
├── logs/
├── scripts/
│   ├── 00_inicializar_gee.py
│   ├── 01_construir_coleccion_dmsp.py   ← diagnóstico, sin extracción masiva
│   ├── 02_extraccion_buffer.py          ← 1 imagen por ronda (no colección), sin agregación temporal
│   ├── 03_indicadores_derivados.py      ← saturación, log, crecimiento 2010→2013
│   ├── 04_unir_variables.py             ← LEFT JOIN por (id_ola, ola) sobre el espinazo de S1
│   └── 05_control_calidad.py
├── utils/{gee_utils.py, io_utils.py, dmsp_utils.py}
├── requirements.txt
└── LISTO_PARA_SALA.md
```

Nota: no hay script equivalente a
`viirs_pipeline/03_series_temporales.py` — DMSP-OLS es un compuesto
ANUAL (una imagen por año), no hay múltiples observaciones dentro de la
ronda que agregar temporalmente.

## Integración con el resto del proyecto

Mismas convenciones que S1/S2/VIIRS: llave `id_ola`/`ola`, prefijo de
columnas (`dmsp_`), coordenadas reutilizadas de
`sentinel1_pipeline/00_preparar_coordenadas.py`, merge final con el panel
ELCA fuera de este sub-pipeline. El panel final de "geoespacial 2010/2013"
de la tesis se construye combinando `variables_dmsp.parquet` (2010, 2013)
con `variables_viirs.parquet` (2016 en adelante) — ninguna ronda debería
tomar datos de ambas fuentes a la vez.
