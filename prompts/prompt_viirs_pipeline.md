# Prompt maestro: Pipeline de extracción de variables satelitales VIIRS (luces nocturnas)

## Rol y contexto

Actúa como un equipo interdisciplinario de expertos: un investigador en **economía del desarrollo** (medición de pobreza y actividad económica con datos satelitales), un especialista en **economía espacial** (escalas de agregación, efectos de vecindario, spillovers), un experto en **teledetección** (sensores DNB/VIIRS, corrección atmosférica, ruido de sensor) y un ingeniero de **machine learning** aplicado a ciencias sociales computacionales, con experiencia en software científico reproducible.

Estoy desarrollando mi tesis de maestría sobre predicción de vulnerabilidad a la pobreza monetaria usando machine learning. Las fuentes de datos son la ELCA/ELCO (Universidad de los Andes, rondas 2010, 2013, 2016, 2019, 2022), complementadas con imágenes públicas georreferenciadas. Ya tengo pipelines funcionales para Google Street View, Sentinel-1 y Sentinel-2. Ahora necesito uno equivalente para **VIIRS DNB (luces nocturnas)**.

## Restricción crítica: el código se ejecuta sin acceso a IA

Idéntica a la de los pipelines anteriores:
1. Cada script debe ser completamente autocontenido y funcional.
2. Documentación exhaustiva: docstrings, comentarios que expliquen el *porqué*, mensajes de log informativos.
3. Mensajes de error diagnósticos (qué se buscaba, dónde, por qué pudo fallar, cómo solucionarlo con comandos concretos).
4. CONFIG centralizado y perfectamente documentado.

## Patrón de código obligatorio

**Exactamente el mismo patrón que `sentinel1_pipeline` y `sentinel2_pipeline`**: estructura de script (docstring QUÉ HACE / INPUTS / OUTPUTS / CÓMO CORRER), separadores de 78 caracteres, CONFIG con `_HERE = Path(__file__).resolve().parent`, logging en vez de print, type hints, funciones reutilizables en `utils/`, reporte `.txt` con estadísticas descriptivas al final de cada script. Ver `prompts/prompt_sentinel1_pipeline.md` para el patrón completo — no se repite aquí.

---

## Fundamento teórico: por qué luces nocturnas para predecir pobreza

### La variable y su lógica de medición

VIIRS DNB (Day/Night Band) mide **radiancia nocturna** captada por el satélite Suomi-NPP: la luz artificial emitida hacia el cielo por alumbrado público, vivienda, comercio e industria. No es un indicador directo de pobreza — es un **proxy de actividad económica y provisión de infraestructura**, con un mecanismo causal específico que hay que documentar explícitamente en la tesis (y sus límites).

**Mecanismo causal (por qué luz nocturna ≈ actividad económica):**
1. La electrificación es un bien complementario al desarrollo económico: hogares y negocios con mayor ingreso consumen más electricidad (iluminación, electrodomésticos, actividad comercial nocturna).
2. La inversión pública en infraestructura (alumbrado vial, alumbrado público) se correlaciona con la capacidad fiscal del municipio, que a su vez se correlaciona con su base económica.
3. La actividad económica nocturna (comercio, industria con turnos) es en sí misma un componente del PIB local.

**Evidencia empírica clave que debe citarse en la tesis:**
- Henderson, Storeygard & Weil (2012, *AER*) — "Measuring Economic Growth from Outer Space": valida luces nocturnas como proxy de crecimiento del PIB subnacional, especialmente donde las estadísticas oficiales son débiles o inexistentes.
- Chen & Nordhaus (2011, *PNAS*) — evalúa dónde el proxy es más/menos informativo que las estadísticas convencionales.
- Jean, Burke, Lobell, Xie, Ermon, Lobell (2016, *Science*) — combina imágenes satelitales diurnas + nocturnas con ML para predecir pobreza a nivel de cluster en África; referencia metodológica directa para el enfoque de esta tesis.
- Elvidge et al. (varios años) — trabajo fundacional de NOAA sobre luces nocturnas y desarrollo económico, y las limitaciones técnicas del sensor (ver abajo).

### Qué NO mide (limitaciones que deben quedar en la tesis)

1. **Insensibilidad a la pobreza rural dispersa.** En zonas rurales de baja densidad, la ausencia de luz nocturna es indistinguible entre "zona sin electrificar por pobreza" y "zona sin electrificar por baja densidad poblacional" (p. ej. una finca próspera pero aislada). El proxy funciona mejor en gradientes urbano-rurales que en la cola inferior de la distribución (zonas ya oscuras, "bottom-coding").
2. **Blooming / overglow.** La luz se dispersa atmosféricamente y "sangra" varios cientos de metros más allá de su fuente real, especialmente en zonas con alta humedad o aerosoles. Esto infla artificialmente la luminosidad medida en el entorno inmediato de fuentes intensas y sesga positivamente zonas cercanas a un centro urbano brillante aunque sean pobres.
3. **Top-coding y saturación en núcleos urbanos densos.** En el centro de ciudades grandes el sensor puede saturarse, comprimiendo la variación real de actividad económica en esas zonas (poca capacidad discriminante en el extremo alto).
4. **Fuentes de luz no económicas.** Quemas agrícolas, incendios forestales, luz lunar residual (mitigada por el producto mensual pero no eliminada del todo), y quema de gas (gas flaring, relevante en zonas petroleras de Colombia como el Meta/Casanare) pueden inflar la radiancia sin relación con bienestar del hogar.
5. **Sesgo hacia lo público/colectivo, no lo privado del hogar.** La luz nocturna capta infraestructura y actividad del entorno (alumbrado público, comercios vecinos), no necesariamente el consumo eléctrico del hogar encuestado. Es una variable de **contexto espacial**, no una característica individual del hogar — hay que ser explícitos sobre este nivel de agregación en la interpretación.
6. **Nubosidad y sombra topográfica.** Colombia tiene alta nubosidad en buena parte del territorio (Andes, Pacífico); el compuesto mensual filtra píxeles nublados pero zonas con nubosidad persistente pueden tener pocas observaciones válidas, afectando la fiabilidad del promedio.

**Dónde podría inducir sesgos:** al usar luces nocturnas como predictor de pobreza, el modelo puede sistemáticamente sub-predecir vulnerabilidad en hogares rurales bien electrificados con baja densidad de vecinos (falso positivo de "no pobre" por brillo bajo pero explicado por dispersión geográfica, no por pobreza) y sobre-predecir bienestar en hogares pobres ubicados cerca de un eje vial iluminado o de zonas de flaring. Este sesgo debe discutirse en la sección de limitaciones metodológicas de la tesis, y —si el tiempo lo permite— contrastarse con un análisis de sensibilidad excluyendo hogares cerca de pozos petroleros conocidos.

---

## Escala espacial: por qué 500 m y no 250 m

Sentinel-1/2 usan un buffer único de 250 m justificado por su resolución nativa de 10 m y el error de georreferenciación de las encuestas (50–200 m). VIIRS DNB tiene una resolución nativa de **~15 arco-segundos (~450–500 m en el ecuador)**, es decir, un solo píxel VIIRS ya cubre un área comparable al buffer completo usado para Sentinel. Mantener 250 m subestimaría el área real de sensado del instrumento (el buffer caería dentro de un único píxel remuestreado, sin aportar información adicional) e ignoraría el efecto de blooming descrito arriba.

**Decisión: buffer circular único de 500 m**, alineado con la resolución nativa del sensor.

- **Ventajas de un buffer circular:** simetría (no favorece ninguna dirección de dispersión de la luz), consistencia con la convención ya usada en S1/S2, cómputo simple en GEE (`geometry.buffer()`).
- **Desventajas:** no captura si la fuente dominante de luz está en una dirección específica (p. ej. una vía principal a 400 m al norte vs una zona oscura al sur se promedian).
- **Por qué no un buffer cuadrado:** no aporta ventaja interpretativa aquí (no estamos alineando con una grilla catastral) y complica el cálculo geodésico sin beneficio.
- **Por qué no una métrica de distancia de red (red vial):** exigiría datos de red vial de alta calidad y cobertura nacional consistente en las 5 rondas, que no están garantizados; el beneficio marginal es bajo para una variable que ya es de por sí de resolución gruesa (500 m). Se documenta como línea de trabajo futura, no como parte de este pipeline.

**Nota de diseño — anillo auxiliar para el índice de posición relativa:** el módulo de indicadores derivados (ver abajo) calcula un índice de brillo relativo del hogar frente a su entorno inmediato más amplio. Esto requiere, además del buffer principal de 500 m, una segunda geometría en forma de anillo (donut) entre 500 m y 2 km, calculada server-side en GEE sin depender de shapefiles externos (municipios, veredas). Esto **no cambia** la filosofía de "buffer único" para las variables núcleo — es un cálculo auxiliar interno de un solo indicador derivado, y se documenta así para que quede claro por qué aparece una segunda geometría en el código.

---

## Producto GEE y ventana temporal

### Colección

**`NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG`** — compuestos mensuales de radiancia promedio, enmascarados por nubes, **sin** corrección de stray-light. Bandas relevantes:
- `avg_rad`: radiancia promedio nocturna libre de nubes, en nW·cm⁻²·sr⁻¹.
- `cf_cvg`: número de observaciones libres de nubes usadas en el compuesto del mes (proxy de calidad/confiabilidad del dato, no de actividad económica).

**Decisión de producto verificada empíricamente contra GEE (no solo documentación) — hay dos variantes del compuesto mensual VIIRS DNB en el catálogo de NOAA, y difieren en cobertura temporal real:**

| Colección | Corrección stray-light | Cobertura real (confirmada contra GEE) |
|---|---|---|
| `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | Sí | **2014-01-01 en adelante** (148 imágenes al momento de verificar) |
| `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG` | No | **2012-04-01 en adelante** (169 imágenes al momento de verificar) |

La documentación general de NOAA sugiere que el producto corregido (VCMSLCFG) existe "desde 2012", pero al filtrar la colección real en GEE por fecha, **VCMSLCFG devuelve 0 imágenes para 2012-2013**. Este pipeline usa **VCMCFG para las 5 rondas** (un solo producto, sin mezclar algoritmos entre rondas), por las siguientes razones:

1. **Consistencia metodológica del panel.** Usar el mismo algoritmo de composición en las 5 rondas evita introducir un salto artificial en el nivel o la varianza de la variable entre rondas que sería indistinguible de un cambio económico real.
2. **Stray-light es un problema de latitud.** La contaminación que corrige VCMSLCFG proviene de luz solar/lunar dispersada durante el crepúsculo prolongado — un fenómeno estacional marcado en **latitudes altas** (crepúsculos largos en verano polar). Colombia está entre ~-4° y 13° de latitud (la ELCA se concentra en la zona andina, ~4°N): el terminador día/noche es abrupto y estable todo el año, por lo que el fenómeno que la corrección busca remover es, en principio, poco relevante para este territorio. **Esta es una justificación geográfica razonada, no una medición directa del sesgo residual** — debe citarse como supuesto explícito en la tesis, con la salvedad de que no se validó cuantitativamente cuánto stray-light residual queda en VCMCFG para Colombia específicamente.
3. **Sin esta decisión, no hay ventana disponible para 2010/2013 con el producto corregido.** La alternativa sería mover el proxy a 2014 (ver ventanas abajo — opción descartada por alejar aún más el proxy de la ronda real) o mezclar productos por ronda (descartado por el punto 1).

**Limitación que debe quedar explícita en la tesis:** las variables VIIRS de este pipeline pueden contener una pequeña sobreestimación de radiancia en meses/ubicaciones con contaminación real de stray-light no corregida. Se recomienda, si el tiempo de la tesis lo permite, un análisis de sensibilidad puntual: recalcular las rondas 2016/2019/2022 (donde SÍ hay traslape con VCMSLCFG) con ambos productos y comparar la magnitud de la diferencia, para dar una cota empírica al supuesto del punto 2 en vez de dejarlo solo como argumento teórico.

El script `00_inicializar_gee.py` verifica en tiempo de ejecución (no asume) que la colección responde y que las bandas esperadas existen — la disponibilidad exacta de variantes de esta colección en GEE puede cambiar y **debe confirmarse al correr el script**, no solo confiar en este documento. De hecho, así fue como se detectó la limitación de cobertura de VCMSLCFG descrita arriba.

### Ventanas temporales por ronda

Con VCMCFG (cobertura real desde abril de 2012), sí existe un primer año calendario completo antes de las rondas 2016/2019/2022: 2013. Esto reproduce el mismo tipo de limitación que Sentinel-1 con las rondas 2010/2013 (proxy compartido), pero ahora con datos reales disponibles para ese proxy.

```python
"ventanas_temporales": {
    # ── LIMITACIÓN IMPORTANTE ──────────────────────────────────────────────
    # VIIRS DNB (VCMCFG) no existe antes de abril 2012. Las rondas 2010 y
    # 2013 usan el mismo proxy: el primer año calendario con cobertura
    # mensual completa (2013). Igual que en Sentinel-1, esto implica que,
    # para hogares que no se mudaron, las filas 2010 y 2013 tendrán
    # valores IDÉNTICOS de VIIRS. No es un error. Documentar en la tesis y
    # repetir el análisis de sensibilidad excluyendo 2010/2013 ya definido
    # para S1.
    2010: ("2013-01-01", "2013-12-31"),  # proxy: primer año calendario completo
    2013: ("2013-01-01", "2013-12-31"),  # proxy: primer año calendario completo
    2016: ("2015-10-01", "2016-09-30"),  # año previo a la ronda (consistente con S1)
    2019: ("2018-10-01", "2019-09-30"),  # año previo a la ronda
    2022: ("2021-10-01", "2022-09-30"),  # año previo a la ronda
},
```

### Autenticación GEE

Idéntica a S1/S2: cuenta de servicio, JSON en `credenciales/`, verificación de existencia con instrucciones de remedio si falta.

### Decisión de arquitectura: la limpieza ocurre sobre la tabla ya extraída, no sobre la imagen

A diferencia de una primera intuición (limpiar a nivel de imagen GEE antes de extraer), se decidió que **01 solo filtra la colección por ventana temporal** (sin tocar los valores), **02 extrae los valores crudos** de `avg_rad`/`cf_cvg` por hogar × ronda × mes, y **03 aplica la limpieza sobre esa tabla numérica ya extraída**, antes de calcular los estadísticos temporales. Razones:
- Mantiene 02 (extracción) simple y genérico — un único responsable de "bajar datos de GEE a una tabla", reutilizable si más adelante cambian las reglas de limpieza sin tener que re-consultar GEE (costoso en tiempo de red).
- Permite iterar sobre el umbral de winsorización o el manejo de `cf_cvg` releyendo el Parquet intermedio de 02, sin volver a golpear la API de GEE.
- Es coherente con el principio ya usado en S1/S2 de minimizar llamadas a GEE y maximizar el trabajo reproducible en Python puro sobre datos ya descargados.

**Limpieza aplicada en 03 (sobre la tabla extraída):**
1. **Recorte de negativos**: `avg_rad = max(avg_rad, 0)`. Los valores negativos son un artefacto de la corrección de stray-light de NOAA (sustracción de fondo), no luz "negativa" real.
2. **Winsorización por techo físico fijo**: `avg_rad = min(avg_rad, CONFIG["techo_winsorizacion_nw"])`, con techo por defecto de **100 nW·cm⁻²·sr⁻¹** — muy por encima de lo que produce alumbrado urbano denso típico, pero característico de gas flaring o incendios activos (ver Elvidge et al. sobre gas flaring y VIIRS). El valor exacto queda documentado y es modificable en CONFIG.
3. **`cf_cvg` NO se usa para descartar meses.** Se decidió conservar todos los meses de la ventana (VIIRS, al ser un compuesto global mensual, siempre tiene un mes disponible por hogar — no hay "hogares sin escena" como en Sentinel). En su lugar, `cf_cvg` se reporta como variable de calidad (`viirs_cf_cvg_media`, `viirs_n_meses_validos`) en el control de calidad final (06), permitiendo decidir en la etapa de modelado si se excluyen filas con baja confiabilidad, en vez de perder datos irreversiblemente en la extracción.

```python
"techo_winsorizacion_nw": 100,  # nW/cm2/sr; techo físico para gas flaring/incendios, ver docs
```

---

## Variables a construir

### Bandas base (tras limpieza)

- `avg_rad`: radiancia promedio, con outliers extremos winsorizados (ver control de calidad) para mitigar gas flaring/incendios puntuales.
- `cf_cvg`: cobertura de observaciones libres de nubes — variable de calidad, no económica.

### Estadísticos temporales (sobre el promedio espacial del buffer de 500 m, para cada mes disponible en la ventana)

- Media, mediana, mínimo, máximo.
- Desviación estándar, coeficiente de variación (std/mean).
- Rango (max − min).
- Percentiles 10, 25, 75, 90.
- Número de meses con observación válida (`n_meses_validos`) y cobertura media (`cf_cvg` media) — variables de calidad del dato, no de contenido económico.
- Pendiente de una regresión lineal simple radiancia ~ tiempo dentro de la ventana (`viirs_tendencia`) — capta si el entorno se está iluminando (creciendo) o apagando (declinando) dentro del período, no solo su nivel promedio.

### Indicadores derivados (módulo 8: construcción de indicadores derivados)

**Decisión: NO se construye `viirs_indice_electrificacion`.** Binarizar avg_rad exige un umbral, y no hay uno defendible: un umbral "de la literatura" no existe de forma consolidada para VIIRS (a diferencia de DMSP-OLS, donde sí hay convenciones citadas como DN≥3), y un umbral derivado de la propia distribución del panel (percentil o valle bimodal) sería reproducible pero arbitrario y con riesgo de circularidad si se usa para predecir la misma pobreza que se busca explicar. Se deja `viirs_log_rad_media` como la variable de nivel a usar; la decisión de binarizar (y con qué criterio) queda diferida a la etapa de modelado, si se justifica ahí con un criterio explícito.

**Decisión: el crecimiento entre rondas se separa en DOS variables según si el hogar se mudó o no.** Calcular un solo "crecimiento" mezclando hogares que no se movieron con hogares que sí se mudaron confundiría dos fenómenos distintos: cambio económico real del lugar vs. cambio de lugar del hogar. Se usa la distancia entre las coordenadas del hogar en las dos rondas (fórmula de Haversine) con un umbral de 100 m (mismo orden de magnitud que el error de georreferenciación ya documentado para las encuestas) para clasificar "no se movió" vs. "se movió".

| Variable | Fórmula | Se calcula cuando... | Qué aproxima |
|---|---|---|---|
| `viirs_log_rad_media` | `log(viirs_rad_media + 1)` | Siempre | Transformación estándar en la literatura (Henderson et al.) para la distribución fuertemente sesgada de la radiancia. Variable de nivel recomendada para modelos lineales. |
| `viirs_brillo_relativo` | `viirs_rad_media(buffer 500 m) / viirs_rad_media(anillo 500 m–2 km)` | Siempre | Posición relativa del hogar frente a su entorno ampliado: >1 indica un núcleo más iluminado que su periferia; <1 sugiere periferia oscura de una zona más brillante (posible asentamiento periférico junto a una zona próspera). |
| `viirs_crecimiento_interanual` | `(rad_media_t − rad_media_t-1) / rad_media_t-1` | El hogar NO se movió entre rondas consecutivas (distancia < 100 m) | Proxy de crecimiento económico REAL del lugar, análogo a Henderson-Storeygard-Weil. Queda `NaN` si el hogar se mudó. |
| `viirs_cambio_nivel_migracion` | `rad_media_t − rad_media_t-1` (diferencia, no razón) | El hogar SÍ se movió entre rondas consecutivas (distancia ≥ 100 m) | Cambio en el nivel de luminosidad del entorno al mudarse — mejora o empeora el entorno, NO es "crecimiento del lugar". Se reporta como diferencia (no como razón) porque dividir por la radiancia del lugar de origen no tiene la misma interpretación cuando el numerador y el denominador son lugares distintos. Queda `NaN` si el hogar no se mudó. |
| `viirs_hogar_se_movio` | booleano | Siempre que exista el par de rondas | Bandera que indica cuál de las dos variables anteriores aplica a esa fila; permite filtrar o controlar por movilidad en el modelo. |

**Nota sobre qué pares de rondas se usan para crecimiento/migración:** se excluye el par (2010, 2013) porque ambas rondas comparan con la MISMA ventana proxy (2013-01 a 2013-12, ver sección de ventanas temporales) — cualquier "crecimiento" ahí sería cero por construcción para no-movidos, o reflejaría dos lugares observados en el mismo año para movidos, no un cambio real en el tiempo. Los pares usados son: (2013, 2016), (2016, 2019), (2019, 2022).

### Variables con interpretación directa (para narrativa de tesis)

| Variable | Interpretación | Justificación |
|---|---|---|
| `viirs_rad_media` / `viirs_log_rad_media` | **Nivel de actividad económica/electrificación del entorno** | Proxy validado en la literatura (Henderson et al. 2012, Jean et al. 2016). |
| `viirs_crecimiento_interanual` | **Dinamismo económico local entre rondas (solo hogares que no se movieron)** | Cambios en infraestructura/actividad económica se reflejan en cambios de radiancia del mismo lugar. |
| `viirs_cambio_nivel_migracion` | **Cambio en la calidad del entorno al mudarse (solo hogares que se movieron)** | Compara el entorno de destino contra el de origen; mide movilidad espacial, no crecimiento económico de un lugar fijo. |
| `viirs_brillo_relativo` | **Centralidad vs. periferia dentro de la zona de influencia local** | Discrimina núcleos consolidados de periferias menos servidas dentro del mismo entorno ampliado. |

### Variables de apoyo (poder predictivo para ML, interpretación limitada)

Percentiles (P10, P25, P75, P90), rango, coeficiente de variación, `viirs_tendencia`, `cf_cvg` media, `n_meses_validos`. Incluir como *features*; no interpretarlas individualmente en la tesis salvo `n_meses_validos`/`cf_cvg`, que deben reportarse como variables de calidad del dato en el control de calidad (zonas con pocas observaciones válidas = alta nubosidad persistente, mayormente Pacífico y Andes altos).

**Nota metodológica sobre texturas GLCM:** a diferencia de S1/S2, **no se calculan métricas GLCM para VIIRS**. GLCM mide heterogeneidad espacial de una imagen; con un buffer de 500 m sobre un sensor de ~500 m de resolución nativa, el buffer contiene 1–4 píxeles reales (el resto es remuestreo), por lo que cualquier "textura" calculada sería un artefacto de interpolación, no señal real. Esta es una diferencia deliberada frente a S1/S2 y debe explicarse así en la tesis si se pregunta por la asimetría entre pipelines.

---

## Estructura del output final: panel hogar × ronda

Mismo formato que S1/S2:

```
consecutivo | ola  | lat    | lon     | viirs_rad_media | viirs_log_rad_media | viirs_brillo_relativo | ...
H001        | 2010 | 4.123  | -74.456 | 2.1              | 1.13                 | 0.87                   | ...
H001        | 2013 | 4.123  | -74.456 | 2.1              | 1.13                 | 0.87                   | ...
H001        | 2016 | 4.125  | -74.460 | 3.4              | 1.48                 | 1.05                   | ...
```

**Convención de nombres**: prefijo `viirs_` para todas las columnas. Merge con el panel de encuesta por `["consecutivo", "ola"]`, `how="left"`.

---

## Arquitectura del pipeline

Los 11 pasos lógicos (configuración → carga de coordenadas → descarga → preprocesamiento → buffers → extracción → variables temporales → indicadores derivados → control de calidad → almacenamiento → reportes) se **agrupan en 7 scripts**, siguiendo exactamente la convención de `sentinel1_pipeline`/`sentinel2_pipeline` (donde "almacenamiento" y "generación de reportes" no son scripts aparte sino la salida y el reporte `.txt` de cada script correspondiente):

```
viirs_pipeline/
├── credenciales/                      ← JSON de service account (NO commitear)
├── data/
│   ├── raw/                           ← coordenadas ELCA/ELCO (reutiliza salida de 00_preparar_coordenadas de S1)
│   └── processed/                     ← outputs intermedios y finales (Parquet)
├── logs/
├── scripts/
│   ├── 00_inicializar_gee.py          ← Módulo 1 (configuración) + verificación de conexión y de la colección VIIRS
│   ├── 01_construir_coleccion_viirs.py← Módulos 2-3 (carga de coordenadas, filtro de la colección mensual por ventana temporal de cada ronda; diagnóstico de disponibilidad)
│   ├── 02_extraccion_buffer.py        ← Módulos 5-6 (creación de buffers circular 500m + anillo 500m-2km, extracción espacial server-side de valores CRUDOS avg_rad/cf_cvg por hogar×ronda×mes)
│   ├── 03_series_temporales.py        ← Módulos 4 y 7 (preprocesamiento aplicado sobre la tabla YA EXTRAÍDA: recorte de negativos y winsorización por techo físico fijo; luego estadísticos temporales: media, percentiles, CV, tendencia)
│   ├── 04_indicadores_derivados.py    ← Módulo 8 (log-transform, crecimiento interanual, índice de electrificación, brillo relativo)
│   ├── 05_unir_variables.py           ← Módulo 10 (almacenamiento: LEFT JOIN sobre el espinazo de TODOS los hogares×ola por ['id_ola','ola'], panel_final en Parquet — el merge con el panel de encuesta ELCA/ELCO vive fuera de este sub-pipeline, igual que en S1/S2)
│   └── 06_control_calidad.py          ← Módulo 9 + 11 (control de calidad y generación de reporte final)
├── utils/
│   ├── gee_utils.py                   ← reutilizado/adaptado de sentinel1_pipeline
│   ├── io_utils.py                    ← reutilizado de sentinel1_pipeline
│   └── viirs_utils.py                 ← funciones específicas de VIIRS (máscara, winsorización, buffers+anillo, estadísticos, indicadores)
├── requirements.txt
└── README.md
```

### Procesamiento por lotes y eficiencia

Mismas reglas que S1/S2: lotes configurables (por defecto 500 hogares), checkpointing/idempotencia, cómputo pesado server-side en GEE, solo se descargan tablas de resultados nunca ráster completo, Parquet como formato intermedio/final.

---

## Flujo de desarrollo

**No generar todos los scripts simultáneamente.** Para cada script: (1) objetivo del módulo, (2) justificación de decisiones metodológicas, (3) inputs/outputs exactos, (4) interacción con el resto del pipeline, (5) parámetros de CONFIG involucrados — y solo después, el código completo y funcional.

## Requisitos de calidad del código

Idénticos a S1/S2: Clean Code, SOLID donde aplique, manejo robusto de errores con mensajes diagnósticos, sin variables globales salvo CONFIG y log, funciones reutilizables en `utils/`, compatible Windows/macOS.

## Nivel esperado

Equivalente al de un investigador especializado en teledetección aplicada a economía del desarrollo. Priorizar: robustez > claridad > eficiencia > extensibilidad. El pipeline debe integrarse sin fricción al resto del proyecto (mismo formato de panel, mismas claves de merge `consecutivo`/`ola`) para eventualmente unir GSV + S1 + S2 + VIIRS en una sola matriz de *features*.
