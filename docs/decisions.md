# Analytical Decisions

## Research Question
Prediction of transitions into monetary poverty.

## Outcome Definition
Transition from non-poor to poor between survey rounds.

## Poverty Measure
Monetary poverty.

## Strategy
Train models on two rounds and evaluate on a third (out-of-sample).

## Log

Nota sobre fechas: las entradas de este log están ordenadas por la fecha en
que se tomó/implementó cada decisión, no por la fecha en que se escribió su
documentación. Para las auditorías retroactivas (agregadas el 2026-08-08 a
pedido explícito, cubriendo scripts que ya existían sin documentar), la
fecha usada es la fecha de creación real del script correspondiente
(verificada por metadata del sistema de archivos), no el día en que se
redactó el texto.

### 2026-03-18
- Initialized reproducible project structure
- Implemented API-based download of ELCA data
- Adopted pipeline-based workflow

### 2026-03-18 (cont.) — Scripts exploratorios iniciales de `02_build/`

Documentación retroactiva (agregada 2026-08-08). Cuatro utilidades de solo
lectura/exploración, creadas el mismo día del arranque del proyecto, que no
transforman ni persisten datos consumidos aguas abajo por el pipeline
oficial:

  - `audit_core_2010.py`: inventario de columnas (tipo, % missing, únicos,
    muestra de valores) para 4 archivos crudos de 2010. Usa la ruta
    `data/raw/elca_2010`, que **ya no existe** en el repo (los crudos se
    reorganizaron después a `data/interim/raw/`) -- el script fallaría si
    se ejecutara hoy tal cual. Sí dejó output persistido de esa ejecución
    original: `docs/variable_audit/elca_2010_core/*.csv`.
  - `inspect_columns_2010.py` / `inspect_columns_persons_2010.py` /
    `inspect_round.py`: utilidades de grep/inspección de columnas por
    palabra clave o listado genérico de archivo, sin output persistido
    (solo `print`). Las 3 usan la misma ruta obsoleta `data/raw/elca_*`
    que `audit_core_2010.py`. Ninguna contiene una decisión metodológica
    no trivial; son herramientas ad-hoc de exploración de un solo
    archivo/ola a la vez, que sirvieron para orientar qué columnas de
    ingreso/pobreza/gasto/pesos existían antes de escribir los scripts de
    consolidación.

### 2026-06-12 — Pipeline visual GSV: esqueleto inicial (variables
visuales y modelos econométricos)

Documentación retroactiva (agregada 2026-08-08). Primeros dos archivos
creados del pipeline paralelo de Google Street View
(`src/01_download/02_scr_GoogleStreetView/`), antes que cualquier otro
script de ese pipeline -- consistente con escribir primero el contrato de
inputs/outputs y después la implementación. Este pipeline es una fuente de
datos completamente independiente de la construcción de pobreza monetaria
ELCA: solo comparte con ella los `.tab` crudos de hogar
(`RHogar-csv.tab`/`UHogar-csv.tab`) como input eventual, no consume ningún
parquet de `04_features/` ni la clasificación de pobreza.

**`04_construir_variables_visuales.py` y `05_modelos_econometricos.py` --
AMBOS SON ESQUELETOS VACÍOS**, solo docstring + `# TODO: implementar`, sin
una sola línea de código funcional (así siguen a la fecha de esta
auditoría). El primero declara la responsabilidad de agregar de nivel
imagen a nivel hogar×ola (input: `embeddings_{modelo}.parquet` de
`03_extraer_embeddings.py` únicamente -- el docstring no menciona
explícitamente CLIP/streetscore/SAM3 como inputs); el criterio de
agregación (promedio simple, ponderado, mediana) no existe todavía. El
segundo declara input/output pero no especifica variable dependiente,
especificación econométrica, ni ningún join contra las bases ELCA.

### 2026-06-17 — GSV: panel de coordenadas y extracción de embeddings
ImageNet/Places365

Documentación retroactiva (agregada 2026-08-08).

**Panel de coordenadas (`00_construir_panel_coordenadas.py`).**
  - Parte directamente de los `.tab` crudos de hogar (no de
    `hogar_elca_longitudinal_clean.parquet`) -- es un linaje de limpieza
    paralelo e independiente al del pipeline de pobreza.
  - **Fuente de coordenadas distinta por ola**: 2010/2013 usan variables
    decimales directas ya provistas por la ELCA (`coord_latitud`/
    `coord_longitud`); 2016 no las trae y se construyen desde componentes
    grados-minutos-segundos con `decimal = grados + minutos/60 +
    segundos/3600`. El signo de la longitud se fuerza siempre negativo
    (Colombia está íntegramente al oeste de Greenwich), sin depender del
    signo capturado en el dato crudo de 2016.
  - **Validación de calidad de coordenadas, distinta por ola y de
    confiabilidad desigual**: 2010/2013 usan `coordenadas_obs`, una
    variable de revisión manual ya incluida en la ELCA. 2016 no tiene esa
    variable; se usa en su lugar un bounding box del territorio colombiano
    (lat −4.2° a 12.5°, lon −79.0° a −66.8°) -- una validación mucho más
    débil, que solo descarta coordenadas fuera del país, no errores de
    captura dentro de Colombia. Solo los hogares que pasan esta validación
    (`coord_valida_ola=1`) generarán consultas a la API en el script
    siguiente -- el criterio de inclusión de todo el pipeline se decide
    aquí.
  - Hogares divididos (`es_split`, sub-hogar ≠ "1" en 2013/2016) se tratan
    como filas independientes en el panel largo, pero heredan la
    coordenada de referencia del hogar original del que se separaron.
  - **Cambio de residencia entre olas**: distancia Haversine entre la
    coordenada de la ola actual y la anterior, con **umbral de tolerancia
    de 50 metros** para considerar "mismo domicilio" -- un supuesto de
    ingeniería explícitamente no validado contra ningún catastro, elegido
    como "suficiente para distinguir domicilios distintos en áreas
    urbanas".
  - Outputs: `coordenadas/panel_coordenadas.csv` (panel largo hogar × ola)
    y un reporte de calidad puramente descriptivo (no filtra el CSV).

**Extracción de embeddings ImageNet/Places365 (`03_extraer_embeddings.py`).**
  - **Tres modelos preentrenados seleccionables, sin fine-tuning (transfer
    learning puro)**: VGG-19 (ImageNet, embedding 4096-d, capa
    clasificadora final eliminada -- el vector son las activaciones de la
    penúltima capa FC), ResNet-50 (ImageNet, 2048-d, capa `fc` reemplazada
    por identidad -- el vector es la salida del average pooling global),
    Places365 (ResNet-50 entrenada sobre escenas en vez de objetos,
    también 2048-d, checkpoint del MIT). Se documenta explícitamente por
    qué Places365 es "especialmente relevante para análisis de entorno
    urbano" frente a ImageNet estándar.
  - Preprocesamiento fijo común a los tres: resize 256 (lado corto) +
    center crop 224×224 + normalización estándar ImageNet, asumida válida
    también para Places365 "porque fue entrenado con el mismo pipeline".
  - Sin selección de vistas/ángulos ni agregación por hogar en este script
    -- procesa una fila por imagen; la agregación a nivel hogar×ola se
    delega explícitamente a `04_construir_variables_visuales.py`.
  - Solo imágenes con `exito==True` en el registro de descargas; imágenes
    corruptas se filtran y cuentan aparte sin detener el proceso.
  - `image_id` = SHA-256 del nombre de archivo (no del `pano_id` de
    Google), truncado a 16 caracteres hex.
  - Output en Parquet, elegido explícitamente por soportar columnas tipo
    lista de forma nativa (el embedding se guarda como lista de float32
    por fila) y compresión por columna.

### 2026-06-18 — GSV: análisis de cobertura y descarga de fotos

Documentación retroactiva (agregada 2026-08-08).

**Análisis de cobertura GSV (`01_analisis_cobertura_gsv.py`).**
  - **Limitación estructural de la API declarada explícitamente**: la
    Metadata API de Google Street View retorna un único panorama -- el más
    cercano -- por consulta, nunca el listado completo de panoramas en un
    área. Por eso se consulta con **4 radios independientes (50, 100, 200,
    400 m)**, y las variables resultantes `gsv_n_panos_{R}m` son
    indicadores de PRESENCIA (0/1: existe al menos un panorama dentro de R
    metros), NO un conteo real de panoramas únicos -- un conteo real
    requeriría la Maps JavaScript API o muestreo en grilla, declarado
    fuera de alcance.
  - Fuente de panoramas restringida a `"outdoor"` (excluye interiores y
    Google Maps Business), por ser más apropiado para captar el entorno
    construido del hogar.
  - Manejo de API: 10 workers en paralelo (~67 req/s vs. ~20 req/s
    secuencial, con nota de bajar a 5 si aparecen errores sostenidos de
    límite de tasa); deduplicación de consultas por coordenada+radio;
    reanudación vía caché en disco (solo se consideran "hechas" las
    consultas con status definitivo; errores transitorios se reintentan
    siempre).
  - **Dos reglas de unicidad de fotografías**: entre radios, un mismo
    panorama se asigna al radio más pequeño donde aparece; entre olas, un
    mismo panorama para el mismo hogar se asigna únicamente a la PRIMERA
    ola elegible según su año de captura, no se repite en olas
    posteriores. Ventanas de elegibilidad foto→ola: 2010 = año ≤2010,
    2013 = 2011-2013, 2016 = 2014-2016 -- el criterio explícito para
    conciliar la fecha real de captura de la imagen (que puede no
    coincidir con la fecha de la entrevista) con la ola de la encuesta.
  - **Panel balanceado**: subconjunto de hogares con coordenada válida
    presentes simultáneamente en las 3 olas, definido como denominador
    preferente de las estadísticas de cobertura y filtro recomendado por
    defecto para la descarga (script siguiente). Nota: este criterio de
    panel balanceado NO es el mismo universo que usan las matrices de
    transición de `build_pobreza_desagregaciones.py` (que excluyen
    hogares divididos, pero no exigen presencia en las 3 olas
    simultáneamente).
  - Dos escenarios de descarga definidos aquí: 1 imagen/hogar (vista
    frontal) o 2 imágenes/hogar (frontal + opuesta, heading+180°).
  - **Análisis de costos incluido en el reporte**: Metadata API gratuita;
    Static API (descarga real) = USD 7.00 por 1.000 requests; crédito
    mensual Google Maps Platform = USD 200 (~28.571 imágenes/mes gratis).
    El script verifica si el volumen total encontrado cabe en el crédito
    gratuito para ambos escenarios, y sugiere (sin implementarlo como
    filtro automático) priorizar por distancia al panorama más cercano si
    se supera el crédito.
  - Limitación explícita: la cobertura de imágenes no es homogénea en el
    tiempo (Google amplió su cobertura en Colombia gradualmente desde
    ~2012), lo que puede introducir sesgo de disponibilidad entre olas.

**Descarga de fotos (`02_descarga_fotos_GSV.py`).**
  - Usa directamente el `pano_id`/`heading` ya verificados por el script
    anterior -- no vuelve a consultar la Metadata API, lo que garantiza
    reproducibilidad (mismo `pano_id` produce siempre la misma imagen).
  - **Modo A (default, recomendado)**: descarga solo fotografías canónicas
    (`es_foto_unica=1`) de hogares en el panel balanceado
    (`en_panel_balanceado=1`). **Modo B (fallback)**: si el inventario no
    trae esas columnas o se desactiva explícitamente, descarga todos los
    panoramas con status OK, deduplicando por radio mínimo pero sin exigir
    panel balanceado -- incluye hogares presentes en solo 1 o 2 olas.
  - Parámetros fijos de la Static API: 640×640 px (máximo sin costo
    adicional), FOV 90°, pitch 0°.
  - **Idempotencia**: antes de descargar, verifica si el archivo ya existe
    en disco y pesa ≥10.000 bytes (umbral heurístico para distinguir un
    JPEG real de un placeholder/error) -- la existencia en disco es la
    fuente de verdad principal, el registro CSV previo es secundario.
    Validación adicional de magic bytes JPEG para descartar respuestas
    HTML/placeholders de panoramas eliminados o expirados.
  - Manejo de errores de API: reintentos hasta 3 con pausa de 2s; HTTP 429
    espera 10s antes de reintentar; HTTP 403 se diagnostica
    automáticamente en el reporte final (si >50% de los fallos son 403, se
    interpreta como problema de configuración de API, no de panoramas
    individuales expirados).
  - Nomenclatura de archivo `{consecutivo}_{ola}_{pano_id}_{heading:03d}.jpg`
    -- permite trazar cada imagen de vuelta al hogar/ola/panorama sin
    depender de metadatos externos.

### 2026-06-27 — Consolidación ELCA: gastos, personas, choques, activos
rurales, comunidades y niños

Documentación retroactiva (agregada 2026-08-08). Seis módulos de
`src/01_download/01_descarga_ELCA/` creados el mismo día. Solo dos
(gastos y personas) alimentan la construcción final de ingreso/gasto/
pobreza de `04_features/`; los otros cuatro (choques, activos rurales,
comunidades, niños) no la alimentan hoy, pero se documentan con el mismo
nivel de detalle (ver nota de relevancia al final de esta sección).

**Consolidación de gastos (`02_consolidacion_bases_gastos.py`).**
  - **Indicador de compra construido distinto por ola**: en 2010, algunos
    artículos son de "compra esporádica" (columna `compro`=Sí/No) y otros
    de "compra rutinaria" (solo tienen `per_compra`, sin `compro`); el
    indicador de compra para estos últimos se infiere como 1 si
    `per_compra` no es nulo ni "No compra" (`compro_a_bin_2010()`). En
    2013/2016 todos los artículos usan `compro`="SI"/"NO" directamente.
    Esta asimetría de diseño del cuestionario, no un error del código, es
    la razón por la que hogares que compraron ciertos artículos en 2010
    pueden tener `per_{a}`=NaN incluso habiendo comprado (ver la auditoría
    de `build_gasto_hogar.py` más abajo, del 2026-08-06, que corrige el
    manejo de esos NaN aguas abajo).
  - **8 columnas por artículo** (`gasto_`, `vr_`, `per_`, `adq_`,
    `vr_obt_`, `finca_`, `pago_`, `regalo_`): `adq_`/`finca_`/`pago_`/
    `regalo_` se construyen como mutuamente excluyentes (`adq_`=1 si y
    solo si alguna de las tres fuentes de obtención sin compra es 1).
  - **12 artículos sin columna `vr_obt_`** (valor de lo obtenido sin
    comprar): no es una omisión, es que ningún hogar de la muestra
    reportó haber obtenido esos 12 artículos específicos sin comprarlos,
    por lo que `pivot_table` nunca genera la columna. Lista completa en
    el docstring del script.
  - **Armonización de nombres de artículo entre olas**
    (`ARMONIZACION_ARTICULOS`, ~30 mapeos): cubre diferencias de redacción
    entre 2010 Rural/Urbano y 2013/2016 para el MISMO artículo (ej. "huevo"
    vs. "hueso" en carne, un typo real en el cuestionario 2010 rural;
    "cerveza, aguardiente" vs. "aguardiente, cerveza"), no diferencias de
    codificación. Sin esta armonización, `build_gasto_hogar.py` trataría
    estos ~30 artículos como inexistentes en una ola cuando en realidad
    solo cambió la redacción -- exactamente el tipo de pérdida silenciosa
    de datos que la auditoría de gasto (más abajo, 2026-08-06) buscó y
    corrigió para el resto del pipeline.
  - **Corrección de codificación** (`???`/`??` → vocales acentuadas,
    `S???`→`SI`): aplicada por reemplazo exacto de subcadena, con
    diccionarios específicos por ola/zona porque la codificación fuente
    corrupta difiere (2010 U usa patrones de 2 signos, el resto usa 3).
    Detalle exhaustivo en el script; no se duplica aquí.

**Consolidación de personas (`06_consolidacion_bases_personas.py`).**
  - **Identificador de persona elegido**: `llaveper`/`llaveper_n16` (llave
    de sub-hogar + orden dentro del hogar), NO `llave_ID_lb` (que rastrea
    al mismo individuo a través de las 3 olas pero tiene NaN para
    cualquier persona incorporada después de 2010 -- nuevos miembros,
    nacimientos, cónyuges -- por lo que no sirve como llave única del
    consolidado). Esta elección es la que permite a
    `build_pobreza_desagregaciones.py` identificar exactamente un jefe de
    hogar por sub-hogar en las 3 olas (verificado: 9.853/9.261/8.818
    personas = número de hogares, ver auditoría de desagregaciones más
    abajo, 2026-08-08).
  - **Alcance deliberadamente limitado de las correcciones de codificación
    en 2013**: la corrupción de caracteres en esa ola solo afecta texto
    libre (`descrip_oficio`, nombres de partidos políticos) y un par de
    etiquetas de opción abierta ("Otra, ¿cuál?"); los campos categóricos
    analíticamente relevantes (`parentesco`, `sexo`, `estado_civil`,
    `etnia`, `nivel_educ`, usados en la desagregación por jefe de hogar)
    ya estaban limpios en la fuente. Se corrigen solo las etiquetas fijas;
    el texto libre se deja tal cual (corregirlo requeriría un diccionario
    de miles de palabras, sin beneficio analítico dado que no se usa texto
    libre en ningún script de `04_features/`).
  - **Calidad de datos, sin corregir**: en 2016 hay personas con
    `llave`/`hogar` en NaN (detectadas en esa ola pero cuyo hogar de
    referencia no tiene sub-hogar asignado en ola 2). Tienen
    `llave_n16`/`llaveper_n16` válidos, así que son identificables, pero
    un join por `llave` no las encontrará. No se investigó si esto afecta
    materialmente el conteo de jefes de hogar en la desagregación por
    características del jefe -- queda como pregunta abierta menor.

**Choques económicos (`01_consolidacion_bases_choques.py`).**
  - Unidad sub-hogar × ola, misma jerarquía de llaves que el resto del
    panel (`consecutivo`/`llave`/`llave_n16`).
  - Los archivos de 2010 vienen sin encabezado, en formato ancho con
    bloques posicionales de columnas (hasta 7 choques en Urbano, 6 en
    Rural); `raw_2010_a_long()` reconstruye el formato largo leyendo esos
    bloques fila por fila. 2013/2016 ya vienen en formato largo.
  - **El indicador `conteo` (número de veces que ocurrió el choque) se
    construye con una lógica DISTINTA por ola** -- no es literalmente la
    misma variable reetiquetada: en 2013 se cuenta cuántas columnas
    `mes_*` no son nulas; en 2016 se suman las columnas `veces_{año}`; en
    2010 viene explícito en el bloque posicional. En los tres casos se
    aplica `.clip(lower=1)`: si el hogar reportó el choque pero no quedó
    ningún mes/año registrado, igual cuenta como 1 ocurrencia en vez de 0,
    para no perder el choque reportado por un hueco de captura del detalle.
  - **Armonización de 3 categorías de choque entre olas**
    (`ARMONIZACION_CHOQUES`): "Enfermedad" y "Accidente" son categorías
    separadas en 2010 pero se fusionan en una sola desde 2013 (se mapean
    ambos nombres de 2010 al nombre canónico posterior, para no duplicar
    la categoría); se depuran paréntesis plurales de dos etiquetas; y
    "Robo, incendio o destrucción de bienes del hogar" (nombre corto, en
    2010 y 2013 Rural) se unifica con la versión larga usada en 2013
    Urbano/2016, evitando que queden como dos columnas `choque_`
    distintas para el mismo concepto.
  - `imp_econ_*` (impacto económico del choque) no existe en 2010, solo en
    2013/2016 -- queda NaN para toda la ola 1, documentado en el docstring.
  - Corrección de codificación estándar (`???`→vocal acentuada/ñ), con un
    caso especial: en `UChoques` 2016 el indicador de filtro `tuvo_choque`
    usa `S???` en vez del patrón habitual y se corrige aparte por ser una
    columna de filtro, no de contenido.
  - Output: `data/processed/choques_elca_longitudinal.parquet` (sin
    dimensiones fijas documentadas en el script; se imprimen al ejecutar).

**Activos productivos rurales (`03_consolidacion_bases_RActivos.py`).**
  - Módulo exclusivo de zona rural -- no existe base urbana equivalente,
    por lo que no requiere armonización Urbano/Rural.
  - **A diferencia de los demás módulos, este NO tiene ninguna
    armonización de nombres entre olas**: los renombramientos reales entre
    2010 y 2013/2016 (`abejas`/`n_abejas`→`colmenas`/`n_colmenas`;
    `peces`/`n_peces`→`estanque`/`n_estanque`; `n_oanim`→`n_otros_anim`/
    `n_otro_cual`) se dejan como columnas separadas, pobladas solo en su
    ola de origen -- a diferencia del criterio ya aplicado en gasto
    (`ARMONIZACION_ARTICULOS`) y en choques (`ARMONIZACION_CHOQUES`). Es
    una limitación de comparabilidad real, documentada en el docstring
    del script pero no resuelta con código.
  - `vr_alquiler_*` (30 columnas, valor de alquiler de cada activo) solo
    existe en 2010; se dejó de preguntar después.
  - **Centinelas numéricos exclusivos de 2010**: las columnas de cantidad
    (`n_*`) y valor de alquiler contienen ocasionalmente los strings
    literales `"No informa"`/`"No sabe"` en vez de un número
    (`CENTINELAS_NUMERICOS_2010`); se convierten a NaN vía
    `pd.to_numeric(errors="coerce")` para mantener tipo `float64`
    consistente con 2013/2016.
  - Corrección de codificación estándar; 2013 además corrige `region`
    (Atlántica/Pacífica) y el campo libre `n_otro_cual`; 2016 no tiene
    columna `region`.
  - Output: `data/processed/RActivos_hogar_unido.parquet`, 12.810 filas ×
    112 columnas (2010=4.578×97, 2013=4.339×79, 2016=3.893×79).

**Comunidades (`05_consolidacion_bases_comunidades.py`).**
  - **Unidad de análisis distinta al resto del pipeline: comunidad × ola**
    (barrio urbano o vereda rural), no hogar/sub-hogar. Único
    identificador `consecutivo_c` -- la encuesta la responden hasta 6
    líderes comunitarios, no los hogares. El join recomendado hacia Hogar
    es `["consecutivo_c", "ola"]` con `how="left"` (nunca `inner`, porque
    2 comunidades quedan con `ola`=NaN en la fuente 2010 y un inner las
    descartaría, perdiendo hogares del panel principal).
  - **`RComunidades-csv.tab` de 2010 no trae encabezado**: el script
    recupera los 235 nombres de columna del archivo compañero
    `RComunidades.tab` y los asigna posicionalmente, verificando primero
    que ambos archivos tengan exactamente 235 columnas.
  - `zona` se llama `zona_2016` en la ola 3 sin que exista una `zona` de
    referencia previa como en Hogar; el script NO la unifica (queda NaN
    para toda la ola 2016), a diferencia de `03_clean/01_limpieza_base_hogar.py`
    (2026-06-30) que sí resuelve ese mismo problema para el módulo de
    hogar.
  - Dos patrones de corrección de codificación distintos por archivo:
    `???` (3 signos) para las 6 bases, `??` (2 signos) exclusivo de
    `RComunidades` 2010. **Orden de aplicación crítico**: el patrón de 3
    signos debe corregirse ANTES que el de 2, porque `S???` contiene `S??`
    como subcadena -- invertir el orden dejaría un signo de interrogación
    residual (`Sí?` en vez de `Sí`).
  - Espacios iniciales residuales en los módulos de desastres de 2013
    (ej. `' Destru???da Totalmente'`) generarían, tras corregir `???`, dos
    categorías distintas para el mismo concepto si no se aplicara
    `.str.strip()` al final de la corrección.
  - Cobertura de módulos no comparable entre olas (conflictos de tierra:
    exclusivo Rural; desastres naturales y ayuda humanitaria: agregados en
    2013; organizaciones sociales: agregado en 2016; protestas
    ciudadanas: exclusivo 2016) -- sin intento de armonizar contenido.
  - Output: `data/processed/comunidades_elca_longitudinal.parquet`, 2.330
    filas × 558 columnas.

**Niños (`07_consolidacion_bases_ninos.py`).**
  - Unidad niño × ola. Identificador más granular disponible cambia por
    ola igual que en Personas (`consecutivo`+`orden` en 2010, `llaveper`
    en 2013, `llaveper_n16` en 2016); `llave_ID_lb` vincula al niño con su
    registro en Personas solo si era miembro original del panel.
  - **Decisión central: NO filtrar por rango de edad al consolidar.** El
    rango cubierto por el cuestionario cambia entre olas (0-9 años en
    2010, 0-13 en 2013, 6-16 en 2016); el único rango presente en las 3
    olas es 6-9 años. El script consolida los 6 archivos tal cual, dejando
    en NaN lo que no aplica -- la responsabilidad de filtrar por edad para
    comparabilidad estricta se delega explícitamente al analista aguas
    abajo.
  - `ola`/`zona` no existen en ningún archivo fuente de este módulo (a
    diferencia de Hogar/Personas, donde `ola` sí viene en 2010); se
    agregan por completo en cada función `procesar_*()`.
  - `UNinos6a16-csv.tab` de 2016 (Urbano) no trae encabezado -- mismo
    patrón de recuperación posicional que en Comunidades, verificando 145
    columnas coincidentes antes de asignar.
  - **El módulo con codificación más heterogénea del panel**: en 2010
    Urbano/Rural, la mayoría de columnas usa `???` (3 signos) pero un
    subconjunto específico (`resultado_m`, `pecho`, `razon_no_asiste_cual`,
    `cuidado_prefiere_cual`) usa `??` (2 signos) -- se aplican ambos
    diccionarios en secuencia. En 2013 Rural, el patrón ` ??? ` (con
    espacios) representa un guion largo "–", no una vocal, y se corrige
    con prioridad por ser inequívoco. En 2016 hay un caso de doble
    sustitución en la misma cadena (`'Otra, ???cu???l?'`→`'Otra, ¿cuál?'`,
    donde el primer `???` es `¿` y el segundo es `á`).
  - `RNinos6a16` 2016 tiene 2 columnas más que `UNinos6a16`
    (`ano_edad_padre`, `ano_edad_madre`, presentes en Rural desde 2010,
    nunca en Urbano) -- sin intento de armonización por no tener
    equivalente urbano.
  - Cobertura de módulos no comparable entre olas: 2013 es la más ancha
    (348 columnas) por incorporar salud neonatal, lactancia extendida,
    vacunas y la prueba de desarrollo cognitivo TVIP; 2016 (145-147
    columnas) elimina esos tres módulos respecto a 2013 aunque amplía el
    rango de edad.
  - Output: `data/processed/ninos_elca_longitudinal.parquet`, 25.636 filas
    × 433 columnas.

**Nota de relevancia potencial para pobreza/vulnerabilidad** (informativa,
no es una decisión tomada): de los 4 módulos que no alimentan
`04_features/`, **choques económicos** es el candidato más directo como
predictor de transición a la pobreza -- pérdida de empleo,
enfermedad/accidente, muerte de un miembro, pérdida de
vivienda/cosechas/animales, desastres, violencia, con su impacto económico
declarado y estrategia de afrontamiento, es exactamente el tipo de
variable que la literatura de vulnerabilidad (Lopez-Calva y Ortiz-Juarez,
ya citado en este documento) usa para explicar transiciones, y se vincula
al panel de hogares con las mismas llaves que ya usa todo el pipeline.
Activos productivos rurales podría aportar una medida de capital físico
rural no capturada por ingreso/gasto, aunque limitada a la submuestra
rural. Comunidades y Niños tienen unidades de análisis distintas al hogar
(comunidad, niño) y su relevancia directa a "predicción de transición a
pobreza monetaria del hogar" (la Research Question de este documento) es
más indirecta -- Comunidades como controles de contexto geográfico si se
hace merge por `consecutivo_c`+`ola`, Niños solo si se explorara
transmisión intergeneracional, fuera del alcance actual.

### 2026-06-30 — Consolidación y limpieza de hogares (incluye hallazgo del
bug de `region` 2013 Urbano)

Documentación retroactiva (agregada 2026-08-08).

**Consolidación de hogares (`04_consolidacion_bases_hogar.py`).**
  - Mismo patrón de corrección de codificación que gastos, con un
    diccionario propio (`CORRECCIONES_HOGAR`) más extenso (variables de
    vivienda, crédito, religión, seguros, desastres naturales).
  - **Decisión explícita de NO unificar `zona`/`region` para 2016 en este
    paso**: los archivos fuente de 2016 no traen `zona`/`region`
    (traen `zona_2016`/`region_2016` en su lugar); este script se limita a
    concatenar tal cual, dejando `zona`/`region` en NaN para toda la ola
    2016. La unificación se delega explícitamente a
    `03_clean/01_limpieza_base_hogar.py` (ver más abajo) -- separación
    deliberada entre "unificar estructura" (este script) y "limpiar
    inconsistencias" (`03_clean`), ambos creados el mismo día.
  - **Normalización de tipos mixtos antes de guardar en parquet**: al
    concatenar 6 archivos con esquemas distintos, algunas columnas quedan
    como `object` con mezcla de `str` y `float` (numéricas en una ola,
    texto en otra); `pyarrow` rechaza esa mezcla. Se convierten a `str`
    preservando NaN. Mismo patrón aplicado en personas y en el script de
    comunidades (ambos del 2026-06-27).

**Limpieza de hogares (`03_clean/01_limpieza_base_hogar.py`) -- HALLAZGO
IMPORTANTE no documentado hasta ahora.**
  - **Bug real de datos, ya corregido en código pero nunca registrado
    aquí**: la columna `region` en la ola 2013 Urbano NO contenía la
    clasificación estándar ELCA de 9 regiones, sino una clasificación
    geográfica sub-regional distinta (posiblemente heredada del diseño
    muestral rural) -- hogares clasificados como "Bogotá" en `RegionLb`
    aparecían con `region`="Cundi-Boyacense" u "Oriental"; hogares de
    "Atlántica"/"Pacífica" aparecían con `region`="Central"/"Eje
    Cafetero". Sin corregir esto, cualquier tabla de pobreza por región
    perdía Bogotá COMPLETA en la ola 2013 (711 hogares → 0) y mostraba
    apenas 13 hogares Atlántica / 10 Pacífica en vez de los ~1.000 que
    tiene cada región en las otras olas -- de forma silenciosa (sin error,
    solo NaN). **Esto afecta directamente** la tabla
    `fgt_por_ola_region.csv` de `build_pobreza_desagregaciones.py`: sin
    esta corrección (que YA estaba aplicada desde la creación de este
    script, meses antes de que existiera esa tabla), la desagregación
    regional de pobreza para 2013 habría estado seriamente distorsionada.
    Corrección aplicada: se sobreescribe `region` con `RegionLb` (la
    columna que sí trae las 9 categorías ELCA estándar, 0 NaN) para TODA
    la ola 2013 -- inofensivo para la zona rural (donde ambas columnas ya
    coincidían) y necesario para la urbana. Se valida con `assert` que
    `RegionLb` no tenga NaN en ola 2 antes de sustituir, para no
    introducir NaN donde antes había datos válidos.
  - **`zona`/`region` en NaN para toda la ola 2016**: corregido con
    `fillna()` desde `zona_2016`/`region_2016`. Se valida con `assert`
    que `zona`/`zona_2016` (e idem `region`) nunca estén pobladas
    simultáneamente en la misma fila, antes de aplicar el fillna, para
    detectar cualquier cambio futuro en los datos fuente que invalidara
    el supuesto. 37 de 8.818 filas de 2016 quedan con `region`=NaN de
    todas formas (el propio `region_2016` tenía esos NaN en la fuente; no
    hay dato alternativo para imputarlos).
  - **Orden de aplicación importante**: la corrección de región 2013 debe
    ejecutarse ANTES del fillna de zona/región 2016, porque el fillna solo
    rellena NaN y no sobreescribe -- si se invirtiera el orden, los NaN de
    2013 U quedarían sin corregir.
  - Este script se documenta a sí mismo como "vivo": nuevos problemas de
    consistencia entre olas que se detecten a futuro deben agregarse aquí
    como nuevas funciones, siguiendo el mismo patrón (una función por
    problema, con el mismo nivel de detalle).

### 2026-07-02 — Descarga de datos ELCA vía API

Documentación retroactiva (agregada 2026-08-08).

**Descarga (`00_descarga_API_bases.py`).**
  - Fuente: API de Dataverse (DataHub Uniandes) para las 3 olas ELCA
    usadas (2010: `doi:10.57924/DPF0M5`, 2013: `doi:10.57924/DE9LP7`, 2016:
    `doi:10.57924/BLUILW`), descarga programática vía metadata JSON +
    descarga en paralelo (`ThreadPoolExecutor`, 6 workers) + extracción
    automática de .zip.
  - El script también descarga 2 olas adicionales (2019, 2022) publicadas
    por el DANE bajo el nombre ELCO (plataforma NADA, no Dataverse) --
    decisión: se descargan porque existen y se documentan, pero NO se usan
    en ningún script de `04_features/`; la tesis actual se limita a las 3
    olas ELCA con datos de ingreso/gasto/pobreza comparables (ver
    `config_dane.py`: la serie de líneas de pobreza metodológicamente
    continua solo cubre 2010-2016).
  - Los archivos crudos se guardan sin modificar en `data/interim/raw/`
    (ver `docs/data.md`); ninguna transformación ocurre en este paso.

### 2026-07-03 — Consolidación ELCO (olas DANE 2019/2022)

Documentación retroactiva (agregada 2026-08-08).

**¿ELCO es una encuesta distinta o la continuación de ELCA?** Ambas cosas
a la vez, verificado en el código: `08_consolidacion_bases_hogar_ELCO.py`
y `09_consolidacion_bases_personas_ELCO.py` conservan explícitamente las
columnas `CONSECUTIVO_DANE_2010`/`_2013`/`_2016` (y `_ELPS_2012`/`_2015`,
de otra encuesta intermedia) en sus identificadores -- confirma que ELCO
2019/2022 SÍ rastrea a los mismos hogares/personas del panel ELCA
2010-2016, es la continuación longitudinal de la misma muestra base. Pero
el INSTRUMENTO es estructuralmente distinto: el propio docstring titula la
encuesta "Encuesta Longitudinal Colombiana de Origen y Destino", con
organización en capítulos con letras y variables tipo `P#####` (formato
DANE estándar), sin relación de nombres con las variables `ing_*`/`gasto_*`
de ELCA; incluso los nombres de capítulo cambian entre 2019 y 2022 sin
cambiar de contenido, y el módulo `O_HISTORIAL_DE_ACTIVIDADES` desaparece
por completo en 2022 -- evidencia de que el cuestionario se rediseñó de
una ola a otra de forma más drástica que entre las olas ELCA 2010/2013/2016.
**Conclusión práctica**: ELCO permitiría en principio extender el panel
más allá de 2016 (el identificador de enlace existe), pero NO es un simple
"pegue" de columnas -- requeriría remapear manualmente las preguntas de
ingreso/gasto de ELCO (estructura y nombres totalmente distintos) antes de
poder alimentar `04_features/`. Ninguno de los dos scripts ELCO intenta
ese remapeo.

**Consolidación de hogares ELCO (`08_consolidacion_bases_hogar_ELCO.py`).**
  - **Llave de unión intra-ola = `DIRECTORIO`, no
    `CONSECUTIVO_DANE_ELCO_2019/2022`**: en 2022 esos identificadores
    llegan corruptos en el archivo fuente (exportados en notación
    científica con solo 6 cifras significativas, ej. `"1,16048E+14"`),
    generando 2.095 de 15.499 filas con el identificador duplicado.
    `DIRECTORIO` es único en ambas olas y se usa para el merge; los
    `CONSECUTIVO_DANE_ELCO_*` se conservan solo como referencia nominal.
  - Módulos cíclicos (varias filas por hogar, ej. gastos por categoría) se
    ensanchan con `pivotear_ciclico()`: `F_CICLO_GASTOS_DEL_HOGAR` usa el
    valor de la columna de categoría (`CAP`) como sufijo de columna;
    `D_CREDITOS` (sin columna de categoría) usa simplemente el orden de
    aparición como sufijo.
  - Cada merge usa `how="left"` + `validate="one_to_one"` explícito: un
    hogar sin registro en un módulo queda en NaN, no desaparece; si algún
    módulo tuviera más de una fila por `DIRECTORIO`, el script falla con
    `MergeError` en vez de mezclar datos silenciosamente.
  - Concatenación (apilado) de olas, no cruce ancho: la unidad de análisis
    es hogar × ola, no hogar único con columnas por ola, porque 2022 no es
    un simple re-cruce de los hogares de 2019 (hay entradas/salidas del
    panel).
  - Lectura con `dtype=str` en todas las columnas (evita truncar
    identificadores largos o convertir códigos con ceros a la izquierda en
    enteros); separador `","` en 2019 y `";"` en 2022.

**Consolidación de personas ELCO (`09_consolidacion_bases_personas_ELCO.py`).**
  - Mismo problema de `CONSECUTIVO_DANE_ELCO_PER_2022` corrupto (22.368 de
    35.766 filas de `A_ENLISTAMIENTO` duplicadas bajo esa llave); se usa
    `DIRECTORIO`+`ORDEN` como llave de unión.
  - **Deduplicación de filas 100% idénticas antes de unir**
    (`deduplicar_exactas()`), necesaria aquí y no en hogares: varios
    módulos traen filas exactamente iguales bajo la misma llave (mismo
    `DIRECTORIO`+`ORDEN` y todas las demás columnas), atribuido a doble
    digitación/exportación. Ejemplos con conteo verificado:
    `A_ENLISTAMIENTO` 2019 (1 fila), `H_MIGRACION*` 2022 (13/16/27 filas),
    `I_CICLO_EDUCACION` 2022 (61 filas), `L_CICLO_SALUD_FECUNDIDAD` 2022
    (55 filas), `M_GASTOS_PERSONALES` 2022 (23 filas). Solo colapsa
    duplicados EXACTOS; una llave repetida con contenido distinto no se
    toca y hace fallar el merge explícitamente.
  - `O_CICLO_HISTORIAL_DE_ACTIVIDADES`/`O_HISTORIAL_DE_ACTIVIDADES` no
    existen en 2022 -- el script omite la lectura con mensaje explícito en
    vez de intentar abrir un archivo inexistente.
  - `L_SALUD.csv` de 2022 es el único de los 19 módulos × 2 olas que no
    está en UTF-8 (viene en Latin-1); el fallback de codificación genérico
    del script solo se activa realmente para este archivo.
  - Mismas decisiones de `dtype=str`, `how="left"`+`validate="one_to_one"`
    y concatenación por apilado que en el script de hogares.

### 2026-07-29 — Exploración de atrición del panel (`eda_hogares_panel.py`)

Documentación retroactiva (agregada 2026-08-08). Script exploratorio de
solo lectura en `src/02_build/`, no forma parte del pipeline oficial (no
alimenta `04_features/`). Responde 5 preguntas descriptivas del panel de
hogares (2010/2013/2016): hogares únicos por ola y su composición
geográfica, hogares que se dividen entre olas, atrición (hogares
perdidos) y su composición por zona/estrato/región/depto/municipio,
distribución de tamaño de hogar, y personas en hogares perdidos. Lee la
ruta VIGENTE del pipeline (`hogar_elca_longitudinal_clean.parquet`, no
una ruta obsoleta) y persiste en `outputs/tables/eda_hogares/` y
`outputs/figures/eda_hogares/`. Funcionó como exploración previa al mismo
tipo de análisis de atrición que luego se formalizó, más adelante
(2026-08-08), en la revisión de panel de `build_pobreza_desagregaciones.py`
(tabla `atricion_panel.csv`).

### 2026-08-05 — GSV: segmentación semántica con SAM 3

Documentación retroactiva (agregada 2026-08-08).

**Segmentación semántica con SAM 3 (`03e_extraer_segmentacion_sam3.py`).**
  - Modelo SAM 3 (Meta, nov. 2025), segmentación promptable por texto
    libre corto (frases nominales tipo "car", "pothole"). Requiere GPU
    CUDA 12.6+; CPU es funcional pero "no soportado ni probado por Meta",
    recomendado solo para pilotos pequeños. Checkpoint de acceso
    restringido (requiere solicitud en Hugging Face).
  - **14 conceptos a segmentar** en 4 categorías (infraestructura vial,
    vivienda/construcción, activos/vehículos/comercio,
    entorno/espacio público), prompts en inglés por convención del
    vocabulario de entrenamiento de SAM3.
  - Por concepto e imagen se generan 3 variables: conteo de instancias
    sobre umbral de confianza 0.5, fracción de píxeles cubierta por la
    UNIÓN de todas las máscaras del concepto (para no contar doble si dos
    instancias se solapan), y confianza promedio de las detecciones.
  - El backbone visual se calcula una sola vez por imagen y se reutiliza
    para los 14 conceptos (solo se recalcula el encoder de texto por
    prompt), optimización explícita para no recomputar el backbone 14
    veces.
  - Limitación explícita de costo: modelo de 848M parámetros; en CPU
    "puede tomar de segundos a minutos por imagen, inviable para el
    corpus completo de la ELCA".

### 2026-08-06 — Ingreso, gasto y pobreza monetaria (ELCA 2010, 2013, 2016)

Pipeline de scripts separados en `src/04_features/`, cada uno con un
artefacto propio (mismo patron que `build_ingreso_hogar.py`, ya existente):

  `config_dane.py`            -> parametros oficiales DANE (LP, LI, insumos IPC)
  `build_deflactor_ipc.py`    -> deflactores IPC total / IPC ingresos bajos, base 2010
  `build_ingreso_hogar.py`    -> ingreso nominal + series reales (ya existia, extendido)
  `build_gasto_hogar.py`      -> gasto mensual del hogar, normalizado por periodicidad
  `build_pobreza_monetaria.py`-> clasificacion pobre/no pobre (ingreso oficial + gasto robustez)

Decisiones metodologicas:

1. **Medida oficial de pobreza = ingreso**, siguiendo la metodologia DANE:
   ingreso per capita nominal del hogar vs. LP/LI nominal de su año y zona
   (Urbano/Rural). El gasto se construye como medida PARALELA de robustez,
   no reemplaza al ingreso como criterio oficial.
2. **Lineas de pobreza**: se usaron los valores oficiales del DANE para
   2010 (MESEP), 2013 y 2016 (boletines de pobreza monetaria), todos en la
   misma serie metodologica continua (linea base ENIG 2006-2007). Fuentes
   documentadas en `docs/fuentes_dane/README.md`.
3. **Deflactor de precios**: se construyeron dos series (IPC total nacional
   e IPC de ingresos bajos), a promedio ANUAL de cada ola (no al mes exacto
   de entrevista, aunque ese dato existe en la ELCA). Los deflactores NO se
   usan para clasificar pobreza (esa comparacion es nominal contra nominal
   del mismo ano); se usan solo para series de ingreso/gasto reales.
4. **Gasto del hogar**: se recalculó mensualizando cada item de gasto según
   su periodicidad declarada (`per_{articulo}`), en vez de usar el
   `total_gasto` ya presente en `gastos_elca_longitudinal.parquet` (que suma
   valores de periodicidades distintas sin normalizar). Ver docstring de
   `build_gasto_hogar.py` para el supuesto de recall anual en los 35
   artículos sin dato de periodicidad.

### 2026-08-06 (cont.) — deflactor por zona y arriendo imputado

- **Deflactor por zona**: `deflactor_ipc_ing_bajos` (el derivado de la LP)
  ahora se calcula por separado para Urbano y Rural, no solo a nivel
  nacional, porque la LP urbana y la LP rural no crecen exactamente igual
  año a año. `deflactor_ipc_total` sigue siendo nacional único: no existe
  una serie de IPC total oficial desagregada por zona urbano/rural
  descargable (el DANE desagrega el IPC por ciudad, no por dominio
  urbano/rural). Esto NO afecta la clasificacion de pobreza (que ya usaba
  LP/LI por zona desde el principio); solo afecta las series de
  ingreso/gasto reales.
- **Arriendo imputado**: se investigo si la ELCA permite estimar el valor
  del arriendo que un hogar propietario "se ahorra" (componente que el DANE
  si incluye en su concepto oficial de ingreso). Resultado: la ELCA NO
  pregunta esto. `valor_arriendo_pagado` (pregunta 117 del modulo Hogar)
  es exclusivamente el arriendo que pagan los hogares que SI arriendan
  (~21-23% de la muestra); no existe una pregunta equivalente para que los
  hogares propietarios estimen cuanto pagarian si tuvieran que arrendar.
  Por lo tanto, `ingreso_total_hogar` sigue sin arriendo imputado -- es una
  limitacion real de los datos fuente, no una omision de construccion, y se
  deja documentada aqui. Alternativa posible (no implementada): estimar un
  arriendo imputado por regresion hedonica (arriendo pagado por
  arrendatarios en funcion de caracteristicas de la vivienda, aplicado a
  los propietarios) -- requiere una decision metodologica aparte antes de
  construirse.

### 2026-08-06 (cont.) — Deflactor IPC: metodología de construcción
completa (`build_deflactor_ipc.py`)

Documentación retroactiva del detalle metodológico completo (agregada
2026-08-08; el script y la decisión son del mismo día, 2026-08-06).

  - **Reconstrucción del IPC total nacional**: el archivo fuente del DANE
    (`IPC_Variacion.xls`) trae variaciones PORCENTUALES mes a mes, no el
    índice en niveles; se reconstruye encadenando esas variaciones desde
    una base arbitraria de 100 en diciembre de 2008 (mes de referencia de
    la metodología IPC-08, vigente sin quiebre metodológico entre 2010 y
    2016 -- por eso es seguro encadenar sin ajuste de empalme). El nivel
    anual de cada ola es el promedio de los 12 meses de ese año.
  - **El deflactor "ingresos bajos" NO se construye de una serie de IPC
    independiente**: no existe un archivo histórico descargable del IPC
    específico para el grupo de "ingresos bajos" (el que el DANE usa para
    actualizar la LP), ni desagregado por zona. En su lugar, se DERIVA
    directamente de la razón entre los valores oficiales de LP de dos años
    (de `config_dane.LP_NACIONAL`/`LINEAS_POBREZA`) -- válido porque el
    DANE documenta explícitamente que la LP se actualiza cada año
    exactamente con ese índice, así que la razón entre dos LP oficiales
    ES por construcción la inflación acumulada de ingresos bajos entre
    esos años. Ver `docs/fuentes_dane/README.md` sección 2.
  - **Año base = 2010** (primera ola): todas las series "reales" del
    pipeline están en pesos de 2010 salvo que se indique lo contrario.

### 2026-08-06 (cont.) — por que la LP se queda en 2 dominios (Urbano/Rural)

Se evaluo si la clasificacion de pobreza deberia usar el nivel de
desagregacion mas fino que reportan los boletines del DANE (13 ciudades y
Areas Metropolitanas / Otras Cabeceras / Centros poblados y rural disperso,
en vez de solo Urbano/Rural). Dos hallazgos, ambos verificados contra
fuente primaria, decidieron mantener 2 dominios:

1. El propio MESEP declara "urbano y rural" como el NIVEL MAXIMO DE
   DESAGREGACION para construir la linea base de pobreza (no departamental
   ni por ciudad) -- `docs/fuentes_dane/Pobreza_nuevametodologia.pdf`,
   seccion de decisiones metodologicas. El desglose "13 AM / Otras
   Cabeceras" que aparece en los boletines 2013 y 2016 es un reporte
   agregado dentro de "Cabeceras", no una canasta/linea distinta.
2. Aunque quisieramos aplicar ese desglose de todas formas: la ELCA SI trae
   columnas `id_dpto` / `id_mpio` a nivel de hogar (se verifico en los .tab
   crudos y sobreviven en `hogar_elca_longitudinal_clean.parquet`), pero el
   diccionario de la encuesta (`data/interim/raw/diccionarios_elca/
   elca_2010_unido.pdf`, variables HR4/HR5) las describe explicitamente
   como **"Identificador FALSO del departamento/municipio"**: son codigos
   anonimizados por el equipo ELCA (probablemente para proteger la
   confidencialidad de hogares en veredas pequenas), sin correspondencia
   real con DIVIPOLA. No hay ninguna forma de saber, con los datos
   publicos de la ELCA, si un hogar esta en una de las 13 ciudades y AM.

Conclusion: la comparacion nominal por zona Urbano/Rural (ya implementada
en `build_pobreza_monetaria.py` desde el principio) es tanto la que exige
la metodologia base del DANE como el maximo nivel que los datos permiten
verificar. No se implemento el cruce de 3 dominios.

### 2026-08-06 (cont.) — Auditoria completa de build_ingreso_hogar.py

Se audito exhaustivamente `build_ingreso_hogar.py`contra las 69 columnas
 `ing_/act_/gan_/gto_` de
`hogar_elca_longitudinal_clean.parquet`, verificando tres cosas: (1) que
ninguna variable de ingreso quedara fuera sin explicacion, (2) que la
periodicidad (mensual vs. anual) de cada variable estuviera bien tratada,
y (3) que las exclusiones fueran reales y no un cambio de nombre entre
olas. Metodo: comparacion programatica de columnas vs. las listas del
script, cruce contra los `.tab` crudos de cada ola (no solo los archivos
`_unido` consolidados), y lectura de los diccionarios especificos por
ola/zona (`RHogar.pdf`/`UHogar.pdf` de cada ano, no solo el consolidado)
para wording exacto de cada pregunta.

**Periodicidad de las variables incluidas** — confirmado sin problemas:
las 8 variables de ingreso ya incluidas (`ing_trabajo`, `ing_trabagr`,
`ing_trabnoagr`, `ing_pensiones`, `ing_arriendos`, `ing_intereses_div`,
`ing_ayudas`, `ing_otros_nrem`) tienen la misma redaccion "ingresos
**mensuales**" en los tres cuestionarios -- se verifico buscando cada
variable en `elca_2010_unido.pdf`, `elca_2013_unido.pdf` y
`elca_2016_unido.pdf`. Ninguna mezcla periodicidades.

**Variables que se estaban perdiendo sin documentar** (ninguna incluida en
el ingreso, ninguna en `EXCLUIDAS_NO_COMPARABLES`):

1. `act_vtaanimdes` (venta de animales de desecho): columna numerica con
   valores reales, existe solo en ola 2010 (confirmado ausente en los
   `.tab` crudos 2013/2016). Se agrego a `EXCLUIDAS_NO_COMPARABLES`.
2. `act_ocasional_vr` ("ingreso ocasional"): columna numerica nueva, existe
   solo en 2013/2016 (confirmado ausente en los `.tab` crudos 2010, la
   columna ni siquiera existe para esa ola). Se investigo su magnitud:
   indicador `act_ocasional` coincide EXACTAMENTE con la presencia de
   valor (137/9261 hogares en 2013, 140/8818 en 2016, montos entre
   $10.000 y $29.000.000 -- dato limpio, sin inconsistencias). Se incluyo
   como `ingreso_ocasional`, pero SOLO para 2013/2016 (queda NaN, no 0, en
   2010, para no fingir que la pregunta existio esa ola).

**Rationale corregido para 6 de las 20 variables `act_*_vr` que estaban
excluidas** (`herencias`, `polizas`, `vtainm`, `vtaneg`, `otrosing`,
`vtaotros`): el docstring original decia "solo existen en ola 2010, sin
equivalente en 2013/2016" -- FALSO. Se verifico en los `.tab` crudos que
la pregunta (indicador Y valor) existe en las 3 olas. Lo que cambia es el
DISEÑO de la pregunta:

- **Rural 2010**: el monto se pedia directamente a (casi) todos los
  hogares, sin filtro previo -- ~100% de respuesta, con muchos ceros
  explicitos (mediana de `act_herencias_vr` en rural 2010 = 0, no NaN).
- **Urbano 2010 y TODAS las filas de 2013/2016**: primero se pregunta un
  filtro Si/No, y el monto solo se pide si "Si" -- por eso la respuesta
  parece "casi inexistente" en esas filas (1-2% del total), pero es
  porque casi nadie pasa el filtro, no porque la pregunta no exista.

Se verifico caso por caso que el indicador coincide 100% (o 99.9%) con la
presencia de valor bajo el diseno de filtro, tanto en 2013/2016 como en
urbano 2010, antes de usar esta logica. Excepcion de nombres: en ola 1
urbana el indicador de `vtaotros` se llama `act_vgtaotros` (no
`act_vtaotros`, que no existe para esa ola) -- se verifico el mismo
100% de coincidencia antes de mapearlo.

Con este hallazgo, las 6 variables se movieron de "excluidas" a
"incluidas": nuevo componente `ingreso_excepcional` desagregado en
`ingreso_herencias`, `ingreso_polizas`, `ingreso_vtainm`, `ingreso_vtaneg`,
`ingreso_otrosing`, `ingreso_vtaotros` (todas divididas entre 12, porque
son preguntas de 12 meses, a diferencia del resto del ingreso que ya es
mensual).

Las otras 14 variables `act_*_vr` (bonos, cesantias, dinero, fondos,
inversiones, roscas, seguros de vehiculo/vida/vivienda/maquinaria/
cosechas) siguen excluidas, y esta vez con el rationale correcto
verificado: la columna `_vr` (valor) directamente NO EXISTE en los `.tab`
crudos de 2013/2016 -- la pregunta de monto fue eliminada del
cuestionario despues de 2010, solo quedo el indicador Si/No.

**`ing_ayudas` en 2010 — hueco de cobertura, no no-respuesta.** Se detecto
que en ola 1, `ing_ayudas` tenia 0% de datos no-nulos en hogares rurales
vs. 17.3% en urbanos, y que a diferencia de `ing_pensiones` (que mezcla
explicitamente valores "0" con positivos) no hay NINGUN "0" explicito en
`ing_ayudas` -- todo blanco es indistinguible entre "no recibe" y "no se
pregunto". Se investigo la causa con tres fuentes independientes:

1. Columnas del `.tab` crudo rural 2010 (`RHogar-csv.tab` y `RHogar.tab`):
   `ing_ayudas` no existe como columna en ninguno de los dos.
2. Diccionario especifico rural (`data/interim/raw/elca_2010/RHogar.pdf`):
   la numeracion de items salta de "d. Intereses o dividendos" directo a
   la siguiente variable, sin ningun "e." intermedio.
3. Diccionario especifico urbano (`data/interim/raw/elca_2010/UHogar.pdf`):
   confirma que "e. Ayudas en dinero" (`ing_ayudas`, HU248) existe ahi, y
   que el item "f." siguiente (`ing_otros_nrem`) tiene la MISMA
   definicion textual ("Otros ingresos diferentes a remesas") en ambos
   cuestionarios -- descartando que "otros" en rural absorbiera "ayudas"
   bajo otro nombre.

Las tres fuentes coinciden: la pregunta "Ayudas en dinero" nunca se hizo a
hogares rurales en 2010 (~4.578 hogares, el 46% de la ola 1). No es
no-respuesta al azar, es un hueco de diseno del cuestionario. Decision:
se mantiene el tratamiento de 0 implicito (no hay forma de recuperar el
dato real, y es la practica mas razonable dado que no se puede imputar
una pregunta que nunca se hizo), pero ahora la razon real queda
documentada en el docstring de `build_ingreso_hogar.py` en vez de
mezclarse con el supuesto generico de "No informa"/"No sabe".

**Resultado**: el ingreso total mensual de la ola 1 subio de un promedio
de no-nulos de 9.818 a 9.853 hogares (todos los hogares rurales ahora
tienen valor en los componentes excepcionales, antes varios quedaban sin
ningun componente reportado), y la media/mediana de `ingreso_total_hogar`
aumento levemente en las 3 olas al incorporar `ingreso_excepcional` e
`ingreso_ocasional`.

### 2026-08-06 (cont.) — Auditoria completa de build_gasto_hogar.py

se aplico a `build_gasto_hogar.py` el mismo tipo de
auditoria hecha para el ingreso: (1) que ningun articulo de gasto se
excluyera por un cambio de nombre entre olas, (2) que la periodicidad
estuviera bien estandarizada para todos los articulos, y (3) que decision
tomar con los articulos sin columna per_. Metodo: comparacion programatica
de cobertura por ola/zona para los 88 articulos, cruce contra el manual
metodologico original (`data/interim/raw/diccionarios_elca/
ELCA-Manual-Hogar-Urbano2010.pdf`) y el `cod_articulo` de los `.tab`
crudos de 2010 urbano.

**Armonizacion de nombres**: no se encontraron articulos perdidos por
cambio de nombre entre olas -- `02_consolidacion_bases_gastos.py` (creado
el 2026-06-27) ya resuelve esos casos explicitamente via
`ARMONIZACION_ARTICULOS`. Lo que si se encontro es una diferencia real de
GRANULARIDAD dentro de la ola 1: la zona urbana de 2010 usa 35 categorias
agregadas (ej. "alimentos procesados" junta pastas, sal, azucar, panela,
cafe, etc. en un solo articulo), mientras que la zona rural de 2010 y
ambas zonas de 2013/2016 usan 72 categorias detalladas para los mismos
conceptos (cada uno por separado). Confirmado con `cod_articulo` de
`UGastos-csv.tab` 2010: los articulos 1-35 mapean exactamente a la lista
agregada urbana. Esto NO afecta el total del hogar (cada hogar reporta en
un solo sistema, sin doble conteo), pero si reduce la comparabilidad a
nivel de articulo individual dentro de la ola 1.

**BUG CRITICO encontrado y corregido** (esto no era un problema de
documentacion, era una perdida real de datos): el codigo decidia si usar
la periodicidad declarada (`per_{articulo}`) mirando si la COLUMNA existia
en el esquema del DataFrame, no si estaba poblada en esa fila especifica.
Para 9 articulos, la columna `per_` solo tiene datos reales en ola 1
urbana; en ola 1 rural y TODA ola 2/3 esta vacia para esos mismos
articulos (aunque el hogar si compro). Como el codigo intentaba
`per_.map(FACTOR_MENSUAL)` para TODAS las filas sin verificar si habia
dato, el resultado era NaN para las filas sin periodo -- y ese NaN no caia
a ningun supuesto de respaldo, sino que borraba el articulo del total.

Verificacion de impacto: para "ropa para hombre, mujer, niño y niña",
**10.296 de 10.301 hogares (99.95%)** que reportaron haberla comprado
quedaban con valor NaN -- el gasto en ropa habia desaparecido del total
para practicamente toda la muestra excepto ola 1 urbana. Los 9 articulos
afectados, con su tasa real de poblacion de `per_` dado que el hogar
compro:

  ropa_para_hombre_mujer_nino_y_nina                    0.05% con periodo (10.301 compraron)
  calzado_para_hombre_mujer_nino_y_nina                 0.01% con periodo ( 9.795 compraron)
  reparacion_de_calzado_y_o_vestuario                   0.05% con periodo ( 2.048 compraron)
  articulos_de_aseo_personal_...                       18.9%  con periodo (27.306 compraron)
  articulos_para_el_aseo_del_hogar_...                 19.0%  con periodo (26.941 compraron)
  corte_de_pelo_arreglo_de_unas                        28.3%  con periodo (13.381 compraron)
  bombillos_pilas_y_otros_articulos_electricos...      29.6%  con periodo ( 9.933 compraron)
  algodon_gasas_...botiquin                            33.6%  con periodo ( 7.419 compraron)
  empleados_del_servicio_domestico_internos            61.2%  con periodo (   307 compraron)

Correccion aplicada en `construir_gasto_mensual()`: ahora se usa
`per_{articulo}` fila por fila cuando esta poblado, y solo se recurre a un
factor de respaldo cuando no lo esta -- nunca se deja el articulo en NaN
por falta de periodo si el hogar si lo compro. El factor de respaldo se
definio por grupo, verificado contra fuente:

  1. Ropa, calzado, reparacion de calzado/vestuario: se penso inicialmente
     en usar la periodicidad "observada" en ola 1 urbana, pero se descubrio
     que esa cifra estaba basada en **una sola observacion por articulo**
     (n=1, dato residual de captura, no una muestra confiable). Se
     descarto y se uso en su lugar el capitulo del manual metodologico
     (confirmado por `cod_articulo`): estos 3 articulos, junto con
     reparacion de vehiculo y libros/discos/CDs, pertenecen al capitulo
     IX-B "Gastos TRIMESTRALES del hogar" (cod_articulo 21-25) -> factor
     de respaldo = Trimestral (/3).
  2. Articulos de aseo personal, aseo del hogar, corte de pelo, bombillos,
     botiquin, servicio domestico: la periodicidad real observada en ola 1
     urbana (unica fuente con dato individual, con muestras de n=188 a
     n=5.148 por articulo -- suficientes para ser confiables) SI se reparte
     entre varias categorias sin un ganador claro (ej. articulos de aseo
     personal: 52.6% mensual, 30.0% quincenal, 14.9% semanal, resto
     marginal). Factor de respaldo = promedio ponderado por esa
     distribucion real, calculado dinamicamente cada vez que corre el
     script (`_factores_respaldo_ponderados()`), no un valor fijo escrito
     a mano.

**Correccion de periodicidad para 2 de los 35 "articulos sin periodo"**:
se cruzo el `cod_articulo` de 2010 urbano contra el manual y se confirmo
que "reparacion_repuestos_y_mantenimiento_de_vehiculo_de_uso_del_hogar"
(cod 24) y "libros_discos_y_cds" (cod 25) son del capitulo IX-B
(TRIMESTRAL), no del IX-C (ANUAL) que se les asumia antes -- se movieron
de /12 a /3. Los otros 28 articulos sin columna `per_` corresponden al
capitulo IX-C del manual (Gastos ANUALES, cod_articulo 26-35: muebles,
electrodomesticos, colchones, ollas, vehiculo/moto, bienes raices,
hoteles, pasajes de avion, primas de seguros) mas gastos excepcionales de
salud/educacion/impuestos sin equivalente documentado en el manual; para
estos se mantiene el supuesto de recall anual (/12), ahora con la fuente
exacta (capitulo IX-C) citada donde aplica.

**Limitaciones documentadas, sin corregir con codigo** (requieren juicio
metodologico, no una formula):
  - "impuesto_a_la_renta_y_complementarios" (solo existe en 2010) e
    "impuesto_a_la_renta_complementarios_y_predial" (solo 2013/2016) NO
    son el mismo concepto -- el segundo agrega impuesto predial. Se
    mantienen como articulos separados, sin fusionar ni excluir; el monto
    de 2010 mide algo mas angosto que 2013/2016 en ese rubro puntual.
  - "Gastos en educación" y "Gastos en salud" en ola 1 RURAL tienen una
    tasa de compra muy baja (6/4.578 y 254/4.578 hogares) frente al
    equivalente urbano (~33%/9% via "educacion/salud de todos los
    miembros del hogar"). Se confirmo que la pregunta rural SI existe en
    `RGastos-csv.tab` (no es un caso como el de `ing_ayudas`), pero no hay
    una variable de filtro que permita distinguir si es una tasa de compra
    real muy baja o un problema de ventana de recall. Queda documentado
    como limitacion, sin ajuste adicional.

**Resultado**: gasto_mensual_hogar promedio subio de ~$961.032 a
~$1.049.783 (ola 1), ~$869.891 a ~$1.064.941 (ola 2), ~$1.072.693 a
~$1.301.705 (ola 3) al recuperar el gasto que antes se perdia
silenciosamente. La pobreza por gasto (medida de robustez) bajo de forma
correspondiente: 54.96%->47.57% (ola 1), 53.67%->40.89% (ola 2),
45.66%->33.43% (ola 3). La pobreza por ingreso (medida oficial) no cambia
por esta correccion -- el gasto no interviene en su calculo.

### 2026-08-06 (cont.) — GSV: embeddings CLIP (esqueleto) y scores de
percepción StreetScore

Documentación retroactiva (agregada 2026-08-08).

**Extracción de embeddings CLIP (`03b_extraer_embeddings_clip.py`) --
ESQUELETO, NO FUNCIONAL.** El propio script lo declara: los módulos de
carga del modelo y de cálculo de similitud lanzan `NotImplementedError`.
Lo que SÍ está definido y es una decisión metodológica real, aunque el
código no corra:
  - Modelo vía librería `open_clip` (no la implementación original de
    OpenAI), variante default `ViT-B-32` con checkpoint `openai`. El
    docstring documenta el trade-off frente a `ViT-L-14` (mejor accuracy
    zero-shot, mucho más lento, preferible para conceptos visualmente
    sutiles) y `RN50` (backbone convolucional, por debajo de ViT-B-32 en
    accuracy).
  - **La decisión central ya tomada es la lista de 18 prompts de scoring
    zero-shot**, agrupados en 5 categorías (vivienda/construcción,
    infraestructura vial, servicios básicos, espacio público/entorno,
    densidad, activos visibles). Cada concepto documenta explícitamente
    una hipótesis de correlación con vulnerabilidad Y un riesgo/objeción
    levantado por el mismo equipo -- ej.: "vía sin pavimentar" arriesga
    medir "¿es rural?" en vez de "¿es vulnerable?" (recomienda controlar
    por el indicador urbano/rural ya existente); "entorno cuidado" tiene
    objeción fuerte de que la limpieza correlaciona con esfuerzo/cultura,
    no con ingreso, posiblemente en dirección contraria a la asumida;
    "asentamiento informal" arriesga capturar sesgo de representación
    visual heredado del entrenamiento de CLIP (fotos etiquetadas
    "slum"/"favela" desde mirada occidental) en vez de informalidad real;
    "rejas de seguridad" tiene dirección ambigua (protección de activos
    vs. miedo a violencia) y se recomienda explícitamente NO construir
    hipótesis direccional hasta ver el dato. El docstring aclara que esta
    lista fue discutida entre perfiles de economía de la pobreza, ML,
    trabajo social, antropología, sociología, y una persona con
    experiencia vivida de pobreza -- y que ningún signo de correlación
    está confirmado, son hipótesis a validar.

**Scores de percepción urbana / StreetScore (`03d_extraer_scores_streetscore.py`).**
  - Modelo ViT-B-16 de Ouyang (2023), entrenado por regresión sobre
    comparaciones pareadas humanas de Place Pulse 2.0 (110.988 imágenes,
    56 ciudades). 6 dimensiones perceptuales fijas, score continuo 0-10:
    seguridad, animación, riqueza percibida, belleza, aburrimiento,
    depresión.
  - **Decisión explícita de reimplementar ~40 líneas en vez de usar el
    paquete `zensvi`** (que expone la misma clase): evita arrastrar ~35
    dependencias no relacionadas (open3d, faiss-cpu, geopandas, osmnx,
    rasterio, groundingdino-py) sin wheels precompiladas para todas las
    plataformas -- decisión de portabilidad, no metodológica sobre el
    modelo.
  - El score final es la probabilidad de la clase índice 1 (formato
    binario de comparación pareada de Place Pulse) escalada ×10, no un
    score de regresión directo.
  - Preprocesamiento: resize 384×384 (no 224 como en el pipeline
    VGG/ResNet/CLIP), sin center crop.
  - **Criterio de calidad más estricto que el resto del pipeline**: una
    imagen queda excluida del output final si le falta el score en
    CUALQUIERA de las 6 dimensiones.
  - Limitación explícita y central, citada textualmente del propio
    script: "estos scores provienen de comparaciones hechas por
    voluntarios de internet mayormente de fuera de Colombia. La
    correlación esperada con vulnerabilidad a pobreza... es una hipótesis
    a validar contra el índice de activos/SISBEN de la ELCA, no un
    supuesto aceptado".

**Conexión con el pipeline de pobreza ELCA: no implementada todavía (a la
fecha de esta auditoría, 2026-08-08).** Los dos pipelines -- GSV y pobreza
monetaria ELCA -- son hoy completamente independientes. El punto de
integración natural sería un merge por `consecutivo`/`llave`/`llave_n16`
(las mismas llaves que ya usa el pipeline de pobreza) entre las variables
visuales agregadas por hogar×ola y `pobreza_monetaria_elca_longitudinal.parquet`
-- técnicamente viable porque los scripts 03/03b/03d/03e ya incorporan
esas llaves en sus outputs, pero el código que haría ese join y la
regresión (scripts 04 y 05, ambos esqueletos vacíos desde 2026-06-12) todavía
no se ha escrito.

### 2026-08-08 — Desagregaciones de pobreza (build_pobreza_desagregaciones.py)

Nuevo script que parte de `pobreza_monetaria_elca_longitudinal.parquet` y
agrega:

1. **Indicadores FGT** (incidencia P0, brecha P1, severidad P2), para
   ingreso y gasto contra LP y LI, replicando la metodologia que el propio
   DANE reporta en sus boletines.
2. **Desagregacion geografica**: por zona (Urbano/Rural) y por las 8
   regiones propias de la ELCA (`region` de hogar_elca_longitudinal_clean)
   -- el nivel geografico mas fino que los datos permiten verificar (ver
   seccion anterior sobre por que la LP no se desagrega a 13 AM/otras
   cabeceras).
3. **Por jefe de hogar y hogar**: sexo, grupo de edad y nivel educativo del
   jefe (identificado via `parentesco` en personas_elca_longitudinal --
   se verifico exactamente un jefe por hogar en las 3 olas, 9853/9261/8818
   personas respectivamente, igual al numero de hogares) y numero de
   niños menores de 12 años en el hogar.
4. **Matrices de transicion de pobreza** entre olas consecutivas
   (2010->2013, 2013->2016), siguiendo la metodologia de Lopez-Calva y
   Ortiz-Juarez (2014) "A Vulnerability Approach to the Definition of the
   Middle Class" (Tabla 3): cruce de estado de pobreza inicial (filas) vs.
   final (columnas) en porcentajes de fila, y clasificacion en 4
   categorias (nunca pobre / siempre pobre / sale de la pobreza / entra en
   pobreza). Emparejamiento de hogares via `consecutivo`. Decision: los 
   hogares que se DIVIDEN entre olas (511 casos 2010->2013, 1.190 casos 2013->2016 
   contando ambos lados del
   emparejamiento) se EXCLUYEN de la matriz -- solo se usan matches 1 a 1 --
   y el conteo de excluidos se reporta junto a cada matriz.

Todas las tablas se guardan como CSV en `outputs/tables/pobreza/` (son
resultados descriptivos para el documento, no insumo de otro script).

Resultado de las transiciones (ingreso, medida oficial):
  2010->2013 (n=8.218): 43.9% siempre pobre, 28.8% nunca pobre, 18.6% sale
  de la pobreza, 8.8% entra en pobreza.
  2013->2016 (n=6.911): 37.0% nunca pobre, 33.3% siempre pobre, 20.6% sale
  de la pobreza, 9.2% entra en pobreza.

### 2026-08-08 (cont.) — Revision del panel de economistas/sociologos (Banco
Mundial, Harvard, MIT, Oxford, LSE) sobre construccion de ingreso, gasto y
pobreza

Se sometio el pipeline completo (`config_dane.py`, `build_ingreso_hogar.py`,
`build_gasto_hogar.py`, `build_pobreza_monetaria.py`,
`build_pobreza_desagregaciones.py`) a una revision externa simulada, con
foco en la variable de interes de la tesis (transiciones de pobreza). De 8
observaciones planteadas, se investigaron y/o implementaron 6; 2 quedan como
limitaciones de la fuente sin correccion posible (documentadas mas abajo).

**1. Pesos muestrales -- ausentes hasta ahora, implementados.** El pipeline
no usaba ningun factor de expansion: todas las incidencias de pobreza
reportadas hasta este punto (incluida la seccion de matrices de transicion
arriba) son proporciones MUESTRALES, no estimaciones poblacionales. Se
encontraron las columnas de peso en `hogar_elca_longitudinal_clean.parquet`
y se verifico su definicion exacta contra los diccionarios especificos de
cada ola (`UHogar.pdf` 2010/2013/2016, via `pdftotext`):

  - `fexhog` (HR254, 2010): "Factor de expansion Longitudinal (Factor
    Unidad)" -- peso transversal de la ola 1 a nivel de hogar (media ~1).
  - `fexhog_2013` (HU324, 2013): "Factor de expansion final del hogar en
    2013 (Factor Unidad)" -- peso transversal propio de la ola 2.
  - `fexhog_2010` (HU325 en el diccionario 2013, HU250 en el diccionario
    2016): "Factor de expansion 2010 Longitudinal (Factor Unidad)" -- un
    UNICO peso longitudinal, anclado a la muestra base de 2010, presente en
    las olas 2 y 3 (para permitir comparaciones panel contra esa base).
  - Se descarto `fhog`/`fhog_2013`/`fhog_2010`/`fhog_2016`: son factores en
    otra escala (media ~500-1500, no ~1), no la variante "Factor Unidad" a
    nivel de hogar.

  **Limitacion real encontrada**: la ola 3 (2016) NO tiene un peso
  transversal propio en los datos consolidados -- el diccionario 2016 solo
  documenta `fhog_2016` (escala grande, no "Factor Unidad") y
  `fexhog_2010` (el longitudinal anclado a 2010). Para las tablas
  descriptivas de 2016 se usa `fexhog_2010` como mejor aproximacion
  disponible, lo cual subrepresenta a cualquier hogar que haya entrado a la
  muestra despues de 2010 (si los hubiera). Queda documentado, sin
  solucion posible con los datos publicos de la ELCA.

  Implementado en `build_pobreza_desagregaciones.py`: `peso_transversal`
  (fexhog / fexhog_2013 / fexhog_2010 segun ola) para las tablas FGT por
  ola/zona/region/jefe de hogar (ahora se generan en version sin ponderar Y
  ponderada, sufijo `_ponderado.csv`); `peso_longitudinal` (fexhog_2010,
  peso de la OLA FINAL de cada periodo) para las matrices de transicion.

  Efecto en la incidencia oficial (ingreso vs. LP, P0): sin ponderar vs.
  ponderada -- 60.7%->60.4% (2010), 52.4%->48.6% (2013), 42.4%->41.8%
  (2016). La brecha mas grande (2013, ~4 puntos) sugiere que la muestra
  sin ponderar sobrerrepresenta hogares pobres en esa ola. En la matriz de
  transicion, ponderar por `peso_longitudinal` sube "siempre pobre"
  2010->2013 de 43.9% a 42.7% (leve) pero la mueve mas en 2013->2016:
  "siempre pobre" baja de 33.3% a 32.9% y "nunca pobre" sube de 37.0% a
  38.6%. Tablas completas: `outputs/tables/pobreza/*_ponderado.csv` y
  `transicion_*_ponderado_ola*.csv`.

**2. Ingreso excepcional en la variable de transicion -- robustez
agregada.** Los 6 componentes de `ingreso_excepcional` (venta de inmueble,
venta de negocio, herencias, polizas, venta de otros activos, otros
ingresos no clasificados) son eventos retrospectivos de 12 meses
prorrateados como flujo mensual (ver seccion de auditoria de ingreso mas
arriba). Un hogar que vendio una propiedad puede aparecer como "sale de la
pobreza" sin que haya cambiado su capacidad de ingreso sostenible -- riesgo
directo para la variable de interes de la tesis (transiciones). Se agrego
`pobre_sin_excepcional` (clasificacion con `ingreso_total_hogar` menos esos
6 componentes) y se repitieron las matrices de transicion con ella. Efecto:
pequeño a nivel agregado -- "siempre pobre" 2010->2013 pasa de 43.9% a
44.4%, "nunca pobre" de 28.8% a 28.6%; 2013->2016 practicamente igual
(36.5%/33.8% vs. 37.0%/33.3%). Conclusion: el ingreso excepcional NO es un
motor importante de las transiciones observadas a nivel agregado, aunque
puede seguir siendo relevante para hogares individuales especificos. Tabla:
`transicion_categorias_sin_excepcional_ola*.csv`.

**3. Sensibilidad de las transiciones a error de medicion cerca del
umbral.** Clasificar pobre/no-pobre con un corte estricto contra la LP
puede generar "transiciones" que son solo ruido de muestreo cerca del
umbral. Se agrego una tabla de sensibilidad con LP*0.9 y LP*1.1 (banda
+-10%) para las dos matrices de transicion. Resultado: la categoria
"siempre pobre" oscila entre 38.6% (LP-10%) y 49.1% (LP+10%) en 2010->2013
(baseline 43.9%), y entre 27.8% y 38.6% en 2013->2016 (baseline 33.3%) --
una banda de ~10 puntos porcentuales, bastante mayor que el efecto del
ingreso excepcional. Esto es evidencia de que buena parte de la variacion
en los resultados de transicion es sensible a la eleccion exacta del
umbral, algo que conviene discutir explicitamente al interpretar el modelo
predictivo. Tabla: `transicion_sensibilidad_banda_ola*.csv`.

**4. Robustez de `ing_ayudas` en rural 2010.** Se investigo cuanto cambia
la transicion 2010->2013 si se excluye `ingreso_ayudas` de las 3 olas (para
aislar el efecto del hueco de cobertura documentado: la pregunta nunca se
hizo en rural 2010). Resultado: "siempre pobre" sube de 43.9% a 46.8%,
"nunca pobre" baja de 28.8% a 27.2% -- un efecto mas grande que el del
ingreso excepcional (punto 2), consistente con que `ing_ayudas` es una
fuente de ingreso mas extendida (no solo eventos raros). El hueco de
cobertura en rural 2010 no altera esta comparacion (ahi ya era 0 en ambas
versiones), pero remover `ing_ayudas` de las otras filas SI mueve la
clasificacion de forma no trivial. Tabla:
`transicion_categorias_sin_ayudas_ola1_a_2.csv`.

**5. Atricion total del panel.** Ademas de los hogares excluidos por
division (ya documentados en las matrices de transicion: 511 y 1.190
casos), se midio la atricion PURA -- hogares de la ola inicial cuyo
`consecutivo` no aparece en absoluto en la ola siguiente (ni como match 1 a
1 ni como division). Resultado: 11.4% entre 2010 y 2013 (1.124 de 9.853
hogares base), y 11.4% otra vez entre 2013 y 2016 (994 de 8.729 hogares
base -- la base aqui ya excluye los perdidos en el primer tramo).
Consistente entre ambos periodos, pero no se investigo si esta atricion es
aleatoria o esta concentrada en algun perfil de hogar (rural, mas pobre,
mas movil) -- queda como pregunta abierta para la interpretacion de
resultados, dado que atricion no aleatoria en paneles de pobreza tipicamente
sesga a la baja las tasas de "entra en pobreza" (los hogares mas
vulnerables tienden a perderse mas). Tabla: `atricion_panel.csv`.

**6-7. Limitaciones de la fuente, sin correccion posible con codigo:**

  - **Ingreso laboral bruto vs. neto en trabajo independiente.**
    `ing_trabajo`/`ing_trabagr`/`ing_trabnoagr` son autoreportados sin
    desagregar costos del negocio informal -- no hay forma de saber, con
    los datos de la ELCA, si un hogar rural agropecuario reporta ingreso
    bruto o neto de insumos/costos de produccion. Esto no es un problema
    de codigo: la pregunta del cuestionario simplemente no distingue
    ambos conceptos. El DANE tiene la misma limitacion en la GEIH, pero
    vale la pena declararla explicitamente en la tesis como limitacion de
    comparabilidad urbano/rural, dado que la informalidad (y por tanto
    esta ambiguedad) es sistematicamente mayor en la zona rural.

  - **Sesgo de recall en items de gasto de recall largo.** La
    mensualizacion de `build_gasto_hogar.py` normaliza correctamente la
    PERIODICIDAD declarada, pero no corrige el sesgo de recall conocido en
    la literatura de modulos de consumo (ej. Beegle et al. 2012): las
    preguntas de recall anual (ropa, muebles, salud, capitulo IX-C del
    manual) tienden a subestimarse por "telescoping"/olvido frente a
    preguntas de recall corto (alimentos, capitulo mensual). No es
    corregible con los datos disponibles (no hay una submuestra con
    recall corto para comparar), pero explica en parte por que la pobreza
    por gasto (47.6%/40.9%/33.4%, ver seccion de auditoria de gasto mas
    arriba) es sistematicamente mas alta que la pobreza por ingreso -- vale
    la pena decirlo explicitamente en la tesis en vez de dejar que se lea
    como evidencia exclusiva de subreporte de ingreso.
