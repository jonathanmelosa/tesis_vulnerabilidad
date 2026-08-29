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

### 2026-08-08 (cont.) — Revision de la construccion de ingreso, gasto y
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

### 2026-08-08 (cont.) — Metodologia del modelo benchmark (pre-GSV):
particion temporal, especificaciones de ingreso, y criterio de agregacion
de covariables

Definicion de la metodologia del modelo
predictivo benchmark -- el que se comparara despues contra la version que
suma variables derivadas de fotos de Google Street View. 

**1. Poblacion de analisis: modelo de "entrada" condicional, no
pobreza incondicional en t+1.** "Caer en pobreza" solo esta definido
para hogares que NO eran pobres en el periodo base -- un hogar que ya era
pobre en t no puede "entrar" en t+1, esa es la categoria "siempre pobre",
no "entra en pobreza" (ver matrices de transicion, seccion de
`build_pobreza_desagregaciones.py` mas arriba). El modelo se entrena y
evalua exclusivamente sobre el subconjunto de hogares NO pobres en la ola
base, con:

    Y_{i,t+1} = 1[hogar i es pobre en t+1 | no pobre en t]

Esto es exactamente la columna "entra en pobreza" vs. "nunca pobre" de
las matrices de transicion ya construidas.

**2. Particion entre las 3 olas.** Especificacion PRINCIPAL: entrenar con
la transicion 2010->2013 (covariables de 2010, outcome observado en 2013)
y evaluar out-of-sample con la transicion 2013->2016 (covariables de
2013, outcome observado en 2016) -- holdout temporal hacia adelante, que
replica el caso de uso real (solo existe la ola base al momento de
predecir) y coincide con el `Strategy` ya declarado al inicio de este
documento. Validacion interna (ajuste de hiperparametros) debe hacerse
DENTRO del periodo de entrenamiento (2010->2013), sin tocar 2013->2016
hasta la evaluacion final.

Como ROBUSTEZ (mismo patron que los ejercicios de robustez de pobreza ya
documentados), se agregan dos especificaciones secundarias, NO como
resultado principal:
  - **Reversa**: entrenar en 2013->2016, evaluar en 2010->2013. Un
    backtest/placebo, no el escenario de despliegue real; informa si el
    poder predictivo es simetrico.
  - **Pooled con k-fold agrupado por hogar**: combinar ambas transiciones
    (con dummy de periodo) y usar k-fold ALEATORIO PERO AGRUPADO POR
    `consecutivo` (nunca por fila), para no filtrar informacion del mismo
    hogar entre folds de entrenamiento y prueba. Da una estimacion de
    desempeño con mas datos pero sacrifica validez temporal genuina (no
    detecta *drift* macro entre periodos).

Justificacion: el holdout temporal hacia adelante es el unico que
prueba generalizacion real ante drift macro (regimenes economicos
distintos 2010-2013 vs. 2013-2016); es standard en literatura de alerta
temprana / credit scoring por la misma razon. El riesgo de un unico split
es que da un solo punto de desempeño sin banda de incertidumbre sobre la
generalizacion misma -- de ahi el valor complementario del pooled
agrupado como robustez, no como reemplazo.

**3. Ingreso/brecha a la LP como covariable: se corren AMBAS
especificaciones y se comparan.** Modelo A incluye el nivel de
ingreso/consumo per capita del periodo base (o la brecha a la LP) como
predictor -- enfoque estandar de vulnerabilidad a la pobreza (Chaudhuri,
Jalan y Suryahadi, 2002, ya citado en `paper/referencias.bib`): que tan
cerca esta un hogar del umbral es el predictor mas informativo de
cruzarlo. Modelo B excluye ingreso/gasto por completo (solo covariables
no monetarias). Se comparan AUC-ROC/recall/F1 e importancia de variables
(SHAP o permutation importance) entre A y B. Motivo adicional para hacer
esta comparacion: establece el marco de referencia para la pregunta que
vendra despues con las variables de GSV -- si el Modelo A domina casi
por completo al B, la pregunta relevante para las variables geoespaciales
pasa a ser "¿ayudan donde la brecha a la LP ya no distingue bien (hogares
cerca del umbral)?" mas que "¿ayudan en general?".

**4. Alcance de covariables: se busca incluir Personas, Niños,
Comunidades y Choques ademas de Hogar/Ingreso/Gasto ya integrados.**
Ninguno de estos 4 modulos esta hoy conectado a `04_features/`; hace
falta construir scripts de features nuevos (patron
`build_ingreso_hogar.py`) para cada uno.

**5. Criterio para decidir que variables de estos modulos entran al
benchmark, dado que ELCA sigue a la MISMA poblacion en el tiempo pero
2010 no tiene una ola anterior dentro del panel.** Dos ejes distintos:

  - **Eje 1 -- comparabilidad del CONSTRUCTO, no de la columna**: la
    regla de construccion debe producir el mismo concepto medible en cada
    ola base, aunque la pregunta/columna fuente difiera entre olas (igual
    que ya se hizo para ingreso y gasto). No exige que exista la misma
    columna en las 3 olas, exige que el RESULTADO sea equivalente.
  - **Eje 2 -- disponibilidad en el momento del despliegue**: una
    variable DINAMICA (cambio, tendencia, choque acumulado entre olas)
    necesita una ola anterior para calcularse. 2010 es la primera ola del
    panel -- no existe una ola previa contra la cual calcular un cambio
    para la muestra de entrenamiento. Por construccion, NINGUNA variable
    dinamica puede entrar al benchmark principal (estaria vacia para el
    100% de la muestra de entrenamiento 2010->2013).

  De estos dos ejes se deriva la regla de tres partes:
    (a) **Nivel en t** (estado del hogar/persona/niño en la ola base) →
        entra al benchmark si pasa el Eje 1.
    (b) **Cambio o acumulado entre t-1 y t** → nunca entra al benchmark
        principal (Eje 2); solo es calculable para la transicion
        2013->2016 (2010 SI existe como "pasado" respecto a 2013).
    (c) La riqueza longitudinal del panel no se pierde, se traslada a una
        especificacion **"modelo dinamico"** aparte, entrenada y evaluada
        UNICAMENTE dentro de la transicion 2013->2016 (la unica con una
        ola previa real dentro del panel), reportada como comparacion
        adicional, no como sustituto del benchmark principal. Mismo
        patron que los ejercicios de robustez de pobreza ya documentados:
        especificacion base + variantes, nunca se mezclan en un solo
        resultado.

**6. Reglas de agregacion Personas/Niños -> Hogar** (Comunidades no
requiere agregacion, es un join/difusion por `consecutivo_c`+`ola`;
Choques ya esta a nivel de hogar, solo requiere merge una vez resuelto el
hallazgo de cobertura del punto 8):

  - **Variables de conteo/composicion** (edad, sexo, parentesco): conteos
    y proporciones directas (numero de niños <12, razon de dependencia
    demografica), sin funcion de agregacion que elegir.
  - **`any()`** ("al menos un miembro cumple X") cuando un solo caso ya
    cambia la vulnerabilidad del hogar completo (ej. algun miembro con
    discapacidad, algun desempleado) -- mecanismo economico de
    contagio/arrastre, no se diluye.
  - **Proporcion/tasa** cuando importa la intensidad, no la mera
    presencia (ej. % de adultos con primaria incompleta o menos, tasa de
    ocupacion del hogar).
  - **Maximo** para capital humano disponible en el hogar (ej. nivel
    educativo mas alto entre todos los miembros) -- el promedio diluye,
    el maximo capta si existe el recurso.
  - **Valor del jefe de hogar, SIN agregar** para las variables donde la
    literatura de pobreza usa al jefe como proxy estandar (ya se hace
    asi en `build_pobreza_desagregaciones.py` para sexo/edad/educacion
    del jefe). Esto no sustituye las versiones agregadas de todo el
    hogar, las complementa.
  - **Restriccion del denominador ANTES de agregar**: las tasas de
    mercado laboral (ocupacion, cotizacion) se calculan solo sobre
    personas en **edad de trabajar, definida como 15+ años** (estandar
    DANE/GEIH para Colombia, decidido explicitamente sobre las
    alternativas de 12+ y 18+ para poder comparar magnitudes con
    estadisticas oficiales de mercado laboral si hiciera falta). Incluir
    niños en el denominador diluiria artificialmente la tasa.
  - **Missing a nivel persona ≠ "no cumple la condicion"**: si una
    persona tiene NaN en una variable (por filtro de edad en la pregunta
    original, ej. `estado_civil` al 39% en ola 1), debe EXCLUIRSE del
    denominador al agregar, no contarse como si no cumpliera. Riesgo
    tecnico concreto: `Serie.any()`/`Serie.mean()` de pandas tratan NaN
    de forma que puede introducir este error silenciosamente si no se
    filtra explicitamente antes de agregar.

**7. Inventario preliminar de candidatas por modulo** (primera pasada,
sujeta a decision variable por variable):

  - **Personas** -- candidatas limpias (**N**, nivel comparable ya):
    `parentesco`, `sexo`, `edad` (100% cobertura las 3 olas),
    `estado_civil` (39%/100%/100%, la brecha de ola 1 es filtro de edad
    esperado, no problema de calidad), `ahorra` (39% identico en las 3
    olas). Candidatas con **anomalia de cobertura sin explicar todavia**
    (**⚠**, requieren investigacion antes de decidir): `nivel_educ`
    (38%/58%/20%, cae en 2016), `etnia` (59%/66%/14%, cae fuerte en 2016),
    `actividad_ppal` (20%/78%/80%, salto grande 2010->2013),
    `ocupacion` (18%/30%/32%). Candidatas que **no existen en ola 1**
    (**X**, excluidas del benchmark, candidatas solo a la especificacion
    dinamica): `cotizando` (pension, 0% en 2010), `migra_ult3` (0% en
    2010). Caso especial: `desempleado` existe SOLO en 2010 (14%, 0% en
    2013/2016) -- lo opuesto del patron usual; si se quiere "desempleo"
    como concepto comparable en las 3 olas hay que reconstruirlo desde
    `actividad_ppal` (Eje 1: mismo constructo, columna fuente distinta
    por ola), no usar la columna `desempleado` directamente.
  - **Niños** (rango 6-9 años, unico comparable en las 3 olas) --
    candidatas identificadas sin verificar aun cobertura especifica
    dentro del subrango: `talla_cm`/`pesonino` (antropometria ->
    desnutricion cronica/aguda), `come_frutas`/`come_verduras`/
    `come_carnes` (calidad de dieta), `trabajo`/`trabajo_horas` (trabajo
    infantil), `carne_vacunas` (esquema de vacunacion).
  - **Comunidades** -- candidatas con cobertura verificada: `homicidios`
    (91%/100%/100%), `seguridad` -percepcion- (91%/100%/100%),
    `hay_desplazados`/`n_desplazados` (65%/70%/69%). Candidatas sin
    verificar: `puesto_salud`, `obra_prioritaria*` (probablemente Eje 1,
    categorias abiertas). Modulo de desastres (`inu_*`/`ava_*`/`desb_*`)
    es **X** para 2010 si no tiene equivalente esa ola (agregado desde
    2013, ver auditoria de comunidades mas arriba).
  - **Choques -- HALLAZGO CRITICO (RESUELTO 2026-08-09, ver seccion
    "Choques: se resuelve el HALLAZGO CRITICO" mas abajo)**: la base
    consolidada de choques NO tenia una fila por cada hogar del panel. De
    9.853 hogares en 2010 solo 3.406 aparecian en la tabla de choques
    (35%); en 2013, 6.067 de 8.729 (70%); en 2016, 6.160 de 8.086 (76%).
    Se confirmo contra los `.tab` crudos que el archivo fuente SI trae una
    fila (2010) o un bloque de filas con tuvo_choque='No' (2013/2016) por
    cada hogar sin excepcion -- el problema era un filtro aplicado ANTES
    de pivotear en `01_consolidacion_bases_choques.py`, no una limitacion
    de los datos. Corregido: reindexado del pivote contra el universo
    completo de hogares del archivo crudo. `imp_econ_*` sigue siendo **X**
    puro (no existe en 2010, ya documentado). `resp_*` tiene cobertura en
    las 3 olas pero decreciente (73%->52%->50%), candidata **A** con
    revision.

**Proximos pasos**: decidir variable por variable dentro de
Personas primero (el modulo con mas candidatas **N** limpias y mas
directamente ligado a vulnerabilidad -- composicion, educacion,
ocupacion), dejando la verificacion de choques como tarea aparte antes de
decidir si ese modulo entra al benchmark.

### 2026-08-08 (cont.) — HALLAZGO CRITICO: corrupcion U+FFFD en
personas_elca_longitudinal (mucho mas extendida de lo documentado) +
03_clean/02_limpieza_base_personas.py

Al construir la primera variable de composicion del hogar (conteo de
niños/adultos mayores, sexo del jefe) se encontro que `parentesco` mezcla
categorias corruptas y limpias como si fueran distintas (ej. "Jefe(a)"
18.079 filas vs. "Jefe de hogar" 9.853 filas todo referido al mismo
concepto entre olas, y "C�nyuge o compa�era(o)" 13.599 filas vs.
"Cónyuge o compañera(o)" 6.037 -- el mismo concepto, corrupto en una ola y
limpio en otra). Investigando el alcance real:

**El problema es mucho mas grande de lo que documentaba
`06_consolidacion_bases_personas.py`.** Ese script (y la auditoria
retroactiva del 2026-08-08 anterior en este mismo documento) describian la
corrupcion de 2013 como acotada a "texto libre y un par de etiquetas de
opcion abierta". Con datos en mano: el caracter de reemplazo Unicode "�"
(U+FFFD) aparece en **542 columnas** de `personas_elca_longitudinal.parquet`
-- 180 en ola 1 (2010) y 435 en ola 2 (2013), ninguna en ola 3 (2016) --
afectando 333.732 celdas en ola 1 y 680.756 en ola 2. Esto incluye
columnas categoricas analiticamente importantes que se habian dado por
limpias: `parentesco`, `etnia`, `afiliacion`, y muchas mas.

**Causa raiz verificada contra los bytes crudos**: no es un problema de
como se lee el archivo (una codificacion mal elegida se puede corregir
re-leyendo). Se confirmo con `data/interim/raw/elca_2013/UPersonas-csv.tab`
que la secuencia de bytes UTF-8 del caracter de reemplazo (EF BF BD) YA
esta en el archivo que distribuye la ELCA -- es una perdida de informacion
irreversible a nivel de caracter ocurrida antes de que este pipeline
tuviera acceso al dato. La unica forma de recuperar el valor es por
contexto (la misma categoria aparece limpia en otra fila/ola, o en el
diccionario de la encuesta).

**Alcance de la correccion aplicada, deliberadamente acotado**: de las 542
columnas afectadas, 484 tienen vocabulario cerrado (<=25 categorias) y 58
son de alta cardinalidad (texto libre: descripciones de oficio, nombres de
programas sociales, etc. -- estas NO se tocan, mismo criterio que
`ARMONIZACION_ARTICULOS`/`CORRECCIONES_2013` en otros modulos: no vale un
diccionario de miles de palabras para campos que no se usan como
categoria). Dentro del vocabulario cerrado:

  1. **Correccion automatica y reproducible** (no una lista escrita a
     mano): para cada valor corrupto de una columna, se construye un
     patron donde cada "�" es un comodin de 1 caracter, y se busca si hay
     EXACTAMENTE UN valor limpio de esa misma columna (en cualquier ola)
     que calce. Si hay 0 o mas de 1 candidato, no se corrige --
     deliberadamente conservador. Resultado: 322 de 956 valores corruptos
     (34%) resueltos asi, en 241 columnas.
  2. **Regla Si/Sí**: cuando el unico comodin resuelve a {"Si","Sí"}, se
     usa "Sí" (grafia que la ELCA usa consistentemente en el resto del
     cuestionario, igual que `SI_TOKENS` en `build_ingreso_hogar.py`).
     Incluida en el conteo anterior.
  3. **Correccion manual verificada** de las 5 variables priorizadas para
     el modelo benchmark que la correccion automatica no pudo resolver:
     `parentesco` (5 valores; cardinalidad 26, nunca entra al barrido
     automatico), `ocupacion` (11 valores; cardinalidad 31, mismo caso),
     `etnia` (1 valor residual), `nivel_educ` (1 valor residual),
     `actividad_ppal` (6 valores residuales, uno de ellos truncado en la
     fuente original -- se conserva el truncamiento, solo se corrige el
     caracter). 24 valores en total, verificados contra ortografia
     estandar del español. Se valida con `assert` que estas 5 columnas
     queden en 0 valores "�" al terminar el script.

**Lo que queda sin resolver, documentado (no oculto)**: 634 valores
corruptos en 273 columnas de vocabulario cerrado que HOY no son
candidatas para el benchmark no tienen ningun valor limpio equivalente en
ningun lado del dataset para inferir la correccion -- resolverlos
requeriria consultar los diccionarios PDF especificos de cada ola, columna
por columna (un esfuerzo de dias, no justificado hoy para columnas que no
se van a usar). Se guarda el inventario completo (334 columnas: 273
vocabulario cerrado sin match + 58 texto libre no tocado) en
`docs/variable_audit/personas_corrupcion_residual.csv`, para no tener que
redescubrir el problema si alguna de esas columnas se necesita mas
adelante.

**Nota aparte, no U+FFFD**: `sexo` no tiene corrupcion de codificacion,
pero SI tiene un problema de capitalizacion inconsistente entre olas
("Mujer"/"Hombre" en unas filas, "MUJER"/"HOMBRE" en mayuscula en otras --
se cuentan como 4 categorias en vez de 2 si no se normaliza). No se
corrige en este script (es un problema de formato, no de perdida de
informacion); se normaliza en el script de construccion de features de
personas con `.str.title()`, mismo patron que ya usa
`build_pobreza_desagregaciones.py` para `sexo_jefe`.

**Script**: `src/03_clean/02_limpieza_base_personas.py`. Sigue el patron
"vivo" de `01_limpieza_base_hogar.py` (una funcion por problema, documentado
con el mismo nivel de detalle del porque). Input:
`personas_elca_longitudinal.parquet`. Output:
`personas_elca_longitudinal_clean.parquet` (mismas dimensiones,
118.824 filas x 1.359 columnas -- esta limpieza no elimina filas ni
columnas, solo corrige valores). A partir de ahora, cualquier script de
`04_features/` que use el modulo de personas debe leer desde el archivo
`_clean`, no desde el original.

### 2026-08-08 (cont.) — Verificacion: ¿se esta perdiendo informacion
valiosa en las 273 columnas sin resolver?

Pregunta razonable dado que la correccion automatica es deliberadamente
conservadora. Verificado con datos, no solo argumentado:

  - **Magnitud real en celdas, no solo en columnas**: de las 1.014.488
    celdas corruptas originales, la correccion (automatica + manual) ya
    resolvio 526.817 (51.9%). Las ~487.671 restantes estan concentradas en
    las 58 columnas de texto libre (descripciones de oficio, codigos CIIU,
    razones de retiro escolar, costos de hospitalizacion) -- variables que
    de todas formas iban a requerir un tratamiento aparte (categorizacion,
    NLP) antes de poder usarse como predictor, independientemente de la
    corrupcion. La perdida de informacion relevante para variables
    CATEGORICAS de vocabulario cerrado es mucho menor a lo que sugiere el
    conteo de columnas.
  - **Chequeo cruzado contra las candidatas ya identificadas**: se cruzo
    `personas_corrupcion_residual.csv` contra la lista de variables
    candidatas para el benchmark identificada en la seccion anterior de
    metodologia. Resultado: **3 SI aparecian** en el backlog
    (`trabajo_padre`, `trabajo_madre`, `nivel_educ_cursa`) -- exactamente
    el tipo de perdida silenciosa que la pregunta buscaba prevenir. Se
    resolvieron de inmediato (13 valores mas, verificados contra
    ortografia estandar, agregados a
    `CORRECCIONES_MANUALES_PRIORITARIAS`), quedando en 0 valores "�". El
    resto de candidatas ya identificadas (`etnia`, `nivel_educ`,
    `actividad_ppal`, `cotizando`, `ahorra`, `desempleado`,
    `migra_ult3`, `estado_civil`) NO aparecen en el backlog -- ya estaban
    limpias o cubiertas.
  - **Proceso hacia adelante, no solo una revision de una sola vez**: el
    backlog CSV se trata como una LISTA DE VERIFICACION activa, no un
    archivo que se escribe y se olvida. Regla operativa: antes de activar
    cualquier variable nueva (de Personas o de cualquier otro modulo con
    el mismo problema) como candidata del benchmark, se cruza contra el
    reporte de corrupcion residual correspondiente; si aparece, se
    resuelve en ese momento (mismo patron que los 8 casos ya resueltos),
    nunca se usa una columna sin verificar primero. Esto acota el riesgo a
    "columnas que nunca se llegan a usar" -- por definicion, esas no
    afectan al modelo.

### 2026-08-08 (cont.) — Auditoria completa de personas_elca_longitudinal
(las 1.359 columnas, no solo las candidatas ya identificadas)

El usuario pidio estar seguro al 100% de no perder informacion valiosa del
modulo Personas -- mismo estandar de rigor que ya se aplico a
`build_ingreso_hogar.py` (69 columnas auditadas) y `build_gasto_hogar.py`
(88 articulos). Se genero el inventario completo, no solo de las
candidatas identificadas hasta ahora.

**Hallazgo metodologico previo, mas importante que el inventario mismo**:
el criterio correcto de "¿esta columna puede ser feature del benchmark?"
NO es "¿existe en las 3 olas?" sino **"¿existe en ola 1 Y ola 2?"** -- por
el diseño ya acordado (ver seccion de metodologia del benchmark mas
arriba), ola 1 y ola 2 son las UNICAS fuentes de features (bases de
entrenamiento 2010->2013 y prueba 2013->2016); ola 3 solo aporta el
resultado observado, nunca es fuente de predictores. Este criterio reduce
el universo a auditar de 1.359 a 139 columnas reales, sin perder nada:
el resto queda excluido por una razon estructural verificable, no por
omision.

**Clasificacion completa** (umbral de "presente" = >1% no-nulo en la ola),
guardada en `docs/variable_audit/personas_hogar_construccion.csv`:

  - `CANDIDATO_BENCHMARK` (139): presente en ola1 Y ola2.
  - `EXCLUIDA_NO_EN_OLA1` (369): presente en ola2 (y quiza ola3) pero NO en
    ola1 -- no se puede calcular para el hogar en la ola de entrenamiento.
    Preguntas agregadas al cuestionario despues de 2010.
  - `EXCLUIDA_NO_EN_OLA2` (310): presente en ola1 pero NO en ola2 -- no se
    puede calcular para el hogar en la ola de prueba. Preguntas eliminadas
    del cuestionario despues de 2010.
  - `EXCLUIDA_CASI_VACIA` (427): menos de 1% de cobertura en las 3 olas;
    193 de estas son 0% exacto en las 3 olas (columnas del esquema nunca
    pobladas). Las otras 234 son preguntas con filtro muy restrictivo
    (ej. "si recibio beca, de que tipo") -- casi ningun hogar cae en el
    universo aplicable.
  - `EXCLUIDA_SOLO_OLA3` (96): nueva en 2016, irrelevante para el
    benchmark porque ola 3 nunca es fuente de features.
  - `IDENTIFICADOR` (11): llaves ya usadas para el merge (`llave`,
    `consecutivo`, `ola`, etc.), no son variables de contenido.
  - `EXCLUIDA_OTRO_PATRON` (7): presentes en ola1 y ola3 pero no en ola2
    (ej. `pareja_embarazo`, `primer_busca`) -- mismo motivo que
    `EXCLUIDA_NO_EN_OLA2`.

Suma verificada: 139+369+310+427+96+11+7 = 1.359, coincide exactamente con
el total de columnas del archivo -- ninguna columna quedo sin clasificar.

**Verificacion cruzada contra el backlog de corrupcion U+FFFD**: de las
139 candidatas reales, 32 todavia aparecian en
`personas_corrupcion_residual.csv` en el primer cruce -- es decir, el
riesgo que el usuario señalaba (perder informacion sin darse cuenta) SI
se habia materializado parcialmente hasta este punto. Se investigaron y
corrigieron las 32 (24 de vocabulario cerrado con match unico verificado a
mano, 5 mas de `razon_noestudia`, 9 de `razon_dejo_trab`, y 3 casos donde
la "corrupcion" en realidad era un token categorico dentro de una columna
mayormente numerica -- `vr_ganancia`/`vr_salario`="No recibió",
`n_empleados`="50 personas y más", mismo patron que `ZERO_TOKENS` en
`build_ingreso_hogar.py`). Total agregado a
`CORRECCIONES_MANUALES_PRIORITARIAS`: 88 valores en 37 columnas.

**Resultado final, verificado con una segunda pasada del cruce**: de las
139 candidatas, **136 quedan 100% libres de corrupcion**. Las 3 restantes
(`descripcion_ciiu`, `cod_oficio2`, `descrip_oficio`) se dejan
deliberadamente sin resolver -- son texto libre de alta cardinalidad
(208-531 valores distintos corruptos) que necesitarian categorizacion
aparte para ser utilizables como predictor, independientemente de la
corrupcion; no es una perdida de informacion analitica, es una decision de
alcance ya tomada (mismo criterio que excluye las 58 columnas de texto
libre del resto del modulo).

**Conclusion para la pregunta de "¿estoy seguro al 100%?"**: si, en el
sentido verificable -- cada una de las 1.359 columnas tiene un estado
documentado (candidata limpia, candidata excluida por estructura de olas,
o candidata con corrupcion residual explicada), no hay ninguna columna
que se haya ignorado sin razon. Lo que queda pendiente (los ~600 valores
residuales en las ~270 columnas fuera del pool de 139, mas las 3 de texto
libre) esta acotado y documentado, no oculto.

### 2026-08-09 — Segunda ronda de rescate: diccionarios PDF como fuente
independiente de verdad, mas 3 hallazgos adicionales

El usuario pidio explicitamente explorar "de otras maneras" si se podia
rescatar mas informacion, no solo aceptar el estado anterior (136 de 139
candidatas limpias, 634 valores residuales sin resolver en el resto del
modulo). Se probaron 4 enfoques distintos.

**1. Busqueda de variables renombradas entre olas (intento fallido, resultado
honesto).** Se investigo si columnas "solo en ola1" (310) y "no en ola1"
(369) eran en realidad la MISMA pregunta con nombre distinto entre olas
(mismo patron que `ARMONIZACION_ARTICULOS` en gasto). Dos metodos:
  - Coincidencia de nombre normalizado (quitando sufijos `_2010`/`_2013`):
    solo 3 coincidencias, todas factores de expansion (`fexpers`/`fpers`),
    no contenido sustantivo.
  - Similitud de vocabulario (Jaccard) entre los valores de categoria de
    ambos grupos: con columnas binarias (Si/No) el metodo genera ruido
    masivo sin valor (cualquier par de columnas Si/No tiene Jaccard=1.0);
    restringiendo a columnas con >=4 categorias, siguen apareciendo
    coincidencias altas que resultan ser artefactos de escalas genericas
    compartidas (meses del año, rangos de horas) entre preguntas NO
    relacionadas, no renombres reales.
  Conclusion original (CORREGIDA el 2026-08-09, ver seccion "Segunda
  vuelta de busqueda de renombrados" mas abajo): con estos dos metodos
  (coincidencia exacta de nombre normalizado; Jaccard de vocabulario) no
  aparecio nada rescatable. Los dos metodos resultaron ser demasiado
  estrictos/ruidosos para el patron real -- ver correccion.

**2. Diccionarios PDF como fuente de verdad independiente -- el hallazgo
principal de esta ronda.** Se verifico que los diccionarios de la encuesta
(`data/interim/raw/elca_{2010,2013}/{U,R}Personas.pdf`, extraidos con
`pdftotext -layout`) contienen las etiquetas de cada categoria en texto
LIMPIO -- son un documento generado por separado del archivo .tab
exportado, no heredan su corrupcion (confirmado contra `parentesco` en el
diccionario 2013: "Cónyuge o compañero(a)" aparece bien escrito ahi,
aunque el .tab trae "C�nyuge o compa�era(o)"). Metodo: se unieron los 4
diccionarios de las olas corruptas (2010 y 2013, U y R) en un solo texto,
y para cada valor corrupto residual se construyo un patron con limites de
palabra (`\b...\b`, cada "�" = 1 comodin) buscado en ese texto completo;
solo se acepta si hay exactamente una coincidencia.

  **Bug encontrado y corregido durante el desarrollo, antes de aplicarlo**:
  una primera version uso `re.match()` sin anclar el string al final
  (`$`), lo que permitia que un patron corto como "S." emparejara el
  INICIO de cualquier cadena mas larga que empezara con "S" (ej. "S�"
  resolvio incorrectamente a "SECRETARIA DE EDUCACION", una respuesta de
  texto libre no relacionada, en 7 columnas `cred_*`). Se detecto en la
  revision manual de una muestra ANTES de aplicar las correcciones al
  script -- exactamente el tipo de error que la revision manual esta
  para atrapar. Corregido anclando todos los patrones de coincidencia por
  columna/familia con `^...$` (match completo); el metodo de diccionario ya
  usaba `\b...\b` (limites de palabra) sobre el texto completo, que no
  tenia este problema. Tras la correccion, una muestra aleatoria de 25
  valores resueltos se verifico manualmente uno por uno, todos correctos.

  Resultado: 163 valores adicionales resueltos en 77 columnas
  (`CORRECCIONES_DICCIONARIO_PDF` en el script), sin ningun conflicto
  frente al metodo automatico existente. Sumado a los metodos previos:
  **reduccion total de celdas corruptas en el dataset completo: de 51.9% a
  72.7%** (de 1.014.488 originales, quedan 276.784 sin resolver, vs.
  487.671 antes de esta ronda).

**3. Segundo patron de corrupcion, distinto e independiente, encontrado por
casualidad al usar el diccionario como contraste.** El texto literal
"???" (tres signos de interrogación, NO el caracter U+FFFD) aparece en
**108 columnas / 98.360 celdas** de personas_elca_longitudinal, un
problema separado del que este script atiende. Verificado: **100% de esas
celdas estan en ola 3 (2016)**, cero en ola 1 y ola 2. Como ola 3 nunca es
fuente de features en el diseño del benchmark (solo aporta el resultado
observado, ver seccion de metodologia), esto NO bloquea el trabajo actual
-- se documenta como hallazgo real y pendiente, no se corrige en este
script (que se enfoca en U+FFFD). Queda para cuando/si se necesite
contenido de ola 3 mas alla del estado de pobreza.

**4. Revision del bucket "casi vacio" (427 columnas, <1% cobertura) --
confirmado que NO esconde informacion valiosa.** De las 427, 234 tienen
algo de cobertura (no exactamente 0%), pero incluso las de MAYOR cobertura
dentro de ese grupo no superan 0.9% en ninguna ola (ej. `control_prenatal`,
`semana_embarazo`, `curso_yoga`, actividades de uso del tiempo en slots
altos como `hor_aactv6_ma`). Son preguntas con filtro previo muy
restrictivo (ej. solo aplica a mujeres actualmente embarazadas en el
momento de la entrevista) -- la baja cobertura es un reflejo correcto del
diseño del cuestionario, no una perdida de datos ni un error de
consolidacion. No se encontro ninguna variable tematicamente relevante
para pobreza escondida en este bucket con cobertura suficiente para ser
usable.

**Conclusion de esta ronda**: se agotaron los metodos razonables para
rescatar mas informacion sin recurrir a revision manual columna por
columna de las ~270 columnas de vocabulario cerrado que siguen sin
resolver (ninguna de ellas es candidata al benchmark hoy) y las 58 de
texto libre. El pool de 139 candidatas al benchmark se mantiene en 136/139
limpias (las 3 restantes son texto libre de alta cardinalidad, decision de
alcance ya tomada, no un residuo tecnico). Script actualizado:
`src/03_clean/02_limpieza_base_personas.py` (nueva funcion
`aplicar_correcciones_diccionario_pdf`, nueva constante
`CORRECCIONES_DICCIONARIO_PDF`, `documentar_residual` rediseñada para
calcular el remanente sobre el resultado final en vez de listas
intermedias, mas preciso y menos propenso a errores de conteo).

### 2026-08-09 (cont.) — Verificacion sistematica (no muestral) de todas
las correcciones + ampliacion del corpus de diccionarios a los 15
disponibles

El usuario pregunto, con razon, como estar seguro de que no quedan mas
errores silenciosos como el detectado en la ronda anterior (el bug de
anclaje de regex), dado que la revision manual previa fue solo sobre una
muestra de 25 valores. Se implemento una verificacion que cubre el 100%
de las correcciones, no una muestra, usando una invariante matematica
simple derivada de como funciona la corrupcion misma:

**Invariante de verificacion**: cada caracter "�" (U+FFFD) reemplaza
EXACTAMENTE un caracter original (confirmado en la investigacion de la
causa raiz, sesion anterior: "cada � reemplaza exactamente una vocal").
De ahi se derivan dos chequeos automaticos, aplicables a CUALQUIER par
(valor corrupto, valor propuesto como correccion):

  1. **Longitud identica**: `len(corrupto) == len(limpio)`, siempre,
     salvo una excepcion documentada explicitamente en el codigo (el caso
     truncado de `actividad_ppal` donde se completo una palabra cortada en
     la fuente, sumando un caracter a proposito).
  2. **Coincidencia caracter por caracter en toda posicion que NO es
     comodin**: si `corrupto[i] != '�'`, entonces `limpio[i]` debe ser
     IDENTICO a `corrupto[i]`.

  Esta es exactamente la regla que el bug de la ronda anterior violaba
  (`len('S�')=2` vs. `len('SECRETARIA DE EDUCACION')=23` -- se habria
  detectado al instante). Se implemento como funcion `validar_correccion()`
  y se corrio contra las TRES fuentes de correccion del script:
  `CORRECCIONES_MANUALES_PRIORITARIAS` (88 valores),
  `CORRECCIONES_DICCIONARIO_PDF` (170 valores), y la correccion automatica
  en tiempo de ejecucion (322 valores, recalculada llamando directamente a
  `corregir_vocabulario_cerrado_automatico()`). **Resultado: 580 de 580
  correcciones pasan la invariante, cero errores nuevos** -- la unica
  excepcion es la ya documentada y deliberada de `actividad_ppal`.

**Metodo adicional para rescatar mas columnas: ampliar el corpus de
diccionarios de 4 a 15.** La ronda anterior solo usaba los diccionarios de
Personas 2010/2013 (2 olas x 2 zonas). Se amplio a los 15 diccionarios
disponibles del proyecto: Personas + Hogar + RActivos_hogar, las 3 olas
(2010/2013/2016) y ambas zonas -- el razonamiento es que categorias
compartidas entre modulos (niveles educativos, "Sí/No", listas de
"razones") pueden aparecer documentadas en un diccionario de un modulo
distinto al de la columna corrupta. El blob de texto de referencia paso de
346.736 a 895.494 caracteres. Con el corpus ampliado (y ya con la
invariante de validacion integrada directamente en la busqueda, no solo
como chequeo posterior) se resolvieron 7 valores adicionales
(`educ_padre`, `educ_madre`, `pcuida_niveledu`, `descrip_activ3-6`, todos
sobre las categorias "Uno o más años de técnica o tecnológica" y
"Agricultura, ganadería, caza, silvicultura y pesca") -- una ganancia
modesta, que confirma que el corpus de 4 diccionarios ya capturaba la
mayor parte de lo recuperable por este metodo.

**Total acumulado de esta y la ronda anterior**: 322 (automatico) + 170
(diccionario, corpus ampliado) + 88 (manual) = 580 valores corregidos y
verificados al 100%, sin muestreo. Reduccion total de celdas corruptas en
el dataset completo respecto al original: de 1.014.488 a ~276.000 (~73%).

**Diagnostico honesto sobre lo que queda (409 valores en ~187 columnas de
vocabulario cerrado, mas 58 de texto libre)**: se agotaron los metodos
automatizables y verificables disponibles (mismo-columna, familia,
regla Sí/Sí, diccionario contra el corpus completo de 15 documentos). Lo
que resta no tiene ninguna version limpia en ningun documento disponible
del proyecto -- o la pregunta fue exclusiva de la ola/zona corrupta sin
documentacion sobreviviente en otro lado, o es una respuesta de filtro
tan rara que nunca aparece sin corrupcion en ningun contexto. La unica via
adicional seria revision manual pagina por pagina de cada diccionario para
cada columna especifica -- viable si en el futuro se decide incorporar
alguna de esas columnas al modelo (mismo criterio ya establecido: se
resuelve en el momento en que una columna se activa como candidata, no
antes). **Importante: ninguna de las 139 candidatas al benchmark esta en
este residual** -- las 3 unicas candidatas afectadas (texto libre de alta
cardinalidad) ya estaban excluidas por decision de alcance, no por falta
de intento. Seguir persiguiendo este residual no tiene impacto en el
modelo benchmark que se esta construyendo; se documenta como backlog
abierto y se retoma la construccion de features.

### 2026-08-09 (cont.) — Primer bloque de features del benchmark:
composicion del hogar (build_personas_hogar.py)

Primer script de construccion de covariables desde el modulo Personas,
sobre `personas_elca_longitudinal_clean.parquet` (no el original -- usar
el sin limpiar contaria "Jefe(a)" y "Jefe de hogar" como personas
distintas entre olas). Cubre el grupo de variables mas directamente ligado
a vulnerabilidad y sin ninguna ambiguedad de cobertura (ver inventario de
139 candidatas, seccion anterior).

**Variables construidas**: `n_ninos_5` (edad<6), `n_ninos_12` (edad<12),
`n_adultos_mayores` (edad>=65), `razon_dependencia_demografica`
((menores 15 + mayores 65) / personas 15-64, NaN si el denominador es 0),
`pct_mujeres_hogar`, `sexo_jefe`/`edad_jefe` (tomados directo del jefe,
identificado con el mismo `JEFE_TOKENS` que ya usa
`build_pobreza_desagregaciones.py`), y `tiene_conyuge_jefe` (1 si algun
miembro del sub-hogar tiene `parentesco`="Cónyuge o compañera(o)").

**Nombrado deliberado de `tiene_conyuge_jefe`, no `hogar_monoparental`**:
la ausencia de conyuge del jefe no implica por si sola que haya hijos en
el hogar: se nombra por lo que mide literalmente, y se deja para el script
de modelado combinarla con `n_ninos_5`/`n_ninos_12` si se quiere una
definicion mas estricta de monoparentalidad.

`tamano_hogar` NO se reconstruye aqui: ya existe como `t_personas` en
`hogar_elca_longitudinal_clean.parquet`, validado que coincide con el
conteo real de filas en Personas (diferencia maxima 3 personas en total).

**Problema de performance encontrado y corregido antes de terminar**: la
primera version uso `groupby().apply()` con funciones lambda de Python (6
llamadas separadas) para las variables de conteo -- tardo mas de 100
segundos para 118.824 filas. Se reescribio vectorizado (precomputar
columnas indicadoras 0/1 por persona, UN solo `groupby().sum()` para
todas juntas): mismo resultado exacto, 3.8 segundos (~30x mas rapido). Con
139 variables candidatas por construir, el patron `groupby().apply()` con
lambdas escalaria mal -- se adopta el patron vectorizado como estandar
para el resto de scripts de este modulo.

**Validacion del resultado**: 27.932 filas (coincide exactamente con el
universo de `hogar_elca_longitudinal_clean.parquet`), 0 hogares sin jefe
identificado (`sexo_jefe` sin nulos, solo categorias Hombre/Mujer), 0
nulos en `pct_mujeres_hogar`, 396 nulos en `razon_dependencia_demografica`
(hogares sin nadie en edad de trabajar -- comportamiento esperado, no un
error).

Output: `data/processed/personas_hogar_elca_longitudinal.parquet`.

### 2026-08-09 (cont.) — Segundo bloque de features del benchmark:
educacion y mercado laboral (build_educacion_ocupacion_hogar.py)

Segundo script de construccion de covariables desde Personas. Antes de
construir las variables se encontraron y corrigieron 2 problemas de
calidad de datos nuevos, NO relacionados con la corrupcion U+FFFD ya
documentada:

  1. **Duplicacion de categoria por doble espacio interno**: `actividad_ppal`
     tiene la MISMA respuesta contada como dos categorias por un doble
     espacio en el texto fuente (ej. ola 2: "...por el que recibe
     ingresos" con un espacio, 173 casos, vs. "...por  el que recibe
     ingresos" con dos espacios, 118 casos). Se corrige con
     `normalizar_espacios()` (colapsar espacios multiples) ANTES de
     cualquier mapeo categorico -- de lo contrario el indicador `ocupado`
     quedaria mal clasificado para esos 118 casos.
  2. **"Sí"/"Si" sin unificar** en `lee_escribe` y `estudia` (mismo patron
     que ya se documento para otras variables): se normaliza a {"Sí",
     "No"}.

**Escala ordinal de `nivel_educ`** (0-9, propia del proyecto, no oficial
del DANE): se unifican "Básica secundaria (6 a 13)" y "Básica secundaria y
media (6 a 13)" como el mismo nivel (ambas cubren los mismos grados 6-13,
la diferencia es solo de redaccion entre olas). Escala completa en el
docstring del script.

**Regrupamiento de `ocupacion` a 7 categorias**: los datos crudos tienen
20 categorias que mezclan niveles de detalle distintos entre olas (ej.
"Asalariado de empresa particular" sin desagregar tipo de contrato en
unas filas, desagregado en "...con contrato a termino indefinido/fijo" en
otras). Se regrupa preservando el gradiente de informalidad: Patrón o
empleador > Asalariado > Cuenta propia > Jornalero > Empleado doméstico >
Sin remuneración > Otro.

**Definicion de `ocupado` y su limitacion documentada**: 1 si la persona
trabajo >=1 hora (incluye trabajador familiar sin remuneracion, criterio
OIT/DANE) o tiene empleo del que esta temporalmente ausente; 0 para
"Ninguna de las anteriores" e "incapacitado permanente"; NaN para "no
informa". La categoria residual "Ninguna de las anteriores" MEZCLA
personas inactivas (estudiantes, hogar, jubilados) con posibles
desocupados buscando trabajo -- esta variable de la ELCA no permite
separar desocupado de inactivo con confianza, asi que este script
construye solo el indicador binario ocupado/no-ocupado, no una tasa de
desempleo propiamente dicha. Documentado como limitacion de los datos, no
resuelto con una distincion forzada.

**Hallazgo adicional en ola 3, no relacionado con U+FFFD, no bloquea el
benchmark**: al validar `tasa_ocupacion_hogar` por ola, ola 3 mostraba
13.1% vs. 69.5%/65.1% en olas 1/2 -- una caida obviamente espuria.
Investigado: `actividad_ppal` en ola 3 tiene DOS variantes de texto que no
calzaban con el conjunto `ACTIVIDAD_OCUPADO` construido sobre olas 1/2:
una truncada sin relacion con U+FFFD ("...en una actividad que le gene",
cortada antes de "generó", 8.088 casos) y otra con el "???" literal ya
documentado como exclusivo de ola 3 (5.907 casos, ver seccion anterior
"HALLAZGO CRITICO"). Se agregaron ambas variantes a `ACTIVIDAD_OCUPADO`
-- tras el ajuste, `tasa_ocupacion_hogar` en ola 3 sube a 65.1%, consistente
con las otras olas. Esto NO afecta al benchmark (ola 3 nunca es fuente de
features), se corrigio solo para que la variable no quede erroneamente
sub-estimada si se usa ola 3 para otra cosa (validacion, EDA, o una
especificacion dinamica futura).

**Validacion del resultado** (ola 1 y 2, las que importan para el
benchmark): `nivel_educ_jefe` sin nulos; los 545 nulos de
`nivel_educ_ordinal_jefe` y los 4.602 de `ocupado_jefe` se verificaron
como missingness genuina de la fuente (valor crudo `None`/"No informa",
no un error de mapeo). `categoria_ocupacional_jefe`/`horas_trabajo_jefe`
en NaN para jefes no ocupados, por diseño (no aplica). `tasa_asistencia_
escolar` en NaN para hogares sin ningun miembro de 6 a 17 años, por diseño.

Output: `data/processed/educacion_ocupacion_hogar_elca_longitudinal.parquet`.

### 2026-08-09 (cont.) — Tercer bloque de features del benchmark: salud y
discapacidad (build_salud_discapacidad_hogar.py) -- hallazgo de
comparabilidad que la auditoria de cobertura original no detectaba

Motivacion: los choques de salud son un mecanismo clasico de entrada a la
pobreza en la literatura de vulnerabilidad (Dercon, 2002, ya citado en
`paper/referencias.bib`). Tercer bloque de Personas (los anteriores:
composicion del hogar, educacion/ocupacion).

**Hallazgo importante: la pregunta de discapacidad cronica
(`dif_moverse/banarse/calle/aprender`, `ceguera/sordera/mudez`) NO se hizo
a la misma poblacion en las 3 olas**, un problema de comparabilidad que
la auditoria de cobertura de la seccion "Auditoria completa de
personas_elca_longitudinal" NO detecto porque esa auditoria solo medía
TASA de no-nulo (>1% = "presente en la ola"), no A QUIEN se le pregunta.
Verificado por edad: en ola 1 (2010) estas 7 preguntas se hicieron
EXCLUSIVAMENTE a niños de 0 a 10 años (edad maxima entre respuestas
no-nulas = 10.0 exacto -- es un sub-modulo de niños integrado en
Personas). En ola 2/3 (2013/2016) la cobertura se amplio a todas las
edades (0 a 97 años). Dos consecuencias practicas:
  - Un indicador a nivel de JEFE de hogar seria 100% NaN en ola 1 (el
    jefe nunca tiene 0-10 años) -- estructuralmente inutilizable como
    feature de entrenamiento. NO se construyo `discapacidad_jefe`.
  - Un indicador "algun miembro del hogar" mezclaria poblaciones distintas
    por ola (solo niños en 2010 vs. todos en 2013/2016), violando el Eje
    1 de comparabilidad del constructo (ver metodologia del benchmark).
  Solucion, mismo principio ya usado para el modulo de niños (restriccion
  a 6-9 años, unico rango comparable en las 3 olas): se restringe el
  indicador a **niños de 0 a 10 años**, produciendo
  `pct_ninos_con_discapacidad` en vez de un indicador de discapacidad del
  hogar en general. Verificado consistente entre olas tras el ajuste:
  2.5%/2.7%/2.9% en olas 1/2/3 (antes del ajuste, sin filtrar por edad, la
  comparacion habria sido invalida por mezclar poblaciones).

**Implicacion metodologica mas amplia**: la auditoria de cobertura de
`docs/variable_audit/personas_hogar_construccion.csv` (basada en tasa de
no-nulo por ola) es una condicion NECESARIA pero NO SUFICIENTE para
comparabilidad -- puede pasar por alto cambios en LA POBLACION
evaluada dentro de una columna que aparenta estar "presente" en las 3
olas. Al construir cada bloque de variables restante hay que verificar
tambien la distribucion de edad (u otro filtro poblacional relevante) de
quienes tienen dato valido, no solo la tasa de cobertura agregada.

**Variables sin problemas de comparabilidad en este bloque** (verificado
por edad, cobertura similar en ola1/ola2, min/max de edad consistentes):
`ev_enfe/acci/odon/ciru` (eventos de salud recientes) y `hospitalizado`
-> `tuvo_evento_salud_jefe`, `n_eventos_salud_hogar`,
`tuvo_hospitalizacion_hogar`; `afiliacion`/`prepagada` ->
`tasa_afiliacion_salud_hogar`, `tiene_prepagada_hogar` (esta ultima es
marcador de MAYOR bienestar, no de vulnerabilidad -- medicina prepagada
privada adicional a la afiliacion basica).

Output: `data/processed/salud_discapacidad_hogar_elca_longitudinal.parquet`.

### 2026-08-09 (cont.) — Cuarto bloque de features del benchmark: ahorro y
capital social (build_ahorro_capital_social_hogar.py)

Cuarto bloque de Personas (los anteriores: composicion del hogar,
educacion/ocupacion, salud/discapacidad). A diferencia del bloque de
salud, se verifico primero la distribucion de edad de cobertura antes de
construir nada (leccion del hallazgo de discapacidad) -- `ahorra` y las
10 `org_*` tienen cobertura consistente entre olas (~39%, edades 13-94 en
ola 1 / 15-97 en ola 2), sin el problema de poblacion distinta por ola.

**`ahorra` tiene mas categorias que Sí/No**: ademas de "Sí"/"Si" (sin
tilde) y "No", incluye "No, no recibe ingresos" y "No recibe ingresos"
(variantes de redaccion del mismo motivo entre olas). Ambas variantes de
"no recibe ingresos" se tratan como "No" para el indicador binario -- el
hogar no esta ahorrando, independientemente del motivo.

**8 sub-variables de motivo de ahorro EXCLUIDAS por decision de alcance**:
`ahorro_futuro/educ/casa/carro/otros_act/recre/montar/otro` tienen solo
5.7% de cobertura (son sub-preguntas filtradas solo para quienes ya
respondieron "Sí" a `ahorra`, un subconjunto ya pequeño) -- demasiado
dispersas para aportar señal util a nivel de hogar, la mayoria de hogares
quedaria en NaN. Documentado como decision, no un descarte accidental.

**Organizaciones sociales**: 10 tipos de participacion comparables en
ambas olas (`org_jac/caridad/comunitaria/religiosa/iestado/etnica/culdep/
educ/mamb/otra`). `org_ninguna` (solo ola 1) y `org_sindicato`/
`org_agremia` (solo ola 2) se excluyen por no existir en ambas olas del
benchmark (Eje 1 de comparabilidad); `org_otra_cual` es texto libre, no se
usa.

**Variables construidas**: `ahorra_jefe`, `participa_organizacion_jefe`,
`n_tipos_organizacion_jefe` (0-10, intensidad de participacion civica) a
nivel de jefe; `tasa_ahorro_hogar`, `pct_hogar_participa_organizacion`
agregadas sobre adultos 15+ del hogar.

**Validacion**: nulos minimos (24-25 de 19.114 filas ola1+2, ~0.13%) y
valores consistentes entre olas sin anomalias (`ahorra_jefe`: 15.5%/17.3%/
20.6% en olas 1/2/3; `pct_hogar_participa_organizacion`: 18.7%/24.8%/
19.6%) -- a diferencia del hallazgo de ocupacion en ola 3 (seccion
anterior), aqui no hubo caidas espurias que investigar.

Output: `data/processed/ahorro_capital_social_hogar_elca_longitudinal.parquet`.

### 2026-08-09 (cont.) — Decision confirmada: variables de ingreso
individual excluidas del benchmark

Se pregunto explicitamente si incluir `vr_salario`, `vr_ganancia`,
`vr_rec_pension`, `vr_rec_arriendos`, `cr_arriendos`, `cr_dividendos`
(ingreso/activos financieros a nivel de PERSONA). Decision: **excluidas**.
Miden esencialmente lo mismo que `ingreso_hogar` (construido desde el
modulo Hogar, ya incluido como covariable principal via la brecha a la
LP -- ver metodologia del benchmark) pero a nivel de persona -- incluirlas
arriesgaria doble conteo/colinealidad severa con la variable de ingreso
que ya es, por diseño, el predictor mas informativo del modelo (enfoque
Chaudhuri et al. 2002). No se construye ningun script para estas 6
columnas.

### 2026-08-09 (cont.) — Regla de alcance para el resto de Personas:
cobertura minima 10% por ola

A partir de este punto, cualquier candidata de las 139 con cobertura POR
OLA menor al 10% (verificada consistente entre ola 1 y ola 2, no una
caida espuria como la de ocupacion en ola 3) se EXCLUYE del benchmark sin
construir un script para ella -- mismo criterio ya aplicado de facto a las
8 sub-variables de motivo de ahorro (5.7% de cobertura,
build_ahorro_capital_social_hogar.py). Un hogar con >90% de NaN en una
columna no aporta señal util a un modelo de prediccion y agrega
complejidad sin beneficio. Se aplica bloque por bloque; el inventario
completo de columnas descartadas por esta regla se documenta al cerrar
cada bloque.

### 2026-08-09 (cont.) — Quinto bloque de features del benchmark:
educacion y ocupacion, extension (build_educacion_ocupacion_hogar_ext.py)
-- 2 bugs encontrados y corregidos en la validacion

Extension de `build_educacion_ocupacion_hogar.py` con variables de menor
cobertura pero alto valor tematico.

**Hallazgo: `poc`/`pin`/`pds` son la clasificacion OIT que faltaba.** Se
habia documentado en el bloque 2 que `actividad_ppal` no permite separar
desocupado de inactivo con confianza. Al revisar el diccionario de la
encuesta se encontro que `poc`/`pin`/`pds` (que parecian 3 preguntas de
baja cobertura, 1.4%-22.7% cada una por separado) son en realidad 3
CATEGORIAS MUTUAMENTE EXCLUYENTES de una clasificacion laboral ya
construida por el equipo ELCA para "personas de seguimiento" (miembros
originales del panel): Ocupada/Inactiva/Desocupada. Combinadas, cobertura
real 20.3% (ola 1) / 40.4% (ola 2). Se construye
`categoria_laboral_oit_jefe`, complementaria (no sustituta) de
`ocupado_jefe` del bloque 2.

**`razon_noestudia`** (38.1%/58.0% cobertura, buena): la categoria mas
frecuente es "Falta de dinero" (12.715 casos) -- señal directa de
vulnerabilidad. Se construye `pct_ninos_no_estudia_razon_economica`
(niños 6-17 que no estudian por "Falta de dinero" o "Necesita trabajar",
sobre el total de niños 6-17 que no estudian con razon informada). Typo
menor corregido: "Por enfermdad" (sin la segunda "e") vs. "Por
enfermedad", tratadas como la misma categoria.

**Bug 1 encontrado en validacion: `registro_mercantil` NO es Sí/No
simple.** La primera version del script aplico `normalizar_si_no()` (la
misma funcion generica usada en otros bloques) a esta columna, dejando el
**100% de las filas en NaN** -- se detecto en la revision de nulos por
columna (rutina ya establecida en este proyecto: nunca asumir que un
resultado esta bien sin revisar la tabla de nulos). La columna en
realidad tiene 4 categorias reales: "No lo necesita", "Lo necesita pero
no lo tiene", "Sí lo tiene y lo renovó este año", "Sí lo tiene pero no lo
renovó este año" (con duplicacion "Si"/"Sí" adicional). Se corrigio con
un mapeo propio (`REGISTRO_MERCANTIL_MAPA`) que colapsa a 4 categorias
limpias. Tras corregir, cobertura en jefes = 19.0%, consistente con lo
esperado.

**Bug 2 (en realidad un problema de diseño real, no un bug de codigo):
`n_empleados` cambia de formato entre olas.** Ola 1 pide el numero EXACTO
de empleados (valores "2.0" a "99.0", el minimo observado es 2, no hay
"1"); ola 2 pide directamente el TRAMO ("Trabaja solo", "De 2 a 5
personas", ..., "50 personas y más"). Sin armonizar, un modelo entrenado
en ola 1 (numeros) no generalizaria a ola 2 (categorias) o viceversa. Se
arma ola 1 a los mismos tramos de ola 2 (`TRAMOS_N_EMPLEADOS`), mismo
patron que `ARMONIZACION_ARTICULOS` en gasto. Nota de cautela sin
resolver: tras armonizar, la categoria "50 personas y más" tiene sole 2
casos en ola 1 vs. 933 en ola 2 -- una diferencia demasiado grande para
explicarse solo por azar muestral. Posible causa (no confirmada): la
pregunta podria referirse al tamaño de la empresa donde trabaja el jefe
(aplicable a asalariados tambien) en una ola, y solo al negocio propio
(aplicable solo a empleadores) en la otra -- requeriria leer el
cuestionario exacto de cada ola para confirmar. Se deja documentado como
limitacion; la variable se conserva porque la mayoria de las categorias
(2-49 personas) son consistentes en magnitud entre olas.

**Variables construidas**: `categoria_laboral_oit_jefe`,
`grado_educ_jefe`, `medio_consiguio_jefe` (con la misma correccion de
doble espacio ya aplicada a `actividad_ppal`), `registro_mercantil_jefe`,
`n_empleados_jefe` (a nivel de jefe, cobertura del hogar completo
demasiado baja para agregar con confianza); `pct_ninos_no_estudia_razon_
economica` a nivel de hogar.

**Excluidas de este bloque por cobertura <10% en al menos una ola**
(regla de alcance de la seccion anterior): `actividad_principal`,
`ocupacion2`, `meses_trabaja`, `meses_ganancia`, `anos_trabaja`,
`anos_superior`, `disponibilidad`, `ofertas_empleo`, `t_busco_trab`,
`t_bustrab_a`, `t_bustrab_m`, `razon_dejo_trab`, `razon_dejo_bus`,
`razon_tiene_negocio`, `grado_educ_cursa`, `nivel_educ_cursa`,
`tipo_estab`, `consulta_libros`, `prestamo_libros`, `edad_dejoestudio`.
`diligencias4`/`diligencias12` se excluyen ademas por un salto de
cobertura entre olas (6.1%->28.0%, 4.8%->25.3%) que no se investigo a
fondo -- mismo tipo de señal de alerta que la discapacidad y la ocupacion
en ola 3, pendiente de verificar si se necesita esta variable en el
futuro.

Output: `data/processed/educacion_ocupacion_hogar_ext_elca_longitudinal.parquet`.

### 2026-08-09 (cont.) — Sexto bloque de features del benchmark: becas y
subsidios escolares (build_becas_subsidios_hogar.py)

Antes de construir se verifico la distribucion de edad de cobertura
(disciplina ya establecida tras el hallazgo de discapacidad): mediana 8-9
años consistente entre ola 1 y ola 2 en las 9 columnas de este bloque, con
una cola de casos hasta 66-68 años (educacion de adultos, minoritaria) --
se restringe el calculo a edad 4-20 para no diluir el indicador con ese
grupo pequeño y de tamaño no necesariamente comparable entre olas.

**Regla de cobertura >=10%** (ver seccion anterior) descarta 11 de las 20
candidatas originales del tema: las 7 `beca_*` (1.9% en ola 1),
`finan_educ_pago`, `rec_alimentos_pago`, `rec_vivienda_pago` (7.5% en ola
2), `rec_otros_pago` (7.3%/7.5%).

**`recibio_beca` tiene mas categorias que Sí/No**: distingue "Sí,
subsidio", "Sí, beca" y "Sí, beca y subsidio" (esta ultima encontrada solo
en ola 3, no afecta el benchmark) ademas de "No recibió ninguno". Se
construye un indicador binario (cualquier "Sí").

**Mismo hallazgo de ola 3 que en bloques anteriores, no bloquea el
benchmark**: `pct_ninos_recibio_beca_subsidio` salio inicialmente en
100.0% para ola 3 -- investigado, causa exactamente la ya documentada
("No recibi??? ninguno", la variante con "???" literal exclusiva de ola 3)
sin match en el set de valores "No" del script. Agregada la variante
corrupta al conjunto `RECIBIO_BECA_NO`; ola 3 baja a 27.9%, consistente
con las otras olas (16.5%/20.0%).

**Variables construidas** (nivel hogar, restringido a edad 4-20):
`pct_ninos_recibio_beca_subsidio`, `pct_ninos_credito_estudiar`,
`pct_ninos_apoyo_alimentario_escolar` (almuerzo/beca alimentaria/
desayuno/refrigerio combinados con OR), `pct_ninos_apoyo_material_escolar`
(fotocopias/transporte/uniformes combinados con OR).

**Validacion**: apoyo alimentario escolar 55-73% (programas de
alimentacion escolar son ampliamente extendidos en Colombia, magnitud
plausible); credito_estudiar practicamente inexistente para esta
poblacion (0.09%-0.35%, esperable -- creditos educativos son raros para
niños); nulos ~57% de las filas ola1+2 (hogares sin niños en el rango de
edad, esperado por diseño).

Output: `data/processed/becas_subsidios_hogar_elca_longitudinal.parquet`.

### 2026-08-09 (cont.) — Septimo bloque de features del benchmark: salud
preventiva y afiliacion infantil (build_salud_discapacidad_hogar_ext.py)

Extension del bloque de salud. Regla de cobertura >=10% descarta
`hospital_veces`/`ultima_hosp`/`ult_hosp_dias` (3.6%-5.6%),
`dias_noasistio` (7.8% en ola 1), `beneficiario_orden` (7.8% en ola 2).
Pasan: `prev_med/odo/opto/malter` (59.3%/65.5%, todas las edades),
`prev_pediatra` y `beneficiario_sss` (20.1%/26.0%, restringidas a niños
0-14 por diseño de la pregunta -- verificado antes de construir, mismo
principio que discapacidad), `prev_planif` (39.2%/52.7%, restringida a
13+ años).

**Variables construidas**: `tasa_control_preventivo_hogar` (control
medico/odontologico/optico/nutricional, cualquier miembro del hogar),
`pct_ninos_control_pediatrico` (0-14 años), `tasa_planificacion_familiar`
(13+ años), `pct_ninos_beneficiario_sss` (0-14 años).

**Validacion**: control preventivo 66-80% (crecimiento consistente con
expansion del sistema de salud colombiano entre 2010-2016), beneficiario
SSS en niños 91-97% (coincide con cobertura casi universal del regimen
subsidiado de salud para menores en Colombia), planificacion familiar
7-10% (magnitud razonable). Sin anomalias que investigar.

### 2026-08-09 (cont.) — Octavo bloque de features del benchmark: demografia
y estructura familiar extendida (build_personas_hogar_ext.py)

Extension del bloque de composicion del hogar (Bloque 1). Cubre
`estado_civil`, `etnia`, `padre_vive`, `madre_vive`, `vive_conyuge`,
`mes_unionm`/`ano_unionm`/`edad_unionm`, `tareas`, `id_dpto_nac`/
`id_mpio_nac`.

**`estado_civil`: cobertura 100% en ola 2 es enganosa.** Salta de 39.2%
(ola 1, edad minima 13) a 100.0% (ola 2, edad minima 0) -- investigado
antes de construir, misma disciplina que en el hallazgo de discapacidad
(Bloque 3): en ola 2 los menores de 10 años quedan codificados como
"Soltero(a)" por defecto (7,272 casos verificados), una respuesta
administrativa, no una evaluacion real. Se construye `estado_civil_jefe`
directamente sobre el jefe de hogar (siempre adulto por definicion), lo
que evita el problema sin necesidad de restringir edad explicitamente.

**`padre_vive`/`madre_vive`: dos formas de decir "no" segun la ola.**
Ademas de "Sí"/"No" existen las variantes "Falleció"/"Ya falleció"
(`padre_vive` value_counts: No=30,078, Ya falleció=18,758, Si=14,635,
Sí=13,195, Falleció=5,874) -- se normalizan como equivalentes ("No"
y las variantes de fallecimiento => padre/madre no vive).

**`tareas`: "del hogar" vs. "de otro hogar" son conceptos distintos.**
Categorias: "Sí, del hogar" (quehaceres domesticos normales, 9,961
casos), "No" (1,505), "Sí, de otro hogar" (964) -- la ultima es trabajo
domestico pagado en la vivienda de OTRA familia, una señal de
vulnerabilidad mucho mas fuerte que hacer quehaceres en el propio hogar.
Se construye la variable especificamente sobre esa categoria
(`pct_ninos_trabaja_otro_hogar`), no sobre "cualquier tarea", para no
mezclar dos fenomenos de gravedad muy distinta.

**Regla de cobertura >=10%** excluye `mes_unionm`/`ano_unionm` (6.9% en
ola 1). Se descarta `vive_conyuge` (33.2%/38.4%, pasa la regla) por
redundancia con `tiene_conyuge_jefe` ya construido en el Bloque 1. Se
descartan `id_dpto_nac`/`id_mpio_nac` (19.9%->100% entre olas) por el
mismo riesgo de identificador falso ya documentado para `id_dpto`/
`id_mpio` a nivel de hogar (ver seccion "por que la LP se queda en 2
dominios") -- no se verifico si el identificador de lugar de NACIMIENTO
tiene el mismo problema, y no se construye sobre una base no verificada.

**Variables construidas**: nivel jefe (directo, sin riesgo de mezclar
poblacion infantil) — `estado_civil_jefe`, `etnia_jefe`,
`edad_union_jefe`; nivel hogar (restringido a niños 0-17) —
`pct_ninos_padre_vivo`, `pct_ninos_madre_viva`; nivel hogar (restringido
a niños 0-14, cobertura real observada de `tareas`) —
`pct_ninos_trabaja_otro_hogar`.

**Validacion**: `estado_civil_jefe` con 100% cobertura en las 3 olas
(consistente con la restriccion a jefe adulto); distribucion ola 1
plausible (unión libre 34%, casado 27%, separado 10%, soltero 8%, viudo
4%). `pct_ninos_trabaja_otro_hogar` sube de 4.9% (ola 1) a 9.1%-11.2%
(olas 2-3), tendencia consistente con mayor reporte de trabajo domestico
infantil pagado en periodos posteriores. `edad_union_jefe` sube de 21.3
a 27.2-27.3 años entre olas -- no se investigo a fondo (posible cambio de
composicion de jefes de hogar entre olas o cambio de pareja), se deja
como caveat sin resolver, igual que el caso de "50 personas y mas" en
`n_empleados` (Bloque 5).

### 2026-08-09 (cont.) — Noveno bloque de features del benchmark:
participacion civica y politica (build_participacion_civica_hogar.py)

Ultimo bloque de features de Personas. Cubre `mov_parpol`, `junta_edif`,
`asoc_vigil` -- las unicas 3 de las 7 candidatas de este tema presentes
en el listado de 139 CANDIDATO_BENCHMARK (`jov_org_social`, `participa`,
`porcentaje_participacion`, `beca_accionsocial` ya estaban excluidas por
no estar en ola 1 o por cobertura casi vacia, verificado contra
`personas_hogar_construccion.csv`).

`asoc_vigil`/`mov_parpol` tienen cobertura pareja (39.2%/39.5%/39.1%,
edad minima 13 en las 3 olas). `junta_edif` cae de 39.2% (ola 1) a
20.3%-20.6% (olas 2-3) con el mismo rango de edad -- no se identifico la
causa exacta (posible cambio de filtro de pregunta entre rondas); se
deja como observacion sin resolver, sin bloquear la construccion porque
20.3% aun supera el umbral minimo de 10%.

Las 3 variables tienen prevalencia de "Sí" muy baja (0.1%-0.4% de quienes
responden) -- se combinan con OR en un unico indicador de participacion
civica, mismo criterio que `apoyo_alimentario_escolar`/
`apoyo_material_escolar` (Bloque 6), para evitar 3 indicadores
individualmente casi degenerados.

**Variable construida**: `tasa_participacion_civica_hogar` (nivel hogar,
restringido a miembros de 13+ años) -- proporcion de adultos del hogar
que participan en junta de edificio, asociacion de vigilancia barrial o
movimiento politico.

**Validacion**: tasa 1.3%-1.7% entre olas, consistente con la baja
prevalencia observada a nivel individual. Sin anomalias.

Con este bloque se completan los 9 bloques de features de Personas.

### 2026-08-09 (cont.) — Verificacion final: las 139 candidatas quedan
todas contabilizadas

Se cruzaron programaticamente las 139 columnas `CANDIDATO_BENCHMARK` de
`personas_hogar_construccion.csv` contra las columnas efectivamente
usadas en los 9 scripts de `src/04_features/` y contra las menciones
explicitas en este documento. 122 de 139 ya estaban cubiertas (usadas
para construir una variable, o mencionadas por nombre o por familia
`prefijo_*` al documentar una exclusion por cobertura <10%). Las 17
restantes no tenian mencion individual -- se verifico su cobertura y se
documentan aqui para cerrar la auditoria sin dejar nada implicito:

- `beca_cajacom`, `beca_emp_pri`, `beca_emp_pub`, `beca_misma_ins`,
  `beca_otro`, `beca_prg_gob`: forman parte de la familia "7 `beca_*`"
  ya excluida en el Bloque 6 (1.9% cobertura en ola 1, junto con
  `beca_accionsocial`) -- confirmado, sin cambios.
- `edad_tenia` (18.4%/2.0%/2.1%) y `lugar_ahorra`/`vr_ahorro`
  (5.3%-7.5%, misma familia de las 8 sub-razones de `ahorro_*` ya
  excluidas en el Bloque 4): **EXCLUIDAS**, cobertura <10% en al menos
  una ola.
- `edad_meses` (100% cobertura): **EXCLUIDA** por redundancia -- es la
  misma informacion que `edad` (años) en otra unidad, ya usada de forma
  transversal en los 9 bloques para restringir poblaciones.
- `nac_ano`, `nac_dia`, `nac_mes` (100% cobertura, fecha de nacimiento
  exacta): **EXCLUIDAS** -- redundantes con `edad`, y una fecha de
  nacimiento exacta es un cuasi-identificador que no aporta señal
  adicional relevante para el benchmark de pobreza.
- `orden_conyuge`, `orden_madre`, `orden_padre`, `orden_tareas`
  (9.5%-49.6% cobertura): **EXCLUIDAS** -- son punteros al numero de
  orden dentro del hogar de otra persona (conyuge/madre/padre/quien
  asigna tareas), no contenido sustantivo de la encuesta; ya se usa la
  informacion sustantiva relacionada (`tareas`, `padre_vive`,
  `madre_vive`, `vive_conyuge`) directamente en los Bloques 1 y 8.

Con esto, las 139 candidatas quedan 100% contabilizadas: construidas en
alguno de los 9 bloques o excluidas con razon documentada (cobertura
<10%, redundancia, o ser un puntero/identificador y no contenido
sustantivo). Con esta verificacion se completa el modulo de Personas.

### 2026-08-09 (cont.) — El usuario pregunta: "¿como estar seguro de que
las 139 son las unicas que se pueden usar y no se esta perdiendo
informacion relevante?" — dos columnas rescatadas, renombradas entre
olas

Pregunta legitima: las 139 vienen de una clasificacion automatica
temprana (ver "Auditoria completa de personas_elca_longitudinal", regla
de "presente" = >1% no-nulo por ola). Se hicieron 3 verificaciones
independientes de esa clasificacion antes de responder:

1. **¿Hay algo excluido por `EXCLUIDA_CASI_VACIA` que en realidad tenga
   señal util?** Se reviso la distribucion completa de cobertura maxima
   (entre las 3 olas) de las 427 columnas de esta categoria: el maximo
   observado es 0.9%, muy por debajo del umbral de 1% -- ninguna esta
   "al borde" del corte. Los 7 `EXCLUIDA_OTRO_PATRON` tambien se
   revisaron uno por uno: todas con cobertura <=1% en las 3 olas, sin
   contenido oculto.
2. **¿Alguna columna de `EXCLUIDA_NO_EN_OLA1` (369) es en realidad la
   MISMA pregunta que una de `EXCLUIDA_NO_EN_OLA2` (310), solo
   renombrada entre olas** (mismo patron ya visto en `n_empleados`,
   Bloque 5)**?** Se cruzaron los 369 nombres contra los 310 por
   similitud de texto (`difflib`, cutoff 0.75): 92 pares candidatos. La
   gran mayoria son falsos positivos (numeros de sub-item de un mismo
   checklist, ej. `cr_activ1_ma`..`cr_activ6_ma` vs `cr_actv_ma6`, o
   preguntas de contenido distinto con nombre parecido, ej. `asma_m`/
   `asma_p` = asma de la madre/el padre en ola 1, vs `asma` = asma
   propia en ola 2/3 -- verificado por cobertura: 23.5%+20.4% en ola 1
   vs. 39.5% en ola 2, consistente con dos preguntas distintas
   consolidadas en un solo referente, no un renombrado). Pero 2 pares
   resultaron ser la MISMA pregunta, confirmado por cobertura y
   estructura de categorias identicas entre olas:
   - `sindicato` (ola 1, cobertura 39.2%) = `org_sindicato` (ola 2/3,
     39.5%/39.1%) -- afiliacion sindical, mismo rango de edad (13+),
     mismas categorias Sí/No/No informa.
   - `fue_jornalero` (ola 1, cobertura 18.3%) = `jornalero` (ola 2/3,
     19.0%/18.3%) -- ejercicio de trabajo como jornalero en el periodo
     de referencia, misma estructura.
3. **Al revisar `sindicato` para armonizarla, aparecio un segundo
   problema mas subtil**: la columna TODAVIA tenia 175 valores "S�" sin
   corregir (U+FFFD) en la base limpia. No es una falla del pipeline de
   correccion -- SI estaba correctamente registrada en
   `personas_corrupcion_residual.csv` desde el principio -- sino que
   nunca se prioriza porque `sindicato` habia quedado fuera del pool de
   139 candidatas (la correccion automatica no pudo resolverla sola
   porque en ola 1 el 100% de las respuestas "Sí" quedaron corruptas,
   sin ningun "Sí" limpio en la misma columna contra el cual hacer
   match). Como el vocabulario de la pregunta es cerrado (Sí/No/No
   informa), "S�" solo puede ser "Sí" -- se agrego a
   `CORRECCIONES_MANUALES_PRIORITARIAS` en
   `02_limpieza_base_personas.py` y se re-corrio la limpieza completa
   (89 valores en 38 columnas, antes 88/37).

**Correccion aplicada**: se agrego `sindicato_armonizada` como undecimo
tipo de organizacion en `build_ahorro_capital_social_hogar.py` (Bloque
4) y `jornalero_armonizado` -> `jornalero_jefe`/
`pct_adultos_fue_jornalero` en `build_educacion_ocupacion_hogar.py`
(Bloque 2). Los 9 scripts de features se re-corrieron sobre la base
limpia regenerada; sin cambios de magnitud en las demas variables
(`sindicato` solo afecta esas 2 columnas). `docs/variable_audit/
personas_hogar_construccion.csv` se actualizo marcando las 4 columnas
(`sindicato`, `org_sindicato`, `fue_jornalero`, `jornalero`) como
rescatadas.

**Respuesta honesta a la pregunta del usuario**: no, las 139 originales
NO eran perfectas -- la clasificacion automatica por presencia/ausencia
de nombre de columna no detecta renombrados entre olas, y eso costo 2
columnas reales (de 139, un 1.4% del pool). El metodo que las encontro
(similitud de nombre + verificacion de cobertura/estructura) es
generalizable pero no se habia aplicado sistematicamente hasta ahora;
las demas categorias de exclusion (`CASI_VACIA`, `OTRO_PATRON`) si se
verificaron a fondo y no mostraron mas casos ocultos. No hay garantia
matematica de que sea imposible que quede algo mas -- el riesgo
remanente mas probable es el mismo patron (renombrado silencioso) en
alguna de las 369+310 columnas restantes con nombres menos similares
entre si (por ejemplo, un cambio de nombre mas drastico que `difflib`
con cutoff 0.75 no detecta), no en las categorias `CASI_VACIA`/
`SOLO_OLA3`/`IDENTIFICADOR` que ya se revisaron con mas detalle.

Siguiente paso (segun instruccion del usuario de terminar todo Personas
antes de consolidar): pasar a los modulos de Niños, Comunidades y
Choques, y solo despues consolidar todos los parquets de features en un
solo dataset para el benchmark.

Output: `data/processed/salud_discapacidad_hogar_ext_elca_longitudinal.parquet`.

### 2026-08-09 (cont.) — Segunda vuelta de busqueda de renombrados: umbral
relajado, un hallazgo adicional (`cotiza_fp`/`cotizando`)

El usuario pidio explicitamente relajar el umbral y volver a revisar,
tras la correccion de `sindicato`/`jornalero` (ver seccion anterior). Se
bajo el cutoff de similitud de nombre (`difflib`) de 0.75 a 0.55 contra
las 367+308 columnas restantes de `EXCLUIDA_NO_EN_OLA1`/
`EXCLUIDA_NO_EN_OLA2` (ya sin las 4 columnas rescatadas en la ronda
anterior): 506 pares candidatos, filtrados a 115 con cobertura similar
entre olas (diferencia <5 puntos porcentuales, cobertura >3%).

**Hallazgo metodologico importante**: la mayoria de estos 115 son falsos
positivos por una razon especifica -- MUCHAS preguntas no relacionadas
comparten la misma poblacion filtrada (~39% de las personas, el mismo
"informante idoneo" al que se le hacen `ahorra`/`org_*`/`cotiza_fp` en
ola 1, o el mismo filtro de edad 13+/15+ en ola 2/3), asi que tienen
cobertura casi identica por coincidir en el filtro, no en el contenido.
Verificado caso por caso con `value_counts()` en los 12 pares con menor
diferencia de cobertura: `califica_salud`/`afilia_cual`,
`informante`/`enf_corazon`, `desempleado`/`seg_medico`,
`act_cotidiana`/`ataq_corazon`, `fundo_negocio`/`vr_negocio` vs
`edad_negocio` -- todos confirmados como preguntas DISTINTAS con
coincidencia de cobertura espuria, no renombres. `t_dejo_trabajar`/
`tiempo_trab` tienen categorias parecidas pero conceptualmente distintas
(tiempo desde que dejo de trabajar vs. antigüedad en el trabajo actual),
ambiguo, no se fuerza el merge (de cualquier forma cae bajo el umbral de
10% de cobertura). `noson_hogar`/`noson_hogar1` SI son la misma pregunta
(categorias identicas, ej. "Por insuficiencia de ingresos" con conteos
casi iguales) pero quedan excluidas de todas formas por cobertura
<10% (3.5%/3.0%/2.7%), sin efecto practico. `fpers`/`fexpers` vs
`fpers_2013`/`fexpers_2013` son factores de expansion muestral (ya
identificados en la ronda anterior), no contenido sustantivo -- no son
features, son metadatos de diseño muestral.

**Un hallazgo real y de valor**: `cotiza_fp` (ola 1, cobertura 39.2%) es
la misma pregunta que `cotizando` (ola 2/3, cobertura 39.5%/39.1%) --
cotizacion activa a fondo de pensiones, un indicador fuerte de
formalidad laboral. El nombre y la cobertura por si solos no lo dejaban
ver claro porque `cotiza_fp` en ola 1 es una pregunta COMBINADA de 9
categorias (estado + motivo si no cotiza, ej. "No cotiza porque no tiene
dinero", "Si está cotizando, pero todavía no es pensionado"), mientras
ola 2/3 la separan en `cotizando` (Sí/No limpio) + `cotiza_cual` (texto
libre, casi vacio, no relacionado). Se colapso `cotiza_fp` a binario
("Sí" = categorias que empiezan con "Si est...", "No" = el resto,
incluye "ya está pensionado" por no ser cotizacion activa): da 15.7% de
"Sí" en ola 1 vs. 16.4% en `cotizando` de ola 2 -- magnitud consistente,
confirma el renombrado. `cotiza_fp` tambien tenia una categoria con
corrupcion U+FFFD sin resolver ("Si est�cotizando y recibe pensi�n", 52
casos) por el mismo motivo que `sindicato` (sin candidato limpio en la
misma columna); se agrego a `CORRECCIONES_MANUALES_PRIORITARIAS` (ahora
90 valores en 39 columnas, antes 89/38).

**Correccion aplicada**: se agrego `cotiza_pension_jefe` y
`tasa_cotizacion_pension_hogar` a `build_ahorro_capital_social_hogar.py`
(Bloque 4) -- misma poblacion filtrada que `ahorra`/`org_*`, mismo rango
de edad. Tasa de cotizacion a nivel hogar: 15.9%/16.4%/18.1% entre olas,
tendencia suave y consistente, sin anomalias. Los 9 scripts de features
se re-corrieron sobre la base limpia regenerada.

**Total de columnas rescatadas por renombrado en las dos rondas: 3**
(`sindicato`, `fue_jornalero`, `cotiza_fp`, mapeadas respectivamente a
`org_sindicato`, `jornalero`, `cotizando` en ola 2/3) sobre 139
candidatas originales -- 2.2% del pool. Con umbral 0.55 y verificacion
manual de los 12 pares mas prometedores no aparecio nada mas. Bajar mas
el umbral (por debajo de 0.5) empieza a generar demasiado ruido para
verificar caso por caso de forma confiable; el riesgo residual de un
renombrado con nombre muy distinto (ej. un cambio de codigo completo,
no solo de prefijo/sufijo) sigue sin poder descartarse por este metodo
y requeriria comparacion directa contra los diccionarios PDF de cada
ola, pregunta por pregunta -- no se hizo por ser demasiado costoso para
el beneficio esperado dado lo bajo de la tasa de hallazgo real (3 en 2
rondas).

### 2026-08-09 (cont.) — Tercera vuelta: comparacion directa contra los
diccionarios PDF (texto de la pregunta, no solo nombre/cobertura), 2
hallazgos mas

El usuario pidio intentar el metodo mas costoso planteado al cierre de la
ronda anterior: en vez de comparar nombres de columna o cobertura
numerica, comparar el TEXTO DE LA PREGUNTA de cada columna contra los
diccionarios oficiales de la encuesta.

**Metodo**: se extrajeron con `pdftotext -layout` los 4 diccionarios de
Personas de las olas con nombres de columna potencialmente distintos
(`elca_2010/{U,R}Personas.pdf`, `elca_2013/{U,R}Personas.pdf`) y se
parsearon con un script ad-hoc (tabla de columnas ID/Variable/Capítulo o
Módulo/Descripción/Tipo/Formato/Valores, reconstruyendo cada descripcion
a partir de las lineas de continuacion usando la posicion horizontal del
texto para separar la columna Descripción de la columna Valores que
comparten renglon en el layout de texto plano). Cobertura del parseo:
231/307 columnas de `EXCLUIDA_NO_EN_OLA2` y 260/366 de
`EXCLUIDA_NO_EN_OLA1` quedaron con descripcion recuperada (el resto no se
pudo extraer de forma confiable, mismo riesgo residual que ya se tenia
documentado, no empeora nada). Se calculo similitud de Jaccard sobre las
palabras de la descripcion (normalizadas: sin tildes, sin stopwords, >=3
caracteres) entre los dos conjuntos: 35 pares con Jaccard>=0.3.

**Verificacion caso por caso de los pares con mayor similitud** (cruzando
tambien cobertura y edad, mismo criterio que en las rondas anteriores):
la mayoria de los 35 son preguntas RELACIONADAS pero no identicas (ej.
`medio_busco`/`medio_bus_trabajo`, ambas maneras de preguntar por medios
de busqueda de empleo pero con categorias distintas, y de cualquier forma
<10% de cobertura, sin efecto practico). Dos pares SI son la misma
pregunta bajo nombre distinto, confirmado por texto de pregunta
practicamente identico + cobertura y rango de edad consistentes:

  - `estaba_sss` (ola 1, cobertura 14.9%, edad 13-71) = `segsoc_salud`
    (ola 2/3, 14.4%/14.8%, edad 17-88) -- afiliacion a seguridad social
    en SALUD por vinculo LABORAL. Distinta de `afiliacion` (Bloque 3,
    39%, todas las edades, afiliacion general sin filtro de tipo de
    vinculo) -- verificado que no son redundantes.
  - `estaba_fp` (ola 1, cobertura 14.9%) = `afiliacion_fp` (ola 2/3,
    14.4%/14.8%) -- afiliacion (no necesariamente cotizacion activa) a
    fondo de PENSIONES. Distinta de `cotiza_pension` (cotizacion activa,
    ya rescatada en la ronda anterior) -- se puede estar afiliado sin
    cotizar activamente, son conceptos relacionados pero no iguales.

Ambas tenian el mismo patron de corrupcion U+FFFD sin resolver que
`sindicato`/`cotiza_fp` (categoria "Sí" 100% corrupta como "S�" en su
columna, sin candidato limpio local para el match automatico) --
agregadas a `CORRECCIONES_MANUALES_PRIORITARIAS` (ahora 92 valores en 41
columnas).

**Correccion aplicada**: se agregaron `afiliado_pension_jefe`/
`tasa_afiliacion_pension_hogar` y `afiliado_salud_laboral_jefe`/
`tasa_afiliacion_salud_laboral_hogar` a
`build_ahorro_capital_social_hogar.py` (Bloque 4). Validacion: afiliacion
a pension 34.2%-42.1% entre olas, afiliacion a salud laboral 34.6%-46.2%
-- ambas mayores que la tasa de cotizacion ACTIVA (15.9%-18.1%), lo cual
es coherente (estar afiliado sin cotizar activamente es comun en mercados
laborales informales). Los 9 scripts de features se re-corrieron sobre la
base regenerada.

**Total de columnas rescatadas en las 3 rondas: 5** (`sindicato`,
`fue_jornalero`, `cotiza_fp`, `estaba_sss`, `estaba_fp`, mapeadas a
`org_sindicato`, `jornalero`, `cotizando`, `segsoc_salud`,
`afiliacion_fp` respectivamente) sobre 139 candidatas originales -- 3.6%
del pool. El metodo de diccionario (mas costoso, basado en texto de
pregunta real) encontro 2 casos que el metodo de similitud de nombre con
umbral relajado NO habia detectado (`estaba_sss`/`estaba_fp` no tienen
nombres parecidos a `segsoc_salud`/`afiliacion_fp`) -- confirma que el
riesgo remanente identificado en la ronda anterior (renombrados con
nombre muy distinto) era real y no solo teorico. Con este metodo se
alcanza la cobertura mas alta posible dado lo que se pudo extraer de los
diccionarios (231/307 y 260/366 descripciones recuperadas); las columnas
sin descripcion recuperable en el parseo (76+106) quedan como el unico
riesgo residual no verificable con los recursos actuales -- requeriria
revisar el PDF visualmente pagina por pagina, desproporcionado frente a
la tasa de hallazgo observada (5 en 3 rondas, con retornos decrecientes:
2 en la primera ronda de nombre estricto habian dado 0, la segunda con
nombre relajado dio 3, la tercera con texto de pregunta dio 2 mas).

### 2026-08-09 (cont.) — Choques: se resuelve el HALLAZGO CRITICO (cobertura
35%/70%/76% -> 100%)

Con Personas cerrado, se retomo el bloqueo pendiente de Choques (ver
"HALLAZGO CRITICO" en la auditoria de modulos, mas arriba). Se volvio a
los `.tab` crudos (`data/interim/raw/elca_{2010,2013,2016}/{U,R}Choques*.tab`)
para responder la pregunta que quedo pendiente: ¿que significa la ausencia
de una fila para un hogar dado?

**Verificacion contra los crudos**: 2010 usa formato ANCHO -- un archivo
`UChoques-csv.tab`/`RChoques-csv.tab` sin encabezado con exactamente una
fila por hogar (5.275+4.579=9.854 filas, cuadra con los 9.853 hogares del
panel); los hogares sin ningun choque tienen las columnas de choque en
blanco, PERO SI aparecen en el archivo. 2013/2016 usan formato LARGO --
una fila por (hogar, tipo de choque) incluyendo las filas con
`tuvo_choque=='No'`, por lo que cada hogar aparece ~16-23 veces
independientemente de si tuvo o no cada choque especifico. Confirmado:
el archivo fuente de ELCA SI tiene cobertura completa del panel en las 3
olas -- el problema nunca fue el dato, sino el script de consolidacion.

**Causa raiz encontrada en `01_consolidacion_bases_choques.py`**: las
funciones `procesar_2013`/`procesar_2016` ejecutaban
`df = df[df["tuvo_choque"] == "SI"].copy()` ANTES de pivotear, eliminando
todas las filas de hogares sin ningun choque (su bloque completo de 'No'
desaparecia); `procesar_2010`/`raw_2010_a_long` hacia `if pd.isna(choque):
continue`, saltandose los bloques vacios de los hogares sin choques. El
resultado: el pivote de conteo (`pivot_table(index=key_col, ...)`) nunca
recibia esos hogares en su indice, asi que quedaban fuera del panel final
en vez de aparecer con `choque_*=0`.

**Correccion aplicada**: se capturo el universo completo de hogares
(`hogares_universo`) de cada archivo crudo ANTES del filtro
`tuvo_choque=='SI'`/`pd.isna`, y se reindexo el pivote de conteo contra
ese universo con `fill_value=0` (`choque_*`, `total_choques`).
`imp_econ_*`/`resp_*` se dejan en NaN para los hogares sin choque -- es
correcto, no aplica reportar impacto economico o respuesta de
afrontamiento cuando no hubo ningun evento. Tambien se corrigio la ruta
hardcodeada del script (`DATA_ROOT`/`OUTPUT_PATH` apuntaban a
`.../Documentos/tesis_vulnerabilidad/...`, sin el prefijo `Tesis_MECA/`
del proyecto actual -- ni siquiera podia ejecutarse antes de este fix) a
`Path(__file__).resolve().parents[3]`, siguiendo la misma convencion
`PROJECT_ROOT` usada en el resto del proyecto.

**Resultado tras re-ejecutar la consolidacion**: 27.932 filas (antes
16.401), exactamente el mismo tamaño de panel hogar-ola que usa el resto
del proyecto (ej. `personas_hogar_elca_longitudinal.parquet`). Hogares
por ola: 9.853 (ola 1, 100% de cobertura, antes 35%) / 9.261 (ola 2) /
8.818 (ola 3). Proporcion de hogares con `total_choques==0`: 65.4% (ola
1) / 31.1% (ola 2) / 25.0% (ola 3) -- magnitud plausible para una
pregunta de recall de choques en un periodo de referencia de ~12 meses
(o mas largo en la primera ola). El maximo de `total_choques` (95 en ola
1, 89 en ola 3) es alto pero no es un artefacto de este fix -- ya existia
en la logica de conteo original (`clip(lower=1)` sobre columnas
`mes_*`/`veces_*`), queda como observacion para revisar al construir el
bloque de features de Choques, no bloquea el uso del modulo.

Con esto, Choques queda desbloqueado y listo para construir su bloque de
features del benchmark.

### 2026-08-09 (cont.) — Modulo de Comunidades: auditoria completa (mismo
rigor que Personas) + bloque de features (build_comunidades_hogar.py)

Con Personas y Choques resueltos, se inicia el modulo de Comunidades. El
usuario pidio explicitamente el mismo nivel de auditoria que Personas, no
un estandar mas liviano -- se rehizo el trabajo con las 4 etapas completas.

A diferencia de Personas, la unidad de analisis original YA es la
comunidad (entrevista a lideres comunitarios) -- cada hogar hereda las
variables de su comunidad via `consecutivo_c` (identificador de comunidad
= `consecutivo` del hogar sin el ultimo digito, confirmado exacto contra
`UHogar-csv.tab`: hogares 111001/111002/111003 -> consecutivo_c 11001).

**1. Limpieza de corrupcion U+FFFD** (`03_limpieza_base_comunidades.py`,
nuevo, mismo patron que `02_limpieza_base_personas.py`): de 558 columnas,
26 tienen "�" (mucho mas acotado que las 542 de Personas) -- 22 de
vocabulario cerrado (<=25 categorias) y 4 de texto libre
(`proy_prioritario1/2/3`, `otros_problemas_cual`, sin tocar). La
correccion automatica (match unico dentro de la misma columna) resuelve
28 valores sin ambiguedad; queda 1 residual, `region` (3 valores: Bogotá/
Atlántica/Pacífica corruptas, sin candidato limpio local porque esas 3
categorias SIEMPRE aparecen corruptas en el archivo -- mismo patron que
`sindicato`/`cotiza_fp`/`estaba_sss`/`estaba_fp` en Personas), corregida a
mano por ser inequivoca (son los 3 nombres de region del diseño muestral
de ELCA). Validado con assert: 0 residual tras la correccion manual.
Output: `comunidades_elca_longitudinal_clean.parquet`.

**2. Clasificacion completa de las 558 columnas**
(`docs/variable_audit/comunidades_construccion.csv`, mismo criterio >1%
de cobertura por ola que Personas): 212 `CANDIDATO_BENCHMARK` (presente
ola1 Y ola2), 207 `EXCLUIDA_NO_EN_OLA1`, 40 `EXCLUIDA_NO_EN_OLA2`, 52
`EXCLUIDA_SOLO_OLA3`, 39 `EXCLUIDA_CASI_VACIA`, 5 `IDENTIFICADOR`, 3
`EXCLUIDA_OTRO_PATRON` (`cod_conf_5`/`n_hogares_5`/`anos_resolver_5`,
verificados triviales, <2% cobertura). Suma verificada: 558.

**3. Busqueda de renombrados entre olas** (nombre relajado cutoff 0.55 +
comparacion contra diccionarios PDF de
`elca_{2010,2013}/{U,R}Comunidades.pdf`, mismo metodo de 2 pasadas usado
en Personas): **sin hallazgos nuevos**. Los pares con mayor similitud
resultaron ser:
  - Bloques de items numerados con distinto N por ola (`sexo_lider5/6`,
    `cargo_lider5/6` sin equivalente porque ola 2 solo pregunta hasta el
    lider 4; `cod_conf_1..4` vs `cod_conf_7..12`, mismo patron para
    conflictos) -- NO son renombrados, son preguntas repetidas con
    distinto limite de repeticion por ola.
  - `grarmados_2001..2010` vs `grarmados_2011..2013` -- diseño de VENTANA
    MOVIL de años (cada ola pregunta por los grupos armados presentes en
    años especificos, sin solapamiento entre olas) -- tampoco es un
    renombrado, es un cambio de referencia temporal por diseño del
    cuestionario.
  - Verificado caso por caso (`camp_prev_embarazo`/`camp_emb_adoles`,
    `venden_mas10`/`ven_vias`, `pasa_tierra_ido`/`pasa_tierra_cual`):
    coincidencias de nombre espurias, contenido distinto en los 3 casos.
  A diferencia de Personas (donde este metodo rescato 5 columnas reales),
  aqui el modulo de Comunidades no tiene el mismo patron de renombrado
  silencioso entre olas.

**4. Aplicacion del umbral >=10% por tema** sobre las 212 candidatas para
decidir que construir. Se verifico estructura de categorias, escalas y
poblacion de cada variable ANTES de construirla (misma disciplina de
Personas):
  - `homicidios` usa escalas DISTINTAS entre zona urbana (Sí/No) y rural
    (Nunca/Algunas veces/Frecuentemente) -- verificado cruzando con
    `zona` antes de construir (mismo principio de comparabilidad
    poblacional de Personas). Armonizado a binario.
  - `seguridad` es ordinal de 4 niveles con variantes de genero (Muy
    seguro/segura, etc.) -- normalizado y mapeado 1-4 hacia percepcion de
    INSEGURIDAD.
  - `solidaridad` es ordinal de 3 niveles (No se ayudan/Se ayudan poco/Se
    ayudan mucho) -- mapeado a 0-2.
  - `desalojos`/`secuestros`/`amenazas_gra` mezclan Sí/No CON Nunca/
    Algunas veces/Frecuentemente/Todo el tiempo dentro de la MISMA
    columna (mismo patron que `homicidios`) -- armonizados a binario
    antes de combinarlos en el indice de conflicto armado.
  - `acude_solucion` tenia una categoria con corrupcion "???" literal
    ("Otro.  ???Cu???l?", mismo patron ya documentado para Personas ola
    3, aqui aparece en Comunidades) -- consolidada con "Otro".
  - `inf_salud`/`inf_educacion`/etc. se verificaron contra el diccionario
    PDF: NO son "tiene puesto de salud" (eso ya lo mide `puesto_salud`)
    sino "hubo obra de infraestructura de salud en los ultimos 2 años" --
    variable de INVERSION RECIENTE, complementaria y no redundante con
    presencia actual (confirmado con `pd.crosstab`: 308 comunidades
    difieren entre ambas).
  - **Anomalia de IDs en el archivo crudo de Comunidades**: 25 comunidades
    urbanas de 2016 (`UComunidades-csv.tab`) tienen `consecutivo_c` de
    8-10 digitos (ej. 8110011099) en vez del rango normal de 5-6 digitos
    -- confirmado que es un problema del archivo CRUDO de ELCA, no de
    esta consolidacion (sin efecto practico, ola 3 nunca es fuente de
    features).
  - **Cobertura del join hogar->comunidad**: el `consecutivo_c` del
    HOGAR (no solo el de Comunidades) tambien trae el mismo tipo de
    codigo malformado en ola 2/3, mas un sentinela "8888888" (4.468
    filas, codigo estandar de "sin dato"). Filtrando esos IDs invalidos,
    queda un segundo problema independiente: en ola 1, 80 de ~792
    comunidades referenciadas por hogares simplemente no aparecen en el
    archivo de Comunidades (8.1% de los hogares sin comunidad
    emparejada; 2.9%/3.6% en ola 2/3) -- cobertura real del cuestionario
    de Comunidades, no un error de calculo del ID. Se deja NaN, sin
    imputar.

**Temas EXCLUIDOS deliberadamente, con razon documentada** (no solo "no
alcanzo el tiempo" -- ver docstring completo en
`build_comunidades_hogar.py` para el detalle variable por variable):
demografia de lideres comunitarios (metadato del informante, no
caracteristica de la comunidad); `hecho_seguridad`/`razon_seguridad`
(sub-pregunta filtrada de baja cobertura); cluster de acceso a mercado
agricola rural (~25 columnas, cobertura 26%-31% que coincide con la
fraccion rural de la muestra -- submodulo rural fuera de alcance);
cluster de tipos de trabajo rural y calendario climatico mensual (mismo
submodulo rural); cluster de proveedores de salud alternativos
(medico/odontologo/enfermera "nopuesto", rural-only); cluster de
campañas de salud/agricolas (salta de 26% a 100% entre olas, mismo
patron de posible problema de comparabilidad poblacional que
discapacidad en Personas, sin investigar a fondo); extension del
problema ambiental (cobertura mas baja y redundante con
`problema_contaminacion_comunidad` ya construido); sub-preguntas de
seguimiento de baja cobertura sobre temas ya cubiertos.

**Variables construidas** (nivel hogar, heredadas de la comunidad):
`percepcion_inseguridad_comunidad` (escala 1-4), `problema_homicidios_comunidad`,
`n_problemas_convivencia_comunidad` (conteo 0-7), `problema_contaminacion_comunidad`,
`riesgo_inundacion_comunidad`, `acceso_agua_comunidad`,
`hay_desplazados_comunidad`/`n_desplazados_comunidad`,
`n_organizaciones_comunidad` (conteo 0-14), `tiene_puesto_salud_comunidad`,
`tiene_escuela_primaria_comunidad`, `tiene_colegio_secundaria_comunidad`,
`tiene_transporte_publico_comunidad`, `barrio_legal_comunidad`,
`solidaridad_comunidad` (escala 0-2), `acude_justicia_formal_comunidad`,
`cortes_agua_comunidad`, `n_obras_infraestructura_reciente_comunidad`
(conteo 0-6), `n_servicios_primera_infancia_comunidad` (conteo 0-4),
`n_espacios_publicos_comunidad` (conteo 0-4),
`n_acciones_conflicto_armado_comunidad` (conteo 0-11).

**Validacion**: percepcion de inseguridad ~2.0 (escala 1-4, centrada),
homicidios reportados 18%-32% (declinante entre olas, consistente con la
tendencia de reduccion de violencia en Colombia 2010-2016), contaminacion
69%-77%, desplazados reportados 46%-53% de comunidades (alto pero
plausible dado que ELCA sobre-muestrea poblacion vulnerable/afectada por
conflicto), `n_acciones_conflicto_armado_comunidad` cae de 1.35 (ola 1) a
~0.49-0.50 (olas 2/3) -- mismo patron declinante que homicidios,
consistente. Dos caveats sin resolver documentados en el docstring del
script: `n_organizaciones_comunidad` cae de 4.25 a 2.10 entre olas
(cobertura estable, no es artefacto) y `n_desplazados_comunidad` cae de
una media de 195 a ~70 personas (posible cambio de periodo de referencia
de la pregunta, no verificado contra diccionario).

Output: `data/processed/comunidades_hogar_elca_longitudinal.parquet`
(27.932 filas, mismo panel hogar-ola que el resto del proyecto).

### 2026-08-09 (cont.) — Modulo de Niños: auditoria completa + bloque de
features (build_ninos_hogar.py)

Mismo nivel de auditoria que Personas y Comunidades, 4 etapas completas.

**1. Limpieza de corrupcion** (`04_limpieza_base_ninos.py`, nuevo): de 433
columnas, 80 tienen U+FFFD ("�") -- 79 cerradas + 1 texto libre
(`descrip_oficio`). Correccion automatica: 131 valores. **Hallazgo
adicional al validar `quien_cuida`**: la misma columna tenia TAMBIEN
corrupcion "???" literal (patron ya visto en Personas, pero ahi exclusivo
de ola 3 -- aqui aparece en las 3 olas, afecta 12 columnas). Se
generalizo el script para manejar ambos marcadores. Residual final: 6
valores rescatados via diccionario PDF de 2013 (`{U,R}Ninos0a13.pdf`), 3
via la regla Sí/No de Personas (candidato ambiguo pero inequivoco por
diseño), resto corregido a mano por reconstrucciones de acento sin
ambiguedad; 2 columnas (`dejo_lactar_cual`, `observ_antrop`) resultaron
ser texto libre "accidentalmente cerrado" por tamaño de muestra chico --
se re-clasificaron y se dejan sin forzar correccion.

**2. Clasificacion completa de las 433 columnas**
(`docs/variable_audit/ninos_construccion.csv`): 83 `CANDIDATO_BENCHMARK`,
256 `EXCLUIDA_NO_EN_OLA1` (el modulo crecio mucho de 2010->2013: nuevo
tramo de edad 0-13 y nuevas baterias de estimulacion/cuidado), 56
`EXCLUIDA_SOLO_OLA3`, 22 `EXCLUIDA_NO_EN_OLA2`, 11 `IDENTIFICADOR`, 4
`EXCLUIDA_CASI_VACIA` (verificadas triviales), 1 `EXCLUIDA_OTRO_PATRON`
(`ano_nac_m`, trivial). Suma verificada: 433.

**3. Busqueda de renombrados entre olas** (nombre relajado + diccionarios
PDF de `elca_{2010,2013}/{U,R}Ninos*.pdf`): **5 renombrados reales
encontrados**, todos del bloque de "estimulacion en el hogar" -- en ola 1
la pregunta era una unica variable de frecuencia; en ola 2 se dividio en
"quien" (quien lo hace) y "freq_*" (frecuencia). Se rescata la parte de
frecuencia:
  - `conversa` = `freq_conversa`, `ensena` = `freq_ensena`,
    `juega_fueradecasa` = `freq_juegafuera`, `lee_libros` = `freq_lee`
    (encontrados por nombre relajado).
  - `juega_encasa` = `freq_juegadentro` -- encontrado SOLO por el
    diccionario PDF ("encasa" vs "dentro" no tienen similitud textual
    suficiente para el metodo de nombre). Confirma otra vez el valor de
    hacer las 2 pasadas (nombre + diccionario), no solo una.
  Mismas 4 categorias de frecuencia en ambas olas, verificado antes de
  armonizar.

**4. Verificacion de poblacion/edad antes de construir**: `edad_ames`
esta codificada como años*100+meses (ej. 207 = 2 años 7 meses) --
decodificada primero. El bloque de salud/vacunacion/estimulacion se
concentra en niños 0-7 en ambas olas (mediana 2.6-4.8 años); el bloque de
oficios/trabajo domestico en niños 5-11 en ambas olas (mediana 5.1-8.2
años) -- ambos comparables entre ola 1 y ola 2 (las 2 fuentes de
features del benchmark; ola 3 solo aporta el resultado observado).

**Bug encontrado en validacion (el mismo tipo de error que motivo esta
auditoria completa desde el principio)**: `fiebrea` se iba a llamar
`pct_ninos_fiebre_reciente_hogar` asumiendo que media "tuvo fiebre" --
al ver una tasa de 93.8%/89.9% (implausible para un sintoma reciente) se
verifico contra el diccionario PDF 2013 y resulto ser **"¿Recibió la
vacuna contra la fiebre amarilla?"** -- una vacuna mas de la bateria, no
un sintoma. Renombrada a `tasa_vacuna_fiebreamarilla_hogar`. Segundo
hallazgo en la misma validacion: `pct_ninos_cuidado_terceros_hogar` daba
93% por un bug de matching (`PADRES_TOKENS` no incluia "Su madre"/"Su
padre", solo "La madre"/"El padre" -- la columna `quien_cuida` usa
ambas formas segun la fila) -- corregido. Tercer hallazgo: el indicador
binario `pct_ninos_oficios_hogar` (¿hace algun oficio?) se satura en
93%-99.7% porque `limpieza` sola ya tiene ~87% de "Sí" (barra baja
segun el diccionario: "de los siguientes oficios, ¿cuáles hizo la
semana pasada?", cualquier ayuda de aseo cuenta) -- se agrego
`n_oficios_promedio_nino_hogar` (conteo 0-8) para preservar varianza
como covariable de intensidad.

**Redundancia deliberadamente NO construida**: `padre_vive`,
`madre_vive`, `educ_padre`, `educ_madre`, `trabajo_padre`,
`trabajo_madre`, `orden_padre`, `orden_madre`, `ano_nac_p` -- ya existen
en Personas (via el informante) con variables equivalentes construidas
(`pct_ninos_padre_vivo`/`pct_ninos_madre_viva`, Bloque 8;
`nivel_educ_max_hogar`, Bloque 2). Duplicar desde Niños no aporta señal
adicional.

**Test cognitivo TVIP**: `puntoinicio`/`itemtope`/`menoserrores`/
`puntuaciondirecta` corresponden al Test de Vocabulario en Imágenes
Peabody (confirmado contra el diccionario PDF: "variable: es igual a la
puntuación... prueba TVIP"). Se usa `puntuaciondirecta` (puntaje bruto),
sin ajustar por edad -- misma limitacion que talla/peso.

**Variables construidas** (nivel hogar, agregadas sobre niños con dato
valido): `tasa_vacunacion_basica_hogar` (antituberculosa+triple viral+
hepatitis B recien nacido), `tasa_control_crecimiento_hogar`,
`tasa_vacuna_fiebreamarilla_hogar`, `talla_promedio_nino_hogar`/
`peso_promedio_nino_hogar` (sin ajustar por edad/sexo, covariable cruda),
`tasa_asistencia_escolar_nino_hogar`, `pct_ninos_oficios_hogar`/
`n_oficios_promedio_nino_hogar`, `horas_oficio_promedio_nino_hogar`,
`pct_ninos_trabajo_remunerado_hogar`, `indice_estimulacion_hogar_nino`
(promedio 0-4 de 5 items de estimulacion, ver renombrados rescatados),
`tvip_puntaje_directo_hogar`, `pct_ninos_cuidado_terceros_hogar`.

**Validacion**: vacunacion basica 93.4%/82.3%, control de crecimiento
84.3%/88.1%, vacuna fiebre amarilla 93.8%/89.9% (magnitud consistente
con las otras vacunas, confirma la reinterpretacion), asistencia escolar
30.4%/52.4% (coherente con la edad 0-7 del subgrupo, incluye bebes no
escolarizables), oficios domesticos n_oficios_promedio 1.48/2.13 (de 8
posibles), trabajo remunerado infantil 2.8%/2.9% (bajo, plausible dado
el rango de edad 5-11), indice de estimulacion 2.57/3.30 (escala 0-4),
TVIP 46.3/50.5 (talla/peso/TVIP muestran valores mas altos en ola 3 por
ser una poblacion de niños mayores 6-16, no comparable en crudo con
ola 1/2 -- pero ola 3 nunca es fuente de features, no bloquea el uso).

Output: `data/processed/ninos_hogar_elca_longitudinal.parquet` (15.473
filas -- solo hogares con al menos un niño en el rango de edad relevante,
menor que el panel completo de 27.932 por diseño).

Con esto se completan los 4 modulos con auditoria completa (Personas,
Comunidades, Niños, mas la correccion de Choques). Pendiente: construir
el bloque de features de Choques (ya desbloqueado) y, solo despues,
consolidar todos los parquets de features en un dataset unico para el
benchmark, segun la instruccion original del usuario.

### 2026-08-09 (cont.) — Bloque de features de Choques
(build_choques_hogar.py)

A diferencia de Personas/Comunidades/Niños, `choques_elca_longitudinal.parquet`
ya esta a nivel de hogar (generada por `01_consolidacion_bases_choques.py`,
ver correccion del HALLAZGO CRITICO mas arriba) -- no requiere agregacion
desde individuos, solo join directo por llave de hogar (verificado 1:1
exacto contra `hogar_elca_longitudinal_clean.parquet` en las 3 olas).

**Verificacion pedida por el usuario: ¿por que 0% de choques de desastre
natural en ola 1 (2010), si coincidio con la Ola Invernal de Colombia?**
Se extrajo el diccionario PDF oficial de 2010 (`UChoques.pdf`) y se
confirmo que el cuestionario de esa ola enumera EXPLICITAMENTE las 18
categorias de choque posibles (todas listadas en el diccionario, items
1-18: enfermedad, accidente, muerte, abandono, separacion, perdida de
empleo, perdida de tierra/vivienda/animales/cosechas, robo, violencia) y
NINGUNA es un desastre natural -- la categoria (inundaciones/avalanchas/
sequias/temblores) se agrego recien en el cuestionario de 2013. Es una
limitacion real y verificada del instrumento de ELCA (posiblemente
porque el cuestionario ya estaba diseñado/en campo antes de que la Ola
Invernal 2010-2011 se agravara), no un artefacto de la consolidacion.

**Otros hallazgos de la auditoria de cobertura**:
  - 2 categorias de "abandono" (`abandono_del_hogar_por_parte_de_un_menor_
    de_18_anos`, `abandono_del_que_era_jefe_del_hogar_o_del_conyuge`)
    desaparecen despues de ola 1 (0% ola2/3) -- preguntas eliminadas del
    cuestionario. Excluidas de los composites (fallan Eje 1).
  - `imp_econ_*` (severidad Alta/Media/Baja) confirmado 0% en ola 1 (ya
    documentado antes) -- no se usa en el benchmark simetrico train/test.
  - `resp_*` (estrategias de afrontamiento) tienen cobertura MENOR
    (34.6%/54.0%/60.9%) que la proporcion de hogares con
    `total_choques>0` (34.6%/68.9%/75.0%). Investigado contra el crudo de
    2013: de 12.439 choques con `tuvo_choque=='SI'`, 3.903 (31.4%) tienen
    la respuesta principal (`hizo_princ`) en blanco -- no-respuesta real
    de ELCA a esa pregunta de seguimiento, no un error de esta
    consolidacion. Se deja NaN, sin imputar.
  - `hipotecaron_algun_activo`/`arrendaron_algun_activo` (2 respuestas
    separadas en ola 1) se fusionaron en `hipotecaron_o_arrendaron_algun_
    activo` (ola 2/3) -- armonizado con OR antes de incluir en el
    composite erosivo.
  - `choque_perdida_o_muerte_de_animales`/`choque_plagas_o_perdida_de_
    cosechas` se preguntan EXCLUSIVAMENTE a hogares rurales (100%
    cobertura rural, 0% urbana, por diseño) -- el ~54% de NaN del
    composite agropecuario a nivel nacional es la fraccion urbana del
    panel, no un error.

**Variables construidas**: incidencia -- `total_choques_hogar`,
`tuvo_algun_choque_hogar`, `n_tipos_choque_hogar` (diversidad, no solo
frecuencia), `tuvo_choque_salud_hogar`, `tuvo_choque_economico_hogar`,
`tuvo_choque_patrimonial_hogar`, `tuvo_choque_agropecuario_hogar`,
`tuvo_choque_familiar_hogar`, `tuvo_choque_severo_hogar` (muerte del
jefe/conyuge, perdida de vivienda o fincas -- subgrupo irreversible).
Afrontamiento -- `afrontamiento_erosivo_hogar` (vendieron bienes,
retiraron hijos del colegio, hipotecaron/arrendaron activo, sacrificaron
animales, redujeron alimentos, migracion internacional -- compromete
capital futuro) vs. `afrontamiento_protector_hogar` (usaron ahorros,
seguro, ayuda de familiares/instituciones -- no compromete capital),
`intensifico_trabajo_hogar` (categoria propia, respuesta de oferta
laboral), mas 5 variables individuales de interes especifico:
`retiro_hijos_colegio_choque_hogar`, `redujo_alimentos_choque_hogar`,
`se_endeudo_formal_choque_hogar` vs. `se_endeudo_informal_choque_hogar`
(formal/informal, proxy de exclusion financiera), `no_ajusto_choque_hogar`
(señal inversa).

**Validacion**: total_choques promedio 0.62/1.42/2.44 (creciente entre
olas, consistente con la ampliacion del cuestionario), n_tipos_choque
(diversidad) maximo 6/8/9 de los ~17-23 tipos posibles -- plausible.
tuvo_choque_severo muy bajo (0.9%-3.0%), consistente con ser un subgrupo
de eventos raros e irreversibles. afrontamiento_erosivo 11.3%-25.6%,
afrontamiento_protector 28.8%-34.3% -- ambos crecen con `total_choques`,
sin anomalias. `no_ajusto_choque_hogar` cae de 41.7% (ola 1) a ~18% (ola
2/3) -- no investigado a fondo, podria reflejar que los choques
disponibles en ola 1 (sin el cluster de desastres/clima, ver arriba) son
en promedio menos severos y requieren menos ajuste; queda como
observacion.

Output: `data/processed/choques_hogar_elca_longitudinal.parquet` (27.932
filas, mismo panel que el resto del proyecto).

Con esto se completan los 4 modulos (Personas, Comunidades, Niños,
Choques) con features construidas y auditadas. Siguiente paso, segun la
instruccion original del usuario: consolidar todos los parquets de
features en un dataset unico hogar-ola para el benchmark.

### 2026-08-09 (cont.) — Consolidacion final: dataset unico hogar-ola
(build_benchmark_consolidado.py)

Une los 12 parquets de features (Personas x9 bloques, Comunidades, Niños,
Choques) mas las variables monetarias (ingreso/gasto/pobreza) en un solo
dataset ancho, join por `llave_compuesta` (mismo esquema de identidad
`consecutivo`/`llave`/`llave_n16` segun ola usado en todo el proyecto) +
`ola`. Este script SOLO consolida (join ancho) -- NO construye todavia la
matriz de entrenamiento train/test (desplazar el outcome de pobreza a la
ola siguiente por hogar y filtrar a poblacion no-pobre en la ola base,
ver metodologia del benchmark puntos 1-2), eso queda como paso siguiente
deliberadamente separado.

**Variables monetarias -- nominal para el label, real para la
covariable de nivel**: `pobreza_monetaria_elca_longitudinal.parquet` trae
ingreso/gasto NOMINAL (correcto para determinar pobreza, comparado contra
LP/LI nominal del mismo año). Para una covariable de NIVEL comparable
entre olas se usa la version REAL con el deflactor "ingresos bajos"
(`_real_ipcbajos` de `ingreso_hogar_elca_longitudinal.parquet`/
`gasto_hogar_elca_longitudinal.parquet`) en vez de IPC total -- mismo
criterio que usa el DANE para actualizar la LP oficial. Se agrega tambien
`brecha_lp_ingreso`/`brecha_lp_gasto` (ingreso o gasto nominal / LP
nominal del mismo año) -- ratio escala-invariante que no necesita
deflactar, siguiendo a Chaudhuri, Jalan y Suryahadi (2002) ya citado en
la metodologia del benchmark.

**Validaciones aplicadas**: `llave_compuesta+ola` unico en la base ancla
(assert); cada merge usa `validate="one_to_one"` y se verifica que el
numero de filas no cambie; se verifica que ningun par de archivos de
features aporte una columna con el MISMO nombre (deteccion de colision
silenciosa). Las 827 columnas RAW de `hogar_elca_longitudinal_clean.parquet`
NO se incorporan al consolidado (nunca fueron auditadas en esta sesion;
solo se usan sus columnas de identidad como ancla del panel).

**Resultado**: 27.932 filas x 128 columnas (6 identidad + 13 monetarias +
109 de features). Sin colisiones de nombre, sin cambio de filas en
ningun merge. Tasa de pobreza por ingreso 60.7%/52.4%/42.4% (declinante,
consistente con la tendencia de reduccion de pobreza en Colombia
2010-2016), `brecha_lp_ingreso` promedio subiendo de 1.24 a 1.75 (hogares
en promedio mas por encima de la LP con el tiempo, consistente). Cobertura
de columnas por archivo de origen entre 21.8% (Niños, esperado -- solo
hogares con niños) y 99.8% (Personas Bloque 1, composicion del hogar).

Output: `data/processed/benchmark_consolidado_elca_longitudinal.parquet`.

Con esto se completa la fase de construccion de features del benchmark.
Pendiente (fuera del alcance de esta sesion): construir la matriz de
entrenamiento train/test desplazando el outcome de pobreza por hogar
entre olas y filtrando a la poblacion no-pobre en la ola base, luego
entrenar y comparar los Modelos A/B (con/sin ingreso) segun la
metodologia ya documentada.

### 2026-08-09 (cont.) — Auditoria completa del modulo Hogar (827 columnas,
mismo rigor que Personas/Comunidades/Niños)

A raiz de la matriz de control de variables (ver seccion siguiente), el
modulo Hogar habia quedado marcado como `CANDIDATO_NO_EVALUADO` -- nunca
paso por el mismo proceso de 4 etapas que los demas modulos. El usuario
pidio auditarlo con el mismo rigor.

**1. Limpieza de corrupcion** (`05_limpieza_corrupcion_hogar.py`, nuevo):
`01_limpieza_base_hogar.py` corrige inconsistencias ESTRUCTURALES entre
olas (armonizacion region/RegionLb), pero la correccion de codificacion
('???'/U+FFFD) se habia hecho AD-HOC en `04_consolidacion_bases_hogar.py`
(solo columnas conocidas), sin el barrido sistematico de 4 capas. Escaneo
completo: 117 columnas con corrupcion residual (92 con U+FFFD, 68 con
"???", con solape), 97 de vocabulario cerrado. Correccion automatica: 86
valores. Rescate via diccionario PDF (`{U,R}Hogar.pdf` 2010/2013): 50
valores mas con match unico. Los ~100 valores restantes se dividieron en
dos grupos: texto libre "accidentalmente cerrado" (plantillas "Otro.
¿Cuál?:_____" con cardinalidad chica por tamaño de muestra, ej.
`con_quien_2/4/5/6/7`, `no_seguros`, `destino_cual_*`) documentado sin
forzar correccion, y reconstrucciones de acento inequivocas corregidas a
mano (`religion1/2`, `destino2013_1..10`, `no_credito_sf1..5`,
`choquec_1/2`, `hizo1c_1/2/3`, etc.). Resultado: **0 valores residuales en
columnas de vocabulario cerrado** tras las 3 capas (verificado). La base
se sobre-escribio en el mismo archivo (`hogar_elca_longitudinal_clean.parquet`)
porque es aditiva (corrige valores dentro de columnas existentes, no
cambia estructura/filas) -- todos los scripts downstream (ingreso, gasto,
pobreza, consolidacion) leen la version ya corregida sin cambios.

**2. Clasificacion completa de las 827 columnas**
(`docs/variable_audit/hogar_construccion.csv`, mismo criterio >1% de
cobertura por ola): 157 `CANDIDATO_BENCHMARK`, 260 `EXCLUIDA_CASI_VACIA`,
226 `EXCLUIDA_NO_EN_OLA1`, 147 `EXCLUIDA_NO_EN_OLA2`, 25
`EXCLUIDA_SOLO_OLA3`, 10 `IDENTIFICADOR`, 2 `EXCLUIDA_OTRO_PATRON`
(`t_hogar`, `jovenes_accion`: verificado que tienen 100% cobertura en ola
1 Y ola 3 pero 0% en ola 2 -- pregunta genuinamente ausente del
cuestionario de 2013, no un error; fallan Eje 1 igual que las demas
exclusiones de este tipo). Suma verificada: 827.

**3. Busqueda de renombrados entre olas** (nombre relajado cutoff 0.55 +
diccionarios PDF de `elca_{2010,2013}/{U,R}Hogar.pdf`, mismo metodo de 2
pasadas): **sin hallazgos de alto valor**. Los pares con mayor similitud
resultaron ser:
  - `fexhog`/`fhog` (2010) vs `fexhog_2010`/`fexhog_2013`/`fhog_2010`/
    `fhog_2013` (2013): factores de expansion e indice de riqueza
    (metadato de diseño muestral, confirmado por texto del diccionario
    "factor indice longitudinal puntaje riqueza") -- mismo patron que
    `fpers`/`fexpers` en Personas, no es contenido sustantivo.
  - La bateria completa de `act_*` (tipos de activos: bonos, inversiones,
    prestamos, oficina, etc.) genera decenas de coincidencias de nombre Y
    de cobertura numerica identica entre si -- confirmado que es
    coincidencia espuria por compartir la MISMA poblacion filtrada
    (mismo patron ya visto en Personas/Comunidades: muchos items de una
    misma bateria comparten cobertura exacta sin ser la misma pregunta).
  - `ayu_emergencias`/`sub_desempleo`/`caja_subsprest` vs
    `guardabosques`/`alianz_prod`/`oport_rural`/`leydevictimas`/etc.:
    alta similitud de texto porque comparten el mismo enunciado base
    ("¿Algún miembro del hogar fue beneficiario de...?") pero son
    PROGRAMAS SOCIALES COLOMBIANOS DISTINTOS con nombre propio (Guardabosques,
    Alianzas Productivas, Oportunidades Rurales, Ley de Víctimas) -- no
    son renombrados, son categorias diferentes de una misma bateria.

**4. Resultado -- 131 candidatas de alto valor identificadas, aun NO
construidas.** De las 157 `CANDIDATO_BENCHMARK`, 131 pasan ademas el
umbral >=10% de cobertura en ambas olas (mismo criterio que el resto del
proyecto) y cubren temas centrales de vulnerabilidad AUSENTES del
consolidado actual:
  - **Vivienda**: `material_paredes`, `material_pisos`,
    `tenencia_vivienda`, `tipo_vivienda`, `servicio_sanitario`,
    `obtencion_agua`, `energia_cocinan`, `eliminan_basura`,
    `t_cuartos_hogar`/`t_cuartos_dormir` (hacinamiento), `sp_acueducto`/
    `alcantarillado`/`energia`/`gasnatural`/`telefono`/`recoleccion_basura`
    (servicios publicos).
  - **Activos/riqueza**: bateria de bienes durables (`n_neveras`,
    `n_lavadoras`, `n_television*`, `n_computadores`, `n_internet`,
    `automoviles`, `motocicletas`, etc.) y **`riqueza_pca`, un indice de
    riqueza YA CALCULADO por componentes principales** (100% cobertura
    las 3 olas) -- candidato directo a covariable de nivel socioeconomico
    complementaria al ingreso monetario.
  - **Programas sociales**: `familias_accion`, `red_juntos`, `sena`,
    `icbf`, `prg_adultomayor` -- beneficiario de los principales
    programas de proteccion social colombianos, altamente relevantes
    para un modelo de pobreza.
  - **Choques/desastres a nivel hogar** (distinto del modulo Choques ya
    construido, que es auto-reportado por categoria).
  - **Acceso a credito**: `credito_financiera`, `credito_cooperativa`,
    `credito_fna`, `subsidios`, `recursos_propios`, `prestamo_familiar`.
  - **Ayudas/remesas recibidas**: `ayu_alimentos`, `ayu_fam_colom`,
    `ayu_fam_ext`, `ayu_ong`, `ayu_religiosas`, `ayu_desplazados`.

Estas 131 candidatas quedan documentadas en la matriz de control
(`data/processed/matriz_control_variables_ELCA.xlsx`, hoja "Hogar") con
razon "candidata valida, aun no construida" -- pendiente de decision del
usuario sobre si construir un bloque `build_hogar_features.py` para
incorporarlas al benchmark antes de la consolidacion final, dado que
enriquecerian sustancialmente el modelo mas alla de ingreso/gasto/pobreza
ya integrados.

### 2026-08-09 (cont.) — Verificacion de "pregunta filtro oculta" en las
candidatas de baja cobertura + documentacion de 2 casos especiales
(riqueza_pca, estrato/sp_estrato)

**1. Verificacion sistematica de filtro oculto.** El usuario pregunto
explicitamente como estar seguros de que las candidatas con cobertura
<10% (excluidas) no son en realidad preguntas condicionales con
cobertura ALTA dentro de su poblacion aplicable (mismo riesgo que ya
habia aparecido con discapacidad en Personas). Se hicieron 2 pruebas:

  - **Prueba de "variable puerta"**: para columnas de detalle/valor con
    una pregunta condicional plausible (ej. `ayu_fe_vr` = "¿cuánto
    recibió de ayuda de familiares en el exterior?"), se busco la
    variable-puerta Sí/No correspondiente (`ayu_fam_ext`) y se calculo
    que % de quienes respondieron "Sí" tambien respondieron el detalle.
    Resultado: 100% en los 7 pares revisados (`ayu_fe_vr`/`ayu_fam_ext`,
    `ayu_ali_vr`/`ayu_alimentos`, `ayu_ong_vr`/`ayu_ong`,
    `act_herencias_vr`/`act_herencias`, `act_otrosing_vr`/
    `act_otrosing_cual`/`act_otrosing`) -- confirma que la baja cobertura
    es por baja PREVALENCIA real del evento (pocos hogares reciben
    herencias, ayuda de ONG, etc.), no por hueco de datos. Las
    variables-puerta correspondientes YA estan en las 131 candidatas de
    alta cobertura.
  - **Inspeccion de colas numeradas**: el resto de las candidatas de baja
    cobertura son la 3ª a 16ª ocurrencia de bloques repetidos (prestamos
    `con_quien_3..14`, acreedores `aquien_deben_5..16`, choques
    comunitarios `choquec_4/5`) -- genuina rareza de tener multiples
    prestamos/acreedores simultaneos, mismo patron ya documentado en
    Comunidades (lideres #5/#6, conflictos #7-12).
  - **Unica excepcion real encontrada**: `ing_ayudas` (9% cobertura
    aparente ola 1 vs. 100% ola 2/3) -- YA investigado a fondo en una
    sesion anterior (ver `build_ingreso_hogar.py`, "AUDITORÍA
    2026-08-06"): tres fuentes cruzadas (`.tab` crudo rural, diccionario
    `RHogar.pdf`, comparacion de letra de item con `UHogar.pdf`)
    confirmaron que la pregunta "e. Ayudas en dinero" NUNCA se hizo a
    hogares RURALES en 2010 -- no es dato recuperable, y ya esta tratado
    como 0 implicito en `ingreso_total_hogar` con la limitacion
    documentada.

  Conclusion: la regla de cobertura cruda (`.notna()`) NO esta siendo
  engañada por tokens de cero en texto (ej. "No Recibe" en `ing_ayudas`
  SI se cuenta como respuesta valida, confirmado). El unico mecanismo que
  si podria ocultar cobertura real (una pregunta condicional cuya puerta
  tiene alta cobertura) fue puesto a prueba explicitamente y no aparecio
  en ningun caso revisado.

**2. Dos casos especiales, documentados SIN sacarlos de las candidatas**
(pedido explicito del usuario: no remover variables ya calculadas, solo
documentar bien):

  - **`riqueza_pca`**: NO es una pregunta de encuesta -- es un indice YA
    CALCULADO por ELCA ("Puntaje del índice de riqueza calculado con
    activos del hogar", confirmado en el diccionario), via Analisis de
    Componentes Principales sobre la MISMA bateria de activos durables
    (`n_neveras`, `n_lavadoras`, `n_television*`, etc.) que tambien
    aparece como columnas individuales en las 131 candidatas. Aparece en
    el diccionario junto a `fexhog` (factor de expansion/indice de
    riqueza, ya excluido por no existir en ola 2) y `tercil2010`/
    `tercil2013` (version discretizada en terciles del mismo indice, ya
    excluidas por existir cada una en una sola ola). Riesgo de
    colinealidad si se usa junto con los activos individuales crudos --
    decision de modelado a tomar en la etapa de seleccion de variables,
    no en esta auditoria. Se mantiene en las candidatas.
  - **`estrato` vs. `sp_estrato`**: dos mediciones del MISMO concepto
    (estrato socioeconomico de la vivienda) por metodos distintos,
    confirmado en el diccionario 2010:
      - `estrato` (HU8): pregunta temprana, antes del modulo detallado de
        servicios publicos, sin especificar metodo de verificacion
        (probablemente registro rapido). Cobertura 53%/50%/49%.
      - `sp_estrato` (HU17): dentro del modulo de servicios publicos,
        pide verificar CONTRA EL RECIBO DE ENERGIA ELECTRICA -- metodo
        documentado/objetivo. Cobertura 53%/100%/100%.
    Se mantienen AMBAS (pedido explicito del usuario): no son redundantes
    en el sentido de eliminar una, son dos mediciones con confiabilidad
    distinta, utiles para chequeo de consistencia entre si.

**3. Correccion menor: 2 identificadores que se habian colado en las
candidatas.** `consecutivo_c` (llave de comunidad, ya usada como tal en
`build_comunidades_hogar.py`) e `id_mpioU` (probable codigo
administrativo de municipio, mismo tipo de cautela ya documentada para
`id_dpto`/`id_mpio` de hogar -- posible identificador anonimizado sin
correspondencia directa con DIVIPOLA) pasaban el filtro automatico de
cobertura por tener datos en las 3 olas, pero no son contenido
sustantivo. Re-clasificadas como `IDENTIFICADOR`. Total de candidatas
reales: **155** (antes 157), de las cuales 131 pasan ademas el umbral de
10% de cobertura.

La matriz de control (`matriz_control_variables_ELCA.xlsx`) se actualizo
con una columna nueva, "Notas metodológicas", que documenta estos casos
sin eliminarlos del listado.

### 2026-08-09 (cont.) — Bloque de features de Hogar (build_hogar_features.py)

Construccion del bloque de features con las 129 candidatas de Hogar
(131 originales menos `consecutivo_c`/`id_mpioU`, re-clasificadas como
identificadores).

**Hallazgos de calidad de dato encontrados al verificar antes de
construir** (misma disciplina aplicada en todos los modulos anteriores):
  - **Bateria `act_*` (15 tipos de activos financieros)**: ola 1 usa
    CODIGOS NUMERICOS ("1"/"2") sin texto, ola 2/3 usan texto Sí/No.
    Verificado cruzando proporciones (~99% de "2" en ola 1 vs ~99% de
    "No" en ola 2/3): confirma 1=Sí/2=No (mismo codigo usado en el resto
    de ELCA). Normalizado antes de construir.
  - **`n_internet`**: mezcla CONTEOS numericos ("0") con texto Sí/No en
    la misma columna -- colapsado a binario.
  - **`con_quien_1`/`con_quien_2`/`con_quien_3`**: tenian corrupcion
    U+FFFD/"???" residual NO detectada en el escaneo original porque su
    cardinalidad (25/26/26) cae justo en el limite de
    `CARDINALIDAD_MAXIMA_CERRADA=25` -- el barrido automatico las trata
    como texto libre y nunca las evalua. Corregido directamente en
    `05_limpieza_corrupcion_hogar.py` (ver su docstring, correccion
    2026-08-09) -- leccion: el umbral de cardinalidad es heuristica, no
    garantia, columnas cerca del limite deben revisarse a mano.
  - **`credito_financiera`/`credito_cooperativa`/`credito_fna`/
    `otra_financiacion`/`recursos_propios`/`prestamo_familiar`/
    `subsidios`/`dcto_vivienda`**: el nombre sugiere credito general,
    pero el diccionario (HU55-56, pregunta 20: "¿Cuáles de las
    siguientes fuentes de financiación utilizaron para la COMPRA O
    CONSTRUCCIÓN DE ESTA VIVIENDA?") confirma que es especificamente
    financiacion de VIVIENDA -- distinto de `con_quien_1/2`
    (endeudamiento general: bancos, amigos, prestamistas, tenderos,
    verificado con las categorias reales de la columna).
  - **`eay_*` vs `ayu_*`**: el diccionario (HU178-179, pregunta 36:
    "¿algún miembro de este hogar ENVIÓ ayuda...?") confirma que `eay_*`
    es ayuda ENVIADA por el hogar (señal de capacidad economica
    relativa), no "ayuda esperada" como sugeriria el nombre -- distinto
    de `ayu_*` (ayuda RECIBIDA, señal de vulnerabilidad). Se construyen
    ambos por ser complementarios, no redundantes.
  - **`uay_*`**: el diccionario (pregunta 35: "Las ayudas... fueron
    utilizadas para...") confirma que es el USO de la ayuda RECIBIDA
    (condicional a `ayu_*`), coherente con su cobertura mas baja.

**Caveat de escala encontrado en validacion**: `n_activos_financieros_hogar`
sube de 0.37 (ola 1) a 0.64 (ola 2) -- investigado: 5 de los 15 `act_*`
se preguntaron SOLO a zona urbana en ola 1 (0% rural, verificado) y a
ambas zonas en ola 2; a la inversa, otros 5 se preguntaron a ambas zonas
en ola 1 pero solo urbana en ola 2. El composite se calcula
correctamente sobre los items disponibles por hogar, pero el NUMERO de
items "preguntables" varia por ola/zona -- el conteo no es perfectamente
comparable en magnitud entre olas, mismo tipo de limitacion ya
documentada para otras variables compuestas (TVIP en Niños).

**Variables construidas**: 61 columnas -- vivienda (tenencia, tipo,
materiales, servicios sanitarios/agua/energia/basura, hacinamiento vía
personas por cuarto/dormitorio, `n_servicios_publicos_hogar`,
`estrato_hogar`/`estrato_verificado_hogar` ambas mantenidas por pedido
explicito), activos/riqueza (`riqueza_pca_hogar` pass-through del indice
ya calculado por ELCA, `n_bienes_durables_hogar`, `tiene_vehiculo_hogar`,
`tiene_internet_hogar`, `n_activos_financieros_hogar`,
`tiene_propiedad_rural_hogar`, `tiene_transporte_carga_hogar`,
`tiene_ingreso_agropecuario_hogar`), programas sociales
(`beneficiario_familias_accion_hogar`, `beneficiario_red_juntos_hogar`,
`n_programas_sociales_hogar`, `beneficiario_algun_programa_hogar`),
desastres (`tuvo_desastre_natural_hogar`), financiacion de vivienda
(`financio_credito_formal_vivienda_hogar`,
`financio_recursos_propios_vivienda_hogar`,
`financio_subsidio_vivienda_hogar`, `financio_otra_fuente_vivienda_hogar`,
`tiene_escritura_vivienda_hogar`, `tiene_titulo_baldio_hogar`,
`valor_arriendo_pagado_hogar`), endeudamiento general
(`tiene_deuda_hogar`, `deuda_formal_hogar`, `deuda_informal_hogar`),
ayudas recibidas/enviadas/uso (14 variables), y `practica_religion_hogar`.

**Verificacion final de cobertura completa**: de las 129 candidatas, 103
se usan (directas o en composites), 26 se excluyen con razon documentada
(5 redundantes con ingreso ya construido, 2 redundantes con gasto, 12
detalle de prestamo demasiado granular ya capturado en
formal/informal, 2 valores condicionales cuya variable-puerta ya esta
capturada, 3 categorias de uso de ayuda con prevalencia negligible).

**Validacion**: hacinamiento declinante (1.65->1.39 personas/cuarto),
tenencia de vehiculo creciente (26%->44%), `riqueza_pca_hogar` con
tendencia positiva entre olas, `beneficiario_red_juntos_hogar` con perfil
de ciclo de vida del programa (1.7%->7.9%->4.9%, consistente con el
lanzamiento y transicion de Red Juntos a Red Unidos), uso de ayuda
dominado por alimentos (78%-91%), religiosidad alta (91%-97%,
consistente con Colombia) -- sin anomalias sin explicar.

Output: `data/processed/hogar_features_elca_longitudinal.parquet` (27.932
filas, mismo panel que el resto del proyecto).

Con esto, el modulo Hogar queda con el mismo nivel de auditoria y
construccion de features que Personas/Comunidades/Niños/Choques.

### 2026-08-09 (cont.) — Re-consolidacion final con Hogar incluido

`hogar_features_elca_longitudinal.parquet` (56 columnas de contenido) se
agrego a `FEATURE_PARQUETS` en `build_benchmark_consolidado.py` y se
re-corrio la consolidacion completa. Resultado: **27.932 filas x 184
columnas** (antes 128; +56 de Hogar). Sin colisiones de nombre, sin
cambio en el numero de filas en ningun merge (validado con
`validate="one_to_one"` como en la corrida anterior). Cobertura promedio
de las columnas de Hogar en el consolidado: 81.4%.

Con esto, los 5 modulos (Personas, Comunidades, Niños, Choques, Hogar)
mas ingreso/gasto/pobreza quedan unidos en `benchmark_consolidado_elca_longitudinal.parquet`,
todos auditados al mismo nivel de rigor. Pendiente (fuera del alcance de
esta sesion): construir la matriz de entrenamiento train/test
desplazando el outcome de pobreza por hogar entre olas y filtrando a la
poblacion no-pobre en la ola base, luego entrenar y comparar los Modelos
A/B (con/sin ingreso) segun la metodologia ya documentada.

### 2026-08-09 (cont.) — Matriz de entrenamiento/prueba del benchmark
(build_benchmark_train_test.py) y primer entrenamiento Modelo A vs. B
(entrenar_benchmark.py)

**Construccion de la matriz** (`src/05_model/build_benchmark_train_test.py`):
emparejamiento de hogares entre olas por `consecutivo`, **solo matches 1
a 1** -- misma politica ya confirmada y usada en las matrices de
transicion de `build_pobreza_desagregaciones.py` (hogares que se
dividieron se excluyen, se reporta el conteo). Poblacion: no-pobres en la
ola base (`pobre_ingreso==False`). Outcome `Y` = pobre en la ola
siguiente.

Resultados de la construccion:
  - **2010->2013** (train principal): 8.218 hogares en panel 1 a 1 (511
    excluidos por division), 3.089 no-pobres en 2010, tasa de entrada a
    pobreza 23.4%.
  - **2013->2016** (test principal): 6.911 hogares en panel 1 a 1 (1.190
    excluidos por division -- mas division que 2010->2013, esperable con
    mas tiempo transcurrido), 3.191 no-pobres en 2013, tasa de entrada a
    pobreza 19.9%.

Modelo A (169 covariables, incluye `ingreso_percapita_hogar_real`/
`gasto_percapita_hogar_real`/`brecha_lp_ingreso`/`brecha_lp_gasto`) vs.
Modelo B (165 covariables, sin esas 4). `pobre_ingreso`/`lp`/`li`/etc. de
la ola base se excluyen de AMBOS por ser constantes dentro de la
poblacion no-pobre (no es fuga de outcome, el outcome es la ola
siguiente) -- no aportan varianza, no por riesgo de fuga.

**Bug encontrado y corregido antes del entrenamiento**: `consecutivo_c`
(identificador de COMUNIDAD, codigo numerico sin significado ordinal) y
`llave_compuesta` no estaban excluidos de las covariables -- solo
`consecutivo` (ID de hogar) lo estaba. `consecutivo_c` aparecio entre las
10 variables mas importantes del Modelo B en una primera corrida. Se
verifico: 663 de ~715-735 comunidades se repiten entre train y test
(mismo panel longitudinal de ELCA) -- el modelo podia estar explotando
un ID arbitrario como si tuviera significado ordinal en vez de capturar
geografia de forma principiada (para eso ya estan `zona` y el bloque
completo de variables de contenido de Comunidades). Se agrego
`consecutivo_c`/`llave_compuesta` a las columnas excluidas y se
re-entreno -- los resultados cambiaron minimamente (AUC-ROC de 0.758 a
0.757 en Modelo A, de 0.739 a 0.738 en Modelo B), confirmando que el ID
no aportaba señal real, solo ruido con apariencia de importancia.

**Algoritmo**: `HistGradientBoostingClassifier` (sklearn) en vez de
regresion logistica -- las covariables tienen missingness ESTRUCTURAL
alta y heterogenea (~89%-99% NaN en variables de Niños para hogares sin
niños, ~83% en variables condicionales de choques); imputar con un valor
generico para 165-169 columnas con estructuras de missingness tan
distintas introduciria supuestos arbitrarios por variable. Se uso un
modelo que maneja NaN nativamente en la particion de arboles (el
missingness se vuelve señal explotable, ej. "sin dato de vacunacion
infantil" ya proxy-codifica "no tiene hijos pequeños", en vez de forzarse
una imputacion). Columnas categoricas via dtype `category` +
`categorical_features="from_dtype"` (soporte nativo, sin one-hot).
`class_weight="balanced"` dado el desbalance moderado (~20%-23% tasa
positiva).

**Resultado principal** (holdout temporal hacia adelante, train
2010->2013, test 2013->2016):

| Modelo | Covariables | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| A (con ingreso) | 169 | 0.757 | 0.609 | 0.368 | 0.459 |
| B (sin ingreso) | 165 | 0.738 | 0.584 | 0.354 | 0.441 |

**Interpretacion (responde a la pregunta de la metodologia)**: la brecha
de AUC-ROC entre A y B es de solo 0.019 -- el Modelo A NO domina casi
por completo al B. Esto sugiere que las covariables NO monetarias
(educacion, vivienda/hacinamiento, activos, formalidad laboral,
estructura demografica) capturan una fraccion sustancial del poder
predictivo de la brecha a la LP, no son redundantes con ella. Variables
mas importantes en A: `ingreso_percapita_hogar_real` (dominante, como
se espera del enfoque Chaudhuri et al. 2002), `riqueza_pca_hogar`,
`nivel_educ_max_hogar`, `brecha_lp_ingreso`, `n_bienes_durables_hogar`.
En B: `nivel_educ_max_hogar`, `riqueza_pca_hogar`,
`personas_por_cuarto_hogar` (hacinamiento), `n_activos_financieros_hogar`,
`tasa_cotizacion_pension_hogar` (formalidad laboral) --
consistentes con la literatura de determinantes de pobreza, sin
variables espurias entre las principales.

Output: `data/processed/benchmark_train_test/` (4 parquets: Modelo A/B x
2 transiciones) y `data/processed/benchmark_resultados/` (metricas +
importancia de variables por modelo).

**Pendiente** (fuera del alcance de esta sesion): especificaciones de
robustez (reversa: train 2013->2016/test 2010->2013; pooled con k-fold
agrupado por hogar), ajuste de hiperparametros dentro del periodo de
entrenamiento (sin tocar el test hasta la evaluacion final, ya
establecido en la metodologia pero no ejecutado aun), y la comparacion
final contra el modelo enriquecido con variables de Google Street View.

## 2026-08-09: Suite de comparacion de algoritmos -- logistica regularizada (benchmark), Random Forest, XGBoost, LightGBM, HistGradientBoosting

Reemplaza el entrenamiento unico de `entrenar_benchmark.py` (eliminado) por
una suite de 5 algoritmos, cada uno en su propio script (`src/05_model/
modelo_<algoritmo>.py`), resultados en carpeta propia
(`data/processed/benchmark_resultados/<algoritmo>/`), y un registro
transversal `registro_modelos.xlsx` (regenerado desde `registro_modelos.csv`,
fuente de verdad, upsert por algoritmo+especificacion) con metricas,
hiperparametros y observaciones de cada corrida. Logica compartida en
`src/05_model/modelo_utils.py`.

**El usuario designo la regresion logistica con regularizacion como
benchmark**, a comparar contra XGBoost, LightGBM (pedidos explicitamente) y
Random Forest (propuesto como "otro algoritmo recomendado", punto medio
entre la logistica lineal y los 3 gradient boosting).

**Decisiones metodologicas -- todas CONFIRMADAS explicitamente con el
usuario antes de implementar** (el usuario detuvo una primera corrida por
asumir demasiado sin consultar; ver mas abajo):

1. **Imputacion de missings** (solo Logistica y Random Forest -- los unicos
   sin soporte nativo de NaN; XGBoost/LightGBM/HistGB no imputan, ver
   entrada anterior): **0 + indicador de faltante, siempre, para toda
   variable numerica** (nunca la mediana). Razon (objecion del usuario que
   cambio la recomendacion inicial): gran parte del missingness esta
   CAUSADO por una pregunta filtro -- la mediana de una columna asi se
   calcula solo sobre quienes pasaron el filtro y respondieron, y
   asignarsela a los excluidos por el filtro les inventa un valor "tipico"
   que no tienen, contradiciendo la logica del filtro. 0 es mas seguro
   porque en un modelo lineal `0 x coeficiente = 0` siempre -- el relleno
   no aporta ninguna contribucion mas alla de lo que capture el indicador,
   que es quien absorbe el efecto de "esto no aplica"; con la mediana, el
   mismo coeficiente que debe ajustar la pendiente real sobre quienes SI
   respondieron queda contaminado por tener que explicar el valor
   inventado del grupo filtrado. Evita ademas clasificar ~165 covariables
   una por una segun si su missingness admite un "cero real". Categoricas:
   categoria explicita "Sin dato" (no moda). One-hot con `drop="first"`
   para Logistica/RF; estandarizacion solo para Logistica (regularizacion
   penaliza magnitud, dependiente de escala).

2. **Comparacion de estrategias de balanceo de clases** (CONFIRMADO): se
   comparan 3 estrategias por AUC-ROC en CV -- `balanced` (reweighting,
   `class_weight`/`scale_pos_weight`), `ninguno` (baseline sin ajuste),
   `oversampling` (`RandomOverSampler` de imbalanced-learn, dentro de cada
   fold via `imblearn.pipeline.Pipeline` para no filtrar informacion del
   fold de validacion). SMOTE se descarto (el usuario no lo selecciono):
   interpola vecinos, mal definido con NaN/categoricas mezcladas;
   RandomOverSampler solo duplica filas, compatible con cualquier tipo.

3. **Metrica de seleccion** (CONFIRMADO): AUC-ROC -- no depende de un
   umbral de clasificacion, y el oversampling distorsiona la calibracion
   de las probabilidades (comparar F1/recall a umbral fijo entre
   estrategias de balanceo distintas no seria comparacion justa).
   Recall/precision/F1 se reportan en paralelo, sin decidir la seleccion.

4. **Busqueda de hiperparametros** (CONFIRMADO): `RandomizedSearchCV`,
   `n_iter=15`, `cv=3` (StratifiedKFold) -- valores de costo computacional,
   no consultados explicitamente, documentados para poder ajustarse.

**Hallazgo de la primera corrida y correccion de umbral** (CONFIRMADO
con el usuario tras mostrarle el problema): XGBoost (A y B) y LightGBM-B
ganaron la comparacion de balanceo con "ninguno" por una diferencia de
AUC-CV insignificante (~0.001, ruido estadistico) frente a "balanced" --
pero sin reponderar clases (~23% tasa positiva), las probabilidades
salen tan comprimidas hacia 0 que casi ninguna fila cruzaba el umbral
fijo de 0.5 (recall 0.03-0.04 en XGBoost, con precision >0.6 -- el modelo
ordenaba bien pero era inservible al umbral estandar). Se agrego
`elegir_umbral_por_cv` (`modelo_utils.py`): probabilidades out-of-fold
via `cross_val_predict` sobre el estimador ya elegido (balanceo +
hiperparametros), se escanea una grilla y se elige el umbral que
maximiza F1 -- AUC-ROC sigue mandando la seleccion previa de
balanceo/hiperparametros, esto solo corrige como se reportan
recall/precision/F1. Tras la correccion, XGBoost-A paso de recall 0.038
a 0.671 (umbral 0.28) sin cambiar el modelo elegido.

**Resultado final** (holdout temporal principal, train 2010->2013, test
2013->2016; umbral de clasificacion elegido por CV, no fijo en 0.5):

| Algoritmo | Espec. | Balanceo | Umbral | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| Random Forest | A | balanced | 0.52 | 0.764 | 0.665 | 0.364 | 0.470 |
| XGBoost | A | ninguno | 0.28 | 0.762 | 0.671 | 0.367 | 0.475 |
| HistGradientBoosting | A | ninguno | 0.23 | 0.759 | 0.745 | 0.334 | 0.461 |
| LightGBM | A | balanced | 0.55 | 0.757 | 0.625 | 0.372 | 0.467 |
| Logistica regularizada (benchmark) | A | balanced | 0.52 | 0.747 | 0.665 | 0.334 | 0.445 |
| HistGradientBoosting | B | balanced | 0.44 | 0.740 | 0.715 | 0.322 | 0.444 |
| Logistica regularizada (benchmark) | B | balanced | 0.48 | 0.739 | 0.704 | 0.313 | 0.433 |
| Random Forest | B | balanced | 0.49 | 0.739 | 0.674 | 0.335 | 0.447 |
| XGBoost | B | ninguno | 0.27 | 0.735 | 0.658 | 0.335 | 0.444 |
| LightGBM | B | ninguno | 0.23 | 0.727 | 0.742 | 0.308 | 0.435 |

Los 5 algoritmos quedan en un rango estrecho de AUC-ROC (0.727-0.764) --
los gradient boosting (RF/XGBoost/HistGB) superan marginalmente a la
logistica regularizada, pero la brecha es pequeña (~0.01-0.02 en Modelo
A), consistente con que la señal predictiva principal esta en variables
con relacion mayormente monotona/aditiva con el outcome (ingreso,
educacion, activos) que la logistica captura razonablemente bien incluso
sin interacciones. **Pendiente, explicitamente diferido por el usuario**:
decidir cual algoritmo (y cual especificacion A/B) es "el" benchmark
final para la comparacion contra el modelo enriquecido con Google Street
View.

Output: `src/05_model/modelo_utils.py` (logica compartida),
`src/05_model/modelo_{logistica_regularizada,random_forest,xgboost,
lightgbm,histgradientboosting}.py`, `data/processed/benchmark_resultados/
<algoritmo>/` (coeficientes o importancia de variables por modelo+especificacion),
`data/processed/benchmark_resultados/registro_modelos.{csv,xlsx}`.

## 2026-08-10: Re-corrida de la suite con multiples semillas + intervalos de confianza; paper actualizado

Re-corrida de los 5 modelos con `evaluar_multiples_semillas` (5 semillas,
balanceo/hiperparametros ya fijados por la corrida anterior, ver docstring
de `modelo_utils.py`). Hallazgo relevante: HistGradientBoosting, LightGBM
y la logistica regularizada resultan practicamente DETERMINISTICOS bajo
los hiperparametros elegidos (desviacion estandar del AUC-ROC entre
semillas ~0.0000) -- no usan submuestreo de filas/columnas en su
configuracion optima, asi que el `random_state` no tiene ningun efecto.
Random Forest y XGBoost si tienen variabilidad pequeña pero no nula (std
0.0005 y 0.0011 respectivamente), heredada del bootstrap/subsample que si
esta activo en sus hiperparametros ganadores.

**Resultado actualizado** (media, 5 semillas; ver tabla completa en
`registro_modelos.csv`):

| Algoritmo | Espec. | Balanceo | Umbral (media) | AUC-ROC (media) | IC95% |
|---|---|---|---|---|---|
| Random Forest | A | balanced | 0.48 | 0.764 | [0.763, 0.764] |
| XGBoost | A | ninguno | 0.26 | 0.763 | [0.761, 0.764] |
| HistGradientBoosting | A | ninguno | 0.23 | 0.759 | [0.759, 0.759] |
| LightGBM | A | balanced | 0.53 | 0.758 | [0.758, 0.758] |
| Logistica regularizada | A | balanced | 0.46 | 0.747 | [0.747, 0.747] |
| HistGradientBoosting | B | balanced | 0.47 | 0.740 | [0.740, 0.740] |
| Logistica regularizada | B | balanced | 0.48 | 0.739 | [0.739, 0.739] |
| Random Forest | B | balanced | 0.48 | 0.739 | [0.738, 0.740] |
| XGBoost | B | ninguno | 0.27 | 0.734 | [0.732, 0.735] |
| LightGBM | B | ninguno | 0.25 | 0.727 | [0.727, 0.727] |

Los numeros de AUC-ROC media casi no cambian frente a la corrida de
semilla unica (2026-08-09) -- confirma que esa corrida no era un
artefacto de una semilla particular. Hallazgo nuevo: en la especificacion
B el patron "gradient boosting/RF superan a la logistica" NO se sostiene
-- XGBoost-B (0.734) y LightGBM-B (0.727) quedan POR DEBAJO de la
logistica regularizada-B (0.739), solo HistGradientBoosting-B (0.740) y
Random Forest-B (empatado, 0.739) igualan o superan. El IC95% reportado
mide solo la aleatoriedad del propio procedimiento de ajuste sobre el
mismo test set (no la incertidumbre muestral del AUC-ROC en si) -- no
sustituye una prueba pareada formal (ej. DeLong) para comparar
algoritmos, que queda pendiente.

**Paper actualizado** (`paper/main.tex`): Tabla de desempeño (columna
IC95% agregada), Figuras 01/02 de `graf_resultados_modelos.py`
regeneradas con barras de error, parrafo de analisis en
Seccion~5.2 corregido (ya no afirma que gradient boosting supera
"consistentemente" a la logistica -- eso solo es cierto en Modelo A),
brecha A-vs-B actualizada a 0.008-0.031 (antes 0.017-0.030), item de
Limitaciones sobre "cuantificar incertidumbre" marcado como resuelto y
reemplazado por "prueba formal de diferencias entre algoritmos"
(pendiente). PDF compila limpio, 38 paginas, sin overfull ni referencias
indefinidas.

## 2026-08-25: Evaluacion de comparabilidad ELCA -> ELCO (2019/2022) antes de pedir acceso a los datos

Antes de solicitar permiso de acceso y procesar ELCO 2019/2022 para
extender el panel a la transicion 2016->2019 (con 2019->2022 como
generalizacion), se evaluo si las variables de ELCA (2010/2013/2016) son
identificables e interpretables igual en ELCO, dado que el DANE cambio de
administrador y de metodologia del cuestionario a partir de 2019.
Herramienta construida para esto: `src/01_download/01_descarga_ELCA/
10_diccionario_elco.py`, que parsea `documentacion_ELCO.pdf` (711
paginas) a una tabla codigo->pregunta->modulo buscable (`data/processed/
diccionario_elco.parquet`), en vez de buscar a mano.

**Resuelto y verificado:**

- Ingreso laboral (`ingreso_laboral` en `build_ingreso_hogar_elco.py`):
  logica de `coalesce()` de las 3 rutas (P158 asalariados / P6749S1-S2
  independientes / P7422S1 desocupados) verificada contra el diseno de
  skip-logic real de `Formulario_Seguimiento_ELCO_2022.pdf` (paginas
  86-91) -- las rutas son mutuamente excluyentes por diseno del
  cuestionario, no un supuesto.
- Ingreso directo y excepcional/ocasional: mapeados y YA CODIFICADOS en
  `build_ingreso_hogar_elco.py`. Hallazgo relevante: ELCO NO desagrega
  herencias/polizas/venta de inmueble/venta de negocio/otros como ELCA
  (6 preguntas independientes de 12 meses) -- las bundlea en una sola
  pregunta catch-all (P2375S6/P2375S6A1, ya mapeada como
  `otras_fuentes`). Se pierde la desagregacion por tipo, no el total.
- 4 de 5 variables prioritarias no monetarias mapeadas: `pct_ninos_
  madre_viva`->P6083, `pct_ninos_padre_vivo`->P6081, `pct_ninos_
  control_pediatrico`->P2286, `pct_ninos_oficios_hogar`->P2298S1-S8
  (bateria "De los siguientes oficios...", mismo diseno que ELCA).
  `problema_homicidios_comunidad` NO tiene equivalente: viene del modulo
  "Comunidades" de ELCA (entrevista a lideres comunitarios), que ELCO no
  tiene -- confirmado que es ausencia real de la fuente, no un problema
  de busqueda.
- Gasto del hogar: estructuralmente comparable -- `F_GASTOS DEL HOGAR`/
  `F_CICLO GASTOS DEL HOGAR` usa el mismo diseno de periodicidad de
  recordacion que ELCA (15 dias / 3 meses / 12 meses), categorias de
  articulo equivalentes. Construirlo requiere el mismo esfuerzo de
  auditoria articulo-por-articulo que tomo `build_gasto_hogar.py` en
  ELCA -- viable, no iniciado.
- Pobreza monetaria -- LP/LI: la clasificacion pobre/no-pobre siempre es
  nominal-contra-nominal del mismo ano (asi lo hace `build_pobreza_
  monetaria.py`), asi que la LP/LI de cada ano es una serie oficial del
  DANE independiente de ELCA/ELCO, no algo que dependa de la encuesta.
  Verificado con boletines oficiales descargados y guardados en
  `docs/fuentes_dane/`:
    - El DANE recalculo la LP/LI a partir de 2019 con una canasta base
      nueva (ENPH 2016-2017), reemplazando la serie ENIG 2006-2007/MESEP
      que usa el proyecto para 2010/2013/2016. El boletin de 2019 dice
      textualmente: "Estas cifras no son comparables con las cifras de
      la serie MESEP."
    - 2019 y 2022 SI estan en la misma serie ENPH entre si (confirmado
      con 2 boletines independientes -- 2019 y 2023, este ultimo reporta
      2022 en su tabla comparativa -- ambos citan "lineas base ENPH
      2016-2017, actualizadas con el deflactor especial de las lineas de
      pobreza"). Valores agregados a `config_dane.LINEAS_POBREZA_ELCO`.
    - Para la transicion 2016->2019 especificamente, se necesita el 2016
      recalculado bajo ENPH (no el de `LINEAS_POBREZA`, que es MESEP).
      Se descargo el dataset oficial del DANE con esa serie (microdatos.
      dane.gov.co/index.php/catalog/689, archivo `lineas_20122018.csv`,
      copiado en `docs/fuentes_dane/lineas_pobreza_2012_2018_enph/`).
      Dominio RURAL diciembre-2016 confirmado 1:1 con zona="Rural"
      (LP=$196.225, LI=$102.020 -- ~23% mas alto que el valor MESEP que
      usa el proyecto hoy para 2016/Rural, confirma que no son series
      intercambiables). Agregado a `config_dane.LP_2016_ENPH_RURAL`/
      `LI_2016_ENPH_RURAL`.

**Pendiente, sin resolver:**

1. LP/LI 2016-ENPH, dominio Urbano (`config_dane.LP_2016_ENPH_URBANO =
   None`): el archivo fuente del DANE solo trae LP/LI por ciudad
   individual (24 dominios) + "Resto Urbano", sin los pesos
   poblacionales de la GEIH para agregarlos en el equivalente a
   "Cabeceras". Se confirmo que el DANE nunca publico esta cifra
   agregada en un boletin (el primer boletin bajo esta metodologia
   compara 2017->2018, no llega a 2016). Camino viable no explorado:
   calcularlo ponderando por proyecciones oficiales de poblacion del
   DANE por municipio/dominio (dato publico independiente) -- pendiente,
   requiere trabajo adicional, se dejo documentado como calculo
   derivado si se hace (no séria una cita directa de una tabla del
   DANE).
2. Deflactor IPC: `IPC_Variacion.xls` (usado por `build_deflactor_ipc.py`)
   solo cubre metodologia IPC-08 hasta dic-2018. El DANE reemplazo esa
   serie con una base nueva (dic-2018=100) desde enero de 2019 -- falta
   descargar esa serie nueva. Solo afecta series descriptivas en pesos
   reales, NO la clasificacion de pobreza (que es nominal-contra-nominal
   del mismo ano).
3. `build_gasto_hogar_elco.py`: no iniciado (ver hallazgo de
   comparabilidad arriba -- viable, esfuerzo similar al de ELCA).
4. `build_pobreza_monetaria_elco.py`: no iniciado. Insumos ya
   disponibles para 2019/2022 completos; para 2016->2019 falta resolver
   el punto 1 (Urbano-ENPH-2016) antes de poder clasificar pobreza en
   zona urbana para ese ano.
5. Extender `build_ninos_hogar.py`-equivalente en ELCO con las 4
   variables prioritarias ya mapeadas (madre_viva, padre_vivo,
   control_pediatrico, oficios_hogar) -- mapeo hecho, codigo no escrito.
6. Decision de alcance no tomada: si se construye el panel 2016->2019
   (entrenamiento) con evaluacion en 2019->2022 (generalizacion), falta
   decidir explicitamente si esto reemplaza o complementa el panel
   2010->2013->2016 ya existente, antes de invertir en construir todos
   los insumos restantes.

Output (nuevo/modificado en esta sesion): `src/04_features/
build_ingreso_hogar_elco.py` (docstring y comentarios actualizados a
"verificado"), `src/04_features/config_dane.py` (`LINEAS_POBREZA_ELCO`,
`LP_2016_ENPH_RURAL`/`LI_2016_ENPH_RURAL`, `LP_2016_ENPH_URBANO`/
`LI_2016_ENPH_URBANO = None`), `docs/fuentes_dane/README.md` (secciones
1.b y 1.c), `docs/fuentes_dane/Boletin-pobreza-monetaria_2019.pdf`,
`docs/fuentes_dane/bol-PM-2023.pdf`, `docs/fuentes_dane/
lineas_pobreza_2012_2018_enph/lineas_20122018.csv`.

### 2026-08-28 — Construccion del Indice de Pobreza Multidimensional
(IPM-Colombia) sobre el panel ELCA

Motivacion: dentro de la investigacion sobre si DMSP-OLS aporta a la
prediccion de vulnerabilidad (ver `~/Desktop/informe_dmsp_ols_evidencia.pdf`,
documento externo al repo), el usuario pidio probar si el hallazgo de "no
aporta" es especifico de la definicion de pobreza monetaria o se sostiene
tambien con pobreza multidimensional. No existia ningun indicador de
pobreza multidimensional en el repositorio (confirmado por busqueda
exhaustiva) -- se construyo desde cero en
`src/04_features/build_ipm_multidimensional.py`.

**Metodologia**: 5 dimensiones (educacion, niñez y juventud, trabajo,
salud, vivienda y servicios publicos), 15 indicadores, estructura de
ponderacion anidada Alkire-Foster (cada dimension 20%, cada indicador
igual peso dentro de su dimension), punto de corte 33,3% -- definiciones
operativas extraidas palabra por palabra del glosario del Boletin Tecnico
DANE "Pobreza Multidimensional en Colombia" (2018,
`bt_pobreza_multidimensional_18.pdf`, via `pdftotext`, no de memoria).

**Bug real encontrado y corregido**: rezago escolar (niños 7-17) se
construyo primero con `grado_educ` (grado YA COMPLETADO), que tiene
0,4%-7,6% de cobertura entre niños en las 3 olas -- la mayoria de los
niños en edad escolar todavia esta cursando, no "completo" nada
recientemente. La variable correcta es `grado_educ_cursa` (grado que
cursa actualmente), con cobertura 26,6%/91,9%/92,5% (ola 1/2/3).
Corregido.

**Limitacion real, sin arreglo posible, encontrada por investigacion
profunda (no solo diferencia de nomenclatura entre olas)**: 4 de los 15
indicadores dependen de `actividad_ppal` (adultos y niños 12-17) o
`estudia` (niños 6-16) -- ambas variables tienen cobertura severamente
baja en la OLA 1 (2010) especificamente, frente a ~100%/~75-80% en olas
2/3:

  - `actividad_ppal` en niños 12-17: 0,4% cobertura ola 1 vs. ~100% ola 2/3.
  - `estudia` en niños 6-16: 35,3% cobertura ola 1 vs. 99,6% ola 2/3.
  - `actividad_ppal` en adultos: ~20% cobertura ola 1 vs. ~75-80% ola 2/3.

A diferencia de otros huecos de cobertura ya resueltos en esta misma
sesion (nivel educativo en ola 3 -- arrastrado desde la ola anterior de
la misma persona via `llave_ID_lb`; aseguramiento en salud -- variable
mal elegida, `segsoc_salud` en vez de `afiliacion`, corregida; barreras
de acceso a salud en ola 2/3 -- variable correcta encontrada,
`tratar_problema`), este NO tiene arreglo posible: 2010 es la primera
ola del panel, no existe una ola anterior de la cual arrastrar el dato
para esas mismas personas. Se buscaron variables alternativas con el
mismo metodo que resolvio los otros casos, sin encontrar ninguna con
mejor cobertura en ola 1 para actividad economica o asistencia escolar.

**Decision (confirmada con el usuario, 2026-08-28)**: se excluyen del
`ipm_score` los 4 indicadores afectados (`priv_inasistencia_escolar`,
`priv_trabajo_infantil`, `priv_empleo_informal`,
`priv_desempleo_larga_duracion`) mas `priv_primera_infancia` (excluido
por una razon distinta -- ver docstring del script, punto 4: el modulo
de cuidado de primera infancia de la encuesta no se pregunto en ola 3 ni
tiene sub-componente de nutricion en ola 1). Las 5 columnas se calculan
y se guardan igual en el parquet de salida (para auditoria/transparencia,
`COLS_EXCLUIDAS_SCORE` en el script), pero no entran en el score.

Al excluir los 2 unicos indicadores de la dimension "Trabajo", esa
dimension queda vacia -- su 20% de peso se redistribuye entre las 4
dimensiones restantes (Educacion, Niñez y juventud, Salud, Vivienda),
que pasan de 20% a 25% cada una (practica estandar Alkire-Foster cuando
una dimension completa no es medible). El IPM final queda con 10
indicadores activos en el score (de los 15 originales), y la tasa de
pobreza IPM por ola resulto notablemente mas estable tras la
redistribucion (20,9%/20,8%/19,2% en 2010/2013/2016, frente a
17,4%/27,8%/25,3% con los indicadores problematicos todavia incluidos) --
consistente con que esos indicadores estaban metiendo ruido/sesgo de
cobertura, no señal real.

**Implicacion para uso futuro**: si se usa 2010 como ola base para
definir "hogares no pobres en IPM al inicio" (target de transicion,
espejo de `build_benchmark_train_test.py` para pobreza monetaria), la
clasificacion de 2010 ya no arrastra el sesgo de los 4 indicadores
excluidos -- pero sigue dependiendo de las aproximaciones documentadas en
el docstring del script (años de educacion via tabla de conversion propia,
no oficial DANE; barreras de acceso a salud via proxy mas agresivo en
ola 1 que en ola 2/3; norma de rezago escolar aproximada linealmente).

**Intento de segunda ronda de exclusion, REVERTIDO (mismo dia)**: se
evaluo excluir tambien `priv_rezago_escolar` (que si tiene cobertura
imperfecta en ola 1 -- 26,6% vs. ~92% en ola 2/3, aunque bastante menos
grave que los 4 ya excluidos). Resultado: con solo 9 indicadores activos
en el score, cada uno pesa proporcionalmente mas, y el sesgo YA CONOCIDO
de `priv_barreras_acceso_salud` (proxy mas agresivo en ola 1 sin arreglo
disponible, ver docstring punto 3) paso a DOMINAR el agregado -- la tasa
de pobreza IPM saltaba a 49,6% en ola 1 vs. 36,1%/34,9% en ola 2/3, una
inestabilidad peor que el problema que se intentaba resolver. Hallazgo
metodologico general, no especifico de este indicador: excluir
indicadores para "limpiar" un sesgo de cobertura tiene un limite --
mas alla de cierto punto, concentra peso en los sesgos que quedan sin
poder excluirse (porque son la unica pregunta disponible, no un
duplicado de otra) en vez de diluirlos. Se revirtio la exclusion de
`priv_rezago_escolar`, quedando la version de 10 indicadores/25% por
dimension como version final (confirmado con el usuario).

Output: `src/04_features/build_ipm_multidimensional.py` (nuevo),
`data/processed/ipm_multidimensional_elca_longitudinal.parquet` (nuevo).
lineas_pobreza_2012_2018_enph/lineas_20122018.csv`.
