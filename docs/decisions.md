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

### 2026-03-18
- Initialized reproducible project structure
- Implemented API-based download of ELCA data
- Adopted pipeline-based workflow

### 2026-08-06 — Ingreso, gasto y pobreza monetaria (ELCA 2010, 2013, 2016)

Pipeline de scripts separados en `src/04_features/`, cada uno con un
artefacto propio (mismo patron que `build_ingreso_hogar.py`, ya existente):

  `config_dane.py`            -> parametros oficiales DANE (LP, LI, insumos IPC)
  `build_deflactor_ipc.py`    -> deflactores IPC total / IPC ingresos bajos, base 2010
  `build_ingreso_hogar.py`    -> ingreso nominal + series reales (ya existia, extendido)
  `build_gasto_hogar.py`      -> gasto mensual del hogar, normalizado por periodicidad
  `build_pobreza_monetaria.py`-> clasificacion pobre/no pobre (ingreso oficial + gasto robustez)

Decisiones metodologicas (consultadas con el usuario, no asumidas):

1. **Medida oficial de pobreza = ingreso**, siguiendo la metodologia DANE:
   ingreso per capita nominal del hogar vs. LP/LI nominal de su ano y zona
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
4. **Gasto del hogar**: se recalculo mensualizando cada item de gasto segun
   su periodicidad declarada (`per_{articulo}`), en vez de usar el
   `total_gasto` ya presente en `gastos_elca_longitudinal.parquet` (que suma
   valores de periodicidades distintas sin normalizar). Ver docstring de
   `build_gasto_hogar.py` para el supuesto de recall anual en los 35
   articulos sin dato de periodicidad.

### 2026-08-06 (cont.) — deflactor por zona y arriendo imputado

- **Deflactor por zona**: `deflactor_ipc_ing_bajos` (el derivado de la LP)
  ahora se calcula por separado para Urbano y Rural, no solo a nivel
  nacional, porque la LP urbana y la LP rural no crecen exactamente igual
  ano a ano. `deflactor_ipc_total` sigue siendo nacional unico: no existe
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

A peticion del usuario, se audito exhaustivamente `build_ingreso_hogar.py`
contra las 69 columnas `ing_/act_/gan_/gto_` de
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

A peticion del usuario, se aplico a `build_gasto_hogar.py` el mismo tipo de
auditoria hecha para el ingreso: (1) que ningun articulo de gasto se
excluyera por un cambio de nombre entre olas, (2) que la periodicidad
estuviera bien estandarizada para todos los articulos, y (3) que decision
tomar con los articulos sin columna per_. Metodo: comparacion programatica
de cobertura por ola/zona para los 88 articulos, cruce contra el manual
metodologico original (`data/interim/raw/diccionarios_elca/
ELCA-Manual-Hogar-Urbano2010.pdf`) y el `cod_articulo` de los `.tab`
crudos de 2010 urbano.

**Armonizacion de nombres**: no se encontraron articulos perdidos por
cambio de nombre entre olas -- `02_consolidacion_bases_gastos.py` ya
resuelve esos casos explicitamente via `ARMONIZACION_ARTICULOS`. Lo que si
se encontro es una diferencia real de GRANULARIDAD dentro de la ola 1: la
zona urbana de 2010 usa 35 categorias agregadas (ej. "alimentos
procesados" junta pastas, sal, azucar, panela, cafe, etc. en un solo
articulo), mientras que la zona rural de 2010 y ambas zonas de 2013/2016
usan 72 categorias detalladas para los mismos conceptos (cada uno por
separado). Confirmado con `cod_articulo` de `UGastos-csv.tab` 2010: los
articulos 1-35 mapean exactamente a la lista agregada urbana. Esto NO
afecta el total del hogar (cada hogar reporta en un solo sistema, sin
doble conteo), pero si reduce la comparabilidad a nivel de articulo
individual dentro de la ola 1.

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

### 2026-08-06 (cont.) — Desagregaciones de pobreza (build_pobreza_desagregaciones.py)

Nuevo script que parte de `pobreza_monetaria_elca_longitudinal.parquet` y
agrega, a peticion del usuario:

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
   pobreza). Emparejamiento de hogares via `consecutivo`. Decision
   confirmada con el usuario: los hogares que se DIVIDEN entre olas (511
   casos 2010->2013, 1.190 casos 2013->2016 contando ambos lados del
   emparejamiento) se EXCLUYEN de la matriz -- solo se usan matches 1 a 1 --
   y el conteo de excluidos se reporta junto a cada matriz.

Todas las tablas se guardan como CSV en `outputs/tables/pobreza/` (son
resultados descriptivos para el documento, no insumo de otro script).

Resultado de las transiciones (ingreso, medida oficial):
  2010->2013 (n=8.218): 43.9% siempre pobre, 28.8% nunca pobre, 18.6% sale
  de la pobreza, 8.8% entra en pobreza.
  2013->2016 (n=6.911): 37.0% nunca pobre, 33.3% siempre pobre, 20.6% sale
  de la pobreza, 9.2% entra en pobreza.
