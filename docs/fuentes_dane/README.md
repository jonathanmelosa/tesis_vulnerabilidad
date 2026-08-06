# Fuentes oficiales DANE — pobreza monetaria e IPC

Documentos oficiales descargados para construir las líneas de pobreza (LP), líneas de
pobreza extrema (LI) y los deflactores de precios usados en
`src/04_features/build_deflactor_ipc.py` y `src/04_features/build_pobreza_monetaria.py`.
Todos los valores de pesos usados en el código están tomados literalmente de las tablas
citadas abajo — no se estimó ni interpoló ningún valor de línea de pobreza.

## 1. Líneas de pobreza (LP) y pobreza extrema (LI)

| Año | Documento local | Fuente oficial | Tabla usada |
|---|---|---|---|
| 2010 | `Pobreza_nuevametodologia.pdf` | [MESEP — Misión para el Empalme de las Series de Empleo, Pobreza y Desigualdad](https://www.dane.gov.co/files/noticias/Pobreza_nuevametodologia.pdf) | Anexo J, Cuadro J1 (p. 92): "Valor mensual promedio por persona de la línea de pobreza extrema y pobreza para Colombia 2002-2010", dominios Nacional/Urbano/Rural |
| 2013 | `bol_pobreza_13.pdf` | [DANE — Boletín técnico, Pobreza Monetaria y Multidimensional en Colombia 2013](https://www.dane.gov.co/files/investigaciones/condiciones_vida/pobreza/bol_pobreza_13.pdf) | Tabla 3 y Tabla 4 (p. 6): "Comportamiento de la Línea de Pobreza / Pobreza Extrema 2012-2013", dominios Nacional/Cabecera/Resto |
| 2016 | `bol_pobreza_16.pdf` | [DANE — Boletín técnico, Pobreza Monetaria y Multidimensional en Colombia 2016](https://www.dane.gov.co/files/investigaciones/condiciones_vida/pobreza/bol_pobreza_16.pdf) | Tabla 1 y Tabla 2 (p. 3-4): "Comportamiento de la Línea de Pobreza / Pobreza Extrema 2015-2016", dominios Total Nacional/Cabeceras/Centros poblados y rural disperso |

Las tres tablas citan la misma metodología en su pie de página: **"Fuente: DANE, línea
base ENIG 2006-2007, actualizadas por IPC total de ingresos bajos"** (LP) y **"... por IPC
de alimentos por ingresos bajos"** (LI). Es decir, los tres años están en la misma serie
metodológica continua (pre actualización ENPH 2016-2017), lo que las hace comparables
entre sí — condición necesaria para clasificar pobreza en un panel 2010-2013-2016.

`cp_pobreza_2011.pdf` (comunicado de prensa, cifras 2010-2011) se conservó como respaldo:
confirma el quiebre metodológico "anterior/nueva metodología" y los valores de incidencia
de pobreza 2010 usados como chequeo cruzado, aunque no se usó para tomar ningún valor de
línea de pobreza.

### Dominios: correspondencia con la variable `zona` de la ELCA

La ELCA reporta `zona` como `"Urbano"` / `"Rural"` en las tres olas (ver
`data/processed/hogar_elca_longitudinal_clean.parquet`). Esto coincide exactamente con el
dominio "Urbano/Rural" del DANE en 2010, y se mapea de forma directa a "Cabecera/Resto"
(2013) y "Cabeceras/Centros poblados y rural disperso" (2016), que son la misma partición
conceptual con nombres distintos según el año de publicación.

## 2. Índice de Precios al Consumidor (IPC)

| Documento local | Fuente oficial | Contenido usado |
|---|---|---|
| `IPC_Variacion.xls` | [DANE — IPC histórico, "Variaciones porcentuales (IPC) 2003-2018 (diciembre)"](https://www.dane.gov.co/files/investigaciones/ipc/dic18/IPC_Variacion.xls) | Hoja `VariaNal`: variación mensual (%) del IPC total nacional, base metodológica IPC-08 (vigente desde enero 2009, canasta actualizada con la Encuesta de Ingresos y Gastos 2006-2007) |

El archivo trae variaciones porcentuales mes a mes, no el índice en niveles. El índice se
reconstruye encadenando esas variaciones desde una base arbitraria (100 en diciembre de
2008, mes de referencia de la metodología IPC-08 vigente en todo el período 2010-2016) —
ver `build_deflactor_ipc.py`. El promedio anual de ese índice encadenado para 2010, 2013 y
2016 es el deflactor **"IPC total nacional"**.

No se encontró un archivo histórico único y descargable con el IPC específico para el
grupo de "ingresos bajos" (el que el DANE usa para actualizar la LP/LI). Como ese vínculo
está documentado explícitamente en los boletines de pobreza ("actualizadas por IPC total
de ingresos bajos"), el deflactor **"IPC ingresos bajos"** se deriva directamente de la
razón entre los valores oficiales de LP nacional de cada año (ver tabla de la sección 1),
que por construcción DANE ya incorporan ese índice. Esto se documenta explícitamente en
`build_deflactor_ipc.py` para que quede claro que es un valor derivado, no una serie de
IPC descargada de forma independiente.

## 3. Notas de alcance

- No se usaron las líneas de pobreza recalculadas con la actualización metodológica ENPH
  2016-2017 (que solo cubre 2012-2018 y no es comparable con 2010).
- Todos los valores están en pesos corrientes mensuales por persona; la comparación de
  pobreza (pobre/no pobre) se hace en términos **nominales** (ingreso nominal del hogar del
  año X vs. LP nominal de ese mismo año y zona), tal como lo hace el DANE oficialmente. Los
  deflactores de esta carpeta solo se usan para construir series de ingreso/gasto en
  términos reales para análisis descriptivo y de tendencia, no para la clasificación de
  pobreza.
- Fecha de descarga de estos documentos: 2026-08-06.
