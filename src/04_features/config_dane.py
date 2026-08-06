"""
Parametros oficiales del DANE para pobreza monetaria (ELCA 2010, 2013, 2016).

Unica fuente de verdad para las lineas de pobreza (LP), lineas de pobreza
extrema (LI) y los insumos del deflactor de precios usados en todo el
pipeline de 04_features. Ningun valor de este archivo fue estimado,
interpolado ni tomado de memoria: todos vienen copiados literalmente de los
documentos oficiales guardados en docs/fuentes_dane/ (ver README.md de esa
carpeta para el enlace exacto y el numero de tabla de cada cifra).

Cobertura metodologica
-----------------------
Los tres anos usan la MISMA serie continua del DANE: linea base ENIG
2006-2007, actualizada anualmente por IPC (total de ingresos bajos para LP;
alimentos de ingresos bajos para LI). Esto es lo que hace comparables los
pesos de 2010, 2013 y 2016 entre si. Deliberadamente NO se usan las lineas
recalculadas con la actualizacion metodologica ENPH 2016-2017 (esa serie solo
cubre 2012-2018 y no tiene un valor comparable para 2010).

Dominio geografico
-------------------
La ELCA reporta la columna `zona` como "Urbano" / "Rural" en las tres olas.
Esto coincide exactamente con el dominio "Urbano/Rural" que el DANE reporta
para 2010, y es la misma particion conceptual que el DANE llama
"Cabecera/Resto" en 2013 y "Cabeceras/Centros poblados y rural disperso" en
2016 (cambia el nombre de la categoria, no la definicion). Por eso todas las
filas de LINEAS_POBREZA usan las etiquetas "Urbano"/"Rural" tal como vienen
en la ELCA, para que el merge con `zona` sea directo.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUENTES_DANE_DIR = PROJECT_ROOT / "docs" / "fuentes_dane"

# ─── Lineas de pobreza (LP) y pobreza extrema (LI) ────────────────────────
# Pesos corrientes mensuales por persona. Fuente y tabla exacta por ano:
#
#   2010 -> docs/fuentes_dane/Pobreza_nuevametodologia.pdf (MESEP)
#           Anexo J, Cuadro J1 (p.92), dominios Nacional/Urbano/Rural
#   2013 -> docs/fuentes_dane/bol_pobreza_13.pdf
#           Tabla 3 y Tabla 4 (p.6), dominios Nacional/Cabecera/Resto
#   2016 -> docs/fuentes_dane/bol_pobreza_16.pdf
#           Tabla 1 y Tabla 2 (p.3-4), dominios Total Nacional/Cabeceras/
#           Centros poblados y rural disperso
#
# La columna `zona` usa las etiquetas de la ELCA ("Urbano"/"Rural"); ver
# docstring del modulo para la correspondencia de dominios.

LINEAS_POBREZA = pd.DataFrame(
    [
        {"ola": 1, "ano": 2010, "zona": "Urbano", "lp": 207_005, "li": 87_401},
        {"ola": 1, "ano": 2010, "zona": "Rural",  "lp": 123_502, "li": 71_392},
        {"ola": 2, "ano": 2013, "zona": "Urbano", "lp": 227_367, "li": 95_884},
        {"ola": 2, "ano": 2013, "zona": "Rural",  "lp": 136_192, "li": 77_947},
        {"ola": 3, "ano": 2016, "zona": "Urbano", "lp": 266_043, "li": 119_685},
        {"ola": 3, "ano": 2016, "zona": "Rural",  "lp": 159_543, "li": 97_867},
    ]
)

# Linea de pobreza y de pobreza extrema a nivel nacional (para chequeos de
# consistencia y para derivar el deflactor "IPC ingresos bajos", ver
# build_deflactor_ipc.py). Mismas fuentes que LINEAS_POBREZA.
LP_NACIONAL = {2010: 187_079, 2013: 206_091, 2016: 241_673}
LI_NACIONAL = {2010: 83_581, 2013: 91_698, 2016: 114_692}

OLA_A_ANO = {1: 2010, 2: 2013, 3: 2016}

# ─── Insumo del deflactor IPC total nacional ──────────────────────────────
# Variaciones porcentuales mensuales del IPC total nacional, metodologia
# IPC-08 (vigente desde enero de 2009, sin quiebre entre 2010 y 2016).
# Fuente: docs/fuentes_dane/IPC_Variacion.xls, hoja "VariaNal"
# (https://www.dane.gov.co/files/investigaciones/ipc/dic18/IPC_Variacion.xls).
IPC_VARIACION_XLS = FUENTES_DANE_DIR / "IPC_Variacion.xls"
