"""
Encuesta Longitudinal Colombiana (ELCA) – Zona Rural
Consolidación de las bases de datos de activos productivos – olas 2010, 2013 y 2016.

Autor: Jonathan Melo Sarta
Fecha: Junio 2026

Contexto
--------
El módulo RActivos_hogar recoge información sobre activos productivos de los
hogares rurales encuestados: maquinaria agrícola, semovientes (ganado, aves,
peces, colmenas, etc.) y otros activos del predio. Esta base NO existe para
la zona urbana — es exclusiva de la muestra rural de la ELCA.

Lógica general
--------------
Las tres olas entran ya en formato ancho (una fila por hogar) por lo que no
se requiere pivoteo. El script se limita a:
  1. Leer cada archivo fuente.
  2. Aplicar correcciones de codificación donde corresponde.
  3. Concatenar las tres olas en un único dataset longitudinal.

Unidad de análisis
------------------
Sub-hogar × ola. Los hogares que se dividen entre olas quedan como filas
separadas gracias a los identificadores llave (2013) y llave_n16 (2016).
En 2010 no hay información de divisiones; la unidad es simplemente hogar.

Identificadores incluidos
-------------------------
- 2010 : ola, consecutivo
- 2013 : ola, consecutivo, hogar, llave, zona, region
- 2016 : ola, consecutivo, hogar, llave, hogar_n16, llave_n16

Diferencias de columnas entre olas
------------------------------------
2010 tiene 97 columnas, 2013 y 2016 tienen 79 cada una. Las diferencias son:

  Solo en 2010 (ausentes en 2013/2016):
    - vr_alquiler_* (30 columnas): valor de alquiler de cada activo. Esta
      información dejó de recogerse en olas posteriores.
    - abejas / n_abejas  : colmenas de abejas (renombradas en 2013 como
      colmenas / n_colmenas).
    - peces / n_peces    : estanques de peces (renombrados en 2013 como
      estanque / n_estanque).
    - n_oanim            : número de otros animales (renombrado en 2013 como
      n_otros_anim / n_otro_cual).

  Solo en 2013/2016 (ausentes en 2010):
    - colmenas / n_colmenas   : equivalen a abejas / n_abejas de 2010.
    - estanque / n_estanque   : equivalen a peces / n_peces de 2010.
    - picapasto / n_picapasto : activo no relevado en 2010.
    - n_otro_cual             : descripción textual de otros animales.
    - zona, region            : zona (Rural) y región geográfica (solo 2013).

  Solo en 2016 (ausentes en 2010 y 2013):
    - hogar_n16, llave_n16  : identificadores de sub-hogar en ola 3.
    - n_invernadro           : invernadero (no relevado antes).

  Las columnas ausentes quedan como NaN al concatenar.

Codificación y correcciones
-----------------------------
- 2010: sin problemas de codificación.
- 2013: los campos binarios de semovientes (bueyes, vacas, cerdos, etc.)
  tienen 'S???' en lugar de 'Si'. La columna 'region' presenta 'Atl???ntica',
  'Pac???fica' y 'Atl???ntica Media'. El campo libre 'n_otro_cual' puede
  tener 'cr???a' en lugar de 'cría'.
- 2016: los mismos campos binarios de semovientes tienen 'S???' en lugar
  de 'Si' (sin afectación en region ya que la columna no existe en 2016).

Dimensiones del output
-----------------------
  2010 : 4 578 filas ×  97 columnas
  2013 : 4 339 filas ×  79 columnas
  2016 : 3 893 filas ×  79 columnas
  Total: 12 810 filas × 112 columnas  (unión de todas las columnas)
"""

import os

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

DATA_ROOT = "/Users/macbook/Documents/Documentos/tesis_vulnerabilidad/data/interim/raw"
OUTPUT_PATH = (
    "/Users/macbook/Documents/Documentos/tesis_vulnerabilidad"
    "/data/processed/RActivos_hogar_unido.parquet"
)

# ─── Identificadores por ola ─────────────────────────────────────────────────

IDS_POR_OLA = {
    2010: {"key": "consecutivo",  "links": []},
    2013: {"key": "llave",        "links": ["consecutivo", "hogar"]},
    2016: {"key": "llave_n16",    "links": ["consecutivo", "llave", "hogar_n16"]},
}

# ─── Correcciones de codificación ────────────────────────────────────────────
#
# En 2013 y 2016 los campos binarios de semovientes registran 'S???' en lugar
# de 'Si'. Adicionalmente 2013 tiene problemas en la columna 'region' y en el
# campo libre 'n_otro_cual'.
#
# La corrección se aplica a TODAS las columnas de texto del DataFrame porque:
#  - 'S???' solo aparece en campos que esperan 'Si'/'No', así que reemplazarlo
#    por 'Si' es siempre correcto.
#  - Las demás subcadenas corregidas ('Atl???ntica', 'cr???a', etc.) tampoco
#    son ambiguas: no pueden aparecer en otro contexto con significado distinto.

# Valores centinela en columnas numéricas de 2010 (cantidades y valores de alquiler).
# Se reemplazan por NaN para evitar conflictos de tipo al concatenar con 2013/2016.
# "No informa": el informante no supo el dato. "No sabe": idem.
# Afecta principalmente n_avescorral (columna que también existe en 2013/2016 como
# float64) y a varias columnas vr_alquiler_* (exclusivas de 2010).
CENTINELAS_NUMERICOS_2010 = {"No informa", "No sabe"}

CORRECCIONES_2013 = {
    "S???":            "Si",           # indicadores binarios de semovientes
    "Atl???ntica":     "Atlántica",    # región geográfica
    "Pac???fica":      "Pacífica",     # región geográfica
    "cr???a":          "cría",         # descripción libre n_otro_cual
}

CORRECCIONES_2016 = {
    "S???": "Si",                       # indicadores binarios de semovientes
}


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def corregir_columnas(df: pd.DataFrame, correcciones: dict) -> pd.DataFrame:
    """
    Aplica reemplazos exactos de subcadena en todas las columnas de texto del
    DataFrame. Devuelve una copia con las correcciones aplicadas.
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        for patron, correcto in correcciones.items():
            df[col] = df[col].str.replace(patron, correcto, regex=False)
    return df


# ─── Procesamiento por ola ────────────────────────────────────────────────────

def procesar_2010() -> pd.DataFrame:
    """
    Lee RActivos_hogar de 2010 (zona Rural).

    Solo tiene 'consecutivo' como identificador de hogar; no existen llave
    ni hogar_n16. Incluye 30 columnas vr_alquiler_* (valor de alquiler de
    cada activo) que no se recogieron en olas posteriores.

    Limpieza específica de 2010:
    - Las columnas de cantidad (n_*) y de valor de alquiler (vr_alquiler_*)
      contienen ocasionalmente los strings 'No informa' y 'No sabe' en lugar
      de un valor numérico. Estos se convierten a NaN para que la columna
      pueda ser tratada como float64, evitando conflictos de tipo al
      concatenar con 2013/2016 (donde las mismas columnas ya son float64).
    """
    path = os.path.join(DATA_ROOT, "elca_2010", "RActivos_hogar-csv.tab")
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Convertir centinelas a NaN en columnas que debieran ser numéricas
    for col in df.select_dtypes(include="object").columns:
        if df[col].isin(CENTINELAS_NUMERICOS_2010).any():
            df[col] = pd.to_numeric(
                df[col].replace({v: None for v in CENTINELAS_NUMERICOS_2010}),
                errors="coerce",
            )

    return df


def procesar_2013() -> pd.DataFrame:
    """
    Lee RActivos_hogar de 2013 (zona Rural).

    Corrige 'S???' → 'Si' en campos binarios de semovientes, errores de
    codificación en la columna 'region' y en el campo libre 'n_otro_cual'.
    Agrega zona, region, llave y hogar como identificadores adicionales.
    """
    path = os.path.join(DATA_ROOT, "elca_2013", "RActivos_hogar-csv.tab")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = corregir_columnas(df, CORRECCIONES_2013)
    return df


def procesar_2016() -> pd.DataFrame:
    """
    Lee RActivos_hogar de 2016 (zona Rural).

    Corrige 'S???' → 'Si' en campos binarios de semovientes.
    Agrega hogar_n16 y llave_n16 como identificadores de sub-hogar en ola 3.
    """
    path = os.path.join(DATA_ROOT, "elca_2016", "RActivos_hogar-csv.tab")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = corregir_columnas(df, CORRECCIONES_2016)
    return df


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Lee, corrige y concatena las tres olas de RActivos_hogar.

    La concatenación usa la unión de columnas (outer join implícito de
    pd.concat); las columnas ausentes en una ola quedan como NaN.
    """
    print("  Procesando 2010 RActivos_hogar...")
    df_2010 = procesar_2010()

    print("  Procesando 2013 RActivos_hogar...")
    df_2013 = procesar_2013()

    print("  Procesando 2016 RActivos_hogar...")
    df_2016 = procesar_2016()

    base_final = pd.concat(
        [df_2010, df_2013, df_2016],
        ignore_index=True,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    base_final.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    print(f"\nBase final guardada en: {OUTPUT_PATH}")
    print(f"Dimensiones           : {base_final.shape[0]} filas × {base_final.shape[1]} columnas")
    print(f"Olas presentes        : {sorted(base_final['ola'].unique())}")
    print(f"Hogares únicos        : {base_final['consecutivo'].nunique()}")
    print(f"  2010 : {len(df_2010):>5} filas × {df_2010.shape[1]:>3} columnas")
    print(f"  2013 : {len(df_2013):>5} filas × {df_2013.shape[1]:>3} columnas")
    print(f"  2016 : {len(df_2016):>5} filas × {df_2016.shape[1]:>3} columnas")

    return base_final


if __name__ == "__main__":
    base_final = main()
