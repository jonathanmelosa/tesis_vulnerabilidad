"""
Encuesta Longitudinal Colombiana de Origen y Destino (ELCO)
Consolidación de las bases de datos de PERSONAS – olas 2019 y 2022.

Autor: Jonathan Melo Sarta
Fecha: Julio 2026

Contexto
--------
ELCO tiene, a nivel de persona, diecinueve módulos temáticos (migración,
educación, salud, mercado laboral, pensiones, ahorro, capital social,
política, etc.). Este script une, para cada ola por separado, los módulos
de persona sobre la llave propia de esa ola, y luego apila (concatena) las
dos olas en un único dataset longitudinal con una fila por persona por ola.

Unidad de análisis
-------------------
Persona × ola.  Cada persona aporta exactamente una fila por ola en la que
participó (no se duplican filas por evento migratorio, ciclo educativo,
episodio de fecundidad, gasto personal ni historial de actividades; ver
"Módulos cíclicos" abajo).

Llave de unión intra-ola
--------------------------
Se usa DIRECTORIO + ORDEN (no CONSECUTIVO_DANE_ELCO_2019 + _PER_2019, ni sus
análogos de 2022) como llave de cruce entre los módulos de una misma ola.
Motivo (mismo hallazgo que en el script de hogares):

  - En 2019 los identificadores CONSECUTIVO_DANE_ELCO_2019 /
    CONSECUTIVO_DANE_ELCO_PER_2019 son limpios y sin duplicados.
  - En 2022, CONSECUTIVO_DANE_ELCO_2022 y CONSECUTIVO_DANE_ELCO_PER_2022
    llegan corrompidos EN EL ARCHIVO FUENTE: son identificadores numéricos
    de 14-16 dígitos exportados en notación científica con solo 6 cifras
    significativas (p.ej. "3,123E+16"), lo que provoca colisiones reales
    entre personas distintas (22 368 de 35 766 filas de A_ENLISTAMIENTO
    2022 quedan con CONSECUTIVO_DANE_ELCO_PER_2022 duplicado).
  - DIRECTORIO + ORDEN sí es único en todos los módulos de persona de
    ambas olas y está presente en todos ellos, por lo que se usa como
    llave de cruce.
  - Las columnas CONSECUTIVO_DANE_ELCO_2019/2022 y sus versiones _PER se
    conservan en el output (identificador nominal / enlace al panel), pero
    NO se usan para el merge.

Módulos incluidos y su archivo por ola
-----------------------------------------
    Módulo                             Archivo 2019                                          Archivo 2022
    A_ENLISTAMIENTO (base)             A_ENLISTAMIENTO DE PERSONAS.csv                        A_ENLISTAMIENTO DE PERSONAS.csv
    H_COM_Y_CARACT_HOGAR               H_COM_Y_CARACT_HOGAR.csv                               H_COMPOSICION Y CARACTERIZACIÓN DEL HOGAR.csv
    H_MIGRACION_CABECERA *             H_MIGRACION CABECERA.csv                               H_COMPOSICION Y CARACTERIZACIÓN DEL HOGAR (MIGRACION CABECERA).csv
    H_MIGRACION_VEREDA *               H_MIGRACION VEREDA.csv                                 H_COMPOSICION Y CARACTERIZACIÓN DEL HOGAR (MIGRACION VEREDA).csv
    H_MIGRACION *                      H_MIGRACION.csv                                        H_COMPOSICION Y CARACTERIZACIÓN DEL HOGAR (MIGRACION).csv
    I_CICLO_EDUCACION_FORM_TRABAJO *   I_CICLO_EDUCACION_FORM_TRABAJO.csv                     I_CICLO EDUCACION Y FORMACIÓN PARA EL TRABAJO.csv
    I_EDUC_FORMACION_PARA_EL_TRABAJO   I_EDUC_FORMACION_PARA_EL_TRABAJO.csv                   I_EDUCACION Y FORMACIÓN PARA EL TRABAJO.csv
    J_INFANCIA_Y_ADOLESCENCIA          J_ INFANCIA Y ADOLESCENCIA.csv                         J_INFANCIA Y ADOLESCENCIA.csv
    K_JOVENES                          K_JOVENES.csv                                          K_JOVENES.csv
    L_SALUD                            L_SALUD.csv                                            L_SALUD.csv
    L_CICLO_SALUD_FECUNDIDAD *         L_CICLO SALUD (FECUNDIDAD).csv                         L_CICLO SALUD (FECUNDIDAD).csv
    M_GASTOS_PERSONALES *              M_GASTOS PERSONALES.csv                                M_GASTOS PERSONALES.csv
    N_MERCADO_LABORAL                  N_MERCADO LABORAL.csv                                  N_MERCADO LABORAL.csv
    O_CICLO_HISTORIAL_DE_ACTIVIDADES * O_CICLO HISTORIAL DE ACTIVIDADES.csv                   -- no existe en 2022 --
    O_HISTORIAL_DE_ACTIVIDADES         O_HISTORIAL DE ACTIVIDADES.csv                         -- no existe en 2022 --
    P_PENSIONES                        P_PENSIONES.csv                                        O_PENSIONES.csv
    Q_AHORRO                           Q_AHORRO.csv                                           P_AHORRO.csv
    R_CAPITAL_SOCIAL                   R_CAPITAL SOCIAL.csv                                   Q_CAPITAL_SOCIAL.csv
    S_POLITICA                         S_POLITICA.csv                                         R_POLITICA.csv

  (*) Módulos cíclicos, ver abajo.

  Nota sobre módulos ausentes en 2022: O_CICLO_HISTORIAL_DE_ACTIVIDADES y
  O_HISTORIAL_DE_ACTIVIDADES (historial laboral detallado) no tienen archivo
  en la carpeta fuente de 2022 — el módulo no se repitió en esa ola. Sus
  columnas quedan como NaN para todas las filas de OLA == 2022.

  Nota sobre letras de capítulo: entre 2019 y 2022 varios módulos cambiaron
  de letra de capítulo sin cambiar de contenido (P_PENSIONES→O_PENSIONES,
  Q_AHORRO→P_AHORRO, R_CAPITAL_SOCIAL→Q_CAPITAL_SOCIAL, S_POLITICA→
  R_POLITICA). El mapeo de archivos de arriba es por CONTENIDO, no por letra.

Módulos cíclicos
-------------------
Los módulos marcados con (*) traen varias filas por persona (un episodio
migratorio, un ciclo educativo, un evento de fecundidad, un gasto personal
o un periodo de actividad por fila). Para preservar "una fila por persona"
se ENSANCHAN antes de unirlos a la base: se numeran las filas de cada
persona en el orden en que aparecen y se usa ese número como sufijo de
columna (p.ej. P15000_1, P15000_2, ... para el primer y segundo evento
migratorio). Ninguno de estos módulos trae una columna de categoría
explícita (a diferencia de F_CICLO_GASTOS_DEL_HOGAR en el script de
hogares), así que el sufijo siempre es el orden de aparición.

Separador y codificación de los archivos fuente
----------------------------------------------------
  - 2019: separador coma (","), UTF-8.
  - 2022: separador punto y coma (";"), UTF-8.
  Ambos se leen con dtype=str para no perder precisión en identificadores
  largos ni introducir NaN/float espurios en columnas categóricas.

Dimensiones esperadas
------------------------
  2019: 40 770 personas.
  2022: 35 766 personas.
  Total: 76 536 filas en la base unificada (una fila por persona por ola).
  (En la práctica quedan 76 535: ver "Filas duplicadas exactas" abajo — una
  fila de A_ENLISTAMIENTO 2019 es un duplicado exacto de otra y se elimina.)

Filas duplicadas exactas en el archivo fuente
-------------------------------------------------
A diferencia del script de hogares (08_consolidacion_bases_hogar_ELCO.py),
aquí SÍ fue necesario limpiar duplicados exactos antes de cada merge (ver
deduplicar_exactas()): varios módulos de persona traen filas 100%
idénticas (mismo DIRECTORIO, ORDEN y TODO el resto de columnas) para la
misma persona. Ejemplos encontrados al construir este script:
  - A_ENLISTAMIENTO 2019: 1 fila duplicada.
  - H_MIGRACION_CABECERA / H_MIGRACION_VEREDA / H_MIGRACION / 2022:
    13, 16 y 27 filas duplicadas respectivamente.
  - I_CICLO_EDUCACION_FORM_TRABAJO 2022: 61 filas duplicadas.
  - L_CICLO_SALUD_FECUNDIDAD 2022: 55 filas duplicadas.
  - M_GASTOS_PERSONALES 2022: 23 filas duplicadas.
Se asume que son errores de captura/exportación (doble digitación de la
misma fila) y no información distinta, por lo que se descartan con
DataFrame.drop_duplicates() ANTES de pivotear o unir. Si tras esa limpieza
persistiera una llave repetida con contenido distinto, el merge
'validate="one_to_one"' fallaría de forma explícita en vez de mezclar
datos incorrectamente (ver unir_modulos).

Estrategia de unión (how="left") y por qué
-----------------------------------------------
Cada módulo secundario se une a la base (A_ENLISTAMIENTO) con un LEFT
JOIN: la base define el universo de personas de la ola, y una persona que
no respondió un módulo específico (p.ej. no reportó pensión) simplemente
queda con NaN en esas columnas en vez de desaparecer de la base unificada.
'validate="one_to_one"' actúa como aserción de integridad en cada merge.
"""

from __future__ import annotations

import os

import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────────────────────

RAW_2019 = (
    "/Users/macbook/Documents/Documentos/tesis_vulnerabilidad"
    "/data/interim/raw/elca_2019/ELCO_2019/csv"
)
RAW_2022 = (
    "/Users/macbook/Documents/Documentos/tesis_vulnerabilidad"
    "/data/interim/raw/elca_2022/BDATOS-ELCO-2022/cvs"
)
OUTPUT_PATH = (
    "/Users/macbook/Documents/Documentos/tesis_vulnerabilidad"
    "/data/processed/personas_ELCO.parquet"
)

KEY = ["DIRECTORIO", "ORDEN"]

# ─── Módulos por ola ──────────────────────────────────────────────────────────
# (nombre_modulo, archivo_o_None_si_no_existe, es_ciclico)

MODULOS_2019 = [
    ("A_ENLISTAMIENTO",                    "A_ENLISTAMIENTO DE PERSONAS.csv",       False),
    ("H_COM_Y_CARACT_HOGAR",                "H_COM_Y_CARACT_HOGAR.csv",              False),
    ("H_MIGRACION_CABECERA",                "H_MIGRACION CABECERA.csv",              True),
    ("H_MIGRACION_VEREDA",                  "H_MIGRACION VEREDA.csv",                True),
    ("H_MIGRACION",                         "H_MIGRACION.csv",                       True),
    ("I_CICLO_EDUCACION_FORM_TRABAJO",      "I_CICLO_EDUCACION_FORM_TRABAJO.csv",    True),
    ("I_EDUC_FORMACION_PARA_EL_TRABAJO",    "I_EDUC_FORMACION_PARA_EL_TRABAJO.csv",  False),
    ("J_INFANCIA_Y_ADOLESCENCIA",           "J_ INFANCIA Y ADOLESCENCIA.csv",        False),
    ("K_JOVENES",                           "K_JOVENES.csv",                         False),
    ("L_SALUD",                             "L_SALUD.csv",                           False),
    ("L_CICLO_SALUD_FECUNDIDAD",            "L_CICLO SALUD (FECUNDIDAD).csv",        True),
    ("M_GASTOS_PERSONALES",                 "M_GASTOS PERSONALES.csv",               True),
    ("N_MERCADO_LABORAL",                   "N_MERCADO LABORAL.csv",                 False),
    ("O_CICLO_HISTORIAL_DE_ACTIVIDADES",    "O_CICLO HISTORIAL DE ACTIVIDADES.csv",  True),
    ("O_HISTORIAL_DE_ACTIVIDADES",          "O_HISTORIAL DE ACTIVIDADES.csv",        False),
    ("P_PENSIONES",                         "P_PENSIONES.csv",                       False),
    ("Q_AHORRO",                            "Q_AHORRO.csv",                          False),
    ("R_CAPITAL_SOCIAL",                    "R_CAPITAL SOCIAL.csv",                  False),
    ("S_POLITICA",                          "S_POLITICA.csv",                        False),
]

MODULOS_2022 = [
    ("A_ENLISTAMIENTO",                 "A_ENLISTAMIENTO DE PERSONAS.csv",                                       False),
    ("H_COM_Y_CARACT_HOGAR",             "H_COMPOSICION Y CARACTERIZACIÓN DEL HOGAR.csv",                        False),
    ("H_MIGRACION_CABECERA",             "H_COMPOSICION Y CARACTERIZACIÓN DEL HOGAR (MIGRACION CABECERA).csv",   True),
    ("H_MIGRACION_VEREDA",               "H_COMPOSICION Y CARACTERIZACIÓN DEL HOGAR (MIGRACION VEREDA).csv",     True),
    ("H_MIGRACION",                      "H_COMPOSICION Y CARACTERIZACIÓN DEL HOGAR (MIGRACION).csv",            True),
    ("I_CICLO_EDUCACION_FORM_TRABAJO",   "I_CICLO EDUCACION Y FORMACIÓN PARA EL TRABAJO.csv",                    True),
    ("I_EDUC_FORMACION_PARA_EL_TRABAJO", "I_EDUCACION Y FORMACIÓN PARA EL TRABAJO.csv",                          False),
    ("J_INFANCIA_Y_ADOLESCENCIA",        "J_INFANCIA Y ADOLESCENCIA.csv",                                        False),
    ("K_JOVENES",                        "K_JOVENES.csv",                                                        False),
    ("L_SALUD",                          "L_SALUD.csv",                                                          False),
    ("L_CICLO_SALUD_FECUNDIDAD",         "L_CICLO SALUD (FECUNDIDAD).csv",                                       True),
    ("M_GASTOS_PERSONALES",              "M_GASTOS PERSONALES.csv",                                              True),
    ("N_MERCADO_LABORAL",                "N_MERCADO LABORAL.csv",                                                False),
    ("O_CICLO_HISTORIAL_DE_ACTIVIDADES", None,                                                                   True),
    ("O_HISTORIAL_DE_ACTIVIDADES",       None,                                                                   False),
    ("P_PENSIONES",                      "O_PENSIONES.csv",                                                      False),
    ("Q_AHORRO",                         "P_AHORRO.csv",                                                         False),
    ("R_CAPITAL_SOCIAL",                 "Q_CAPITAL_SOCIAL.csv",                                                 False),
    ("S_POLITICA",                       "R_POLITICA.csv",                                                       False),
]

# Columnas identificadoras que se repiten en todos los módulos y no aportan
# información nueva más allá del módulo base.
COLS_IDENTIFICADORAS = [
    "CONSECUTIVO_DANE_2010", "CONSECUTIVO_DANE_2010_PER",
    "CONSECUTIVO_DANE_2013", "CONSECUTIVO_DANE_2013_PER",
    "CONSECUTIVO_DANE_2016", "CONSECUTIVO_DANE_2016_PER",
    "CONSECUTIVO_DANE_ELPS_2012", "CONSECUTIVO_DANE_ELPS_PER_2012",
    "CONSECUTIVO_DANE_ELPS_2015", "CONSECUTIVO_DANE_ELPS_PER_2015",
    "CONSECUTIVO_DANE_ELCO_2019", "CONSECUTIVO_DANE_ELCO_PER_2019",
    "CONSECUTIVO_DANE_ELCO_PER_19",
    "CONSECUTIVO_DANE_ELCO_2022", "CONSECUTIVO_DANE_ELCO_PER_2022",
    "DIRECTORIO", "SECUENCIA_ENCUESTA", "SECUENCIA_P", "ORDEN", "ORDEN_1",
]


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def leer_modulo(carpeta: str, archivo: str, sep: str) -> pd.DataFrame:
    """
    Lee un módulo de persona desde CSV.

    Decisiones:
    - dtype=str en TODAS las columnas: evita que pandas infiera tipos
      numéricos y trunque/redondee identificadores largos (el problema que
      ya afecta a CONSECUTIVO_DANE_ELCO_PER_2022 en el archivo fuente, ver
      docstring del módulo) o convierta columnas categóricas (códigos como
      "01", "02") en enteros, perdiendo los ceros a la izquierda.
    - Se intenta UTF-8 primero porque es la codificación de la mayoría de
      los archivos. L_SALUD.csv de 2022 es el ÚNICO archivo de todo el
      conjunto (los 19 módulos x 2 olas) que no está en UTF-8: viene en
      Latin-1, así que el except captura específicamente ese caso y
      reintenta con esa codificación. El resto de módulos nunca cae en la
      rama except, pero se deja el mecanismo genérico por robustez ante
      nuevas descargas del mismo instrumento.
    """
    path = os.path.join(carpeta, archivo)
    try:
        return pd.read_csv(path, sep=sep, dtype=str, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, dtype=str, low_memory=False, encoding="latin-1")


def deduplicar_exactas(df: pd.DataFrame, key: list, nombre: str) -> pd.DataFrame:
    """
    Elimina filas 100% idénticas (todas las columnas iguales, no solo la
    llave) que comparten la llave de persona DIRECTORIO+ORDEN.

    Por qué es necesario en este script y no en el de hogares
    -----------------------------------------------------------
    Varios módulos de persona (ver "Filas duplicadas exactas" en el
    docstring del módulo) traen filas repetidas al 100% para la misma
    persona — errores de digitación/exportación del archivo fuente, no
    información distinta. Sin esta limpieza, el merge 'validate=
    "one_to_one"' de unir_modulos() fallaría con MergeError apenas
    encontrara la primera llave repetida, deteniendo todo el script.

    Comportamiento ante duplicados NO exactos
    --------------------------------------------
    drop_duplicates() solo colapsa filas cuyo contenido es idéntico en
    TODAS las columnas. Si dos filas comparten DIRECTORIO+ORDEN pero
    difieren en alguna otra columna (un duplicado "real" con datos
    distintos, que indicaría un problema más serio en la fuente), esta
    función NO las toca — persisten como duplicado de llave y el merge
    posterior fallará de forma explícita en unir_modulos(), dejando el
    problema visible en vez de elegir una fila arbitrariamente.
    """
    antes = len(df)
    df = df.drop_duplicates()
    if len(df) < antes:
        print(f"      [{nombre}] {antes - len(df)} fila(s) duplicada(s) exacta(s) eliminada(s)")
    return df


def pivotear_ciclico(df: pd.DataFrame, key: list) -> pd.DataFrame:
    """
    Ensancha (long → wide) un módulo con varias filas por persona a
    exactamente una fila por persona, agregando un sufijo de orden a cada
    columna de contenido. Es el equivalente, para personas, de
    pivotear_ciclico() en el script de hogares — ver ese docstring para el
    razonamiento completo del algoritmo (agrupar por sufijo, indexar por
    llave, renombrar columnas, concatenar por columnas).

    Diferencia clave frente al script de hogares
    ------------------------------------------------
    Ningún módulo cíclico de persona (H_MIGRACION*, I_CICLO_EDUCACION_
    FORM_TRABAJO, L_CICLO_SALUD_FECUNDIDAD, M_GASTOS_PERSONALES,
    O_CICLO_HISTORIAL_DE_ACTIVIDADES) trae una columna de categoría con
    significado propio (como "CAP" en F_CICLO_GASTOS_DEL_HOGAR). Por eso
    esta función no recibe un parámetro cat_col: el sufijo siempre es el
    número de orden de aparición de la fila dentro de la persona
    (1, 2, 3, ...), p.ej. P15000_1 = datos del primer evento migratorio
    reportado, P15000_2 = datos del segundo, etc. El orden de captura en
    la encuesta se preserva pero no tiene un significado temático
    adicional más allá de "primer episodio", "segundo episodio", etc.

    Se descartan las columnas de COLS_IDENTIFICADORAS (excepto la llave)
    antes de pivotear, por ser redundantes con el módulo base.
    """
    cols_id_sin_key = [c for c in COLS_IDENTIFICADORAS if c in df.columns and c not in key]
    df = df.drop(columns=cols_id_sin_key)
    value_cols = [c for c in df.columns if c not in key]

    sufijo = (df.groupby(key).cumcount() + 1).astype(str)
    df = df[key + value_cols].copy()
    df["_sufijo"] = sufijo.values

    anchas = []
    for suf, sub in df.groupby("_sufijo"):
        sub = sub.drop(columns="_sufijo").set_index(key)
        sub.columns = [f"{c}_{suf}" for c in sub.columns]
        anchas.append(sub)

    ancha = pd.concat(anchas, axis=1).reset_index()
    return ancha


def unir_modulos(carpeta: str, sep: str, modulos: list, ola: int) -> pd.DataFrame:
    """
    Une, para UNA ola, todos sus módulos de persona en una sola tabla ancha
    (una fila por persona).

    Pasos
    -----
    1. El primer elemento de `modulos` es siempre el módulo base
       (A_ENLISTAMIENTO). Se lee y se deduplica (deduplicar_exactas), pero
       NO se le quitan columnas: es la única tabla que conserva sus
       columnas identificadoras completas (CONSECUTIVO_DANE_*, ORDEN_1,
       SECUENCIA_*), que sirven como referencia/documentación de la
       persona en el output final.
    2. Cada módulo siguiente:
       - si su archivo es None (módulo que no existe en esta ola, p.ej.
         O_HISTORIAL_DE_ACTIVIDADES en 2022), se omite explícitamente con
         un mensaje — no se intenta leer un archivo inexistente.
       - si no, se lee, se deduplica (deduplicar_exactas) y:
           - si es cíclico, se ensancha con pivotear_ciclico() para que
             quede una fila por persona;
           - si no, se le quitan las columnas identificadoras redundantes
             (COLS_IDENTIFICADORAS) porque ya están en la base.
    3. El módulo resultante se une a la base acumulada con LEFT JOIN sobre
       DIRECTORIO+ORDEN y `validate="one_to_one"` (ver docstring del
       módulo, sección "Estrategia de unión").
    4. Al final se agrega la columna OLA (2019 o 2022) para poder
       distinguir e identificar cada fila tras apilar ambas olas en main().

    El progreso (nombre de módulo, filas, columnas, duplicados quitados) se
    imprime en cada paso para poder auditar visualmente que ningún módulo
    se leyó vacío o con una cantidad de columnas inesperada.
    """
    nombre_base, archivo_base, _ = modulos[0]
    base = leer_modulo(carpeta, archivo_base, sep)
    base = deduplicar_exactas(base, KEY, nombre_base)
    print(f"    {nombre_base:38s}: {base.shape[0]:>6} filas x {base.shape[1]:>3} cols")

    for nombre, archivo, es_ciclico in modulos[1:]:
        if archivo is None:
            print(f"    {nombre:38s}: no existe en esta ola -> se omite (columnas quedan NaN)")
            continue

        mod = leer_modulo(carpeta, archivo, sep)

        if es_ciclico:
            mod = deduplicar_exactas(mod, KEY, nombre)
            mod = pivotear_ciclico(mod, KEY)
            print(f"    {nombre:38s}: {mod.shape[0]:>6} personas x {mod.shape[1]:>3} cols (pivoteado)")
        else:
            mod = deduplicar_exactas(mod, KEY, nombre)
            cols_id_sin_key = [c for c in COLS_IDENTIFICADORAS if c in mod.columns and c not in KEY]
            mod = mod.drop(columns=cols_id_sin_key)
            print(f"    {nombre:38s}: {mod.shape[0]:>6} filas x {mod.shape[1]:>3} cols")

        # how="left": la base define el universo de personas de la ola; una
        # persona sin registro en este módulo queda con NaN, no desaparece.
        # validate="one_to_one": aserción de integridad — si `mod` tuviera
        # más de una fila por DIRECTORIO+ORDEN tras deduplicar_exactas() y
        # pivotear_ciclico(), el script debe fallar aquí en vez de
        # multiplicar filas de `base` silenciosamente.
        base = base.merge(mod, on=KEY, how="left", validate="one_to_one")

    base["OLA"] = ola
    return base


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """
    Pipeline completo: construye la tabla ancha de cada ola por separado
    (unir_modulos) y las apila en un único dataset longitudinal.

    Por qué concatenar (apilar) y no cruzar (wide panel)
    ----------------------------------------------------
    Igual que en el script de hogares: se procesa cada ola por separado y
    se concatena con pd.concat(axis=0), produciendo una fila por persona
    POR OLA (2019 y 2022 son filas distintas). Es la opción confirmada con
    el usuario, coherente con que la unidad de análisis es la persona y
    con que ELCO 2022 no es un simple re-cruce 1 a 1 de las personas de
    2019 (hay nacimientos, muertes, entradas y salidas del panel). La
    UNIÓN de columnas de pd.concat implica que las columnas exclusivas de
    una ola (p.ej. las de O_HISTORIAL_DE_ACTIVIDADES, que no existe en
    2022) quedan en NaN para las filas de la otra ola.

    Orden de columnas del output
    -------------------------------
    Primero los identificadores clave (OLA, DIRECTORIO, ORDEN y los
    CONSECUTIVO_DANE_ELCO_*/_PER_* nominales), y después el resto de
    columnas en el orden en que fueron apareciendo al unir los módulos.
    """
    print("Procesando ola 2019...")
    personas_2019 = unir_modulos(RAW_2019, ",", MODULOS_2019, 2019)

    print("\nProcesando ola 2022...")
    personas_2022 = unir_modulos(RAW_2022, ";", MODULOS_2022, 2022)

    base_final = pd.concat([personas_2019, personas_2022], axis=0, ignore_index=True)

    # Identificadores primero, luego el resto en orden de aparición.
    id_cols = [c for c in ["OLA", "DIRECTORIO", "ORDEN",
                            "CONSECUTIVO_DANE_ELCO_2019", "CONSECUTIVO_DANE_ELCO_PER_2019",
                            "CONSECUTIVO_DANE_ELCO_2022", "CONSECUTIVO_DANE_ELCO_PER_2022"]
               if c in base_final.columns]
    otras_cols = [c for c in base_final.columns if c not in id_cols]
    base_final = base_final[id_cols + otras_cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    base_final.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    print(f"\nBase final guardada en : {OUTPUT_PATH}")
    print(f"Dimensiones            : {base_final.shape[0]:>6} filas x {base_final.shape[1]:>4} columnas")
    for ola_val in sorted(base_final["OLA"].unique()):
        n = (base_final["OLA"] == ola_val).sum()
        print(f"  Ola {ola_val}: {n:>6} personas")

    return base_final


if __name__ == "__main__":
    base_final = main()
