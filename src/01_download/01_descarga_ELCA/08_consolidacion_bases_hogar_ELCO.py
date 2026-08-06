"""
Encuesta Longitudinal Colombiana de Origen y Destino (ELCO)
Consolidación de las bases de datos de HOGARES – olas 2019 y 2022.

Autor: Jonathan Melo Sarta
Fecha: Julio 2026

Contexto
--------
ELCO tiene, a nivel de hogar, ocho módulos temáticos: datos de la vivienda,
servicios del hogar, bienes y activos, créditos, riesgos y choques, y gastos
(módulo puntual y módulo cíclico). Este script une, para cada ola por
separado, los módulos de hogar sobre la llave propia de esa ola, y luego
apila (concatena) las dos olas en un único dataset longitudinal con una fila
por hogar por ola.

Unidad de análisis
-------------------
Hogar × ola.  Cada hogar aporta exactamente una fila por ola en la que
participó (no se duplican filas por categoría de gasto ni por crédito, ver
sección "Módulos cíclicos" abajo).

Llave de unión intra-ola
--------------------------
Se usa DIRECTORIO (no CONSECUTIVO_DANE_ELCO_2019 / CONSECUTIVO_DANE_ELCO_2022)
como llave de cruce entre los módulos de una misma ola. Motivo:

  - En 2019, CONSECUTIVO_DANE_ELCO_2019 es una llave limpia y sin
    duplicados, pero en 2022 esa misma columna (y su análoga
    CONSECUTIVO_DANE_ELCO_2022) llegan ya corrompidas EN EL ARCHIVO FUENTE:
    el identificador numérico de 14-15 dígitos fue exportado en notación
    científica con solo 6 cifras significativas (p.ej. "1,16048E+14"), lo
    que provoca colisiones reales entre hogares distintos (2 095 de 15 499
    filas de 2022 quedan con un CONSECUTIVO_DANE_ELCO_2022 duplicado).
  - DIRECTORIO sí es único en todos los módulos de hogar de ambas olas y
    está presente en todos ellos, por lo que se usa como llave de cruce.
  - Las columnas CONSECUTIVO_DANE_ELCO_2019 y CONSECUTIVO_DANE_ELCO_2022
    igual se conservan en el output (identificador nominal / enlace al
    panel), pero NO se usan para el merge.

Módulos incluidos y su archivo por ola
-----------------------------------------
    Módulo                          Archivo 2019                                    Archivo 2022
    B_DATOS_DE_LA_VIVIENDA (base)   B_DATOS DE LA VIVIENDA.csv                       B_DATOS DE LA VIVIENDA.csv
    C_SERVICIOS_DEL_HOGAR           C_SERVICIOS DEL HOGAR.csv                        C_SERVICIOS DEL HOGAR.csv
    D_BIENES_Y_ACTIVOS_DEL_HOGAR    D_BIENES Y ACTIVOS DEL HOGAR.csv                 D_BIENES Y ACTIVOS DEL HOGAR.csv
    E_RIESGOS_Y_CHOQUES_DEL_HOGAR   E_RIESGOS Y CHOQUES DEL HOGAR.csv                E_RIESGOS Y CHOQUES DEL HOGAR.csv
    F_GASTOS_DEL_HOGAR              F_GASTOS DEL HOGAR.csv                           F_GASTOS DEL HOGAR.csv
    F_CICLO_GASTOS_DEL_HOGAR *      F_CICLO GASTOS DEL HOGAR.csv                     F_CICLO GASTOS DEL HOGAR.csv
    D_CREDITOS *                    D_CREDITOS.csv                                   D_CICLO BIENES Y ACTIVOS DEL HOGAR (CREDITOS).csv

  (*) Módulos cíclicos, ver abajo. D_BIENES_Y_ACTIVOS_DEL_HOGAR fue listado
      dos veces en el pedido original; se incluye una sola vez.

Módulos cíclicos (F_CICLO_GASTOS_DEL_HOGAR y D_CREDITOS)
------------------------------------------------------------
Ambos módulos traen varias filas por hogar (F_CICLO: hasta 64 filas/hogar,
una por categoría de gasto CAP ∈ {G1..G5, "0"}; D_CREDITOS: 1 a 5 filas por
hogar, una por crédito). Para preservar "una fila por hogar" se ENSANCHAN
antes de unirlos a la base:

  - F_CICLO_GASTOS_DEL_HOGAR: se pivotea usando el valor de CAP como sufijo
    de columna (p.ej. P10270_G1, P10270_G2, ..., P2127_G5). Cuando un hogar
    tiene más de una fila para la misma categoría CAP (subciclos), se
    numeran con un sufijo adicional _2, _3... en orden de aparición.
  - D_CREDITOS: no trae una categoría explícita, así que se numeran los
    créditos en el orden en que aparecen por hogar (sufijo _1, _2, ..., _5:
    P2097_1 = primer crédito, P2097_2 = segundo crédito, etc.).

Todas las columnas identificadoras (llaves, DIRECTORIO, SECUENCIA_*, ORDEN)
se descartan de estos dos módulos antes de pivotear (ya están en la base).

Separador y codificación de los archivos fuente
----------------------------------------------------
  - 2019: separador coma (","), UTF-8.
  - 2022: separador punto y coma (";"), UTF-8.
  Ambos se leen con dtype=str para no perder precisión en identificadores
  largos ni introducir NaN/float espurios en columnas categóricas.

Dimensiones esperadas
------------------------
  2019: 16 938 hogares.
  2022: 15 499 hogares.
  Total: 32 437 filas en la base unificada (una fila por hogar por ola).

Filas duplicadas exactas en el archivo fuente
-------------------------------------------------
A diferencia del script de personas (09_consolidacion_bases_personas_ELCO.py),
aquí NO se aplica una limpieza de filas 100% idénticas antes de unir. Se
verificó manualmente que ninguno de los 7 módulos de hogar, en ninguna de
las dos olas, trae DIRECTORIO repetido con contenido idéntico (ver
'validate="one_to_one"' en unir_modulos, que habría fallado si existiera
algún duplicado). Si en una actualización futura de los datos aparece un
duplicado exacto, el script se detendrá con un MergeError en vez de
mezclar información silenciosamente; en ese caso replicar aquí la función
deduplicar_exactas() del script de personas.

Estrategia de unión (how="left") y por qué
-----------------------------------------------
Cada módulo secundario se une a la base (B_DATOS_DE_LA_VIVIENDA) con un
LEFT JOIN: la base define el universo de hogares de la ola, y un hogar que
no respondió un módulo específico (p.ej. no tuvo créditos) simplemente
queda con NaN en esas columnas en vez de desaparecer de la base unificada.
Se usa 'validate="one_to_one"' en cada merge como una aserción explícita:
si algún módulo tuviera más de una fila por DIRECTORIO (que no haya sido
ya resuelto por pivotear_ciclico), pandas lanza un error inmediatamente en
vez de multiplicar filas silenciosamente.
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
    "/data/processed/hogares_ELCO.parquet"
)

KEY = "DIRECTORIO"

# ─── Módulos por ola ──────────────────────────────────────────────────────────
# (nombre_modulo, archivo, es_ciclico, columna_categoria_o_None)

MODULOS_2019 = [
    ("B_DATOS_DE_LA_VIVIENDA",        "B_DATOS DE LA VIVIENDA.csv",       False, None),
    ("C_SERVICIOS_DEL_HOGAR",         "C_SERVICIOS DEL HOGAR.csv",        False, None),
    ("D_BIENES_Y_ACTIVOS_DEL_HOGAR",  "D_BIENES Y ACTIVOS DEL HOGAR.csv", False, None),
    ("E_RIESGOS_Y_CHOQUES_DEL_HOGAR", "E_RIESGOS Y CHOQUES DEL HOGAR.csv",False, None),
    ("F_GASTOS_DEL_HOGAR",            "F_GASTOS DEL HOGAR.csv",           False, None),
    ("F_CICLO_GASTOS_DEL_HOGAR",      "F_CICLO GASTOS DEL HOGAR.csv",     True,  "CAP"),
    ("D_CREDITOS",                    "D_CREDITOS.csv",                   True,  None),
]

MODULOS_2022 = [
    ("B_DATOS_DE_LA_VIVIENDA",        "B_DATOS DE LA VIVIENDA.csv",                        False, None),
    ("C_SERVICIOS_DEL_HOGAR",         "C_SERVICIOS DEL HOGAR.csv",                         False, None),
    ("D_BIENES_Y_ACTIVOS_DEL_HOGAR",  "D_BIENES Y ACTIVOS DEL HOGAR.csv",                  False, None),
    ("E_RIESGOS_Y_CHOQUES_DEL_HOGAR", "E_RIESGOS Y CHOQUES DEL HOGAR.csv",                 False, None),
    ("F_GASTOS_DEL_HOGAR",            "F_GASTOS DEL HOGAR.csv",                            False, None),
    ("F_CICLO_GASTOS_DEL_HOGAR",      "F_CICLO GASTOS DEL HOGAR.csv",                      True,  "CAP"),
    ("D_CREDITOS",                    "D_CICLO BIENES Y ACTIVOS DEL HOGAR (CREDITOS).csv", True,  None),
]

# Columnas identificadoras que se repiten en todos los módulos y no aportan
# información nueva más allá del módulo base (se descartan al unir módulos
# secundarios para no duplicar columnas).
COLS_IDENTIFICADORAS = [
    "CONSECUTIVO_DANE_2010", "CONSECUTIVO_DANE_2013", "CONSECUTIVO_DANE_2016",
    "CONSECUTIVO_DANE_ELPS_2012", "CONSECUTIVO_DANE_ELPS_2015",
    "CONSECUTIVO_DANE_ELCO_2019", "CONSECUTIVO_DANE_ELCO_2022",
    "DIRECTORIO", "SECUENCIA_ENCUESTA", "SECUENCIA_P", "ORDEN",
]


# ─── Funciones utilitarias ────────────────────────────────────────────────────

def leer_modulo(carpeta: str, archivo: str, sep: str) -> pd.DataFrame:
    """
    Lee un módulo de hogar desde CSV.

    Decisiones:
    - dtype=str en TODAS las columnas: evita que pandas infiera tipos
      numéricos y trunque/redondee identificadores largos (el problema que
      ya afecta a CONSECUTIVO_DANE_ELCO_2022 en el archivo fuente, ver
      docstring del módulo) o convierta columnas categóricas (códigos como
      "01", "02") en enteros, perdiendo los ceros a la izquierda.
    - Se intenta UTF-8 primero porque es la codificación de la mayoría de
      los archivos de ambas olas. Ningún módulo de HOGAR resultó estar en
      otra codificación durante la construcción de este script (a
      diferencia de L_SALUD.csv de personas 2022, que sí requiere el
      fallback), pero se deja el mismo mecanismo de respaldo por robustez
      ante nuevas descargas del mismo instrumento.
    """
    path = os.path.join(carpeta, archivo)
    try:
        return pd.read_csv(path, sep=sep, dtype=str, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, dtype=str, low_memory=False, encoding="latin-1")


def pivotear_ciclico(df: pd.DataFrame, key: str, cat_col: str | None) -> pd.DataFrame:
    """
    Ensancha (long → wide) un módulo con varias filas por hogar a exactamente
    una fila por hogar, agregando un sufijo a cada columna de contenido.
    Es la pieza central que permite cumplir el requisito "una fila por
    hogar" para F_CICLO_GASTOS_DEL_HOGAR y D_CREDITOS (ver docstring del
    módulo, sección "Módulos cíclicos").

    Parámetros
    ----------
    df : módulo ya leído (todavía en formato largo, varias filas/hogar).
    key : nombre de la columna llave de hogar ("DIRECTORIO").
    cat_col : nombre de la columna que clasifica cada fila en una categoría
        con significado propio (p.ej. "CAP" en F_CICLO_GASTOS_DEL_HOGAR,
        que vale "G1".."G5"). Si el módulo no tiene una columna así
        (D_CREDITOS), se pasa None.

    Lógica del sufijo
    ------------------
    - cat_col definido: el sufijo es el valor de esa columna (p.ej. "G1"),
      de forma que las columnas resultantes quedan agrupadas por categoría
      de gasto (P10270_G1, P10270_G2, ...) y son interpretables sin mirar
      el orden de captura. Si, dentro de la misma categoría, un hogar tiene
      más de una fila (un subciclo repetido), se distingue con un sufijo de
      orden adicional: "G1" para la primera, "G1_2" para la segunda, etc.
    - cat_col es None: no hay una categoría con significado propio, así que
      el sufijo es simplemente el número de orden de aparición de la fila
      dentro del hogar (1, 2, 3, ...) — p.ej. P2097_1 = datos del primer
      crédito reportado, P2097_2 = datos del segundo, etc. El orden de
      captura del crédito en la encuesta se preserva pero no tiene un
      significado temático adicional (a diferencia de CAP).

    Columnas identificadoras
    -------------------------
    Antes de pivotear se descartan todas las columnas de COLS_IDENTIFICADORAS
    excepto la llave: esas columnas (SECUENCIA_*, ORDEN, los distintos
    CONSECUTIVO_DANE_*) son redundantes con lo que ya aporta el módulo base
    y no varían de forma útil entre las filas de un mismo hogar.

    Implementación
    ---------------
    Se agrupa por el sufijo calculado y, para cada grupo, se indexa por la
    llave de hogar y se renombra cada columna con el sufijo correspondiente;
    luego se concatenan todos los grupos por columnas (axis=1), alineando
    por la llave de hogar. Un hogar que no tiene fila para un sufijo dado
    queda con NaN en ese bloque de columnas.
    """
    df = df.drop(columns=[c for c in COLS_IDENTIFICADORAS if c in df.columns and c != key])
    value_cols = [c for c in df.columns if c != key]

    if cat_col is not None and cat_col in value_cols:
        value_cols = [c for c in value_cols if c != cat_col]
        orden_dentro_cat = df.groupby([key, cat_col]).cumcount() + 1
        sufijo = df[cat_col].astype(str)
        sufijo = sufijo.where(orden_dentro_cat == 1, sufijo + "_" + orden_dentro_cat.astype(str))
    else:
        sufijo = (df.groupby(key).cumcount() + 1).astype(str)

    df = df[[key] + value_cols].copy()
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
    Une, para UNA ola, todos sus módulos de hogar en una sola tabla ancha
    (una fila por hogar).

    Pasos
    -----
    1. El primer elemento de `modulos` es siempre el módulo base
       (B_DATOS_DE_LA_VIVIENDA). Se lee tal cual, SIN descartar ninguna
       columna: es la única tabla que conserva sus columnas identificadoras
       completas (CONSECUTIVO_DANE_*, SECUENCIA_*, ORDEN), que sirven como
       referencia/documentación del hogar en el output final.
    2. Cada módulo siguiente se lee y:
       - si es cíclico (es_ciclico=True), se ensancha primero con
         pivotear_ciclico() para que quede una fila por hogar;
       - si no, simplemente se le quitan las columnas identificadoras
         redundantes (COLS_IDENTIFICADORAS) porque ya están en la base y
         mantenerlas duplicaría información sin agregar nada.
    3. El módulo resultante (ya con una fila por hogar) se une a la base
       acumulada con LEFT JOIN sobre DIRECTORIO y `validate="one_to_one"`
       (ver docstring del módulo, sección "Estrategia de unión").
    4. Al final se agrega la columna OLA (2019 o 2022) para poder
       distinguir e identificar cada fila tras apilar ambas olas en main().

    El progreso (nombre de módulo, filas, columnas) se imprime en cada paso
    para poder auditar visualmente que ningún módulo se leyó vacío o con
    una cantidad de columnas inesperada.
    """
    nombre_base, archivo_base, _, _ = modulos[0]
    base = leer_modulo(carpeta, archivo_base, sep)
    print(f"    {nombre_base:32s}: {base.shape[0]:>6} filas x {base.shape[1]:>3} cols")

    for nombre, archivo, es_ciclico, cat_col in modulos[1:]:
        mod = leer_modulo(carpeta, archivo, sep)

        if es_ciclico:
            mod = pivotear_ciclico(mod, KEY, cat_col)
            print(f"    {nombre:32s}: {mod.shape[0]:>6} hogares x {mod.shape[1]:>3} cols (pivoteado)")
        else:
            mod = mod.drop(
                columns=[c for c in COLS_IDENTIFICADORAS if c in mod.columns and c != KEY]
            )
            print(f"    {nombre:32s}: {mod.shape[0]:>6} filas x {mod.shape[1]:>3} cols")

        # how="left": la base define el universo de hogares de la ola; un
        # hogar sin registro en este módulo queda con NaN, no desaparece.
        # validate="one_to_one": aserción de integridad — si `mod` tuviera
        # más de una fila por DIRECTORIO, el script debe fallar aquí en vez
        # de multiplicar filas de `base` silenciosamente.
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
    Cada ola se procesa de forma independiente y el resultado se concatena
    con pd.concat(axis=0): esto produce una fila por hogar POR OLA (2019 y
    2022 son filas distintas, no columnas distintas de un mismo hogar). Es
    la opción que se confirmó con el usuario: la unidad de análisis es el
    hogar, y ELCO 2022 no es solo un re-cruce de los mismos hogares de 2019
    (hay hogares nuevos y hogares que salen del panel), por lo que un cruce
    ancho (wide) perdería o generaría NaN innecesarios. pd.concat usa la
    UNIÓN de columnas: las columnas que solo existen en una ola (p.ej. las
    nuevas preguntas P747S1/P810S5/P2_S1... que 2022 agrega en
    B_DATOS_DE_LA_VIVIENDA) quedan en NaN para las filas de la otra ola.

    Orden de columnas del output
    -------------------------------
    Primero los identificadores clave (OLA, DIRECTORIO y los
    CONSECUTIVO_DANE_ELCO_* nominales), y después el resto de columnas en
    el orden en que fueron apareciendo al unir los módulos. Esto facilita
    ubicar visualmente la fila de un hogar específico sin tener que buscar
    la llave entre cientos de columnas de contenido.
    """
    print("Procesando ola 2019...")
    hogares_2019 = unir_modulos(RAW_2019, ",", MODULOS_2019, 2019)

    print("\nProcesando ola 2022...")
    hogares_2022 = unir_modulos(RAW_2022, ";", MODULOS_2022, 2022)

    base_final = pd.concat([hogares_2019, hogares_2022], axis=0, ignore_index=True)

    # Identificadores primero, luego el resto en orden de aparición.
    id_cols = [c for c in ["OLA", "DIRECTORIO", "CONSECUTIVO_DANE_ELCO_2019",
                            "CONSECUTIVO_DANE_ELCO_2022"] if c in base_final.columns]
    otras_cols = [c for c in base_final.columns if c not in id_cols]
    base_final = base_final[id_cols + otras_cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    base_final.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")

    print(f"\nBase final guardada en : {OUTPUT_PATH}")
    print(f"Dimensiones            : {base_final.shape[0]:>6} filas x {base_final.shape[1]:>4} columnas")
    for ola_val in sorted(base_final["OLA"].unique()):
        n = (base_final["OLA"] == ola_val).sum()
        print(f"  Ola {ola_val}: {n:>6} hogares")

    return base_final


if __name__ == "__main__":
    base_final = main()
