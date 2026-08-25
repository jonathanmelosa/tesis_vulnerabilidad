"""
build_ingreso_hogar_elco.py
==============================

Construye el ingreso total mensual del hogar para ELCO 2019/2022,
replicando la lógica de `build_ingreso_hogar.py` (ELCA): 0 vs. NaN,
agregación a nivel hogar, sobre un cuestionario con estructura de
códigos totalmente distinta (ver Sección "Comparabilidad ELCA/ELCO" en
la tesis y docs/decisions.md).

La lógica de ingreso laboral (la parte más riesgosa) SE VERIFICÓ
directamente contra el diseño en papel del cuestionario
(`Formulario_Seguimiento_ELCO_2022.pdf`, sección N. MERCADO LABORAL,
páginas 85-91) -- no es una aproximación, ver detalle más abajo. Lo que
SIGUE sin verificar son los componentes marcados explícitamente como
"gap" al final del docstring (excepcionales de 12 meses, deflactor,
gasto, pobreza).

MAPEO DE COMPONENTES (código ELCO -> concepto ELCA), CONFIRMADO CONTRA
EL DICCIONARIO (`10_diccionario_elco.py`) Y LOS DATOS CRUDOS 2019
-------------------------------------------------------------------------
  Archivo N_MERCADO LABORAL (persona x ola):
    P2374S1 (Sí/No) + P2374S1A1 (valor)  -> arriendos          ~ ing_arriendos
    P2374S2 (Sí/No) + P2374S2A1 (valor)  -> pensión/jubilación ~ ing_pensiones
    P2374S3 (Sí/No) + P2374S3A1 (valor)  -> pensión alimenticia (SIN equivalente
                                             directo en ing_* de ELCA -- se suma
                                             a "otros no laborales" por defecto,
                                             revisar si se prefiere aparte)
    P2375S1 (Sí/No) + P2375S1A1 (valor)  -> dinero de hogares/personas del país
    P2375S2 (Sí/No) + P2375S2A1 (valor)  -> dinero de hogares/personas del exterior
    P2375S3 (Sí/No) + P2375S3A1 (valor)  -> ayudas en dinero de instituciones ~ ing_ayudas
    P2375S4 (Sí/No) + P2375S4A1 (valor)  -> intereses/CDT/dividendos ~ ing_intereses_div
    P2375S5 (Sí/No) + P2375S5A1 (valor)  -> cesantías (SIN equivalente directo)
    P2375S6 (Sí/No) + P2375S6A1 (valor)  -> otras fuentes (loterías, indemnizaciones) ~ ing_otros_nrem

  Ingreso LABORAL -- VERIFICADO contra Formulario_Seguimiento_ELCO_2022.pdf,
  sección N. MERCADO LABORAL:
    P158    (valor directo, "cuánto ganó el mes pasado en este empleo")
            -- pregunta 19, bloque "EMPLEO PRINCIPAL ASALARIADOS" (p. 86).
    P6749S1 (valor) + P6749S2 (a cuántos meses corresponde) -- pregunta 29,
            bloque "EMPLEO PRINCIPAL INDEPENDIENTES" (p. 88): "Honorarios o
            ganancia neta en el MES PASADO" + "A cuántos meses corresponde
            lo que recibió" -- CONFIRMADO que puede ser un valor de varios
            meses, la división por P6749S2 (cuando > 1) es la regla correcta,
            no una suposición.
    P7422S1 (valor, tras P7422 Sí/No) -- pregunta 42, bloque "DESOCUPADOS E
            INACTIVOS" (p. 91): "¿...recibió o ganó el mes pasado ingresos
            por concepto de trabajo?".
    Estas tres rutas NO se suman entre sí (se combinan con `coalesce`,
    tomando la primera no-nula en ese orden) -- CONFIRMADO en el formulario
    que son mutuamente excluyentes por construcción, no una aproximación:
    la pregunta 18 (posición ocupacional, P153) tiene saltos explícitos
    ("Pase a 29") que enrutan a empleados dependientes hacia P158,
    independientes/patrones hacia P6749S1, y trabajadores sin remuneración
    directamente a la pregunta de horas (sin pasar por ninguna de las dos);
    P7422 pertenece a un bloque totalmente distinto del cuestionario
    (personas DESOCUPADAS/INACTIVAS, no OCUPADAS), así que nunca coincide
    con P158 ni P6749S1 para la misma persona-mes.

  Módulo P_PENSIONES / O_PENSIONES (nombre de archivo cambia 2019->2022,
  código de pregunta NO cambia, verificado):
    P2415 (valor, "cuánto recibió el mes pasado por concepto de pensiones")
    -- se compara contra P2374S2A1 (misma pregunta, en otro módulo) y se
    usa el mayor de los dos no-nulos por persona, para no perder
    respuesta si una de las dos rutas quedó vacía. NO se suman (evitar
    doble conteo de la misma pensión).

QUÉ NO SE INCLUYE TODAVÍA (gaps conocidos, no ausencias silenciosas)
-------------------------------------------------------------------------
  - Componentes excepcionales retrospectivos de 12 meses (equivalente a
    herencias/pólizas/venta de inmueble/negocio de ELCA,
    `COMPONENTES_EXCEPCIONALES` en build_ingreso_hogar.py): no se buscaron
    todavía en el diccionario ELCO.
  - ingreso_ocasional (ELCA 2013/2016): sin buscar en ELCO.
  - Deflactación a pesos reales (build_deflactor_ipc.py cubre 2010-2016;
    para 2019/2022 haría falta extender esa serie con el IPC oficial de
    esos años).
  - Gasto del hogar (build_gasto_hogar.py) y clasificación de pobreza
    (build_pobreza_monetaria.py): fuera de alcance de este script.

CODIFICACIÓN VERIFICADA CONTRA LOS DATOS CRUDOS 2019 (no asumida)
-------------------------------------------------------------------------
  Indicadores Sí/No/No sabe: 1=Sí, 2=No, 9=No sabe/No informa, NaN=pregunta
  no aplicó a esa persona (filtro de la encuesta, no equivale a 0).
  Mismo criterio que ELCA: 2 (No) -> 0 explícito; 9 (No sabe) -> NaN;
  Sí (1) sin valor reportado -> NaN (no-respuesta real sobre el monto).
  NaN en el indicador (pregunta no aplicó) -> NaN, NO se asume 0.

Unidad de agregación: DIRECTORIO (hogar) + ORDEN (persona) -> se suma a
nivel DIRECTORIO. Concatenación de 2019 y 2022 (no cruce ancho), columna
`ola` = 2019 o 2022, igual que el resto del pipeline ELCO
(`08_consolidacion_bases_hogar_ELCO.py`).

Output: data/processed/ingreso_hogar_elco_longitudinal.parquet (ingreso laboral verificado contra el formulario; ver gaps pendientes arriba)
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "interim" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ingreso_hogar_elco_longitudinal.parquet"

# ── Archivos crudos por ola (nombre de archivo cambia 2019->2022 para el ─────
# módulo de pensiones: P_PENSIONES -> O_PENSIONES; N_MERCADO LABORAL y
# los demás mantienen el mismo nombre -- verificado listando ambas carpetas).
ARCHIVOS_POR_OLA = {
    2019: {
        "mercado_laboral": RAW_DIR / "elca_2019" / "ELCO_2019" / "csv" / "N_MERCADO LABORAL.csv",
        "pensiones": RAW_DIR / "elca_2019" / "ELCO_2019" / "csv" / "P_PENSIONES.csv",
    },
    2022: {
        "mercado_laboral": RAW_DIR / "elca_2022" / "BDATOS-ELCO-2022" / "cvs" / "N_MERCADO LABORAL.csv",
        "pensiones": RAW_DIR / "elca_2022" / "BDATOS-ELCO-2022" / "cvs" / "O_PENSIONES.csv",
    },
}

SEPARADOR_POR_OLA = {2019: ",", 2022: ";"}  # verificado en docs/decisions.md, consolidación ELCO

SI, NO, NO_SABE = "1", "2", "9"

# (columna_indicador, columna_valor, nombre_componente)
COMPONENTES_INDICADOR_VALOR = [
    ("P2374S1", "P2374S1A1", "arriendos"),
    ("P2374S2", "P2374S2A1", "pension_via_laboral"),
    ("P2374S3", "P2374S3A1", "pension_alimenticia"),
    ("P2375S1", "P2375S1A1", "dinero_hogares_pais"),
    ("P2375S2", "P2375S2A1", "dinero_hogares_exterior"),
    ("P2375S3", "P2375S3A1", "ayudas_instituciones"),
    ("P2375S4", "P2375S4A1", "intereses_dividendos"),
    ("P2375S5", "P2375S5A1", "cesantias"),
    ("P2375S6", "P2375S6A1", "otras_fuentes"),
]


def _a_numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


def _resolver_indicador_valor(df: pd.DataFrame, col_indicador: str, col_valor: str) -> pd.Series:
    """2 (No) -> 0 explícito; 9 (No sabe) -> NaN; 1 (Sí) sin valor -> NaN;
    NaN en el indicador (pregunta no aplicó) -> NaN. Mismo criterio que
    ZERO_TOKENS/MISSING_TOKENS de build_ingreso_hogar.py (ELCA)."""
    if col_indicador not in df.columns or col_valor not in df.columns:
        return pd.Series(np.nan, index=df.index)
    indicador = df[col_indicador].astype(str).str.strip()
    valor = _a_numero(df[col_valor])

    resultado = pd.Series(np.nan, index=df.index, dtype="float64")
    resultado.loc[indicador == NO] = 0.0
    resultado.loc[indicador == SI] = valor.loc[indicador == SI]  # NaN si no reportó monto
    # indicador == NO_SABE o NaN (no aplicó) -> se queda en NaN
    return resultado


def _ingreso_laboral_persona(df: pd.DataFrame) -> pd.Series:
    """Coalesce de las 3 rutas de ingreso laboral -- verificado contra
    Formulario_Seguimiento_ELCO_2022.pdf (ver docstring del módulo)."""
    p158 = _a_numero(df["P158"]) if "P158" in df.columns else pd.Series(np.nan, index=df.index)

    if "P6749S1" in df.columns:
        ganancia = _a_numero(df["P6749S1"])
        meses = _a_numero(df.get("P6749S2", pd.Series(np.nan, index=df.index)))
        # Mensual si meses es NaN o <=1; se divide si meses > 1 -- confirmado
        # contra P6749S2 ("¿A cuántos meses corresponde lo que recibió?") en el formulario.
        divisor = meses.where(meses > 1, 1.0)
        p6749 = ganancia / divisor
    else:
        p6749 = pd.Series(np.nan, index=df.index)

    if "P7422" in df.columns and "P7422S1" in df.columns:
        p7422 = _resolver_indicador_valor(df, "P7422", "P7422S1")
    else:
        p7422 = pd.Series(np.nan, index=df.index)

    return p158.combine_first(p6749).combine_first(p7422)


def cargar_ola(ola: int) -> pd.DataFrame:
    """Carga N_MERCADO LABORAL + pensiones de una ola, calcula los
    componentes de ingreso a nivel PERSONA (antes de agregar a hogar)."""
    rutas = ARCHIVOS_POR_OLA[ola]
    sep = SEPARADOR_POR_OLA[ola]

    if not rutas["mercado_laboral"].exists():
        raise FileNotFoundError(f"No se encontró {rutas['mercado_laboral']}")
    ml = pd.read_csv(rutas["mercado_laboral"], sep=sep, dtype=str, encoding="utf-8", on_bad_lines="skip")

    componentes = pd.DataFrame(index=ml.index)
    componentes["DIRECTORIO"] = ml["DIRECTORIO"]
    componentes["ORDEN"] = ml["ORDEN"]
    componentes["ola"] = ola

    componentes["ing_trabajo"] = _ingreso_laboral_persona(ml)
    for col_ind, col_val, nombre in COMPONENTES_INDICADOR_VALOR:
        componentes[f"ing_{nombre}"] = _resolver_indicador_valor(ml, col_ind, col_val)

    # Pensión: comparar la ruta de N_MERCADO_LABORAL (P2374S2A1, ya en
    # componentes["ing_pension_via_laboral"]) contra la de PENSIONES
    # (P2415) -- tomar la mayor no-nula por persona para no perder
    # respuesta si una de las dos rutas quedó vacía, sin sumarlas (misma
    # pregunta formulada dos veces en el cuestionario).
    if rutas["pensiones"].exists():
        pens = pd.read_csv(rutas["pensiones"], sep=sep, dtype=str, encoding="utf-8", on_bad_lines="skip")
        if "P2415" in pens.columns:
            pens_directa = pens[["DIRECTORIO", "ORDEN"]].copy()
            pens_directa["ing_pension_directa"] = _a_numero(pens["P2415"])
            componentes = componentes.merge(pens_directa, on=["DIRECTORIO", "ORDEN"], how="left")
        else:
            componentes["ing_pension_directa"] = np.nan
    else:
        componentes["ing_pension_directa"] = np.nan

    componentes["ing_pensiones"] = componentes[["ing_pension_via_laboral", "ing_pension_directa"]].max(axis=1, skipna=True)
    componentes = componentes.drop(columns=["ing_pension_via_laboral", "ing_pension_directa"])

    return componentes


def agregar_a_hogar(componentes: pd.DataFrame) -> pd.DataFrame:
    """Suma los componentes de ingreso de todas las personas del hogar
    (DIRECTORIO). Un componente NaN para TODAS las personas del hogar
    queda NaN (no 0) en el agregado; si al menos una persona tiene valor
    no-nulo, se suma tratando el resto como 0 (misma lógica de pandas
    .sum(skipna=True), coherente con el criterio de ELCA)."""
    cols_ing = [c for c in componentes.columns if c.startswith("ing_")]
    agregado = componentes.groupby(["DIRECTORIO", "ola"], as_index=False)[cols_ing].sum(min_count=1)
    agregado["ingreso_total_hogar"] = agregado[cols_ing].sum(axis=1, min_count=1)
    return agregado


def main() -> None:
    print("=== build_ingreso_hogar_elco.py (ingreso laboral verificado; otros componentes con gaps documentados) ===\n")
    partes = []
    for ola in [2019, 2022]:
        print(f"Cargando ola {ola}...")
        componentes = cargar_ola(ola)
        print(f"  {len(componentes):,} personas -- agregando a hogar...")
        agregado = agregar_a_hogar(componentes)
        print(f"  {len(agregado):,} hogares. Ingreso total hogar: media={agregado['ingreso_total_hogar'].mean():,.0f}  mediana={agregado['ingreso_total_hogar'].median():,.0f}  % NaN={agregado['ingreso_total_hogar'].isna().mean():.1%}")
        partes.append(agregado)

    resultado = pd.concat(partes, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nExportado: {OUTPUT_PATH} (ver docstring para que esta verificado y que falta)")
    print(f"Total: {len(resultado):,} filas hogar x ola")


if __name__ == "__main__":
    main()
