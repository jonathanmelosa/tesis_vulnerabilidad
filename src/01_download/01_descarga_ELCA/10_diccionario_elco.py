"""
10_diccionario_elco.py
========================

Extrae del PDF de documentación de ELCO (`documentacion_ELCO.pdf`, 711
páginas) una tabla código->pregunta->módulo->estadísticas, para poder
identificar qué variable P##### corresponde a qué pregunta del
cuestionario sin tener que leer el PDF a mano. Insumo necesario para
remapear las variables ELCO 2019/2022 a los nombres descriptivos que ya
usa el pipeline ELCA (`ing_*`, `gasto_*`, `serv_*`, etc.) -- ver
docs/decisions.md, sección "Consolidación ELCO", y la discusión sobre
comparabilidad de variables entre ELCA y ELCO.

QUÉ HACE

    1. Convierte el PDF a texto plano (pdftotext, si no existe ya en
       CONFIG['cache_texto']).
    2. Parsea cada entrada del diccionario: busca el patrón
       "<texto>(<código>)\\nArchivo: <módulo>" seguido de metadatos
       (Tipo, Casos válidos, etc.) y el bloque "Pregunta literal".
    3. Exporta una tabla larga: codigo, modulo, pregunta_literal,
       tipo, casos_validos.

    Este script NO decide qué variable ELCO corresponde a qué variable
    ELCA -- eso requiere criterio humano sobre el contenido de la
    pregunta. Solo deja la búsqueda por código o por texto de pregunta
    lista para hacerse en segundos en vez de buscar a mano en 711
    páginas.

INPUTS

    data/interim/raw/elca_2019/documentacion_ELCO.pdf

OUTPUTS

    data/processed/diccionario_elco.parquet
    data/processed/diccionario_elco.csv (para revisión rápida en Excel)

CÓMO CORRER

    python 10_diccionario_elco.py
    python 10_diccionario_elco.py --buscar "arriendo"     # busca por texto de pregunta
    python 10_diccionario_elco.py --codigo P158            # busca por código exacto
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data" / "interim" / "raw" / "elca_2019" / "documentacion_ELCO.pdf"
CACHE_TEXTO = PROJECT_ROOT / "data" / "interim" / "raw" / "elca_2019" / "documentacion_ELCO.txt"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "diccionario_elco.parquet"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "diccionario_elco.csv"

# Patrón de cada entrada: "<algo, hasta 120 chars>(<CODIGO>)\nArchivo: <modulo>\n"
PATRON_ENTRADA = re.compile(r"\n([^\n]{0,120}?)\s*\(([A-Z][0-9A-Z]+)\)\s*\nArchivo:\s*([^\n]+)\n")


def convertir_pdf_a_texto(pdf_path: Path, cache_path: Path) -> str:
    """Convierte el PDF a texto plano con pdftotext, cacheando el resultado
    (711 páginas -- convertir de nuevo cada vez es lento)."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    if not pdf_path.exists():
        print(f"ERROR: no se encontró {pdf_path}", file=sys.stderr)
        sys.exit(1)
    resultado = subprocess.run(["pdftotext", str(pdf_path), str(cache_path)], capture_output=True, text=True)
    if resultado.returncode != 0:
        print(f"ERROR al convertir PDF: {resultado.stderr}", file=sys.stderr)
        sys.exit(1)
    return cache_path.read_text(encoding="utf-8")


def parsear_diccionario(texto: str) -> pd.DataFrame:
    """Extrae (codigo, modulo, pregunta_literal, tipo, casos_validos) para cada entrada del diccionario."""
    posiciones = list(PATRON_ENTRADA.finditer(texto))
    filas = []
    for i, m in enumerate(posiciones):
        codigo, modulo = m.group(2), m.group(3).strip()
        inicio = m.end()
        fin = posiciones[i + 1].start() if i + 1 < len(posiciones) else len(texto)
        bloque = texto[inicio:fin]

        idx_preg = bloque.find("Pregunta literal")
        pregunta = ""
        if idx_preg >= 0:
            pregunta = " ".join(bloque[idx_preg + len("Pregunta literal"):idx_preg + 400].split())

        m_tipo = re.search(r"Tipo:\s*([^\n]+)", bloque)
        tipo = m_tipo.group(1).strip() if m_tipo else ""

        m_casos = re.search(r"Casos válidos:\s*([\d.,]+)", bloque)
        casos_validos = m_casos.group(1).strip() if m_casos else ""

        filas.append({
            "codigo": codigo, "modulo": modulo, "pregunta_literal": pregunta,
            "tipo": tipo, "casos_validos": casos_validos,
        })

    df = pd.DataFrame(filas)
    # DIRECTORIO/ORDEN/CONSECUTIVO_* se repiten en cada módulo (son llaves, no
    # preguntas de contenido) -- se conservan pero se marcan para poder
    # filtrarlas fácilmente en búsquedas de contenido.
    df["es_llave_identificador"] = df["codigo"].isin(
        ["DIRECTORIO", "ORDEN", "SECUENCIA_ENCUESTA", "SECUENCIA_P"]
    ) | df["codigo"].str.startswith("CONSECUTIVO_DANE")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buscar", help="Busca por texto en pregunta_literal (insensible a mayúsculas)")
    parser.add_argument("--codigo", help="Busca por código exacto (ej. P158)")
    parser.add_argument("--modulo", help="Filtra por módulo (ej. N_MERCADO_LABORAL)")
    args = parser.parse_args()

    if OUTPUT_PARQUET.exists():
        df = pd.read_parquet(OUTPUT_PARQUET)
    else:
        print("Convirtiendo PDF a texto (puede tardar unos segundos, se cachea después)...")
        texto = convertir_pdf_a_texto(PDF_PATH, CACHE_TEXTO)
        print("Parseando diccionario...")
        df = parsear_diccionario(texto)
        OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PARQUET, index=False)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Diccionario exportado: {len(df):,} variables -- {OUTPUT_PARQUET}")

    resultado = df
    if args.buscar:
        resultado = resultado[resultado["pregunta_literal"].str.contains(args.buscar, case=False, na=False)]
    if args.codigo:
        resultado = resultado[resultado["codigo"] == args.codigo]
    if args.modulo:
        resultado = resultado[resultado["modulo"] == args.modulo]

    if args.buscar or args.codigo or args.modulo:
        pd.set_option("display.max_colwidth", 100)
        print(resultado[["codigo", "modulo", "pregunta_literal"]].to_string(index=False))
    else:
        print(f"\n{len(df):,} variables en el diccionario, {df['modulo'].nunique()} módulos.")
        print("Usa --buscar/--codigo/--modulo para consultar.")


if __name__ == "__main__":
    main()
