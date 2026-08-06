"""
04_construir_variables_visuales.py
===================================
Procesa los embeddings extraídos por 03_extraer_embeddings.py y construye
variables visuales agregadas a nivel hogar × ola, listas para los modelos
econométricos.

RESPONSABILIDAD EXCLUSIVA
    Transformar los vectores de embedding (uno por imagen) en variables que
    puedan ser utilizadas directamente como regresores a nivel hogar × ola.

    Este script NO realiza:
    · Extracción de embeddings (responsabilidad de script 03).
    · Descarga de imágenes ni consultas a la API (scripts 01 y 02).
    · Estimación de modelos ni inferencia causal (responsabilidad de script 05).

INPUTS
    data/processed/embeddings/embeddings_{modelo}.parquet
    (output de 03_extraer_embeddings.py — uno por modelo)

OUTPUTS
    data/processed/variables_visuales/variables_visuales_{modelo}.csv
        → variables agregadas a nivel hogar × ola (una fila por hogar × ola)
    data/processed/variables_visuales/reporte_variables_{modelo}.txt
        → estadísticas descriptivas del proceso de construcción

CÓMO CORRER
    python 04_construir_variables_visuales.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# TODO: implementar
# ──────────────────────────────────────────────────────────────────────────────
