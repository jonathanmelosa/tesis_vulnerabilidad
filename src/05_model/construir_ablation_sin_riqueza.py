"""
construir_ablation_sin_riqueza.py
====================================

ABLATION: simula un contexto SIN encuesta de hogares rica en preguntas de
infraestructura (sin n_servicios_publicos_hogar ni n_bienes_durables_hogar
-- las dos variables que explican por que DMSP-OLS resulta redundante,
ver Seccion 5.3 "Contribucion marginal de las variables geoespaciales" y
la correlacion parcial calculada en esa misma conversacion). Pregunta que
responde: si esas dos preguntas de la encuesta NO existieran (como
pasaria con datos administrativos pobres en infraestructura, o entre
olas de encuesta), ¿DMSP-OLS empezaria a aportar como sustituto?

QUE HACE
--------
    1. Para cada especificacion base (A, B) y su version +DMSP-OLS
       (AgeoDMSP, BgeoDMSP), carga los archivos ya construidos
       (train=2010_2013, test=2013_2016).
    2. Elimina n_servicios_publicos_hogar y n_bienes_durables_hogar de
       las covariables (si existen en ese archivo).
    3. Exporta 4 especificaciones nuevas: Anoriq, AnoriqGeo, Bnoriq,
       BnoriqGeo -- "noriq" = sin las variables de riqueza/servicios
       identificadas como redundantes con DMSP-OLS. Comparar Anoriq vs.
       AnoriqGeo (y Bnoriq vs. BnoriqGeo) aisla el efecto de agregar
       DMSP-OLS EXACTAMENTE en el escenario donde, por hipotesis, deberia
       ayudar mas.

INPUTS
------
    data/processed/benchmark_train_test/modelo_{A,B,AgeoDMSP,BgeoDMSP}_{2010_2013,2013_2016}.parquet

OUTPUTS
-------
    data/processed/benchmark_train_test/modelo_{Anoriq,AnoriqGeo,Bnoriq,BnoriqGeo}_{2010_2013,2013_2016}.parquet

CORRER
------
    python construir_ablation_sin_riqueza.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"

COLS_RIQUEZA_REDUNDANTES = ["n_servicios_publicos_hogar", "n_bienes_durables_hogar"]

# (especificacion_origen, especificacion_nueva)
PARES = [
    ("A", "Anoriq"), ("AgeoDMSP", "AnoriqGeo"),
    ("B", "Bnoriq"), ("BgeoDMSP", "BnoriqGeo"),
]
TRANSICIONES = ["2010_2013", "2013_2016"]


def main() -> None:
    print("=== construir_ablation_sin_riqueza.py ===\n")
    for espec_origen, espec_nueva in PARES:
        for transicion in TRANSICIONES:
            ruta_in = DATA_DIR / f"modelo_{espec_origen}_{transicion}.parquet"
            if not ruta_in.exists():
                print(f"ERROR: no se encontro {ruta_in}", file=sys.stderr)
                sys.exit(1)
            df = pd.read_parquet(ruta_in)

            presentes = [c for c in COLS_RIQUEZA_REDUNDANTES if c in df.columns]
            df_ablation = df.drop(columns=presentes)

            ruta_out = DATA_DIR / f"modelo_{espec_nueva}_{transicion}.parquet"
            df_ablation.to_parquet(ruta_out, index=False)
            print(f"  modelo_{espec_nueva}_{transicion}.parquet: {df_ablation.shape[0]:,} filas x {df_ablation.shape[1]:,} columnas (eliminadas: {presentes})")

    print("\nListo. Especificaciones nuevas: Anoriq, AnoriqGeo, Bnoriq, BnoriqGeo")
    print("(agregar a modelo_utils.ESPECIFICACIONES_PRINCIPAL o correr con una lista aparte)")


if __name__ == "__main__":
    main()
