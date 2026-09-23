"""
diagnostico_shap_monotonia.py
=====================================
Comprueba si la direccion inestable de algunas variables del nucleo SHAP
(`diagnostico_shap_signo.py`) se debe a que la relacion valor -> SHAP NO
es monotona. La correlacion de Spearman resume la relacion en un solo
signo: si el efecto tiene forma de U o de U invertida (por ejemplo, la
edad del jefe), ese signo es enganoso y "inestable" seria, en realidad,
"no monotono".

Para cada variable de la lista y cada combinacion fuente x algoritmo x
especificacion (se omite Logistica: es lineal, monotona por construccion),
se agrupa a los hogares en hasta 5 tramos (cuantiles; los valores unicos
si la variable tiene <= 6) y se calcula el SHAP medio de cada tramo.
Luego se clasifica la forma con el Spearman entre el ORDEN del tramo y su
SHAP medio: |rho_tramos| >= UMBRAL_MONOTONA la llama monotona (creciente o
decreciente segun el signo); en otro caso, no monotona. Una variable
binaria (2 tramos) es monotona por definicion y se omite.

Mismos modelos y mismos SHAP de los scripts originales (misma funcion
`calcular_shap`); no se reentrena nada distinto ni se cambia ningun
hiperparametro. Mismas cautelas que `diagnostico_shap_signo.py`
(2013->2016 in-sample con hiperparametros heredados).

INPUTS y ESTRUCTURA: los de `diagnostico_shap_signo.py`.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_shap_monotonia_tramos.csv
    (una fila por combinacion x variable x tramo: media de la variable,
    SHAP medio, n)
    data/processed/benchmark_resultados/diagnostico_shap_monotonia_resumen.csv
    (una fila por combinacion x variable con la forma clasificada)

COMO CORRER

    cd src/05_model && python -u diagnostico_shap_monotonia.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd

import modelo_utils as mu
from algoritmos_suite import algoritmos_presentes_en_registro, resolver_algoritmo
from diagnostico_shap import calcular_shap, entrenar
from diagnostico_shap_2013_2016 import entrenar_sobre_2013_2016
from diagnostico_shap_signo import FUENTES

VARIABLES = [
    # estables o moderadas (referencia)
    "riqueza_pca_hogar", "nivel_educ_max_hogar", "razon_dependencia_demografica",
    "valor_arriendo_pagado_hogar", "tasa_control_preventivo_hogar",
    # inestables o con signo dependiente de la fuente
    "edad_jefe", "n_desplazados_comunidad", "pct_ninos_madre_viva", "pct_ninos_padre_vivo",
    "tvip_puntaje_directo_hogar", "n_servicios_publicos_hogar", "n_bienes_durables_hogar",
    "tasa_cotizacion_pension_hogar",
]
EXCLUIR_ALGORITMOS = {"Logistica regularizada (elastic net, benchmark)"}
N_TRAMOS = 5
MAX_VALORES_UNICOS = 6
UMBRAL_MONOTONA = 0.8

RUTA_TRAMOS = mu.RESULTADOS_DIR / "diagnostico_shap_monotonia_tramos.csv"
RUTA_RESUMEN = mu.RESULTADOS_DIR / "diagnostico_shap_monotonia_resumen.csv"
_INICIO = time.time()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{(time.time() - _INICIO)/60:6.1f} min] {msg}", flush=True)


def tramos(x: pd.Series) -> pd.Series:
    """Etiqueta ordinal de tramo por hogar (NaN si no hay dato)."""
    if x.nunique() <= MAX_VALORES_UNICOS:
        return x.rank(method="dense")
    return pd.qcut(x, N_TRAMOS, labels=False, duplicates="drop")


def resumir(fuente, nombre_algo, espec, shap_values, nombres, x_shap):
    shap_df = pd.DataFrame(np.asarray(shap_values), columns=list(nombres))
    x_df = x_shap.reset_index(drop=True) if isinstance(x_shap, pd.DataFrame) else pd.DataFrame(x_shap, columns=list(nombres))
    filas_tramos, filas_resumen = [], []
    for v in VARIABLES:
        col = next((c for c in shap_df.columns if c == v or c == f"num__{v}"), None)
        if col is None:
            continue
        x = pd.to_numeric(x_df[col], errors="coerce")
        if x.nunique() <= 2:
            continue
        t = tramos(x)
        g = pd.DataFrame({"x": x, "shap": shap_df[col], "tramo": t}).dropna().groupby("tramo").agg(
            x_medio=("x", "mean"), shap_medio=("shap", "mean"), n=("x", "size")).reset_index()
        base = {"fuente": fuente, "algoritmo": nombre_algo, "especificacion": espec, "variable": v}
        for _, f in g.iterrows():
            filas_tramos.append({**base, **f.to_dict()})
        if len(g) >= 3:
            rho = g["tramo"].corr(g["shap_medio"], method="spearman")
            forma = "monotona" if abs(rho) >= UMBRAL_MONOTONA else "no monotona"
            signo = "creciente" if rho > 0 else "decreciente"
            filas_resumen.append({**base, "n_tramos": len(g), "rho_tramos": round(rho, 3),
                                  "forma": forma if forma == "no monotona" else f"monotona {signo}",
                                  "shap_tramo_bajo": g["shap_medio"].iloc[0], "shap_tramo_alto": g["shap_medio"].iloc[-1]})
    return filas_tramos, filas_resumen


def main() -> None:
    tr, rs = [], []
    for fuente, cfg in FUENTES.items():
        registro = pd.read_csv(cfg["registro"])
        for espec in cfg["especs"]:
            for algoritmo_raw in algoritmos_presentes_en_registro(cfg["registro"]):
                nombre_algo = resolver_algoritmo(algoritmo_raw)["nombre_bonito"]
                if nombre_algo in EXCLUIR_ALGORITMOS:
                    continue
                if not ((registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)).any():
                    continue
                log(f"=== {fuente} -- {nombre_algo} -- {espec} ===")
                if cfg["entrenar"] == "test":
                    pipe, x_train, x_ev, _, cat_cols = entrenar(algoritmo_raw, espec, registro)
                else:
                    pipe, x_train, x_ev, cat_cols = entrenar_sobre_2013_2016(algoritmo_raw, espec, registro)
                shap_values, nombres, x_shap = calcular_shap(algoritmo_raw, pipe, x_train, x_ev, cat_cols)
                a, b = resumir(fuente, nombre_algo, espec + cfg["sufijo_espec"], shap_values, nombres, x_shap)
                tr += a
                rs += b
    pd.DataFrame(tr).to_csv(RUTA_TRAMOS, index=False)
    resumen = pd.DataFrame(rs)
    resumen.to_csv(RUTA_RESUMEN, index=False)

    tabla = resumen.groupby("variable")["forma"].value_counts().unstack(fill_value=0)
    log("Formas por variable (n de combinaciones):\n" + tabla.to_string())
    log(f"FIN. Guardado: {RUTA_RESUMEN}")


if __name__ == "__main__":
    main()
