"""
Genera matrices de confusion (conteos absolutos y porcentajes) para los 3
algoritmos x 4 especificaciones x 2 targets (pobreza monetaria e IPM) de
la comparacion robusta (F-beta=2, CV_FOLDS=10, N_ITER_BUSQUEDA=30) --
pedido por el usuario (2026-08-30).

Fuente: los CSV `metricas_multiples_semillas_modelo_{espec}.csv` que ya
genera cada corrida (uno por semilla, con columnas tn/fp/fn/tp) -- NO
reentrena nada. Se promedian los conteos de las 5 semillas (redondeando
al entero mas cercano) para tener una matriz representativa por
algoritmo/especificacion, mas la version de la semilla de referencia
(42) para quien prefiera un unico corte reproducible exacto en vez de un
promedio.

Porcentaje: cada celda como % del total de hogares de test (las 4 celdas
de una matriz suman 100%) -- la convencion mas comun para una matriz de
confusion reportada de forma standalone (distinta de recall/precision,
que ya se reportan aparte en las tablas de desempeno).

OUTPUTS

    data/processed/benchmark_resultados/matrices_confusion.csv
    (formato largo: una fila por algoritmo x especificacion x target,
    con conteos promedio y de semilla 42, y sus porcentajes)

COMO CORRER

    python src/tabla_matrices_confusion.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = REPO_ROOT / "data" / "processed" / "benchmark_resultados"

ALGORITMOS_DIRS = {
    "XGBoost": "xgboost",
    "HistGradientBoosting": "histgradientboosting",
    "Logistica regularizada": "logistica_regularizada",
}

TARGETS = {
    "Monetaria": {
        "carpeta": RESULTADOS_DIR / "fbeta2_cv10",
        "especificaciones": ["A", "AgeoDMSP", "B", "BgeoDMSP"],
    },
    "IPM": {
        "carpeta": RESULTADOS_DIR / "ipm",
        "especificaciones": ["Aipm", "AipmgeoDMSP", "Bipm", "BipmgeoDMSP"],
    },
}


def cargar_matriz(carpeta: Path, algoritmo_dir: str, espec: str) -> dict:
    ruta = carpeta / algoritmo_dir / f"metricas_multiples_semillas_modelo_{espec}.csv"
    if not ruta.exists():
        return None
    df = pd.read_csv(ruta)

    fila_ref = df[df["semilla"] == 42].iloc[0]
    prom = df[["tn", "fp", "fn", "tp"]].mean()

    n_test_ref = int(fila_ref[["tn", "fp", "fn", "tp"]].sum())
    n_test_prom = prom.sum()

    return {
        "tn_semilla42": int(fila_ref["tn"]), "fp_semilla42": int(fila_ref["fp"]),
        "fn_semilla42": int(fila_ref["fn"]), "tp_semilla42": int(fila_ref["tp"]),
        "tn_pct_semilla42": round(100 * fila_ref["tn"] / n_test_ref, 2),
        "fp_pct_semilla42": round(100 * fila_ref["fp"] / n_test_ref, 2),
        "fn_pct_semilla42": round(100 * fila_ref["fn"] / n_test_ref, 2),
        "tp_pct_semilla42": round(100 * fila_ref["tp"] / n_test_ref, 2),
        "n_test_semilla42": n_test_ref,
        "tn_promedio_5semillas": round(prom["tn"], 1), "fp_promedio_5semillas": round(prom["fp"], 1),
        "fn_promedio_5semillas": round(prom["fn"], 1), "tp_promedio_5semillas": round(prom["tp"], 1),
        "tn_pct_promedio": round(100 * prom["tn"] / n_test_prom, 2),
        "fp_pct_promedio": round(100 * prom["fp"] / n_test_prom, 2),
        "fn_pct_promedio": round(100 * prom["fn"] / n_test_prom, 2),
        "tp_pct_promedio": round(100 * prom["tp"] / n_test_prom, 2),
    }


def imprimir_matriz(nombre: str, m: dict) -> None:
    print(f"\n--- {nombre} (semilla 42, n={m['n_test_semilla42']}) ---")
    print(f"                 Predicho: No entra    Predicho: Entra")
    print(f"  Real: No entra   TN={m['tn_semilla42']:5d} ({m['tn_pct_semilla42']:5.2f}%)   FP={m['fp_semilla42']:5d} ({m['fp_pct_semilla42']:5.2f}%)")
    print(f"  Real: Entra      FN={m['fn_semilla42']:5d} ({m['fn_pct_semilla42']:5.2f}%)   TP={m['tp_semilla42']:5d} ({m['tp_pct_semilla42']:5.2f}%)")


def main() -> None:
    filas = []
    for target_nombre, target_cfg in TARGETS.items():
        for algoritmo_nombre, algoritmo_dir in ALGORITMOS_DIRS.items():
            for espec in target_cfg["especificaciones"]:
                m = cargar_matriz(target_cfg["carpeta"], algoritmo_dir, espec)
                if m is None:
                    print(f"AVISO: no se encontro {target_nombre}/{algoritmo_dir}/{espec}, se omite")
                    continue
                imprimir_matriz(f"{algoritmo_nombre} -- {target_nombre} -- {espec}", m)
                filas.append({"target": target_nombre, "algoritmo": algoritmo_nombre, "especificacion": espec, **m})

    out = pd.DataFrame(filas)
    out_path = RESULTADOS_DIR / "matrices_confusion.csv"
    out.to_csv(out_path, index=False)
    print(f"\n\nCSV consolidado: {out_path}  ({len(out)} matrices)")


if __name__ == "__main__":
    main()
