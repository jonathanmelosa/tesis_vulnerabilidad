"""
comparar_auc_cv_vs_test_fbeta2_cv10.py
=============================

Version de `comparar_auc_cv_vs_test.py` sobre `registro_modelos_fbeta2_cv10.csv`
(umbral por F-beta beta=2, CV_FOLDS=10, N_ITER_BUSQUEDA=30 -- ver
`modelo_fbeta2_cv10_comparacion.py`), para el mismo diagnostico de
sobreajuste al agregar variables geoespaciales (Seccion~\\ref{subsec:marginal}
de la tesis) bajo la metodologia final adoptada. Misma logica exacta que el
script original, solo cambia el registro de entrada.

INPUTS
------
    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv

OUTPUTS
-------
    data/processed/benchmark_resultados/comparacion_auc_cv_vs_test_fbeta2_cv10.csv

CORRER
------
    python comparar_auc_cv_vs_test_fbeta2_cv10.py
"""

import pandas as pd

import modelo_utils as mu

ADVERTENCIA_BRECHA = 0.03  # diferencia en brecha (geo - base) que se marca como advertencia

REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv"

PARES_COMPARACION = [
    ("A", "AgeoDMSP"),
    ("B", "BgeoDMSP"),
]


def cargar_registro() -> pd.DataFrame:
    if not REGISTRO_CSV.exists():
        print(f"ERROR: no se encontro {REGISTRO_CSV} -- corre primero modelo_fbeta2_cv10_comparacion.py")
        raise SystemExit(1)
    return pd.read_csv(REGISTRO_CSV)


def calcular_brecha(registro: pd.DataFrame) -> pd.DataFrame:
    registro = registro.copy()

    def auc_cv_de_fila(fila):
        col = f"auc_cv_{fila['balanceo_elegido']}"
        return fila[col] if col in registro.columns and pd.notna(fila.get(col)) else float("nan")

    registro["auc_cv_elegido"] = registro.apply(auc_cv_de_fila, axis=1)
    registro["brecha_cv_menos_test"] = registro["auc_cv_elegido"] - registro["auc_roc_media"]
    return registro


def comparar(registro: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for algoritmo in sorted(registro["algoritmo"].unique()):
        sub = registro[registro["algoritmo"] == algoritmo].set_index("especificacion")
        for espec_base, espec_geo in PARES_COMPARACION:
            if espec_base not in sub.index or espec_geo not in sub.index:
                continue
            brecha_base = sub.loc[espec_base, "brecha_cv_menos_test"]
            brecha_geo = sub.loc[espec_geo, "brecha_cv_menos_test"]
            delta = brecha_geo - brecha_base
            filas.append({
                "algoritmo": algoritmo,
                "especificacion_base": espec_base,
                "auc_cv_base": round(sub.loc[espec_base, "auc_cv_elegido"], 4),
                "auc_test_base": round(sub.loc[espec_base, "auc_roc_media"], 4),
                "brecha_base": round(brecha_base, 4),
                "especificacion_geo": espec_geo,
                "auc_cv_geo": round(sub.loc[espec_geo, "auc_cv_elegido"], 4),
                "auc_test_geo": round(sub.loc[espec_geo, "auc_roc_media"], 4),
                "brecha_geo": round(brecha_geo, 4),
                "delta_brecha": round(delta, 4),
                "advertencia_sobreajuste": delta > ADVERTENCIA_BRECHA,
                "auc_test_mejora": sub.loc[espec_geo, "auc_roc_media"] > sub.loc[espec_base, "auc_roc_media"],
            })
    return pd.DataFrame(filas)


def main() -> None:
    print("=== comparar_auc_cv_vs_test_fbeta2_cv10.py ===\n")
    registro = cargar_registro()
    registro = calcular_brecha(registro)
    comparacion = comparar(registro)

    if comparacion.empty:
        print("Sin pares base/geo entrenados todavia.")
        return

    out_path = mu.RESULTADOS_DIR / "comparacion_auc_cv_vs_test_fbeta2_cv10.csv"
    comparacion.to_csv(out_path, index=False)

    for _, fila in comparacion.iterrows():
        print(f"--- {fila['algoritmo']} ---")
        print(f"  {fila['especificacion_base']}: AUC-CV={fila['auc_cv_base']:.4f}  AUC-test={fila['auc_test_base']:.4f}  brecha={fila['brecha_base']:.4f}")
        print(f"  {fila['especificacion_geo']}: AUC-CV={fila['auc_cv_geo']:.4f}  AUC-test={fila['auc_test_geo']:.4f}  brecha={fila['brecha_geo']:.4f}")
        print(f"  Cambio en brecha (geo - base): {fila['delta_brecha']:+.4f}")
        if fila["advertencia_sobreajuste"]:
            print(f"  ADVERTENCIA: la brecha CV-test crece mas de {ADVERTENCIA_BRECHA} al agregar geoespaciales -- posible sobreajuste, no solo mejor prediccion.")
        if fila["auc_test_mejora"]:
            print(f"  AUC de prueba SI mejora con geoespaciales ({fila['auc_test_geo']:.4f} > {fila['auc_test_base']:.4f}).")
        else:
            print(f"  AUC de prueba NO mejora con geoespaciales ({fila['auc_test_geo']:.4f} <= {fila['auc_test_base']:.4f}).")
        print()

    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
