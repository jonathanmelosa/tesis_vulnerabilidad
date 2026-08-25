"""
comparar_auc_cv_vs_test.py
=============================

Diagnostico de sobreajuste al agregar variables geoespaciales: compara,
para cada algoritmo, el AUC-ROC de validacion cruzada usado para elegir
balanceo/hiperparametros (calculado SOLO sobre el conjunto de
entrenamiento) contra el AUC-ROC real de prueba (holdout temporal
2013->2016, o probabilidades out-of-fold para las especificaciones Ageo3/
Bgeo3). Una brecha que se ENSANCHA al pasar de A/B a AgeoDMSP/BgeoDMSP es
la señal de que las columnas geoespaciales agregan sobreajuste sin aportar
poder predictivo real fuera de muestra -- exactamente lo que un ratio
n/p ajustado (n~3.089, p sube de ~165-169 a ~196-200 con DMSP-OLS, o a
~302-306 con las tres fuentes) haria esperable.

QUE HACE
--------
    1. Lee registro_modelos.csv (todas las filas ya entrenadas).
    2. Para cada fila, toma el AUC-CV de la estrategia de balanceo
       elegida (columna auc_cv_{balanceo_elegido}).
    3. Calcula la brecha: auc_cv_elegido - auc_roc_media (test/OOF).
    4. Agrupa por algoritmo y compara la brecha de cada especificacion
       "+Geo" contra su version base (A vs. AgeoDMSP; B vs. BgeoDMSP).
    5. Imprime y exporta una tabla de comparacion, marcando con una
       advertencia las combinaciones donde la brecha crece de forma
       importante (umbral configurable, ADVERTENCIA_BRECHA).

INPUTS
------
    data/processed/benchmark_resultados/registro_modelos.csv

OUTPUTS
-------
    data/processed/benchmark_resultados/comparacion_auc_cv_vs_test.csv

CORRER
------
    python comparar_auc_cv_vs_test.py
"""

import pandas as pd

import modelo_utils as mu

ADVERTENCIA_BRECHA = 0.03  # diferencia en brecha (geo - base) que se marca como advertencia

# Pares (especificacion_base, especificacion_geo) a comparar.
PARES_COMPARACION = [
    ("A", "AgeoDMSP"),
    ("B", "BgeoDMSP"),
]


def cargar_registro() -> pd.DataFrame:
    if not mu.REGISTRO_CSV.exists():
        print(f"ERROR: no se encontro {mu.REGISTRO_CSV} -- corre primero los scripts modelo_*.py")
        raise SystemExit(1)
    return pd.read_csv(mu.REGISTRO_CSV)


def calcular_brecha(registro: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna auc_cv_elegido (el AUC-CV de la estrategia de
    balanceo que gano) y brecha = auc_cv_elegido - auc_roc_media."""
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
    print("=== comparar_auc_cv_vs_test.py ===\n")
    registro = cargar_registro()
    registro = calcular_brecha(registro)
    comparacion = comparar(registro)

    if comparacion.empty:
        print("Sin pares base/geo entrenados todavia -- corre modelo_*.py primero con ESPECIFICACIONES_PRINCIPAL completo.")
        return

    out_path = mu.RESULTADOS_DIR / "comparacion_auc_cv_vs_test.csv"
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
