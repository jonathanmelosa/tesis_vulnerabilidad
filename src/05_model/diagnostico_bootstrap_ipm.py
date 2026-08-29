"""
Version de `diagnostico_bootstrap_dmsp.py` para el benchmark con target
IPM (pobreza multidimensional) en vez de pobreza monetaria -- mismo test
de significancia formal (bootstrap pareado sobre los hogares de test),
misma logica de reutilizar hiperparametros ya encontrados en
`registro_modelos_ipm.csv` (sin repetir RandomizedSearchCV -- UN solo fit
por combinacion). Motivacion: bajo IPM, XGBoost mostro deltas de AUC-ROC
notablemente mas grandes que bajo pobreza monetaria (+0.011 a +0.018,
vs. maximo ~0.002-0.003 con pobre_ingreso) -- este test formal confirma
si esos deltas son distinguibles del ruido de muestreo del test set.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_bootstrap_ipm.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_bootstrap_ipm.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb
import modelo_logistica_regularizada as m_log
import modelo_xgboost as m_xgb

REGISTRO = mu.RESULTADOS_DIR / "registro_modelos_ipm.csv"
N_BOOT = 2000
RNG = np.random.default_rng(mu.RANDOM_STATE)

ALGOS = {
    "XGBoost": "XGBoost",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "Logistica regularizada (elastic net, benchmark)": "Logistica",
}
PARES = [("Aipm", "AipmgeoDMSP"), ("Bipm", "BipmgeoDMSP")]


def filtrar_params_modelo(hiperparametros_json: str) -> dict:
    d = json.loads(hiperparametros_json)
    return {k: v for k, v in d.items() if k.startswith("modelo__")}


def entrenar_y_predecir(algoritmo_raw: str, espec: str, registro: pd.DataFrame):
    fila = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)].iloc[0]
    balanceo = fila["balanceo_elegido"]
    params = filtrar_params_modelo(fila["hiperparametros"])

    if algoritmo_raw == "XGBoost":
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)
        pipe = m_xgb.construir_pipeline(x_train, y_train, balanceo)
    elif algoritmo_raw == "HistGradientBoosting (sklearn)":
        train, test = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)
        pipe = m_hgb.construir_pipeline(balanceo)
    else:  # Logistica
        train, test = mu.cargar_datos(espec)
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)
        pipe = m_log.construir_pipeline(x_train, balanceo)

    if params:
        pipe.set_params(**params)
    pipe.fit(x_train, y_train)
    proba_test = pipe.predict_proba(x_test)[:, 1]
    return proba_test, y_test.reset_index(drop=True).values


def bootstrap_delta_auc(y: np.ndarray, proba_base: np.ndarray, proba_geo: np.ndarray, n_boot: int = N_BOOT):
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            idx = RNG.integers(0, n, size=n)
            y_b = y[idx]
        auc_base = roc_auc_score(y_b, proba_base[idx])
        auc_geo = roc_auc_score(y_b, proba_geo[idx])
        deltas[b] = auc_geo - auc_base
    return deltas


def main() -> None:
    registro = pd.read_csv(REGISTRO)
    filas = []

    for algoritmo_raw, nombre in ALGOS.items():
        for base, geo in PARES:
            print(f"\n=== {nombre} -- {base} vs {geo} ===")
            proba_base, y_base = entrenar_y_predecir(algoritmo_raw, base, registro)
            proba_geo, y_geo = entrenar_y_predecir(algoritmo_raw, geo, registro)
            assert (y_base == y_geo).all(), "y_test debe ser identico entre Xipm y XipmgeoDMSP (mismo test set)"
            y = y_base

            auc_base = roc_auc_score(y, proba_base)
            auc_geo = roc_auc_score(y, proba_geo)
            delta_obs = auc_geo - auc_base

            deltas = bootstrap_delta_auc(y, proba_base, proba_geo)
            ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
            p_valor = 2 * min((deltas > 0).mean(), (deltas < 0).mean())
            p_valor = min(p_valor, 1.0)

            print(f"  AUC {base}={auc_base:.4f}  AUC {geo}={auc_geo:.4f}  delta={delta_obs:+.4f}")
            print(f"  IC95% bootstrap del delta: [{ci_low:+.4f}, {ci_high:+.4f}]  p={p_valor:.3f}  {'(cruza 0 -- NO significativo)' if ci_low <= 0 <= ci_high else '(NO cruza 0 -- significativo)'}")

            filas.append({
                "algoritmo": nombre, "especificacion_base": base, "especificacion_geo": geo,
                "n_test": len(y), "auc_base": round(auc_base, 4), "auc_geo": round(auc_geo, 4),
                "delta_auc": round(delta_obs, 4), "ci95_low": round(ci_low, 4), "ci95_high": round(ci_high, 4),
                "p_valor": round(p_valor, 4), "cruza_cero": bool(ci_low <= 0 <= ci_high),
            })

    out = pd.DataFrame(filas)
    out_path = mu.RESULTADOS_DIR / "diagnostico_bootstrap_ipm.csv"
    out.to_csv(out_path, index=False)
    print("\n=== Resumen ===")
    print(out.to_string(index=False))
    print(f"\nCSV: {out_path}")


if __name__ == "__main__":
    main()
