"""
Version de `generar_predicciones_test_dmsp.py` para el target IPM --
genera y persiste las probabilidades de test (con metadatos por hogar) de
los 12 modelos ya calibrados en `registro_modelos_ipm.csv` (3 algoritmos x
Aipm/AipmgeoDMSP/Bipm/BipmgeoDMSP). Artefacto compartido para el bootstrap
con clustering por comunidad y el analisis de heterogeneidad sobre el
hallazgo de que DMSP-OLS SI aporta bajo IPM en XGBoost (ver
`diagnostico_bootstrap_ipm.py`).

NO vuelve a correr RandomizedSearchCV -- reutiliza `balanceo_elegido` y
`mejores_params` ya encontrados, UN solo fit por combinacion
(semilla=RANDOM_STATE=42).

OUTPUTS

    data/processed/benchmark_resultados/predicciones_test_ipm_Aipm.parquet
    data/processed/benchmark_resultados/predicciones_test_ipm_Bipm.parquet

COMO CORRER

    cd src/05_model && python -u generar_predicciones_test_ipm.py
"""

import json
from typing import Optional

import pandas as pd

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb
import modelo_logistica_regularizada as m_log
import modelo_xgboost as m_xgb

REGISTRO = mu.RESULTADOS_DIR / "registro_modelos_ipm.csv"
OUT_DIR = mu.RESULTADOS_DIR

ALGOS = {
    "XGBoost": "XGBoost",
    "HistGradientBoosting (sklearn)": "HistGradientBoosting",
    "Logistica regularizada (elastic net, benchmark)": "Logistica",
}
PARES = [("Aipm", "AipmgeoDMSP"), ("Bipm", "BipmgeoDMSP")]

METADATA_COLS = [
    "consecutivo", "consecutivo_c", "zona", "brecha_lp_ingreso",
    "estrato_verificado_hogar", "n_servicios_publicos_hogar", "n_bienes_durables_hogar",
]


def filtrar_params_modelo(hiperparametros_json: str) -> dict:
    d = json.loads(hiperparametros_json)
    return {k: v for k, v in d.items() if k.startswith("modelo__")}


def entrenar_y_predecir(algoritmo_raw: str, espec: str, registro: pd.DataFrame, metadata_extra: Optional[pd.DataFrame] = None):
    fila = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec)].iloc[0]
    balanceo = fila["balanceo_elegido"]
    params = filtrar_params_modelo(fila["hiperparametros"])

    train, test_raw = mu.cargar_datos(espec)
    faltantes = [c for c in METADATA_COLS if c not in test_raw.columns]
    if faltantes:
        assert metadata_extra is not None, f"faltan columnas {faltantes} y no hay metadata_extra para completarlas"
        test_raw = test_raw.merge(metadata_extra[["consecutivo"] + faltantes], on="consecutivo", how="left")

    if algoritmo_raw == "XGBoost":
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test_raw)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)
        pipe = m_xgb.construir_pipeline(x_train, y_train, balanceo)
    elif algoritmo_raw == "HistGradientBoosting (sklearn)":
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)
        x_test, y_test, _ = mu.preparar_arboles_nativos(test_raw)
        x_train, x_test = mu.alinear_columnas_categoricas(x_train, x_test, cat_cols)
        pipe = m_hgb.construir_pipeline(balanceo)
    else:
        x_train, y_train = mu.preparar_xy_crudo(train)
        x_test, y_test = mu.preparar_xy_crudo(test_raw)
        x_train, x_test = x_train.align(x_test, join="inner", axis=1)
        pipe = m_log.construir_pipeline(x_train, balanceo)

    if params:
        pipe.set_params(**params)
    pipe.fit(x_train, y_train)
    proba_test = pipe.predict_proba(x_test)[:, 1]

    meta = test_raw[METADATA_COLS].reset_index(drop=True).copy()
    meta["Y"] = y_test.reset_index(drop=True).values
    meta["proba"] = proba_test
    return meta


def main() -> None:
    registro = pd.read_csv(REGISTRO)
    _, metadata_ref = mu.cargar_datos("Aipm")

    for base, geo in PARES:
        piezas = []
        for algoritmo_raw, nombre in ALGOS.items():
            print(f"=== {nombre} -- {base} ===")
            m_base = entrenar_y_predecir(algoritmo_raw, base, registro, metadata_ref)
            print(f"=== {nombre} -- {geo} ===")
            m_geo = entrenar_y_predecir(algoritmo_raw, geo, registro, metadata_ref)

            assert (m_base["consecutivo"].values == m_geo["consecutivo"].values).all(), "hogares desalineados entre base y geo"
            assert (m_base["Y"].values == m_geo["Y"].values).all(), "Y debe coincidir entre base y geo"

            combinado = m_base.drop(columns=["proba"]).copy()
            combinado["algoritmo"] = nombre
            combinado["proba_base"] = m_base["proba"].values
            combinado["proba_geo"] = m_geo["proba"].values
            piezas.append(combinado)

        out = pd.concat(piezas, ignore_index=True)
        ruta = OUT_DIR / f"predicciones_test_ipm_{base}.parquet"
        out.to_parquet(ruta, index=False)
        print(f"Guardado: {ruta}  ({out.shape})")


if __name__ == "__main__":
    main()
