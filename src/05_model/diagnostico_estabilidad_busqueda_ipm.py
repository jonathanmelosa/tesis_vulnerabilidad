"""
Chequeo de estabilidad de la busqueda de hiperparametros frente a
distintas semillas -- mismo tipo de diagnostico que
`diagnostico_estabilidad_balanceo.py` (usado alli para HistGB/pobreza
monetaria), aplicado ahora a XGBoost bajo target IPM, donde
`diagnostico_bootstrap_ipm.py` encontro un efecto SIGNIFICATIVO de
DMSP-OLS (Aipm->AipmgeoDMSP: +0.0128, p=0.002; Bipm->BipmgeoDMSP:
+0.0198, p<0.001). Pregunta: ¿ese delta refleja informacion real de
DMSP-OLS, o parte de el es que la busqueda aleatoria de RandomizedSearchCV
para AipmgeoDMSP/BipmgeoDMSP tuvo mas suerte encontrando buenos
hiperparametros que la busqueda (independiente) de Aipm/Bipm?

QUE HACE

    Repite `comparar_balanceo_y_tunear` completo (3 balanceos x
    RandomizedSearchCV, n_iter=30, cv_folds=10 -- misma config que la
    corrida ya hecha) 5 veces por especificacion, variando `random_state`
    (mismas semillas SEMILLAS=[42,1,2,3,4]), para las 4 especificaciones
    XGBoost-IPM (Aipm, AipmgeoDMSP, Bipm, BipmgeoDMSP). Si el AUC-CV de la
    especificacion con DMSP-OLS le gana consistentemente a la sin DMSP-OLS
    en las 5 semillas (no solo en la semilla 42 ya usada), la ganancia no
    es un artefacto de una busqueda con suerte.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_estabilidad_busqueda_ipm.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_estabilidad_busqueda_ipm.py
"""

import time
from datetime import datetime

import pandas as pd

import modelo_utils as mu
import modelo_xgboost as m_xgb

CV_FOLDS = 10
N_ITER_BUSQUEDA = 30
ESPECIFICACIONES = ["Aipm", "AipmgeoDMSP", "Bipm", "BipmgeoDMSP"]
SEMILLAS_DIAGNOSTICO = mu.SEMILLAS

OUT_CSV = mu.RESULTADOS_DIR / "diagnostico_estabilidad_busqueda_ipm.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def main() -> None:
    filas = []
    for espec in ESPECIFICACIONES:
        train, _ = mu.cargar_datos(espec)
        x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)

        log(f"=== XGBoost -- {espec} -- {len(SEMILLAS_DIAGNOSTICO)} semillas ===")
        for semilla in SEMILLAS_DIAGNOSTICO:
            resultado = mu.comparar_balanceo_y_tunear(
                construir_pipeline_fn=lambda b: m_xgb.construir_pipeline(x_train, y_train, b),
                param_distributions_fn=lambda b: m_xgb.PARAM_DIST,
                x_train=x_train, y_train=y_train,
                cv_folds=CV_FOLDS, n_iter_busqueda=N_ITER_BUSQUEDA,
                random_state=semilla,
            )
            auc = resultado["auc_cv_por_balanceo"]
            mejor_auc = max(auc.values())
            filas.append({
                "especificacion": espec, "semilla": semilla,
                "balanceo_elegido": resultado["balanceo_elegido"],
                "auc_cv_mejor": mejor_auc, **{f"auc_cv_{b}": auc[b] for b in mu.BALANCEOS},
            })
            log(f"  semilla={semilla}: elegido={resultado['balanceo_elegido']}  AUC-CV mejor={mejor_auc:.4f}  (detalle: {auc})")

    detalle = pd.DataFrame(filas)
    detalle.to_csv(OUT_CSV, index=False)

    print("\n=== Resumen: AUC-CV mejor por especificacion x semilla ===")
    tabla = detalle.pivot(index="semilla", columns="especificacion", values="auc_cv_mejor")
    tabla = tabla[ESPECIFICACIONES]
    print(tabla.to_string())

    print("\n=== Delta (con DMSP-OLS - sin) por semilla ===")
    tabla["delta_A"] = tabla["AipmgeoDMSP"] - tabla["Aipm"]
    tabla["delta_B"] = tabla["BipmgeoDMSP"] - tabla["Bipm"]
    print(tabla[["delta_A", "delta_B"]].to_string())
    print(f"\nMedia delta_A: {tabla['delta_A'].mean():+.4f}  (semillas con delta>0: {(tabla['delta_A']>0).sum()}/5)")
    print(f"Media delta_B: {tabla['delta_B'].mean():+.4f}  (semillas con delta>0: {(tabla['delta_B']>0).sum()}/5)")

    print(f"\nCSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
