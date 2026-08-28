"""
Diagnostico: estabilidad de la comparacion de balanceo de clases frente a
distintos splits de CV -- HistGradientBoosting, Modelo A, el caso donde
`balanceo_elegido` cambio de "ninguno" (folds=3/iter=8) a "balanced"
(folds=10/iter=30), con un margen chico entre las 3 estrategias en la
corrida robusta (balanced=0.7603, oversampling=0.7602, ninguno=0.759 --
ver conversacion con el usuario, 2026-08-28). El diagnostico de
convergencia por iteraciones (`diagnostico_convergencia_busqueda.py`) ya
mostro que N_ITER=30 esta razonablemente convergido para "balanced"
(+0.0006 entre iter 30 y 50) -- el paso que falta es chequear si ESE
margen entre estrategias sobrevive a un split de CV distinto, o si se
invierte (convergencia espuria: "balanced" gano por el split particular
que le toco, no porque sea realmente mejor).

QUE HACE

    Repite `comparar_balanceo_y_tunear` completo (3 balanceos x
    RandomizedSearchCV, n_iter=30, cv_folds=10 -- igual a la corrida
    robusta) 5 veces, variando `random_state` (mismas semillas que
    SEMILLAS=[42,1,2,3,4] usadas en el resto de la suite) -- esto cambia
    TANTO el split de StratifiedKFold COMO el muestreo de candidatos de
    RandomizedSearchCV (la pregunta real es "otro analista con otra
    semilla, hubiera llegado a la misma conclusion", no solo "otro split
    con los mismos candidatos"). Si `balanceo_elegido` es "balanced" en
    las 5 semillas, la eleccion es robusta. Si cambia, confirma que estaba
    al borde del ruido de la estimacion por CV.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_estabilidad_balanceo_histgb_A.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_estabilidad_balanceo.py
"""

import time
from datetime import datetime

import pandas as pd

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb

CV_FOLDS_DIAGNOSTICO = 10
N_ITER_DIAGNOSTICO = 30
SEMILLAS_DIAGNOSTICO = mu.SEMILLAS  # [42, 1, 2, 3, 4] -- mismas semillas que el resto de la suite

OUT_CSV = mu.RESULTADOS_DIR / "diagnostico_estabilidad_balanceo_histgb_A.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def main() -> None:
    train, _ = mu.cargar_datos("A")
    x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)

    log(f"HistGradientBoosting -- Modelo A -- {len(SEMILLAS_DIAGNOSTICO)} semillas, cv_folds={CV_FOLDS_DIAGNOSTICO}, n_iter={N_ITER_DIAGNOSTICO}")
    log(f"train: {x_train.shape}")

    filas = []
    for semilla in SEMILLAS_DIAGNOSTICO:
        resultado = mu.comparar_balanceo_y_tunear(
            construir_pipeline_fn=m_hgb.construir_pipeline,
            param_distributions_fn=lambda b: m_hgb.PARAM_DIST,
            x_train=x_train, y_train=y_train,
            cv_folds=CV_FOLDS_DIAGNOSTICO, n_iter_busqueda=N_ITER_DIAGNOSTICO,
            random_state=semilla,
        )
        auc = resultado["auc_cv_por_balanceo"]
        fila = {"semilla": semilla, "balanceo_elegido": resultado["balanceo_elegido"], **{f"auc_cv_{b}": auc[b] for b in mu.BALANCEOS}}
        filas.append(fila)
        log(f"  semilla={semilla}: elegido={resultado['balanceo_elegido']}  AUC-CV={auc}")

    detalle = pd.DataFrame(filas)
    detalle.to_csv(OUT_CSV, index=False)

    print("\n=== Resumen ===")
    print(detalle.to_string(index=False))
    conteo = detalle["balanceo_elegido"].value_counts()
    print(f"\nVeces que gano cada balanceo (de {len(SEMILLAS_DIAGNOSTICO)} semillas): {conteo.to_dict()}")
    for b in mu.BALANCEOS:
        col = f"auc_cv_{b}"
        print(f"  {b:12s}: media={detalle[col].mean():.4f}  std={detalle[col].std():.4f}  rango=[{detalle[col].min():.4f}, {detalle[col].max():.4f}]")

    log(f"CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
