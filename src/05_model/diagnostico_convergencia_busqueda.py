"""
Diagnostico: convergencia de RandomizedSearchCV (N_ITER_BUSQUEDA) para el
caso mas ambiguo de la comparacion folds=3/iter=8 vs. folds=10/iter=30 --
HistGradientBoosting, Modelo A, donde `balanceo_elegido` cambio de
"ninguno" a "balanced" con una ganancia de AUC-CV chica (+0.0088). Ver
conversacion con el usuario (Jonathan Melo, 2026-08-28): antes de aceptar
esa comparacion de balanceo como una diferencia real (no ruido de una
busqueda que no convergio), hay que chequear si el AUC-CV del candidato
ganador ya se habia estabilizado bien antes de la iteracion 30, o si
seguia subiendo -- en cuyo caso 30 iteraciones tampoco alcanzarian.

QUE HACE

    Corre UN solo RandomizedSearchCV con n_iter=50 (mas alla de las 30 de
    la corrida "robusta") y cv=10 folds, para el balanceo ganador
    ("balanced") de HistGradientBoosting Modelo A. `cv_results_` preserva
    el orden en que RandomizedSearchCV genero/evaluo los candidatos (no
    esta ordenado por score) -- se calcula el maximo acumulado de
    mean_test_score en ese orden, que es exactamente la curva "mejor score
    visto hasta la iteracion k" que se necesita para juzgar si N_ITER=30
    ya habia convergido.

    No reentrena el modelo final (`refit=False`) -- no hace falta para
    este diagnostico, ahorra tiempo.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_convergencia_histgb_A.csv
    outputs/figures/modelos/diagnostico_convergencia_histgb_A.png

COMO CORRER

    cd src/05_model && python -u diagnostico_convergencia_busqueda.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import modelo_utils as mu
import modelo_histgradientboosting as m_hgb

N_ITER_DIAGNOSTICO = 50
CV_FOLDS_DIAGNOSTICO = 10
BALANCEO = "balanced"  # el ganador en la corrida folds=10/iter=30

OUT_CSV = mu.RESULTADOS_DIR / "diagnostico_convergencia_histgb_A.csv"
OUT_PNG = mu.PROJECT_ROOT / "outputs" / "figures" / "modelos" / "diagnostico_convergencia_histgb_A.png"


def main() -> None:
    train, _ = mu.cargar_datos("A")
    x_train, y_train, cat_cols = mu.preparar_arboles_nativos(train)

    print(f"HistGradientBoosting -- Modelo A -- balanceo={BALANCEO}")
    print(f"train: {x_train.shape} -- n_iter={N_ITER_DIAGNOSTICO}, cv_folds={CV_FOLDS_DIAGNOSTICO}")

    pipeline = m_hgb.construir_pipeline(BALANCEO)
    cv = StratifiedKFold(n_splits=CV_FOLDS_DIAGNOSTICO, shuffle=True, random_state=mu.RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline, param_distributions=m_hgb.PARAM_DIST, n_iter=N_ITER_DIAGNOSTICO,
        scoring=mu.SCORING, cv=cv, random_state=mu.RANDOM_STATE, n_jobs=-1, refit=False,
    )
    search.fit(x_train, y_train)

    detalle = pd.DataFrame({
        "iteracion": range(1, N_ITER_DIAGNOSTICO + 1),
        "mean_test_score": search.cv_results_["mean_test_score"],
        "std_test_score": search.cv_results_["std_test_score"],
    })
    detalle["mejor_acumulado"] = detalle["mean_test_score"].cummax()
    detalle.to_csv(OUT_CSV, index=False)

    for k in [8, 15, 30, 50]:
        fila = detalle.iloc[k - 1]
        print(f"  Hasta iteracion {k:2d}: mejor AUC-CV visto = {fila['mejor_acumulado']:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(detalle["iteracion"], detalle["mejor_acumulado"], marker="o", markersize=3, label="Mejor AUC-CV acumulado")
    plt.scatter(detalle["iteracion"], detalle["mean_test_score"], s=10, alpha=0.35, color="gray", label="AUC-CV por candidato")
    plt.axvline(8, color="tab:orange", linestyle="--", label="N_ITER original (8)")
    plt.axvline(30, color="tab:red", linestyle="--", label="N_ITER corrida robusta (30)")
    plt.xlabel("Iteracion de RandomizedSearchCV")
    plt.ylabel("AUC-ROC (CV)")
    plt.title("Convergencia de la busqueda de hiperparametros\nHistGradientBoosting, Modelo A, balanceo=balanced")
    plt.legend(fontsize=8)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)

    print(f"\nCSV: {OUT_CSV}")
    print(f"PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
