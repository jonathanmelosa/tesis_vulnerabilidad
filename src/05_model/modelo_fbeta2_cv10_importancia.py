"""
modelo_fbeta2_cv10_importancia.py
=====================================
Genera la importancia de variables de Random Forest y los coeficientes de
la regresion logistica regularizada, Modelos A y B, bajo la metodologia
final de la tesis (umbral por F-beta beta=2, CV_FOLDS=10,
N_ITER_BUSQUEDA=30 -- ver `modelo_fbeta2_cv10_comparacion.py`). Las
Figuras "Variables mas importantes, Random Forest, Modelo A" y "Mayores
coeficientes... regresion logistica" de la tesis (Seccion "Desempeno
comparativo de modelos"), y el parrafo que discute el Modelo B sin
figura, usaban hasta ahora la corrida ORIGINAL (umbral F1, CV_FOLDS=3,
N_ITER_BUSQUEDA=8) porque `modelo_fbeta2_cv10_comparacion.py` y sus
variantes (`..._rf_lgbm.py`, `..._random_forest_A.py`) solo exportan
metricas agregadas por semilla, no importancia por variable -- este
script llena ese hueco para las dos especificaciones.

NO REPITE LA BUSQUEDA DE HIPERPARAMETROS
-----------------------------------------------------------------------
El balanceo y los hiperparametros ganadores para cada (algoritmo,
especificacion) YA quedaron decididos y registrados (columnas
`balanceo_elegido`/`hiperparametros`) en `registro_modelos_fbeta2_cv10.csv`
por `modelo_fbeta2_cv10_comparacion_rf_lgbm.py`/`modelo_fbeta2_cv10_comparacion.py`
-- ese `RandomizedSearchCV` (30 iteraciones x 10 folds) ya se corrio una
vez y no hace falta pagarlo de nuevo solo para leer `feature_importances_`/
`coef_`. Este script LEE esas columnas directamente del registro y
reconstruye el pipeline ganador exacto (`construir_pipeline` del modulo
correspondiente + `set_params` con el JSON de `hiperparametros` ya
guardado), lo reentrena UNA sola vez sobre todo `x_train` (sin busqueda,
sin CV) y extrae la importancia/coeficientes de ese ajuste.

REPRODUCIBILIDAD
-----------------------------------------------------------------------
Determinista de punta a punta: el balanceo/hiperparametros no se vuelven a
elegir (se leen tal cual del registro, ya fijos), y el reentrenamiento usa
el mismo `RANDOM_STATE=42` de siempre
(`RandomForestClassifier(random_state=42)`/`LogisticRegression(
random_state=42)`, sin componente aleatoria adicional una vez fijados los
hiperparametros) sobre el mismo parquet de entrenamiento -- correr este
script dos veces, en cualquier maquina, produce exactamente el mismo CSV
de salida (se compara explicitamente al final contra la corrida anterior
si el archivo ya existe). No depende de ningun modelo previamente
serializado en disco (no se guarda ningun `.pkl`/`.joblib` en el
proyecto): la reproducibilidad viene de reconstruir el pipeline desde el
registro + un ajuste determinista, no de reusar un objeto en memoria de
otra corrida.

INPUTS

    data/processed/benchmark_train_test/modelo_{A,B}_2010_2013.parquet
    data/processed/benchmark_resultados/registro_modelos_fbeta2_cv10.csv
        (balanceo_elegido / hiperparametros ya decididos por
        modelo_fbeta2_cv10_comparacion.py / ..._rf_lgbm.py -- debe existir)

OUTPUTS

    data/processed/benchmark_resultados/fbeta2_cv10/random_forest/importancia_variables_modelo_{A,B}.csv
    data/processed/benchmark_resultados/fbeta2_cv10/logistica_regularizada/coeficientes_modelo_{A,B}.csv

COMO CORRER

    cd src/05_model && python -u modelo_fbeta2_cv10_importancia.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import modelo_utils as mu
import modelo_logistica_regularizada as m_log
import modelo_random_forest as m_rf

ESPECIFICACIONES = ["A", "B"]
REGISTRO_CSV = mu.RESULTADOS_DIR / "registro_modelos_fbeta2_cv10.csv"

_INICIO = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _INICIO
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed/60:6.1f} min] {msg}", flush=True)


def leer_config_ganadora(algoritmo: str, espec: str) -> tuple[str, dict]:
    """Lee balanceo_elegido/hiperparametros ya decididos para
    algoritmo/espec desde registro_modelos_fbeta2_cv10.csv -- no se
    vuelven a elegir aqui."""
    if not REGISTRO_CSV.exists():
        print(f"ERROR: no se encontro {REGISTRO_CSV} -- correr primero modelo_fbeta2_cv10_comparacion.py / "
              f"modelo_fbeta2_cv10_comparacion_rf_lgbm.py", file=sys.stderr)
        sys.exit(1)
    registro = pd.read_csv(REGISTRO_CSV)
    fila = registro[(registro["algoritmo"] == algoritmo) & (registro["especificacion"] == espec)]
    if fila.empty:
        print(f"ERROR: no hay fila para {algoritmo!r}/{espec!r} en {REGISTRO_CSV}", file=sys.stderr)
        sys.exit(1)
    fila = fila.iloc[0]
    balanceo = fila["balanceo_elegido"]
    hiperparametros = json.loads(fila["hiperparametros"])
    # class_weight/scale_pos_weight quedan fijados por `balanceo` dentro de
    # construir_pipeline(...) -- se descartan aqui para no pasarlos dos
    # veces a set_params.
    params_modelo = {k: v for k, v in hiperparametros.items() if k.startswith("modelo__")}
    return balanceo, params_modelo


def _comparar_con_corrida_previa(ruta: Path, df_nuevo: pd.DataFrame, columna_valor: str) -> None:
    if not ruta.exists():
        return
    df_previo = pd.read_csv(ruta)
    previo = df_previo.set_index("variable")[columna_valor].reindex(df_nuevo["variable"]).values
    nuevo = df_nuevo.set_index("variable")[columna_valor].reindex(df_nuevo["variable"]).values
    if np.allclose(previo, nuevo, equal_nan=True):
        log(f"  Reproducibilidad OK: {ruta.name} identico a la corrida anterior.")
    else:
        max_diff = np.nanmax(np.abs(previo - nuevo))
        log(f"  ADVERTENCIA: {ruta.name} difiere de la corrida anterior (diferencia maxima: {max_diff:.2e}).")


def correr_random_forest(espec: str) -> None:
    out_dir = mu.RESULTADOS_DIR / "fbeta2_cv10" / "random_forest"
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta = out_dir / f"importancia_variables_modelo_{espec}.csv"

    balanceo, params_modelo = leer_config_ganadora("Random Forest", espec)
    log(f"=== Random Forest -- Modelo {espec}: balanceo={balanceo!r}, params={params_modelo} ===")

    train, _ = mu.cargar_datos(espec)
    x_train, y_train = mu.preparar_xy_crudo(train)

    pipe = m_rf.construir_pipeline(x_train, balanceo)
    pipe.set_params(**params_modelo)
    pipe.fit(x_train, y_train)

    modelo_final = pipe.named_steps["modelo"]
    nombres_features = pipe.named_steps["prep"].get_feature_names_out()
    importancias = pd.DataFrame({
        "variable": nombres_features,
        "importancia": modelo_final.feature_importances_,
    }).sort_values("importancia", ascending=False)

    _comparar_con_corrida_previa(ruta, importancias, "importancia")
    importancias.to_csv(ruta, index=False)
    log(f"  Guardado: {ruta}")
    print(importancias.head(10).to_string(index=False))


def correr_logistica(espec: str) -> None:
    out_dir = mu.RESULTADOS_DIR / "fbeta2_cv10" / "logistica_regularizada"
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta = out_dir / f"coeficientes_modelo_{espec}.csv"

    balanceo, params_modelo = leer_config_ganadora("Logistica regularizada (elastic net, benchmark)", espec)
    log(f"=== Logistica regularizada -- Modelo {espec}: balanceo={balanceo!r}, params={params_modelo} ===")

    train, _ = mu.cargar_datos(espec)
    x_train, y_train = mu.preparar_xy_crudo(train)

    pipe = m_log.construir_pipeline(x_train, balanceo)
    pipe.set_params(**params_modelo)
    pipe.fit(x_train, y_train)

    modelo_final = pipe.named_steps["modelo"]
    nombres_features = pipe.named_steps["prep"].get_feature_names_out()
    coeficientes = pd.DataFrame({
        "variable": nombres_features,
        "coeficiente": modelo_final.coef_[0],
    }).sort_values("coeficiente", key=np.abs, ascending=False)

    _comparar_con_corrida_previa(ruta, coeficientes, "coeficiente")
    coeficientes.to_csv(ruta, index=False)
    log(f"  Guardado: {ruta}")
    print(coeficientes.head(10).to_string(index=False))


def main() -> None:
    for espec in ESPECIFICACIONES:
        correr_random_forest(espec)
        correr_logistica(espec)
    log("FIN.")


if __name__ == "__main__":
    main()
