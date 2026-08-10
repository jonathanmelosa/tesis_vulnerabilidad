"""
Utilidades compartidas por todos los scripts `modelo_*.py` de la suite de
comparacion de algoritmos del benchmark (ver docs/decisions.md, "Suite de
comparacion de algoritmos: logistica regularizada, random forest, XGBoost,
LightGBM, HistGradientBoosting").

Cada script `modelo_*.py` entrena y evalua UN algoritmo (para Modelo A y
Modelo B, ver `build_benchmark_train_test.py`) y usa estas utilidades para:
  1. Cargar los parquets de train/test ya construidos.
  2. Preparar X/Y -- dos rutas distintas segun si el algoritmo maneja NaN y
     categoricas nativamente (arboles de gradient boosting) o no (logistica,
     random forest), ver `preparar_arboles_nativos` / `preparar_lineal_rf`.
  3. Comparar estrategias de balanceo de clases y tunear hiperparametros
     por validacion cruzada (`comparar_balanceo_y_tunear`), decisiones
     confirmadas explicitamente por el usuario (no asumidas).
  4. Calcular metricas estandarizadas (`calcular_metricas`).
  5. Registrar el resultado en `registro_modelos.csv` (fuente de verdad,
     un upsert por (algoritmo, especificacion)) y regenerar el Excel
     formateado `registro_modelos.xlsx` a partir de ese CSV completo.

Estrategia de imputacion (para logistica y random forest, que NO manejan
NaN nativamente) -- CONFIRMADO CON EL USUARIO
--------------------------------------------------------------------------
Se usa SIEMPRE 0 (nunca la mediana) como constante de relleno para
variables numericas, combinada con una columna indicadora binaria
`missingindicator_{col}` (via `SimpleImputer(strategy="constant",
fill_value=0, add_indicator=True)`).

Por que 0 y no la mediana: gran parte de la missingness de este dataset es
CAUSADA por una pregunta filtro (ej. "¿recibio ayuda?" -> si "No", el
monto de la ayuda queda sin preguntar). La mediana de una columna asi solo
se calcula sobre quienes SI pasaron el filtro y respondieron -- asignarle
ese valor "tipico de quien si aplica" a alguien a quien la pregunta NO le
aplica contradice la logica del filtro (le inventa un valor que no tiene).
Usar 0 en cambio tiene una propiedad limpia en un modelo lineal: 0 x
coeficiente = 0 siempre, sin importar el coeficiente -- el relleno no
aporta ninguna contribucion a la prediccion de esas filas mas alla de lo
que capture el indicador de faltante, que es quien efectivamente absorbe
el efecto de "esto no aplica". Con la mediana, el mismo coeficiente que
debe ajustar la pendiente real sobre quienes SI respondieron queda
ademas contaminado por tener que "explicar" el valor inventado del grupo
filtrado, sesgando potencialmente esa pendiente. Esto aplica sin importar
si el "cero real" tiene sentido semantico para esa variable en particular
(monto de ayuda) o no (edad de titulacion para quien no tiene titulo) --
en ambos casos 0+indicador es mas seguro, y evita tener que clasificar
~165 covariables una por una segun si su missingness admite un cero
"real". Decision explicita del usuario, no un supuesto por defecto.

Categoricas: nueva categoria explicita "Sin dato" (NO moda -- la moda
borraria la señal de missingness estructural tratandola como el valor mas
comun).

Comparacion de estrategias de balanceo de clases -- CONFIRMADO CON EL
USUARIO
--------------------------------------------------------------------------
Se comparan 3 estrategias (sin SMOTE, no se incluyo en la seleccion del
usuario): "balanced" (reweighting via class_weight/scale_pos_weight),
"ninguno" (baseline, sin ajuste), "oversampling" (RandomOverSampler de
imbalanced-learn -- duplica filas de la clase minoritaria; se prefirio
sobre SMOTE porque SMOTE interpola vecinos y no esta bien definido con
NaN/categoricas mezcladas, mientras que RandomOverSampler solo duplica
filas existentes, compatible con cualquier tipo de dato). El oversampling
se hace DENTRO de cada fold de CV (via `imblearn.pipeline.Pipeline`, que
excluye el paso de resampling en tiempo de prediccion/validacion) para no
filtrar informacion del fold de validacion al de entrenamiento.

Metrica de seleccion: AUC-ROC (CONFIRMADO) -- no depende de un umbral de
clasificacion, y evita el problema de que el oversampling/SMOTE distorsiona
la calibracion de las probabilidades (comparar F1/recall a umbral fijo 0.5
entre estrategias de balanceo distintas no seria una comparacion justa).
Recall/precision/F1 se reportan en paralelo como referencia, no deciden.

Busqueda de hiperparametros: `RandomizedSearchCV` (CONFIRMADO) -- factible
en tiempo dado que son 5 algoritmos x 2 especificaciones x 3 estrategias de
balanceo. `N_ITER_BUSQUEDA=15` iteraciones aleatorias, `CV_FOLDS=3`
(StratifiedKFold) -- valores elegidos por costo computacional (no se
consultaron con el usuario explicitamente; documentados aqui para que
pueda pedir ajustarlos si el tiempo de computo lo permite).

Umbral de clasificacion -- CONFIRMADO CON EL USUARIO (hallazgo de la
primera corrida)
--------------------------------------------------------------------------
En la primera corrida de la suite, XGBoost y LightGBM-B ganaron la
comparacion de balanceo con la estrategia "ninguno" por una diferencia de
AUC-CV insignificante (~0.001, ruido) frente a "balanced" -- pero el
modelo resultante, sin reponderar clases ~23%/77%, produce probabilidades
tan comprimidas hacia 0 que casi ninguna fila cruza el umbral fijo de 0.5
(recall ~0.03-0.04 en XGBoost, precision >0.6). El umbral 0.5 es arbitrario
cuando la clase positiva es ~23% de la muestra, y distintas estrategias de
balanceo calibran las probabilidades de forma distinta -- comparar
recall/precision/F1 a 0.5 entre ellas no es una comparacion justa.

Solucion: el umbral de clasificacion NO se fija en 0.5 -- se elige por CV
(`elegir_umbral_por_cv`, sobre probabilidades out-of-fold via
`cross_val_predict`, maximizando F1) para cada modelo ya elegido (balanceo
+ hiperparametros). AUC-ROC (umbral-independiente) sigue mandando esa
seleccion previa -- esto solo corrige COMO se reportan recall/precision/F1,
no cambia que gana la comparacion de balanceo/hiperparametros.

Multiples semillas e intervalos de confianza -- CONFIRMADO CON EL USUARIO
--------------------------------------------------------------------------
Pregunta del usuario ("¿el mejor modelo fue Random Forest?") expuso que
las diferencias de AUC-ROC entre los 3-4 algoritmos mejor rankeados
(~0.002-0.005) son del orden de la variabilidad que ya se habia observado
por puro azar en la comparacion de balanceo -- sin una nocion de
incertidumbre, "el mejor" no es una afirmacion defendible.

Se re-entrena el modelo FINAL (balanceo y mejores_params YA elegidos por
`comparar_balanceo_y_tunear` -- eso NO se repite por semilla, seria
excesivo en tiempo de computo) con `SEMILLAS = [42, 1, 2, 3, 4]` (5
semillas, CONFIRMADO): cada semilla cambia el random_state del modelo (y
del remuestreo si el balanceo elegido es "oversampling") y del particionado
usado para re-elegir el umbral por CV -- mide la variabilidad debida a la
aleatoriedad propia del algoritmo (bootstrap de arboles, orden de
convergencia, etc.), NO la sensibilidad a la busqueda de
balanceo/hiperparametros (mantenida fija, ver `evaluar_multiples_semillas`).
Se reporta media, desviacion estandar e intervalo de confianza al 95%
(t de Student, 4 grados de libertad dado n=5) para AUC-ROC/recall/
precision/F1 -- estos, no un unico valor de una sola semilla, son los que
deben compararse entre algoritmos.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"
RESULTADOS_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados"
REGISTRO_CSV = RESULTADOS_DIR / "registro_modelos.csv"
REGISTRO_XLSX = RESULTADOS_DIR / "registro_modelos.xlsx"

COLS_NO_FEATURE = ["consecutivo", "consecutivo_c", "llave_compuesta", "Y"]

BALANCEOS = ["balanced", "ninguno", "oversampling"]
N_ITER_BUSQUEDA = 15
CV_FOLDS = 3
SCORING = "roc_auc"
RANDOM_STATE = 42
SEMILLAS = [42, 1, 2, 3, 4]

METRICAS_RESUMEN = ["auc_roc", "recall", "precision", "f1"]

COLUMNAS_REGISTRO = [
    "algoritmo", "especificacion", "fecha_entrenamiento",
    "n_train", "n_test", "n_covariables_originales", "n_covariables_modelo",
    "tasa_entrada_train", "tasa_entrada_test",
    "balanceo_elegido", "auc_cv_balanced", "auc_cv_ninguno", "auc_cv_oversampling",
    "n_semillas", "umbral_clasificacion_media",
    "auc_roc_media", "auc_roc_std", "auc_roc_ci95_low", "auc_roc_ci95_high",
    "recall_media", "recall_std", "recall_ci95_low", "recall_ci95_high",
    "precision_media", "precision_std", "precision_ci95_low", "precision_ci95_high",
    "f1_media", "f1_std", "f1_ci95_low", "f1_ci95_high",
    "tn_semilla42", "fp_semilla42", "fn_semilla42", "tp_semilla42",
    "estrategia_imputacion", "hiperparametros", "observaciones",
]


def cargar_datos(especificacion: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """especificacion: 'A' (con ingreso/gasto) o 'B' (sin). Train =
    2010->2013, test = 2013->2016 (holdout temporal principal, ver
    `build_benchmark_train_test.py`)."""
    train = pd.read_parquet(DATA_DIR / f"modelo_{especificacion}_2010_2013.parquet")
    test = pd.read_parquet(DATA_DIR / f"modelo_{especificacion}_2013_2016.parquet")
    return train, test


def preparar_arboles_nativos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Para algoritmos con soporte nativo de NaN + categoricas (HistGB,
    XGBoost, LightGBM): sin imputar, columnas de texto a dtype category."""
    y = df["Y"]
    x = df.drop(columns=[c for c in COLS_NO_FEATURE if c in df.columns])
    cols_categoricas = x.select_dtypes(include="object").columns.tolist()
    for c in cols_categoricas:
        x[c] = x[c].astype("category")
    return x, y, cols_categoricas


def alinear_columnas_categoricas(x_train: pd.DataFrame, x_test: pd.DataFrame, cat_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alinea columnas y categorias entre train y test (pueden diferir
    ligeramente en categorias observadas de una ola a otra)."""
    cols_comunes = [c for c in x_train.columns if c in x_test.columns]
    x_train = x_train[cols_comunes].copy()
    x_test = x_test[cols_comunes].copy()
    for c in cat_cols:
        if c not in cols_comunes:
            continue
        categorias = pd.api.types.union_categoricals([x_train[c], x_test[c]]).categories
        x_train[c] = x_train[c].cat.set_categories(categorias)
        x_test[c] = x_test[c].cat.set_categories(categorias)
    return x_train, x_test


def preparar_xy_crudo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X/Y sin transformar -- para pasar a un Pipeline de sklearn
    con imputacion/encoding/escalado incluido (logistica, random forest)."""
    y = df["Y"]
    x = df.drop(columns=[c for c in COLS_NO_FEATURE if c in df.columns])
    return x, y


def construir_preprocesador(x: pd.DataFrame, escalar: bool = True) -> ColumnTransformer:
    """ColumnTransformer para algoritmos SIN soporte nativo de NaN/
    categoricas -- 0 + indicador (numericas), 'Sin dato' + one-hot
    (categoricas). Ver docstring del modulo para la justificacion de 0
    sobre mediana (confirmada con el usuario)."""
    cols_numericas = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    cols_categoricas = x.select_dtypes(include="object").columns.tolist()

    pasos_numerico = [("imputar", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True))]
    if escalar:
        pasos_numerico.append(("escalar", StandardScaler()))
    pipe_numerico = ImbPipeline(pasos_numerico)

    pipe_categorico = ImbPipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Sin dato")),
        ("codificar", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", pipe_numerico, cols_numericas),
        ("cat", pipe_categorico, cols_categoricas),
    ])


def _class_weight_dict(y_train) -> dict:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    n = n_pos + n_neg
    return {0: n / (2 * n_neg), 1: n / (2 * n_pos)}


def comparar_balanceo_y_tunear(
    construir_pipeline_fn,
    param_distributions_fn,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """Para cada estrategia de balanceo en BALANCEOS: construye el pipeline
    correspondiente (via `construir_pipeline_fn(balanceo)`), corre
    RandomizedSearchCV con los hiperparametros de `param_distributions_fn(balanceo)`,
    y se queda con la estrategia+configuracion de mayor AUC-ROC promedio en
    CV. Retorna el mejor estimador ya reentrenado sobre todo x_train/y_train,
    mas el detalle de la comparacion para el registro y la consola.
    """
    resultados = {}
    mejor_balanceo, mejor_score = None, -np.inf
    mejor_estimador, mejor_params = None, {}

    for balanceo in BALANCEOS:
        pipeline = construir_pipeline_fn(balanceo)
        param_dist = param_distributions_fn(balanceo)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        if param_dist:
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_dist, n_iter=N_ITER_BUSQUEDA,
                scoring=SCORING, cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True,
            )
            search.fit(x_train, y_train)
            score, mejores_params, estimador = search.best_score_, search.best_params_, search.best_estimator_
        else:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(pipeline, x_train, y_train, scoring=SCORING, cv=cv, n_jobs=-1)
            score, mejores_params = float(scores.mean()), {}
            estimador = pipeline.fit(x_train, y_train)

        resultados[balanceo] = {"auc_cv": round(float(score), 4), "mejores_params": mejores_params}
        if score > mejor_score:
            mejor_balanceo, mejor_score, mejor_estimador, mejor_params = balanceo, score, estimador, mejores_params

    return {
        "balanceo_elegido": mejor_balanceo,
        "auc_cv_por_balanceo": {b: resultados[b]["auc_cv"] for b in BALANCEOS},
        "mejores_params": mejor_params,
        "estimador": mejor_estimador,
    }


def elegir_umbral_por_cv(estimador, x_train: pd.DataFrame, y_train: pd.Series, semilla: int = RANDOM_STATE) -> float:
    """Probabilidades out-of-fold (cross_val_predict, mismos folds que la
    busqueda de hiperparametros) -- se escanea una grilla de umbrales y se
    elige la que maximiza F1. Evita fijar 0.5 quando la calibracion de las
    probabilidades depende de la estrategia de balanceo elegida (ver
    docstring del modulo)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=semilla)
    proba_oof = cross_val_predict(estimador, x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    mejor_umbral, mejor_f1 = 0.5, -1.0
    for umbral in np.linspace(0.02, 0.98, 97):
        f1 = f1_score(y_train, (proba_oof >= umbral).astype(int), zero_division=0)
        if f1 > mejor_f1:
            mejor_umbral, mejor_f1 = float(umbral), f1
    return mejor_umbral


def evaluar_multiples_semillas(
    construir_pipeline_fn,
    mejores_params: dict,
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    semillas: list = SEMILLAS,
) -> dict:
    """Re-entrena el modelo FINAL (balanceo y mejores_params ya elegidos,
    fijos) con cada semilla en `semillas` -- `construir_pipeline_fn(semilla)`
    debe devolver el pipeline SIN tunear (misma estructura/balanceo que el
    elegido por `comparar_balanceo_y_tunear`, solo cambia el random_state
    interno) -- se le aplican los `mejores_params` via `set_params` antes de
    entrenar. Para cada semilla se re-elige tambien el umbral por CV (mismos
    folds, misma semilla). Retorna el detalle por semilla y un resumen
    (media, std, IC 95% via t de Student) para AUC-ROC/recall/precision/F1
    -- ver docstring del modulo, "Multiples semillas e intervalos de
    confianza"."""
    filas = []
    for semilla in semillas:
        pipe = construir_pipeline_fn(semilla)
        if mejores_params:
            pipe.set_params(**mejores_params)
        pipe.fit(x_train, y_train)
        umbral = elegir_umbral_por_cv(pipe, x_train, y_train, semilla=semilla)
        proba_test = pipe.predict_proba(x_test)[:, 1]
        metricas = calcular_metricas(y_test, proba_test, umbral=umbral)
        filas.append({"semilla": semilla, "umbral": umbral, **metricas})

    detalle = pd.DataFrame(filas)

    n = len(semillas)
    t_mult = float(scipy_stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    resumen = {"umbral_media": round(float(detalle["umbral"].mean()), 4)}
    for metrica in METRICAS_RESUMEN:
        media = float(detalle[metrica].mean())
        std = float(detalle[metrica].std(ddof=1)) if n > 1 else 0.0
        margen = t_mult * std / np.sqrt(n) if n > 1 else 0.0
        resumen[metrica] = {
            "media": round(media, 4), "std": round(std, 4),
            "ci95_low": round(media - margen, 4), "ci95_high": round(media + margen, 4),
        }

    fila_ref = detalle[detalle["semilla"] == RANDOM_STATE].iloc[0]
    resumen["confusion_semilla42"] = {
        "tn": int(fila_ref["tn"]), "fp": int(fila_ref["fp"]),
        "fn": int(fila_ref["fn"]), "tp": int(fila_ref["tp"]),
    }

    return {"detalle": detalle, "resumen": resumen}


def calcular_metricas(y_test, proba_test, umbral: float = 0.5) -> dict:
    pred_test = (proba_test >= umbral).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_test, proba_test),
        "recall": recall_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def registrar_resultado(
    algoritmo: str,
    especificacion: str,
    x_train_shape: tuple,
    x_test_shape: tuple,
    n_covariables_originales: int,
    y_train: pd.Series,
    y_test: pd.Series,
    multi_resultado: dict,
    estrategia_imputacion: str,
    balanceo_info: dict,
    hiperparametros: dict,
    observaciones: str,
) -> None:
    """Upsert de una fila en registro_modelos.csv (clave: algoritmo +
    especificacion) y regeneracion completa de registro_modelos.xlsx.
    `multi_resultado` es el dict retornado por `evaluar_multiples_semillas`
    -- se registran media/std/IC95 por metrica, no un unico valor."""
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    auc_cv = balanceo_info.get("auc_cv_por_balanceo", {})
    resumen = multi_resultado["resumen"]
    conf = resumen["confusion_semilla42"]
    fila = {
        "algoritmo": algoritmo,
        "especificacion": especificacion,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_train": x_train_shape[0],
        "n_test": x_test_shape[0],
        "n_covariables_originales": n_covariables_originales,
        "n_covariables_modelo": x_train_shape[1],
        "tasa_entrada_train": round(float(y_train.mean()), 4),
        "tasa_entrada_test": round(float(y_test.mean()), 4),
        "balanceo_elegido": balanceo_info.get("balanceo_elegido", ""),
        "auc_cv_balanced": auc_cv.get("balanced", ""),
        "auc_cv_ninguno": auc_cv.get("ninguno", ""),
        "auc_cv_oversampling": auc_cv.get("oversampling", ""),
        "n_semillas": len(SEMILLAS),
        "umbral_clasificacion_media": resumen["umbral_media"],
        "tn_semilla42": conf["tn"], "fp_semilla42": conf["fp"],
        "fn_semilla42": conf["fn"], "tp_semilla42": conf["tp"],
        "estrategia_imputacion": estrategia_imputacion,
        "hiperparametros": json.dumps(hiperparametros, ensure_ascii=False, default=str),
        "observaciones": observaciones,
    }
    for metrica in METRICAS_RESUMEN:
        m = resumen[metrica]
        fila[f"{metrica}_media"] = m["media"]
        fila[f"{metrica}_std"] = m["std"]
        fila[f"{metrica}_ci95_low"] = m["ci95_low"]
        fila[f"{metrica}_ci95_high"] = m["ci95_high"]

    if REGISTRO_CSV.exists():
        registro = pd.read_csv(REGISTRO_CSV)
        registro = registro[~((registro["algoritmo"] == algoritmo) & (registro["especificacion"] == especificacion))]
        registro = pd.concat([registro, pd.DataFrame([fila])], ignore_index=True)
    else:
        registro = pd.DataFrame([fila])

    registro = registro[COLUMNAS_REGISTRO].sort_values(["algoritmo", "especificacion"]).reset_index(drop=True)
    registro.to_csv(REGISTRO_CSV, index=False)
    _regenerar_excel(registro)


def _regenerar_excel(registro: pd.DataFrame) -> None:
    with pd.ExcelWriter(REGISTRO_XLSX, engine="openpyxl") as writer:
        registro.to_excel(writer, sheet_name="Registro modelos", index=False)
        ws = writer.sheets["Registro modelos"]
        ws.freeze_panes = "A2"
        for col_cells in ws.columns:
            letra = col_cells[0].column_letter
            largo = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[letra].width = min(max(largo + 2, 10), 60)
        from openpyxl.styles import Font
        for cell in ws[1]:
            cell.font = Font(bold=True)
