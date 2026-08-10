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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

COLUMNAS_REGISTRO = [
    "algoritmo", "especificacion", "fecha_entrenamiento",
    "n_train", "n_test", "n_covariables_originales", "n_covariables_modelo",
    "tasa_entrada_train", "tasa_entrada_test",
    "balanceo_elegido", "auc_cv_balanced", "auc_cv_ninguno", "auc_cv_oversampling",
    "auc_roc", "recall", "precision", "f1",
    "tn", "fp", "fn", "tp",
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
    metricas: dict,
    estrategia_imputacion: str,
    balanceo_info: dict,
    hiperparametros: dict,
    observaciones: str,
) -> None:
    """Upsert de una fila en registro_modelos.csv (clave: algoritmo +
    especificacion) y regeneracion completa de registro_modelos.xlsx."""
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    auc_cv = balanceo_info.get("auc_cv_por_balanceo", {})
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
        "auc_roc": round(float(metricas["auc_roc"]), 4),
        "recall": round(float(metricas["recall"]), 4),
        "precision": round(float(metricas["precision"]), 4),
        "f1": round(float(metricas["f1"]), 4),
        "tn": metricas["tn"], "fp": metricas["fp"], "fn": metricas["fn"], "tp": metricas["tp"],
        "estrategia_imputacion": estrategia_imputacion,
        "hiperparametros": json.dumps(hiperparametros, ensure_ascii=False, default=str),
        "observaciones": observaciones,
    }

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
