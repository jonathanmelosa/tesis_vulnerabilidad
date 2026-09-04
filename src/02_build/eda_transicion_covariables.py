"""
Caracterizacion de los 4 grupos de transicion de pobreza (2010->2013),
pobreza monetaria e IPM, para la seccion de la tesis sobre diferencias en
covariables entre poblaciones de estudio.

Grupos (Lopez-Calva y Ortiz-Juarez 2014, Tabla 3, ya implementados en
`construir_matriz_transicion` de build_pobreza_desagregaciones.py):
Siempre pobre / Sale de la pobreza / Entra en pobreza / Nunca pobre.

ETAPA 1 de este script (la unica implementada por ahora, pendiente de
validacion con el usuario antes de construir el ranking de covariables y
el mapa): agregacion de los 4 grupos por `region` (unica variable
geografica valida en ELCA -- `id_dpto`/`id_mpio` son "identificador falso"
segun el diccionario oficial de la encuesta, ver docs/decisions.md linea
739 y elca_2010_unido.pdf HR4/HR5) y perfil de iluminacion nocturna
(dmsp_stable_lights, ola 2010) por grupo.

Ponderacion: `peso_longitudinal` (fexhog_2010) para AMBAS definiciones de
pobreza, calculado aqui para IPM reusando `cargar_pesos_muestrales`
(extraida de build_pobreza_desagregaciones.py, misma logica que ya usa
pobreza monetaria) -- decision del usuario 2026-09-04 para que monetaria e
IPM sean comparables en la seccion de comparacion. Los MODELOS de ML no
usan estos pesos (confirmado: no hay peso/fexhog/sample_weight en
src/05_model/); son exclusivos de este analisis descriptivo.

INPUTS
    data/processed/pobreza_monetaria_elca_longitudinal.parquet
    data/processed/ipm_multidimensional_elca_longitudinal.parquet
    data/processed/hogar_elca_longitudinal_clean.parquet (region, pesos)
    data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet (DMSP ola 2010)

OUTPUTS
    outputs/tables/eda_transicion_covariables/region_x_categoria_{monetaria,ipm}.csv
    outputs/tables/eda_transicion_covariables/dmsp_por_categoria_{monetaria,ipm}.csv

COMO CORRER
    cd src/02_build && python eda_transicion_covariables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
from build_pobreza_desagregaciones import (  # noqa: E402
    HOGAR_PATH,
    _llave_compuesta,
    cargar_pesos_muestrales,
    construir_matriz_transicion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POBREZA_PATH = PROJECT_ROOT / "data" / "processed" / "pobreza_monetaria_elca_longitudinal.parquet"
IPM_PATH = PROJECT_ROOT / "data" / "processed" / "ipm_multidimensional_elca_longitudinal.parquet"
GEO_PATH = PROJECT_ROOT / "data" / "processed" / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"
CONSOLIDADO_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_consolidado_elca_longitudinal.parquet"
INVENTARIO_PATH = PROJECT_ROOT / "outputs" / "tables" / "eda_variables_modelo" / "01_inventario_variables.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"

CATEGORIAS_ORDEN = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]

# Variable insignia de la tesis -- entra SIEMPRE a la seleccion final, sin
# importar su posicion en el ranking estadistico (decision del usuario).
VARIABLE_OBLIGATORIA = "dmsp_stable_lights"

COLS_ID = {"consecutivo", "consecutivo_c", "ola", "llave", "llave_n16", "llave_compuesta"}
# Columnas que son (casi) el propio label de pobreza monetaria en la ola
# base (ver build_benchmark_train_test.py:COLS_EXCLUIR_SIEMPRE): excluidas
# SOLO del ranking de MONETARIA por circularidad, no del de IPM (ahi son
# covariables legitimas, no el label -- el label de IPM, `pobre_ipm`/
# `ipm_score`/`priv_*`, ni siquiera vive en el consolidado que usamos como
# universo de covariables).
COLS_LABEL_MONETARIA = {
    "pobre_ingreso", "pobre_extremo_ingreso", "pobre_gasto", "pobre_extremo_gasto",
    "concuerdan_ingreso_gasto", "lp", "li",
}

UMBRAL_ETA2 = 0.01  # "efecto pequeno" (Cohen) para variables Numericas (eta2 sobre rangos)
UMBRAL_V = 0.10  # analogo para Categorica/Booleana (V de Cramer sesgo-corregida)
UMBRAL_CI_ETA2 = 0.005  # piso de "no ruido" para el limite inferior del bootstrap
UMBRAL_CI_V = 0.05
N_BOOT = 300
UMBRAL_REDUNDANCIA = 0.70  # |rho| de Spearman para colapsar variables casi-colineales
N_VARIABLES_OBJETIVO = 11  # ademas de DMSP (10-12, decision del usuario) -> 12 en tabla final


def cargar_peso_longitudinal_por_consecutivo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Equivalente a `cargar_pesos_muestrales` pero para dataframes SIN
    `llave`/`llave_n16` (caso de `ipm_multidimensional_elca_longitudinal.parquet`
    -- a diferencia de pobreza monetaria, IPM no trae el desglose de
    sub-hogar por division). Se une por (consecutivo, ola) directamente en
    vez de por llave compuesta: es seguro porque `construir_matriz_transicion`
    excluye los hogares con `consecutivo` duplicado (division) ANTES de leer
    `peso_col`, asi que una fila duplicada temporal con el mismo peso no
    afecta el resultado.
    """
    hogar = pd.read_parquet(HOGAR_PATH, columns=["consecutivo", "ola", "fexhog_2010"])
    hogar_ola_fin = hogar[hogar["ola"] != 1].drop_duplicates(subset=["consecutivo", "ola"])
    df["peso_longitudinal"] = df.merge(
        hogar_ola_fin, on=["consecutivo", "ola"], how="left", validate="many_to_one"
    )["fexhog_2010"].to_numpy()
    return df


def cargar_region_por_consecutivo() -> pd.Series:
    """`region` de ola 1 (2010), indexada por `consecutivo` -- unica variable
    geografica valida en ELCA (ver docstring del modulo)."""
    hogar = pd.read_parquet(HOGAR_PATH, columns=["consecutivo", "ola", "region"])
    hogar_ola1 = hogar[hogar["ola"] == 1]
    return hogar_ola1.set_index("consecutivo")["region"]


def cargar_dmsp_por_consecutivo() -> pd.Series:
    """dmsp_stable_lights de ola 2010, indexada por `consecutivo` (100% de
    cobertura en esta ola, ver eda_variables_modelo.py)."""
    geo = pd.read_parquet(GEO_PATH, columns=["consecutivo", "ola", "dmsp_stable_lights"])
    geo_2010 = geo[geo["ola"] == 2010]
    assert not geo_2010["consecutivo"].duplicated().any(), "consecutivo debe ser unico en ola 2010"
    return geo_2010.set_index("consecutivo")["dmsp_stable_lights"]


def perfilar_transicion(panel_categorias: pd.DataFrame, region: pd.Series, dmsp: pd.Series) -> dict:
    """A partir de `panel_categorias` (consecutivo, categoria, peso_longitudinal),
    construye la agregacion por region y el perfil de DMSP por grupo."""
    df = panel_categorias.copy()
    df["region"] = df["consecutivo"].map(region)
    df["dmsp_stable_lights"] = df["consecutivo"].map(dmsp)
    df["categoria"] = pd.Categorical(df["categoria"], categories=CATEGORIAS_ORDEN, ordered=True)

    n_sin_peso = df["peso_longitudinal"].isna().sum()
    n_sin_region = df["region"].isna().sum()
    n_sin_dmsp = df["dmsp_stable_lights"].isna().sum()

    # Distribucion de categoria dentro de cada region (% de fila, ponderado
    # por peso_longitudinal) -- para identificar patrones espaciales.
    pivote_peso = df.pivot_table(
        index="region", columns="categoria", values="peso_longitudinal", aggfunc="sum",
        fill_value=0, observed=False,
    )
    region_x_categoria_pct = (pivote_peso.div(pivote_peso.sum(axis=1), axis=0) * 100).round(1)
    region_x_categoria_n = pd.crosstab(df["region"], df["categoria"])

    # DMSP promedio (ponderado) y mediana (sin ponderar, robusta) por grupo,
    # y por region x grupo para ver si el patron espacial se sostiene.
    def _media_ponderada(g):
        peso = g["peso_longitudinal"]
        valor = g["dmsp_stable_lights"]
        mask = valor.notna() & peso.notna()
        if not mask.any():
            return float("nan")
        return (valor[mask] * peso[mask]).sum() / peso[mask].sum()

    dmsp_por_categoria = df.groupby("categoria", observed=True).apply(
        lambda g: pd.Series(
            {
                "dmsp_media_ponderada": _media_ponderada(g),
                "dmsp_mediana": g["dmsp_stable_lights"].median(),
                "n_hogares": len(g),
            }
        ),
        include_groups=False,
    )
    dmsp_region_x_categoria = df.pivot_table(
        index="region", columns="categoria", values="dmsp_stable_lights", aggfunc="median",
        observed=False,
    ).round(1)

    return {
        "region_x_categoria_pct": region_x_categoria_pct,
        "region_x_categoria_n": region_x_categoria_n,
        "dmsp_por_categoria": dmsp_por_categoria,
        "dmsp_region_x_categoria": dmsp_region_x_categoria,
        "n_sin_peso": n_sin_peso,
        "n_sin_region": n_sin_region,
        "n_sin_dmsp": n_sin_dmsp,
        "n_total": len(df),
    }


def cargar_tipos_variables() -> dict:
    """variable -> tipo (Numerica/Categorica/Booleana), del inventario ya
    construido por eda_variables_modelo.py sobre este mismo consolidado."""
    inv = pd.read_csv(INVENTARIO_PATH)
    tipos = dict(zip(inv["variable"], inv["tipo"]))
    tipos["dmsp_stable_lights"] = "Numerica"
    tipos.setdefault("zona", "Categorica")
    return tipos


def cargar_covariables_ola1(dmsp: pd.Series) -> pd.DataFrame:
    """Universo de covariables candidatas: consolidado ML (ola 2010) + DMSP
    (no vive en el consolidado, se agrega aparte). Indexado por consecutivo,
    1 fila = 1 hogar -- mismo insumo (antes del filtro no-pobre-en-base) que
    usan los modelos de ML, para que el ranking hable de las MISMAS
    covariables que ya se evaluan en el modelado."""
    consolidado = pd.read_parquet(CONSOLIDADO_PATH)
    base = consolidado[consolidado["ola"] == 1].set_index("consecutivo")
    base = base.drop(columns=[c for c in COLS_ID if c in base.columns], errors="ignore")
    base["dmsp_stable_lights"] = dmsp
    return base


def _kish_ess(pesos: np.ndarray) -> float:
    """Tamano de muestra efectivo de Kish: penaliza la varianza de los
    pesos (un peso muy desigual reduce la precision real por debajo del N
    crudo) -- se usa en vez de N para el sesgo de V de Cramer y como base
    del remuestreo bootstrap."""
    return float(pesos.sum() ** 2 / np.sum(pesos ** 2))


def weighted_rank_eta2(valores: pd.Series, grupo: pd.Series, peso: pd.Series) -> tuple:
    """eta^2 sobre rangos (ANOVA de rangos ponderado por `peso`): analogo
    ponderado y robusto a colas pesadas/outliers del eta^2 de Kruskal-Wallis.
    Acotado en [0,1], no crece con N (solo con la separacion real entre
    grupos) -- ver metodologia acordada con el usuario."""
    df = pd.DataFrame({"v": valores, "g": grupo, "w": peso}).dropna()
    if df["g"].nunique() < 2 or len(df) < 10:
        return float("nan"), 0.0
    df["rango"] = df["v"].rank(method="average")
    n_eff = _kish_ess(df["w"].to_numpy())
    media_global = np.average(df["rango"], weights=df["w"])
    ss_total = np.sum(df["w"] * (df["rango"] - media_global) ** 2)
    if ss_total == 0:
        return 0.0, n_eff
    ss_between = 0.0
    for _, sub in df.groupby("g", observed=True):
        media_g = np.average(sub["rango"], weights=sub["w"])
        ss_between += sub["w"].sum() * (media_g - media_global) ** 2
    return float(ss_between / ss_total), n_eff


def weighted_cramers_v(valores: pd.Series, grupo: pd.Series, peso: pd.Series) -> tuple:
    """V de Cramer con correccion de sesgo (Bergsma 2013), usando el N
    efectivo de Kish en vez del N crudo -- evita que el ponderado infle
    artificialmente la "certeza" del estadistico."""
    df = pd.DataFrame({"v": valores, "g": grupo, "w": peso}).dropna()
    if df["v"].nunique() < 2 or df["g"].nunique() < 2 or len(df) < 10:
        return float("nan"), 0.0
    tabla = df.pivot_table(index="v", columns="g", values="w", aggfunc="sum", fill_value=0.0, observed=False)
    n_eff = _kish_ess(df["w"].to_numpy())
    total = tabla.to_numpy().sum()
    fila = tabla.sum(axis=1).to_numpy()
    col = tabla.sum(axis=0).to_numpy()
    esperado = np.outer(fila, col) / total
    chi2 = np.nansum(np.where(esperado > 0, (tabla.to_numpy() - esperado) ** 2 / esperado, 0.0))
    r, c = tabla.shape
    if n_eff <= 1:
        return 0.0, n_eff
    phi2_corr = max(0.0, chi2 / n_eff - (r - 1) * (c - 1) / (n_eff - 1))
    r_corr = r - (r - 1) ** 2 / (n_eff - 1)
    c_corr = c - (c - 1) ** 2 / (n_eff - 1)
    denom = min(r_corr - 1, c_corr - 1)
    if denom <= 0:
        return 0.0, n_eff
    return float(np.sqrt(phi2_corr / denom)), n_eff


def bootstrap_ci(valores: pd.Series, grupo: pd.Series, peso: pd.Series, tipo: str, seed: int = 0) -> tuple:
    """Bootstrap ponderado: remuestrea hogares con probabilidad proporcional
    al peso (asi el remuestreo ya representa a la poblacion ponderada) y
    recalcula el efecto SIN ponderar dentro de cada replica -- IC percentil
    95%. Solo se llama sobre el subconjunto que ya paso el umbral de efecto,
    para mantener el costo computacional acotado."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": valores, "g": grupo, "w": peso}).dropna()
    n = len(df)
    probs = (df["w"] / df["w"].sum()).to_numpy()
    unos = pd.Series(np.ones(n))
    replicas = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True, p=probs)
        muestra = df.iloc[idx].reset_index(drop=True)
        if tipo == "Numerica":
            efecto, _ = weighted_rank_eta2(muestra["v"], muestra["g"], unos)
        else:
            efecto, _ = weighted_cramers_v(muestra["v"], muestra["g"], unos)
        replicas.append(efecto)
    return float(np.nanpercentile(replicas, 2.5)), float(np.nanpercentile(replicas, 97.5))


def rankear_covariables(covariables: pd.DataFrame, tipos: dict, panel: pd.DataFrame, excluir: set) -> pd.DataFrame:
    """Ranking de covariables por tamano de efecto (eta2 de rangos para
    Numerica, V de Cramer sesgo-corregida para Categorica/Booleana), ambos
    ponderados por `peso_longitudinal` y acotados en [0,1] -- comparables
    entre si pese a la diferencia de escala/tipo. Devuelve 1 fila por
    variable candidata que paso el filtro minimo de datos."""
    df = panel.merge(covariables, left_on="consecutivo", right_index=True, how="left")
    grupo, peso = df["categoria"], df["peso_longitudinal"]
    filas = []
    for var in covariables.columns:
        if var in excluir:
            continue
        tipo = tipos.get(var, "Numerica")
        valores = df[var]
        if tipo == "Numerica":
            if not pd.api.types.is_numeric_dtype(valores):
                continue
            efecto, n_eff = weighted_rank_eta2(valores, grupo, peso)
        else:
            efecto, n_eff = weighted_cramers_v(valores.astype(object), grupo, peso)
        if pd.isna(efecto):
            continue
        filas.append({"variable": var, "tipo": tipo, "efecto": efecto, "n_efectivo": n_eff})

    ranking = pd.DataFrame(filas).sort_values("efecto", ascending=False).reset_index(drop=True)
    ranking["umbral"] = np.where(ranking["tipo"] == "Numerica", UMBRAL_ETA2, UMBRAL_V)
    ranking["pasa_umbral"] = ranking["efecto"] >= ranking["umbral"]

    ci_low, ci_high = [], []
    for _, fila in ranking.iterrows():
        if not fila["pasa_umbral"]:
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            continue
        lo, hi = bootstrap_ci(df[fila["variable"]], grupo, peso, fila["tipo"])
        ci_low.append(lo)
        ci_high.append(hi)
    ranking["ci_low"], ranking["ci_high"] = ci_low, ci_high
    umbral_ci = np.where(ranking["tipo"] == "Numerica", UMBRAL_CI_ETA2, UMBRAL_CI_V)
    ranking["robusto"] = ranking["pasa_umbral"] & (ranking["ci_low"] >= umbral_ci)
    return ranking


def deduplicar_por_correlacion(ranking_robusto: pd.DataFrame, covariables: pd.DataFrame) -> pd.DataFrame:
    """Colapsa clusters de correlacion de Spearman (|rho| > UMBRAL_REDUNDANCIA)
    entre variables Numericas Y Booleanas (estas ultimas recodificadas 0/1 --
    Spearman sobre binarias es el coeficiente phi, perfectamente valido, y es
    necesario: `pobre_ingreso`/`pobre_extremo_ingreso`/`pobre_gasto`/
    `pobre_extremo_gasto` son booleanas casi-redundantes entre si) y se queda
    con la de mayor efecto por cluster -- evita mostrar varias variantes de
    la misma señal. Categorica multinivel pasa sin cambios (redundancia
    entre categoricas de >2 niveles queda para revision manual, limitacion
    conocida)."""
    candidatas = ranking_robusto[ranking_robusto["tipo"].isin(["Numerica", "Booleana"])]["variable"].tolist()
    categoricas = ranking_robusto[~ranking_robusto["tipo"].isin(["Numerica", "Booleana"])]
    if len(candidatas) < 2:
        return ranking_robusto

    valores = covariables[candidatas].apply(
        lambda s: s.astype(float) if s.dtype != object else pd.to_numeric(s, errors="coerce")
    )
    corr = valores.corr(method="spearman").fillna(0.0)
    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)
    condensada = squareform(dist.values, checks=False)
    clusters = fcluster(linkage(condensada, method="average"), t=1 - UMBRAL_REDUNDANCIA, criterion="distance")

    ranking_cand = ranking_robusto[ranking_robusto["tipo"].isin(["Numerica", "Booleana"])].copy()
    ranking_cand["cluster"] = clusters
    representantes = ranking_cand.sort_values("efecto", ascending=False).drop_duplicates(subset="cluster")
    return pd.concat([representantes.drop(columns="cluster"), categoricas]).sort_values(
        "efecto", ascending=False
    )


def seleccionar_variables_finales(ranking: pd.DataFrame, covariables: pd.DataFrame, tipos_modulo: dict) -> list:
    """De las variables robustas y no-redundantes, arma la seleccion final:
    fuerza DMSP, y completa hasta N_VARIABLES_OBJETIVO priorizando efecto
    pero evitando repetir un mismo `modulo` mas de 2 veces (diversidad
    tematica, ver metodologia acordada)."""
    robustas = ranking[ranking["robusto"]].copy()
    sin_redundancia = deduplicar_por_correlacion(robustas, covariables)

    seleccion = [VARIABLE_OBLIGATORIA]
    conteo_modulo = {}
    for _, fila in sin_redundancia.sort_values("efecto", ascending=False).iterrows():
        var = fila["variable"]
        if var == VARIABLE_OBLIGATORIA or var in seleccion:
            continue
        modulo = tipos_modulo.get(var, "Otro")
        if conteo_modulo.get(modulo, 0) >= 2 and len(seleccion) < N_VARIABLES_OBJETIVO:
            continue  # da preferencia a otros modulos mientras haya cupo
        seleccion.append(var)
        conteo_modulo[modulo] = conteo_modulo.get(modulo, 0) + 1
        if len(seleccion) >= N_VARIABLES_OBJETIVO + 1:  # +1 por DMSP
            break
    return seleccion


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    region = cargar_region_por_consecutivo()
    dmsp = cargar_dmsp_por_consecutivo()
    covariables = cargar_covariables_ola1(dmsp)
    tipos = cargar_tipos_variables()
    inv = pd.read_csv(INVENTARIO_PATH)
    tipos_modulo = dict(zip(inv["variable"], inv["modulo"]))
    tipos_modulo["dmsp_stable_lights"] = "Geoespacial"
    tipos_modulo.setdefault("zona", "Vivienda")

    pobreza = pd.read_parquet(POBREZA_PATH)
    llave_pobreza = _llave_compuesta(pobreza)
    cargar_pesos_muestrales(pobreza, llave_pobreza)
    resultado_monetaria = construir_matriz_transicion(
        pobreza, 1, 2, col_pobre="pobre_ingreso", peso_col="peso_longitudinal"
    )

    ipm = pd.read_parquet(IPM_PATH)
    cargar_peso_longitudinal_por_consecutivo(ipm)
    resultado_ipm = construir_matriz_transicion(
        ipm, 1, 2, col_pobre="pobre_ipm", peso_col="peso_longitudinal"
    )

    for nombre, resultado in [("monetaria", resultado_monetaria), ("ipm", resultado_ipm)]:
        perfil = perfilar_transicion(resultado["panel_categorias"], region, dmsp)
        print(f"\n=== {nombre.upper()} (n panel={perfil['n_total']}) ===")
        print(f"Sin peso_longitudinal: {perfil['n_sin_peso']} | sin region: {perfil['n_sin_region']} | sin DMSP: {perfil['n_sin_dmsp']}")
        print("\nDistribucion de categoria por region (% fila, ponderado):")
        print(perfil["region_x_categoria_pct"].reindex(columns=CATEGORIAS_ORDEN))
        print("\nN hogares por region x categoria (sin ponderar):")
        print(perfil["region_x_categoria_n"].reindex(columns=CATEGORIAS_ORDEN))
        print("\nDMSP por categoria (media ponderada / mediana / n):")
        print(perfil["dmsp_por_categoria"].reindex(CATEGORIAS_ORDEN))
        print("\nDMSP mediana por region x categoria:")
        print(perfil["dmsp_region_x_categoria"].reindex(columns=CATEGORIAS_ORDEN))

        perfil["region_x_categoria_pct"].reindex(columns=CATEGORIAS_ORDEN).to_csv(
            TABLES_DIR / f"region_x_categoria_{nombre}.csv"
        )
        perfil["dmsp_por_categoria"].reindex(CATEGORIAS_ORDEN).to_csv(
            TABLES_DIR / f"dmsp_por_categoria_{nombre}.csv"
        )

        excluir = COLS_LABEL_MONETARIA if nombre == "monetaria" else set()
        panel = resultado["panel_categorias"]
        ranking = rankear_covariables(covariables, tipos, panel, excluir)
        seleccion = seleccionar_variables_finales(ranking, covariables, tipos_modulo)

        print(f"\n--- Ranking de covariables ({nombre}), top 20 ---")
        with pd.option_context("display.max_rows", 20, "display.width", 120):
            print(ranking.head(20).to_string(index=False))
        print(f"\nCandidatas robustas (pasan umbral y CI): {ranking['robusto'].sum()} de {len(ranking)}")
        print(f"\nSeleccion final ({len(seleccion)} variables, incluye {VARIABLE_OBLIGATORIA}):")
        print(seleccion)

        ranking.to_csv(TABLES_DIR / f"ranking_covariables_{nombre}.csv", index=False)
        pd.Series(seleccion, name="variable").to_csv(
            TABLES_DIR / f"seleccion_final_{nombre}.csv", index=False
        )

    print(f"\nGuardado en: {TABLES_DIR}")


if __name__ == "__main__":
    main()
