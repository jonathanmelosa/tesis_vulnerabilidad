"""
Desagregaciones de la pobreza monetaria (ELCA 2010, 2013, 2016).

Parte de pobreza_monetaria_elca_longitudinal.parquet (build_pobreza_monetaria.py)
y agrega cuatro tipos de analisis:

1. Indicadores FGT (Foster-Greer-Thorbecke): ademas de la incidencia (P0,
   ya calculada como pobre_ingreso/pobre_gasto), se agregan la BRECHA (P1)
   y la SEVERIDAD (P2) de la pobreza, exactamente como las reporta el DANE
   en sus boletines (ver docs/fuentes_dane/). Formula estandar:

       P_alpha = (1/N) * sum_i [ ((LP - y_i) / LP)^alpha * 1(y_i < LP) ]

   alpha=0 -> incidencia (proporcion de pobres)
   alpha=1 -> brecha (que tan lejos esta el ingreso promedio de los pobres
              respecto a la LP, como fraccion de la LP)
   alpha=2 -> severidad (igual que la brecha, pero pondera mas a los
              hogares mas alejados de la LP -- sensible a la desigualdad
              entre los pobres)

   Se calculan para ingreso Y gasto, contra LP y LI, por ola.

2. Desagregacion geografica: por zona (Urbano/Rural, ya en el pipeline) y
   por las 8 macro-regiones propias de la ELCA (Atlantica, Pacifica,
   Oriental, etc. -- columna `region` de hogar_elca_longitudinal_clean).
   Es el nivel geografico mas fino que los datos permiten (ver
   docs/decisions.md, seccion sobre por que la LP se queda en 2 dominios:
   no hay forma de identificar departamento/municipio real).

3. Desagregacion por caracteristicas del jefe de hogar y del hogar,
   replicando el tipo de tablas que publica el DANE (bol_pobreza_16.pdf,
   Tablas 3-9): sexo, grupo de edad y nivel educativo del jefe de hogar
   (identificado via parentesco="Jefe de hogar"/"Jefe(a)" en
   personas_elca_longitudinal.parquet -- se verifico que hay exactamente
   un jefe por hogar en las 3 olas) y numero de niños menores de 12 años
   en el hogar.

4. Matrices de transicion de pobreza entre olas consecutivas (2010->2013,
   2013->2016), siguiendo la metodologia de Lopez-Calva y Ortiz-Juarez
   (2014), "A Vulnerability Approach to the Definition of the Middle
   Class", Tabla 3: tabla cruzada de estado de pobreza en el periodo
   inicial (filas) contra el periodo final (columnas), en porcentajes de
   fila (cada fila suma 100%). De esa tabla se deriva la clasificacion en
   4 categorias: nunca pobre, siempre pobre, sale de la pobreza (pobre ->
   no pobre) y entra en pobreza (no pobre -> pobre).

   Emparejamiento de hogares entre olas: se usa `consecutivo` (el
   identificador de hogar de 2010, que se mantiene estable en las 3 olas).
   Importante: los hogares que se DIVIDEN entre una
   ola y la siguiente (ver docs/decisions.md) se EXCLUYEN de la matriz de
   transicion -- solo se usan los matches 1 a 1. El numero de hogares
   excluidos por division se reporta aparte.

Todas las tablas de salida son CSV en outputs/tables/pobreza/, no parquet:
son resultados descriptivos para el documento de tesis, no un insumo para
otro script del pipeline. Las graficas correspondientes se guardan como PNG
en outputs/figures/pobreza/ -- ver graficar_resultados() al final del
archivo. Paleta y reglas de color siguen la skill de dataviz del proyecto
(colores categoricos en orden fijo, validados para daltonismo; nunca doble
eje; etiquetas directas en vez de solo leyenda cuando el espacio alcanza).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POBREZA_PATH = PROJECT_ROOT / "data" / "processed" / "pobreza_monetaria_elca_longitudinal.parquet"
HOGAR_PATH = PROJECT_ROOT / "data" / "processed" / "hogar_elca_longitudinal_clean.parquet"
PERSONAS_PATH = PROJECT_ROOT / "data" / "processed" / "personas_elca_longitudinal.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "pobreza"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "pobreza"

# Paleta categorica validada (dataviz skill, references/palette.md), modo claro.
# Orden fijo: nunca se reasigna por categoria, siempre en este orden de slot.
PALETA = {
    "azul": "#2a78d6",
    "naranja": "#eb6834",
    "aguamarina": "#1baf7a",
    "amarillo": "#eda100",
    "magenta": "#e87ba4",
    "verde": "#008300",
    "violeta": "#4a3aa7",
    "rojo": "#e34948",
}
INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECUNDARIO,
    "text.color": INK_PRIMARIO,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

JEFE_TOKENS = {"jefe de hogar", "jefe(a)"}

NIVEL_EDUC_GRUPOS = [
    ("preescolar", "Ninguno/preescolar"),
    ("ninguno", "Ninguno/preescolar"),
    ("primaria", "Primaria"),
    ("secundaria", "Secundaria"),
    ("t.cnico", "Tecnica"),
    ("tecnol.gico", "Tecnologica"),
    ("universitario", "Universitaria"),
    ("posgrado", "Posgrado"),
]


def _llave_compuesta(df: pd.DataFrame) -> pd.Series:
    """consecutivo (ola 1) / llave (ola 2) / llave_n16 (ola 3): 1 fila = 1 sub-hogar."""
    return df["llave_n16"].where(
        df["ola"] == 3, df["llave"].where(df["ola"] == 2, df["consecutivo"])
    )


def _agrupar_nivel_educ(serie: pd.Series) -> pd.Series:
    """Colapsa las variantes de codificacion (acentos rotos '?'/'�') a categorias limpias."""
    texto = serie.astype(str).str.strip().str.lower().str.replace("�", ".", regex=False)
    texto = texto.where(~texto.isin(["nan", "no informa"]), np.nan)
    resultado = pd.Series(np.nan, index=serie.index, dtype=object)
    for patron, etiqueta in NIVEL_EDUC_GRUPOS:
        resultado[texto.str.contains(patron, na=False, regex=True)] = etiqueta
    return resultado


def _grupo_edad(edad: pd.Series) -> pd.Series:
    edad = pd.to_numeric(edad, errors="coerce")
    return pd.cut(
        edad,
        bins=[0, 25, 35, 45, 55, 65, 200],
        labels=["Hasta 25", "26 a 35", "36 a 45", "46 a 55", "56 a 65", "Mayor a 65"],
        right=True,
        include_lowest=True,
    )


def cargar_pobreza_con_covariables() -> pd.DataFrame:
    pobreza = pd.read_parquet(POBREZA_PATH)

    hogar = pd.read_parquet(HOGAR_PATH)[
        ["consecutivo", "ola", "llave", "llave_n16", "region", "t_personas"]
    ]
    llave_pobreza = _llave_compuesta(pobreza)
    llave_hogar = _llave_compuesta(hogar)
    hogar_por_llave = hogar.set_index(llave_hogar)[["region", "t_personas"]]
    pobreza["region"] = llave_pobreza.map(hogar_por_llave["region"])
    pobreza["t_personas"] = llave_pobreza.map(hogar_por_llave["t_personas"])

    personas = pd.read_parquet(PERSONAS_PATH)
    llave_personas = _llave_compuesta(personas)

    es_jefe = personas["parentesco"].astype(str).str.strip().str.lower().isin(JEFE_TOKENS)
    jefes = personas[es_jefe].set_index(llave_personas[es_jefe])
    if jefes.index.duplicated().any():
        raise ValueError("Mas de un jefe de hogar por sub-hogar: revisar supuesto de unicidad.")

    sexo_jefe = jefes["sexo"].astype(str).str.strip().str.title()
    sexo_jefe = sexo_jefe.where(sexo_jefe.isin(["Hombre", "Mujer"]), np.nan)
    pobreza["sexo_jefe"] = llave_pobreza.map(sexo_jefe)
    pobreza["edad_jefe"] = llave_pobreza.map(pd.to_numeric(jefes["edad"], errors="coerce"))
    pobreza["grupo_edad_jefe"] = _grupo_edad(pobreza["edad_jefe"])
    pobreza["nivel_educ_jefe"] = llave_pobreza.map(_agrupar_nivel_educ(jefes["nivel_educ"]))

    edad_personas = pd.to_numeric(personas["edad"], errors="coerce")
    ninos = personas.assign(llave=llave_personas, es_nino=edad_personas < 12)
    t_ninos = ninos.groupby("llave")["es_nino"].sum()
    pobreza["t_ninos_12"] = llave_pobreza.map(t_ninos).fillna(0).astype(int)

    return pobreza


def fgt(bienestar: pd.Series, linea: pd.Series, alpha: int) -> float:
    """P_alpha de Foster-Greer-Thorbecke. bienestar/linea en las mismas unidades (pesos nominales)."""
    valido = bienestar.notna() & linea.notna()
    b, l = bienestar[valido], linea[valido]
    if len(b) == 0:
        return np.nan
    brecha_relativa = ((l - b) / l).clip(lower=0)
    if alpha == 0:
        return (brecha_relativa > 0).mean()
    return (brecha_relativa**alpha).mean()


def tabla_fgt(df: pd.DataFrame, groupby_cols: list) -> pd.DataFrame:
    """P0/P1/P2 para ingreso y gasto, contra LP y LI, por los grupos indicados."""
    combinaciones = [
        ("ingreso", "lp", "ingreso_percapita_hogar"),
        ("ingreso", "li", "ingreso_percapita_hogar"),
        ("gasto", "lp", "gasto_percapita_hogar"),
        ("gasto", "li", "gasto_percapita_hogar"),
    ]
    filas = []
    for claves, grupo in df.groupby(groupby_cols, dropna=False):
        claves = claves if isinstance(claves, tuple) else (claves,)
        for medida, linea_nombre, columna_bienestar in combinaciones:
            fila = dict(zip(groupby_cols, claves))
            fila["medida"] = medida
            fila["linea"] = linea_nombre
            fila["n_hogares"] = grupo[columna_bienestar].notna().sum()
            for alpha in (0, 1, 2):
                fila[f"P{alpha}"] = fgt(grupo[columna_bienestar], grupo[linea_nombre], alpha)
            filas.append(fila)
    return pd.DataFrame(filas)


def construir_matriz_transicion(pobreza: pd.DataFrame, ola_ini: int, ola_fin: int) -> dict:
    """
    Matriz de transicion pobre/no pobre entre dos olas consecutivas, siguiendo
    Lopez-Calva y Ortiz-Juarez (2014), Tabla 3. Solo hogares con match 1 a 1
    por `consecutivo` (decision confirmada: excluir hogares que se dividieron).
    """
    ini = pobreza[pobreza["ola"] == ola_ini][["consecutivo", "pobre_ingreso"]].dropna()
    fin = pobreza[pobreza["ola"] == ola_fin][["consecutivo", "pobre_ingreso"]].dropna()

    ini_unicos = ini[~ini["consecutivo"].duplicated(keep=False)]
    fin_unicos = fin[~fin["consecutivo"].duplicated(keep=False)]
    n_excluidos_division = (
        ini["consecutivo"].nunique() - ini_unicos["consecutivo"].nunique()
        + fin["consecutivo"].nunique() - fin_unicos["consecutivo"].nunique()
    )

    panel = ini_unicos.merge(
        fin_unicos, on="consecutivo", suffixes=(f"_{ola_ini}", f"_{ola_fin}")
    )
    col_ini, col_fin = f"pobre_ingreso_{ola_ini}", f"pobre_ingreso_{ola_fin}"
    panel[col_ini] = panel[col_ini].map({True: "Pobre", False: "No pobre"})
    panel[col_fin] = panel[col_fin].map({True: "Pobre", False: "No pobre"})

    matriz_pct = (
        pd.crosstab(panel[col_ini], panel[col_fin], normalize="index") * 100
    ).round(1)
    matriz_n = pd.crosstab(panel[col_ini], panel[col_fin])

    def clasificar(fila):
        if fila[col_ini] == "No pobre" and fila[col_fin] == "No pobre":
            return "Nunca pobre"
        if fila[col_ini] == "Pobre" and fila[col_fin] == "Pobre":
            return "Siempre pobre"
        if fila[col_ini] == "Pobre" and fila[col_fin] == "No pobre":
            return "Sale de la pobreza"
        return "Entra en pobreza"

    panel["categoria"] = panel.apply(clasificar, axis=1)
    distribucion_categorias = (panel["categoria"].value_counts(normalize=True) * 100).round(1)

    return {
        "ola_inicial": ola_ini,
        "ola_final": ola_fin,
        "n_hogares_panel": len(panel),
        "n_excluidos_por_division": n_excluidos_division,
        "matriz_porcentaje_fila": matriz_pct,
        "matriz_conteo": matriz_n,
        "distribucion_categorias": distribucion_categorias,
    }


ANO_POR_OLA = {1: 2010, 2: 2013, 3: 2016}


def _guardar(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    ruta = FIGURES_DIR / nombre
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"Guardado: {ruta}")


def graf_incidencia_series(tabla_ola: pd.DataFrame) -> None:
    """Incidencia (P0) de pobreza por ingreso vs. gasto, contra la LP, en el tiempo."""
    sub = tabla_ola[tabla_ola["linea"] == "lp"]
    fig, ax = plt.subplots(figsize=(6, 4))
    colores = {"ingreso": PALETA["azul"], "gasto": PALETA["naranja"]}
    for medida, color in colores.items():
        serie = sub[sub["medida"] == medida].sort_values("ola")
        anos = serie["ola"].map(ANO_POR_OLA)
        ax.plot(anos, serie["P0"] * 100, marker="o", markersize=6, linewidth=2, color=color, label=medida.capitalize())
        for x, y in zip(anos, serie["P0"] * 100):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=INK_SECUNDARIO)
    ax.set_ylim(0, max(sub["P0"]) * 100 * 1.25)
    ax.set_xticks(list(ANO_POR_OLA.values()))
    ax.set_ylabel("Incidencia de pobreza (%)")
    ax.set_title("Pobreza monetaria por ingreso vs. gasto (P0, vs. LP)")
    ax.legend(frameon=False, loc="upper right")
    _guardar(fig, "01_incidencia_ingreso_vs_gasto.png")


def graf_fgt_trio(tabla_ola: pd.DataFrame) -> None:
    """P0/P1/P2 del ingreso contra la LP, por ola -- incidencia, brecha y severidad."""
    sub = tabla_ola[(tabla_ola["medida"] == "ingreso") & (tabla_ola["linea"] == "lp")].sort_values("ola")
    anos = sub["ola"].map(ANO_POR_OLA).astype(str)
    indicadores = [("P0", "Incidencia", PALETA["azul"]), ("P1", "Brecha", PALETA["aguamarina"]), ("P2", "Severidad", PALETA["violeta"])]

    x = np.arange(len(anos))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (col, etiqueta, color) in enumerate(indicadores):
        valores = sub[col].to_numpy() * 100
        barras = ax.bar(x + (i - 1) * ancho, valores, width=ancho * 0.9, color=color, label=etiqueta)
        ax.bar_label(barras, fmt="%.1f", padding=2, fontsize=8, color=INK_SECUNDARIO)
    ax.set_xticks(x)
    ax.set_xticklabels(anos)
    ax.set_ylabel("% de la linea de pobreza (LP)")
    ax.set_title("Indicadores FGT de pobreza por ingreso (vs. LP)")
    ax.legend(frameon=False, loc="upper right")
    _guardar(fig, "02_fgt_incidencia_brecha_severidad.png")


def graf_incidencia_zona(tabla_zona: pd.DataFrame) -> None:
    """Incidencia (P0) de pobreza por ingreso, contra la LP, por zona Urbano/Rural."""
    sub = tabla_zona[(tabla_zona["medida"] == "ingreso") & (tabla_zona["linea"] == "lp")]
    fig, ax = plt.subplots(figsize=(6, 4))
    colores = {"Urbano": PALETA["azul"], "Rural": PALETA["naranja"]}
    for zona, color in colores.items():
        serie = sub[sub["zona"] == zona].sort_values("ola")
        anos = serie["ola"].map(ANO_POR_OLA)
        ax.plot(anos, serie["P0"] * 100, marker="o", markersize=6, linewidth=2, color=color, label=zona)
        for x, y in zip(anos, serie["P0"] * 100):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=INK_SECUNDARIO)
    ax.set_ylim(0, max(sub["P0"]) * 100 * 1.25)
    ax.set_xticks(list(ANO_POR_OLA.values()))
    ax.set_ylabel("Incidencia de pobreza (%)")
    ax.set_title("Pobreza monetaria por ingreso, Urbano vs. Rural (P0, vs. LP)")
    ax.legend(frameon=False, loc="upper right")
    _guardar(fig, "03_incidencia_zona.png")


def graf_incidencia_region(tabla_region: pd.DataFrame, ola: int) -> None:
    """Incidencia (P0) de pobreza por ingreso, contra la LP, por region, en una ola dada."""
    sub = tabla_region[
        (tabla_region["medida"] == "ingreso")
        & (tabla_region["linea"] == "lp")
        & (tabla_region["ola"] == ola)
        & tabla_region["region"].notna()
    ].sort_values("P0")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    barras = ax.barh(sub["region"], sub["P0"] * 100, color=PALETA["azul"])
    ax.bar_label(barras, fmt="%.1f%%", padding=4, fontsize=9, color=INK_SECUNDARIO)
    ax.set_xlabel("Incidencia de pobreza (%)")
    ax.set_title(f"Pobreza monetaria por ingreso, por region -- {ANO_POR_OLA[ola]} (P0, vs. LP)")
    ax.set_xlim(0, sub["P0"].max() * 100 * 1.2)
    _guardar(fig, f"04_incidencia_region_{ANO_POR_OLA[ola]}.png")


def graf_incidencia_jefe(tabla_sexo: pd.DataFrame, tabla_edad: pd.DataFrame, tabla_educ: pd.DataFrame, ola: int) -> None:
    """3 paneles: incidencia por sexo, grupo de edad y nivel educativo del jefe de hogar."""
    filtro = lambda t, col: t[  # noqa: E731
        (t["medida"] == "ingreso") & (t["linea"] == "lp") & (t["ola"] == ola) & t[col].notna()
    ]

    orden_educ = ["Ninguno/preescolar", "Primaria", "Secundaria", "Tecnica", "Tecnologica", "Universitaria", "Posgrado"]
    orden_edad = ["Hasta 25", "26 a 35", "36 a 45", "46 a 55", "56 a 65", "Mayor a 65"]

    sexo = filtro(tabla_sexo, "sexo_jefe").set_index("sexo_jefe").reindex(["Mujer", "Hombre"])
    edad = filtro(tabla_edad, "grupo_edad_jefe").set_index("grupo_edad_jefe").reindex(orden_edad)
    educ = filtro(tabla_educ, "nivel_educ_jefe").set_index("nivel_educ_jefe").reindex(orden_educ)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    paneles = [
        (axes[0], sexo, "Sexo del jefe de hogar", False),
        (axes[1], edad, "Edad del jefe de hogar", True),
        (axes[2], educ, "Nivel educativo del jefe de hogar", True),
    ]
    for ax, tabla, titulo, rotar in paneles:
        valores = tabla["P0"] * 100
        barras = ax.bar(tabla.index.astype(str), valores, color=PALETA["azul"])
        ax.bar_label(barras, fmt="%.0f", padding=2, fontsize=8, color=INK_SECUNDARIO)
        ax.set_title(titulo, fontsize=10)
        if rotar:
            ax.tick_params(axis="x", rotation=40)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    axes[0].set_ylabel("Incidencia de pobreza (%)")
    fig.suptitle(f"Pobreza monetaria por ingreso segun jefe de hogar -- {ANO_POR_OLA[ola]} (P0, vs. LP)")
    _guardar(fig, f"05_incidencia_jefe_hogar_{ANO_POR_OLA[ola]}.png")


def graf_transiciones(resumenes: list) -> None:
    """Distribucion de categorias de transicion (nunca/siempre pobre, entra/sale) por periodo."""
    categorias = ["Nunca pobre", "Sale de la pobreza", "Entra en pobreza", "Siempre pobre"]
    colores = [PALETA["azul"], PALETA["aguamarina"], PALETA["naranja"], PALETA["rojo"]]

    periodos = [f"{ANO_POR_OLA[r['ola_inicial']]}-{ANO_POR_OLA[r['ola_final']]}" for r in resumenes]
    matriz = np.array([
        [r["distribucion_categorias"].get(cat, 0.0) for cat in categorias] for r in resumenes
    ])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    base = np.zeros(len(periodos))
    for i, (cat, color) in enumerate(zip(categorias, colores)):
        valores = matriz[:, i]
        ax.bar(periodos, valores, bottom=base, color=color, label=cat, width=0.55)
        for j, v in enumerate(valores):
            if v > 3:
                ax.text(j, base[j] + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=8, color="white")
        base += valores

    ax.set_ylabel("% de hogares (panel emparejado)")
    ax.set_ylim(0, 100)
    ax.set_title("Matriz de transicion de pobreza monetaria entre olas\n(Lopez-Calva y Ortiz-Juarez, 2014)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
    _guardar(fig, "06_transiciones_pobreza.png")


def graficar_resultados(
    tabla_ola: pd.DataFrame,
    tabla_zona: pd.DataFrame,
    tabla_region: pd.DataFrame,
    tabla_sexo: pd.DataFrame,
    tabla_edad: pd.DataFrame,
    tabla_educ: pd.DataFrame,
    resumen_transiciones: list,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    graf_incidencia_series(tabla_ola)
    graf_fgt_trio(tabla_ola)
    graf_incidencia_zona(tabla_zona)
    graf_incidencia_region(tabla_region, ola=3)
    graf_incidencia_jefe(tabla_sexo, tabla_edad, tabla_educ, ola=3)
    graf_transiciones(resumen_transiciones)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pobreza = cargar_pobreza_con_covariables()

    # 1) FGT por ola (linea base) + 2) por zona/region + 3) por jefe de hogar/ninos
    especificaciones = {
        "fgt_por_ola.csv": ["ola"],
        "fgt_por_ola_zona.csv": ["ola", "zona"],
        "fgt_por_ola_region.csv": ["ola", "region"],
        "fgt_por_ola_sexo_jefe.csv": ["ola", "sexo_jefe"],
        "fgt_por_ola_grupo_edad_jefe.csv": ["ola", "grupo_edad_jefe"],
        "fgt_por_ola_nivel_educ_jefe.csv": ["ola", "nivel_educ_jefe"],
        "fgt_por_ola_t_ninos_12.csv": ["ola", "t_ninos_12"],
    }
    tablas = {}
    for nombre_archivo, cols in especificaciones.items():
        tabla = tabla_fgt(pobreza, cols)
        tabla.to_csv(OUTPUT_DIR / nombre_archivo, index=False)
        tablas[nombre_archivo] = tabla
        print(f"Guardado: {OUTPUT_DIR / nombre_archivo} ({len(tabla)} filas)")

    # 4) Matrices de transicion
    resumen_transiciones = []
    for ola_ini, ola_fin in [(1, 2), (2, 3)]:
        resultado = construir_matriz_transicion(pobreza, ola_ini, ola_fin)
        sufijo = f"{ola_ini}_a_{ola_fin}"
        resultado["matriz_porcentaje_fila"].to_csv(OUTPUT_DIR / f"transicion_pct_ola{sufijo}.csv")
        resultado["matriz_conteo"].to_csv(OUTPUT_DIR / f"transicion_conteo_ola{sufijo}.csv")
        resultado["distribucion_categorias"].to_csv(
            OUTPUT_DIR / f"transicion_categorias_ola{sufijo}.csv", header=["porcentaje"]
        )
        print(
            f"\nTransicion ola {ola_ini} -> ola {ola_fin} "
            f"(n={resultado['n_hogares_panel']}, excluidos por division={resultado['n_excluidos_por_division']}):"
        )
        print(resultado["matriz_porcentaje_fila"])
        print(resultado["distribucion_categorias"])
        resumen_transiciones.append(resultado)

    print(f"\nTodas las tablas guardadas en: {OUTPUT_DIR}")

    graficar_resultados(
        tabla_ola=tablas["fgt_por_ola.csv"],
        tabla_zona=tablas["fgt_por_ola_zona.csv"],
        tabla_region=tablas["fgt_por_ola_region.csv"],
        tabla_sexo=tablas["fgt_por_ola_sexo_jefe.csv"],
        tabla_edad=tablas["fgt_por_ola_grupo_edad_jefe.csv"],
        tabla_educ=tablas["fgt_por_ola_nivel_educ_jefe.csv"],
        resumen_transiciones=resumen_transiciones,
    )
    print(f"Todas las graficas guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
