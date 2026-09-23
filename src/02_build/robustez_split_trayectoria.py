"""
robustez_split_trayectoria.py
=====================================
Dos chequeos de robustez sobre la separacion del grupo "Entra en pobreza"
(2010->2013) en transitorio y persistente (Seccion 5.2 de `main.tex`,
Cuarto hallazgo; `eda_perfil_split_trayectoria.py`), que hoy solo se
reportan como promedios puntuales sobre 610 de 723 hogares:

  1. ATRICION. 113 de los 723 hogares que entran en pobreza no tienen dato
     de 2016 y por eso quedan fuera de la separacion. Si esos 113 difirieran
     de los 610 en las covariables de la ola base, la separacion (y su
     58%/42%) podria estar sesgada por quien se queda en el panel. Se
     compara, sobre las 53 variables del perfil, a los hogares "con dato
     2016" frente a "sin dato 2016" (diferencia estandarizada, prueba de
     diferencia, correccion por comparaciones multiples).
  2. INCERTIDUMBRE. Con 355 y 255 hogares, las diferencias entre
     transitorio y persistente (por ejemplo, deuda informal 17.3% frente a
     41.9%) necesitan un intervalo y una prueba, no solo el punto. Se
     calcula, para las 53 variables, la diferencia persistente - transitorio
     con su diferencia estandarizada, una prueba de diferencia de medias
     (Welch; equivale a la prueba de dos proporciones para variables 0/1) y
     el valor p ajustado por Benjamini-Hochberg; y para las variables
     ejemplo, intervalos de Wilson del 95% para cada subgrupo.

Nota metodologica: las pruebas y los intervalos son SIN ponderar. Los
promedios del perfil (`perfil_split_transitorio_persistente_monetaria.csv`)
son ponderados por `peso_longitudinal`; se reporta tambien la version
ponderada de las variables ejemplo para mostrar que la diferencia no
depende de ponderar. La inferencia asume hogares independientes (sin
ajuste por conglomerado de comunidad).

INPUTS
    outputs/tables/eda_transicion_covariables/perfil_split_transitorio_persistente_monetaria.csv
        (lista de las 53 variables y su tipo/nivel)
    Fuentes de `eda_perfil_split_trayectoria.py` (panel de categorias y
    covariables de la ola base 2010)

OUTPUTS
    outputs/tables/eda_transicion_covariables/robustez_split_atricion.csv
    outputs/tables/eda_transicion_covariables/robustez_split_transitorio_vs_persistente.csv
    outputs/tables/eda_transicion_covariables/robustez_split_ejemplos_ic.csv

COMO CORRER
    cd src/02_build && python robustez_split_trayectoria.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_features"))
import eda_transicion_covariables as etc  # noqa: E402
from eda_perfil_split_trayectoria import construir_panel_split  # noqa: E402

TABLES_DIR = etc.TABLES_DIR
UMBRAL_SMD = 0.20
ALFA = 0.05
EJEMPLOS = [
    "categoria_ocupacional_jefe", "cotiza_pension_jefe", "deuda_informal_hogar",
    "deuda_formal_hogar", "tiene_transporte_publico_comunidad",
]


def indicador(covariables: pd.DataFrame, fila: pd.Series) -> pd.Series:
    """Serie numerica (0/1 o continua) de una fila del perfil: la variable,
    o el indicador del nivel mostrado si es categorica."""
    col = covariables[fila["variable"]]
    if fila["tipo"] == "Categorica":
        return (col.astype(str) == str(fila["nivel_mostrado"])).astype(float).where(col.notna())
    return pd.to_numeric(col.astype(float), errors="coerce")


def comparar(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna(), b.dropna()
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    smd = (a.mean() - b.mean()) / sd if sd > 0 else np.nan
    p = stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 and sd > 0 else np.nan
    return {"n_a": len(a), "n_b": len(b), "media_a": a.mean(), "media_b": b.mean(), "smd": smd, "p": p}


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """Valores p ajustados (q) de Benjamini-Hochberg; los NaN se dejan NaN."""
    ok = p.dropna().sort_values()
    m = len(ok)
    q = (ok * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def wilson(x: int, n: int, z: float = 1.96) -> tuple:
    p = x / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    mitad = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - mitad, centro + mitad


def main() -> None:
    panel = construir_panel_split()
    dmsp = etc.cargar_dmsp_por_consecutivo(2010)
    covariables = etc.cargar_covariables_ola_base(dmsp, 1)
    perfil = pd.read_csv(TABLES_DIR / "perfil_split_transitorio_persistente_monetaria.csv")

    entra = panel[panel["categoria"].str.startswith("Entra")].set_index("consecutivo")
    trans = entra.index[entra["categoria"] == "Entra (transitorio)"]
    pers = entra.index[entra["categoria"] == "Entra (persistente)"]
    sin_dato = entra.index[entra["categoria"] == "Entra (sin dato 2016)"]
    con_dato = trans.union(pers)
    print(f"Entra: {len(entra)} | con dato 2016: {len(con_dato)} (transitorio {len(trans)}, persistente {len(pers)}) | sin dato: {len(sin_dato)}")

    filas_at, filas_tp = [], []
    for _, fila in perfil.iterrows():
        x = indicador(covariables, fila)
        etiqueta = {"variable": fila["variable"], "nivel": fila["nivel_mostrado"]}
        filas_at.append({**etiqueta, **comparar(x.reindex(con_dato), x.reindex(sin_dato))})
        filas_tp.append({**etiqueta, **comparar(x.reindex(pers), x.reindex(trans))})

    at = pd.DataFrame(filas_at).rename(columns={"n_a": "n_con_dato", "n_b": "n_sin_dato", "media_a": "media_con_dato", "media_b": "media_sin_dato"})
    at["q_bh"] = benjamini_hochberg(at["p"])
    at.to_csv(TABLES_DIR / "robustez_split_atricion.csv", index=False)

    tp = pd.DataFrame(filas_tp).rename(columns={"n_a": "n_persistente", "n_b": "n_transitorio", "media_a": "media_persistente", "media_b": "media_transitorio"})
    tp["q_bh"] = benjamini_hochberg(tp["p"])
    tp.to_csv(TABLES_DIR / "robustez_split_transitorio_vs_persistente.csv", index=False)

    print("\n=== 1. Atricion: con dato 2016 frente a sin dato 2016 (53 variables) ===")
    print(f"  |SMD| > {UMBRAL_SMD}: {int((at['smd'].abs() > UMBRAL_SMD).sum())} de {len(at)}"
          f" | q(BH) < {ALFA}: {int((at['q_bh'] < ALFA).sum())} de {at['q_bh'].notna().sum()}")
    print(at.reindex(at["smd"].abs().sort_values(ascending=False).index).head(8)[
        ["variable", "nivel", "media_con_dato", "media_sin_dato", "smd", "p", "q_bh"]].round(3).to_string(index=False))

    print("\n=== 2. Persistente frente a transitorio (53 variables) ===")
    print(f"  |SMD| > {UMBRAL_SMD}: {int((tp['smd'].abs() > UMBRAL_SMD).sum())} de {len(tp)}"
          f" | q(BH) < {ALFA}: {int((tp['q_bh'] < ALFA).sum())} de {tp['q_bh'].notna().sum()}")
    print(tp.reindex(tp["smd"].abs().sort_values(ascending=False).index).head(10)[
        ["variable", "nivel", "media_persistente", "media_transitorio", "smd", "p", "q_bh"]].round(3).to_string(index=False))

    # Intervalos de Wilson y version ponderada de las variables ejemplo
    peso = panel.set_index("consecutivo")["peso_longitudinal"]
    filas_ic = []
    for v in EJEMPLOS:
        for _, fila in perfil[perfil["variable"] == v].iterrows():
            x = indicador(covariables, fila)
            for nombre, idx in [("transitorio", trans), ("persistente", pers)]:
                s = x.reindex(idx).dropna()
                w = peso.reindex(s.index)
                lo, hi = wilson(int(round(s.sum())), len(s))
                filas_ic.append({
                    "variable": v, "nivel": fila["nivel_mostrado"], "grupo": nombre, "n": len(s),
                    "pct_sin_ponderar": 100 * s.mean(), "ic95_bajo": 100 * lo, "ic95_alto": 100 * hi,
                    "pct_ponderado": 100 * np.average(s, weights=w),
                })
    ic = pd.DataFrame(filas_ic)
    ic.to_csv(TABLES_DIR / "robustez_split_ejemplos_ic.csv", index=False)
    print("\n=== Variables ejemplo: % con IC de Wilson 95% ===")
    print(ic.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
