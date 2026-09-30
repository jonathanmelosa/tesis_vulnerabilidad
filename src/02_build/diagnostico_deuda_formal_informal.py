"""
diagnostico_deuda_formal_informal.py
=====================================
Diagnostico de la construccion de `deuda_formal_hogar` y
`deuda_informal_hogar` (`build_hogar_features.py`), pedido del usuario
(2026-09-30) tras notar que la deuda informal, variable principal del
antiguo Cuarto hallazgo, no esta en el nucleo SHAP. Documentado en
docs/decisions.md, entrada del 2026-09-30. No modifica ningun dato.

Que verifica
    1. Clasificacion de `con_quien_1` (prestamista del PRIMER prestamo) por
       ola: formal, informal, sin clasificar ("otro") o NaN (sin prestamo).
    2. Que los NaN de `deuda_informal_hogar` coinciden con
       `tiene_deuda_hogar == 0` (sin deuda codificado como faltante).
    3. Cuantos hogares tienen un segundo prestamo (`con_quien_2`), que la
       construccion no usa.
    4. Correlacion de Spearman entre las dos variables (con signo) en los
       benchmarks del Modelo B.

INPUTS
    data/processed/hogar_elca_longitudinal_clean.parquet
    data/processed/hogar_features_elca_longitudinal.parquet
    data/processed/benchmark_train_test/modelo_B_{2010_2013,2013_2016}.parquet

OUTPUTS
    Solo consola (diagnostico).

COMO CORRER
    python src/02_build/diagnostico_deuda_formal_informal.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "04_features"))
from build_hogar_features import DEUDA_FORMAL_TOKENS, DEUDA_INFORMAL_TOKENS, normalizar_espacios  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    h = pd.read_parquet(PROCESSED / "hogar_elca_longitudinal_clean.parquet")
    h["ola"] = pd.to_numeric(h["ola"], errors="coerce")
    cq = normalizar_espacios(h["con_quien_1"]).str.lower().where(h["con_quien_1"].notna())
    clase = np.select(
        [cq.isna(), cq.isin(DEUDA_FORMAL_TOKENS), cq.isin(DEUDA_INFORMAL_TOKENS)],
        ["sin prestamo (NaN)", "formal", "informal"], default="sin clasificar",
    )
    print("1. Prestamista del primer prestamo, por ola:")
    print(pd.crosstab(clase, h["ola"]).to_string())
    print("\n   Categorias sin clasificar:")
    print(cq[clase == "sin clasificar"].value_counts().to_string())

    f = pd.read_parquet(PROCESSED / "hogar_features_elca_longitudinal.parquet")
    print("\n2. tiene_deuda_hogar x deuda_informal_hogar (-1 = NaN):")
    print(pd.crosstab(f["tiene_deuda_hogar"], f["deuda_informal_hogar"].fillna(-1)).to_string())

    print("\n3. Hogares con segundo prestamo (con_quien_2 no nulo), por ola:")
    print(h.groupby("ola")["con_quien_2"].apply(lambda s: int(s.notna().sum())).to_string())

    print("\n4. Modelo B: faltantes y Spearman (con signo) entre deuda formal e informal:")
    for ventana in ["2010_2013", "2013_2016"]:
        d = pd.read_parquet(PROCESSED / "benchmark_train_test" / f"modelo_B_{ventana}.parquet")
        rho = d["deuda_formal_hogar"].corr(d["deuda_informal_hogar"], method="spearman")
        faltante = d["deuda_informal_hogar"].isna().mean() * 100
        print(f"   {ventana}: n={len(d)}, faltante={faltante:.1f}%, rho={rho:.3f}")


if __name__ == "__main__":
    main()
