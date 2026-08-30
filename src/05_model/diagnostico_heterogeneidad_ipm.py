"""
Version IPM de `diagnostico_heterogeneidad_dmsp.py` -- mismos 4 ejes de
heterogeneidad, aplicados al hallazgo significativo de
`diagnostico_bootstrap_ipm.py` (XGBoost). Pregunta: dado que el efecto
agregado ya es significativo (a diferencia del caso de pobreza
monetaria), ¿esta concentrado en alguna subpoblacion en particular, o es
parejo?

Usa las predicciones ya generadas por `generar_predicciones_test_ipm.py`
-- no reentrena nada.

OUTPUTS

    data/processed/benchmark_resultados/diagnostico_heterogeneidad_ipm.csv

COMO CORRER

    cd src/05_model && python -u diagnostico_heterogeneidad_ipm.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import modelo_utils as mu

N_MIN_POSITIVOS = 20


def auc_por_grupo(sub: pd.DataFrame) -> dict:
    y = sub["Y"].values
    n = len(sub)
    n_pos = int(y.sum())
    if n_pos < 2 or n_pos > n - 2:
        return {"n": n, "n_pos": n_pos, "auc_base": np.nan, "auc_geo": np.nan, "delta": np.nan, "confiable": False}
    auc_base = roc_auc_score(y, sub["proba_base"].values)
    auc_geo = roc_auc_score(y, sub["proba_geo"].values)
    return {
        "n": n, "n_pos": n_pos, "auc_base": round(auc_base, 4), "auc_geo": round(auc_geo, 4),
        "delta": round(auc_geo - auc_base, 4), "confiable": n_pos >= N_MIN_POSITIVOS,
    }


def construir_ejes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["eje_zona"] = df["zona"]
    df["eje_brecha_lp"] = pd.qcut(df["brecha_lp_ingreso"], 4, labels=["Q1 (mas cerca de LP)", "Q2", "Q3", "Q4 (mas lejos)"])
    df["eje_estrato"] = "Estrato " + df["estrato_verificado_hogar"].astype("Int64").astype(str)

    riqueza_z = (
        (df["n_servicios_publicos_hogar"] - df["n_servicios_publicos_hogar"].mean()) / df["n_servicios_publicos_hogar"].std()
        + (df["n_bienes_durables_hogar"] - df["n_bienes_durables_hogar"].mean()) / df["n_bienes_durables_hogar"].std()
    )
    df["eje_riqueza_proxy"] = pd.qcut(riqueza_z, 3, labels=["Bajo (proxy menos discriminante)", "Medio", "Alto (proxy menos discriminante)"])
    return df


def main() -> None:
    filas = []
    for base, geo in [("Aipm", "AipmgeoDMSP"), ("Bipm", "BipmgeoDMSP")]:
        pred = pd.read_parquet(mu.RESULTADOS_DIR / f"predicciones_test_ipm_{base}.parquet")
        pred = construir_ejes(pred)

        for algoritmo in pred["algoritmo"].unique():
            sub_algo = pred[pred["algoritmo"] == algoritmo]

            print(f"\n=== {algoritmo} -- {base} vs {geo} ===")
            general = auc_por_grupo(sub_algo)
            print(f"  [Global] n={general['n']} n_pos={general['n_pos']} AUC {base}={general['auc_base']} AUC {geo}={general['auc_geo']} delta={general['delta']}")
            filas.append({"algoritmo": algoritmo, "especificacion_base": base, "eje": "Global", "grupo": "Global", **general})

            for eje in ["eje_zona", "eje_brecha_lp", "eje_estrato", "eje_riqueza_proxy"]:
                for grupo, sub_grupo in sub_algo.groupby(eje, observed=True):
                    r = auc_por_grupo(sub_grupo)
                    marca = "" if r["confiable"] else "  [n_pos<20, POCO CONFIABLE]"
                    print(f"  [{eje}={grupo}] n={r['n']} n_pos={r['n_pos']} AUC {base}={r['auc_base']} AUC {geo}={r['auc_geo']} delta={r['delta']}{marca}")
                    filas.append({"algoritmo": algoritmo, "especificacion_base": base, "eje": eje, "grupo": str(grupo), **r})

    out = pd.DataFrame(filas)
    out_path = mu.RESULTADOS_DIR / "diagnostico_heterogeneidad_ipm.csv"
    out.to_csv(out_path, index=False)

    print("\n\n=== Candidatos con mayor |delta| entre celdas confiables (n_pos>=20) ===")
    confiables = out[(out["confiable"]) & (out["eje"] != "Global")].copy()
    confiables["abs_delta"] = confiables["delta"].abs()
    print(confiables.sort_values("abs_delta", ascending=False).head(15)[["algoritmo", "especificacion_base", "eje", "grupo", "n", "n_pos", "delta"]].to_string(index=False))

    print(f"\nCSV: {out_path}")


if __name__ == "__main__":
    main()
