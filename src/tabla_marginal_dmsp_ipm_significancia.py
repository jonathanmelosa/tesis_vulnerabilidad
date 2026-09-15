"""
tabla_marginal_dmsp_ipm_significancia.py
=====================================
Tabla de contribucion marginal de DMSP-OLS bajo pobreza MULTIDIMENSIONAL
(IPM) -- analoga a `tabla_marginal_dmsp_fbeta2_cv10.py` (monetaria), pero
con el foco puesto en la SIGNIFICANCIA del cambio de AUC-ROC al agregar
DMSP-OLS (columna Delta + IC95% + p-valor del bootstrap pareado), no solo
en el valor de AUC de cada especificacion por separado -- por eso el
formato es una fila por PAR (base vs. +DMSP-OLS), no una fila por
especificacion como en la version monetaria.

Por que esta tabla existe: a diferencia de monetaria (efecto agregado nulo
en las 10 combinaciones, bootstrap sin cluster y con cluster), bajo IPM el
bootstrap de `diagnostico_bootstrap_ipm.py`/`diagnostico_bootstrap_cluster_
ipm.py` encuentra un efecto ESTADISTICAMENTE SIGNIFICATIVO (IC95% no cruza
cero) en 3 de 10 combinaciones (XGBoost-A, XGBoost-B, LightGBM-B) -- un
resultado que hasta ahora no tenia tabla propia en el documento.

INPUTS

    data/processed/benchmark_resultados/registro_modelos_ipm.csv
    data/processed/benchmark_resultados/diagnostico_bootstrap_ipm.csv
    data/processed/benchmark_resultados/diagnostico_bootstrap_cluster_ipm.csv
        (cobertura parcial: solo XGBoost, HistGradientBoosting, Logistica
        -- Random Forest y LightGBM no tienen bootstrap cluster-robusto
        corrido; se marca "--" en esas filas, no se omite la fila)

OUTPUTS

    paper/tables/tab_marginal_dmsp_ipm_significancia.tex

COMO CORRER

    python src/tabla_marginal_dmsp_ipm_significancia.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = REPO_ROOT / "data" / "processed" / "benchmark_resultados"

CONFIG = {
    "registro": RESULTADOS_DIR / "registro_modelos_ipm.csv",
    "bootstrap": RESULTADOS_DIR / "diagnostico_bootstrap_ipm.csv",
    "bootstrap_cluster": RESULTADOS_DIR / "diagnostico_bootstrap_cluster_ipm.csv",
    "algoritmos_orden": [
        "XGBoost",
        "Random Forest",
        "LightGBM",
        "HistGradientBoosting (sklearn)",
        "Logistica regularizada (elastic net, benchmark)",
    ],
    "nombres_algoritmo": {
        "XGBoost": "XGBoost",
        "Random Forest": "Random Forest",
        "LightGBM": "LightGBM",
        "HistGradientBoosting (sklearn)": "HistGradientBoosting",
        "Logistica regularizada (elastic net, benchmark)": "Logística regularizada",
    },
    # nombres cortos usados en diagnostico_bootstrap*_ipm.csv (columna "algoritmo")
    "nombres_bootstrap": {
        "XGBoost": "XGBoost",
        "Random Forest": "Random Forest",
        "LightGBM": "LightGBM",
        "HistGradientBoosting (sklearn)": "HistGradientBoosting",
        "Logistica regularizada (elastic net, benchmark)": "Logistica",
    },
    "pares_especificacion": [("Aipm", "AipmgeoDMSP"), ("Bipm", "BipmgeoDMSP")],
    "etiqueta_especificacion": {"Aipm": "A", "Bipm": "B"},
    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar(cfg: dict) -> tuple:
    faltantes = [p for p in (cfg["registro"], cfg["bootstrap"], cfg["bootstrap_cluster"]) if not p.exists()]
    if faltantes:
        print(f"ERROR: no se encontraron {faltantes}", file=sys.stderr)
        sys.exit(1)
    return (
        pd.read_csv(cfg["registro"]),
        pd.read_csv(cfg["bootstrap"]),
        pd.read_csv(cfg["bootstrap_cluster"]),
    )


def fila_tex(registro: pd.DataFrame, boot: pd.DataFrame, boot_cl: pd.DataFrame,
             algoritmo_raw: str, espec_base: str, espec_geo: str, cfg: dict) -> str:
    nombre = cfg["nombres_algoritmo"][algoritmo_raw]
    nombre_boot = cfg["nombres_bootstrap"][algoritmo_raw]
    etiqueta = cfg["etiqueta_especificacion"][espec_base]

    r_base = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec_base)].iloc[0]
    r_geo = registro[(registro.algoritmo == algoritmo_raw) & (registro.especificacion == espec_geo)].iloc[0]

    b = boot[(boot.algoritmo == nombre_boot) & (boot.especificacion_base == espec_base)]
    if b.empty:
        print(f"ERROR: sin fila de bootstrap para {nombre_boot}/{espec_base}", file=sys.stderr)
        sys.exit(1)
    b = b.iloc[0]

    bc = boot_cl[(boot_cl.algoritmo == nombre_boot) & (boot_cl.especificacion_base == espec_base)]
    p_cluster = f"{bc.iloc[0]['p_valor_cluster']:.3f}" if not bc.empty else "--"

    delta_top10 = r_geo["precision_top10_media"] - r_base["precision_top10_media"]

    negrita = not bool(b["cruza_cero"])
    fmt_delta = f"\\textbf{{{b['delta_auc']:+.4f}}}" if negrita else f"{b['delta_auc']:+.4f}"
    fmt_p = f"\\textbf{{{b['p_valor']:.3f}}}" if negrita else f"{b['p_valor']:.3f}"

    return (
        f"    {nombre:<24s} & {etiqueta} & {r_base['auc_roc_media']:.4f} & {r_geo['auc_roc_media']:.4f} & "
        f"{fmt_delta} & [{b['ci95_low']:.4f}, {b['ci95_high']:.4f}] & {fmt_p} & {p_cluster} & "
        f"{delta_top10:+.3f} \\\\"
    )


def generar_tex(registro: pd.DataFrame, boot: pd.DataFrame, boot_cl: pd.DataFrame, cfg: dict) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Contribución marginal de DMSP-OLS bajo pobreza multidimensional",
        r"  (IPM), holdout temporal (train 2010$\to$2013, test 2013$\to$2016). $\Delta$",
        r"  AUC-ROC y su IC95\% son del bootstrap pareado sobre el conjunto de prueba",
        r"  (\texttt{diagnostico\_bootstrap\_ipm.py}); $p$ (hogar) no corrige por",
        r"  correlación intra-comunidad, $p$ (cluster) sí",
        r"  (\texttt{diagnostico\_bootstrap\_cluster\_ipm.py}, 1{,}696 clusters).",
        r"  Filas en \textbf{negrita}: IC95\% no cruza cero. AUC base/+DMSP son la",
        r"  media de 5 semillas (Sección~\ref{subsec:desempeno}); $\Delta$/IC95\%/$p$",
        r"  vienen de un bootstrap sobre una sola semilla (42) del mismo conjunto de",
        r"  prueba, por eso pueden diferir del AUC de 5 semillas en el tercer decimal.}",
        r"  \label{tab:marginal_dmsp_ipm_significancia}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llccccccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Esp.} & \textbf{AUC base} & \textbf{AUC +DMSP} & "
        r"$\boldsymbol{\Delta}$\textbf{AUC} & \textbf{IC95\%} & \textbf{$p$ (hogar)} & "
        r"\textbf{$p$ (cluster)} & $\boldsymbol{\Delta}$\textbf{top-10\%} \\",
        r"    \midrule",
    ]
    for i, (espec_base, espec_geo) in enumerate(cfg["pares_especificacion"]):
        for j, algoritmo_raw in enumerate(cfg["algoritmos_orden"]):
            lineas.append(fila_tex(registro, boot, boot_cl, algoritmo_raw, espec_base, espec_geo, cfg))
            if j < len(cfg["algoritmos_orden"]) - 1:
                lineas.append(r"    \addlinespace")
        if i < len(cfg["pares_especificacion"]) - 1:
            lineas.append(r"    \addlinespace")
    lineas += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}"]
    return "\n".join(lineas)


def main() -> None:
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    registro, boot, boot_cl = cargar(cfg)
    tex = generar_tex(registro, boot, boot_cl, cfg)
    ruta_tex = out_dir / "tab_marginal_dmsp_ipm_significancia.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada: {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
