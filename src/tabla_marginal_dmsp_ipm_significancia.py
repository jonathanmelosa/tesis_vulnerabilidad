"""
tabla_marginal_dmsp_ipm_significancia.py
=====================================
Tabla de contribucion marginal de DMSP-OLS bajo pobreza MULTIDIMENSIONAL
(IPM) -- analoga a `tabla_marginal_dmsp_fbeta2_cv10.py` (monetaria), pero
con el foco puesto en la SIGNIFICANCIA del cambio de AUC-ROC al agregar
DMSP-OLS (columna Delta del bootstrap pareado), no solo en el valor de AUC
de cada especificacion por separado -- por eso el formato es una fila por
PAR (base vs. +DMSP-OLS), no una fila por especificacion como en la
version monetaria. El IC95% y los valores p (sin y con corregir por
clustering comunitario) NO van en columnas propias de la tabla (pedido
explicito del usuario, 2026-09-18, para no sobrecargarla) -- se resumen en
la nota al pie, generada dinamicamente en `_nota_significancia()` a partir
de los pares con Delta AUC-ROC significativo, junto con la explicacion de
por que se corrige por clustering comunitario (resolucion espacial gruesa
de DMSP-OLS, verificada: 30 arcosegundos/~1km vs. 15 arcosegundos/~500m de
VIIRS, su sucesor, con mejor calibracion y menos saturacion -- la
limitacion de granularidad de DMSP-OLS es precisamente lo que motivo el
diseno de VIIRS).

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
        corrido; se resume como "sin cobertura" en la nota si aplica a un
        par significativo)

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


def fila_tex(registro: pd.DataFrame, boot: pd.DataFrame,
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

    delta_top10 = r_geo["precision_top10_media"] - r_base["precision_top10_media"]

    negrita = not bool(b["cruza_cero"])
    fmt_delta = f"\\textbf{{{b['delta_auc']:+.4f}}}" if negrita else f"{b['delta_auc']:+.4f}"

    return (
        f"    {nombre:<24s} & {etiqueta} & {r_base['auc_roc_media']:.4f} & {r_geo['auc_roc_media']:.4f} & "
        f"{fmt_delta} & {delta_top10:+.3f} \\\\"
    )


def _nota_significancia(boot: pd.DataFrame, boot_cl: pd.DataFrame, cfg: dict) -> list:
    """Arma la frase de la nota que resume, por fuera de la tabla, los IC95%
    y valores p (con y sin corregir por clustering comunitario) de los
    pares con Delta AUC-ROC significativo -- en vez de columnas propias en
    la tabla."""
    sig = boot[~boot["cruza_cero"]].copy()
    partes = []
    for _, fila in sig.iterrows():
        bc = boot_cl[(boot_cl.algoritmo == fila["algoritmo"]) & (boot_cl.especificacion_base == fila["especificacion_base"])]
        etiqueta = cfg["etiqueta_especificacion"][fila["especificacion_base"]]
        if bc.empty:
            texto_p = f"$p={fila['p_valor']:.3f}$ sin corregir por clustering (sin cobertura de bootstrap cluster-robusto)"
        else:
            texto_p = f"$p={fila['p_valor']:.3f}$ sin corregir y $p={bc.iloc[0]['p_valor_cluster']:.3f}$ corrigiendo por clustering"
        partes.append(
            f"{fila['algoritmo']}-{etiqueta} (IC95\\% [{fila['ci95_low']:.4f}, {fila['ci95_high']:.4f}], {texto_p})"
        )
    return partes


def generar_tex(registro: pd.DataFrame, boot: pd.DataFrame, boot_cl: pd.DataFrame, cfg: dict) -> str:
    detalle_significancia = "; ".join(_nota_significancia(boot, boot_cl, cfg))
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Contribución marginal de DMSP-OLS bajo pobreza",
        r"  multidimensional (IPM), holdout temporal (train 2010$\to$2013,",
        r"  test 2013$\to$2016).}",
        r"  \label{tab:marginal_dmsp_ipm_significancia}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    \textbf{Algoritmo} & \textbf{Esp.} & \textbf{AUC base} & \textbf{AUC +DMSP} & "
        r"$\boldsymbol{\Delta}$\textbf{AUC} & $\boldsymbol{\Delta}$\textbf{top-10\%} \\",
        r"    \midrule",
    ]
    for i, (espec_base, espec_geo) in enumerate(cfg["pares_especificacion"]):
        for j, algoritmo_raw in enumerate(cfg["algoritmos_orden"]):
            lineas.append(fila_tex(registro, boot, algoritmo_raw, espec_base, espec_geo, cfg))
            if j < len(cfg["algoritmos_orden"]) - 1:
                lineas.append(r"    \addlinespace")
        if i < len(cfg["pares_especificacion"]) - 1:
            lineas.append(r"    \addlinespace")
    lineas += [
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"  \begin{minipage}{0.95\textwidth}",
        r"    \vspace{4pt}",
        r"    \footnotesize \textit{Nota:} AUC base/+DMSP son la media de 5",
        r"    semillas (Sección~\ref{subsec:desempeno}); $\Delta$AUC viene de",
        r"    un bootstrap pareado sobre una sola semilla (42) del mismo",
        r"    conjunto de prueba, por eso puede diferir del AUC de 5 semillas",
        r"    en el tercer decimal. Filas en \textbf{negrita}: el IC95\% de",
        r"    $\Delta$AUC no cruza cero -- " + detalle_significancia + ". La",
        r"    corrección por clustering comunitario (remuestreo agrupado por",
        r"    \texttt{id\_comunidad}, 1{,}696 comunidades, disponible para",
        r"    XGBoost, HistGradientBoosting y Logística) es necesaria porque",
        r"    DMSP-OLS tiene una resolución espacial gruesa (30 arcosegundos,",
        r"    $\sim$1\,km) -- la limitación que llevó al desarrollo de VIIRS",
        r"    (15 arcosegundos, $\sim$500\,m, con mejor calibración y menos",
        r"    saturación en zonas brillantes) como sucesor de DMSP-OLS-- así",
        r"    que hogares de una misma comunidad comparten, en buena medida,",
        r"    el mismo píxel o uno adyacente de iluminación nocturna, y el",
        r"    error estándar ingenuo por hogar puede subestimar la",
        r"    incertidumbre si no se corrige por esa correlación compartida.",
        r"    Fuente: cálculos propios.",
        r"  \end{minipage}",
        r"\end{table}",
    ]
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
