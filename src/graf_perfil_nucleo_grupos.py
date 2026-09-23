"""
graf_perfil_nucleo_grupos.py
===================================
Figura de la Seccion 5.2 (Tercer hallazgo): donde queda el grupo
"Entra en pobreza" (los vulnerables) respecto a "Sale de la pobreza" y a
los dos extremos, para las variables del nucleo SHAP cuya DIRECCION es
estable (analisis de sensibilidad del signo) y que ademas estan en el
perfil univariado de 53 variables.

Por que estas variables: el hallazgo une los dos analisis de la seccion.
El nucleo y su direccion salen de SHAP (`diagnostico_shap_signo.py`,
`sensibilidad_shap_nucleo.py`); la posicion de los cuatro grupos de la
matriz de transicion sale del perfil univariado. Se usan las variables de
direccion estable o moderada (consistentes en >= 8 de las 12 celdas de la
malla de umbrales de `sensibilidad_shap_nucleo.py`) que tienen medias por
grupo en el perfil: las demas estables (grado educativo del jefe,
controles preventivos) no son parte de las 53 y no tienen medias de los
cuatro grupos calculadas.

POSICION RELATIVA (para poder comparar variables de escalas distintas):

    posicion = (media del grupo - media de Siempre pobre)
               / (media de Nunca pobre - media de Siempre pobre)

    0 = media de Siempre pobre, 1 = media de Nunca pobre. No depende del
    sentido de la variable (mas hacinamiento es peor, mas riqueza es
    mejor): ambos extremos quedan fijos. Un grupo esta "mas cerca de sale
    que de nunca pobre" si su posicion queda mas cerca de la de "Sale" que
    de 1.

INPUTS

    outputs/tables/eda_transicion_covariables/perfil_completo_monetaria_{2010_2013,2013_2016}.csv
    data/processed/benchmark_resultados/sensibilidad_signo_nucleo_por_variable.csv
    outputs/tables/pobreza/transicion_conteo_{ola1_a_2,ola2_a_3}.csv  (n de cada panel)

OUTPUTS

    outputs/figures/eda_transicion_covariables/perfil_nucleo_grupos_monetaria.png
    outputs/tables/eda_transicion_covariables/perfil_nucleo_grupos_posicion_monetaria.csv
    (posicion relativa de cada grupo por variable y ventana, detras de la figura)

COMO CORRER

    python src/graf_perfil_nucleo_grupos.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda_transicion_covariables"
RUTA_SIGNO_VARIABLE = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "sensibilidad_signo_nucleo_por_variable.csv"
RUTA_CONTEOS = PROJECT_ROOT / "outputs" / "tables" / "pobreza"

MIN_CELDAS_DIRECCION = 8  # de 12: direccion estable o moderada
VENTANAS = [("2010_2013", "2010 → 2013", "transicion_conteo_ola1_a_2.csv"),
            ("2013_2016", "2013 → 2016", "transicion_conteo_ola2_a_3.csv")]

ETIQUETAS = {
    "riqueza_pca_hogar": "Índice de riqueza (PCA)",
    "n_activos_financieros_hogar": "N.º de activos financieros",
    "estrato_verificado_hogar": "Estrato (verificado)",
    "nivel_educ_max_hogar": "Máx. nivel educativo del hogar",
    "nivel_educ_ordinal_jefe": "Nivel educativo del jefe",
    "personas_por_cuarto_hogar": "Personas por cuarto",
    "valor_arriendo_pagado_hogar": "Valor del arriendo pagado",
    "razon_dependencia_demografica": "Razón de dependencia",
}

# Misma paleta categorica que el resto de la tesis.
COLOR_SALE = "#1baf7a"
COLOR_ENTRA = "#eb6834"
INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECUNDARIO, "text.color": INK_PRIMARIO,
    "xtick.color": INK_MUTED, "ytick.color": INK_SECUNDARIO, "font.family": "sans-serif",
    "font.size": 10.5, "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def variables_a_graficar() -> list:
    por_variable = pd.read_csv(RUTA_SIGNO_VARIABLE)
    estables = set(por_variable.loc[por_variable["celdas_consistente"] >= MIN_CELDAS_DIRECCION, "variable"])
    perfiles = [pd.read_csv(TABLES_DIR / f"perfil_completo_monetaria_{v}.csv") for v, _, _ in VENTANAS]
    en_perfil = set.intersection(*[set(p["variable"]) for p in perfiles])
    return sorted(estables & en_perfil)


def posiciones(variables: list) -> pd.DataFrame:
    filas = []
    for ventana, _, _ in VENTANAS:
        perfil = pd.read_csv(TABLES_DIR / f"perfil_completo_monetaria_{ventana}.csv").set_index("variable")
        for v in variables:
            f = perfil.loc[v]
            siempre, nunca = f["Siempre pobre"], f["Nunca pobre"]
            filas.append({
                "ventana": ventana, "variable": v,
                "pos_sale": (f["Sale de la pobreza"] - siempre) / (nunca - siempre),
                "pos_entra": (f["Entra en pobreza"] - siempre) / (nunca - siempre),
            })
    return pd.DataFrame(filas)


def n_panel(archivo: str) -> int:
    return int(pd.read_csv(RUTA_CONTEOS / archivo, index_col=0).to_numpy().sum())


def main() -> None:
    variables = variables_a_graficar()
    pos = posiciones(variables)
    pos.to_csv(TABLES_DIR / "perfil_nucleo_grupos_posicion_monetaria.csv", index=False)

    orden = pos[pos["ventana"] == VENTANAS[0][0]].sort_values("pos_entra")["variable"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (ventana, rotulo, archivo) in zip(axes, VENTANAS):
        g = pos[pos["ventana"] == ventana].set_index("variable").loc[orden]
        y = range(len(orden))
        ax.hlines(y, g["pos_sale"], g["pos_entra"], color=INK_MUTED, linewidth=1.4, zorder=2)
        ax.scatter(g["pos_sale"], y, s=70, color=COLOR_SALE, marker="o", zorder=3, label="Sale de la pobreza")
        ax.scatter(g["pos_entra"], y, s=70, color=COLOR_ENTRA, marker="D", zorder=4, label="Entra en pobreza")
        ax.axvline(0, color=INK_MUTED, linewidth=1.0)
        ax.axvline(1, color=INK_MUTED, linewidth=1.0)
        ax.set_yticks(list(y))
        ax.set_yticklabels([ETIQUETAS.get(v, v) for v in orden])
        ax.set_xlim(-0.12, 1.12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["Siempre\npobre", "0.5", "Nunca\npobre"])
        ax.set_title(f"{rotulo} (n = {n_panel(archivo):,})", fontsize=12, loc="left", pad=10)
    fig.supxlabel("Posición entre las medias de Siempre pobre (0) y Nunca pobre (1)",
                  fontsize=10.5, color=INK_SECUNDARIO, y=0.02)
    handles, rotulos = axes[0].get_legend_handles_labels()
    fig.legend(handles, rotulos, loc="lower center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=2, fontsize=10)
    fig.suptitle(
        "Los hogares que entran en pobreza quedan más cerca de los que salen que de los que nunca fueron pobres\n"
        "(variables del núcleo SHAP con dirección estable o moderada; pobreza monetaria; metodología López-Calva y Ortiz-Juárez, 2014)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.subplots_adjust(wspace=0.06)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "perfil_nucleo_grupos_monetaria.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(pos.round(3).to_string(index=False))
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
