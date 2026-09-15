"""
graf_perfil_categorias_sparse.py
=====================================
Genera una figura (PNG) por cada categoria del perfil completo de
covariables que tiene POCOS items (Zona de residencia, Vivienda:
hacinamiento, Geoespacial, Programas sociales y deuda, Composición del
hogar) -- pedido del usuario (2026-09-11): estas categorias, al tener
solo 1-3 variables, se muestran mejor como grafico (linea + puntos por
grupo de transicion, igual estilo que el artifact de exploracion) que
como tabla en el cuerpo de la Seccion 5.2. Las categorias con muchos
items (Ingreso y gasto, Vivienda: materiales y servicios, Activos del
hogar, Educacion y empleo del jefe) SI se quedan como tabla
(`tabla_perfil_completo.py`).

Estilo: eje X con las 4 categorias de transicion en texto (no numeros),
"Entra en pobreza" marcado como VULNERABLE con una franja de fondo --
mismo criterio ya usado en el artifact HTML de exploracion de esta
sesion.

INPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_monetaria_2010_2013.csv

OUTPUTS
    outputs/figures/eda_transicion_covariables/perfil_<categoria-slug>_monetaria.png
    (una por cada categoria en CATEGORIAS_SPARSE)

COMO CORRER
    python src/graf_perfil_categorias_sparse.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
FIG_DIR = REPO_ROOT / "outputs" / "figures" / "eda_transicion_covariables"

GRUPOS = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]
GRUPOS_ETIQUETA = ["Siempre\npobre", "Sale de la\npobreza", "Entra en\npobreza\n(VULNERABLE)", "Nunca\npobre"]

CATEGORIAS_SPARSE = [
    "Zona de residencia",
    "Vivienda: hacinamiento",
    "Geoespacial",
    "Programas sociales y deuda",
    "Composición del hogar",
    "Comunidad",
]

ETIQUETAS = {
    "zona": "Vive en zona rural (%)",
    "dmsp_stable_lights": "Iluminación nocturna (DMSP, 0-63)",
    "valor_arriendo_pagado_hogar": "Valor del arriendo (miles $/mes)",
    "personas_por_cuarto_hogar": "Personas por cuarto",
    "personas_por_dormitorio_hogar": "Personas por dormitorio",
    "n_programas_sociales_hogar": "N.º de programas sociales",
    "beneficiario_familias_accion_hogar": "Beneficiario Familias en Acción (%)",
    "beneficiario_algun_programa_hogar": "Beneficiario de algún programa (%)",
    "n_ninos_12": "N.º de niños en el hogar",
    "razon_dependencia_demografica": "Razón de dependencia demográfica",
    # -- Extensión 2026-09-15 (VARIABLES_ESTABLES_AMPLIADAS, ver
    # eda_perfil_completo.py) --
    "deuda_formal_hogar": "Tiene deuda formal (%)",
    "deuda_informal_hogar": "Tiene deuda informal (%)",
    "pct_ninos_cuidado_terceros_hogar": "Niños al cuidado de terceros (%)",
    "pct_ninos_apoyo_alimentario_escolar": "Niños con apoyo alimentario escolar (%)",
    "n_espacios_publicos_comunidad": "N.º de espacios públicos en la comunidad",
    "tiene_transporte_publico_comunidad": "Comunidad con transporte público (%)",
}

COLOR_LINEA = "#5b6570"
COLORES_SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def slug(categoria: str) -> str:
    s = categoria.lower()
    for a, b in [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")]:
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def formatear(valor: float, es_pct: bool) -> str:
    if es_pct:
        return f"{valor:.0f}%"
    if abs(valor) >= 1000:
        return f"{valor / 1000:,.0f}k".replace(",", ".")
    if abs(valor) >= 10:
        return f"{valor:.1f}"
    return f"{valor:.2f}"


def graficar_panel(ax, sub: pd.DataFrame, mostrar_leyenda: bool) -> None:
    x = [0, 1, 2, 3]
    ax.axvspan(1.6, 2.4, color="#eb6834", alpha=0.07, zorder=0)

    for i, (_, fila) in enumerate(sub.iterrows()):
        es_pct = "%" in ETIQUETAS.get(fila["variable"], "") or fila["tipo"] in ("Categorica", "Booleana")
        valores = [fila[g] for g in GRUPOS]
        color = COLORES_SERIE[i % len(COLORES_SERIE)]
        ax.plot(x, valores, "-o", color=color, linewidth=2, markersize=7,
                 label=ETIQUETAS.get(fila["variable"], fila["variable"]))
        for xi, v in zip(x, valores):
            ax.annotate(formatear(v, es_pct), (xi, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5, color=color, fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels(GRUPOS_ETIQUETA, fontsize=8)
    for label in ax.get_xticklabels():
        if "VULNERABLE" in label.get_text():
            label.set_color("#c1531a")
            label.set_fontweight("bold")

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.3, 3.3)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.15 or 1
    ax.set_ylim(ymin - pad * 0.1, ymax + pad)

    if mostrar_leyenda:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.20), fontsize=8.5, frameon=False, ncol=1)


def graficar_categoria(sub_categoria: pd.DataFrame, categoria: str) -> None:
    """Un panel por cada `subpanel_unidad` distinto dentro de la categoria
    (mismo criterio de unidad-compatible ya usado en el artifact HTML) --
    nunca se mezclan 2 unidades en el mismo eje Y."""
    grupos_unidad = list(sub_categoria.groupby("subpanel_unidad", sort=False))
    n = len(grupos_unidad)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.0), squeeze=False)
    axes = axes[0]

    for ax, (unidad, sub) in zip(axes, grupos_unidad):
        graficar_panel(ax, sub, mostrar_leyenda=len(sub) > 1)
        if len(sub) == 1:
            ax.set_title(ETIQUETAS.get(sub.iloc[0]["variable"], sub.iloc[0]["variable"]), fontsize=9.5)
        else:
            ax.set_title(unidad, fontsize=9.5)

    fig.tight_layout()
    out_path = FIG_DIR / f"perfil_{slug(categoria)}_monetaria.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path} ({n} panel(es): {[u for u,_ in grupos_unidad]})")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tabla = pd.read_csv(TABLES_DIR / "perfil_completo_monetaria_2010_2013.csv")
    for categoria in CATEGORIAS_SPARSE:
        sub = tabla[tabla["categoria"] == categoria]
        if sub.empty:
            raise ValueError(f"Categoria sin datos: {categoria}")
        graficar_categoria(sub, categoria)


if __name__ == "__main__":
    main()
