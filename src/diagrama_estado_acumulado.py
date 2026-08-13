"""
Diagrama pedagógico del diseño "estado vs. ventana acumulada" usado por
dmsp_ols_pipeline/, alos_palsar_pipeline/ y landsat5_pipeline/ (ver tesis,
Sección 3.3.7). No procesa ningún dato del proyecto — es una ilustración
conceptual de las ventanas temporales ya definidas en el código de esos
tres pipelines.

Output: paper/figures/fig_diagrama_estado_acumulado.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "paper" / "figures"

COLOR_ACUMULADO = "#2a78d6"   # azul — años que solo entran a la ventana acumulada
COLOR_ESTADO = "#d03b3b"      # rojo — año de estado (coincide con el color de "Ola" en la Figura 5)
COLOR_ENTRE_OLAS = "#4a3aa7"  # violeta — comparación entre los dos estados
COLOR_TEXTO = "#2b2b28"

fig, ax = plt.subplots(figsize=(9.5, 3.6))

anios = list(range(2008, 2014))
y_2010, y_2013 = 1.4, 0.5
alto_marca = 0.28

ventanas = [
    {"ronda": 2010, "anios": [2008, 2009, 2010], "anio_estado": 2010, "y": y_2010},
    {"ronda": 2013, "anios": [2011, 2012, 2013], "anio_estado": 2013, "y": y_2013},
]

for v in ventanas:
    y = v["y"]
    x0, x1 = v["anios"][0] - 0.4, v["anios"][-1] + 0.4
    # Barra de fondo: ventana acumulada completa
    ax.add_patch(Rectangle((x0, y - alto_marca / 2), x1 - x0, alto_marca,
                            facecolor=COLOR_ACUMULADO, alpha=0.18, edgecolor=COLOR_ACUMULADO, linewidth=1.0, zorder=1))
    ax.text(x0 - 0.15, y, f"Ola {v['ronda']}", ha="right", va="center", fontsize=11, fontweight="bold", color=COLOR_TEXTO)

    for anio in v["anios"]:
        es_estado = anio == v["anio_estado"]
        color = COLOR_ESTADO if es_estado else COLOR_ACUMULADO
        ax.scatter([anio], [y], s=260 if es_estado else 140, color=color, zorder=3,
                   edgecolor="white", linewidth=1.2, marker="*" if es_estado else "o")
        ax.text(anio, y - alto_marca / 2 - 0.16, str(anio), ha="center", va="top", fontsize=9, color=COLOR_TEXTO)

    ax.annotate("ventana acumulada\n(3 años)", xy=((x0 + x1) / 2, y + alto_marca / 2 + 0.02),
                ha="center", va="bottom", fontsize=8, color=COLOR_ACUMULADO, style="italic")
    ax.annotate("estado", xy=(v["anio_estado"], y + 0.22), ha="center", va="bottom",
                fontsize=8.5, color=COLOR_ESTADO, fontweight="bold")

# Flecha "cambio entre olas": conecta el estado 2010 con el estado 2013
arrow = FancyArrowPatch((2010, y_2010 - alto_marca / 2 - 0.02), (2013, y_2013 + alto_marca / 2 + 0.02),
                          connectionstyle="arc3,rad=-0.4", arrowstyle="-|>", mutation_scale=16,
                          color=COLOR_ENTRE_OLAS, linewidth=1.6, linestyle="--", zorder=2)
ax.add_patch(arrow)
ax.text(2011.5, 1.62, "cambio entre olas\n(2010 → 2013)", ha="center", va="center", fontsize=8.5,
        color=COLOR_ENTRE_OLAS, style="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR_ENTRE_OLAS, linewidth=0.8))

ax.set_xlim(2007.3, 2013.9)
ax.set_ylim(-0.05, 2.35)
ax.set_xticks(anios)
ax.set_yticks([])
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c0")
ax.tick_params(axis="x", colors=COLOR_TEXTO)
ax.set_title(
    "Tres formas de medir cambio, sin confundirlas entre sí\n"
    "(diseño usado por DMSP-OLS, ALOS PALSAR y Landsat 5 TM)",
    fontsize=11, color=COLOR_TEXTO,
)

leyenda = [
    mpatches.Patch(facecolor=COLOR_ESTADO, edgecolor="none", label="Estado — nivel en el año exacto de la ola (★)"),
    mpatches.Patch(facecolor=COLOR_ACUMULADO, alpha=0.5, edgecolor=COLOR_ACUMULADO, label="Ventana acumulada — media/tendencia/crecimiento sobre 3 años"),
    mpatches.Patch(facecolor="white", edgecolor=COLOR_ENTRE_OLAS, label="Cambio entre olas — estado 2010 vs. estado 2013", linestyle="--"),
]
ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, frameon=False, fontsize=8.5, alignment="left")

fig.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
out_path = FIG_DIR / "fig_diagrama_estado_acumulado.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Figura guardada en: {out_path}")
