"""
graficar_residual_dmsp_riqueza.py
==================================
Grafica el diagnostico de `diagnostico_residual_dmsp_riqueza.py`: la
correlacion de dmsp_stable_lights con caer en pobreza es, en crudo,
comparable en magnitud a la de las 4 variables que dominan el nucleo de
importancia multivariada (servicios publicos, estrato verificado, bienes
durables, personas por cuarto), pero colapsa de -0.214 a +0.010 al
residualizar contra las 4 -- la evidencia visual de que la redundancia,
no la ausencia de señal, explica por que DMSP-OLS no aporta poder
predictivo incremental bajo pobreza monetaria (paper/main.tex, seccion
"Por que DMSP-OLS no aporta").

El color codifica el contraste que importa para el argumento: correlacion
bruta (gris, el punto de comparacion) vs. residualizada (rojo, el
resultado clave), no de quien es cada variable.

NO recalcula nada: solo lee el CSV que ya produjo
`diagnostico_residual_dmsp_riqueza.py`.

INPUT

    data/processed/benchmark_resultados/diagnostico_residual_dmsp_riqueza.csv

OUTPUT

    outputs/figures/modelos/05_residual_dmsp_riqueza.png

COMO CORRER

    cd src/05_model && python graficar_residual_dmsp_riqueza.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_residual_dmsp_riqueza.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "modelos" / "05_residual_dmsp_riqueza.png"

COLOR_DMSP_BRUTA = "#4D4D4D"
COLOR_REFERENCIA_BRUTA = "#8C8C8C"
COLOR_RESIDUALIZADA = "#C44E52"

ETIQUETAS = [
    ("Iluminación nocturna\n(DMSP-OLS)", "cruda", "dmsp_stable_lights", COLOR_DMSP_BRUTA),
    ("N.º de servicios públicos\n(referencia)", "cruda (control vs outcome)", "n_servicios_publicos_hogar", COLOR_REFERENCIA_BRUTA),
    ("Estrato verificado\n(referencia)", "cruda (control vs outcome)", "estrato_verificado_hogar", COLOR_REFERENCIA_BRUTA),
    ("N.º de bienes durables\n(referencia)", "cruda (control vs outcome)", "n_bienes_durables_hogar", COLOR_REFERENCIA_BRUTA),
    ("Personas por cuarto\n(referencia)", "cruda (control vs outcome)", "personas_por_cuarto_hogar", COLOR_REFERENCIA_BRUTA),
    ("Iluminación nocturna\n(DMSP-OLS), residualizada$^{*}$", "parcial (sklearn.LinearRegression)", "dmsp_stable_lights", COLOR_RESIDUALIZADA),
]


def main() -> None:
    tabla = pd.read_csv(INPUT_PATH)

    filas = []
    for etiqueta, comparacion, variable, color in ETIQUETAS:
        fila = tabla[(tabla["comparacion"] == comparacion) & (tabla["variable"] == variable)].iloc[0]
        filas.append({"etiqueta": etiqueta, "r": fila["r"], "n": int(fila["n"]), "color": color})
    datos = pd.DataFrame(filas).iloc[::-1]  # para que la primera quede arriba en barh

    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    barras = ax.barh(datos["etiqueta"], datos["r"], color=datos["color"], height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.3, 0.3)
    ax.set_xlabel("Correlación de Pearson (r) con caer en pobreza monetaria")

    UMBRAL_BARRA_CORTA = 0.05  # bajo este |r|, la barra es muy angosta para el texto adentro
    for barra, r in zip(barras, datos["r"]):
        y_centro = barra.get_y() + barra.get_height() / 2
        if abs(r) >= UMBRAL_BARRA_CORTA:
            ax.text(r / 2, y_centro, f"{r:+.3f}", va="center", ha="center",
                    fontsize=10, fontweight="bold", color="white")
        else:
            desplazamiento = 0.012 if r >= 0 else -0.012
            alineacion = "left" if r >= 0 else "right"
            ax.text(r + desplazamiento, y_centro, f"{r:+.3f}", va="center", ha=alineacion,
                    fontsize=10, fontweight="bold", color="black")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_DMSP_BRUTA, label="DMSP-OLS, correlación bruta"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_REFERENCIA_BRUTA, label="Referencias, correlación bruta"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RESIDUALIZADA, label="DMSP-OLS, residualizada$^{*}$"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=1, frameon=False, fontsize=9)

    ax.set_title("La correlación de DMSP-OLS con caer en pobreza es comparable a la de\nlas variables de encuesta más predictivas -- y colapsa al controlar por ellas", fontsize=11)
    fig.text(0.01, -0.02, "$^{*}$Controlando por las 4 variables de referencia (correlación parcial vía residuos,\nteorema de Frisch–Waugh–Lovell).",
              fontsize=8, color="#555555")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
