"""
tabla_covariables_transicion.py
=====================================
Genera la tabla compacta de covariables por grupo de transicion de
pobreza (Panel B de la pagina de resultados, una tabla por definicion de
pobreza) a partir de `tabla_comparativa_{monetaria,ipm}_2010_2013.csv`
(transicion 2010->2013, la unica publicada en el documento por ahora), ya
producida por `src/02_build/eda_transicion_covariables.py` (ranking +
seleccion de 12 covariables, incluyendo DMSP obligatoria). Este script NO
recalcula nada, solo formatea/traduce esas 12 filas a una tabla LaTeX
lista para el documento -- misma convencion que el resto de
`src/tabla_*.py` (booktabs, `\\resizebox`, salida a `paper/tables/`).

Formato de valores por variable (`ESPECIFICACION_VARIABLE` mas abajo):
  - "pct" / "pct_nivel": porcentaje del grupo en esa categoria/condicion
    (0 decimales, "%"). "pct_nivel" ademas inserta el nombre del nivel
    mostrado (`nivel_mostrado` del CSV) en la etiqueta de la fila.
  - "num1" / "num2": promedio en unidades originales (1 o 2 decimales).
  - "miles": promedio en unidades originales / 1000 (para variables
    monetarias, mas legible en la tabla).

INPUTS
    outputs/tables/eda_transicion_covariables/tabla_comparativa_{monetaria,ipm}_2010_2013.csv

OUTPUTS
    paper/tables/tab_covariables_transicion_{monetaria,ipm}.tex

COMO CORRER
    python src/tabla_covariables_transicion.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

CATEGORIAS_ORDEN = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]

# variable -> (etiqueta de fila, "{nivel}" se reemplaza si aplica; formato)
ESPECIFICACION_VARIABLE = {
    "brecha_lp_ingreso": ("Brecha a la línea de pobreza (ingreso, razón)", "num2"),
    "zona": ("Zona rural", "pct_nivel"),
    "material_pisos_hogar": ("Material de piso: {nivel}", "pct_nivel"),
    "nivel_educ_jefe": ("Educación del jefe: {nivel}", "pct_nivel"),
    "energia_cocinan_hogar": ("Combustible de cocina: {nivel}", "pct_nivel"),
    "servicio_sanitario_hogar": ("Servicio sanitario: {nivel}", "pct_nivel"),
    "gasto_percapita_hogar": ("Gasto per cápita del hogar (miles \\$/mes)", "miles"),
    "categoria_ocupacional_jefe": ("Ocupación del jefe: {nivel}", "pct_nivel"),
    "medio_consiguio_jefe": ("Consiguió trabajo: {nivel}", "pct_nivel"),
    "dmsp_stable_lights": ("Iluminación nocturna (DMSP, 2010)", "num1"),
    "n_espacios_publicos_comunidad": ("N.\\textsuperscript{o} espacios públicos en la comunidad", "num1"),
    "tiene_deuda_hogar": ("Hogar con deuda formal", "pct"),
    "pobre_extremo_ingreso": ("Pobre extremo por ingreso (2010)", "pct"),
    "obtencion_agua_hogar": ("Obtención de agua: {nivel}", "pct_nivel"),
    "pobre_gasto": ("Pobre por gasto (2010)", "pct"),
    "pobre_extremo_gasto": ("Pobre extremo por gasto (2010)", "pct"),
    "pct_adultos_alfabetizados": ("Adultos alfabetizados en el hogar", "pct"),
    "tenencia_vivienda_hogar": ("Tenencia de vivienda: {nivel}", "pct_nivel"),
}

# Errores de tipeo conocidos en las etiquetas de nivel de la encuesta
# fuente (ELCA) -- se corrigen solo para presentacion, no en los datos.
CORRECCIONES_NIVEL = {"Bldosa, vinilo, tableta o ladrillo": "Baldosa, vinilo, tableta o ladrillo"}


def formatear_valor(valor: float, formato: str) -> str:
    if formato in ("pct", "pct_nivel"):
        return f"{valor:.0f}\\%"
    if formato == "num1":
        return f"{valor:.1f}"
    if formato == "num2":
        return f"{valor:.2f}"
    if formato == "miles":
        return f"{valor / 1000:,.0f}".replace(",", ".")
    raise ValueError(f"formato desconocido: {formato}")


def etiqueta_fila(variable: str, nivel_mostrado: str) -> str:
    etiqueta, formato = ESPECIFICACION_VARIABLE[variable]
    if formato == "pct_nivel":
        nivel = CORRECCIONES_NIVEL.get(nivel_mostrado, nivel_mostrado)
        etiqueta = etiqueta.format(nivel=nivel)
    return etiqueta, formato


def generar_tex(tabla: pd.DataFrame, caption: str, label: str) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Covariable} & \textbf{Siempre pobre} & \textbf{Sale} & \textbf{Entra} & \textbf{Nunca pobre} \\",
        r"    \midrule",
    ]
    for _, fila in tabla.iterrows():
        if fila["variable"] not in ESPECIFICACION_VARIABLE:
            raise KeyError(f"Falta especificacion de formato para la variable {fila['variable']!r}")
        etiqueta, formato = etiqueta_fila(fila["variable"], fila["nivel_mostrado"])
        valores = " & ".join(formatear_valor(fila[cat], formato) for cat in CATEGORIAS_ORDEN)
        lineas.append(f"    {etiqueta} & {valores} \\\\")
    lineas += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}"]
    return "\n".join(lineas)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    especificaciones = [
        ("monetaria", "Covariables por grupo de transición de pobreza monetaria (2010$\\to$2013).", "tab:covariables_transicion_monetaria"),
        ("ipm", "Covariables por grupo de transición de pobreza multidimensional -- IPM (2010$\\to$2013).", "tab:covariables_transicion_ipm"),
    ]
    for nombre, caption, label in especificaciones:
        tabla = pd.read_csv(TABLES_DIR / f"tabla_comparativa_{nombre}_2010_2013.csv")
        tex = generar_tex(tabla, caption, label)
        ruta = OUTPUT_DIR / f"tab_covariables_transicion_{nombre}.tex"
        ruta.write_text(tex + "\n", encoding="utf-8")
        print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
