"""
tabla_shap_nucleo_perfil.py
=====================================
Tabla de perfil del "núcleo" de variables (Sección~5.3 de `main.tex`,
Hallazgo Primero): las 22 variables que aparecen, en al menos 5 de las 10
combinaciones algoritmo/especificación, dentro del conjunto que acumula
el 80\\% del SHAP total (`src/05_model/diagnostico_shap_nucleo_perfil.py`
calcula el núcleo; este script solo formatea, no calcula). Reutiliza las
etiquetas legibles de `tabla_perfil_completo.py` (mismas usadas en la
tabla de perfil de la Sección~5.1) para las 14 variables que ya estaban en
ese perfil univariado de 53 variables, y agrega etiquetas nuevas para las
8 que no --propuestas al usuario y aprobadas, ver
`docs/decisions.md`--.

INPUTS

    data/processed/benchmark_resultados/diagnostico_shap_nucleo_perfil.csv

OUTPUTS

    paper/tables/tab_shap_nucleo_perfil.tex

COMO CORRER

    python src/tabla_shap_nucleo_perfil.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabla_perfil_completo import ETIQUETAS as ETIQUETAS_PERFIL_53

REPO_ROOT = Path(__file__).resolve().parents[1]

RUTA_ENTRADA = REPO_ROOT / "data" / "processed" / "benchmark_resultados" / "diagnostico_shap_nucleo_perfil.csv"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

# Etiquetas de las 8 variables del núcleo que no están en el perfil
# univariado de 53 variables de la Sección 5.1 (y por tanto no tienen
# entrada en ETIQUETAS_PERFIL_53).
ETIQUETAS_NUEVAS = {
    # \char`\%{} en vez de \%: babel-spanish redefine \% con un \unskip
    # que borra el \quad de sangria que va justo antes (la fila quedaba
    # sin sangria); \char imprime el mismo glifo sin pasar por babel.
    "pct_ninos_madre_viva": "\\char`\\%{} de niños con madre viva",
    "pct_ninos_padre_vivo": "\\char`\\%{} de niños con padre vivo",
    "edad_jefe": "Edad del jefe de hogar",
    "grado_educ_jefe": "Grado educativo del jefe (años)",
    "tvip_puntaje_directo_hogar": "Puntaje de vocabulario infantil (test TVIP)",
    "tuvo_choque_economico_hogar": "Tuvo un choque económico (hogar)",
    "tasa_control_preventivo_hogar": "Tasa de controles médicos preventivos (hogar)",
    "n_desplazados_comunidad": "N.\\textsuperscript{o} de desplazados en la comunidad",
}

ORDEN_CATEGORIAS = [
    "Activos del hogar", "Educación, empleo y seguridad social",
    "Vivienda: materiales y servicios", "Vivienda: hacinamiento",
    "Composición del hogar", "Desarrollo infantil (6--9 años)", "Salud",
    "Programas sociales y deuda", "Choques y afrontamiento", "Comunidad",
]


# Variables categóricas cuya etiqueta en `tabla_perfil_completo.py` es una
# plantilla por nivel ("Material de piso: {nivel}"), pensada para la tabla
# de perfil de la Sección 5.1 (una fila por categoría). Aquí el SHAP ya
# está agregado a nivel de VARIABLE completa (todas las categorías
# sumadas), así que se usa el nombre de la variable sin nivel.
ETIQUETAS_SIN_NIVEL = {
    "material_pisos_hogar": "Material de piso del hogar",
    "estado_civil_jefe": "Estado civil del jefe",
}


def etiqueta(variable: str) -> str:
    if variable in ETIQUETAS_SIN_NIVEL:
        return ETIQUETAS_SIN_NIVEL[variable]
    if variable in ETIQUETAS_PERFIL_53:
        return ETIQUETAS_PERFIL_53[variable]
    if variable in ETIQUETAS_NUEVAS:
        return ETIQUETAS_NUEVAS[variable]
    raise ValueError(f"Variable del núcleo sin etiqueta legible: {variable}")


def formatear_fila(fila: pd.Series) -> str:
    marca_nueva = "" if fila["en_perfil_53"] else "$^{\\dagger}$"
    return (
        f"    \\quad {etiqueta(fila['variable'])}{marca_nueva} & "
        f"{fila['n_combinaciones']}/10 \\\\"
    )


def generar_tex(df: pd.DataFrame) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Núcleo de variables del análisis de importancia",
        r"  multivariada: aparecen, en al menos 5 de las 10 combinaciones",
        r"  algoritmo/especificación, dentro del 80\% de la masa SHAP",
        r"  acumulada (Sección~\ref{subsec:importancia_resultados}).",
        r"  $^{\dagger}$variable fuera del perfil univariado de 53",
        r"  variables robustas de la Sección~\ref{subsec:caracterizacion_grupos}.}",
        r"  \label{tab:shap_nucleo_perfil}",
        r"  \footnotesize",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    \textbf{Variable} & \textbf{Combinaciones (de 10)} \\",
        r"    \midrule",
    ]
    for categoria in ORDEN_CATEGORIAS:
        bloque = df[df["categoria"] == categoria].sort_values("n_combinaciones", ascending=False)
        if bloque.empty:
            continue
        lineas.append(f"    \\textbf{{{categoria}}} & \\\\")
        for _, fila in bloque.iterrows():
            lineas.append(formatear_fila(fila))
        lineas.append(r"    \addlinespace")
    if lineas[-1] == r"    \addlinespace":
        lineas.pop()
    lineas += [r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lineas)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUTA_ENTRADA)

    categorias_csv = set(df["categoria"])
    categorias_orden = set(ORDEN_CATEGORIAS)
    assert categorias_csv <= categorias_orden, (
        f"Categorías en el CSV no contempladas en ORDEN_CATEGORIAS: {categorias_csv - categorias_orden}"
    )

    tex = generar_tex(df)
    ruta_tex = OUTPUT_DIR / "tab_shap_nucleo_perfil.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"Tabla exportada (cuerpo de tabular; encabezado table/caption/nota siguen a mano en main.tex): {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
