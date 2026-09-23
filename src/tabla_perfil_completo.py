"""
tabla_perfil_completo.py
=====================================
Genera UNA TABLA LATEX POR CATEGORIA (no una tabla unica con todas las
variables) a partir de `perfil_completo_{monetaria,ipm}_2010_2013.csv`,
ya producidos por `src/02_build/eda_perfil_completo.py` /
`eda_perfil_completo_ipm.py`.

Cambio 2026-09-11 (pedido del usuario, tras detectar que la tabla unica
en formato `longtable` con filas de sub-encabezado por categoria se
desbordaba hacia la derecha y no mostraba los 4 grupos): se abandona
`longtable` y se vuelve a `table`+`tabular`+`resizebox` (mismo patron que
`tabla_covariables_transicion.py`, ya probado), UNA POR CATEGORIA -- mas
angosto, sin fila de sub-encabezado que rompa el ancho de columna.

El cuerpo de la Seccion 5.2 YA NO usa las tablas por categoria de
monetaria (las reemplazan `tab_perfil_consolidada`,
`tab_perfil_split_trayectoria` y `tab_perfil_nucleo_comun`); las
`tab_perfil_*_monetaria.tex` fueron borradas de `paper/tables/` el
2026-09-23 y este script las regenera si se corre de nuevo. El anexo de
IPM SI usa una tabla por cada una de sus 8 categorias (es una tabla de
referencia completa, no narrativa).

El formato de cada fila (pct / pct_nivel / num) se INFIERE
automaticamente de `tipo` + `nivel_mostrado` del CSV -- ver
`inferir_formato`.

INPUTS
    outputs/tables/eda_transicion_covariables/perfil_completo_{monetaria,ipm}_2010_2013.csv

OUTPUTS
    paper/tables/tab_perfil_<categoria-slug>_monetaria.tex (1 por categoria)
    paper/tables/tab_perfil_<categoria-slug>_ipm.tex (1 por categoria)

COMO CORRER
    python src/tabla_perfil_completo.py
"""

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "outputs" / "tables" / "eda_transicion_covariables"
OUTPUT_DIR = REPO_ROOT / "paper" / "tables"

CATEGORIAS_ORDEN_GRUPO = ["Siempre pobre", "Sale de la pobreza", "Entra en pobreza", "Nunca pobre"]

CORRECCIONES_NIVEL = {"Bldosa, vinilo, tableta o ladrillo": "Baldosa, vinilo, tableta o ladrillo"}

ETIQUETAS = {
    "brecha_lp_ingreso": "Ingreso (veces la línea de pobreza)",
    "brecha_lp_gasto": "Gasto (veces la línea de pobreza)",
    "ingreso_percapita_hogar_real": "Ingreso per cápita",
    "gasto_percapita_hogar_real": "Gasto per cápita",
    "lp": "Línea de pobreza (referencia)",
    "li": "Línea de indigencia (referencia)",
    "zona": "Vive en zona rural",
    "dmsp_stable_lights": "Iluminación nocturna (DMSP)",
    "material_pisos_hogar": "Material de piso: {nivel}",
    "energia_cocinan_hogar": "Combustible de cocina: {nivel}",
    "servicio_sanitario_hogar": "Servicio sanitario: {nivel}",
    "eliminan_basura_hogar": "Recolección de basura: {nivel}",
    "obtencion_agua_hogar": "Obtención de agua: {nivel}",
    "tipo_vivienda_hogar": "Tipo de vivienda: {nivel}",
    "material_paredes_hogar": "Material de paredes: {nivel}",
    "tenencia_vivienda_hogar": "Tenencia de vivienda: {nivel}",
    "personas_por_cuarto_hogar": "Personas por cuarto",
    "personas_por_dormitorio_hogar": "Personas por dormitorio",
    "valor_arriendo_pagado_hogar": "Valor del arriendo pagado",
    "n_bienes_durables_hogar": "N.\\textsuperscript{o} de bienes durables",
    "n_servicios_publicos_hogar": "N.\\textsuperscript{o} de servicios públicos",
    "n_activos_financieros_hogar": "N.\\textsuperscript{o} de activos financieros",
    "estrato_hogar": "Estrato (autorreportado)",
    "estrato_verificado_hogar": "Estrato (verificado)",
    "riqueza_pca_hogar": "Índice de riqueza (PCA)",
    "tiene_internet_hogar": "Tiene internet en el hogar",
    "n_programas_sociales_hogar": "N.\\textsuperscript{o} de programas sociales",
    "beneficiario_familias_accion_hogar": "Beneficiario de Familias en Acción",
    "beneficiario_algun_programa_hogar": "Beneficiario de algún programa social",
    "nivel_educ_jefe": "Educación del jefe: {nivel}",
    "categoria_ocupacional_jefe": "Ocupación del jefe: {nivel}",
    "medio_consiguio_jefe": "Consiguió trabajo: {nivel}",
    "registro_mercantil_jefe": "Registro mercantil del jefe: {nivel}",
    "n_empleados_jefe": "Negocio del jefe: {nivel}",
    "etnia_jefe": "Etnia del jefe: {nivel}",
    "estado_civil_jefe": "Estado civil del jefe: {nivel}",
    "sexo_jefe": "Sexo del jefe: {nivel}",
    "tasa_cotizacion_pension_hogar": "Cotización a pensión (hogar)",
    "cotiza_pension_jefe": "Jefe cotiza a pensión",
    "nivel_educ_max_hogar": "Máx. nivel educativo del hogar",
    "nivel_educ_ordinal_jefe": "Nivel educativo del jefe (ordinal)",
    "pct_adultos_alfabetizados": "Adultos alfabetizados en el hogar",
    "n_ninos_12": "N.\\textsuperscript{o} de niños en el hogar",
    "pct_ninos_cuidado_terceros_hogar": "Niños al cuidado de terceros",
    "razon_dependencia_demografica": "Razón de dependencia demográfica",
    "pobre_ingreso": "Pobre por ingreso (2010)",
    "pobre_extremo_ingreso": "Pobre extremo por ingreso (2010)",
    "pobre_gasto": "Pobre por gasto (2010)",
    "pobre_extremo_gasto": "Pobre extremo por gasto (2010)",
    # -- Extensión 2026-09-15 (VARIABLES_ESTABLES_AMPLIADAS) --
    "tasa_afiliacion_pension_hogar": "Afiliación a pensión (hogar)",
    "tasa_afiliacion_salud_laboral_hogar": "Afiliación a salud laboral (hogar)",
    "afiliado_pension_jefe": "Jefe afiliado a pensión",
    "afiliado_salud_laboral_jefe": "Jefe afiliado a salud laboral",
    "deuda_formal_hogar": "Tiene deuda formal",
    "deuda_informal_hogar": "Tiene deuda informal",
    "tiene_escritura_vivienda_hogar": "Tiene escritura de la vivienda",
    "financio_credito_formal_vivienda_hogar": "Financió la vivienda con crédito formal",
    "tiene_vehiculo_hogar": "Tiene vehículo",
    "pct_ninos_apoyo_alimentario_escolar": "Niños con apoyo alimentario escolar",
    "n_espacios_publicos_comunidad": "N.\\textsuperscript{o} de espacios públicos en la comunidad",
    "tiene_transporte_publico_comunidad": "Comunidad con transporte público",
}


def slug(categoria: str) -> str:
    s = categoria.lower()
    s = s.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def inferir_formato(tipo: str, nivel_mostrado: str) -> str:
    if "%" in nivel_mostrado:
        return "pct"
    if tipo in ("Categorica", "Booleana"):
        return "pct_nivel" if tipo == "Categorica" else "pct"
    return "num"


def formatear_valor(valor: float, formato: str) -> str:
    if formato in ("pct", "pct_nivel"):
        return f"{valor:.0f}\\%"
    if formato == "num":
        if abs(valor) >= 1000:
            return f"{valor / 1000:,.0f}".replace(",", ".") + "k"
        if abs(valor) >= 10:
            return f"{valor:.1f}"
        return f"{valor:.2f}"
    raise ValueError(f"formato desconocido: {formato}")


def etiqueta_fila(variable: str, nivel_mostrado: str) -> str:
    etiqueta = ETIQUETAS.get(variable, variable.replace("_", "\\_"))
    if "{nivel}" in etiqueta:
        nivel = CORRECCIONES_NIVEL.get(nivel_mostrado, nivel_mostrado)
        etiqueta = etiqueta.format(nivel=nivel)
    return etiqueta


def generar_tabla_categoria(sub: pd.DataFrame, caption: str, label: str) -> str:
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
        r"    \textbf{Variable} & \textbf{Siempre pobre} & \textbf{Sale} & \textbf{Entra} & \textbf{Nunca pobre} \\",
        r"    \midrule",
    ]
    for _, fila in sub.iterrows():
        formato = inferir_formato(fila["tipo"], fila["nivel_mostrado"])
        etiqueta = etiqueta_fila(fila["variable"], fila["nivel_mostrado"])
        valores = " & ".join(formatear_valor(fila[cat], formato) for cat in CATEGORIAS_ORDEN_GRUPO)
        lineas.append(f"    {etiqueta} & {valores} \\\\")
    lineas += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}"]
    return "\n".join(lineas)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fuentes = {
        "monetaria": ("pobreza monetaria", "monetaria"),
        "ipm": ("pobreza multidimensional (IPM)", "ipm"),
    }
    for nombre, (etiqueta_pobreza, sufijo) in fuentes.items():
        tabla = pd.read_csv(TABLES_DIR / f"perfil_completo_{nombre}_2010_2013.csv")
        for categoria, sub in tabla.groupby("categoria", sort=False):
            cat_slug = slug(categoria)
            caption = f"{categoria} -- perfil por grupo de transición, {etiqueta_pobreza} (2010$\\to$2013)."
            label = f"tab:perfil_{cat_slug}_{sufijo}"
            tex = generar_tabla_categoria(sub, caption, label)
            ruta = OUTPUT_DIR / f"tab_perfil_{cat_slug}_{sufijo}.tex"
            ruta.write_text(tex + "\n", encoding="utf-8")
            print(f"Guardado: {ruta.name} ({len(sub)} filas)")


if __name__ == "__main__":
    main()
