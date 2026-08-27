"""
tabla_cobertura_modulos.py
=====================================
Genera la tabla de cobertura de covariables por módulo en el dataset
consolidado (Sección "Estadísticas descriptivas",
Tabla~\\ref{tab:cobertura_modulos} de la tesis).

QUÉ HACE Y QUÉ NO PUEDE RECALCULAR DESDE CERO
----------------------------------------------------------------------
"Candidatas auditadas" (columna 2) SÍ es 100% reproducible: es el conteo
de filas con `clasificacion == "CANDIDATO_BENCHMARK"` en cada archivo de
auditoría de `docs/variable_audit/*_construccion.csv` (uno por módulo,
producto de la auditoría de calidad documentada en `docs/decisions.md`).
Choques no tiene archivo de auditoría de este tipo (no pasó por el mismo
proceso columna-por-columna que Hogar/Personas/Comunidades/Niños), de ahí
el "--" en esa celda -- igual que en la tesis.

"Usadas en el modelo" (columna 3) tiene DOS definiciones distintas según
el módulo, ninguna inventada por este script:
  - Personas, Comunidades, Niños, Choques: es el conteo de columnas de
    contenido (sin llaves de identidad) que el módulo aporta al dataset
    consolidado final (`benchmark_consolidado_elca_longitudinal.parquet`)
    -- SÍ reproducible directamente de los datos, se recalcula abajo.
  - Hogar: el número publicado (103) NO es el conteo de columnas finales
    del bloque de features de Hogar (ese conteo es 56-61 según se cuenten
    o no columnas de identidad, ver `hogar_features_elca_longitudinal.parquet`).
    Es el número de candidatas RAW que se usan -directas o absorbidas en
    variables compuestas- documentado explícitamente en
    `docs/decisions.md` (sección "Bloque de features de Hogar", verificación
    final de cobertura: "de las 129 candidatas, 103 se usan (directas o en
    composites), 26 se excluyen con razón documentada"). No existe un
    artefacto de datos del que recalcular ese 103 de forma mecánica (la
    equivalencia candidata-raw -> feature-compuesta se decidió variable por
    variable durante la construcción, no quedó tabulada aparte) -- se deja
    hardcodeado aquí con esta cita como única fuente, en vez de fingir que
    se recalcula. Si se regenera esa auditoría, este valor debe actualizarse
    a mano.

INPUTS

    docs/variable_audit/hogar_construccion.csv
    docs/variable_audit/personas_hogar_construccion.csv
    docs/variable_audit/comunidades_construccion.csv
    docs/variable_audit/ninos_construccion.csv
    data/processed/benchmark_consolidado_elca_longitudinal.parquet

OUTPUTS

    paper/tables/tab_cobertura_modulos.tex

CÓMO CORRER

    python src/tabla_cobertura_modulos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "docs" / "variable_audit"
PROCESSED = REPO_ROOT / "data" / "processed"

# HALLAZGO 103 (Hogar) -- ver docstring arriba. Única fuente: docs/decisions.md,
# sección "Bloque de features de Hogar (build_hogar_features.py)",
# subsección "Verificación final de cobertura completa".
HOGAR_USADAS_EN_MODELO = 103
HOGAR_USADAS_FUENTE = (
    "docs/decisions.md, 'Bloque de features de Hogar': "
    "'de las 129 candidatas, 103 se usan (directas o en composites)'"
)

# Columnas de identidad/llave que no cuentan como covariable de contenido
# al recontar "usadas en el modelo" desde el consolidado (mismo criterio
# que build_benchmark_consolidado.py e src/02_build/eda_variables_modelo.py).
ID_COLS = {"consecutivo", "llave", "llave_n16", "ola", "zona", "llave_compuesta", "consecutivo_c"}

# Columnas del consolidado que pertenecen a cada módulo (mismo mapeo que
# src/02_build/eda_variables_modelo.py, MODULE_RANGES) -- se importa la
# lista por nombre de archivo fuente en vez de re-declararla para no
# desincronizar los dos scripts si el consolidado cambia.
FEATURE_PARQUETS_POR_MODULO = {
    "Personas (9 bloques)": [
        "personas_hogar_elca_longitudinal.parquet",
        "educacion_ocupacion_hogar_elca_longitudinal.parquet",
        "salud_discapacidad_hogar_elca_longitudinal.parquet",
        "ahorro_capital_social_hogar_elca_longitudinal.parquet",
        "educacion_ocupacion_hogar_ext_elca_longitudinal.parquet",
        "becas_subsidios_hogar_elca_longitudinal.parquet",
        "salud_discapacidad_hogar_ext_elca_longitudinal.parquet",
        "personas_hogar_ext_elca_longitudinal.parquet",
        "participacion_civica_hogar_elca_longitudinal.parquet",
    ],
    "Comunidades": ["comunidades_hogar_elca_longitudinal.parquet"],
    "Niños (6-9 años)": ["ninos_hogar_elca_longitudinal.parquet"],
    "Choques": ["choques_hogar_elca_longitudinal.parquet"],
}

CANDIDATAS_AUDITORIA = {
    "Hogar": AUDIT_DIR / "hogar_construccion.csv",
    "Personas (9 bloques)": AUDIT_DIR / "personas_hogar_construccion.csv",
    "Comunidades": AUDIT_DIR / "comunidades_construccion.csv",
    "Niños (6-9 años)": AUDIT_DIR / "ninos_construccion.csv",
    "Choques": None,
}

MODULO_COBERTURA_PANEL = {
    # Cobertura (% del panel) ya documentada/verificada en el texto de la
    # tesis (Sección "Estadísticas descriptivas"): Hogar/Personas/Choques
    # se preguntan a todo hogar (100%); Comunidades es comunidad x ola, no
    # hogar x ola (91.9-96.4%); Niños solo aplica a hogares con >=1 niño de
    # 6-9 años (55.4%, 15,473/27,932 -- ver eda_variables_modelo.py y
    # build_ninos_hogar.py). No se recalculan aquí por no ser el foco de
    # este script (que es "candidatas"/"usadas"); si se automatizan, deben
    # tomarse de las mismas fuentes ya citadas en el cuerpo del texto.
    "Hogar": "100\\%",
    "Personas (9 bloques)": "100\\%",
    "Comunidades": "91.9--96.4\\%",
    "Niños (6-9 años)": "55.4\\% (15{,}473 / 27{,}932)",
    "Choques": "100\\%",
}


def contar_candidatas(ruta: Path | None) -> str:
    if ruta is None:
        return "--"
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(ruta)
    n = int((df["clasificacion"] == "CANDIDATO_BENCHMARK").sum())
    return str(n)


def contar_usadas_en_modelo(modulo: str) -> int:
    if modulo == "Hogar":
        return HOGAR_USADAS_EN_MODELO
    total = 0
    for nombre_archivo in FEATURE_PARQUETS_POR_MODULO[modulo]:
        df = pd.read_parquet(PROCESSED / nombre_archivo)
        cols_contenido = [c for c in df.columns if c not in ID_COLS]
        total += len(cols_contenido)
    return total


def generar_tex(filas: list[dict]) -> str:
    lineas = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Cobertura de covariables por módulo en el dataset consolidado}",
        r"  \label{tab:cobertura_modulos}",
        r"  \small",
        r"  \begin{tabular}{lp{2.6cm}p{2.4cm}p{3.2cm}}",
        r"    \toprule",
        r"    \textbf{Módulo} & \textbf{Candidatas auditadas} & \textbf{Usadas en el modelo} & \textbf{Cobertura (\% del panel)} \\",
        r"    \midrule",
    ]
    for f in filas:
        lineas.append(f"    {f['modulo']:<12s} & {f['candidatas']} & {f['usadas']}  & {f['cobertura']} \\\\")
    lineas += [r"    \bottomrule", r"  \end{tabular}"]
    return "\n".join(lineas)


def main() -> None:
    modulos = ["Hogar", "Personas (9 bloques)", "Comunidades", "Niños (6-9 años)", "Choques"]
    filas = []
    for modulo in modulos:
        filas.append({
            "modulo": modulo,
            "candidatas": contar_candidatas(CANDIDATAS_AUDITORIA[modulo]),
            "usadas": contar_usadas_en_modelo(modulo),
            "cobertura": MODULO_COBERTURA_PANEL[modulo],
        })

    print("Candidatas auditadas y usadas en el modelo, por módulo:")
    for f in filas:
        print(f"  {f['modulo']:<22s} candidatas={f['candidatas']:>4s}  usadas={f['usadas']:>4}")
    print(f"\nNota sobre 'Hogar' -> usadas=103: {HOGAR_USADAS_FUENTE}")

    out_dir = REPO_ROOT / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = generar_tex(filas)
    ruta_tex = out_dir / "tab_cobertura_modulos.tex"
    ruta_tex.write_text(tex, encoding="utf-8")
    print(f"\nTabla exportada (cuerpo de tabular; encabezado table/caption/nota siguen a mano en main.tex): {ruta_tex}")
    print("\n" + tex)


if __name__ == "__main__":
    main()
