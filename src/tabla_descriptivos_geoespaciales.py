"""
tabla_descriptivos_geoespaciales.py
=====================================
Genera las estadísticas descriptivas de las tres fuentes geoespaciales
nuevas (DMSP-OLS, ALOS PALSAR, Landsat 5 TM) para la ola 2010 -- la única
ola en la que las tres tienen datos reales simultáneamente (ver
Sección 3.3, Tabla de cobertura temporal: ALOS PALSAR y Landsat 5 TM no
tienen ninguna observación real en 2013 ni 2016; DMSP-OLS sí tiene datos
reales en 2013, pero se excluye aquí por decisión explícita de mantener
la comparación entre las tres fuentes simétrica dentro de una sola tabla).

QUÉ HACE

    1. Carga variables_geoespaciales_unificadas.parquet (output de
       unificacion_geoespacial/scripts/01_unificar_variables.py).
    2. Filtra a ola == 2010.
    3. Para cada fuente, calcula cobertura (n, %) y estadísticos
       descriptivos (media, std, min, max) de:
         a. La variable de ESTADO insignia (la misma que cada pipeline ya
            identifica como "variable clave" en sus propios reportes de
            control de calidad): dmsp_stable_lights, alos_hh_db, l5_ndvi.
         b. La TENDENCIA de la ventana acumulada de esa misma variable
            (dmsp_stable_lights_acum_tendencia, alos_hh_db_acum_tendencia,
            l5_ndvi_acum_tendencia).
    4. Adicionalmente, exporta un detalle COMPLETO (todas las variables de
       estado de las tres fuentes, no solo la insignia) a un reporte de
       texto aparte, para trazabilidad total -- la tabla de la tesis
       muestra solo el resumen compacto de (3).
    5. Exporta dos tablas .tex listas para incorporar al documento y un
       reporte de texto con el detalle completo.

INPUTS

    ../data/processed/SALE_13082026/variables_geoespaciales_unificadas.parquet

OUTPUTS

    paper/tables/tab_geo_descriptivos_2010.tex
    paper/tables/reporte_descriptivos_geoespaciales_2010.txt

CÓMO CORRER

    python tabla_descriptivos_geoespaciales.py
"""

# ── Librería estándar ──────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── Librerías externas ─────────────────────────────────────────────────────────
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "variables_geoespaciales": (
        REPO_ROOT / "data" / "processed" / "SALE_13082026" / "variables_geoespaciales_unificadas.parquet"
    ),
    "ola_objetivo": 2010,

    # Variable insignia de ESTADO por fuente -- la misma que cada pipeline
    # ya reporta como "variable clave" en su propio control de calidad.
    "variable_insignia": {
        "DMSP-OLS": "dmsp_stable_lights",
        "ALOS PALSAR": "alos_hh_db",
        "Landsat 5 TM": "l5_ndvi",
    },

    # Todas las variables de ESTADO por fuente (para el reporte de detalle
    # completo, no para la tabla compacta de la tesis).
    "variables_estado_completas": {
        "DMSP-OLS": ["dmsp_avg_vis", "dmsp_cf_cvg", "dmsp_stable_lights", "dmsp_log_stable_lights"],
        "ALOS PALSAR": [
            "alos_hh_db", "alos_hv_db", "alos_hh_hv_diff", "alos_hh_hv_ratio",
            "alos_hv_hh_ratio", "alos_indice_normalizado", "alos_log_hh_hv",
        ],
        "Landsat 5 TM": [
            "l5_sr_b1", "l5_sr_b2", "l5_sr_b3", "l5_sr_b4", "l5_sr_b5", "l5_sr_b7",
            "l5_ndvi", "l5_ndbi", "l5_bsi", "l5_evi", "l5_mndwi", "l5_ndwi", "l5_savi",
            "l5_tc_brillo", "l5_tc_humedad", "l5_tc_verdor",
        ],
    },

    "output_tables_dir": REPO_ROOT / "paper" / "tables",
}


def cargar_datos(cfg: dict) -> pd.DataFrame:
    """Lee variables_geoespaciales_unificadas.parquet y filtra a la ola objetivo."""
    ruta = Path(cfg["variables_geoespaciales"])
    if not ruta.exists():
        print(f"ERROR: no se encontró {ruta}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(ruta)
    df_ola = df[df["ola"] == cfg["ola_objetivo"]].copy()
    print(f"Cargado: {len(df):,} filas totales -- {len(df_ola):,} filas en ola {cfg['ola_objetivo']}")
    return df_ola


def calcular_descriptivos(serie: pd.Series, n_total: int) -> dict:
    """Calcula n, cobertura (%), media, std, min, max de una serie, ignorando NaN."""
    validos = serie.dropna()
    return {
        "n": len(validos),
        "cobertura_pct": 100 * len(validos) / n_total if n_total else float("nan"),
        "media": validos.mean() if len(validos) else float("nan"),
        "std": validos.std() if len(validos) else float("nan"),
        "min": validos.min() if len(validos) else float("nan"),
        "max": validos.max() if len(validos) else float("nan"),
    }


def construir_tabla_compacta(df: pd.DataFrame, cfg: dict) -> list:
    """Construye las filas de la tabla compacta (variable insignia: estado + tendencia acumulada)."""
    n_total = len(df)
    filas = []
    for fuente, var_estado in cfg["variable_insignia"].items():
        d_estado = calcular_descriptivos(df[var_estado], n_total)
        filas.append({"fuente": fuente, "tipo": "Estado", "variable": var_estado, **d_estado})

        var_tendencia = f"{var_estado}_acum_tendencia"
        if var_tendencia in df.columns:
            d_tend = calcular_descriptivos(df[var_tendencia], n_total)
            filas.append({"fuente": fuente, "tipo": "Ventana acumulada (tendencia)", "variable": var_tendencia, **d_tend})
    return filas


def generar_tex_tabla_compacta(filas: list, cfg: dict) -> str:
    """Genera el código LaTeX (booktabs) de la tabla compacta."""
    lineas = [
        r"\begin{table}",
        r"\caption{Estadísticas descriptivas de las variables insignia de DMSP-OLS, ALOS PALSAR y Landsat 5 TM, ola 2010 (única ola con datos reales simultáneos en las tres fuentes).}",
        r"\label{tab:geo_descriptivos_2010}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Fuente & Variable & n & Cobertura (\%) & Media & Std & Min & Max \\",
        r"\midrule",
    ]
    for f in filas:
        lineas.append(
            f"{f['fuente']} & {f['tipo']} & {f['n']:,} & {f['cobertura_pct']:.1f} & "
            f"{f['media']:.3f} & {f['std']:.3f} & {f['min']:.3f} & {f['max']:.3f} \\\\"
        )
    lineas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lineas)


def generar_reporte_completo(df: pd.DataFrame, cfg: dict) -> str:
    """Genera el reporte de texto con el detalle COMPLETO de todas las variables de estado, por fuente."""
    n_total = len(df)
    sep = "=" * 70
    lineas = [
        sep,
        f"REPORTE COMPLETO — ESTADÍSTICAS DESCRIPTIVAS GEOESPACIALES (OLA {cfg['ola_objetivo']})",
        sep,
        "",
        f"Hogares en la ola {cfg['ola_objetivo']}: {n_total:,}",
        "",
    ]
    for fuente, variables in cfg["variables_estado_completas"].items():
        lineas.append(f"── {fuente} — TODAS LAS VARIABLES DE ESTADO ──────────────")
        lineas.append("")
        for var in variables:
            if var not in df.columns:
                lineas.append(f"  {var}: NO ENCONTRADA EN EL ARCHIVO")
                continue
            d = calcular_descriptivos(df[var], n_total)
            lineas.append(
                f"  {var}: n={d['n']:,} ({d['cobertura_pct']:.1f}%)  "
                f"media={d['media']:.4f}  std={d['std']:.4f}  min={d['min']:.4f}  max={d['max']:.4f}"
            )
        lineas.append("")
    lineas.append(sep)
    return "\n".join(lineas)


def main():
    """Orquesta la carga, el cálculo y la exportación de tablas y reporte."""
    cfg = CONFIG
    out_dir = Path(cfg["output_tables_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = cargar_datos(cfg)

    filas_compactas = construir_tabla_compacta(df, cfg)
    tex_compacta = generar_tex_tabla_compacta(filas_compactas, cfg)
    ruta_tex = out_dir / "tab_geo_descriptivos_2010.tex"
    ruta_tex.write_text(tex_compacta, encoding="utf-8")
    print(f"Tabla compacta exportada: {ruta_tex}")

    reporte = generar_reporte_completo(df, cfg)
    ruta_reporte = out_dir / "reporte_descriptivos_geoespaciales_2010.txt"
    ruta_reporte.write_text(reporte, encoding="utf-8")
    print(f"Reporte completo exportado: {ruta_reporte}")

    print("\n" + tex_compacta)
    print("\n" + reporte)


if __name__ == "__main__":
    main()
