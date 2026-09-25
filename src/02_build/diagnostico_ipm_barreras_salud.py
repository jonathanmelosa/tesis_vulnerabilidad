"""
Diagnostico del indicador `priv_barreras_acceso_salud` del IPM, que no se
mide igual en las tres olas (ver `src/04_features/build_ipm_multidimensional.py`,
docstring punto 3): en 2013/2016 se usa la pregunta directa
`tratar_problema`; en 2010 un proxy (evento de salud sin hospitalizacion).

Hallazgo del 2026-09-24 (revisando por que la salida de la pobreza IPM
2010->2013 es tan alta): la prevalencia cae de ~50% a ~20% entre 2010 y
2013, y la mayoria de los hogares que salen de la pobreza IPM en esa
ventana lo hacen perdiendo justamente esta privacion. Se evaluo armonizar
el indicador con la pregunta directa, que SI existe en 2010
(`tratar_enfe`/`tratar_acci`/`tratar_odon`/`tratar_ciru`, mismas
categorias de respuesta que `tratar_problema`), pero la responde una
fraccion mucho menor de personas en 2010, lo que sugiere un filtro
distinto y deja una brecha de nivel. Decision del usuario (mismo dia): NO
armonizar (costo de regenerar todo el IPM y la brecha remanente);
documentarlo como limitacion en la Seccion de Limitaciones. Este script
produce las cifras citadas ahi y en el parrafo de transiciones IPM.

La version "armonizada" se calcula SOLO para este diagnostico: no se usa
en el IPM ni en ningun modelo.

INPUTS
    data/processed/ipm_multidimensional_elca_longitudinal.parquet
    data/processed/personas_elca_longitudinal_clean.parquet

OUTPUTS
    outputs/tables/ipm/diagnostico_barreras_salud.csv  (metrica, valor)

COMO CORRER
    python src/02_build/diagnostico_ipm_barreras_salud.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
IPM_PATH = PROCESSED / "ipm_multidimensional_elca_longitudinal.parquet"
PERSONAS_PATH = PROCESSED / "personas_elca_longitudinal_clean.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "ipm"

COLS_TRATAR_2010 = ["tratar_enfe", "tratar_acci", "tratar_odon", "tratar_ciru"]

# Respuestas que NO son atencion medica/odontologica profesional (misma
# lista que build_ipm_multidimensional.py). Se compara por PREFIJO para
# cubrir las variantes con caracteres corruptos de la fuente ("Se auto
# recet�" en 2010, "Us??? remedios caseros" en 2013).
PREFIJOS_NO_ATENCION = ("Se auto recet", "Nada", "Usó remedios", "Us??? remedios", "Acudió al boticario")


def _normalizar(serie: pd.Series) -> pd.Series:
    return serie.astype("string").str.split().str.join(" ")


def _respondio(serie: pd.Series) -> pd.Series:
    s = _normalizar(serie)
    return s.notna() & ~s.isin(["None", "nan", ""])


def _no_atencion(serie: pd.Series) -> pd.Series:
    s = _normalizar(serie)
    return (s.str.startswith(PREFIJOS_NO_ATENCION) & _respondio(serie)).fillna(False)


def barrera_armonizada_hogar(personas: pd.DataFrame) -> pd.Series:
    """Privacion por hogar x ola usando la pregunta directa en las tres
    olas (tratar_* en 2010, tratar_problema en 2013/2016)."""
    ola1 = personas["ola"] == 1
    barrera_2010 = pd.concat([_no_atencion(personas[c]) for c in COLS_TRATAR_2010], axis=1).any(axis=1)
    barrera_2013_16 = _no_atencion(personas["tratar_problema"])
    personas = personas.assign(barrera=barrera_2010.where(ola1, barrera_2013_16))
    return personas.groupby(["consecutivo", "ola"])["barrera"].max()


def cobertura_pregunta_directa(personas: pd.DataFrame) -> pd.Series:
    """% de personas que responden la pregunta directa, por ola."""
    ola1 = personas["ola"] == 1
    resp_2010 = pd.concat([_respondio(personas[c]) for c in COLS_TRATAR_2010], axis=1).any(axis=1)
    resp = resp_2010.where(ola1, _respondio(personas["tratar_problema"]))
    return resp.groupby(personas["ola"]).mean()


def salida_2010_2013_por_barrera(ipm: pd.DataFrame) -> tuple[int, float]:
    """Panel 2010->2013 emparejado por `consecutivo`, excluyendo hogares
    divididos (consecutivo repetido dentro de una ola) -- mismo universo
    que la matriz de transicion IPM (n = 8,218). Devuelve el numero de
    hogares que salen de la pobreza IPM y la fraccion de ellos que pierde
    `priv_barreras_acceso_salud` (1 en 2010 -> 0 en 2013)."""
    ipm = ipm[~ipm.duplicated(["consecutivo", "ola"], keep=False)]
    a = ipm[ipm["ola"] == 1].set_index("consecutivo")
    b = ipm[ipm["ola"] == 2].set_index("consecutivo")
    comunes = a.index.intersection(b.index)
    a, b = a.loc[comunes], b.loc[comunes]
    sale = (a["pobre_ipm"] == 1) & (b["pobre_ipm"] == 0)
    pierde = (a["priv_barreras_acceso_salud"] == 1) & (b["priv_barreras_acceso_salud"] == 0)
    return len(comunes), int(sale.sum()), float(pierde[sale].mean())


def main() -> None:
    ipm = pd.read_parquet(IPM_PATH)
    personas = pd.read_parquet(PERSONAS_PATH, columns=["consecutivo", "ola", "tratar_problema"] + COLS_TRATAR_2010)

    filas = []
    for ola, valor in ipm.groupby("ola")["priv_barreras_acceso_salud"].mean().items():
        filas.append((f"prevalencia_hogar_actual_ola{ola}", valor))
    for ola, valor in barrera_armonizada_hogar(personas).groupby("ola").mean().items():
        filas.append((f"prevalencia_hogar_armonizada_ola{ola}", valor))
    for ola, valor in cobertura_pregunta_directa(personas).items():
        filas.append((f"cobertura_personas_pregunta_directa_ola{ola}", valor))

    n_panel, n_salen, frac = salida_2010_2013_por_barrera(ipm)
    filas += [
        ("n_panel_2010_2013", n_panel),
        ("n_salen_ipm_2010_2013", n_salen),
        ("frac_salen_que_pierden_barrera_salud_2010_2013", frac),
    ]

    tabla = pd.DataFrame(filas, columns=["metrica", "valor"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta = OUTPUT_DIR / "diagnostico_barreras_salud.csv"
    tabla.to_csv(ruta, index=False)
    print(f"Guardado: {ruta}\n")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
