"""
SUPERADO (2026-09-12) -- ver eda_dmsp_nacional_gee.py y eda_dmsp_fondo_mapa.py.
Este script usaba el dataset alternativo de Zenodo (ver mas abajo) por
punto-muestreo en centroides de municipio; se reemplazo por un raster
CONTINUO extraido de la misma fuente (Google Earth Engine) que ya genera
`dmsp_stable_lights` en el resto de la tesis, eliminando la advertencia de
escala "no intercambiable" que se documenta abajo. Se conserva este
archivo sin borrar por trazabilidad, pero no debe usarse para el mapa
territorial final -- ver docs/decisions.md, entrada 2026-09-12.

Iluminacion nocturna DMSP-OLS 2010 para TODOS los municipios de Colombia
(no solo los que tienen hogares ELCA) -- pedido del usuario (2026-09-11)
para que el mapa territorial muestre el "brillo" de luces nocturnas del
pais completo (como en la foto de referencia de luces nocturnas reales),
en vez de solo los ~800 hogares de la muestra ELCA con dato de DMSP.

Fuente: "A consistent and corrected nighttime light dataset (CCNL
1992-2013) from DMSP-OLS data" (Zenodo, DOI 10.5281/zenodo.6644980,
archivo CCNL_DMSP_2010_V1.tif, ~369MB, EPSG:4326, cobertura global,
descarga publica sin registro -- a diferencia de la fuente original EOG/
NOAA que ahora exige login). MISMO SENSOR (DMSP-OLS, F18 2010) que ya usa
el resto de la tesis para dmsp_stable_lights, pero un producto
"corregido y armonizado" distinto al procesamiento original de NOAA NGDC
usado en `variables_geoespaciales_unificadas.parquet` -- los valores NO
son identicos punto a punto, son comparables en magnitud/patron pero no
intercambiables 1 a 1. Se documenta esta diferencia explicitamente aqui
y en el script del mapa final para no mezclar fuentes sin aclararlo.

Metodo: en vez de cruzar por poligono de municipio (GADM nivel 2 no trae
codigo DIVIPOLA -- ver commit de graf_mapa_territorial_4_grupos.py, campo
CC_2 vacio en las 1,119 filas), se extrae el valor del raster DIRECTAMENTE
en el centroide de cada uno de los 1,122 municipios de la tabla oficial
DIVIPOLA (data/interim/geo_referencia/divipola_municipios_coordenadas.csv,
DANE, agosto 2026) -- un punto por municipio, sin ningun cruce por nombre.

INPUTS
    /tmp/dmsp_global/CCNL_DMSP_2010_V1.tif (raster, NO versionado en el repo
        por tamano -- ver COMO CORRER para volver a descargarlo)
    data/interim/geo_referencia/divipola_municipios_coordenadas.csv

OUTPUTS
    data/processed/dmsp_nacional_municipios_2010.csv
        (cod_mpio, nombre_mpio, nombre_dpto, longitud, latitud, dmsp_2010)

COMO CORRER
    # 1. Descargar el raster (una sola vez, ~189MB comprimido):
    mkdir -p /tmp/dmsp_global
    curl -s -o /tmp/dmsp_global/CCNL_DMSP_2010_V1.tif \\
        "https://zenodo.org/api/records/6644980/files/CCNL_DMSP_2010_V1.tif/content"
    # 2. Correr este script:
    cd src/02_build && python eda_dmsp_nacional_municipios.py
"""

from pathlib import Path

import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIVIPOLA_PATH = PROJECT_ROOT / "data" / "interim" / "geo_referencia" / "divipola_municipios_coordenadas.csv"
RASTER_PATH = Path("/tmp/dmsp_global/CCNL_DMSP_2010_V1.tif")
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dmsp_nacional_municipios_2010.csv"


def main() -> None:
    if not RASTER_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el raster en {RASTER_PATH}. "
            "Descargarlo primero (ver docstring del modulo, seccion COMO CORRER)."
        )
    if not DIVIPOLA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro la tabla DIVIPOLA en {DIVIPOLA_PATH}. "
            "Ver src/02_build/ (descarga de DIVIPOLA_Municipios.xlsx del DANE)."
        )

    divipola = pd.read_csv(DIVIPOLA_PATH)
    print(f"Municipios DIVIPOLA: {len(divipola)}")

    with rasterio.open(RASTER_PATH) as src:
        print(f"Raster: {src.width}x{src.height}, bounds={src.bounds}, crs={src.crs}")
        assert str(src.crs) == "EPSG:4326", f"CRS inesperado: {src.crs} (se esperaba EPSG:4326)"

        coords = list(zip(divipola["longitud"], divipola["latitud"]))
        valores = [v[0] for v in src.sample(coords)]

    divipola["dmsp_2010"] = valores
    n_fuera_de_rango = divipola["dmsp_2010"].isna().sum()
    if n_fuera_de_rango:
        print(f"ADVERTENCIA: {n_fuera_de_rango} municipios sin valor de raster "
              "(coordenada fuera de cobertura o en un pixel nulo) -- revisar manualmente.")

    print(divipola[["nombre_dpto", "nombre_mpio", "dmsp_2010"]].describe())
    print()
    print("Top 10 municipios por DMSP 2010:")
    print(divipola.nlargest(10, "dmsp_2010")[["nombre_dpto", "nombre_mpio", "dmsp_2010"]].to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    divipola.to_csv(OUTPUT_PATH, index=False)
    print(f"\nGuardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
