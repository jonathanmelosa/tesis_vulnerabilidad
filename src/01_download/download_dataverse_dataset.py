# %%
from __future__ import annotations
from datetime import datetime
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlencode

import requests


BASE_URL = "https://datahub.uniandes.edu.co"
TIMEOUT = 60

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


DATASETS = {
    "elca_2010": "doi:10.57924/DPF0M5", 
    "elca_2013": "doi:10.57924/DE9LP7", 
    "elca_2016": "doi:10.57924/BLUILW",
}

def safe_filename(name: str) -> str:
    """
    Limpia nombres de archivos para evitar problemas con caracteres especiales.
    """
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^\w\-.() ]", "_", name, flags=re.UNICODE)
    return name


def get_dataset_metadata(base_url: str, dataset_doi: str) -> dict:
    """
    Consulta los metadatos de un dataset en Dataverse usando su DOI.
    """
    url = f"{base_url}/api/datasets/:persistentId/?{urlencode({'persistentId': dataset_doi})}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "OK":
        raise RuntimeError(f"No se pudo obtener el dataset {dataset_doi}: {payload}")

    return payload["data"]


def extract_files_from_metadata(metadata: dict) -> list[dict]:
    """
    Extrae la lista de archivos disponibles dentro de los metadatos del dataset.
    """
    latest_version = metadata.get("latestVersion", {})
    files = latest_version.get("files", [])

    extracted = []
    for f in files:
        data_file = f.get("dataFile", {})
        extracted.append(
            {
                "id": data_file.get("id"),
                "filename": data_file.get("filename"),
                "description": f.get("description"),
                "contentType": data_file.get("contentType"),
                "filesize": data_file.get("filesize"),
                "md5": data_file.get("md5"),
                "persistentId": data_file.get("persistentId"),
            }
        )
    return extracted


def download_file_by_id(base_url: str, file_id: int, destination: Path) -> None:
    """
    Descarga un archivo individual usando el file id de Dataverse.
    """
    url = f"{base_url}/api/access/datafile/{file_id}"
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with open(destination, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)


def save_json(data: dict | list, path: Path) -> None:
    """
    Guarda un objeto JSON con codificación UTF-8 e indentación legible.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def download_dataset(dataset_name: str, dataset_doi: str) -> None:
    """
    Descarga todos los archivos de un dataset y los guarda en su carpeta propia.
    """
    output_dir = RAW_DATA_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Procesando {dataset_name} ===")
    print(f"DOI: {dataset_doi}")
    print(f"Carpeta de salida: {output_dir}")

    metadata = get_dataset_metadata(BASE_URL, dataset_doi)
    save_json(metadata, output_dir / "dataset_metadata.json")

    files = extract_files_from_metadata(metadata)
    save_json(files, output_dir / "files_manifest.json")

    if not files:
        print(f"No se encontraron archivos en {dataset_name}.")
        return

    print(f"Se encontraron {len(files)} archivos en {dataset_name}.")

    for i, file_info in enumerate(files, start=1):
        file_id = file_info["id"]
        filename = safe_filename(file_info["filename"] or f"file_{file_id}")
        destination = output_dir / filename

        if destination.exists():
            print(f"[{i}/{len(files)}] Ya existe, se omite: {filename}")
            continue

        print(f"[{i}/{len(files)}] Descargando: {filename}")
        try:
            download_file_by_id(BASE_URL, file_id, destination)
        except Exception as e:
            print(f"Error descargando {filename}: {e}")


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_doi in DATASETS.items():
        try:
            download_dataset(dataset_name, dataset_doi)
        except Exception as e:
            print(f"Error procesando {dataset_name} ({dataset_doi}): {e}")

    print("\nProceso terminado.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error general: {exc}")
        sys.exit(1)

download_info = {
    "dataset_name": dataset_name,
    "dataset_doi": dataset_doi,
    "download_timestamp": datetime.now().isoformat(),
}
save_json(download_info, output_dir / "download_info.json")


