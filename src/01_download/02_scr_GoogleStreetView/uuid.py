import uuid, json
from pathlib import Path

ruta_mapa = Path(r"\\ECON-E420004947\Users\DANE 3\Documents\Usuarios\Jonathan Melo\SALE\mapa_photo_id.json")

if ruta_mapa.exists():
    with open(ruta_mapa, "r") as f:
        mapa_ids = json.load(f)
else:
    mapa_ids = {}

def generar_photo_id_uuid(pano_id, heading):
    clave = f"{pano_id}_{heading}"
    if clave not in mapa_ids:
        mapa_ids[clave] = uuid.uuid4().hex[:12]
    return mapa_ids[clave]

base["photo_id"] = base.apply(
    lambda row: generar_photo_id_uuid(row["pano_id"], row["heading"]), axis=1
)
base["n_repeticiones"] = base.groupby("photo_id")["photo_id"].transform("count")

with open(ruta_mapa, "w") as f:
    json.dump(mapa_ids, f)

# Solo se eliminan los identificadores de la FOTO, no del hogar
base = base.drop(columns=["pano_id", "nombre_archivo", "heading"])
base