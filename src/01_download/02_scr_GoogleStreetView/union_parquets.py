import pandas as pd
from pathlib import Path

carpeta_embeddings = Path(r"\\ECON-E420004947\Users\DANE 3\Documents\Usuarios\Jonathan Melo\SALE\SALE_04082026\gsv\embeddings")

archivos = {
    "clip": carpeta_embeddings / "embeddings_clip.parquet",
    "vgg19": carpeta_embeddings / "embeddings_vgg19.parquet",
    "resnet50": carpeta_embeddings / "embeddings_resnet50.parquet",
    "places365": carpeta_embeddings / "embeddings_places365.parquet",
}

cols_llave = ["consecutivo", "ola", "llave", "llave_n16"]

dfs = {nombre: pd.read_parquet(ruta) for nombre, ruta in archivos.items()}

# --- Verificación previa: filas y llaves únicas por archivo ---
print("Filas y llaves únicas por archivo:")
for nombre, df in dfs.items():
    n_filas = len(df)
    n_llaves_unicas = df[cols_llave].drop_duplicates().shape[0]
    dup = n_filas - n_llaves_unicas
    print(f"  {nombre:12s} filas={n_filas:6d}  llaves_unicas={n_llaves_unicas:6d}  "
          f"{'⚠ duplicados=' + str(dup) if dup else 'OK, sin duplicados'}")

# --- Verificación de tipos de dato en las columnas llave (causa común de fallos de merge) ---
print("\nTipos de dato por archivo:")
for nombre, df in dfs.items():
    print(f"  {nombre:12s}", {c: str(df[c].dtype) for c in cols_llave})

# Base: identificadores de hogar + identificadores de foto
base = dfs["clip"][cols_llave + ["pano_id", "heading", "nombre_archivo"]].copy()
n_base_inicial = len(base)
print(f"\nFilas base inicial (CLIP): {n_base_inicial}")

# Une los embeddings de cada modelo, con verificación de filas tras cada merge
for nombre, df in dfs.items():
    col_emb = [c for c in df.columns if "embedding" in c.lower() and c != "embedding_dim"]
    df_emb = df[cols_llave + col_emb].rename(columns={c: f"{c}_{nombre}" for c in col_emb})

    n_antes = len(base)
    base = base.merge(df_emb, on=cols_llave, how="outer", indicator=True)

    n_despues = len(base)
    solo_izq = (base["_merge"] == "left_only").sum()
    solo_der = (base["_merge"] == "right_only").sum()
    ambos = (base["_merge"] == "both").sum()
    base = base.drop(columns="_merge")

    estado = "OK" if n_despues == max(n_antes, len(df_emb)) else "⚠ REVISAR — el merge generó más filas de las esperadas (posible duplicado en la llave)"
    print(f"\nMerge con {nombre}: antes={n_antes}  después={n_despues}  "
          f"solo_base={solo_izq}  solo_{nombre}={solo_der}  coinciden={ambos}  -> {estado}")

print(f"\nFilas finales tras todos los merges: {len(base)}")
base