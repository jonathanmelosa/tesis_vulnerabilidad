source .venv/bin/activatefrom pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# 👇 CAMBIA esto cuando quieras explorar otra ronda
ROUND_NAME = "elca_2010"


def list_files(folder: Path):
    return [p for p in folder.iterdir() if p.is_file()]


def read_data(path: Path):
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tab":
            return pd.read_csv(path, sep="\t")
        if suffix == ".dta":
            return pd.read_stata(path)
        if suffix == ".sav":
            return pd.read_spss(path)
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
    except Exception as e:
        print(f"❌ Error leyendo {path.name}: {e}")
        return None

    print(f"⚠️ Formato no soportado: {path.name}")
    return None


def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def inspect_file(path: Path):
    print(f"\n===== {path.name} =====")

    df = read_data(path)

    if df is None:
        return

    df = clean_columns(df)

    print(f"Shape: {df.shape}")
    print("Columnas:")
    print(df.columns.tolist()[:30])  # primeras 30

    print("\nPrimeras filas:")
    print(df.head(2))


def main():
    folder = RAW_DIR / ROUND_NAME

    if not folder.exists():
        print(f"❌ No existe la carpeta: {folder}")
        return

    files = list_files(folder)

    print(f"\n📂 Explorando: {ROUND_NAME}")
    print(f"Archivos encontrados: {len(files)}\n")

    for path in files:
        inspect_file(path)


if __name__ == "__main__":
    main()
