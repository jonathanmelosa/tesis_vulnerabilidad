from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "elca_2010"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "variable_audit" / "elca_2010_core"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "UHogar": "UHogar.tab",
    "RHogar": "RHogar.tab",
    "UPersonas": "UPersonas.tab",
    "RPersonas": "RPersonas-csv.tab",
}

def read_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".tab":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".dta":
        return pd.read_stata(path)
    raise ValueError(f"Formato no soportado: {path.name}")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df

def summarize_variable(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    sample_values = s.dropna().unique()[:5]
    sample_values = [str(x) for x in sample_values]

    out = {
        "variable": col,
        "dtype": str(s.dtype),
        "n_obs": len(s),
        "n_missing": int(s.isna().sum()),
        "pct_missing": round(float(s.isna().mean()), 4),
        "n_unique": int(s.nunique(dropna=True)),
        "sample_values": " | ".join(sample_values),
        "min": None,
        "max": None,
    }

    if pd.api.types.is_numeric_dtype(s):
        out["min"] = s.min()
        out["max"] = s.max()

    return out

def audit_files():
    all_columns = {}
    registry_rows = []

    for short_name, filename in FILES.items():
        path = RAW_DIR / filename
        print(f"\nAuditando {short_name}: {filename}")

        df = read_data(path)
        df = clean_columns(df)

        registry_rows.append({
            "dataset": short_name,
            "file_name": filename,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        })

        all_columns[short_name] = set(df.columns)

        summary = [summarize_variable(df, col) for col in df.columns]
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(OUTPUT_DIR / f"audit_{short_name}.csv", index=False)

    registry_df = pd.DataFrame(registry_rows)
    registry_df.to_csv(OUTPUT_DIR / "file_registry_core_2010.csv", index=False)

    all_vars = sorted(set().union(*all_columns.values()))
    presence_rows = []

    for var in all_vars:
        row = {"variable": var}
        for short_name in FILES.keys():
            row[short_name] = int(var in all_columns[short_name])
        presence_rows.append(row)

    presence_df = pd.DataFrame(presence_rows)
    presence_df.to_csv(OUTPUT_DIR / "variable_presence_core_2010.csv", index=False)

    print("\nListo. Archivos guardados en:")
    print(OUTPUT_DIR)

if __name__ == "__main__":
    audit_files()