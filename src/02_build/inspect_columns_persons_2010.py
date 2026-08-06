from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "elca_2010"

FILES = [
    "UPersonas.tab",
    "RPersonas.tab",
]

KEYWORDS = [
    "consecutivo",
    "llave",
    "orden",
    "edad",
    "sexo",
    "educ",
    "esc",
    "estudia",
    "asiste",
    "trab",
    "ocupa",
    "desoc",
    "ing",
    "salario",
    "labor",
    "fpers",
    "fexpers",
    "parentesco",
    "seguimiento",
]

def read_tab(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)

def main():
    for fname in FILES:
        path = RAW_DIR / fname
        print(f"\n===== {fname} =====")
        df = read_tab(path)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        matched = []
        for col in df.columns:
            for kw in KEYWORDS:
                if kw in col:
                    matched.append(col)
                    break

        matched = sorted(set(matched))
        print(f"Columnas encontradas ({len(matched)}):")
        for col in matched:
            print(col)

if __name__ == "__main__":
    main()