# script_3a_clean_vsx_classes_south.py

import pandas as pd
from pathlib import Path
from src.utils.normalization_dict import normalize_label

INPUT_PARQUET = Path("data/processed/dataset_vsx_tic_labeled_south.parquet")
OUTPUT_PARQUET = Path("data/processed/dataset_vsx_tic_labeled_south_clean.parquet")
SUMMARY_CSV = Path("data/processed/summary/dataset_vsx_tic_labeled_south_clean_summary.csv")
SUMMARY_TXT = Path("data/processed/summary/dataset_vsx_tic_labeled_south_clean_summary_info.txt")

print(f"📥 Leyendo: {INPUT_PARQUET}")
df = pd.read_parquet(INPUT_PARQUET)
df["clase_variable_normalizada"] = df["clase_variable"].apply(normalize_label)
df.to_parquet(OUTPUT_PARQUET, index=False)

summary = df["clase_variable_normalizada"].value_counts()
summary.to_csv(SUMMARY_CSV)

with open(SUMMARY_TXT, "w") as f:
    for clase, count in summary.items():
        f.write(f"{clase}: {count}\n")

print(f"✅ Exportado: {OUTPUT_PARQUET}")
print(f"📊 Resumen: {SUMMARY_TXT}")
