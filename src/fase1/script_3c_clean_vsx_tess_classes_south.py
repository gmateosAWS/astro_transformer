# script_3c_clean_vsx_tess_classes.py
import pandas as pd
import os
import pyarrow.dataset as ds
from collections import Counter
from tqdm import tqdm
from datetime import datetime
import json
import csv
from src.utils.normalization_dict import normalize_label
from src.utils.inspect_and_export_summary import inspect_and_export_summary

INPUT_PATH = "data/processed/dataset_vsx_tess_labeled_south.parquet"
OUTPUT_PATH = "data/processed/dataset_vsx_tess_labeled_south_clean.parquet"
SUMMARY_DIR = "data/processed/summary"

# Limpieza del archivo y aplicación de la normalización
def limpiar_dataset():
    df = pd.read_parquet(INPUT_PATH)
    print(f"📥 Leídas {len(df)} filas desde {INPUT_PATH}")

    df["clase_variable"] = df["clase_variable"].fillna("UNKNOWN")
    df["clase_variable_normalizada"] = df["clase_variable"].apply(normalize_label)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Guardado dataset limpio en: {OUTPUT_PATH}")

    return OUTPUT_PATH

if __name__ == "__main__":
    path = limpiar_dataset()
    inspect_and_export_summary(path, output_format="csv")
