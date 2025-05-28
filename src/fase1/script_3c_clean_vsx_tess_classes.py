# script_3c_clean_vsx_tess_classes.py
import pandas as pd
import os
import pyarrow.dataset as ds
from collections import Counter
from tqdm import tqdm
from datetime import datetime
import json
import csv
from src.utils.inspect_and_export_summary import inspect_and_export_summary

INPUT_PATH = "data/processed/dataset_vsx_tess_labeled.parquet"
OUTPUT_PATH = "data/processed/dataset_vsx_tess_labeled_clean.parquet"

SUMMARY_DIR = "data/processed/summary"

# Normalización de clases (reutilizable y extensible)
def normalizar_clase(clase):
    if not isinstance(clase, str):
        return "UNKNOWN"
    clase = clase.strip().upper()

    if clase in ["", "UNKNOWN", ","]:
        return "UNKNOWN"
    if "ROT" in clase:
        return "Rotational"
    if clase.startswith("RS"):
        return "RS_CVn"
    if clase.startswith("BY"):
        return "BY_Dra"
    if "DSCT" in clase:
        return "Delta_Scuti"
    if "RR" in clase:
        return "RR_Lyrae"
    if "EB" in clase or "EA" in clase or "ECLIPSING" in clase or clase.startswith("E"):
        return "Eclipsing"
    if "SR" in clase or "M" in clase or "LPV" in clase or "LB" in clase:
        return "Irregular"
    if "CV" in clase or "UG" in clase or "NL" in clase:
        return "Cataclysmic"
    if "WD" in clase:
        return "White_Dwarf"
    if "ACV" in clase:
        return "ACV"
    if "BCEP" in clase or "SPB" in clase:
        return "Beta_Cep"
    if "GDOR" in clase:
        return "Gamma_Dor"
    if "HADS" in clase:
        return "Delta_Scuti"
    if "S" == clase:
        return "Irregular"
    if "L" == clase:
        return "Irregular"
    if "VAR" in clase or "MISC" in clase:
        return "Irregular"
    if "YSO" in clase:
        return "YSO"
    if "WD" in clase:
        return "White_Dwarf"
    return "RARE"

# Limpieza del archivo y aplicación de la normalización
def limpiar_dataset():
    df = pd.read_parquet(INPUT_PATH)
    print(f"📥 Leídas {len(df)} filas desde {INPUT_PATH}")

    df["clase_variable"] = df["clase_variable"].fillna("UNKNOWN")
    df["clase_variable_normalizada"] = df["clase_variable"].apply(normalizar_clase)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Guardado dataset limpio en: {OUTPUT_PATH}")

    return OUTPUT_PATH

if __name__ == "__main__":
    path = limpiar_dataset()
    inspect_and_export_summary(path, output_format="csv")
