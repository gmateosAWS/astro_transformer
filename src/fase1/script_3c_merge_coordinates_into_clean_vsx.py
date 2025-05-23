# script_3c_merge_coordinates_into_clean_vsx.py

import pandas as pd
from pathlib import Path
import sys

# Archivos de entrada y salida
CLEAN_PATH = Path("data/processed/dataset_vsx_tic_labeled_clean.parquet")
ORIGINAL_PATH = Path("data/processed/dataset_vsx_tic_labeled.parquet")
OUTPUT_PATH = Path("data/processed/dataset_vsx_tic_labeled_clean_with_coords.parquet")

# Cargar datasets
print(f"📂 Cargando dataset limpio: {CLEAN_PATH.name}")
df_clean = pd.read_parquet(CLEAN_PATH)

print(f"📂 Cargando dataset original: {ORIGINAL_PATH.name}")
df_orig = pd.read_parquet(ORIGINAL_PATH)
print(f"🧾 Columnas del original: {df_orig.columns.tolist()}")

# 🔧 Normalizar 'id_objeto'
def clean_id(series):
    return "TIC_" + series.astype(str).str.extract(r"TIC_(\d+)", expand=False).fillna("MISSING")

df_clean["id_objeto"] = clean_id(df_clean["id_objeto"])
df_orig["id_objeto"] = clean_id(df_orig["id_objeto"])

# ❌ Eliminar posibles columnas antiguas de coordenadas para evitar _x / _y
cols_to_drop = [col for col in df_clean.columns if col.lower() in ["ra", "dec", "tic_ra", "tic_dec"]]
if cols_to_drop:
    print(f"🧹 Eliminando columnas conflictivas del limpio: {cols_to_drop}")
    df_clean.drop(columns=cols_to_drop, inplace=True)

# Seleccionar coordenadas
coord_pairs = [("tic_ra", "tic_dec"), ("ra", "dec")]
selected_pair = next(((ra, dec) for ra, dec in coord_pairs if ra in df_orig.columns and dec in df_orig.columns), None)

if not selected_pair:
    print("❌ No se encontraron columnas válidas de coordenadas en el dataset original.")
    sys.exit(1)

ra_col, dec_col = selected_pair
print(f"🧭 Usando coordenadas: {ra_col} → ra, {dec_col} → dec")

# Preparar coordenadas
df_coords = df_orig[["id_objeto", ra_col, dec_col]].drop_duplicates("id_objeto")
df_coords = df_coords.rename(columns={ra_col: "ra", dec_col: "dec"})

# Merge limpio
df_merged = pd.merge(df_clean, df_coords, on="id_objeto", how="left")

# Verificación
print(f"✅ Columnas finales en df_merged: {df_merged.columns.tolist()}")
missing_coords = df_merged[["ra", "dec"]].isnull().any(axis=1).sum()
print(f"⚠️ Filas sin coordenadas tras merge: {missing_coords}")

# Guardar
df_merged.to_parquet(OUTPUT_PATH, index=False)
print(f"✅ Guardado con coordenadas: {OUTPUT_PATH} ({len(df_merged)} filas)")
