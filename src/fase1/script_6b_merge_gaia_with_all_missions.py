# script_6c_append_gaia_to_all_missions.py

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import gc

INPUT_DIR = Path("data/processed")
FILE_BASE = INPUT_DIR / "all_missions_labeled.parquet"
FILE_NEW = INPUT_DIR / "dataset_gaia_dr3_vsx_tic_labeled_with_coords_clean_complemented.parquet"
OUTPUT_FILE = INPUT_DIR / "all_missions_labeled_completo.parquet"
SUMMARY_FILE = INPUT_DIR / "summary" / "clase_variable_normalizada_summary_completo.csv"
SUMMARY_FILE.parent.mkdir(exist_ok=True)

NORMALIZATION_DICT = {
    "EB": "Eclipsing Binary", "EA": "Eclipsing Binary", "EW": "Eclipsing Binary",
    "EA/RS": "Eclipsing Binary", "EA|EB": "Eclipsing Binary", "EA:": "Eclipsing Binary",
    "ELL/RS": "Eclipsing Binary", "ELL/DW:": "Eclipsing Binary", "EB,": "Eclipsing Binary",
    "RRAB": "RR Lyrae", "RRC": "RR Lyrae", "RRAB/BL": "RR Lyrae", "RRAB:": "RR Lyrae",
    "RR": "RR Lyrae", "RRD": "RR Lyrae", "RRC|EC": "RR Lyrae", "ACEP": "RR Lyrae",
    "DSCT": "Delta Scuti", "DSCTC": "Delta Scuti", "DSCT|GDOR|SXPHE": "Delta Scuti",
    "DSCT|EC|ESD": "Delta Scuti", "DSCT:": "Delta Scuti",
    "ROT": "Rotational", "BY": "Rotational", "BY:": "Rotational", "ROT/WD": "Rotational",
    "RS": "Rotational", "RS:": "Rotational", "RS_CVn": "Rotational",
    "M": "Irregular", "L": "Irregular", "SR": "Irregular", "SR:": "Irregular",
    "SRB": "Irregular", "SRB:": "Irregular", "SRA": "Irregular", "SRS": "Irregular",
    "LPV": "Irregular", "LB": "Irregular", "SR|M": "Irregular", "LC": "Irregular",
    "CV": "Cataclysmic", "CV:": "Cataclysmic", "NL:": "Cataclysmic",
    "ZZA": "White Dwarf", "WD": "White Dwarf",
    "YSO": "Young Stellar Object", "T Tauri": "Young Stellar Object",
    "VAR": "Variable", "VAR:": "Variable", "UNKNOWN": "Unknown", "": "Unknown"
}

# Paso 1: obtener esquema base
print(f"📂 Leyendo esquema base desde: {FILE_BASE.name}")
base_dataset = ds.dataset(str(FILE_BASE), format="parquet")
schema_base = base_dataset.schema
all_columns = schema_base.names

# Detectar columnas a forzar como float64
print("🔍 Detectando columnas numéricas (float64)...")
float64_columns = set()
for field in schema_base:
    if pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
        float64_columns.add(field.name)

schema_final = schema_base
writer = pq.ParquetWriter(OUTPUT_FILE, schema=schema_final)
label_counter = Counter()
rows_written = 0

# Paso 2: copiar el contenido base directamente
print(f"📥 Copiando datos de {FILE_BASE.name} al nuevo fichero (sin modificar)...")
for batch in base_dataset.to_batches():
    writer.write_table(pa.Table.from_batches([batch], schema=schema_final))

# Paso 3: procesar y normalizar nuevo fichero
print(f"➕ Añadiendo datos nuevos desde: {FILE_NEW.name}")
new_dataset = ds.dataset(str(FILE_NEW), format="parquet")
scanner = new_dataset.scanner(batch_size=500)

for i, batch in enumerate(tqdm(scanner.to_batches(), desc="Procesando nuevos datos", unit="batch")):
    df = pa.Table.from_batches([batch]).to_pandas()

    df["clase_variable_normalizada"] = df["clase_variable"].map(NORMALIZATION_DICT).fillna("Other")
    df["source_dataset"] = FILE_NEW.name
    label_counter.update(df["clase_variable_normalizada"])
    rows_written += len(df)

    # Asegurar columnas del esquema
    for col in all_columns:
        if col not in df.columns:
            df[col] = None

    for col in float64_columns:
        if col in df.columns:
            try:
                df[col] = df[col].astype("float64")
            except Exception:
                df[col] = pd.NA

    df = df[all_columns]
    table = pa.Table.from_pandas(df, schema=schema_final, preserve_index=False)
    writer.write_table(table)
    del df, table, batch
    gc.collect()

writer.close()

# Paso 4: exportar resumen
print(f"✅ Dataset final guardado como: {OUTPUT_FILE}")
print(f"📊 Nuevas filas añadidas: {rows_written:,}")

df_summary = pd.DataFrame(label_counter.items(), columns=["clase_variable_normalizada", "Recuento"])
df_summary.sort_values(by="Recuento", ascending=False, inplace=True)
df_summary.to_csv(SUMMARY_FILE, index=False)
print(f"📄 Resumen de clases exportado: {SUMMARY_FILE.relative_to(INPUT_DIR.parent)}")
