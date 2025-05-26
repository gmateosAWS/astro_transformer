# script_6c_normalize_gaiadr3.py (actualizado con diccionario centralizado)

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import gc
from src.utils.normalization_dict import normalize_label

INPUT_DIR = Path("data/processed")
FILE_BASE = INPUT_DIR / "all_missions_labeled.parquet"
FILE_GAIA = INPUT_DIR / "dataset_gaia_dr3_vsx_tic_labeled_with_coords_clean_complemented.parquet"
OUTPUT_PARQUET = INPUT_DIR / "dataset_gaia_complemented_normalized.parquet"
SUMMARY_FILE = INPUT_DIR / "summary" / "clase_variable_normalizada_summary_gaia.csv"
SUMMARY_FILE.parent.mkdir(exist_ok=True)

print("📥 Leyendo esquema desde all_missions_labeled.parquet")
base_schema = ds.dataset(str(FILE_BASE), format="parquet").schema
all_columns = base_schema.names
float64_columns = [f.name for f in base_schema if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)]

writer = pq.ParquetWriter(OUTPUT_PARQUET, schema=base_schema)
label_counter = Counter()
rows_written = 0

print("🔄 Procesando Gaia y normalizando clases...")
gaia_dataset = ds.dataset(str(FILE_GAIA), format="parquet")
scanner = gaia_dataset.scanner(batch_size=2000)

for i, batch in enumerate(tqdm(scanner.to_batches(), desc="Procesando Gaia", unit="batch")):
    df = pa.Table.from_batches([batch]).to_pandas()
    df["clase_variable_normalizada"] = df["clase_variable"].apply(normalize_label)
    df = df[df["clase_variable_normalizada"] != "Unknown"]
    df["source_dataset"] = FILE_GAIA.name
    label_counter.update(df["clase_variable_normalizada"])
    rows_written += len(df)

    missing_cols = [col for col in all_columns if col not in df.columns]
    if missing_cols:
        df = pd.concat([df, pd.DataFrame({col: pd.NA for col in missing_cols}, index=df.index)], axis=1)

    for col in float64_columns:
        if col in df.columns:
            try:
                df[col] = df[col].astype("float64")
            except Exception:
                df[col] = pd.NA

    df = df[all_columns]
    table = pa.Table.from_pandas(df, schema=base_schema, preserve_index=False)
    writer.write_table(table)

    del df, table, batch
    gc.collect()

writer.close()

print(f"✅ Dataset guardado como: {OUTPUT_PARQUET}")
print(f"📊 Total filas escritas: {rows_written:,}")

df_summary = pd.DataFrame(label_counter.items(), columns=["clase_variable_normalizada", "Recuento"])
df_summary.sort_values(by="Recuento", ascending=False, inplace=True)
df_summary.to_csv(SUMMARY_FILE, index=False)
print(f"📄 Resumen exportado: {SUMMARY_FILE.relative_to(INPUT_DIR.parent)}")
