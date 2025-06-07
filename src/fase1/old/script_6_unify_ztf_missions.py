import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import pandas as pd
import gc
from tqdm import tqdm

INPUT_DIR = Path("data/processed")
FILES = [
    "dataset_ztf_labeled_1.parquet",
    "dataset_ztf_labeled_2.parquet"
]
OUTPUT_FILE = INPUT_DIR / "dataset_ztf_labeled.parquet"
GLOBAL_SCHEMA_FILE = INPUT_DIR / "all_missions_labeled.parquet"

# 1. Leer el esquema global
print("🔎 Leyendo esquema global de all_missions_labeled.parquet...", flush=True)
global_dataset = ds.dataset(str(GLOBAL_SCHEMA_FILE), format="parquet")
schema_final = global_dataset.schema
all_columns = schema_final.names

# Detectar columnas que deben ser float64 (por consistencia)
float64_columns = [f.name for f in schema_final if pa.types.is_float64(f.type)]

# 2. Unificar los dos parquets de ZTF con progreso
writer = None
first_batch = True
total_rows = 0

for fname in tqdm(FILES, desc="Archivos ZTF"):
    path = INPUT_DIR / fname
    print(f"\n📂 Procesando: {fname}", flush=True)
    dataset = ds.dataset(str(path), format="parquet")
    batches = list(dataset.to_batches())
    batch_bar = tqdm(batches, desc=f"Batches en {fname}", leave=False)
    for batch in batch_bar:
        df_batch = pa.Table.from_batches([batch]).to_pandas()

        # Añadir columnas faltantes
        for col in all_columns:
            if col not in df_batch.columns:
                df_batch[col] = None

        # Ajustar tipos float64
        for col in float64_columns:
            if col in df_batch.columns:
                try:
                    df_batch[col] = df_batch[col].astype("float64")
                except Exception:
                    df_batch[col] = pd.NA

        # Ordenar columnas
        df_batch = df_batch[all_columns]
        table_out = pa.Table.from_pandas(df_batch, schema=schema_final, preserve_index=False)

        if first_batch:
            writer = pq.ParquetWriter(OUTPUT_FILE, schema_final)
            first_batch = False

        writer.write_table(table_out)
        total_rows += len(df_batch)
        batch_bar.set_postfix_str(f"Total filas: {total_rows:,}")
        del df_batch, table_out, batch
        gc.collect()

if writer:
    writer.close()
    print(f"\n✅ Dataset ZTF unificado guardado como: {OUTPUT_FILE} ({total_rows} filas)", flush=True)
else:
    print("❌ No se consolidó ningún dataset.")