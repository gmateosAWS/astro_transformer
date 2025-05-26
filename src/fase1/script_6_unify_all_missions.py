# script_6_unify_all_missions.py (actualizado con normalización centralizada)

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import pandas as pd
import gc
from collections import defaultdict
from src.utils.normalization_dict import normalize_label

INPUT_DIR = Path("data/processed")
OUTPUT_FILE = INPUT_DIR / "all_missions_labeled.parquet"
SUMMARY_FILE = INPUT_DIR / "summary" / "clase_variable_normalizada_summary.csv"
SUMMARY_FILE.parent.mkdir(exist_ok=True)

FILES = [
    "dataset_eb_kepler_labeled_fixed.parquet",
    "dataset_eb_tess_labeled_fixed.parquet",
    "dataset_k2varcat_labeled_fixed.parquet",
    "dataset_vsx_tess_labeled_fixed.parquet"
    #"dataset_gaia_dr3_vsx_tic_labeled.parquet"
]

REQUIRED_COLS = ["id_objeto", "tiempo", "magnitud", "clase_variable"]
WRITER = None
first_batch = True
all_normalized_labels = []

print("🔎 Detectando tipos reales por columna...", flush=True)
column_types = defaultdict(set)

# Paso 1: detectar tipos reales por columna en todos los ficheros
for fname in FILES:
    path = INPUT_DIR / fname
    dataset = ds.dataset(str(path), format="parquet")
    for batch in dataset.to_batches():
        df_sample = pa.Table.from_batches([batch]).to_pandas()
        df_sample["clase_variable_normalizada"] = df_sample["clase_variable"].apply(normalize_label)
        df_sample["source_dataset"] = fname
        for col in df_sample.columns:
            column_types[col].add(str(df_sample[col].dropna().dtype))
        break

# Paso 2: crear esquema unificado basado en tipos comunes
print("🧩 Inferiendo tipos compatibles por columna", flush=True)
pyarrow_type_map = {
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "object": pa.string(),
    "string": pa.string(),
    "datetime64[ns]": pa.timestamp("ns")
}

schema_fields = []
float64_columns = set()
for col, dtypes in column_types.items():
    if "object" in dtypes or "string" in dtypes:
        schema_fields.append(pa.field(col, pa.string()))
    elif "float64" in dtypes or "int64" in dtypes:
        schema_fields.append(pa.field(col, pa.float64()))
        float64_columns.add(col)
    elif "bool" in dtypes:
        schema_fields.append(pa.field(col, pa.bool_()))
    elif "datetime64[ns]" in dtypes:
        schema_fields.append(pa.field(col, pa.timestamp("ns")))
    else:
        schema_fields.append(pa.field(col, pa.string()))

schema_final = pa.schema(schema_fields)
all_columns = schema_final.names
print(f"✅ Esquema global construido con {len(all_columns)} columnas\n", flush=True)

# Paso 3: consolidar ficheros
for fname in FILES:
    path = INPUT_DIR / fname
    print(f"📂 Procesando: {fname}", flush=True)
    dataset = ds.dataset(str(path), format="parquet")
    schema = dataset.schema

    missing = [col for col in REQUIRED_COLS if col not in schema.names]
    if missing:
        print(f"⚠️ {fname} no contiene columnas requeridas: {missing}. Se omitirá.", flush=True)
        continue

    for batch in dataset.to_batches():
        df_batch = pa.Table.from_batches([batch]).to_pandas()
        df_batch["clase_variable_normalizada"] = df_batch["clase_variable"].apply(normalize_label)
        df_batch = df_batch[df_batch["clase_variable_normalizada"] != "Unknown"]
        df_batch["source_dataset"] = fname
        all_normalized_labels.extend(df_batch["clase_variable_normalizada"].tolist())

        for col in all_columns:
            if col not in df_batch.columns:
                df_batch[col] = None

        # 🔧 Conversión preventiva para columnas marcadas como float64
        for col in float64_columns:
            if col in df_batch.columns:
                try:
                    df_batch[col] = df_batch[col].astype("float64")
                except Exception:
                    df_batch[col] = pd.NA

        df_batch = df_batch[all_columns]
        table_out = pa.Table.from_pandas(df_batch, schema=schema_final, preserve_index=False)

        if first_batch:
            WRITER = pq.ParquetWriter(OUTPUT_FILE, schema_final)
            first_batch = False

        WRITER.write_table(table_out)
        del df_batch, table_out, batch
        gc.collect()

if WRITER:
    WRITER.close()
    print(f"✅ Dataset final guardado como: {OUTPUT_FILE}", flush=True)
    df_summary = pd.Series(all_normalized_labels).value_counts().reset_index()
    df_summary.columns = ["clase_variable_normalizada", "Recuento"]
    df_summary.to_csv(SUMMARY_FILE, index=False)
    print(f"📄 Resumen de clases exportado: {SUMMARY_FILE.relative_to(INPUT_DIR.parent)}")
else:
    print("❌ No se consolidó ningún dataset. Todos fueron omitidos.")
