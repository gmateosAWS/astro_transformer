# script_6d_normalize_gaia_only.py (versión ampliada y optimizada)

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import gc

INPUT_DIR = Path("data/processed")
FILE_BASE = INPUT_DIR / "all_missions_labeled.parquet"
FILE_GAIA = INPUT_DIR / "dataset_gaia_dr3_vsx_tic_labeled_with_coords_clean_complemented.parquet"
OUTPUT_PARQUET = INPUT_DIR / "dataset_gaia_complemented_normalized.parquet"
SUMMARY_FILE = INPUT_DIR / "summary" / "clase_variable_normalizada_summary_gaia.csv"
SUMMARY_FILE.parent.mkdir(exist_ok=True)

NORMALIZATION_DICT = {
    # Eclipsing Binaries
    "EB": "Eclipsing Binary", "EA": "Eclipsing Binary", "EW": "Eclipsing Binary", "E": "Eclipsing Binary",
    "EC": "Eclipsing Binary", "EA/RS": "Eclipsing Binary", "EA|EB": "Eclipsing Binary", "EA:": "Eclipsing Binary",
    "ELL/RS": "Eclipsing Binary", "ELL/DW:": "Eclipsing Binary", "EB,": "Eclipsing Binary", "EW/RS": "Eclipsing Binary",
    "EA+BY+UV": "Eclipsing Binary", "EC|ESD": "Eclipsing Binary", "EC|RRC|ESD": "Eclipsing Binary",
    "EC|BCEP|DSCT|DSCTr|ESD": "Eclipsing Binary", "CW-FU|EC": "Eclipsing Binary", "ESD|CW-FU|EC": "Eclipsing Binary",
    "ESD|CW-FU|CW-FO|EC": "Eclipsing Binary", "ESD|EC": "Eclipsing Binary",

    # RR Lyrae
    "RRAB": "RR Lyrae", "RRAB/BL": "RR Lyrae", "RRAB/BL:": "RR Lyrae", "RRC": "RR Lyrae", "RRC|EC": "RR Lyrae",
    "RR": "RR Lyrae", "RRD": "RR Lyrae", "ACEP": "RR Lyrae",

    # Delta Scuti
    "DSCT": "Delta Scuti", "DSCT:": "Delta Scuti", "DSCTC": "Delta Scuti", "DSCT|GDOR|SXPHE": "Delta Scuti",
    "DSCT|EC|ESD": "Delta Scuti", "BCEP|DSCT": "Delta Scuti",

    # Rotational
    "ROT": "Rotational", "RS": "Rotational", "RS:": "Rotational", "RS_CVn": "Rotational", "BY": "Rotational",
    "BY:": "Rotational", "UV": "Rotational", "ROAP": "Rotational", "ACV": "Rotational", "ACV:": "Rotational",
    "ACV|roAm|roAp|SXARI": "Rotational",

    # Irregular
    "M": "Irregular", "L": "Irregular", "SR": "Irregular", "SR:": "Irregular", "SRB": "Irregular", "SRB:": "Irregular",
    "SRA": "Irregular", "SRS": "Irregular", "S": "Irregular", "SR|M": "Irregular", "LC": "Irregular",
    "LB": "Irregular", "LPV": "Irregular", "SRD": "Irregular",

    # Cataclysmic
    "CV": "Cataclysmic", "CV:": "Cataclysmic", "NL:": "Cataclysmic", "UG": "Cataclysmic", "UGSU+E": "Cataclysmic",
    "UGZ": "Cataclysmic", "UGWZ": "Cataclysmic",

    # White Dwarf
    "ZZA": "White Dwarf", "WD": "White Dwarf",

    # Young Stellar Object
    "YSO": "Young Stellar Object", "T Tauri": "Young Stellar Object",

    # Variable
    "VAR": "Variable", "VAR:": "Variable", "V1093HER": "Variable", "CST": "Variable", "EP": "Variable",
    "BCEP+SPB": "Variable", "V361HYA": "Variable", "MISC": "Variable",

    # Other
    "ED": "Other", "roAp": "Other", "S: ": "Other", "ESD|ACV|ED": "Other", "ACEP": "Other"
}

print("📥 Leyendo esquema desde all_missions_labeled.parquet")
base_schema = ds.dataset(str(FILE_BASE), format="parquet").schema
all_columns = base_schema.names
float64_columns = [f.name for f in base_schema if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)]

writer = pq.ParquetWriter(OUTPUT_PARQUET, schema=base_schema)
label_counter = Counter()
rows_written = 0

print("🔄 Procesando Gaia y normalizando clases...")
gaia_dataset = ds.dataset(str(FILE_GAIA), format="parquet")
scanner = gaia_dataset.scanner(batch_size=500)

for i, batch in enumerate(tqdm(scanner.to_batches(), desc="Procesando Gaia", unit="batch")):
    df = pa.Table.from_batches([batch]).to_pandas()
    df["clase_variable_normalizada"] = df["clase_variable"].map(NORMALIZATION_DICT).fillna("Other")
    df["source_dataset"] = FILE_GAIA.name
    label_counter.update(df["clase_variable_normalizada"])
    rows_written += len(df)

    # Añadir columnas que faltan en bloque para evitar fragmentación
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
