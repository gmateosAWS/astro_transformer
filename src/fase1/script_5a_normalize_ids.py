# script_5a_normalize_ids.py

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import re
from multiprocessing import Pool, cpu_count
import argparse
import gc
import shutil

# Configuración de ficheros
DATASETS = {
    "kepler": "dataset_eb_kepler_labeled.parquet",
    "tess": "dataset_eb_tess_labeled.parquet",
    "k2": "dataset_k2varcat_labeled.parquet",
    "vsx_tess": "dataset_vsx_tess_labeled.parquet"
}

INPUT_DIR = Path("data/processed")
TEMP_DIR = Path("data/processed/normalized_temp")
OUTPUT_DIR = Path("data/processed")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Formatos esperados por misión
PREFIX_MAP = {
    "kepler": "KIC",
    "k2": "EPIC",
    "tess": "TIC",
    "vsx_tess": "TIC"
}

# Expresión robusta para extraer el ID numérico
def extract_numeric_id(raw):
    match = re.search(r"(\d+)", str(raw))
    return match.group(1) if match else None

# Procesa un dataset y unifica los batches
def procesar_dataset(mision, max_batches=None):
    try:
        input_path = INPUT_DIR / DATASETS[mision]
        output_parquet = OUTPUT_DIR / (input_path.stem + "_fixed.parquet")
        batch_prefix = TEMP_DIR / f"{input_path.stem}_batch"
        prefix = PREFIX_MAP[mision]

        print(f"\n🔧 Procesando {mision.upper()} → {input_path.name}", flush=True)
        dataset = ds.dataset(str(input_path), format="parquet")
        schema = dataset.schema

        id_field = "id_objeto"
        if id_field not in schema.names:
            print(f"❌ {input_path.name} no tiene columna 'id_objeto'", flush=True)
            return

        batch_paths = []
        for i, batch in enumerate(dataset.to_batches()):
            if max_batches is not None and i >= max_batches:
                print(f"⏹️ Límite alcanzado: solo procesados {max_batches} batches", flush=True)
                break

            print(f"📝 Procesando batch #{i+1}...", flush=True)
            table = pa.Table.from_batches([batch])
            df = table.to_pandas()

            if df["id_objeto"].astype(str).str.match(f"^{prefix}_\\d+$").all():
                print("ℹ️ id_objeto ya está bien formateado. No se modifica.", flush=True)
            else:
                df["id_numeric"] = df["id_objeto"].apply(extract_numeric_id)
                df = df[df["id_numeric"].notnull()]
                df["id_objeto"] = prefix + "_" + df["id_numeric"]
                df.drop(columns=["id_numeric"], inplace=True)

            table_clean = pa.Table.from_pandas(df, preserve_index=False)
            batch_path = batch_prefix.with_name(f"{batch_prefix.stem}_{i:04d}.parquet")
            pq.write_table(table_clean, batch_path)
            batch_paths.append(batch_path)

            print(f"✅ Batch #{i+1} guardado en {batch_path.name} ({len(df)} filas)", flush=True)
            del df, table, table_clean, batch
            gc.collect()

        # Merge de todos los batches
        if batch_paths:
            print(f"🔗 Unificando {len(batch_paths)} batches en {output_parquet.name}...", flush=True)
            batch_dataset = ds.dataset(batch_paths, format="parquet")
            pq.write_table(batch_dataset.to_table(), output_parquet)
            print(f"✅ Dataset unificado guardado como {output_parquet.name}", flush=True)

            # Limpieza
            for path in batch_paths:
                path.unlink()
            print(f"🧹 Archivos temporales eliminados.", flush=True)
        else:
            print(f"⚠️ No se generaron batches para {mision}, no se creó el dataset final", flush=True)

    except Exception as e:
        print(f"❌ Error procesando {mision}: {e}", flush=True)

# Permitir ejecución individual o paralela
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mision", type=str, help="Nombre de misión a procesar individualmente")
    parser.add_argument("--parallel", action="store_true", help="Procesar todas en paralelo")
    parser.add_argument("--max_batches", type=int, default=None, help="Número máximo de batches a procesar (modo test)")
    args = parser.parse_args()

    if args.mision:
        if args.mision in DATASETS:
            procesar_dataset(args.mision, max_batches=args.max_batches)
        else:
            print(f"❌ Misión desconocida: {args.mision}", flush=True)
    elif args.parallel:
        with Pool(processes=min(4, cpu_count())) as pool:
            pool.starmap(procesar_dataset, [(m, args.max_batches) for m in DATASETS.keys()])
    else:
        print("🔹 Usa --mision <nombre> para prueba o --parallel para procesar todo", flush=True)
