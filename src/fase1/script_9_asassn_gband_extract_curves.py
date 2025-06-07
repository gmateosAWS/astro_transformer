import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import concurrent.futures
import gc

# Add the src directory to sys.path to resolve imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.normalization_dict import normalize_label
from src.utils.inspect_and_export_summary import inspect_and_export_summary

# Configuración
CATALOG_PATH = Path("catalogs/asassn_catalog_variables_stars_gband.csv")
TAR_PATH = Path("catalogs/g_band_lcs.tar.gz")
TEMP_CURVES_DIR = Path("data/temp/asassn_gband_curves/g_band_lcs")
OUTPUT_PATH = Path("data/processed/dataset_asassn_gband.parquet")

ROW_GROUP_SIZE = 50000


def print_normalization_report(df):
    print("\n=== REPORTE DE NORMALIZACIÓN ===")
    unique_classes = sorted(df["clase_variable"].unique())
    unknowns = []
    for val in unique_classes:
        norm = normalize_label(val)
        print(f"{val:15} → {norm}")
        if norm == "Unknown":
            unknowns.append(val)
    if unknowns:
        print("\nClases que deberías añadir al diccionario (Unknown):")
        for u in unknowns:
            print(" -", u)
    else:
        print("\nTodas las clases del catálogo están normalizadas correctamente.")


def load_catalog(path):
    df = pd.read_csv(path)
    df.rename(columns={"ID": "id", "ML_classification": "clase_variable"}, inplace=True)
    df["clase_variable_normalizada"] = df["clase_variable"].apply(normalize_label)
    # El informe de normalización se realiza sobre todo el catálogo, no solo sobre los .dat seleccionados
    print_normalization_report(df)
    return df


def extract_tar_to_temp(tar_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        print(f"🟢 Carpeta temporal ya contiene archivos, no se extrae {tar_path.name}")
        return
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
        print(f"✅ Extraído {tar_path.name} a {output_dir}")


def load_lightcurve_file(filepath):
    try:
        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            comment="#",
            names=["time", "camera", "mag", "mag_err", "flux", "flux_err", "fwhm", "image"]
        )
        if df.empty:
            return None
        # El id debe coincidir exactamente con el del catálogo (reemplaza '_' por ' ')
        df['id'] = filepath.stem.replace("_", " ")
        df = df[["id", "time", "mag", "mag_err", "flux", "flux_err"]]
        return df
    except Exception as e:
        print(f"❌ Error con {filepath.name}: {e}")
        return None


def process_lightcurves_parallel(
    curves_dir, use_parallel=True, max_files=None, batch_size=1000,
    save_intermediate=False, intermediate_dir=None, resume=True
):
    files = list(curves_dir.glob("*.dat"))
    if max_files is not None:
        files = files[:max_files]
    tqdm_desc = f"🔄 Procesando {len(files)} curvas"
    if not files:
        print("⚠️ No se encontraron archivos DAT para procesar.")
        return pd.DataFrame()
    batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
    all_results = []
    if save_intermediate:
        if intermediate_dir is None:
            intermediate_dir = Path("data/processed/asassn_gband_batches")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        # Marca de batches ya procesados
        processed_batches = {p.stem for p in intermediate_dir.glob("batch_*.parquet")}
    else:
        processed_batches = set()
    for i, batch in enumerate(batches):
        batch_name = f"batch_{i+1:04d}"
        batch_path = None
        if save_intermediate:
            batch_path = intermediate_dir / f"{batch_name}.parquet"
            if resume and batch_path.exists():
                print(f"⏩ Saltando {batch_name} (ya procesado)")
                continue
        print(f"🗂️ Procesando {batch_name} ({len(batch)} archivos)...")
        if use_parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(load_lightcurve_file, batch), total=len(batch), desc=f"{tqdm_desc} ({batch_name})"))
        else:
            results = [load_lightcurve_file(f) for f in tqdm(batch, desc=f"{tqdm_desc} ({batch_name})")]
        valid_results = [r for r in results if r is not None]
        if valid_results:
            batch_df = pd.concat(valid_results, ignore_index=True)
            if save_intermediate:
                batch_df.to_parquet(batch_path)
                print(f"💾 Batch guardado: {batch_path}")
            else:
                all_results.append(batch_df)
        # Liberar memoria tras cada batch
        del results, valid_results
        gc.collect()
    if save_intermediate:
        print("🔄 Cargando todos los batches intermedios para concatenar...")
        all_batches = sorted(intermediate_dir.glob("batch_*.parquet"))
        all_results = [pd.read_parquet(p) for p in tqdm(all_batches, desc="Concatenando batches")]
    if not all_results:
        print("⚠️ Ningún archivo DAT válido tras la carga.")
        return pd.DataFrame()
    curves_df = pd.concat(all_results, ignore_index=True)
    return curves_df


def normalize_classes_and_merge(df_cat, df_curves, batch_size=2_000_000, output_dir="data/processed/asassn_gband_merged_batches"):
    """
    Realiza el merge por batches para evitar MemoryError.
    Devuelve la ruta de los parquet intermedios.
    """
    if df_curves is None or df_curves.empty:
        print("⚠️ No hay curvas para unir, devolviendo DataFrame vacío.")
        columns = ["id", "time", "mag", "mag_err", "flux", "flux_err", "clase_variable", "clase_variable_normalizada", "mission", "mission_id", "source_dataset", "label_source", "band"]
        return []

    df_curves["id"] = df_curves["id"].astype(str)
    df_cat["id"] = df_cat["id"].astype(str)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(df_curves)
    parquet_paths = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df_curves.iloc[start:end].copy()
        merged = batch.merge(df_cat[["id", "clase_variable", "clase_variable_normalizada"]], on="id", how="left")
        merged["mission"] = "ASASSN"
        merged["mission_id"] = "ASASSN_gband"
        merged["source_dataset"] = "asassn_gband"
        merged["label_source"] = "ASASSN_Catalog"
        merged["band"] = "g"
        batch_path = output_dir / f"merged_{start:09d}_{end:09d}.parquet"
        merged.to_parquet(batch_path)
        parquet_paths.append(batch_path)
        print(f"✅ Guardado merge batch {start}-{end} de {n} en {batch_path}")
        del batch, merged
        gc.collect()
    return parquet_paths


def concat_parquet_batches(parquet_paths, output_path, batch_size=250_000):
    """
    Escribe el Parquet final directamente desde los batches, sin concatenar en memoria.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import os

    if os.path.exists(output_path):
        os.remove(output_path)

    writer = None
    total_rows = 0
    for i, p in enumerate(tqdm(parquet_paths, desc="Escribiendo batches finales")):
        df = pd.read_parquet(p)
        n = len(df)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = df.iloc[start:end]
            table = pa.Table.from_pandas(chunk)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
            total_rows += len(chunk)
        del df, chunk, table
        gc.collect()
    if writer is not None:
        writer.close()
    print(f"✅ Dataset final guardado en {output_path} ({total_rows} filas)")


def save_to_parquet(df, output_path, batch_size=500_000):
    """
    Guarda el DataFrame en Parquet por partes para evitar problemas de memoria.
    Compatible con versiones de pyarrow que NO soportan append en write_table.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import os

    n = len(df)
    if n == 0:
        print("⚠️ DataFrame vacío, no se guarda Parquet.")
        return

    # Si el fichero existe, bórralo antes de empezar
    if os.path.exists(output_path):
        os.remove(output_path)

    writer = None
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = df.iloc[start:end]
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        print(f"✅ Guardado filas {start} - {end} en {output_path}")
    if writer is not None:
        writer.close()
    print(f"✅ Guardado parquet completo: {output_path}")


def main(
    use_parallel=True, max_files=None, batch_size=1000,
    save_intermediate=True, intermediate_dir=None, resume=True
):
    print("📂 Cargando catálogo...")
    df_catalog = load_catalog(CATALOG_PATH)

    print("📦 Extrayendo curvas del .tar.gz...")
    extract_tar_to_temp(TAR_PATH, TEMP_CURVES_DIR)

    print("📊 Procesando curvas...")
    df_curves = process_lightcurves_parallel(
        TEMP_CURVES_DIR,
        use_parallel=use_parallel,
        max_files=max_files,
        batch_size=batch_size,
        save_intermediate=save_intermediate,
        intermediate_dir=intermediate_dir,
        resume=resume
    )

    print("🔗 Unificando catálogo y curvas por batches...")
    parquet_paths = normalize_classes_and_merge(df_catalog, df_curves, batch_size=250_000)

    print("💾 Concatenando y guardando en formato Parquet final...")
    concat_parquet_batches(parquet_paths, OUTPUT_PATH, batch_size=250_000)

    print("📋 Inspeccionando dataset final...")
    inspect_and_export_summary(OUTPUT_PATH)

if __name__ == "__main__":
    parquet_batches_dir = Path("data/processed/asassn_gband_merged_batches")
    parquet_paths = sorted(parquet_batches_dir.glob("merged_*.parquet"))
    if parquet_paths:
        print(f"💾 Concatenando y guardando en formato Parquet final a partir de {len(parquet_paths)} batches...")
        concat_parquet_batches(parquet_paths, OUTPUT_PATH, batch_size=250_000)
        print("📋 Inspeccionando dataset final...")
        inspect_and_export_summary(OUTPUT_PATH)
    else:
        print("⚠️ No se encontraron batches intermedios, ejecutando el proceso completo...")
        main()
