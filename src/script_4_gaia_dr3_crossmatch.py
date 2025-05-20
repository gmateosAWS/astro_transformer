# script_4_gaia_dr3_crossmatch.py
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
from pyarrow.parquet import ParquetDataset
from pathlib import Path
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy import units as u
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import csv
import json
import shutil
from multiprocessing import Pool

# --- Configuración ---
DATA_DIR = Path("data/processed")
OUTPUT_PARQUET = DATA_DIR / "dataset_gaia_dr3_labeled.parquet"
TEMP_DIR = DATA_DIR / "temp_gaia_dr3"
SUMMARY_DIR = DATA_DIR / "summary"
SEARCH_RADIUS_ARCSEC = 2.0

DATASETS = [
    DATA_DIR / "dataset_vsx_tess_labeled_clean.parquet"
]

# --- Consolidación eficiente con pyarrow.dataset ---
def consolidar_datasets():
    columnas_candidatas = ["id_objeto", "clase_variable", "mision", "origen_etiqueta", "id_mision", "ra", "dec", "tic_ra", "tic_dec"]
    dataframes = []

    for path in DATASETS:
        if not path.exists():
            continue
        print(f"📄 Cargando: {path.name}")
        dataset = ds.dataset(str(path), format="parquet")
        available_cols = dataset.schema.names

        if not any(col in available_cols for col in ["ra", "tic_ra"]) or not any(col in available_cols for col in ["dec", "tic_dec"]):
            print(f"⏭️  {path.name} omitido (sin columnas de coordenadas)")
            continue

        used_columns = [col for col in columnas_candidatas if col in available_cols]
        table = dataset.to_table(columns=used_columns)
        df = table.to_pandas()

        # Renombrar columnas si están en formato tic_ra / tic_dec
        if "tic_ra" in df.columns and "ra" not in df.columns:
            df = df.rename(columns={"tic_ra": "ra"})
        if "tic_dec" in df.columns and "dec" not in df.columns:
            df = df.rename(columns={"tic_dec": "dec"})

        df = df.dropna(subset=["id_objeto", "clase_variable", "ra", "dec"])
        df["id_objeto"] = df["id_objeto"].astype(str).str.extract(r"TIC_(\d+)", expand=False).fillna("MISSING")
        df = df[df["id_objeto"] != "MISSING"]
        df["id_objeto"] = "TIC_" + df["id_objeto"].astype(str)
        dataframes.append(df)

    if not dataframes:
        raise RuntimeError("❌ Ninguno de los datasets incluye coordenadas válidas. No se puede hacer el cruce con Gaia DR3.")

    df_all = pd.concat(dataframes, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["id_objeto"])
    print(f"📦 Dataset consolidado: {len(df_all)} objetos únicos con coordenadas")
    return df_all

# --- Consulta a Gaia ---
def buscar_gaia_variables(ra, dec, radius_arcsec=SEARCH_RADIUS_ARCSEC):
    try:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        query = f"""
            SELECT TOP 1 *
            FROM gaiadr3.vari_summary
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                            CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius_arcsec / 3600.0}))
        """
        job = Gaia.launch_job(query)
        results = job.get_results()
        return results.to_pandas() if len(results) > 0 else None
    except Exception:
        return None

# --- Procesar una fila ---
def procesar_objeto(obj):
    try:
        ra, dec, id_objeto, clase = obj
        result = buscar_gaia_variables(ra, dec)
        if result is None or result.empty:
            return None
        df = result.copy()
        df["id_objeto"] = id_objeto
        df["clase_variable"] = clase
        df["id_mision"] = id_objeto
        df["mision"] = "GAIA"
        df["origen_etiqueta"] = "GAIA_DR3"
        return df
    except Exception:
        return None

# --- Función para inspección y resumen ---
def inspect_and_export_summary(parquet_path, output_format="csv"):
    print(f"\n📁 Inspeccionando: {parquet_path}")
    dataset = ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema

    summary = {
        "file": str(parquet_path),
        "columns": {field.name: str(field.type) for field in schema},
        "class_distribution": {},
        "total_rows": 0,
        "total_objects": 0,
        "timestamp": datetime.now().isoformat()
    }

    class_counter = Counter()
    objetos = set()

    for batch in tqdm(dataset.to_batches(columns=["clase_variable", "id_objeto"]), desc="🧮 Procesando por lotes"):
        summary["total_rows"] += batch.num_rows
        if "clase_variable" in batch.schema.names:
            clases = batch.column("clase_variable").to_pylist()
            class_counter.update(clases)
        if "id_objeto" in batch.schema.names:
            objetos.update(batch.column("id_objeto").to_pylist())

    summary["class_distribution"] = dict(class_counter)
    summary["total_objects"] = len(objetos)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    basename = Path(parquet_path).stem
    output_path = SUMMARY_DIR / f"{basename}_summary.{output_format}"

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    elif output_format == "csv":
        with open(output_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Clase", "Recuento"])
            for clase, count in class_counter.items():
                writer.writerow([clase, count])
        with open(str(output_path).replace(".csv", "_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Fichero: {summary['file']}\n")
            f.write(f"Filas totales: {summary['total_rows']}\n")
            f.write(f"Curvas únicas (id_objeto): {summary['total_objects']}\n")
            f.write(f"Columnas: {list(summary['columns'].keys())}\n")
            f.write(f"Fecha: {summary['timestamp']}\n")

    print(f"✅ Resumen exportado a: {output_path}")

# --- Proceso completo ---
def procesar_cruce(df, workers=4):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    entradas = list(zip(df["ra"], df["dec"], df["id_objeto"], df["clase_variable"]))
    resultados = []

    with Pool(processes=workers) as pool:
        for result in tqdm(pool.imap_unordered(procesar_objeto, entradas), total=len(entradas)):
            if result is not None:
                resultados.append(result)

    if not resultados:
        print("⚠️ No se encontraron coincidencias GAIA DR3")
        return

    df_final = pd.concat(resultados, ignore_index=True)
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✅ Guardado en {OUTPUT_PARQUET} ({len(df_final)} filas)")

    inspect_and_export_summary(OUTPUT_PARQUET, output_format="csv")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

# --- MAIN ---
def main(limit=None, workers=4):
    print("📥 Consolidando datasets...")
    df = consolidar_datasets()

    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"🔍 Ejecutando prueba con {len(df)} objetos aleatorios")
    else:
        print(f"📊 Procesando dataset completo con {len(df)} objetos")

    procesar_cruce(df, workers=workers)

if __name__ == "__main__":
    main(limit=None, workers=4)
