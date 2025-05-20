# script_4_gaia_dr3_vsx_tic_crossmatch.py
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
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
DATASET_PATH = Path("data/processed/dataset_vsx_tic_labeled_clean_fixed.parquet")
OUTPUT_PARQUET = Path("data/processed/dataset_gaia_dr3_vsx_tic_labeled.parquet")
TEMP_DIR = Path("data/processed/temp_gaia_vsx_tic")
SUMMARY_DIR = Path("data/processed/summary")
SEARCH_RADIUS_ARCSEC = 2.0

# --- Consolidación directa ---
def cargar_dataset():
    print(f"📄 Cargando: {DATASET_PATH.name}")
    dataset = ds.dataset(str(DATASET_PATH), format="parquet")
    columnas = dataset.schema.names

    # Verificamos variantes de coordenadas
    if "ra" not in columnas and "tic_ra" not in columnas:
        raise RuntimeError("❌ El dataset no contiene ninguna columna RA válida")
    if "dec" not in columnas and "tic_dec" not in columnas:
        raise RuntimeError("❌ El dataset no contiene ninguna columna DEC válida")

    columnas_usar = ["id_objeto", "clase_variable"]
    if "ra" in columnas:
        columnas_usar.append("ra")
    elif "tic_ra" in columnas:
        columnas_usar.append("tic_ra")

    if "dec" in columnas:
        columnas_usar.append("dec")
    elif "tic_dec" in columnas:
        columnas_usar.append("tic_dec")

    table = dataset.to_table(columns=columnas_usar)
    df = table.to_pandas()

    # Renombrar coordenadas si vienen como tic_ra / tic_dec
    if "tic_ra" in df.columns:
        df.rename(columns={"tic_ra": "ra"}, inplace=True)
    if "tic_dec" in df.columns:
        df.rename(columns={"tic_dec": "dec"}, inplace=True)

    print("📊 Total inicial:", len(df))

    # Normalizar id_objeto
    df["id_numeric"] = df["id_objeto"].astype(str).str.extract(r"TIC_(\d+)", expand=False)
    df = df[~df["id_numeric"].isna()]
    df["id_objeto"] = "TIC_" + df["id_numeric"].astype(str)
    df.drop(columns=["id_numeric"], inplace=True)

    print("📊 Tras normalizar id_objeto:", len(df))

    # Rellenar clase vacía
    df["clase_variable"] = df["clase_variable"].fillna("UNKNOWN")

    # Filtrar coordenadas
    df = df.dropna(subset=["ra", "dec"])
    print("📊 Tras filtrar coordenadas válidas:", len(df))

    # Eliminar duplicados
    df = df.drop_duplicates(subset=["id_objeto"])
    print("📊 Final tras drop_duplicates:", len(df))

    print(f"📦 Dataset listo: {len(df)} objetos únicos con coordenadas")


    print(f"📦 Dataset listo: {len(df)} objetos únicos con coordenadas")
    return df



# --- Consulta a Gaia ---
def buscar_gaia_variables(ra, dec, radius_arcsec=SEARCH_RADIUS_ARCSEC):
    try:
        query = f"""
            SELECT TOP 1 vs.*
            FROM gaiadr3.vari_summary AS vs
            JOIN gaiadr3.gaia_source AS g
            ON vs.source_id = g.source_id
            WHERE 1=CONTAINS(
                POINT('ICRS', g.ra, g.dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec / 3600.0})
            )
        """
        job = Gaia.launch_job(query)
        results = job.get_results()
        return results.to_pandas() if len(results) > 0 else None
    except Exception as e:
        print(f"❌ Error en consulta Gaia: {e}")
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

# --- Inspección y resumen ---
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
    df = cargar_dataset()

    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"🔍 Ejecutando prueba con {len(df)} objetos aleatorios")
    else:
        print(f"📊 Procesando dataset completo con {len(df)} objetos")

    procesar_cruce(df, workers=workers)

if __name__ == "__main__":
    main(limit=None, workers=4)
