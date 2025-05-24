# script_3e_complement_gaia_curves.py

import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
from lightkurve import search_lightcurve
import pyarrow.parquet as pq
from multiprocessing import Pool
from astropy.coordinates import SkyCoord
from astropy import units as u
import shutil
import pyarrow.dataset as ds
from collections import Counter
from datetime import datetime
import csv, json

# === Configuración ===
INPUT_PARQUET = Path("data/processed/dataset_gaia_dr3_vsx_tic_labeled_with_coords_clean.parquet")
OUTPUT_PARQUET = Path("data/processed/dataset_gaia_dr3_vsx_tic_labeled_with_coords_clean_complemented.parquet")
TEMP_DIR = Path("data/processed/temp_gaia_complemented")

# === Descarga robusta de curva ===
def descargar_curva_tess(tic_id, ra, dec):
    try:
        search = search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if len(search) == 0:
            search = search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            search = search_lightcurve(coord, radius=0.002 * u.deg, mission="TESS")
            if len(search) == 0:
                return None

        lc_collection = search.download_all()
        if lc_collection is None:
            return None

        registros = []
        for lc in lc_collection:
            lc = lc.remove_nans().normalize()
            time = lc.time.value
            flux = lc.flux.value
            flux_err = lc.flux_err.value if lc.flux_err is not None else np.zeros_like(flux)

            registros.append(pd.DataFrame({
                "tiempo": time,
                "magnitud": flux,
                "error": flux_err,
                "id_objeto": f"TIC_{tic_id}"
            }))
        return pd.concat(registros, ignore_index=True)
    except Exception:
        return None

# === Procesamiento de una curva ===
def procesar_lote(tic_id, clase, clase_norm, ra, dec):
    curva = descargar_curva_tess(tic_id, ra, dec)
    if curva is None or curva.empty:
        return None
    curva["clase_variable"] = clase
    curva["clase_variable_normalizada"] = clase_norm
    curva["id_mision"] = f"TIC_{tic_id}"
    curva["mision"] = "TESS"
    curva["origen_etiqueta"] = "GaiaDR3"
    return curva

def desempaquetar_lote(tupla):
    return procesar_lote(*tupla)

# === Descarga paralela ===
def procesar_todos(df, num_workers=4):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    entradas = list(zip(df["tic_id"], df["clase_variable"], df["clase_variable_normalizada"], df["ra"], df["dec"]))
    resultados_totales = 0

    with Pool(processes=num_workers) as pool:
        for i, resultado in enumerate(tqdm(pool.imap_unordered(desempaquetar_lote, entradas), total=len(entradas))):
            if resultado is None:
                continue
            resultado = resultado.dropna(subset=["tiempo", "magnitud"])
            if len(resultado) == 0:
                continue
            output_file = TEMP_DIR / f"gaia_complemented_chunk_{i:04d}.parquet"
            resultado.to_parquet(output_file, index=False)
            resultados_totales += len(resultado)

    if resultados_totales == 0:
        print("⚠️ No se encontraron curvas válidas para ningún objeto. Proceso finalizado sin generar archivo.")
        return

    print(f"✅ Guardados {resultados_totales} registros en parquet temporales.")

    print("📦 Unificando resultados...")
    files = sorted(TEMP_DIR.glob("gaia_complemented_chunk_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✅ Dataset final guardado en: {OUTPUT_PARQUET} ({len(df_final)} filas)")

    inspect_and_export_summary(OUTPUT_PARQUET, output_format="csv")

    print("🧹 Eliminando temporales...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

# === Generar resumen .csv ===
def inspect_and_export_summary(parquet_path, output_format="csv"):
    print(f"\n📁 Inspeccionando: {parquet_path}")
    dataset = ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema

    summary = {
        "file": parquet_path,
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

    output_dir = "data/processed/summary"
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(parquet_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_summary.{output_format}")

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    elif output_format == "csv":
        with open(output_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Clase", "Recuento"])
            for clase, count in class_counter.items():
                writer.writerow([clase, count])
        with open(output_path.replace(".csv", "_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Fichero: {summary['file']}\n")
            f.write(f"Filas totales: {summary['total_rows']}\n")
            f.write(f"Curvas únicas (id_objeto): {summary['total_objects']}\n")
            f.write(f"Columnas: {list(summary['columns'].keys())}\n")
            f.write(f"Fecha: {summary['timestamp']}\n")
    else:
        raise ValueError("❌ Formato no soportado. Usa 'json' o 'csv'.")

    print(f"✅ Resumen exportado a: {output_path}")

# === MAIN ===
def main(limit=None, workers=4):
    print(f"📥 Cargando dataset: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    df = df.drop_duplicates(subset=["tic_id"]).dropna(subset=["tic_id", "ra", "dec"])

    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"🔍 Ejecutando solo con {limit} estrellas (modo test aleatorio)")

    procesar_todos(df, num_workers=workers)

if __name__ == "__main__":
    main(limit=None, workers=4)
