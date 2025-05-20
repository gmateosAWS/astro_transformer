# script_3_vsx_tic_match.py
import os
import time
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as u
from collections import Counter
import pyarrow.dataset as ds
from datetime import datetime
import json
import csv
from astroquery.vizier import Vizier
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree
import numpy as np
import shutil

VSX_PATH = "catalogs/vsx_catalog.csv"
TIC_DIR = Path("data/raw/tic_chunks")
OUTPUT_PARQUET = "data/processed/dataset_vsx_tic_labeled.parquet"
TEMP_DIR = Path("data/processed/temp_vsx_tic")

DEC_MIN = -20
DEC_MAX = +20

# === Generar nombre de bin DEC ===
def dec_bin_from_value(dec):
    bin_start = int((dec // 2) * 2)
    bin_end = bin_start + 2
    def fmt(d):
        return f"{abs(d):02d}_00{'N' if d >= 0 else 'S'}"
    if bin_start == -2 and bin_end == 0:
        return "tic_dec02_00S__00_00N.csv.gz"
    return f"tic_dec{fmt(bin_start)}__{fmt(bin_end)}.csv.gz"

# === Descarga por bloques del catálogo VSX completo ===
def descargar_catalogo_vsx_por_bloques(destino="catalogs/vsx_catalog.csv", bloque=50000):
    print("🔄 Descargando catálogo VSX por bloques...")
    columnas = ["Name", "RAJ2000", "DEJ2000", "Type"]
    v = Vizier(columns=columnas, row_limit=bloque)

    try:
        result = v.query_constraints(catalog="B/vsx/vsx")
        if not result:
            raise RuntimeError("❌ No se pudo recuperar el catálogo VSX")
        df = result[0].to_pandas()
        df = df.rename(columns={
            "Name": "nombre_vsx",
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "Type": "clase_variable"
        })
        df = df[["nombre_vsx", "ra", "dec", "clase_variable"]]
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        df.to_csv(destino, index=False)
        print(f"✅ Catálogo guardado en: {destino} ({len(df)} registros)")
    except Exception as e:
        raise RuntimeError(f"❌ Error al descargar el catálogo VSX: {e}")

# === Bins TIC necesarios desde VSX ===
def get_needed_dec_bins(vsx_df, limit_bins=None):
    dec_bins = sorted(set(dec_bin_from_value(dec) for dec in vsx_df["dec"]))
    if limit_bins:
        dec_bins = dec_bins[:limit_bins]
        vsx_df = vsx_df[vsx_df["dec"].apply(lambda d: dec_bin_from_value(d) in dec_bins)].reset_index(drop=True)
        print(f"🎯 Limitando a {limit_bins} regiones DEC (bins únicos)")
    return dec_bins, vsx_df

# === Descargar ficheros TIC individuales ===
def download_tic_dec_file(filename):
    url = f"https://archive.stsci.edu/missions/tess/catalogs/tic_v82/{filename}"
    local_path = TIC_DIR / filename
    TIC_DIR.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        print(f"🟡 Ya existe: {filename}. No se vuelve a descargar.")
        return local_path
    print(f"⬇️ Descargando {filename}...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    else:
        print(f"❌ Error al descargar {filename}: {r.status_code}")
        return None

# === Descarga en paralelo ===
def descargar_tic_en_paralelo(filenames, max_workers=4):
    tic_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(download_tic_dec_file, fname): fname for fname in filenames
        }
        for future in tqdm(as_completed(future_to_file), total=len(filenames), desc="⬇️ Descargando TIC"):
            fname = future_to_file[future]
            try:
                result = future.result()
                if result:
                    tic_files.append(result)
            except Exception as e:
                print(f"❌ Error en {fname}: {e}")
    return tic_files

# === Cruce VSX ↔ TIC usando cKDTree ===
def cruzar_vsx_con_tic(vsx_df, tic_path, radio_arcsec=2.0):
    resultados = []
    try:
        print(f"📂 Procesando en chunks {tic_path.name}...")
        inicio = time.time()
        vsx_coords = SkyCoord(ra=vsx_df["ra"].values * u.deg, dec=vsx_df["dec"].values * u.deg)
        vsx_xyz = vsx_coords.cartesian.get_xyz().T.value

        reader = pd.read_csv(
            tic_path,
            compression='gzip',
            header=None,
            usecols=[0, 13, 14],
            names=["ID", "RA", "DEC"],
            dtype={0: "Int64", 13: "float64", 14: "float64"},
            chunksize=250_000,
            low_memory=False
        )
        for chunk in reader:
            chunk = chunk.dropna()
            tic_coords = SkyCoord(ra=chunk["RA"].values * u.deg, dec=chunk["DEC"].values * u.deg)
            tic_xyz = tic_coords.cartesian.get_xyz().T.value

            tree = cKDTree(tic_xyz)
            idxs = tree.query_ball_point(vsx_xyz, r=np.deg2rad(radio_arcsec / 3600))

            for i, indices in enumerate(idxs):
                if not indices:
                    continue
                # Obtener el más cercano
                separaciones = vsx_coords[i].separation(tic_coords[indices])
                idx_min = separaciones.argmin()
                match_idx = indices[idx_min]
                match = chunk.iloc[match_idx]
                row = vsx_df.iloc[i]
                resultados.append({
                    "id_objeto": f"VSX_{row['nombre_vsx']}",
                    "nombre_vsx": row['nombre_vsx'],
                    "clase_variable": row['clase_variable'],
                    "ra": row['ra'],
                    "dec": row['dec'],
                    "tic_id": match["ID"],
                    "tic_ra": match["RA"],
                    "tic_dec": match["DEC"],
                    "origen_etiqueta": "VSX"
                })
        print(f"⏱️ Chunks procesados en {time.time() - inicio:.2f} s")
    except Exception as e:
        print(f"❌ Error leyendo {tic_path.name}: {e}")
    return resultados

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
def main(limit=None, radio_arcsec=2.0, limit_bins=None, max_download_workers=4):
    if not os.path.exists(VSX_PATH):
        descargar_catalogo_vsx_por_bloques(destino=VSX_PATH)

    vsx_df = pd.read_csv(VSX_PATH)
    vsx_df = vsx_df[(vsx_df["dec"] >= DEC_MIN) & (vsx_df["dec"] <= DEC_MAX)].reset_index(drop=True)
    print(f"📉 Catálogo VSX reducido a franja DEC {DEC_MIN} → {DEC_MAX} ({len(vsx_df)} objetos)")

    if limit and limit < len(vsx_df):
        vsx_df = vsx_df.sample(n=limit, random_state=42).reset_index(drop=True)

    dec_files, vsx_df = get_needed_dec_bins(vsx_df, limit_bins=limit_bins)
    print(f"📦 Archivos DEC necesarios: {dec_files}")

    tic_files = descargar_tic_en_paralelo(dec_files, max_workers=max_download_workers)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    parquet_paths = []

    for tic_file in tic_files:
        resultados = cruzar_vsx_con_tic(vsx_df, tic_file, radio_arcsec=radio_arcsec)
        if resultados:
            df_temp = pd.DataFrame(resultados)
            temp_parquet_path = TEMP_DIR / f"{tic_file.stem}.parquet"
            df_temp.to_parquet(temp_parquet_path, index=False)
            parquet_paths.append(temp_parquet_path)

    if not parquet_paths:
        print("⚠️ No se encontraron coincidencias en ningún fichero TIC.")
        return

    df_final = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✅ Guardado en {OUTPUT_PARQUET} ({len(df_final)} coincidencias)")

    if not df_final.empty:
        inspect_and_export_summary(OUTPUT_PARQUET, output_format="csv")

    # Limpieza de temporales
    for p in parquet_paths:
        p.unlink()
    
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("🧹 Archivos temporales eliminados.")

if __name__ == "__main__":
    main(limit=5000, radio_arcsec=3.0, limit_bins=30, max_download_workers=4)