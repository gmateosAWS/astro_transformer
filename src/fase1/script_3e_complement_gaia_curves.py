# script_3e_complement_gaia_curves.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
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
from src.utils.inspect_and_export_summary import inspect_and_export_summary

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
            lc = lc.remove_nans()
            time = lc.time.value
            flux = lc.flux.value
            flux_err = lc.flux_err.value if lc.flux_err is not None else np.zeros_like(flux)

            registros.append(pd.DataFrame({
                "tiempo": time,
                "magnitud": flux,
                "error": flux_err,
                "id_objeto": f"GAIA_TIC_{tic_id}"
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
    import gc
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    entradas = list(zip(df["tic_id"], df["clase_variable"], df["clase_variable_normalizada"], df["ra"], df["dec"]))
    resultados_totales = 0
    resumen_csv = []  # Para exportar resumen final

    pool = Pool(processes=num_workers)
    try:
        print("🚀 Iniciando procesamiento paralelo con Pool...")
        for i, (entrada, resultado) in enumerate(tqdm(zip(entradas, pool.imap_unordered(desempaquetar_lote, entradas)), total=len(entradas))):
            tic_id, clase, clase_norm, ra, dec = entrada
            if resultado is None:
                print(f"❌ {i+1}: TIC {tic_id} → Sin resultados (None)")
                resumen_csv.append((tic_id, "none", 0))
                continue
            resultado = resultado.dropna(subset=["tiempo", "magnitud"])
            if len(resultado) == 0:
                print(f"⚠️ {i+1}: TIC {tic_id} → Curva vacía")
                resumen_csv.append((tic_id, "vacio", 0))
                continue

            print(f"✅ {i+1}: TIC {tic_id} → {len(resultado)} puntos")
            resumen_csv.append((tic_id, "ok", len(resultado)))
            output_file = TEMP_DIR / f"gaia_complemented_chunk_{i:04d}.parquet"
            resultado.to_parquet(output_file, index=False)
            resultados_totales += len(resultado)
            del resultado
            gc.collect()
    except Exception as e:
        print(f"❌ Error durante el procesamiento paralelo: {e}")
    finally:
        print("🔚 Cerrando pool de procesos...")
        pool.close()
        pool.join()

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

    print("📝 Guardando resumen por TIC en gaia_download_summary.csv...")
    with open("gaia_download_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tic_id", "estado", "n_puntos"])
        writer.writerows(resumen_csv)
    print("✅ Resumen guardado.")


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
