# script_3b_download_tess_curves_from_vsx.py
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
from src.utils.normalization_dict import normalize_label
from src.utils.inspect_and_export_summary import inspect_and_export_summary

INPUT_PARQUET = "data/processed/dataset_vsx_tic_labeled_south.parquet"
OUTPUT_PARQUET = "data/processed/dataset_vsx_tess_labeled_south.parquet"
TEMP_DIR = Path("data/processed/temp_vsx_tess_south")

# --- Descarga y procesamiento de una curva ---
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

# --- Procesar un lote de curvas ---
def procesar_lote(tic_id, clase, ra, dec):
    curva = descargar_curva_tess(tic_id, ra, dec)
    if curva is None or curva.empty:
        return None
    curva["clase_variable"] = clase
    curva["clase_variable_normalizada"] = curva["clase_variable"].apply(normalize_label)
    curva["id_mision"] = f"TIC_{tic_id}"
    curva["mision"] = "TESS"
    curva["origen_etiqueta"] = "VSX"
    return curva

# --- Función auxiliar para desempaquetar tuplas ---
def desempaquetar_lote(tupla):
    return procesar_lote(*tupla)

# --- Procesamiento paralelo con multiproceso y guardado incremental ---
def procesar_todos(df, num_workers=4):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    entradas = list(zip(df["tic_id"], df["clase_variable"], df["tic_ra"], df["tic_dec"]))
    resultados_totales = 0

    with Pool(processes=num_workers) as pool:
        for i, resultado in enumerate(tqdm(pool.imap_unordered(desempaquetar_lote, entradas), total=len(entradas))):
            if resultado is None:
                continue
            resultado = resultado.dropna(subset=["tiempo", "magnitud"])
            if len(resultado) == 0:
                continue
            output_file = TEMP_DIR / f"vsx_tess_chunk_{i:04d}.parquet"
            resultado.to_parquet(output_file, index=False)
            resultados_totales += len(resultado)

    if resultados_totales == 0:
        print("⚠️ No se encontraron curvas válidas para ningún objeto. Proceso finalizado sin generar archivo.")
        return

    print(f"✅ Guardados {resultados_totales} registros en parquet temporales.")

    print("📦 Unificando resultados...")
    files = sorted(TEMP_DIR.glob("vsx_tess_chunk_*.parquet"))
    if not files:
        print("⚠️ No se generaron archivos temporales para unir.")
        return

    dfs = [pd.read_parquet(f) for f in files]
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✅ Dataset final guardado en: {OUTPUT_PARQUET} ({len(df_final)} filas)")

    inspect_and_export_summary(OUTPUT_PARQUET, output_format="csv")

    print("🧹 Eliminando temporales...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

# --- MAIN ---
def main(limit=None, workers=4):
    print(f"📥 Cargando dataset: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    df = df.drop_duplicates(subset=["tic_id"]).dropna(subset=["tic_id"])

    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"🔍 Ejecutando solo con {limit} estrellas (modo test aleatorio)")

    procesar_todos(df, num_workers=workers)

if __name__ == "__main__":
    main(limit=None, workers=4)
