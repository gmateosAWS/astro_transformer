# script_1_eb.py

"""
- Descarga los catálogos de Kepler EB y TESS EB
- Prepara el CSV de entrada para descarga de curvas
- Ejecuta download_from_csv() desde el script download_curves.py
- Lee automáticamente todos los CSV descargados
- Consolida el dataset final estructurado en formato Parquet
"""

import pandas as pd
import time
from pathlib import Path
from download_curves import download_from_csv
from dataset_builder import DatasetBuilder
from utils.merge_downloaded_curves import read_and_merge_curves

# URLs de los catálogos de Kepler y TESS
KEPLER_EB_URL = "https://keplerebs.villanova.edu/data/kep_eclipsing_binary_catalog.csv"
TESS_EB_URL = "https://archive.stsci.edu/hlsps/tess-ebs/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"

CATALOG_DIR = Path("catalogs")
SAMPLE_CATALOG_DIR = Path("catalogs/samples")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LIST_IDS = Path("data/lists/eb_ids.csv")
OUTPUT_FILE = "dataset_eb"

def download_sample_catalogs():
    print("[⬇] Descargando catálogo reducido de Kepler EB para pruebas...")
    df_kepler = pd.read_csv(SAMPLE_CATALOG_DIR / "kepler_eb_sample.csv")
    df_kepler["id"] = df_kepler["KIC"]
    df_kepler["mission"] = "Kepler"
    df_kepler["clase_variable"] = "EB"
    df_kepler.to_csv(CATALOG_DIR / "kepler_eb.csv", index=False)

    print("[⬇] Descargando catálogo reducido de TESS EB para pruebas...")
    df_tess = pd.read_csv(SAMPLE_CATALOG_DIR / "tess_eb_sample.csv")
    df_tess["id"] = df_tess["TIC_ID"]
    df_tess["mission"] = "TESS"
    df_tess["clase_variable"] = "EB"
    df_tess.to_csv(CATALOG_DIR / "tess_eb.csv", index=False)

    return df_kepler, df_tess

def download_catalogs():
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    print("[⬇] Descargando catálogo Kepler EB...")
    LOCAL_KEPLER_CSV = Path(CATALOG_DIR / "kepler_eb_local.csv")    
    if LOCAL_KEPLER_CSV.exists():
        print("[📂] Cargando catálogo Kepler EB desde copia local...")
        df_kepler = pd.read_csv(LOCAL_KEPLER_CSV, comment="#")
    else:
        print("[⬇] Descargando catálogo Kepler EB...")
        df_kepler = pd.read_csv(KEPLER_EB_URL)    
    df_kepler = pd.read_csv(KEPLER_EB_URL)
    df_kepler = df_kepler.rename(columns={"kepid": "KIC"})
    df_kepler["id"] = df_kepler["KIC"]
    df_kepler["mission"] = "Kepler"
    df_kepler["clase_variable"] = "EB"
    df_kepler.to_csv(CATALOG_DIR / "kepler_eb.csv", index=False)

    print("[⬇] Descargando catálogo TESS EB...")
    df_tess = pd.read_csv(TESS_EB_URL)
    df_tess = df_tess.rename(columns={"TIC ID": "TIC_ID"})
    df_tess["id"] = df_tess["TIC_ID"]
    df_tess["mission"] = "TESS"
    df_tess["clase_variable"] = "EB"
    df_tess.to_csv(CATALOG_DIR / "tess_eb.csv", index=False)

    return df_kepler, df_tess

def generar_csv_descarga(df_kepler, df_tess):
    df_total = pd.concat([df_kepler, df_tess], ignore_index=True)
    LIST_IDS.parent.mkdir(parents=True, exist_ok=True)
    df_total[["id", "mission"]].to_csv(LIST_IDS, index=False)
    return df_total

def main(use_sample=True):
    t0 = time.perf_counter()

    # Paso 1: descarga de los catálogos de Kepler y TESS
    if use_sample:
        print("[⬇] Descargando catálogos de pruebas de Kepler y TESS...")
        df_kepler, df_tess = download_sample_catalogs()
    else:
        print("[⬇] Descargando catálogos completos de Kepler y TESS...")
        df_kepler, df_tess = download_catalogs()

    # Paso 2: generar CSV de entrada para el downloader
    print("[⬇] Generando CSV de entrada para descarga de curvas...")
    df_ids = generar_csv_descarga(df_kepler, df_tess)

    # Paso 3: descarga de curvas
    print("[⬇] Descargando curvas de luz...")
    download_from_csv(LIST_IDS, base_output_dir=RAW_DIR)

    # Paso 4: lectura y fusión de las curvas descargadas
    print("[⭢] Leyendo y fusionando curvas descargadas...")
    df_curvas = read_and_merge_curves(RAW_DIR)

    # Paso 5: merge de metadatos (etiquetas)
    print("[⭢] Fusionando metadatos (etiquetas)...")
    df_ids["id"] = df_ids["id"].astype(str)
    df_curvas["id_objeto"] = df_curvas["id_objeto"].astype(str)
    df_merged = df_curvas.merge(df_ids, left_on=["id_objeto", "mision"], right_on=["id", "mission"], how="left")
    df_merged["origen_etiqueta"] = "EB-catalog"
    df_merged["clase_variable"] = df_merged["clase_variable"].fillna("Unknown")

    print(f"[✓] Curvas unificadas: {df_merged['id_objeto'].nunique()} estrellas")
    print(f"[✓] Total de filas: {len(df_merged):,}")

    # Paso 6: guardar dataset final
    builder = DatasetBuilder(base_dir=PROCESSED_DIR)
    builder.add_source("EB", df_merged, "clase_variable", "EB-catalog")
    builder.save(df_merged, OUTPUT_FILE, format="parquet")

    print(f"[⏱] Tiempo total: {time.perf_counter() - t0:.2f} segundos")

if __name__ == "__main__":
    # Cambia a False cuando lo ejecutes en SageMaker
    main(use_sample=True)