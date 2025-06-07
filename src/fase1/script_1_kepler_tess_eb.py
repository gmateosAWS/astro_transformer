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
from utils.download_curves import download_from_csv, download_from_csv_parallel
from utils.dataset_builder import DatasetBuilder
from utils.merge_downloaded_curves import read_and_merge_curves
import logging
import glob
import argparse

logging.getLogger("lightkurve").setLevel(logging.ERROR)
logging.getLogger("astropy").setLevel(logging.ERROR)

import urllib3
urllib3.util.connection.HAS_IPV6 = False  # previene ciertos timeouts

from requests.adapters import HTTPAdapter
from requests.sessions import Session

session = Session()
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("https://", adapter)
session.mount("http://", adapter)

# URLs de los catálogos de Kepler y TESS
# La URL de Kepler devuelve error por lo que se descarga el CSV local
KEPLER_EB_URL = "https://keplerebs.villanova.edu/data/kep_eclipsing_binary_catalog.csv"
TESS_EB_URL = "https://archive.stsci.edu/hlsps/tess-ebs/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"

CATALOG_DIR = Path("catalogs")
SAMPLE_CATALOG_DIR = Path("catalogs/samples")
RAW_DIR = Path("/home/ec2-user/backup/data/raw")
#PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR = Path("/home/ec2-user/backup/processed")
LIST_IDS = Path("data/lists/eb_ids.csv")
OUTPUT_FILE = "dataset_eb"
# En SageMaker el path a /data/raw es diferente
RAW_DIR_CANDIDATES = [
    RAW_DIR,
    Path("/home/ec2-user/backup/data/raw")
]
RAW_DIR = next((p for p in RAW_DIR_CANDIDATES if p.exists()), None)
if RAW_DIR is None:
    raise FileNotFoundError("❌ No se encontró ninguna ruta válida para data/raw")
print(f"📁 Usando RAW_DIR: {RAW_DIR}")

def download_sample_catalogs():
    print("[⬇] Descargando catálogo reducido de Kepler EB para pruebas...", flush=True)
    df_kepler = pd.read_csv(SAMPLE_CATALOG_DIR / "kepler_eb_sample.csv")
    df_kepler["id"] = df_kepler["KIC"]
    df_kepler["mission"] = "Kepler"
    df_kepler["clase_variable"] = "EB"
    df_kepler.to_csv(CATALOG_DIR / "kepler_eb.csv", index=False)

    print("[⬇] Descargando catálogo reducido de TESS EB para pruebas...", flush=True)
    df_tess = pd.read_csv(SAMPLE_CATALOG_DIR / "tess_eb_sample.csv")
    df_tess["id"] = df_tess["tess_id"]
    df_tess["mission"] = "TESS"
    df_tess["clase_variable"] = "EB"
    df_tess.to_csv(CATALOG_DIR / "tess_eb.csv", index=False)

    return df_kepler, df_tess

def download_catalogs():
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    print("[⬇] Descargando catálogo Kepler EB...", flush=True)
    LOCAL_KEPLER_CSV = Path(CATALOG_DIR / "kepler_eb_local.csv")    
    if LOCAL_KEPLER_CSV.exists():
        print("[📂] Cargando catálogo Kepler EB desde copia local...", flush=True)
        df_kepler = pd.read_csv(LOCAL_KEPLER_CSV, comment="#")
    else:
        print("[⬇] Descargando catálogo Kepler EB...", flush=True)
        df_kepler = pd.read_csv(KEPLER_EB_URL)    
    df_kepler = df_kepler.rename(columns={"kepid": "KIC"})
    df_kepler["id"] = df_kepler["KIC"]
    df_kepler["mission"] = "Kepler"
    df_kepler["clase_variable"] = "EB"
    df_kepler.to_csv(CATALOG_DIR / "kepler_eb.csv", index=False)

    print("[⬇] Descargando catálogo TESS EB...", flush=True)
    df_tess = pd.read_csv(TESS_EB_URL)
    df_tess = df_tess.rename(columns={"TIC ID": "TIC_ID"})
    df_tess["id"] = df_tess["tess_id"]
    df_tess["mission"] = "TESS"
    df_tess["clase_variable"] = "EB"
    df_tess.to_csv(CATALOG_DIR / "tess_eb.csv", index=False)

    return df_kepler, df_tess

def filter_pending(df_ids, mission, raw_dir):
    df_m = df_ids[df_ids["mission"] == mission].copy()
    raw_path = Path(raw_dir) / mission.lower()
    files = list(raw_path.glob(f"{mission.lower()}_*.csv"))
    ids_done = {f.stem.split("_")[1] for f in files}
    return df_m[~df_m["id"].astype(str).isin(ids_done)]

def generar_csv_descarga(df_kepler, df_tess, mission="ALL", only_pending=True):
    """
    Genera el CSV de entrada para descarga de curvas.
    Permite filtrar por misión y por estrellas pendientes (no descargadas aún).
    """
    mission = mission.upper()
    dfs = []

    if mission in ["ALL", "KEPLER"]:
        df_k = df_kepler.copy()
        if only_pending:
            df_k = filter_pending(df_k, "kepler", raw_dir=RAW_DIR)
        dfs.append(df_k)

    if mission in ["ALL", "TESS"]:
        df_t = df_tess.copy()
        if only_pending:
            df_t = filter_pending(df_t, "tess", raw_dir=RAW_DIR)
        dfs.append(df_t)

    df_total = pd.concat(dfs, ignore_index=True)

    # Asegurar columna 'clase_variable'
    if "clase_variable" not in df_total.columns:
        df_total["clase_variable"] = "EB"

    # Guardar CSV
    LIST_IDS.parent.mkdir(parents=True, exist_ok=True)
    LIST_IDS.write_text("")  # opcional: vaciar antes
    df_total[["id", "mission", "clase_variable"]].to_csv(LIST_IDS, index=False)

    print(f"📝 CSV generado con {len(df_total)} estrellas → {LIST_IDS}", flush=True)
    return df_total
   
   
def main(mission="ALL", only_pending=True, max_workers=8, use_sample=False):
    t0 = time.perf_counter()

    # Paso 1: descarga de los catálogos de Kepler y TESS
    if use_sample:
        print("[⬇] Descargando catálogos de pruebas de Kepler y TESS...", flush=True)
        df_kepler, df_tess = download_sample_catalogs()
    else:
        print("[⬇] Descargando catálogos completos de Kepler y TESS...", flush=True)
        df_kepler, df_tess = download_catalogs()

    # Paso 2: generar CSV de entrada para el downloader
    print("[⬇] Generando CSV de entrada para descarga de curvas...", flush=True)
    #df_ids = generar_csv_descarga(df_kepler, df_tess)
    df_ids = generar_csv_descarga(df_kepler, df_tess, mission=mission, only_pending=only_pending)

    # Paso 3: descarga de curvas por misión
    print("[⬇] Descargando curvas de luz...", flush=True)
    missions = ["Kepler", "TESS"] if mission.upper() == "ALL" else [mission]

    for m in missions:
        print(f"\n🚀 Procesando misión: {m} (only_pending={only_pending})", flush=True)
        df_m = filter_pending(df_ids, m, RAW_DIR) if only_pending else df_ids[df_ids["mission"] == m]
        out_csv = Path(f"data/lists/eb_ids_{m.lower()}_pendientes.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_m.to_csv(out_csv, index=False)
        print(f"📝 CSV generado con {len(df_m)} estrellas → {out_csv}", flush=True)
        download_from_csv_parallel(str(out_csv), base_output_dir=str(RAW_DIR), max_workers=max_workers)

    # Paso 4: lectura y fusión de las curvas descargadas
    print("[⭢] Leyendo y fusionando curvas descargadas...", flush=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if mission.upper() == "KEPLER":
        read_and_merge_curves(RAW_DIR / "kepler", output_path=PROCESSED_DIR / "dataset_eb_kepler.parquet", df_catalog=df_kepler)
    elif mission.upper() == "TESS":
        read_and_merge_curves(RAW_DIR / "tess", output_path=PROCESSED_DIR / "dataset_eb_tess.parquet", df_catalog=df_tess)
    else:
        read_and_merge_curves(RAW_DIR / "kepler", output_path=PROCESSED_DIR / "dataset_eb_kepler.parquet", df_catalog=df_kepler)
        read_and_merge_curves(RAW_DIR / "tess", output_path=PROCESSED_DIR / "dataset_eb_tess.parquet", df_catalog=df_tess)

    print(f"[⏱] Tiempo total: {time.perf_counter() - t0:.2f} segundos", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descargar y procesar curvas de Kepler/TESS")
    parser.add_argument("--mission", type=str, default="ALL", help="Kepler, TESS o ALL")
    parser.add_argument("--only_pending", action="store_true", help="Solo descargar estrellas pendientes")
    parser.add_argument("--max_workers", type=int, default=8, help="Número de hilos paralelos")
    parser.add_argument("--use_sample", action="store_true", help="Usar catálogos de prueba")
    args = parser.parse_args()

    main(mission=args.mission, only_pending=args.only_pending, max_workers=args.max_workers, use_sample=args.use_sample)