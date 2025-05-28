# script_6g_refuerzo_gaia_completo.py
# Descarga curvas TESS desde objetos Gaia DR3 etiquetados como clases minoritarias

import pandas as pd
import numpy as np
from pathlib import Path
from lightkurve import search_lightcurve
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import gc

CLASES_OBJETIVO = [
    "Cataclysmic", "White Dwarf", "RR Lyrae",
    "Young Stellar Object", "Irregular", "Variable", "Delta Scuti"
]

# Paths
INPUT_DIR = Path("data/processed")
FILE_SOURCE = INPUT_DIR / "dataset_gaia_dr3_vsx_tic_labeled_with_coords_clean.parquet"
EXISTING_IDS_PATH = INPUT_DIR / "all_missions_labeled.parquet"
OUTPUT_PATH = INPUT_DIR / "dataset_refuerzo_gaia_clases_minor.parquet"
SUMMARY_PATH = INPUT_DIR / "summary/clase_variable_normalizada_summary_refuerzo_gaia.csv"
SUMMARY_PATH.parent.mkdir(exist_ok=True)

# Leer ids ya existentes para evitar duplicados
print("📂 Cargando IDs ya presentes en el dataset consolidado...")
existing_ids = set(pd.read_parquet(EXISTING_IDS_PATH, columns=["id_objeto"])["id_objeto"].unique())

# Leer candidatos desde Gaia DR3
print("🔍 Cargando candidatos desde Gaia DR3...")
df = pd.read_parquet(FILE_SOURCE)
required = {"id_objeto", "tic_id", "clase_variable_normalizada", "ra", "dec"}
if not required.issubset(df.columns):
    raise ValueError(f"❌ Faltan columnas requeridas en {FILE_SOURCE.name}: {required - set(df.columns)}")

df = df[df["clase_variable_normalizada"].isin(CLASES_OBJETIVO)]
df = df[~df["id_objeto"].isin(existing_ids)]
df_total = df.drop_duplicates("id_objeto").reset_index(drop=True)
print(f"🎯 Total candidatos únicos desde Gaia: {len(df_total)}")

# Descargar curvas reales desde TESS
curvas = []
labels = []
ids = []
masks = []

print("📥 Descargando curvas TESS con Lightkurve...")
for _, row in tqdm(df_total.iterrows(), total=len(df_total)):
    tic_id = str(row["tic_id"])
    id_objeto = row["id_objeto"]
    clase = row["clase_variable_normalizada"]
    ra = row["ra"]
    dec = row["dec"]

    try:
        search = search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search) == 0 and not pd.isna(ra) and not pd.isna(dec):
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            search = search_lightcurve(coord, radius=0.01 * u.deg, mission="TESS")
        if search is None or len(search) == 0:
            continue

        lcfs = search.download_all()
        if lcfs is None:
            continue
        for lc in lcfs:
            df = lc.to_table().to_pandas()
            df = df[["time", "flux", "flux_err"]].dropna()
            if len(df) < 500:
                continue
            curvas.append(df["flux"].values)
            labels.append(clase)
            ids.append(id_objeto)
            masks.append(np.ones(len(df), dtype=bool))
            break  # solo una curva por objeto
    except Exception:
        continue

print(f"✅ Curvas válidas descargadas: {len(curvas)}")

# Guardar a Parquet
if curvas:
    df_out = pd.DataFrame({
        "id_objeto": ids,
        "clase_variable_normalizada": labels,
        "flux": curvas,
        "mask": masks
    })
    table = pa.Table.from_pandas(df_out)
    pq.write_table(table, OUTPUT_PATH)
    df_out["clase_variable_normalizada"].value_counts().to_csv(SUMMARY_PATH)
    print(f"💾 Dataset guardado en: {OUTPUT_PATH}")
    print(f"📄 Resumen por clase en: {SUMMARY_PATH}")
else:
    print("⚠️ No se han descargado curvas válidas.")