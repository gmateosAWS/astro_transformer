# script_2_K2_k2varcat_everest_chunks.py
import os
import pandas as pd
import requests
from tqdm import tqdm
from lightkurve import search_lightcurve
from multiprocessing import Pool, cpu_count
import pyarrow as pa
import pyarrow.parquet as pq

RAW_CSV_PATH = "catalogs/k2varcat_official.csv"
CSV_URL = "https://cygnus.astro.warwick.ac.uk/phrlbj/k2varcat/K2VarCat.csv"
CHUNKS_DIR = "data/processed/k2varcat_chunks/"
FINAL_PARQUET = "data/processed/dataset_k2varcat_labeled.parquet"

# === DESCARGA CSV SI NO EXISTE ===
def descargar_catalogo_k2varcat(destino, url):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino):
        print("📥 Descargando catálogo oficial K2VARCAT...")
        r = requests.get(url)
        if not r.ok:
            raise RuntimeError(f"❌ Error al descargar CSV desde {url}")
        with open(destino, "wb") as f:
            f.write(r.content)
        print(f"✅ Guardado en: {destino}")
    else:
        print(f"✅ Catálogo ya descargado: {destino}")

# === CARGA CON CABECERA MANUAL ===
def cargar_catalogo_k2varcat(csv_path, limit=None):
    columnas = ["EPIC_ID", "K2_TYPE", "RANGE_PCT", "PERIOD_DAYS", "AMPLITUDE_PCT", "PROPOSAL_INFO", "AMP_ERR1", "AMP_ERR2"]
    df = pd.read_csv(csv_path, names=columnas, header=None)
    df = df[["EPIC_ID", "K2_TYPE"]].dropna()
    df = df.rename(columns={"EPIC_ID": "id_objeto", "K2_TYPE": "clase_variable"})
    df["id_objeto"] = df["id_objeto"].astype(int)
    if limit:
        df = df.head(limit * 10)
    return df

# === FILTRO POR CURVAS EN EVEREST ===
def filtrar_epics_con_everest(df, max_validos=10):
    seleccionados = []
    for epic_id, clase in tqdm(df.itertuples(index=False, name=None)):
        result = search_lightcurve(f"EPIC {epic_id}", mission="K2", author="EVEREST")
        if result and len(result) > 0:
            seleccionados.append((epic_id, clase))
            print(f"✅ {epic_id} tiene curvas EVEREST")
        if len(seleccionados) >= max_validos:
            break
    return pd.DataFrame(seleccionados, columns=["id_objeto", "clase_variable"])

# === PROCESAMIENTO INDIVIDUAL ===
def procesar_y_guardar_curva(args):
    epic_id, clase = args
    output_path = os.path.join(CHUNKS_DIR, f"k2_{epic_id}.parquet")
    if os.path.exists(output_path):
        return "🟡"
    try:
        result = search_lightcurve(f"EPIC {epic_id}", mission="K2", author="EVEREST")
        lcs = result.download_all()
        if not lcs:
            return "❌"

        registros = []
        for lc in lcs:
            if not hasattr(lc, "flux") or lc.flux is None:
                continue

            df = lc.remove_nans().normalize().to_pandas()
            colmap = {}
            if "cadn" in df.columns and "fraw" in df.columns:
                colmap = {"cadn": "tiempo", "fraw": "magnitud", "fraw_err": "error"}
            elif "time" in df.columns and "flux" in df.columns:
                colmap = {"time": "tiempo", "flux": "magnitud", "flux_err": "error"}
            df = df.rename(columns=colmap)

            if not {"tiempo", "magnitud", "error"}.issubset(df.columns):
                return "❌"

            df["id_objeto"] = epic_id
            df["id_mision"] = f"K2_{epic_id}"
            df["mision"] = "K2"
            df["fecha_inicio"] = lc.time.min().value
            df["fecha_fin"] = lc.time.max().value
            df["clase_variable"] = clase
            df["origen_etiqueta"] = "K2VARCAT"
            registros.append(df)

        if registros:
            final_df = pd.concat(registros, ignore_index=True)
            final_df.to_parquet(output_path, index=False)
            return "✅"
        return "❌"
    except Exception:
        return "❌"

# === UNION FINAL DE TODOS LOS PARQUETS ===
def unir_parquets_chunks(directorio_chunks, output_final):
    files = [os.path.join(directorio_chunks, f) for f in os.listdir(directorio_chunks) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(f) for f in files]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_parquet(output_final, index=False)
    print(f"✅ Dataset final guardado en: {output_final}")

# === MAIN ===
def main(limit=None, workers=4):
    descargar_catalogo_k2varcat(RAW_CSV_PATH, CSV_URL)
    df_k2 = cargar_catalogo_k2varcat(RAW_CSV_PATH, limit=limit)
    if limit:
        df_k2 = filtrar_epics_con_everest(df_k2, max_validos=limit)

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    with Pool(processes=workers) as pool:
        resultados = list(tqdm(
            pool.imap(procesar_y_guardar_curva, df_k2.itertuples(index=False, name=None)),
            total=len(df_k2),
            desc="📦 Procesando curvas EVEREST"
        ))

    print("Resumen de estado:", pd.Series(resultados).value_counts().to_dict())
    unir_parquets_chunks(CHUNKS_DIR, FINAL_PARQUET)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de curvas para test")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()), help="Número de procesos paralelos")
    args = parser.parse_args()
    main(limit=args.limit, workers=args.workers)
