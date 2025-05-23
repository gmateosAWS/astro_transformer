import os
import pandas as pd
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from lightkurve import search_lightcurve

# === CONFIGURACIÓN ===
RAW_CSV_PATH = "catalogs/k2varcat_official.csv"
CSV_URL = "https://cygnus.astro.warwick.ac.uk/phrlbj/k2varcat/K2VarCat.csv"
OUTPUT_PARQUET = "data/processed/dataset_k2varcat_labeled.parquet"

# === DESCARGA DEL CSV ===
def descargar_catalogo_k2varcat(destino, url):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino):
        print(f"📥 Descargando catálogo oficial K2VARCAT...")
        r = requests.get(url)
        if not r.ok:
            raise RuntimeError(f"❌ Error al descargar CSV desde {url}")
        with open(destino, "wb") as f:
            f.write(r.content)
        print(f"✅ Catálogo guardado en: {destino}")
    else:
        print(f"✅ Catálogo ya descargado: {destino}")

# === CARGA CON CABECERAS MANUALES ===
def cargar_catalogo_k2varcat(csv_path, limit=None):
    columnas = [
        "EPIC_ID", "K2_TYPE", "RANGE_PCT", "PERIOD_DAYS",
        "AMPLITUDE_PCT", "PROPOSAL_INFO", "AMP_ERR1", "AMP_ERR2"
    ]
    df = pd.read_csv(csv_path, names=columnas, header=None)
    df = df[["EPIC_ID", "K2_TYPE"]].dropna()
    df = df.rename(columns={"EPIC_ID": "id_objeto", "K2_TYPE": "clase_variable"})
    df["id_objeto"] = df["id_objeto"].astype(int)
    if limit:
        df = df.head(limit * 10)  # Sobreselección para filtrar después
    return df

# === PRE-FILTRADO DE IDS CON CURVAS ===
def obtener_ids_con_datos(df, max_validos=10):
    seleccionados = []
    for i, (epic_id, clase) in enumerate(df.itertuples(index=False, name=None)):
        resultado = search_lightcurve(f"EPIC {epic_id}", mission="K2")
        if resultado is not None and len(resultado) > 0:
            seleccionados.append((epic_id, clase))
            print(f"✅ {epic_id} tiene datos")
        if len(seleccionados) >= max_validos:
            break
    return pd.DataFrame(seleccionados, columns=["id_objeto", "clase_variable"])

# === DESCARGA DE UNA CURVA INDIVIDUAL ===
def descargar_una_curva(args):
    epic_id, clase = args
    try:
        star_id = f"EPIC {epic_id}"
        lc_search = search_lightcurve(star_id, mission="K2")
        if lc_search is None or len(lc_search) == 0:
            return None

        lcs = lc_search.download_all()
        if lcs is None or len(lcs) == 0:
            return None

        registros = []
        for lc in lcs:
            if not hasattr(lc, "flux") or lc.flux is None:
                continue

            lc_clean = lc.remove_nans().normalize()

            df = lc_clean.to_pandas().rename(columns={
                "time": "tiempo",
                "flux": "magnitud",
                "flux_err": "error"
            })
            df["id_objeto"] = epic_id
            df["id_mision"] = f"K2_{epic_id}"
            df["mision"] = "K2"
            df["fecha_inicio"] = lc_clean.time.min().value
            df["fecha_fin"] = lc_clean.time.max().value
            df["clase_variable"] = clase
            df["origen_etiqueta"] = "K2VARCAT"
            registros.append(df)

        if registros:
            return pd.concat(registros, ignore_index=True)
        else:
            return None

    except Exception as e:
        print(f"❌ Error con {epic_id}: {e}")
        return None


# === PROCESAMIENTO EN PARALELO ===
def procesar_lote_parallel(df_ids, output_path, num_workers=4):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    schema = pa.schema([
        ("tiempo", pa.float64()),
        ("magnitud", pa.float32()),
        ("error", pa.float32()),
        ("id_objeto", pa.int64()),
        ("id_mision", pa.string()),
        ("mision", pa.string()),
        ("fecha_inicio", pa.float64()),
        ("fecha_fin", pa.float64()),
        ("clase_variable", pa.string()),
        ("origen_etiqueta", pa.string()),
    ])
    writer = pq.ParquetWriter(output_path, schema)
    with Pool(processes=num_workers) as pool:
        resultados = list(tqdm(
            pool.imap(descargar_una_curva, df_ids.itertuples(index=False, name=None)),
            total=len(df_ids),
            desc="🔄 Descargando curvas en paralelo"
        ))
    n_exitos = 0
    for curva_df in resultados:
        if curva_df is not None and not curva_df.empty:
            batch = pa.RecordBatch.from_pandas(curva_df, schema=schema, preserve_index=False)
            writer.write_batch(batch)
            n_exitos += 1
    writer.close()
    print(f"\n✅ Guardado {n_exitos} curvas en: {output_path}")
    print(f"❌ Fallos: {len(df_ids) - n_exitos}")

# === MAIN INVOCABLE ===
def main(limit=None, workers=4):
    descargar_catalogo_k2varcat(RAW_CSV_PATH, CSV_URL)
    df_k2 = cargar_catalogo_k2varcat(RAW_CSV_PATH, limit=limit)
    if limit:
        df_k2 = obtener_ids_con_datos(df_k2, max_validos=limit)
    procesar_lote_parallel(df_k2, OUTPUT_PARQUET, num_workers=workers)

# === LLAMADA POR TERMINAL ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de curvas para test")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()), help="Número de procesos paralelos")
    args = parser.parse_args()
    main(limit=args.limit, workers=args.workers)
