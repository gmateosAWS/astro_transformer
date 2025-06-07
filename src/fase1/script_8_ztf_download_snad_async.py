import pandas as pd
import aiohttp
import asyncio
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from io import StringIO
from utils.normalization_dict import normalize_label
from utils.inspect_and_export_summary import inspect_and_export_summary
from collections import Counter
import os


# === Configuración ===
CATALOG_PATH = Path("catalogs/ztf_variable_candidates.tsv")
OUTPUT_DIR = Path("data/processed")
TEMP_CURVES_DIR = OUTPUT_DIR / "ztf_curves"
OUTPUT_PARQUET = OUTPUT_DIR / "dataset_ztf_labeled.parquet"
#CLASES_OBJETIVO = ["Cataclysmic", "White Dwarf", "RR Lyrae", "Young Stellar Object", "Variable"]
#CLASES_OBJETIVO = ["Delta Scuti", "Irregular"]
CLASES_OBJETIVO = ["Delta Scuti", "Rotational", "Irregular"]
BASE_URL = "https://db.ztf.snad.space/api/v3/data/latest/circle/full/json"

RADIUS_ARCSEC = 10.0
MAX_CONCURRENT_REQUESTS = 5  # puedes ajustar entre 10–50 según red


# === Leer fichero TSV en formato VOTable embebido ===
def cargar_votable_csv_local(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find("<![CDATA[") + len("<![CDATA[")
    end = content.find("]]></CSV>")
    csv_text = content[start:end].strip()
    df = pd.read_csv(StringIO(csv_text), sep=";", low_memory=False)
    return df

# === Preparar catálogo y filtrar ===
def preparar_catalogo(catalog_path):
    print(f"\U0001F4C2 Cargando catálogo desde: {catalog_path}")
    df = cargar_votable_csv_local(catalog_path)
    df = df.rename(columns={
        "ID": "id_objeto",
        "RAJ2000": "ra",
        "DEJ2000": "dec",
        "Type": "clase_variable"
    })
    df["clase_variable_normalizada"] = df["clase_variable"].apply(normalize_label)
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["ra", "dec"])
    df_filtrado = df[df["clase_variable_normalizada"].isin(CLASSES_OBJETIVO)].reset_index(drop=True)
    print(f"✅ Filtradas {len(df_filtrado)} curvas con clases objetivo: {CLASSES_OBJETIVO}")
    return df_filtrado[["id_objeto", "ra", "dec", "clase_variable_normalizada"]]

# === Descargar curvas asincrónicamente ===
async def fetch_curve(session, row, output_dir, sem):
    id_objeto = row["id_objeto"]
    ra = row["ra"]
    dec = row["dec"]
    clase = row["clase_variable_normalizada"]

    # Saltar si ya existe algún CSV para este objeto
    filename_pattern = f"{id_objeto}_*.csv"
    if list(output_dir.glob(filename_pattern)):
        #print(f"✅ Ya existe: {id_objeto}")
        return f"✅ Ya existe: {id_objeto}"

    params = {"ra": ra, "dec": dec, "radius_arcsec": RADIUS_ARCSEC}

    try:
        async with sem:
            try:
                async with session.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        print(f"❌ HTTP {resp.status} para {id_objeto}")
                        return f"❌ HTTP {resp.status} - {id_objeto}"
                    data = await resp.json()
            except Exception as e:
                print(f"❌ Excepción HTTP para {id_objeto}: {e}")
                return f"❌ Excepción con {id_objeto}: {e}"

        if not data:
            print(f"⚠️ Sin datos para {id_objeto}")
            return f"⚠️ Sin datos para {id_objeto}"

        curvas_guardadas = 0
        for object_id, entry in data.items():
            if "lc" not in entry or not entry["lc"]:
                continue
            df = pd.DataFrame(entry["lc"])
            if df.empty:
                continue
            df = df.rename(columns={"mjd": "tiempo", "mag": "magnitud"})
            df["band"] = entry["meta"].get("filter", "unknown")
            df["id_objeto"] = id_objeto
            df["clase_variable_normalizada"] = clase
            fname = output_dir / f"{id_objeto}_{df['band'].iloc[0]}.csv"
            df[["tiempo", "magnitud", "id_objeto", "clase_variable_normalizada", "band"]].to_csv(fname, index=False)
            curvas_guardadas += 1

        if curvas_guardadas == 0:
            print(f"⚠️ Sin curvas válidas para {id_objeto}")
            return f"⚠️ Sin curvas válidas para {id_objeto}"

        print(f"✅ fetch_curve terminada para id_objeto={id_objeto} ({curvas_guardadas} curvas)")
        #print(f"[{time.strftime('%X')}] FIN {id_objeto}")
        return f"⬇ OK: {id_objeto} ({curvas_guardadas} curvas)"
    except asyncio.TimeoutError:
        print(f"⏰ Timeout para {id_objeto}")
        return f"❌ Timeout para {id_objeto}"
    except Exception as e:
        print(f"❌ Excepción para {id_objeto}: {e}")
        return f"❌ Excepción para {id_objeto}: {e}"

# === Descargar curvas ZTF asincrónicamente ===

async def descargar_curvas_snad_async(df, output_dir, chunk_size=500):
    output_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    resultados = []

    async with aiohttp.ClientSession() as session:
        for start in range(0, len(df), chunk_size):
            print(f"🟦 Procesando chunk {start}-{start+chunk_size}...")
            chunk = df.iloc[start:start+chunk_size]
            tareas = [
                asyncio.create_task(fetch_curve(session, row, output_dir, sem))
                for _, row in chunk.iterrows()
            ]
            print(f"🟩 Lanzadas {len(tareas)} tareas para este chunk.")
            for f in tqdm_async.as_completed(
                tareas,
                total=len(tareas),
                desc=f"⬇ Descargando ZTF {start}-{start+len(chunk)}",
                leave=True
            ):
                try:
                    resultado = await f
                    resultados.append(resultado)
                except Exception as e:
                    resultados.append(f"❌ Error inesperado: {e}")

# === Consolidar CSVs en parquet final ===
def consolidar_csvs(output_dir, output_parquet):
    csvs = list(output_dir.glob("*.csv"))
    if not csvs:
        raise RuntimeError("❌ No se encontraron archivos .csv.")

    dfs = [pd.read_csv(path) for path in csvs]
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_parquet(output_parquet, index=False)
    print(f"✅ Consolidado: {len(df_final)} filas → {output_parquet}")

    for path in csvs:
        path.unlink()
    print("🧹 Temporales eliminados.")

# === Función asíncrona para ejecutar el proceso completo ===
async def run_async_download(df=None, chunk_size = 5):
    if df is None:
        df = preparar_catalogo(CATALOG_PATH)
    await descargar_curvas_snad_async(df, TEMP_CURVES_DIR, chunk_size)
    consolidar_csvs(TEMP_CURVES_DIR, OUTPUT_PARQUET)
    inspect_and_export_summary(OUTPUT_PARQUET, output_format="csv")

# === MAIN ===
def main():
    df = preparar_catalogo(CATALOG_PATH)
    try:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(descargar_curvas_snad_async(df, TEMP_CURVES_DIR))
    except RuntimeError:
        asyncio.run(descargar_curvas_snad_async(df, TEMP_CURVES_DIR))
    consolidar_csvs(TEMP_CURVES_DIR, OUTPUT_PARQUET)
    inspect_and_export_summary(OUTPUT_PARQUET, output_format="csv")


if __name__ == "__main__":
    main()

