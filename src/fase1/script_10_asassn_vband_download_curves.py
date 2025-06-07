import sys
import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm
import concurrent.futures
import pyarrow as pa
import pyarrow.parquet as pq
import time
import urllib.parse
from bs4 import BeautifulSoup  # <-- Añadido para scraping

# Añadir src/ al path para los imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.normalization_dict import normalize_label, VALID_CLASSES

# Configuración
CATALOG_PATH = Path("catalogs/asassn_catalog_variable_stars.csv")
TEMP_CURVES_DIR = Path("data/processed/temp/asassn_vband_curves")
OUTPUT_PARQUET = Path("data/processed/dataset_asassn_vband-unified.parquet")
ROW_GROUP_SIZE = 50000
CLASES_OBJETIVO = {"Cataclysmic", "White Dwarf", "Young Stellar Object"}

# URL base para la descarga
BASE_URL = "https://asas-sn.osu.edu/variables?"

# 1. Cargar y filtrar el catálogo
def filtrar_catalogo():
    df = pd.read_csv(CATALOG_PATH)
    df["clase_variable_normalizada"] = df["variable_type"].apply(normalize_label)
    df_filtrado = df[df["clase_variable_normalizada"].isin(CLASES_OBJETIVO)].copy()
    df_filtrado = df_filtrado.drop_duplicates(subset="source_id")
    return df_filtrado[["source_id", "asassn_name", "variable_type", "clase_variable_normalizada"]]

# 2. Descargar una curva
def descargar_curva(row):
    source_id = row.source_id
    asassn_name = row.asassn_name

    tipos = [
        "YSO", "CV", "CV%252BE", "CV%253A", "UG", "UGER", "UGSS", "UGSU", "UGSU%252BE", "UGSU%253A",
        "UGWZ", "UGZ", "AM", "AM%252BE", "AM%253A", "DQ", "DQ%253A", "ZZ", "ZZ%253A", "ZZA", "ZZB", "ZZLep", "ZZO"
    ]
    url = (
        "https://asas-sn.osu.edu/variables?"
        f"&name={urllib.parse.quote(asassn_name)}"
    )
    for t in tipos:
        url += f"&variable_type[]={t}"

    dest = TEMP_CURVES_DIR / f"{source_id}.csv"
    if dest.exists():
        print(f"✔️ Ya existe: {dest}")
        return source_id  # ya descargada

    def get_with_retries(url, max_retries=5, backoff_factor=0.5, timeout=10):
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    wait = backoff_factor * (2 ** attempt)
                    print(f"⚠️ [429 Too Many Requests] Esperando {wait:.1f}s antes de reintentar...")
                    time.sleep(wait)
                else:
                    print(f"❌ [{response.status_code}] {url}")
                    return None
            except Exception as e:
                wait = backoff_factor * (2 ** attempt)
                print(f"⚠️ [{type(e).__name__}] Esperando {wait:.1f}s antes de reintentar...")
                time.sleep(wait)
        print(f"❌ Fallo persistente tras {max_retries} intentos: {url}")
        return None

    try:
        # Paso 1: Descargar la página HTML con reintentos
        response = get_with_retries(url)
        if response is None or response.status_code != 200:
            print(f"❌ No se pudo acceder a la página: {source_id} (status: {response.status_code if response else 'N/A'})")
            return None

        # Paso 2: Buscar el enlace correcto con BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        # Buscar todos los enlaces que apunten a /variables/<uuid>
        links = soup.select("a[href^='/variables/']")
        uuid_link = None
        for link in links:
            href = link.get("href", "")
            # UUID: 36 caracteres, contiene 4 guiones, no es 'lookup'
            if (
                len(href) == len("/variables/") + 36 and
                href.startswith("/variables/") and
                href != "/variables/lookup" and
                link.text.strip() == asassn_name
            ):
                uuid_link = link
                break
        #print(f"[TRACE] Enlace UUID encontrado: {uuid_link.get('href') if uuid_link else None}")
        if not uuid_link or not uuid_link.get("href"):
            print(f"❌ No se encontró enlace UUID para {asassn_name}")
            return None
        uuid_path = uuid_link.get("href")
        uuid = uuid_path.split("/variables/")[1]
        curve_url = f"https://asas-sn.osu.edu/variables/{uuid}.csv"
        #print(f"[TRACE] URL curva CSV: {curve_url}")

        # Paso 3: Descargar la curva CSV con reintentos
        response_curve = get_with_retries(curve_url)
        time.sleep(0.05)
        # if response_curve:
        #     print(f"[TRACE] Status CSV: {response_curve.status_code}, tamaño: {len(response_curve.content)}")
        if response_curve and response_curve.status_code == 200:
            # Guardar el contenido binario tal cual, para evitar problemas de codificación
            try:
                with open(dest, "wb") as f:
                    f.write(response_curve.content)
                #print(f"✅ Guardado: {dest}")
                # Verificación rápida de cabecera
                with open(dest, "r", encoding="utf-8") as f_check:
                    head = f_check.readline()
                    if "HJD" not in head and "hjd" not in head:
                        print(f"⚠️ El archivo guardado no parece tener cabecera HJD/hjd: {head.strip()}")
                return source_id
            except Exception as e:
                print(f"❌ Error guardando CSV {dest}: {e}")
                return None
        else:
            print(f"❌ No válido o sin datos HJD: {source_id} (status: {response_curve.status_code if response_curve else 'N/A'})")
            return None
    except Exception as e:
        print(f"❌ {source_id}: {e}")
        return None

# 3. Consolidar todas las curvas descargadas
def consolidar_curvas(df_metadata):
    curves = []
    for row in tqdm(df_metadata.itertuples(), total=len(df_metadata), desc="📊 Leyendo curvas"):
        file = TEMP_CURVES_DIR / f"{row.source_id}.csv"
        if not file.exists():
            continue
        try:
            df = pd.read_csv(file)
            # Adaptar para aceptar 'HJD' o 'hjd'
            if df.empty or (("HJD" not in df.columns and "hjd" not in df.columns) or "mag" not in df.columns):
                continue
            if "hjd" in df.columns:
                df.rename(columns={"hjd": "time"}, inplace=True)
            else:
                df.rename(columns={"HJD": "time"}, inplace=True)
            df["id"] = row.source_id
            df["clase_variable"] = row.variable_type
            df["clase_variable_normalizada"] = row.clase_variable_normalizada
            df["mission"] = "ASASSN"
            df["mission_id"] = "ASASSN_vband"
            df["source_dataset"] = "asassn_vband"
            df["label_source"] = "ASASSN_Catalog"
            df["band"] = "V"
            # Añadir columnas adicionales si existen
            if "mag_err" in df.columns:
                df["mag_err"] = df["mag_err"]
            if "flux" in df.columns:
                df["flux"] = df["flux"]
            if "flux_err" in df.columns:
                df["flux_err"] = df["flux_err"]
            # Seleccionar columnas según disponibilidad
            columnas = [
                "id", "time", "mag", "clase_variable", "clase_variable_normalizada",
                "mission", "mission_id", "source_dataset", "label_source", "band"
            ]
            if "mag_err" in df.columns:
                columnas.append("mag_err")
            if "flux" in df.columns:
                columnas.append("flux")
            if "flux_err" in df.columns:
                columnas.append("flux_err")
            curves.append(df[columnas])
        except Exception as e:
            print(f"❌ Error leyendo {file.name}: {e}")
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()

def main(max_workers=12, max_ejemplos=None):
    TEMP_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    print("📂 Filtrando catálogo ASAS-SN V-band...")
    df_filtrado = filtrar_catalogo()

    # Limitar el número de ejemplos para pruebas
    if max_ejemplos is not None:
        df_filtrado = df_filtrado.head(max_ejemplos)
        print(f"[TRACE] Procesando solo {len(df_filtrado)} ejemplos para pruebas.")

    print("⬇️ Descargando curvas (modo paralelo)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(descargar_curva, df_filtrado.itertuples(index=False)), total=len(df_filtrado)))

    print("🧩 Consolidando dataset...")
    df_final = consolidar_curvas(df_filtrado)
    if df_final.empty:
        print("⚠️ No se generó ningún dataset")
        return

    print(f"💾 Guardando en formato parquet ({len(df_final)} filas)...")
    table = pa.Table.from_pandas(df_final)
    pq.write_table(table, OUTPUT_PARQUET, row_group_size=ROW_GROUP_SIZE)
    print(f"✅ Guardado: {OUTPUT_PARQUET}")

    print(f"📈 Total de curvas procesadas: {df_final['id'].nunique()}")
    print(f"🧪 Clases presentes:\n{df_final['clase_variable_normalizada'].value_counts()}")


if __name__ == "__main__":
    main()
