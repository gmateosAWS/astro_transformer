import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from io import StringIO
from utils.normalization_dict import normalize_label
from utils.inspect_and_export_summary import inspect_and_export_summary

# === Configuración ===
CATALOG_PATH = Path("catalogs/ztf_variable_candidates.tsv")
OUTPUT_DIR = Path("data/processed")
TEMP_CURVES_DIR = OUTPUT_DIR / "ztf_curves"
OUTPUT_PARQUET = OUTPUT_DIR / "dataset_ztf_labeled.parquet"
CLASSES_OBJETIVO = ["Cataclysmic", "White Dwarf", "RR Lyrae", "Young Stellar Object", "Variable"]
RADIUS_ARCSEC = 10.0

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
    print(f"📂 Cargando catálogo desde: {catalog_path}")
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

# === Descargar curvas desde https://ztf.snad.space ===
def descargar_curvas_snad(df_filtrado, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://db.ztf.snad.space/api/v3/data/latest/circle/full/json"
    session = requests.Session()

    for _, row in tqdm(df_filtrado.iterrows(), total=len(df_filtrado), desc="⬇ Descargando curvas ZTF"):
        ra = row["ra"]
        dec = row["dec"]
        id_objeto = row["id_objeto"]
        clase = row["clase_variable_normalizada"]

        params = {
            "ra": ra,
            "dec": dec,
            "radius_arcsec": RADIUS_ARCSEC
        }

        try:
            response = session.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if not data or "data" not in data or len(data["data"]) == 0:
                print(f"⚠️ Sin datos para {id_objeto}")
                continue

            df_all = pd.DataFrame(data["data"])
            df_all = df_all.rename(columns={"mjd": "tiempo", "mag": "magnitud", "fid": "band"})
            df_all["band"] = df_all["band"].map({1: "g", 2: "r"}).fillna("unknown")
            df_all["id_objeto"] = id_objeto
            df_all["clase_variable_normalizada"] = clase

            for band in df_all["band"].unique():
                df_band = df_all[df_all["band"] == band]
                if not df_band.empty:
                    filename = output_dir / f"{id_objeto}_{band}.csv"
                    df_band[["tiempo", "magnitud", "id_objeto", "clase_variable_normalizada", "band"]].to_csv(filename, index=False)

        except Exception as e:
            print(f"❌ Error con {id_objeto}: {e}")

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

# === MAIN ===
def main():
    df = preparar_catalogo(CATALOG_PATH)
    descargar_curvas_snad(df, TEMP_CURVES_DIR)
    consolidar_csvs(TEMP_CURVES_DIR, OUTPUT_PARQUET)
    inspect_and_export_summary(OUTPUT_PARQUET, output_format="csv")

if __name__ == "__main__":
    main()
