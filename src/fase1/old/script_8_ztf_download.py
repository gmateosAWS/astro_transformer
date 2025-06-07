import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from astroquery.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from utils.normalization_dict import normalize_label
from utils.inspect_and_export_summary import inspect_and_export_summary
from io import StringIO

# === Configuración ===
CATALOG_PATH = Path("catalogs/ztf_variable_candidates.tsv")
OUTPUT_DIR = Path("data/processed")
TEMP_CURVES_DIR = OUTPUT_DIR / "ztf_curves"
OUTPUT_PARQUET = OUTPUT_DIR / "dataset_ztf_labeled.parquet"
CLASSES_OBJETIVO = ["Cataclysmic", "White Dwarf", "RR Lyrae", "Young Stellar Object", "Variable"]
BANDS = ["zg", "zr"]  # nomenclatura oficial en IRSA

Irsa.ROW_LIMIT = -1  # sin límite

# === Leer fichero TSV en formato VOTable embebido
def cargar_votable_csv_local(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find("<![CDATA[") + len("<![CDATA[")
    end = content.find("]]></CSV>")
    csv_text = content[start:end].strip()
    df = pd.read_csv(StringIO(csv_text), sep=";", low_memory=False)
    return df

# === Paso 1: Cargar catálogo y filtrar
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

    # Convertir coordenadas a numérico y eliminar filas inválidas
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["ra", "dec"])

    df_filtrado = df[df["clase_variable_normalizada"].isin(CLASSES_OBJETIVO)].reset_index(drop=True)
    print(f"✅ Filtradas {len(df_filtrado)} curvas con clases objetivo: {CLASSES_OBJETIVO}")
    return df_filtrado[["id_objeto", "ra", "dec", "clase_variable_normalizada"]]


# === Paso 2: Descargar curvas desde IRSA por coordenadas
def descargar_curvas_irsa(df_filtrado, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(df_filtrado.iterrows(), total=len(df_filtrado), desc="⬇ Descargando curvas ZTF"):
        object_id = row["id_objeto"]
        ra = row["ra"]
        dec = row["dec"]
        clase = row["clase_variable_normalizada"]
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        for band in BANDS:
            try:
                table = Irsa.query_region(coord, catalog="ztf_lightcurves", spatial="Cone", radius=2 * u.arcsec)
                filtered = table[(table["filtercode"] == band)]
                if len(filtered) == 0:
                    print(f"⚠️ Sin datos para {object_id} en banda {band}")
                    continue
                df = filtered.to_pandas()
                df = df.rename(columns={"mjd": "tiempo", "mag": "magnitud"})
                df["id_objeto"] = object_id
                df["clase_variable_normalizada"] = clase
                df["band"] = band
                filename = output_dir / f"{object_id}_{band}.csv"
                df[["tiempo", "magnitud", "id_objeto", "clase_variable_normalizada", "band"]].to_csv(filename, index=False)
            except Exception as e:
                print(f"❌ Error con {object_id} ({band}): {e}")

# === Paso 3: Consolidar en .parquet final
def consolidar_csvs(output_dir, output_parquet):
    print(f"🛠️ Consolidando archivos .csv → {output_parquet}")
    csvs = list(output_dir.glob("*.csv"))
    if not csvs:
        raise RuntimeError("❌ No se encontraron archivos .csv a consolidar.")

    dfs = []
    for path in csvs:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error leyendo {path.name}: {e}")

    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_parquet(output_parquet, index=False)
    print(f"✅ Consolidación completa: {len(df_final)} curvas → {output_parquet}")

    for path in csvs:
        path.unlink()
    print("🧹 Archivos temporales eliminados.")

# === MAIN
def main():
    df = preparar_catalogo(CATALOG_PATH)
    descargar_curvas_irsa(df, TEMP_CURVES_DIR)
    consolidar_csvs(TEMP_CURVES_DIR, OUTPUT_PARQUET)
    inspect_and_export_summary(OUTPUT_PARQUET, output_format="csv")

if __name__ == "__main__":
    main()
