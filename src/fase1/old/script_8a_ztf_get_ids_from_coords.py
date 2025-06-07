from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
from pathlib import Path
from io import StringIO

# ⚙️ Función para cargar archivo con contenido <![CDATA[
def cargar_votable_csv_local(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find("<![CDATA[") + len("<![CDATA[")
    end = content.find("]]></CSV>")
    csv_text = content[start:end].strip()
    df = pd.read_csv(StringIO(csv_text), sep=";", low_memory=False)
    return df

# 📂 Configuración
CATALOG_PATH = Path("catalogs/ztf_variable_candidates.tsv")
OUTPUT_PATH = Path("catalogs/ztf_object_ids_from_coords.csv")
RADIUS_ARCSEC = 1.5

# 🧪 Cargar catálogo
df = cargar_votable_csv_local(CATALOG_PATH)
df = df[['_RAJ2000', '_DEJ2000', 'ID']].dropna().drop_duplicates()

# ✅ Conversión segura de coordenadas a float
df['_RAJ2000'] = pd.to_numeric(df['_RAJ2000'], errors='coerce')
df['_DEJ2000'] = pd.to_numeric(df['_DEJ2000'], errors='coerce')
df = df.dropna(subset=['_RAJ2000', '_DEJ2000'])

# 🔍 Consultas a IRSA
results = []
for i, row in df.iterrows():
    ra, dec, id_objeto = row['_RAJ2000'], row['_DEJ2000'], row['ID']
    try:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        table = Irsa.query_region(coord, catalog='ztf_objects', spatial='Cone', radius=RADIUS_ARCSEC * u.arcsec)
        if len(table) > 0:
            results.append({
                '_RAJ2000': ra,
                '_DEJ2000': dec,
                'id_objeto': id_objeto,
                'ztf_object_id': table['ztf_object_id'][0]
            })
    except Exception as e:
        print(f"❌ Error al consultar RA={ra}, DEC={dec} ({id_objeto}): {e}")

# 💾 Guardar resultados
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ IDs ZTF guardados en {OUTPUT_PATH}")
