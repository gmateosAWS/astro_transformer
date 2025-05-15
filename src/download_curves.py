import os
import pandas as pd
from lightkurve import search_lightcurve
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def normalize_id(star_id, mission):
    if star_id.isdigit():
        if mission.lower() == "kepler":
            return f"KIC {star_id}"
        elif mission.lower() == "tess":
            return f"TIC {star_id}"
    return star_id  # Devolver sin cambios si ya incluye el prefijo

# Este script descarga curvas de luz de las misiones Kepler y TESS
# y las guarda en formato CSV.
# Toma un archivo CSV con IDs de estrellas y nombres de misiones.
def download_curve(target_id, mission, output_dir):
    try:
        star_id = normalize_id(target_id, mission)
        print(f"Descargando {star_id} ({mission})...")

        # Buscar archivos de curva de luz
        # y descargar todos los archivos disponibles
        # Cuando Lightkurve busca curvas en TESS, se puede especificar qué conjunto de curvas quieres:
        # "SPOC" (Science Processing Operations Center) → es el pipeline oficial de reducción de datos de TESS, operado por NASA: curvas de 2 minutos de cadencia reducidas oficialmente.
        # "QLP" (Quick Look Pipeline) → datos procesados por MIT, más amplios pero con menos detalle.
        if mission.lower() == "tess":
            lc_search = search_lightcurve(star_id, mission=mission, author="SPOC")
        else:
            lc_search = search_lightcurve(star_id, mission=mission)

        # Si no se encuentra la curva de luz, salir de la función
        # y mostrar un mensaje de advertencia
        if lc_search is None or len(lc_search) == 0:
            print(f"⚠️ No se encontraron datos para {star_id} ({mission})")
            return

        # Descargar todas las curvas de luz disponibles
        lcs = lc_search.download_all()
        if lcs is None:
            print(f"⚠️ No se pudieron descargar curvas para {star_id}")
            return
        # Crear el directorio de salida si no existe        
        os.makedirs(output_dir, exist_ok=True)

        # Iterar sobre los archivos de curva de luz y guardar el flujo PDCSAP_FLUX en CSV
        print(f"[🌓] {star_id} → {len(lcs)} curvas encontradas")
        for i, lc in tqdm(enumerate(lcs), total=len(lcs), desc=f"  ↪ Procesando {star_id}", leave=False):
            # Usar solo curvas etiquetadas como PDCSAP_FLUX si están disponibles
            # PDCSAP_FLUX (Presearch Data Conditioning Simple Aperture Photometry):
            #   Es el producto más limpio y corregido que ofrece Kepler/TESS.
    	    #   Se le han eliminado efectos sistemáticos como ruido instrumental, variaciones orbitales, etc.
            #   Se utiliza para obtener la curva de luz más precisa y confiable.
            #   Ideal para análisis científico, clasificación, y entrenamiento de modelos.
            if lc.label == "PDCSAP_FLUX":
                flux_type = "pdcsap"
                lc_clean = lc.remove_nans().normalize()
            # Si no hay PDCSAP_FLUX, usar SAP_FLUX (Simple Aperture Photometry):
            # SAP_FLUX: Es el flujo sin corregir, que incluye ruido y variaciones. Es menos confiable, pero puede ser útil para algunos análisis.
            elif "sap_flux" in lc.columns:
                flux_type = "sap"
                print(f"⚠️ {target_id} no tiene PDCSAP_FLUX, usando SAP_FLUX")
                lc_clean = lc.remove_nans().normalize()
            else:
                print(f"⚠️ {target_id} no tiene columnas válidas de flujo. Se omite.")
                continue

            # Extraer metadatos del sector / quarter / campaign
            meta = ""
            if "quarter" in lc.meta:
                meta = f"q{lc.meta['quarter']}"
            elif "sector" in lc.meta:
                meta = f"s{lc.meta['sector']}"
            elif "campaign" in lc.meta:
                meta = f"c{lc.meta['campaign']}"

            obs_date = lc.meta.get("DATE-OBS", f"{i}").replace(":", "-")
            filename = f"{mission.lower()}_{target_id}_{meta}_{flux_type}_{obs_date}.csv"
            path = os.path.join(output_dir, filename)

            if os.path.exists(path):
                print(f"🟡 Ya existe: {path}. Se omite.")
                continue

            lc_clean.to_csv(path)
            print(f"✅ Guardado: {path}")
    except Exception as e:
        print(f"❌ Error descargando {star_id} ({mission}): {e}")

def download_from_csv(csv_path, base_output_dir="data"):
    df = pd.read_csv(csv_path)
    print(f"[⬇] Descargando {len(df)} curvas de luz...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="⬇ Descargando curvas"):
        star_id = str(row['id']).strip()
        mission = row['mission'].strip()
        out_dir = os.path.join(base_output_dir, mission.lower())
        download_curve(star_id, mission, out_dir)

def download_from_csv_parallel(csv_path, base_output_dir="data", max_workers=8):
    """
    Descarga curvas de luz en paralelo usando múltiples hilos.
    Respeta los archivos ya descargados.
    """
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"[⬇] Descargando {total} curvas en paralelo con {max_workers} hilos...")

    def process_row(row):
        from download_curves import download_curve
        star_id = str(row["id"]).strip()
        mission = row["mission"].strip()
        out_dir = os.path.join(base_output_dir, mission.lower())
        try:
            print(f"⬇ Iniciando descarga: {star_id} ({mission})")
            download_curve(star_id, mission, out_dir)
            return star_id, "OK"
        except Exception as e:
            print(f"❌ Error con {star_id} ({mission}): {e}")
            raise  # <-- permite detener el hilo si quieres depurar duro

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row, row): row["id"]
            for _, row in df.iterrows()
        }

        results = []
        for future in tqdm(as_completed(futures), total=total, desc="🚀 Descargando curvas"):
            result = future.result()
            results.append(result)

    errors = [r for r in results if "Error" in r[1]]
    print(f"[✓] Descarga finalizada. Éxitos: {len(results)-len(errors)}  Errores: {len(errors)}")

    if errors:
        print("⚠️ Errores encontrados:")
        for star_id, err in errors[:10]:  # muestra solo los 10 primeros
            print(f" - {star_id}: {err}")