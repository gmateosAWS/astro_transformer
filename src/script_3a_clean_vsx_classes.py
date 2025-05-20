# script_3a_clean_vsx_classes.py
import pandas as pd
import os
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
import re

INPUT_PARQUET = "data/processed/dataset_vsx_tic_labeled.parquet"
OUTPUT_PARQUET = "data/processed/dataset_vsx_tic_labeled_clean.parquet"
SUMMARY_CSV = "data/processed/summary/dataset_vsx_tic_labeled_clean_summary.csv"
SUMMARY_TXT = "data/processed/summary/dataset_vsx_tic_labeled_clean_summary_info.txt"

# Diccionario de agrupación extendido
AGRUPACION = {
    "RRAB": "RR_Lyrae", "RRC": "RR_Lyrae", "RRD": "RR_Lyrae", "RR": "RR_Lyrae", "RRAB/BL": "RR_Lyrae",
    "DSCT": "Delta_Scuti", "DSCTC": "Delta_Scuti", "HADS": "Delta_Scuti", "DSCT|GDOR|SXPHE": "Delta_Scuti",
    "GDOR": "Gamma_Dor", "BCEP": "Beta_Cep", "SPB": "Beta_Cep", "BCEP+SPB": "Beta_Cep",
    "EA": "EA", "EB": "EB", "EW": "EW", "RS": "RS_CVn", "RS_CVn": "RS_CVn",
    "ROT": "Rotational", "ACV": "ACV", "BY": "BY_Dra", "BY_Dra": "BY_Dra",
    "VAR": "Irregular", "MISC": "Irregular", "LPV": "Irregular", "LC": "Irregular",
    "SRA": "Irregular", "SR": "Irregular", "SRB": "Irregular", "SRD": "Irregular", "SRS": "Irregular", "SR|M": "Irregular",
    "ZZA": "ZZ_Ceti", "WD": "White_Dwarf", "CV": "Cataclysmic", "UG": "Cataclysmic", "UGZ": "Cataclysmic",
    "UGWZ": "Cataclysmic", "UGSU": "Cataclysmic", "UGSU+E": "Cataclysmic", "CWB": "Cepheid", "DCEP": "Cepheid", "ACEP": "Cepheid",
    "ED": "Eclipsing", "ESD": "Eclipsing", "EP": "Eclipsing", "EC": "Eclipsing",
    "UNKNOWN": "UNKNOWN", "": "UNKNOWN", None: "UNKNOWN"
}

RARE_THRESHOLD = 5

# Limpieza básica + mapeo a clases conocidas
def normalizar_clase(valor):
    if pd.isna(valor) or valor.strip() == "":
        return "UNKNOWN"
    # Eliminar sufijos raros y dividir por separadores comunes
    limpio = re.split(r'[|/+,:]', valor.strip().upper())[0]
    limpio = limpio.replace("-", "").replace(" ", "")
    return AGRUPACION.get(limpio, limpio)

def main():
    print(f"📥 Leyendo: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)

    print("🔧 Normalizando clases...")
    df["clase_variable_normalizada"] = df["clase_variable"].apply(normalizar_clase)

    # Contar ocurrencias
    conteo = Counter(df["clase_variable_normalizada"])
    print(f"🔎 Clases únicas encontradas: {len(conteo)}")

    # Marcar clases raras como RARE
    df["clase_variable_normalizada"] = df["clase_variable_normalizada"].apply(
        lambda c: "RARE" if conteo[c] < RARE_THRESHOLD else c
    )

    print(f"💾 Guardando dataset limpio en: {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("📊 Exportando resumen de clases...")
    final_conteo = Counter(df["clase_variable_normalizada"])
    Path(os.path.dirname(SUMMARY_CSV)).mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("Clase,Recuento\n")
        for clase, count in final_conteo.items():
            f.write(f"{clase},{count}\n")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(f"Fichero: {OUTPUT_PARQUET}\n")
        f.write(f"Filas totales: {len(df)}\n")
        f.write(f"Curvas únicas (id_objeto): {df['id_objeto'].nunique()}\n")
        f.write(f"Columnas: {list(df.columns)}\n")

    print("✅ Proceso completado")

if __name__ == "__main__":
    main()
