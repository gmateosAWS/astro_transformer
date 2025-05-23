# script_3c_clean_vsx_tess_classes.py
import pandas as pd
import os
import pyarrow.dataset as ds
from collections import Counter
from tqdm import tqdm
from datetime import datetime
import json
import csv

INPUT_PATH = "data/processed/dataset_vsx_tess_labeled.parquet"
OUTPUT_PATH = "data/processed/dataset_vsx_tess_labeled_clean.parquet"
SUMMARY_DIR = "data/processed/summary"

# Normalización de clases (reutilizable y extensible)
def normalizar_clase(clase):
    if not isinstance(clase, str):
        return "UNKNOWN"
    clase = clase.strip().upper()

    if clase in ["", "UNKNOWN", ","]:
        return "UNKNOWN"
    if "ROT" in clase:
        return "Rotational"
    if clase.startswith("RS"):
        return "RS_CVn"
    if clase.startswith("BY"):
        return "BY_Dra"
    if "DSCT" in clase:
        return "Delta_Scuti"
    if "RR" in clase:
        return "RR_Lyrae"
    if "EB" in clase or "EA" in clase or "ECLIPSING" in clase or clase.startswith("E"):
        return "Eclipsing"
    if "SR" in clase or "M" in clase or "LPV" in clase or "LB" in clase:
        return "Irregular"
    if "CV" in clase or "UG" in clase or "NL" in clase:
        return "Cataclysmic"
    if "WD" in clase:
        return "White_Dwarf"
    if "ACV" in clase:
        return "ACV"
    if "BCEP" in clase or "SPB" in clase:
        return "Beta_Cep"
    if "GDOR" in clase:
        return "Gamma_Dor"
    if "HADS" in clase:
        return "Delta_Scuti"
    if "S" == clase:
        return "Irregular"
    if "L" == clase:
        return "Irregular"
    if "VAR" in clase or "MISC" in clase:
        return "Irregular"
    if "YSO" in clase:
        return "YSO"
    if "WD" in clase:
        return "White_Dwarf"
    return "RARE"

# Limpieza del archivo y aplicación de la normalización
def limpiar_dataset():
    df = pd.read_parquet(INPUT_PATH)
    print(f"📥 Leídas {len(df)} filas desde {INPUT_PATH}")

    df["clase_variable"] = df["clase_variable"].fillna("UNKNOWN")
    df["clase_variable_normalizada"] = df["clase_variable"].apply(normalizar_clase)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Guardado dataset limpio en: {OUTPUT_PATH}")

    return OUTPUT_PATH

# Exportar resumen .csv y .txt
def inspect_and_export_summary(parquet_path, output_format="csv"):
    print(f"\n📁 Inspeccionando: {parquet_path}")
    dataset = ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema

    summary = {
        "file": parquet_path,
        "columns": {field.name: str(field.type) for field in schema},
        "class_distribution": {},
        "total_rows": 0,
        "total_objects": 0,
        "timestamp": datetime.now().isoformat()
    }

    class_counter = Counter()
    objetos = set()

    for batch in tqdm(dataset.to_batches(columns=["clase_variable_normalizada", "id_objeto"]), desc="🧮 Procesando por lotes"):
        summary["total_rows"] += batch.num_rows
        if "clase_variable_normalizada" in batch.schema.names:
            clases = batch.column("clase_variable_normalizada").to_pylist()
            class_counter.update(clases)
        if "id_objeto" in batch.schema.names:
            objetos.update(batch.column("id_objeto").to_pylist())

    summary["class_distribution"] = dict(class_counter)
    summary["total_objects"] = len(objetos)

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(parquet_path))[0]
    output_path = os.path.join(SUMMARY_DIR, f"{basename}_summary.{output_format}")

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    elif output_format == "csv":
        with open(output_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Clase", "Recuento"])
            for clase, count in class_counter.items():
                writer.writerow([clase, count])
        with open(output_path.replace(".csv", "_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Fichero: {summary['file']}\n")
            f.write(f"Filas totales: {summary['total_rows']}\n")
            f.write(f"Curvas únicas (id_objeto): {summary['total_objects']}\n")
            f.write(f"Columnas: {list(summary['columns'].keys())}\n")
            f.write(f"Fecha: {summary['timestamp']}\n")

    print(f"✅ Resumen exportado a: {output_path}")

if __name__ == "__main__":
    path = limpiar_dataset()
    inspect_and_export_summary(path, output_format="csv")
