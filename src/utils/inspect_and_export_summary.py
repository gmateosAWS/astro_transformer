import os
import csv
from datetime import datetime
from collections import Counter
import pyarrow.dataset as ds
from tqdm import tqdm

# Importa la función de normalización
from src.utils.normalization_dict import normalize_label

def inspect_and_export_summary(parquet_path, output_format="csv"):
    print(f"\n📁 Inspeccionando: {parquet_path}")
    dataset = ds.dataset(parquet_path, format="parquet")
    schema = dataset.schema

    summary = {
        "file": parquet_path,
        "columns": {field.name: str(field.type) for field in schema},
        "class_distribution": {},
        "class_distribution_normalized": {},
        "total_rows": 0,
        "total_objects": 0,
        "timestamp": datetime.now().isoformat()
    }

    counter_original = Counter()
    counter_normalized = Counter()
    objetos = set()

    # Detectar columnas disponibles y alias
    columns_to_load = []
    id_col = None
    for candidate in ["id_objeto", "id"]:
        if candidate in schema.names:
            id_col = candidate
            columns_to_load.append(candidate)
            break

    # Buscar ambas columnas de clase
    has_clase_variable = "clase_variable" in schema.names
    has_normalized = "clase_variable_normalizada" in schema.names

    if has_clase_variable:
        columns_to_load.append("clase_variable")
    if has_normalized:
        columns_to_load.append("clase_variable_normalizada")

    for batch in tqdm(dataset.to_batches(columns=columns_to_load), desc="🧮 Procesando por lotes"):
        summary["total_rows"] += batch.num_rows
        # Extrae ambas columnas si existen
        col_var = batch.column("clase_variable").to_pylist() if has_clase_variable and "clase_variable" in batch.schema.names else None
        col_norm = batch.column("clase_variable_normalizada").to_pylist() if has_normalized and "clase_variable_normalizada" in batch.schema.names else None

        # Selecciona la columna que tenga valores no nulos
        if col_norm is not None and any(v not in [None, "", "nan", "NaN"] for v in col_norm):
            clases = [v for v in col_norm if v not in [None, "", "nan", "NaN"]]
        elif col_var is not None:
            clases = [v for v in col_var if v not in [None, "", "nan", "NaN"]]
        else:
            clases = []

        counter_original.update(clases)
        clases_norm = [normalize_label(c) for c in clases]
        counter_normalized.update(clases_norm)
        if id_col:
            objetos.update(batch.column(id_col).to_pylist())

    summary["class_distribution"] = dict(counter_original)
    summary["class_distribution_normalized"] = dict(counter_normalized)
    summary["total_objects"] = len(objetos)

    output_dir = "data/processed/summary"
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(parquet_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_summary.csv")

    # === Exportar único fichero CSV combinado ===
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Clase (sin normalizar)", "Recuento", "Clase normalizada", "Recuento"])

        # Solo escribir si hay datos
        if counter_original or counter_normalized:
            all_keys = set(counter_original.keys()) | set(counter_normalized.keys())
            for key in sorted(all_keys, key=lambda x: str(x)):
                count_orig = counter_original.get(key, "")
                norm_key = normalize_label(key)
                count_norm = counter_normalized.get(norm_key, "")
                writer.writerow([key, count_orig, norm_key, count_norm])
        else:
            print("⚠️ No se encontraron clases para exportar en el resumen.")

    # === Exportar resumen TXT ===
    with open(os.path.join(output_dir, f"{basename}_summary_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Fichero: {summary['file']}\n")
        f.write(f"Filas totales: {summary['total_rows']}\n")
        f.write(f"Curvas únicas (id_objeto): {summary['total_objects']}\n")
        f.write(f"Columnas: {list(summary['columns'].keys())}\n")
        f.write(f"Fecha: {summary['timestamp']}\n")

    print(f"✅ Resumen exportado a: {output_path}")
