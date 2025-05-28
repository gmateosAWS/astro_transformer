import os
import csv
from datetime import datetime
from collections import Counter
import pyarrow.dataset as ds  # ⬅️ ESTA ES LA CLAVE QUE FALTA
from tqdm import tqdm


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

    columns_to_load = ["clase_variable", "id_objeto"]
    has_normalized = "clase_variable_normalizada" in schema.names
    if has_normalized:
        columns_to_load.append("clase_variable_normalizada")

    for batch in tqdm(dataset.to_batches(columns=columns_to_load), desc="🧮 Procesando por lotes"):
        summary["total_rows"] += batch.num_rows
        if "clase_variable" in batch.schema.names:
            clases = batch.column("clase_variable").to_pylist()
            counter_original.update(clases)
        if has_normalized:
            clases_norm = batch.column("clase_variable_normalizada").to_pylist()
            counter_normalized.update(clases_norm)
        if "id_objeto" in batch.schema.names:
            objetos.update(batch.column("id_objeto").to_pylist())

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

        max_len = max(len(counter_original), len(counter_normalized))
        orig_items = list(counter_original.items())
        norm_items = list(counter_normalized.items())

        for i in range(max_len):
            clase_orig, count_orig = orig_items[i] if i < len(orig_items) else ("", "")
            clase_norm, count_norm = norm_items[i] if i < len(norm_items) else ("", "")
            writer.writerow([clase_orig, count_orig, clase_norm, count_norm])

    # === Exportar resumen TXT ===
    with open(os.path.join(output_dir, f"{basename}_summary_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Fichero: {summary['file']}\n")
        f.write(f"Filas totales: {summary['total_rows']}\n")
        f.write(f"Curvas únicas (id_objeto): {summary['total_objects']}\n")
        f.write(f"Columnas: {list(summary['columns'].keys())}\n")
        f.write(f"Fecha: {summary['timestamp']}\n")

    print(f"✅ Resumen exportado a: {output_path}")
