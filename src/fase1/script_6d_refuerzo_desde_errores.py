import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import gc

INPUT_ERRORS = Path("outputs/errores_mal_clasificados_con_id.csv")
DATASET_PATHS = [
    "data/processed/all_missions_labeled.parquet",
    "data/processed/dataset_gaia_complemented_normalized.parquet"
]
OUTPUT_PARQUET = Path("data/processed/dataset_refuerzo_desde_errores.parquet")

# === Utilidad para cargar los parquet y agrupar por id_objeto ===
def load_and_group_batches(paths, max_per_class=None):
    dataset = ds.dataset(paths, format="parquet")
    scanner = dataset.scanner(columns=["id_objeto", "magnitud", "clase_variable_normalizada"], batch_size=256)

    grouped_data = {}
    class_counts = defaultdict(int)

    for batch in tqdm(scanner.to_batches(), desc="Agrupando curvas por objeto", unit="batch"):
        df = batch.to_pandas()
        df["id_objeto"] = df["id_objeto"].astype(str)
        for id_obj, group in df.groupby("id_objeto"):
            clase = group["clase_variable_normalizada"].iloc[0]
            if not isinstance(clase, str) or clase.strip() == "":
                continue
            if max_per_class is not None and class_counts[clase] >= max_per_class:
                continue
            if id_obj not in grouped_data:
                grouped_data[id_obj] = group
                class_counts[clase] += 1

    return grouped_data

# === Paso principal ===
def main():
    print(f"📥 Leyendo errores desde {INPUT_ERRORS}")
    df_errores = pd.read_csv(INPUT_ERRORS)
    ids_a_extraer = set(df_errores["id_objeto"].dropna().astype(str).unique())
    print(f"🔍 IDs a extraer: {len(ids_a_extraer):,}")

    print("📦 Cargando datasets originales...")
    grouped_data = load_and_group_batches(DATASET_PATHS)

    print("🎯 Extrayendo solo curvas asociadas a errores...")
    seleccionadas = [grouped_data[oid] for oid in ids_a_extraer if oid in grouped_data]
    if not seleccionadas:
        print("⚠️ No se encontraron coincidencias. Proceso abortado.")
        return

    df_resultado = pd.concat(seleccionadas, ignore_index=True)
    df_resultado.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✅ Dataset de refuerzo guardado en: {OUTPUT_PARQUET} ({len(df_resultado):,} filas)")

if __name__ == "__main__":
    main()
