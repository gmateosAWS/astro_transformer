import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset, DataLoader
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import random
import gc
import time  # <-- Añadido para medir tiempos

# Importar función de normalización de clases
from src.utils.normalization_dict import normalize_label
from src.utils.column_mapping import COLUMN_MAPPING, map_column_name, find_column
from src.utils.dataset_paths import DATASET_PATHS

discard_reasons = {
    "all_nan": 0,
    "low_std": 0,
    "short_curve": 0,
    "nan_or_inf_after_norm": 0,
    "ok": 0
}
MIN_POINTS = 30
MIN_STD = 1e-4

class LightCurveDataset(Dataset):
    def __init__(self, sequences, labels, masks):
        self.sequences = sequences
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.float32),
        )

def load_and_group_batches(DATASET_PATHS, max_per_class_global=None, max_per_class_override=None, batch_size=128):
    print(f"\U000023F3 [INFO] Iniciando agrupación de curvas por objeto...", flush=True)
    start_time = time.time()
    dataset = ds.dataset(DATASET_PATHS, format="parquet")
    # Determinar nombres físicos de columnas
    schema = dataset.schema
    id_col = find_column(schema.names, "id")
    mag_col = find_column(schema.names, "magnitud")
    clase_col = find_column(schema.names, "clase_variable_normalizada")
    cols = [id_col, mag_col, clase_col]
    cols = [c for c in cols if c is not None]

    scanner = dataset.scanner(columns=cols, batch_size=batch_size)
    grouped_data = {}
    class_counts = defaultdict(int)

    total_batches = scanner.count_rows() // batch_size + 1
    processed_batches = 0

    for batch in tqdm(scanner.to_batches(), desc="Agrupando curvas por objeto", unit="batch"):
        processed_batches += 1
        if processed_batches % 10 == 0:
            elapsed = time.time() - start_time
            #print(f"\U000023F3 [INFO] Procesados {processed_batches} batches, tiempo transcurrido: {elapsed:.1f}s", flush=True)
        df = batch.to_pandas()
        # Usar los nombres físicos detectados
        df_id = find_column(df.columns, "id")
        df_mag = find_column(df.columns, "magnitud")
        df_clase = find_column(df.columns, "clase_variable_normalizada")
        df["id"] = df[df_id].astype(str)
        for id_obj, group in df.groupby("id"):
            clase = group[df_clase].iloc[0]
            if not isinstance(clase, str) or clase.strip() == "":
                continue
            max_limit = max_per_class_override.get(clase, max_per_class_global)
            if max_limit is not None and class_counts[clase] >= max_limit:
                continue
            if id_obj not in grouped_data:
                grouped_data[id_obj] = group
                class_counts[clase] += 1
        del df, batch
        gc.collect()

    elapsed = time.time() - start_time
    # Mostrar el tiempo en minutos y segundos
    elapsed_minutes = elapsed // 60
    elapsed_seconds = elapsed % 60
    print(f"\U000023F3 [INFO] Agrupación finalizada en {elapsed_minutes:.0f} minutos y {elapsed_seconds:.1f} segundos", flush=True)
    print(f"\U0001F4C8 [INFO] Total de objetos agrupados: {len(grouped_data)}", flush=True)
    return grouped_data

def process_single_curve(group, seq_length, device):
    id_objeto, df = group
    # Usar mapeo robusto de columnas
    mag_col = find_column(df.columns, "magnitud")
    clase_col = find_column(df.columns, "clase_variable_normalizada")
    magnitudes = df[mag_col].to_numpy()

    if np.all(np.isnan(magnitudes)):
        discard_reasons["all_nan"] += 1
        return None

    if len(magnitudes) < MIN_POINTS:
        discard_reasons["short_curve"] += 1
        return None

    magnitudes = np.nan_to_num(magnitudes, nan=np.nanmedian(magnitudes))
    if np.std(magnitudes) < MIN_STD:
        discard_reasons["low_std"] += 1
        return None

    median = np.median(magnitudes)
    iqr = np.subtract(*np.percentile(magnitudes, [75, 25]))
    magnitudes_norm = (magnitudes - median) / iqr if iqr != 0 else magnitudes - median

    if np.isnan(magnitudes_norm).any() or np.isinf(magnitudes_norm).any():
        discard_reasons["nan_or_inf_after_norm"] += 1
        return None

    magnitudes_norm = np.clip(magnitudes_norm, -1000, 1000)

    attention_mask = np.zeros(seq_length)
    effective_length = min(len(magnitudes_norm), seq_length)
    padded_curve = np.zeros(seq_length)
    padded_curve[:effective_length] = magnitudes_norm[:effective_length]
    attention_mask[:effective_length] = 1

    # Normalizar la clase usando el diccionario común
    clase = normalize_label(df[clase_col].iloc[0])
    discard_reasons["ok"] += 1
    return padded_curve, clase, attention_mask

def main(
    seq_length=20000,
    batch_size=64,
    num_workers=4,
    limit_objects=None,
    device="cpu",
    max_per_class=None,
    max_per_class_override=None,
    parquet_dir="data/processed"
):
    print("\U0001F4C2 Cargando datos en lotes con PyArrow...", flush=True)
    global_start = time.time()

    if max_per_class_override is None:
        max_per_class_override = {}

    # --- Agrupación de datos ---
    t0 = time.time()
    grouped_data = load_and_group_batches(
        DATASET_PATHS,
        max_per_class_global=max_per_class,
        max_per_class_override=max_per_class_override,
        batch_size=128  # Reduce si hay problemas de memoria
    )
    t1 = time.time()
    print(f"\U000023F3 [INFO] Tiempo en agrupación de datos: {t1-t0:.1f} segundos", flush=True)

    grouped_list = list(grouped_data.items())
    random.shuffle(grouped_list)

    print(f"\U0001F680 Procesando {len(grouped_list)} curvas en paralelo usando {num_workers} CPUs...", flush=True)
    t2 = time.time()
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_single_curve,
            [(group, seq_length, device) for group in grouped_list]
        )
    t3 = time.time()
    print(f"\U000023F3 [INFO] Tiempo en procesamiento paralelo: {t3-t2:.1f} segundos", flush=True)

    results = [r for r in results if r is not None]
    filtered = [r for r in results if not (
        np.isnan(r[0]).any() or np.isinf(r[0]).any() or
        np.isnan(r[2]).any() or np.isinf(r[2]).any()
    )]

    print(f"\U0001F50B [INFO] Curvas válidas tras filtrado: {len(filtered)}", flush=True)

    sequences, labels, masks = zip(*filtered)
    unique_labels = sorted(set(labels))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_encoder[label] for label in labels]

    os.makedirs("data/train", exist_ok=True)
    print(f"\U0001F4BE [INFO] Guardando label_encoder.pkl...", flush=True)
    with open("data/train/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    counter = Counter(encoded_labels)
    print("\U0001F4CA Recuento por clase codificada:")
    for label, count in counter.items():
        name = [k for k, v in label_encoder.items() if v == label][0]
        print(f"{label:>2} ({name}): {count}")

    df_debug = pd.DataFrame({
        "id": list(grouped_data.keys())[:len(labels)],
        "clase_variable": labels,
        "clase_codificada": encoded_labels
    })
    df_debug.to_csv("data/train/debug_clases_codificadas.csv", index=False)

    sequences = np.array(sequences)
    encoded_labels = np.array(encoded_labels)
    masks = np.array(masks)

    print(f"\U0001F4DD [INFO] Realizando split train/val...", flush=True)
    train_idx, val_idx = train_test_split(
        np.arange(len(encoded_labels)),
        test_size=0.2,
        stratify=encoded_labels,
        random_state=42
    )

    train_dataset = LightCurveDataset(sequences[train_idx], encoded_labels[train_idx], masks[train_idx])
    val_dataset = LightCurveDataset(sequences[val_idx], encoded_labels[val_idx], masks[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Guardar datasets serializados para no perderlos al reiniciar el kernel y poder subirlos a SageMaker
    print(f"\U0001F4BE [INFO] Guardando datasets serializados...", flush=True)
    torch.save(train_loader.dataset, "data/train/train_dataset.pt")
    torch.save(val_loader.dataset, "data/train/val_dataset.pt")

    print("\n\U0001F4C9 Resumen de curvas descartadas:")
    for reason, count in discard_reasons.items():
        print(f"\U0001F538 {reason.replace('_', ' ').capitalize():30}: {count}")

    pd.DataFrame(discard_reasons.items(), columns=["motivo", "cantidad"]).to_csv("data/train/debug_descartes.csv", index=False)
    total_time = time.time() - global_start
    print(f"\U00002705 Datos preparados como secuencias normalizadas y máscaras.")
    print(f"\U000023F3 [INFO] Tiempo total de ejecución: {total_time/60:.2f} minutos ({total_time:.1f} segundos)", flush=True)
    return train_loader, val_loader
