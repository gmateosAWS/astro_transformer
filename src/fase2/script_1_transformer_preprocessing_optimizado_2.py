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

def process_single_curve(group, seq_length, device):
    id_objeto, df = group
    magnitudes = df["magnitud"].to_numpy()

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

    clase = df["clase_variable_normalizada"].iloc[0]
    discard_reasons["ok"] += 1
    return padded_curve, clase, attention_mask

def load_and_group_batches(DATASET_PATHS, max_per_class):
    dataset = ds.dataset(DATASET_PATHS, format="parquet")
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

def load_and_group_batches(DATASET_PATHS, max_per_class_global=None, max_per_class_override=None):
    dataset = ds.dataset(DATASET_PATHS, format="parquet")
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
            max_limit = max_per_class_override.get(clase, max_per_class_global)
            if max_limit is not None and class_counts[clase] >= max_limit:
                continue
            if id_obj not in grouped_data:
                grouped_data[id_obj] = group
                class_counts[clase] += 1

    return grouped_data


def main(
    seq_length=20000,
    batch_size=64,
    num_workers=4,
    limit_objects=None,
    device="cpu",
    max_per_class=None,
    max_per_class_override=None
):
    print("\U0001F4C2 Cargando datos en lotes con PyArrow...", flush=True)

    DATASET_PATHS = [
        "data/processed/all_missions_labeled.parquet",
        "data/processed/dataset_gaia_complemented_normalized.parquet",
        "data/processed/dataset_vsx_tess_labeled_south.parquet",
        "data/processed/dataset_vsx_tess_labeled_north.parquet",
        "data/processed/dataset_vsx_tess_labeled_ampliado.parquet"
    ]

    # Si no se pasa override explícito, usar uno vacío
    if max_per_class_override is None:
        max_per_class_override = {}

    grouped_data = load_and_group_batches(
        DATASET_PATHS,
        max_per_class_global=max_per_class,
        max_per_class_override=max_per_class_override
    )

    grouped_list = list(grouped_data.items())
    random.shuffle(grouped_list)

    print(f"\U0001F680 Procesando {len(grouped_list)} curvas en paralelo usando {num_workers} CPUs...", flush=True)
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_single_curve,
            [(group, seq_length, device) for group in grouped_list]
        )

    results = [r for r in results if r is not None]
    filtered = [r for r in results if not (
        np.isnan(r[0]).any() or np.isinf(r[0]).any() or
        np.isnan(r[2]).any() or np.isinf(r[2]).any()
    )]

    sequences, labels, masks = zip(*filtered)
    unique_labels = sorted(set(labels))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_encoder[label] for label in labels]

    os.makedirs("data/train", exist_ok=True)
    with open("data/train/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    counter = Counter(encoded_labels)
    print("\U0001F4CA Recuento por clase codificada:")
    for label, count in counter.items():
        name = [k for k, v in label_encoder.items() if v == label][0]
        print(f"{label:>2} ({name}): {count}")

    df_debug = pd.DataFrame({
        "id_objeto": list(grouped_data.keys())[:len(labels)],
        "clase_variable": labels,
        "clase_codificada": encoded_labels
    })
    df_debug.to_csv("data/train/debug_clases_codificadas.csv", index=False)

    sequences = np.array(sequences)
    encoded_labels = np.array(encoded_labels)
    masks = np.array(masks)

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

    print("\n\U0001F4C9 Resumen de curvas descartadas:")
    for reason, count in discard_reasons.items():
        print(f"\U0001F538 {reason.replace('_', ' ').capitalize():30}: {count}")

    pd.DataFrame(discard_reasons.items(), columns=["motivo", "cantidad"]).to_csv("data/train/debug_descartes.csv", index=False)
    print("\U00002705 Datos preparados como secuencias normalizadas y máscaras.")
    return train_loader, val_loader
