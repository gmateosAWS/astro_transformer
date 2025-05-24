import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset, DataLoader
import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import random

# Al inicio del script (zona global o dentro de main())
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

# Procesar cada curva de luz
# def process_single_curve(group, seq_length, device):
#     id_objeto, df = group
#     magnitudes = df["magnitud"].to_numpy()

#     if np.all(np.isnan(magnitudes)):
#         return None

#     magnitudes = np.nan_to_num(magnitudes, nan=np.nanmedian(magnitudes))
#     if np.std(magnitudes) == 0:
#         return None

#     median = np.median(magnitudes)
#     iqr = np.subtract(*np.percentile(magnitudes, [75, 25]))
#     if iqr == 0:
#         magnitudes_norm = (magnitudes - median)
#     else:
#         magnitudes_norm = (magnitudes - median) / iqr

#     if np.isnan(magnitudes_norm).any() or np.isinf(magnitudes_norm).any():
#         print(f"⚠️ Curva con NaN/Inf detectada y descartada — id_objeto: {id_objeto}")
#         return None

#     magnitudes_norm = np.clip(magnitudes_norm, -1000, 1000)

#     attention_mask = np.zeros(seq_length)
#     effective_length = min(len(magnitudes_norm), seq_length)
#     padded_curve = np.zeros(seq_length)
#     padded_curve[:effective_length] = magnitudes_norm[:effective_length]
#     attention_mask[:effective_length] = 1

#     clase = df["clase_variable_normalizada"].iloc[0]
#     return padded_curve, clase, attention_mask

# Reemplaza tu función por esta:
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
    if iqr == 0:
        magnitudes_norm = magnitudes - median
    else:
        magnitudes_norm = (magnitudes - median) / iqr

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


def main(seq_length=20000, batch_size=64, num_workers=4, limit_objects=None, device="cpu", max_per_class=100000):
    print("📂 Cargando datos en lotes con PyArrow...", flush=True)
    dataset = ds.dataset("data/processed/all_missions_labeled.parquet", format="parquet")
    scanner = dataset.scanner(columns=["id_objeto", "magnitud", "clase_variable_normalizada"])

    grouped_data = {}
    class_counts = defaultdict(int)
    total_rows = 0

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        for id_obj, group in df.groupby("id_objeto"):
            clase = group["clase_variable_normalizada"].iloc[0]
            if max_per_class is not None and class_counts[clase] >= max_per_class:
                continue
            if id_obj not in grouped_data:
                grouped_data[id_obj] = group
                class_counts[clase] += 1

        total_rows += len(df)

    if max_per_class is None:
        print("⚠️ No se ha aplicado balanceo por clase (max_per_class=None). Algunas clases pueden estar sobrerrepresentadas.")

    grouped_list = list(grouped_data.items())
    random.shuffle(grouped_list)

    print(f"🚀 Procesando {len(grouped_list)} curvas en paralelo usando {num_workers} CPUs...", flush=True)
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_single_curve,
            [(group, seq_length, device) for group in grouped_list]
        )

    results = [r for r in results if r is not None]
    filtered = []
    for curve, label, mask in results:
        if (
            np.isnan(curve).any() or np.isinf(curve).any() or
            np.isnan(mask).any() or np.isinf(mask).any()
        ):
            print("⚠️ Curva con NaN o Inf detectada al final del preprocesado. Descartada.")
            continue
        filtered.append((curve, label, mask))
    results = filtered

    sequences, labels, masks = zip(*results)

    unique_labels = sorted(set(labels))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_encoder[label] for label in labels]

    os.makedirs("data/train", exist_ok=True)
    with open("data/train/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    counter = Counter(encoded_labels)
    print("📊 Recuento por clase codificada:")
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

    print("\n📉 Resumen de curvas descartadas:")
    for reason, count in discard_reasons.items():
        print(f"🔸 {reason.replace('_', ' ').capitalize():30}: {count}")


    print("✅ Datos preparados como secuencias normalizadas y máscaras.")
    return train_loader, val_loader
