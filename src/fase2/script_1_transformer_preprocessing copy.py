# script_1_transformer_preprocessing.py

import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
import sys
import os
from argparse import Namespace

# Añadir AstroConformer al path
astroconformer_path = os.path.abspath('./Astroconformer')
if astroconformer_path not in sys.path:
    sys.path.append(astroconformer_path)

from Astroconformer.Astroconformer.Model.models import Astroconformer as AstroConformer

# Configuración de Paths
DATASET_PATH = Path("data/processed/all_missions_labeled.parquet")

# Clase dataset PyTorch
class LightCurveDataset(Dataset):
    def __init__(self, embeddings, labels, attention_masks):
        self.embeddings = embeddings
        self.labels = labels
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (torch.tensor(self.embeddings[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.attention_masks[idx], dtype=torch.float32))

# Procesado individual de curva con AstroConformer
def process_single_curve(group, seq_length, device):
    id_objeto, df = group
    magnitudes = df["magnitud"].values
    magnitudes_norm = (magnitudes - np.mean(magnitudes)) / np.std(magnitudes)
    attention_mask = np.zeros(seq_length)
    effective_length = min(len(magnitudes_norm), seq_length)
    padded_curve = np.zeros(seq_length)
    padded_curve[:effective_length] = magnitudes_norm[:effective_length]
    attention_mask[:effective_length] = 1

    args = Namespace(
        input_dim=1,                    # no está en yaml, pero es obligatorio en nuestro script
        in_channels=1,
        encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False,
        num_layers=5,
        stride=20,
        encoder_dim=128,
        num_heads=8,
        kernel_size=3,
        dropout=0.1,                    # asumimos que dropout == dropout_p
        dropout_p=0.1,                  # si el modelo lo usa explícitamente
        hidden_dim=128,                # alineado con encoder_dim (si se requiere)
        output_dim=10,
        norm="postnorm",
        device=device
    )

    model = AstroConformer(args)
    model.eval()
    with torch.no_grad():
        embedding = model(torch.tensor(padded_curve, dtype=torch.float32).unsqueeze(0))

    clase = df["clase_variable_normalizada"].iloc[0]
    return embedding.squeeze(0).numpy(), clase, attention_mask

# Función principal parametrizable
def main(seq_length=20000, batch_size=128, num_workers=None, limit_objects=None, device="cpu"):
    if num_workers is None:
        num_workers = cpu_count()

    print("📂 Cargando datos en lotes con PyArrow...", flush=True)
    dataset = ds.dataset(str(DATASET_PATH), format="parquet")
    scanner = dataset.scanner(columns=["id_objeto", "magnitud", "clase_variable_normalizada"])

    grouped_data = {}
    for batch in scanner.to_batches():
        df_batch = batch.to_pandas()
        for id_objeto, group in df_batch.groupby('id_objeto'):
            if id_objeto not in grouped_data:
                grouped_data[id_objeto] = group
            else:
                grouped_data[id_objeto] = pd.concat([grouped_data[id_objeto], group])

    if limit_objects:
        print(f"🔍 Limitando procesamiento a los primeros {limit_objects} objetos", flush=True)
        grouped_data = dict(list(grouped_data.items())[:limit_objects])

    print(f"🚀 Procesando {len(grouped_data)} curvas en paralelo usando {num_workers} CPUs...", flush=True)
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_single_curve,
            [(group, seq_length, device) for group in grouped_data.items()]
        )

    embeddings, labels, attention_masks = zip(*results)

    label_encoder = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    labels_encoded = [label_encoder[label] for label in labels]

    X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(
        embeddings, labels_encoded, attention_masks, test_size=0.2, stratify=labels_encoded, random_state=42)

    train_dataset = LightCurveDataset(X_train, y_train, masks_train)
    val_dataset = LightCurveDataset(X_val, y_val, masks_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("✅ Datos preparados con AstroConformer y máscaras de atención.", flush=True)

    return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--limit_objects", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_loader, val_loader = main(
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_objects=args.limit_objects,
        device=args.device
    )
