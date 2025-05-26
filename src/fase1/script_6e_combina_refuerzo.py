# script_6e_combina_refuerzo_corregido.py

import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
import os
import sys
from src.fase2.script_1_transformer_preprocessing_optimizado import LightCurveDataset, process_single_curve

SEQ_LENGTH = 25000
DEVICE = "cpu"
PATH_REFUERZO = "data/processed/dataset_refuerzo_desde_errores.parquet"
PATH_TRAIN = "data/train/train_dataset.pt"
PATH_ENCODER = "data/train/label_encoder.pkl"
PATH_ERRORES = "outputs/errores_mal_clasificados_con_id.csv"

print("🗓️ Cargando datasets existentes...")
train_dataset = torch.load(PATH_TRAIN, weights_only=False)
label_encoder = pickle.load(open(PATH_ENCODER, "rb"))

print("🔍 Cargando dataset de refuerzo...")
df = pd.read_parquet(PATH_REFUERZO)
df["id_objeto"] = df["id_objeto"].astype(str)

print("📂 Cargando IDs de errores conocidos y entrenamiento...")
df_errores = pd.read_csv(PATH_ERRORES)
id_errores = set(df_errores["id_objeto"].astype(str))

# Extraemos los IDs directamente del dataset
id_train = set()
for i in range(len(train_dataset)):
    try:
        id_train.add(str(train_dataset.ids[i]))
    except AttributeError:
        pass  # Si no hay atributo de IDs, se ignora

id_candidatos = id_errores - id_train
print(f"📊 Total errores: {len(id_errores)}, en train: {len(id_errores & id_train)}, candidatos: {len(id_candidatos)}")

agrupadas = df[df["id_objeto"].isin(id_candidatos)].groupby("id_objeto")
print(f"🔹 {len(agrupadas)} curvas candidatas desde refuerzo")

curvas_nuevas = []
labels_nuevas = []
masks_nuevas = []
id_nuevos = []

for id_obj, group in tqdm(agrupadas):
    resultado = process_single_curve((id_obj, group), SEQ_LENGTH, DEVICE)
    if resultado is None:
        continue
    curva, clase, mask = resultado
    if clase not in label_encoder:
        continue
    if mask.mean() < 0.02:
        continue  # ❌ curva casi vacía, no aporta información útil
    curvas_nuevas.append(torch.tensor(curva, dtype=torch.float32))
    labels_nuevas.append(label_encoder[clase])
    masks_nuevas.append(torch.tensor(mask, dtype=torch.float32))
    id_nuevos.append(id_obj)

print(f"✅ {len(curvas_nuevas)} nuevas curvas se agregarán al entrenamiento")

if len(curvas_nuevas) == 0:
    print("⚠️ No se han añadido nuevas curvas. Se mantienen los datasets originales.")
    sys.exit()

# Convertir a tensores
x_train_old = torch.stack([t for t, _, _ in train_dataset])
y_train_old = torch.tensor([l for _, l, _ in train_dataset], dtype=torch.long)
mask_train_old = torch.stack([m for _, _, m in train_dataset])

x_train_new = torch.stack(curvas_nuevas)
y_train_new = torch.tensor(labels_nuevas, dtype=torch.long)
mask_train_new = torch.stack(masks_nuevas)

x_train = torch.cat([x_train_old, x_train_new], dim=0)
y_train = torch.cat([y_train_old, y_train_new], dim=0)
mask_train = torch.cat([mask_train_old, mask_train_new], dim=0)

train_dataset_updated = LightCurveDataset(x_train, y_train, mask_train)

print(f"📏 Dataset final: {x_train.shape[0]} curvas")
print("💾 Guardando datasets actualizados...")
os.makedirs("data/train", exist_ok=True)
torch.save(train_dataset_updated, PATH_TRAIN)
print("🎉 Listo. Puedes lanzar el fine-tuning con el dataset reforzado.")
