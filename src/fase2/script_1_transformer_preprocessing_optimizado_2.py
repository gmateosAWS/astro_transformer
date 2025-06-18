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
import time
import sys
import time
import sys
from scipy.stats import skew, kurtosis

# Importar función de normalización de clases
from src.utils.normalization_dict import normalize_label
from src.utils.column_mapping import COLUMN_MAPPING, map_column_name, find_column

# Define directories as constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "../../outputs")

# Selección automática de paths según entorno
def detect_aws_env():
    # Detecta SageMaker/Jupyter en AWS por variables de entorno y paths típicos
    if "SM_HOSTS" in os.environ or "SAGEMAKER_JOB_NAME" in os.environ:
        return True
    if "HOME" in os.environ and "SageMaker" in os.environ["HOME"]:
        return True
    if "JPY_PARENT_PID" in os.environ and "SageMaker" in os.getcwd():
        return True
    return False

if detect_aws_env() or os.environ.get("RUN_ENV", "").lower() == "aws":
    from src.utils.dataset_paths import DATASET_PATHS_AWS as DATASET_PATHS
else:
    from src.utils.dataset_paths import DATASET_PATHS

discard_reasons = {
    "all_nan": 0,
    "all_invalid": 0,
    "low_std": 0,
    "short_curve": 0,
    "nan_or_inf_after_norm": 0,
    "unknown_class": 0,
    "removed_class": 0,
    "ok": 0
}
MIN_POINTS = 30
MIN_STD = 1e-4

class LightCurveDataset(Dataset):
    def __init__(self, sequences, labels, masks, features, ids):
        self.sequences = sequences
        self.labels = labels
        self.masks = masks
        self.features = features
        self.ids = ids  # ← nuevo

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.features[idx], dtype=torch.float32),
            self.ids[idx]  # ← nuevo
        )

def load_and_group_batches(DATASET_PATHS, max_per_class_global=None, max_per_class_override=None, batch_size=128, cache_path=None, ids_refuerzo=None, ids_a_excluir=None):
    # Set default cache_path using DATA_DIR
    if cache_path is None:
        cache_path = os.path.join(DATA_DIR, "train/grouped_data.pkl")
    # Si existe el cache, cargarlo y devolverlo
    if os.path.exists(cache_path):
        print(f"\U0001F4BE [INFO] Cargando agrupación de curvas desde cache: {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            grouped_data = pickle.load(f)
        print(f"\U00002705 [INFO] Agrupación cargada desde cache. Total objetos: {len(grouped_data)}", flush=True)
        return grouped_data

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
        # Limpiar valores tipo '>23.629' en la columna de magnitud
        if df_mag is not None:
            df[df_mag] = df[df_mag].astype(str).str.replace('>', '', regex=False).str.strip()
            df[df_mag] = pd.to_numeric(df[df_mag], errors='coerce')
        df["id"] = df[df_id].astype(str)
        for id_obj, group in df.groupby("id"):
            # Excluir IDs si están en la lista de exclusión
            if ids_a_excluir and id_obj in ids_a_excluir:
                discard_reasons["removed_class"] += 1
                continue
            
            clase = group[df_clase].iloc[0]
            # Crear y normalizar clase_norm
            clase_norm = normalize_label(clase)

            # Validar que la clase sea un string válido y no sea "unknown"
            if not isinstance(clase, str) or clase.strip() == "" or clase_norm.lower() == "unknown":
                discard_reasons["unknown_class"] += 1
                continue

            # Excluir clases con max_per_class_override=0 (clases que se quieren eliminar)
            if max_per_class_override is not None and max_per_class_override.get(clase_norm, max_per_class_global) == 0:
                discard_reasons["removed_class"] += 1
                continue

            # Controlar si ya se ha alcanzado el límite por clase
            max_limit = max_per_class_override.get(clase_norm, max_per_class_global)
            if max_limit is not None and class_counts[clase_norm] >= max_limit:
                # Incluir IDs de refuerzo incluso si se supera el límite
                if ids_refuerzo and id_obj in ids_refuerzo:
                    grouped_data[id_obj] = group
                    class_counts[clase_norm] += 1
                continue

            # Control de duplicados 
            if id_obj not in grouped_data:
                grouped_data[id_obj] = group
                class_counts[clase_norm] += 1
            # Forzar inclusión de IDs de refuerzo, pero evitar duplicados
            if ids_refuerzo and id_obj in ids_refuerzo and id_obj not in grouped_data:
                grouped_data[id_obj] = group
                class_counts[clase_norm] += 1
        del df, batch
        gc.collect()

    elapsed = time.time() - start_time
    # Mostrar el tiempo en minutos y segundos
    elapsed_minutes = elapsed // 60
    elapsed_seconds = elapsed % 60
    print(f"\U000023F3 [INFO] Agrupación finalizada en {elapsed_minutes:.0f} minutos y {elapsed_seconds:.1f} segundos", flush=True)
    print(f"\U0001F4C8 [INFO] Total de objetos agrupados: {len(grouped_data)}", flush=True)
    # Guardar agrupación en cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(grouped_data, f)
    print(f"\U0001F4BE [INFO] Agrupación guardada en cache: {cache_path}", flush=True)
    return grouped_data
  
  
# --- NUEVA FUNCIÓN ---
def compute_aux_features(magnitudes):
    if len(magnitudes) == 0:
        return [0.0] * 7

    magnitudes = np.asarray(magnitudes)

    # Cálculo de estadísticas básicas
    std = float(np.std(magnitudes)) if len(magnitudes) > 1 else 0.0
    iqr = float(np.percentile(magnitudes, 75) - np.percentile(magnitudes, 25)) if len(magnitudes) > 1 else 0.0
    amplitude = float(np.max(magnitudes) - np.min(magnitudes)) if len(magnitudes) > 0 else 0.0
    median = float(np.median(magnitudes)) if len(magnitudes) > 0 else 0.0
    mad = float(np.median(np.abs(magnitudes - median))) if len(magnitudes) > 1 else 0.0

    # Skewness y kurtosis con clipping
    skewness = float(skew(magnitudes)) if len(magnitudes) > 2 else 0.0
    kurt = float(kurtosis(magnitudes)) if len(magnitudes) > 2 else 0.0

    # Reemplazar valores no finitos
    features = [std, iqr, amplitude, median, mad, skewness, kurt]
    features = [f if np.isfinite(f) else 0.0 for f in features]

    # Aplicar límites razonables a skewness y kurtosis (clipping conservador)
    features[5] = np.clip(features[5], -10, 10)   # skewness
    features[6] = np.clip(features[6], -20, 20)   # kurtosis

    # También puedes aplicar clip a amplitude si esperas picos extremos
    features[2] = np.clip(features[2], 0, 100)    # amplitude máx razonable

    return features


def process_single_curve(group, seq_length, device):
    id_objeto, df = group
    # Usar mapeo robusto de columnas
    mag_col = find_column(df.columns, "magnitud")
    clase_col = find_column(df.columns, "clase_variable_normalizada")
    # Limpiar valores tipo '>23.629' en la columna de magnitud
    magnitudes_str = df[mag_col].astype(str).str.replace('>', '', regex=False).str.strip()
    magnitudes = pd.to_numeric(magnitudes_str, errors='coerce').to_numpy()

    # Normalizar la clase usando el diccionario común
    clase = normalize_label(df[clase_col].iloc[0])
    if clase.lower() == "unknown":
        discard_reasons["unknown_class"] += 1
        return None, "unknown_class"

    # Extraer y limpiar magnitudes
    magnitudes_str = df[mag_col].astype(str).str.replace('>', '', regex=False).str.strip()
    magnitudes = pd.to_numeric(magnitudes_str, errors='coerce').to_numpy()
    # Validar si todos los valores son NaN o Inf
    if not np.isfinite(magnitudes).any():
        return None, "all_invalid"
    # Calcular mediana solo con los valores finitos
    med = np.median(magnitudes[np.isfinite(magnitudes)])
    # Sustituir NaN, +Inf, -Inf por la mediana válida
    magnitudes = np.nan_to_num(magnitudes, nan=med, posinf=med, neginf=med)
    
    if clase.lower() == "unknown":
        discard_reasons["unknown_class"] += 1
        return None, "unknown_class"
        
    if np.all(np.isnan(magnitudes)):
        return None, "all_nan"

    if len(magnitudes) < MIN_POINTS:
        return None, "short_curve"

    if np.std(magnitudes) < MIN_STD:
        return None, "low_std"

    # Compute auxiliary features
    features = compute_aux_features(magnitudes)

    # Validar valores NaN o Inf en las características calculadas
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize magnitudes
    median = np.median(magnitudes)
    iqr = np.subtract(*np.percentile(magnitudes, [75, 25]))
    magnitudes_norm = (magnitudes - median) / iqr if iqr != 0 else magnitudes - median

    if np.isnan(magnitudes_norm).any() or np.isinf(magnitudes_norm).any():
        return None, "nan_or_inf_after_norm"

    magnitudes_norm = np.clip(magnitudes_norm, -1000, 1000)
    attention_mask = np.zeros(seq_length)
    effective_length = min(len(magnitudes_norm), seq_length)
    padded_curve = np.zeros(seq_length)
    padded_curve[:effective_length] = magnitudes_norm[:effective_length]
    attention_mask[:effective_length] = 1

    discard_reasons["ok"] += 1
    return (padded_curve, clase, attention_mask, features), "ok"

def main(
    seq_length=20000,
    parquet_batch_size=64,
    dataloader_batch_size=128,
    num_workers=4,
    limit_objects=None,
    device="cpu",
    max_per_class=None,
    max_per_class_override=None,
    errores_csv_path=None,
    filtrar_curvas_malas=None
):
    print("\U0001F4C2 Cargando datos en lotes con PyArrow...", flush=True)
    global_start = time.time()

    if max_per_class_override is None:
        max_per_class_override = {}

    ids_refuerzo = set()
    if errores_csv_path:
        errores_df = pd.read_csv(errores_csv_path)
        ids_refuerzo = set(errores_df["indice"].astype(str))
        print(f"\U0001F4C2 [INFO] IDs de refuerzo cargados: {len(ids_refuerzo)}")

    # --- NUEVO: Filtrado de curvas malas ---
    ids_a_excluir = set()
    ids_a_excluir_por_clase = {}
    if filtrar_curvas_malas:
        df_malas = pd.read_csv(filtrar_curvas_malas)
        ids_a_excluir = set(df_malas["id_objeto"].astype(str))
        # Contar por clase original
        ids_a_excluir_por_clase = df_malas.groupby("clase_original")["id_objeto"].nunique().to_dict()
        print(f"\U0001F4C2 [INFO] IDs a excluir por filtrado: {len(ids_a_excluir)}")
        print(f"\U0001F4C2 [INFO] Exclusiones por clase: {ids_a_excluir_por_clase}")

        # Ajustar límites de clase
        if max_per_class_override:
            for clase, n_excluir in ids_a_excluir_por_clase.items():
                if clase in max_per_class_override and max_per_class_override[clase] is not None:
                    nuevo_limite = max(0, max_per_class_override[clase] - n_excluir)
                    print(f"   - Ajustando max_per_class_override[{clase}] de {max_per_class_override[clase]} a {nuevo_limite}")
                    max_per_class_override[clase] = nuevo_limite
        elif max_per_class is not None:
            # Si no hay override, solo hay un límite global
            total_excluir = sum(ids_a_excluir_por_clase.values())
            if max_per_class is not None:
                nuevo_limite = max(0, max_per_class - total_excluir)
                print(f"   - Ajustando max_per_class de {max_per_class} a {nuevo_limite}")
                max_per_class = nuevo_limite

    # --- Agrupación de datos ---
    t0 = time.time()
    grouped_data = load_and_group_batches(
        DATASET_PATHS,
        max_per_class_global=max_per_class,
        max_per_class_override=max_per_class_override,
        batch_size=parquet_batch_size,
        cache_path=os.path.join(DATA_DIR, "train/grouped_data.pkl"),
        ids_refuerzo=ids_refuerzo,
        ids_a_excluir=ids_a_excluir if ids_a_excluir else None
    )
    t1 = time.time()
    print(f"\U000023F3 [INFO] Tiempo en agrupación de datos: {t1-t0:.1f} segundos", flush=True)

    # Convertir grouped_data a listas para procesar
    grouped_list = list(grouped_data.items())
    random.shuffle(grouped_list)

    # Guardar IDs de objetos para referencia posterior
    id_objetos = [id_obj for id_obj, _ in grouped_list]

    print(f"\U0001F680 Procesando {len(grouped_list)} curvas en paralelo usando {num_workers} CPUs...", flush=True)
    t2 = time.time()
    # Procesar curvas en paralelo
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_single_curve,
            [((id_obj, group), seq_length, device) for id_obj, group in grouped_list]
        )
    t3 = time.time()
    print(f"\U000023F3 [INFO] Tiempo en procesamiento paralelo: {t3-t2:.1f} segundos", flush=True)

    # Nuevo acumulador
    discard_reasons = defaultdict(int)
    #processed = []

    # Filtrar resultados válidos
    results = [r for r in results if r is not None]
    filtered = []
    for r, reason in results:
        if r is None:
            discard_reasons[reason] += 1
            continue
        discard_reasons["ok"] += 1
        try:
            # Crear una nueva tupla con valores corregidos
            corrected_r = (
                np.nan_to_num(np.asarray(r[0]), nan=0.0, posinf=0.0, neginf=0.0),  # sequences
                r[1],  # label
                np.nan_to_num(np.asarray(r[2]), nan=0.0, posinf=0.0, neginf=0.0),  # mask
                np.asarray(r[3])  # features
            )
        except Exception as e:
            print(f"❌ Error al convertir r en batch: {e}")
            discard_reasons["all_invalid"] += 1
            continue
        # Validación final
        if not (np.isnan(corrected_r[0]).any() or np.isinf(corrected_r[0]).any() or
                np.isnan(corrected_r[2]).any() or np.isinf(corrected_r[2]).any()):
            filtered.append(corrected_r)

    print(f"\U0001F50B [INFO] Curvas válidas tras filtrado: {len(filtered)}", flush=True)

    # Excluir unknown del label_encoder y de las etiquetas
    sequences, labels, masks, features = zip(*filtered)
    filtered_indices = [i for i, label in enumerate(labels) if str(label).lower() != "unknown"]
    sequences = [sequences[i] for i in filtered_indices]
    labels = [labels[i] for i in filtered_indices]
    masks = [masks[i] for i in filtered_indices]
    features = [features[i] for i in filtered_indices]

    # Crear label_encoder antes de codificar
    unique_labels = sorted(set(labels))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_encoder[label] for label in labels], dtype=np.int32)

    # Convertir secuencias y máscaras a float16 (ahorro de memoria)
    # sequences = np.array(sequences, dtype=np.float16)
    # masks = np.array(masks, dtype=np.float16)
    sequences = np.array(sequences)
    masks = np.array(masks)
    # Convertir y normalizar las características auxiliares en float32
    # Normalización características auxiliares (Robust Scaler por columna)
    features = np.array(features, dtype=np.float32)  # Asegura tipo y estructura
    features_median = np.median(features, axis=0)
    features_iqr = np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0) + 1e-8
    features = (features - features_median) / features_iqr
    # Validar valores NaN o Inf en las características normalizadas
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    #features = features.astype(np.float16)

    # Llamar a la prueba rápida después de generar las características
    quick_test(features)

    # Guardar features_median y features_iqr en un archivo .pkl
    features_stats_path = os.path.join(DATA_DIR, "train/features_stats.pkl")
    os.makedirs(os.path.dirname(features_stats_path), exist_ok=True)
    with open(features_stats_path, "wb") as f:
        pickle.dump({"median": features_median, "iqr": features_iqr}, f)
    print(f"📁 Features median y IQR guardados en: {features_stats_path}")

    # Mostrar el uso de memoria de los arrays
    print(f"[INFO] Uso de memoria de sequences: {sequences.nbytes/1024/1024:.2f} MB")
    print(f"[INFO] Uso de memoria de masks: {masks.nbytes/1024/1024:.2f} MB")
    print(f"[INFO] Uso de memoria de features: {features.nbytes/1024/1024:.2f} MB")
    print(f"[INFO] Uso de memoria de labels: {encoded_labels.nbytes/1024/1024:.2f} MB")

    # ADVERTENCIA: El uso de memoria será muy alto con 85k curvas x 25k puntos, incluso en float16 (~40GB solo para sequences)
    # Si tienes problemas de memoria, considera guardar los arrays en disco y usar un Dataset que lea bajo demanda.
    print(f"[INFO] N curvas: {len(sequences)}, seq_length: {sequences[0].shape[0] if len(sequences) > 0 else 0}")
    print(f"[INFO] Estimación memoria sequences (float16): {len(sequences)*seq_length*2/1024/1024/1024:.2f} GB")
    print(f"[INFO] Estimación memoria sequences (float32): {len(sequences)*seq_length*4/1024/1024/1024:.2f} GB")
    print(f"[INFO] Si tienes problemas de memoria, considera usar almacenamiento en disco y Dataset bajo demanda.")

    os.makedirs("data/train", exist_ok=True)
    print(f"\U0001F4BE [INFO] Guardando label_encoder.pkl...", flush=True)
    with open(os.path.join(DATA_DIR, "train/label_encoder.pkl"), "wb") as output_file:
        pickle.dump(label_encoder, output_file)

    counter = Counter(encoded_labels)
    print("\U0001F4CA Recuento por clase codificada:")
    for label, count in counter.items():
        name = [k for k, v in label_encoder.items() if v == label][0]
        print(f"{label:>2} ({name}): {count}")

    # Filtrar también los IDs con los mismos índices válidos
    id_objetos_filtrados = [id_objetos[i] for i in filtered_indices]
    df_debug = pd.DataFrame({
        "id": id_objetos_filtrados,
        "clase_variable": labels,
        "clase_codificada": encoded_labels
    })
    df_debug.to_csv(os.path.join(DATA_DIR, "train/debug_clases_codificadas.csv"), index=False)

    print(f"\U0001F4DD [INFO] Realizando split train/val/test...", flush=True)

    # Crear índices de refuerzo
    refuerzo_indices = [i for i, (id_obj, _) in enumerate(grouped_list) if id_obj in ids_refuerzo]

    # División inicial: train + val/test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(encoded_labels)),
        test_size=0.1,  # 10% para test final
        stratify=encoded_labels,
        random_state=42
    )

    # División secundaria: train y val dentro del 90% restante
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2222,  # ≈ 20% del 90% restante = 20% val total    # División inicial: train + val/test
        stratify=encoded_labels[train_val_idx],
        random_state=42
    )

    # Asegurar que los IDs de refuerzo están en train
    train_idx = set(train_idx)  # Usar un set para evitar duplicados
    train_idx.update(refuerzo_indices)  # Añadir índices de refuerzo
    train_idx = np.array(list(train_idx))  # Convertir de nuevo a arraytro del 90% restante

    # Crear datasets
    sequences = np.asarray(sequences, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    train_dataset = LightCurveDataset(
        [sequences[i] for i in train_idx],
        [encoded_labels[i] for i in train_idx],
        [masks[i] for i in train_idx],
        [features[i] for i in train_idx],  # Incluir características auxiliares
        [id_objetos[i] for i in train_idx]
    )
    val_dataset = LightCurveDataset(
        [sequences[i] for i in val_idx],
        [encoded_labels[i] for i in val_idx],
        [masks[i] for i in val_idx],
        [features[i] for i in val_idx],  # Incluir características auxiliares
        [id_objetos[i] for i in val_idx]
    )
    test_dataset = LightCurveDataset(
        [sequences[i] for i in test_idx],
        [encoded_labels[i] for i in test_idx],
        [masks[i] for i in test_idx],
        [features[i] for i in test_idx],  # Incluir características auxiliares
        [id_objetos[i] for i in test_idx]
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"\U0001F4CA [INFO] IDs de refuerzo incluidos en train: {len(refuerzo_indices)}")

    train_loader = DataLoader(train_dataset, batch_size=dataloader_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=dataloader_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=dataloader_batch_size, shuffle=False)

    # Guardar datasets serializados para no perderlos al reiniciar el kernel y poder subirlos a SageMaker
    print(f"\U0001F4BE [INFO] Guardando datasets serializados en formato .pt...", flush=True)
    # Save datasets in smaller chunks using torch.save
    def save_large_object(obj, filepath):
        torch.save(obj, filepath)
    save_large_object(train_loader.dataset, os.path.join(DATA_DIR, "train/train_dataset.pt"))
    save_large_object(val_loader.dataset, os.path.join(DATA_DIR, "train/val_dataset.pt"))
    save_large_object(test_loader.dataset, os.path.join(DATA_DIR, "train/test_dataset.pt"))

    print("\n\U0001F4C9 Resumen de curvas descartadas:")
    for reason, count in discard_reasons.items():
        print(f"\U0001F538 {reason.replace('_', ' ').capitalize():30}: {count}")

    pd.DataFrame(discard_reasons.items(), columns=["motivo", "cantidad"]).to_csv(
        os.path.join(DATA_DIR, "train/debug_descartes.csv"), index=False
    )
    total_time = time.time() - global_start
    print(f"\U00002705 Datos preparados como secuencias normalizadas y máscaras.")
    print(f"\U000023F3 [INFO] Tiempo total de ejecución: {total_time/60:.2f} minutos ({total_time:.1f} segundos)", flush=True)
    return train_loader, val_loader

def quick_test(features, num_samples=10):
    """
    Realiza una prueba rápida en un subconjunto de las características para verificar valores NaN o Inf.
    """
    print("\n🔍 Realizando prueba rápida en características auxiliares...")
    subset = features[:num_samples]
    for i, feature in enumerate(subset):
        if np.isnan(feature).any() or np.isinf(feature).any():
            print(f"⚠️ Valores NaN o Inf detectados en el sample {i}: {feature}")
        else:
            print(f"✅ Sample {i} sin problemas: {feature}")
    print("✅ Prueba rápida completada.")
