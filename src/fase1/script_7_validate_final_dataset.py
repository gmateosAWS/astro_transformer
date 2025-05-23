# script_7_validate_final_dataset.py (optimizado con lectura parcial y eficiente)

import pyarrow.dataset as ds
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Configuración
DATASET_PATH = Path("data/processed/all_missions_labeled.parquet")

print(f"📂 Validando: {DATASET_PATH.name}\n")

# Dataset streaming por lotes
dataset = ds.dataset(str(DATASET_PATH), format="parquet")
schema = dataset.schema

# Esquema
print("🧩 Esquema detectado:")
for field in schema:
    print(f"- {field.name}: {field.type}")

# Conteo de filas total
total_rows = dataset.count_rows()
print(f"\n🔢 Filas totales: {total_rows:,}\n")

# Leer primeras filas para muestra
print("🔍 Mostrando 5 filas de ejemplo:\n")
batch_head = next(dataset.to_batches(batch_size=5))
df_head = pa.Table.from_batches([batch_head]).to_pandas()
print(df_head)

# Recuento de nulos en campos clave
subset_cols = ["id_objeto", "tiempo", "magnitud", "clase_variable", "clase_variable_normalizada", "mision"]
print("\n🧼 Recuento de nulos:")
nulos = {col: 0 for col in subset_cols}
total_checked = 0

scanner = dataset.scanner(columns=subset_cols)
for record_batch in scanner.to_batches():
    df_batch = pa.Table.from_batches([record_batch]).to_pandas()
    total_checked += len(df_batch)
    for col in subset_cols:
        nulos[col] += df_batch[col].isnull().sum()

for col in subset_cols:
    print(f"- {col}: {nulos[col]} nulos")

# Recuento de clases
print("\n📊 Recuento por clase normalizada:")
class_counts = {}
scanner = dataset.scanner(columns=["clase_variable_normalizada"])
for batch in scanner.to_batches():
    df = pa.Table.from_batches([batch]).to_pandas()
    for c in df["clase_variable_normalizada"].dropna():
        class_counts[c] = class_counts.get(c, 0) + 1
class_counts_series = pd.Series(class_counts).sort_values(ascending=False)
print(class_counts_series.to_string())

# Gráfico de clases
plt.figure(figsize=(12, 5))
sns.barplot(x=class_counts_series.index, y=class_counts_series.values, palette="tab10")
plt.title("Distribución por clase_variable_normalizada")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Número de curvas")
plt.xlabel("Clase")
plt.tight_layout()
plt.show()

# Recuento por misión
print("\n🛰️ Recuento por mision:")
mision_counts = {}
scanner = dataset.scanner(columns=["mision"])
for batch in scanner.to_batches():
    df = pa.Table.from_batches([batch]).to_pandas()
    for m in df["mision"].dropna():
        mision_counts[m] = mision_counts.get(m, 0) + 1
mision_counts_series = pd.Series(mision_counts).sort_values(ascending=False)
print(mision_counts_series.to_string())

# Gráfico de misiones
plt.figure(figsize=(8, 4))
sns.barplot(x=mision_counts_series.index, y=mision_counts_series.values, palette="Set2")
plt.title("Distribución por misión")
plt.ylabel("Número de curvas")
plt.xlabel("Misión")
plt.tight_layout()
plt.show()

# Recuento por dataset fuente
if "source_dataset" in schema.names:
    print("\n📁 Recuento por dataset fuente:")
    source_counts = {}
    scanner = dataset.scanner(columns=["source_dataset"])
    for batch in scanner.to_batches():
        df = pa.Table.from_batches([batch]).to_pandas()
        for s in df["source_dataset"].dropna():
            source_counts[s] = source_counts.get(s, 0) + 1
    source_counts_series = pd.Series(source_counts).sort_values(ascending=False)
    print(source_counts_series.to_string())

    # Gráfico de dataset fuente
    plt.figure(figsize=(10, 4))
    sns.barplot(x=source_counts_series.index, y=source_counts_series.values, palette="Set3")
    plt.title("Distribución por dataset fuente")
    plt.ylabel("Número de curvas")
    plt.xlabel("Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
