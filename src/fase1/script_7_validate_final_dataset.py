# script_7_validate_final_dataset.py (optimizado con lectura parcial y eficiente)

import pyarrow.dataset as ds
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
from tqdm import tqdm 
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Importar función de normalización de clases ---
from src.utils.normalization_dict import normalize_label
from src.utils.column_mapping import COLUMN_MAPPING, map_column_name, find_column
from src.utils.dataset_paths import DATASET_PATHS

print("🔎 Verificando archivos unified:")
for fpath in DATASET_PATHS:
    p = Path(fpath)
    try:
        if not p.exists():
            print(f"❌ NO EXISTE: {p.name}")
            continue
        pf = pq.ParquetFile(p)
        nrows = pf.metadata.num_rows
        nrow_groups = pf.num_row_groups
        print(f"✅ {p.name}: {nrows:,} filas, {nrow_groups} row groups")
    except Exception as e:
        print(f"⚠️ ERROR en {p.name}: {e}")
print("✔️ Verificación terminada.\n")

print("🔎 Comprobando columnas presentes en cada archivo:")
for fpath in DATASET_PATHS:
    p = Path(fpath)
    try:
        if not p.exists():
            print(f"❌ NO EXISTE: {p.name}")
            continue
        pf = pq.ParquetFile(p)
        schema_cols = [pf.schema_arrow.names[i] for i in range(len(pf.schema_arrow.names))]
        # Mostrar todas las columnas presentes
        print(f"Columnas en {p.name}: {schema_cols}")
        # Usar find_column para comprobar si 'tiempo' está mapeando correctamente
        mapped = find_column(schema_cols, "tiempo")
        if mapped:
            print(f"✅ {p.name}: 'tiempo' mapeado a columna física '{mapped}'")
        else:
            print(f"⚠️ {p.name}: 'tiempo' NO mapeado a ninguna columna física (alias probados: {COLUMN_MAPPING.get('tiempo', [])})")
        # Comprobación clásica para referencia (solo para columnas físicas directas)
        # Solo mostrar advertencia si tampoco hay mapeo lógico
        missing = [col for col in ["tiempo", "mag", "mission"] if col not in schema_cols]
        if missing and not mapped:
            print(f"⚠️ {p.name} NO contiene columnas físicas directas: {missing}")
        else:
            print(f"✅ {p.name} contiene todas las columnas requeridas (directas o mapeadas).")
    except Exception as e:
        print(f"⚠️ ERROR en {p.name}: {e}")
print("✔️ Comprobación de columnas terminada.\n")

# Dataset streaming por lotes
import pyarrow as pa

def safe_batches(dataset, columns):
    """
    Itera sobre los fragmentos y lotes, intentando convertir a pandas.
    Si falla por ArrowInvalid al convertir el fragmento completo, lo procesa batch a batch,
    y si falla en un batch, fuerza todas las columnas a object.
    """
    for fragment in dataset.get_fragments():
        # Obtener columnas presentes en el fragmento usando fragment.physical_schema
        fragment_cols = set(fragment.physical_schema.names)
        cols_present = [col for col in columns if col in fragment_cols]
        if not cols_present:
            continue
        try:
            table = fragment.to_table(columns=cols_present)
            for record_batch in table.to_batches():
                yield record_batch.to_pandas()
        except pa.ArrowInvalid:
            scanner = fragment.scanner(columns=cols_present)
            for record_batch in scanner.to_batches():
                try:
                    yield record_batch.to_pandas()
                except pa.ArrowInvalid:
                    yield record_batch.to_pandas(types={col: object for col in record_batch.schema.names})

dataset = ds.dataset(DATASET_PATHS, format="parquet")

print("📂 Validando los siguientes archivos:")
for path in DATASET_PATHS:
    print(f"- {path}")
print()

schema = dataset.schema

# Esquema
print("🧩 Esquema detectado:")
for field in schema:
    print(f"- {field.name}: {field.type}")

# Conteo de filas total
try:
    total_rows = dataset.count_rows()
    print(f"\n🔢 Filas totales: {total_rows:,}\n")
except Exception as e:
    print(f"⚠️ Error contando filas: {e}")
    total_rows = None

# Leer primeras filas para muestra
print("🔍 Mostrando 5 filas de ejemplo:\n")
try:
    batch_head = next(dataset.to_batches(batch_size=5))
    df_head = pa.Table.from_batches([batch_head]).to_pandas()
    print(df_head)
except Exception as e:
    print(f"⚠️ Error mostrando filas de ejemplo: {e}")

# --- Unificación de agregados en un solo bucle para máxima eficiencia ---

# Determinar columnas físicas presentes para cada lógica
subset_cols_logical = ["id", "tiempo", "mag", "clase_variable", "clase_variable_normalizada", "mission"]
subset_cols = [find_column(schema.names, logical) for logical in subset_cols_logical]
subset_cols = [col for col in subset_cols if col is not None]

# Detectar y avisar si falta alguna columna lógica importante
missing_logical = [logical for logical in subset_cols_logical if find_column(schema.names, logical) is None]
if missing_logical:
    print(f"\n⚠️ ATENCIÓN: Las siguientes columnas lógicas no están presentes en el esquema físico y no serán procesadas: {missing_logical}\n")

print(f"[TRACE] Columnas físicas seleccionadas para lectura: {subset_cols}")

print(f"\n🧼 Recuento de nulos y agregados principales (solo columnas: {subset_cols}):")
nulos = {col: 0 for col in subset_cols}
class_labels = []
mision_labels = []
class_counts = {}
mision_counts = {}
total_checked = 0

scanner = dataset.scanner(columns=subset_cols)
t0 = time.time()
batch_total = 0

# Calcular número total de batches para la barra de progreso (estimado)
try:
    batch_size = 65536  # Tamaño de batch recomendado para eficiencia
    n_batches = (total_rows // batch_size) + 1 if total_rows else None
except Exception:
    n_batches = None

if n_batches:
    batch_iter = tqdm(safe_batches(dataset, subset_cols), total=n_batches, desc="🔄 Procesando batches", unit="batch")
else:
    batch_iter = tqdm(safe_batches(dataset, subset_cols), desc="🔄 Procesando batches", unit="batch")

for df_batch in batch_iter:
    total_checked += len(df_batch)
    # Nulos (más eficiente: vectorizado)
    # Solo procesa columnas que realmente están en el batch
    cols_in_batch = [col for col in subset_cols if col in df_batch.columns]
    nulos_batch = df_batch[cols_in_batch].isnull().sum()
    for col in cols_in_batch:
        nulos[col] += nulos_batch.get(col, 0)
    # Clases agrupadas: usa 'clase_variable' si existe y no es nulo, si no 'clase_variable_normalizada'
    col_var = find_column(df_batch.columns, "clase_variable")
    col_norm = find_column(df_batch.columns, "clase_variable_normalizada")
    if col_var and col_norm:
        clases = df_batch[col_var].where(df_batch[col_var].notnull(), df_batch[col_norm])
    elif col_var:
        clases = df_batch[col_var]
    elif col_norm:
        clases = df_batch[col_norm]
    else:
        clases = None
    if clases is not None:
        batch_counts = clases.value_counts()
        for k, v in batch_counts.items():
            class_counts[k] = class_counts.get(k, 0) + v
    # Misión (ajustar NONE a ZTF)
    col_mision = find_column(df_batch.columns, "mission")
    if col_mision:
        mvals = df_batch[col_mision].astype(str)
        mvals = [v.strip().upper().replace("NONE", "ZTF") for v in mvals]
        # En vez de acumular todo en memoria, cuenta por batch
        for m in mvals:
            mision_counts[m] = mision_counts.get(m, 0) + 1
    del df_batch
    batch_total += 1

elapsed = time.time() - t0
print(f"\n⏱️ Tiempo total para todos los agregados: {elapsed:.1f} s ({elapsed/60:.1f} min)")

# Mostrar recuento de nulos
for logical in subset_cols_logical:
    col = find_column(schema.names, logical)
    alias_list = COLUMN_MAPPING.get(logical, [logical])
    if col and col in nulos:
        print(f"- {logical} ({col}, alias={alias_list}): {nulos[col]} nulos")
    else:
        print(f"- {logical}: columna no encontrada (alias probados: {alias_list})")

# Mostrar recuento de clases agrupadas
if class_counts:
    print("\n📊 Recuento por clase_variable (sin normalizar):")
    class_counts_series = pd.Series(class_counts).sort_values(ascending=False)
    print(class_counts_series.to_string())
    if not class_counts_series.empty:
        plt.figure(figsize=(12, 5))
        sns.barplot(x=class_counts_series.index, y=class_counts_series.values, palette="tab10")
        plt.title("Distribución por clase_variable_normalizada (sin normalizar)")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Número de curvas")
        plt.xlabel("Clase (sin normalizar)")
        plt.tight_layout()
        plt.show()
else:
    print("\n⚠️ 'clase_variable_normalizada' o 'clase_variable' no están en el esquema.")

# Normalizar las clases
class_counts_normalized = {}
for k, v in class_counts.items():
    normalized_class = normalize_label(k)
    class_counts_normalized[normalized_class] = class_counts_normalized.get(normalized_class, 0) + v
# Mostrar recuento por clase_variable_normalizada (normalizado)
if class_counts_normalized:
    print("\n📊 Recuento por clase_variable_normalizada (normalizado):")
    class_counts_series = pd.Series(class_counts_normalized).sort_values(ascending=False)
    print(class_counts_series.to_string())
    if not class_counts_series.empty:
        plt.figure(figsize=(12, 5))
        sns.barplot(x=class_counts_series.index, y=class_counts_series.values, palette="tab10")
        plt.title("Distribución por clase_variable_normalizada (normalizado)")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Número de curvas")
        plt.xlabel("Clase (normalizada)")
        plt.tight_layout()
        plt.show()

# Mostrar recuento por misión (ajustando NONE a ZTF)
if mision_counts:
    print("\n🛰️ Recuento por mision:")
    mision_counts_series = pd.Series(mision_counts).sort_values(ascending=False)
    print(mision_counts_series.to_string())
    if not mision_counts_series.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=mision_counts_series.index, y=mision_counts_series.values, palette="Set2")
        plt.title("Distribución por misión")
        plt.ylabel("Número de curvas")
        plt.xlabel("Misión")
        plt.tight_layout()
        plt.show()
else:
    print("\n⚠️ 'mission' no está en el esquema.")

# --- Paso final: resumen global de curvas únicas por clase normalizada ---

print("\n📊 Recuento final de curvas únicas por clase normalizada (todos los datasets):\n")

# Acumulador eficiente: por clase, un set de ids únicos
class_to_ids = defaultdict(set)
# Acumulador eficiente: por clase normalizada, un set de ids únicos
class_to_ids_norm = defaultdict(set)

for path in tqdm(DATASET_PATHS, desc="🔍 Procesando datasets"):
    try:
        table = pq.read_table(path, columns=["id", "clase_variable_normalizada"])
        df = table.to_pandas()
        for row in df.itertuples(index=False):
            # Sin normalizar
            class_to_ids[row.clase_variable_normalizada].add(row.id)
            # Normalizado
            clase_norm = normalize_label(row.clase_variable_normalizada)
            class_to_ids_norm[clase_norm].add(row.id)
    except Exception as e:
        print(f"[ERROR] {path}: {e}")

# Construir resumen sin normalizar
resumen = {clase: len(ids) for clase, ids in class_to_ids.items()}
resumen = pd.Series(resumen).sort_values(ascending=False)

# Construir resumen normalizado
resumen_norm = {clase: len(ids) for clase, ids in class_to_ids_norm.items()}
resumen_norm = pd.Series(resumen_norm).sort_values(ascending=False)

print("📊 Recuento final de curvas únicas por clase (sin normalizar):\n")
print(resumen)
print("\n📊 Recuento final de curvas únicas por clase normalizada:\n")
print(resumen_norm)