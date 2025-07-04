import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import pandas as pd
from collections import defaultdict
from src.utils.normalization_dict import normalize_label
from tqdm import tqdm
import gc
import itertools

INPUT_DIR = Path("data/processed")
FILES = [
    "dataset_k2varcat_labeled_fixed.parquet",
    "dataset_vsx_tess_labeled_fixed.parquet",
    "dataset_gaia_complemented_normalized.parquet",
    "dataset_eb_kepler_labeled_fixed.parquet",
    "dataset_eb_tess_labeled_fixed.parquet",
    "dataset_vsx_tess_labeled_south.parquet",
    "dataset_vsx_tess_labeled_north.parquet",
    "dataset_vsx_tess_labeled_ampliado.parquet",
    "dataset_ztf_labeled.parquet"
]

ALL_MISSIONS_PATH = INPUT_DIR / "all_missions_labeled.parquet"
if ALL_MISSIONS_PATH.exists():
    print(f"🧩 Usando esquema base de: {ALL_MISSIONS_PATH.name}")
    base_schema = pq.read_schema(ALL_MISSIONS_PATH)
else:
    base_schema = None

COLUMN_MAPPING = {
    "id": ["id", "source_id", "ID", "id_objeto", "oid"],
    "mission_id": ["id_mision"],
    "mission": ["mision"],
    "start_date": ["fecha_inicio"],
    "end_date": ["fecha_fin"],
    "label_source": ["origen_etiqueta"],
    "clase_variable": ["clase_variable", "variable_class"],
    "clase_variable_normalizada": ["clase_variable_normalizada", "normalized_class"],
    "period": ["period", "PERIOD", "period_days", "period_day"],
    "amplitude": ["amplitude", "AMPLITUDE", "ampli", "ampli_max"],
    "ra": ["ra", "RA", "ra_deg"],
    "dec": ["dec", "DEC", "dec_deg"],
    "flux": ["flux"],
    "flux_err": ["flux_err", "error"],
    "mag": ["mag", "magnitud"],
    "magerr": ["magerr"],
    "magzp": ["magzp"],
    "magzprms": ["magzprms"],
    "limitmag": ["limitmag"],
    "airmass": ["airmass"],
    "cadenceno": ["cadenceno"],
    "quality": ["quality", "sap_quality"],
    "fcor": ["fcor"],
    "cbv01": ["cbv01"],
    "cbv02": ["cbv02"],
    "cbv03": ["cbv03"],
    "cbv04": ["cbv04"],
    "cbv05": ["cbv05"],
    "cbv06": ["cbv06"],
    "bkg": ["bkg", "sap_bkg"],
    "bkg_err": ["sap_bkg_err"],
    "centroid_col": ["centroid_col"],
    "centroid_row": ["centroid_row"],
    "sap_flux": ["sap_flux"],
    "sap_flux_err": ["sap_flux_err"],
    "pdcsap_flux": ["pdcsap_flux"],
    "pdcsap_flux_err": ["pdcsap_flux_err"],
    "psf_centr1": ["psf_centr1"],
    "psf_centr1_err": ["psf_centr1_err"],
    "psf_centr2": ["psf_centr2"],
    "psf_centr2_err": ["psf_centr2_err"],
    "mom_centr1": ["mom_centr1"],
    "mom_centr1_err": ["mom_centr1_err"],
    "mom_centr2": ["mom_centr2"],
    "mom_centr2_err": ["mom_centr2_err"],
    "pos_corr1": ["pos_corr1"],
    "pos_corr2": ["pos_corr2"],
    "band": ["band", "filtercode"],
    "tiempo": ["tiempo", "mjd", "hjd"],
    "timecorr": ["timecorr"],
    "field": ["field"],
    "filefracday": ["filefracday"],
    "catflags": ["catflags"],
    "ccdid": ["ccdid"],
    "chi": ["chi"],
    "clrcoeff": ["clrcoeff"],
    "clrcounc": ["clrcounc"],
    "expid": ["expid"],
    "exptime": ["exptime"],
    "programid": ["programid"],
    "qid": ["qid"],
    "sharp": ["sharp"],
    "source_dataset": ["source_dataset"],
    # ...añade más mapeos si aparecen nuevas columnas...
}

# Invertir el mapeo para búsqueda rápida: nombre alternativo -> nombre estándar
ALT_TO_STD = {}
for std, alts in COLUMN_MAPPING.items():
    for alt in alts:
        ALT_TO_STD[alt] = std

def map_column_name(col):
    return ALT_TO_STD.get(col, col)

def infer_global_schema():
    # Unifica el esquema de all_missions_labeled.parquet (si existe) y de todos los .parquet a procesar
    pyarrow_type_map = {
        "int64": pa.int64(),
        "int32": pa.float64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
        "object": pa.string(),
        "string": pa.string(),
        "datetime64[ns]": pa.timestamp("ns")
    }
    column_types = defaultdict(set)
    # Añadir tipos del esquema base si existe
    if base_schema:
        for field in base_schema:
            std_name = map_column_name(field.name)
            column_types[std_name].add(str(field.type))
    # Unificar tipos de todos los parquet a procesar
    for fname in FILES:
        path = INPUT_DIR / fname
        try:
            schema = pq.read_schema(path)
            for field in schema:
                std_name = map_column_name(field.name)
                column_types[std_name].add(str(field.type))
        except Exception:
            # Si no puede leer el schema, lo ignora
            continue
    # Construir campos finales
    schema_fields = []
    float64_columns = set()
    for col, dtypes in column_types.items():
        if "object" in dtypes or "string" in dtypes or "string[pyarrow]" in dtypes:
            schema_fields.append(pa.field(col, pa.string()))
        elif "float64" in dtypes or "int64" in dtypes or "int32" in dtypes or "double" in dtypes:
            schema_fields.append(pa.field(col, pa.float64()))
            float64_columns.add(col)
        elif "bool" in dtypes or "bool[pyarrow]" in dtypes:
            schema_fields.append(pa.field(col, pa.bool_()))
        elif "datetime64[ns]" in dtypes or "timestamp[ns]" in dtypes:
            schema_fields.append(pa.field(col, pa.timestamp("ns")))
        else:
            schema_fields.append(pa.field(col, pa.string()))
    schema_final = pa.schema(schema_fields)
    all_columns = schema_final.names
    print(f"🧩 Esquema global unificado:")
    print(f"  Número de columnas: {len(all_columns)}")
    print(f"  Columnas: {all_columns}\n")
    return schema_final, all_columns, float64_columns

def merge_batches_pyarrow(unified_path, tmp_dir, schema_final):
    """
    Une todos los batches temporales en un solo archivo, recorriendo todos los temporales de forma rápida,
    pero puede llenar la memoria si hay demasiados. Úsalo solo si el número de batches es manejable o
    si vas a hacer la unión en varias pasadas.
    """
    import shutil

    batch_files = sorted(tmp_dir.glob("batch_*.parquet"))
    print(f"Total batches temporales a unir: {len(batch_files)}", flush=True)
    with pq.ParquetWriter(unified_path, schema_final) as writer:
        for tmp_file in tqdm(batch_files, desc="Unificando batches", leave=True):
            table = pq.read_table(tmp_file)
            writer.write_table(table)
            del table
            gc.collect()
    # ...el resto igual...

def unify_schema_for_file(fname, schema_final, all_columns, float64_columns, start_batch=0, end_batch=None):
    path = INPUT_DIR / fname
    print(f"\n📂 Procesando: {fname} (batches {start_batch} a {end_batch if end_batch is not None else 'end'})", flush=True)
    tmp_dir = path.parent / f"{path.stem}_unify_tmp"
    tmp_dir.mkdir(exist_ok=True)

    batch_size = 1000
    total_rows = pq.read_metadata(path).num_rows
    total_batches = (total_rows + batch_size - 1) // batch_size
    if end_batch is None or end_batch == "end":
        end_batch = total_batches

    batches_created = 0
    existing_tmp = set(f.name for f in tmp_dir.glob("batch_*.parquet"))

    dataset = ds.dataset(str(path), format="parquet")
    for i, batch in enumerate(tqdm(dataset.to_batches(batch_size=batch_size), desc=f"Batches {fname}", leave=True)):
        if i < start_batch or i >= end_batch:
            del batch
            continue
        tmp_batch_name = f"batch_{i:04d}.parquet"
        tmp_batch_path = tmp_dir / tmp_batch_name
        if tmp_batch_name in existing_tmp or tmp_batch_path.exists():
            del batch
            continue
        try:
            df_batch = pa.Table.from_batches([batch]).to_pandas()
            rename_dict = {col: map_column_name(col) for col in df_batch.columns if map_column_name(col) != col}
            df_batch = df_batch.rename(columns=rename_dict)
            cols = pd.Series(df_batch.columns)
            duplicated = cols[cols.duplicated()].unique()
            for col in duplicated:
                dup_cols = [j for j, c in enumerate(df_batch.columns) if c == col]
                combined = df_batch.iloc[:, dup_cols].bfill(axis=1).iloc[:, 0]
                df_batch = df_batch.drop(df_batch.columns[dup_cols], axis=1)
                df_batch[col] = combined
            df_batch = df_batch.loc[:, ~df_batch.columns.duplicated()]
            if "clase_variable" in df_batch.columns and "clase_variable_normalizada" not in df_batch.columns:
                df_batch["clase_variable_normalizada"] = df_batch["clase_variable"].apply(normalize_label)
            elif "clase_variable_normalizada" in df_batch.columns and "clase_variable" not in df_batch.columns:
                df_batch["clase_variable"] = df_batch["clase_variable_normalizada"]
            elif "clase_variable" not in df_batch.columns and "clase_variable_normalizada" not in df_batch.columns:
                print(f"Batch {i} descartado: columnas={df_batch.columns.tolist()}, shape={df_batch.shape}", flush=True)
                del df_batch, batch
                continue
            for col in all_columns:
                if col not in df_batch.columns:
                    df_batch[col] = None
            for col in float64_columns:
                if col in df_batch.columns:
                    try:
                        df_batch[col] = df_batch[col].astype("float64")
                    except Exception:
                        df_batch[col] = pd.NA
            df_batch = df_batch[all_columns]
            df_batch.to_parquet(tmp_batch_path, index=False)
            print(f"✅ Batch {i} creado: {tmp_batch_path}", flush=True)
            batches_created += 1
            del df_batch, batch
        except Exception as e:
            print(f"❌ Error procesando batch {i}: {e}", flush=True)
            if 'df_batch' in locals():
                del df_batch
            del batch
            continue

    print(f"✅ Batches nuevos creados en esta ejecución: {batches_created}", flush=True)

def merge_partial_parquets(fname, schema_final):
    path = INPUT_DIR / fname
    tmp_dir = path.parent / f"{path.stem}_unify_tmp"
    unified_path = path.with_name(path.stem + "-unified.parquet")
    batch_files = sorted(tmp_dir.glob("batch_*.parquet"))
    if not batch_files:
        print("⚠️ No hay batches temporales para unir.")
        return
    print(f"🔗 Uniendo {len(batch_files)} batches temporales en {unified_path.name} usando PyArrow puro...", flush=True)
    with pq.ParquetWriter(unified_path, schema_final) as writer:
        for tmp_file in tqdm(batch_files, desc="Unificando batches", leave=True):
            table = pq.read_table(tmp_file)
            writer.write_table(table)
            del table
            gc.collect()
    print(f"✅ {fname} normalizado y guardado como {unified_path.name}.", flush=True)

def run_unify_schema(fname, start=0, end=None, merge_final=False):
    schema_final, all_columns, float64_columns = infer_global_schema()
    unify_schema_for_file(fname, schema_final, all_columns, float64_columns, start_batch=start, end_batch=end)
    if merge_final:
        merge_partial_parquets(fname, schema_final)

def main(selected_file=None):
    print("🔎 Unificando esquemas de todos los ficheros...", flush=True)
    schema_final, all_columns, float64_columns = infer_global_schema()
    if selected_file:
        files_to_process = [selected_file]
    else:
        files_to_process = FILES
    for fname in files_to_process:
        unify_schema_for_file(fname, schema_final, all_columns, float64_columns)
        path = INPUT_DIR / fname
        tmp_dir = path.parent / f"{path.stem}_unify_tmp"
        unified_path = path.with_name(path.stem + "-unified.parquet")
        if tmp_dir.exists() and any(tmp_dir.glob("batch_*.parquet")):
            print(f"🔗 Uniendo batches temporales en {unified_path.name} usando PyArrow puro...", flush=True)
            try:
                merge_batches_pyarrow(unified_path, tmp_dir, schema_final)
                print(f"✅ {fname} normalizado y guardado como {unified_path.name}.", flush=True)
                if unified_path.exists() and unified_path.stat().st_size > 0:
                    print(f"🗑️ Puedes borrar los temporales manualmente si lo deseas: {tmp_dir}", flush=True)
                else:
                    print(f"⚠️ El archivo final {unified_path} no se generó correctamente. No se eliminan temporales.", flush=True)
                gc.collect()
            except Exception as e:
                print(f"❌ Error al unir batches con PyArrow: {e}", flush=True)
                raise
        else:
            print(f"⚠️ No se generaron batches temporales para {fname}, no se crea el parquet final.", flush=True)

def export_partial_parquet(fname, batch_start, batch_end, output_parquet):
    """
    Une los batches temporales en el rango [batch_start, batch_end) en un parquet parcial.
    Fuerza el tipo string en columnas string aunque el batch tenga solo None.
    Si encuentra un batch corrupto, lo omite y avisa.
    """
    path = INPUT_DIR / fname
    tmp_dir = path.parent / f"{path.stem}_unify_tmp"
    schema_final, all_columns, float64_columns = infer_global_schema()
    batch_files = []
    for i in range(batch_start, batch_end):
        batch_file = tmp_dir / f"batch_{i:04d}.parquet"
        if batch_file.exists():
            batch_files.append(batch_file)
    if not batch_files:
        print(f"⚠️ No hay batches temporales en el rango {batch_start}-{batch_end} para unir.")
        return
    print(f"🔗 Uniendo {len(batch_files)} batches temporales ({batch_start}-{batch_end-1}) en {output_parquet} ...", flush=True)
    with pq.ParquetWriter(output_parquet, schema_final) as writer:
        for tmp_file in tqdm(batch_files, desc=f"Unificando batches parciales", leave=True):
            try:
                table = pq.read_table(tmp_file)
            except Exception as e:
                print(f"⚠️ Batch corrupto o no-parquet: {tmp_file} ({e}), se omite.")
                continue
            # --- Forzar tipos string en columnas string del schema ---
            for field in schema_final:
                if pa.types.is_string(field.type):
                    if table.schema.field_by_name(field.name).type != pa.string():
                        if field.name in table.column_names:
                            arr = table[field.name]
                            arr = arr.cast(pa.string(), safe=False)
                            table = table.set_column(table.schema.get_field_index(field.name), field.name, arr)
                        else:
                            arr = pa.array([None] * table.num_rows, type=pa.string())
                            table = table.append_column(field.name, arr)
                elif pa.types.is_null(field.type):
                    if field.name in table.column_names:
                        arr = table[field.name].cast(pa.string(), safe=False)
                        table = table.set_column(table.schema.get_field_index(field.name), field.name, arr)
            table = table.select(all_columns)
            writer.write_table(table)
            del table
            gc.collect()
    print(f"✅ Parcial guardado como {output_parquet}", flush=True)

def merge_final_parquets(parquet_paths, output_parquet, schema_final, batch_size=1000):
    """
    Une varios parquet parciales en el parquet final.
    batch_size: tamaño de batch para la escritura, debe ser igual al usado en los parciales si quieres mantener la estructura.
    """
    print(f"🔗 Uniendo {len(parquet_paths)} parciales en {output_parquet} ...", flush=True)
    with pq.ParquetWriter(output_parquet, schema_final) as writer:
        for part in parquet_paths:
            try:
                # Lee y escribe por lotes para evitar OOM en ficheros grandes
                for batch in pq.ParquetFile(part).iter_batches(batch_size=batch_size):
                    table = pa.Table.from_batches([batch])
                    writer.write_table(table)
                    del table, batch
                    gc.collect()
            except Exception as e:
                print(f"❌ Error procesando {part}: {e}", flush=True)
                continue
    print(f"✅ Parquet final guardado como {output_parquet}", flush=True)

def generate_missing_batches_from_rowgroup(fname, batch_start, batch_end, batch_size=1000):
    """
    Procesa los batches del rango [batch_start, batch_end) leyendo UNA SOLA VEZ cada row group (o todo el parquet si solo hay uno)
    y generando solo los batches temporales necesarios.
    Este método es eficiente y escalable si el parquet tiene muchos row groups pequeños.
    """
    path = INPUT_DIR / fname
    tmp_dir = path.parent / f"{path.stem}_unify_tmp"
    tmp_dir.mkdir(exist_ok=True)
    schema_final, all_columns, float64_columns = infer_global_schema()
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    total_batches = (total_rows + batch_size - 1) // batch_size

    if batch_end > total_batches:
        batch_end = total_batches

    existing_tmp = set(f.name for f in tmp_dir.glob("batch_*.parquet"))

    print(f"Procesando batches {batch_start}-{batch_end-1} ... (lectura por row group)", flush=True)
    # Procesa por row group, pero solo guarda los batches del rango deseado
    for rg in range(pf.num_row_groups):
        rg_start = sum(pf.metadata.row_group(i).num_rows for i in range(rg))
        rg_end = rg_start + pf.metadata.row_group(rg).num_rows
        # Si el row group no solapa con el rango de interés, lo saltamos
        if rg_end <= batch_start * batch_size or rg_start >= batch_end * batch_size:
            continue
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        # Calcula los batches que caen en este row group y están en el rango pedido
        first_batch = max(batch_start, rg_start // batch_size)
        last_batch = min(batch_end, (rg_end + batch_size - 1) // batch_size)
        for batch_idx in range(first_batch, last_batch):
            s = batch_idx * batch_size - rg_start
            e = s + batch_size
            if s >= len(df) or batch_idx < batch_start or batch_idx >= batch_end:
                continue
            df_batch = df.iloc[s:e].copy()
            tmp_batch_name = f"batch_{batch_idx:04d}.parquet"
            tmp_batch_path = tmp_dir / tmp_batch_name
            if tmp_batch_name in existing_tmp or tmp_batch_path.exists():
                del df_batch
                continue
            # ...normalización, renombrado, etc, igual que en el resto del código...
            rename_dict = {col: map_column_name(col) for col in df_batch.columns if map_column_name(col) != col}
            df_batch = df_batch.rename(columns=rename_dict)
            cols = pd.Series(df_batch.columns)
            duplicated = cols[cols.duplicated()].unique()
            for col in duplicated:
                dup_cols = [j for j, c in enumerate(df_batch.columns) if c == col]
                combined = df_batch.iloc[:, dup_cols].bfill(axis=1).iloc[:, 0]
                df_batch = df_batch.drop(df_batch.columns[dup_cols], axis=1)
                df_batch[col] = combined
            df_batch = df_batch.loc[:, ~df_batch.columns.duplicated()]
            if "clase_variable" in df_batch.columns and "clase_variable_normalizada" not in df_batch.columns:
                df_batch["clase_variable_normalizada"] = df_batch["clase_variable"].apply(normalize_label)
            elif "clase_variable_normalizada" in df_batch.columns and "clase_variable" not in df_batch.columns:
                df_batch["clase_variable"] = df_batch["clase_variable_normalizada"]
            elif "clase_variable" not in df_batch.columns and "clase_variable_normalizada" not in df_batch.columns:
                print(f"Batch {batch_idx} descartado: columnas={df_batch.columns.tolist()}, shape={df_batch.shape}", flush=True)
                del df_batch
                continue
            for col in all_columns:
                if col not in df_batch.columns:
                    df_batch[col] = None
            for col in float64_columns:
                if col in df_batch.columns:
                    try:
                        df_batch[col] = df_batch[col].astype("float64")
                    except Exception:
                        df_batch[col] = pd.NA
            df_batch = df_batch[all_columns]
            df_batch.to_parquet(tmp_batch_path, index=False)
            #print(f"✅ Batch {batch_idx} creado: {tmp_batch_path}", flush=True)
            del df_batch
            gc.collect()
        del df, table
        gc.collect()
    print(f"✅ Batches faltantes generados en el rango {batch_start}-{batch_end-1}", flush=True)

# --- USO MANUAL PARA TU CASO ---
# 1. Exporta los batches temporales 0-49999 a un parcial
# export_partial_parquet("dataset_eb_tess_labeled_fixed.parquet", 0, 50000, "data/processed/dataset_eb_tess_labeled_fixed.part1.parquet")
# 2. Exporta los batches temporales 50000-102426 a otro parcial
# export_partial_parquet("dataset_eb_tess_labeled_fixed.parquet", 50000, 102427, "data/processed/dataset_eb_tess_labeled_fixed.part2.parquet")
# 3. Procesa y exporta los batches nuevos 102427-120748 a un tercer parcial
# export_partial_parquet_new_batches("dataset_eb_tess_labeled_fixed.parquet", 102427, 120749, "data/processed/dataset_eb_tess_labeled_fixed.part3.parquet")
# 4. Une los tres parciales en el unified final
# from src.fase1.script_6a_unify_schema_tess import merge_final_parquets, infer_global_schema
# schema_final = infer_global_schema()[0]
# merge_final_parquets([
#     "data/processed/dataset_eb_tess_labeled_fixed.part1.parquet",
#     "data/processed/dataset_eb_tess_labeled_fixed.part2.parquet",
#     "data/processed/dataset_eb_tess_labeled_fixed.part3.parquet"
# ], "data/processed/dataset_eb_tess_labeled_fixed-unified.parquet", schema_final)