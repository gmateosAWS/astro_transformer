import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
import shutil
import pyarrow.parquet as pq
import pyarrow as pa


def read_and_merge_curves(data_dir: Path, output_path: Path, batch_size: int = 500):
    """
    Lee archivos CSV de curvas de luz en lotes y construye un fichero Parquet final de forma eficiente.
    Si ya existen ficheros Parquet temporales previos, los reutiliza directamente.
    """
    temp_dir = Path("/home/ec2-user/temp_batches")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Si ya hay archivos parquet previos, los usamos directamente
    parquet_files = sorted(temp_dir.glob("batch_*.parquet"))
    if not parquet_files:
        pattern_flat = os.path.join(data_dir, "*.csv")
        pattern_nested = os.path.join(data_dir, "*/*.csv")
        files = glob(pattern_flat) + glob(pattern_nested)

        print(f"[📂] Procesando {len(files)} curvas en {data_dir} → guardando por lotes en {temp_dir}...")

        batch_rows = []
        batch_index = 0

        for i, f in enumerate(tqdm(files, desc="📦 Procesando por lotes")):
            parts = os.path.basename(f).split("_")
            if len(parts) < 5:
                continue
            mission = parts[0]
            target_id = parts[1]
            meta = parts[2]
            try:
                df = pd.read_csv(f)
                df = df.rename(columns={"time": "tiempo", "flux": "magnitud", "flux_err": "error"})
                df["id_objeto"] = target_id
                df["id_mision"] = f"{mission}_{target_id}_{meta}"
                df["mision"] = mission.capitalize()
                df["fecha_inicio"] = df["tiempo"].min()
                df["fecha_fin"] = df["tiempo"].max()
                batch_rows.append(df)

                if len(batch_rows) >= batch_size:
                    df_batch = pd.concat(batch_rows, ignore_index=True)
                    batch_file = temp_dir / f"batch_{batch_index:03d}.parquet"
                    df_batch.to_parquet(batch_file, index=False)
                    batch_index += 1
                    batch_rows.clear()
            except Exception as e:
                print(f"❌ Error leyendo {f}: {e}")

        # Guardar último lote si queda algo
        if batch_rows:
            df_batch = pd.concat(batch_rows, ignore_index=True)
            batch_file = temp_dir / f"batch_{batch_index:03d}.parquet"
            df_batch.to_parquet(batch_file, index=False)

        print(f"[✅] Total de lotes guardados: {batch_index + 1}")
    else:
        print(f"[📁] Usando {len(parquet_files)} lotes ya existentes en {temp_dir}")

    # Concatenar todos los .parquet al fichero final usando pyarrow
    print(f"[⏳] Uniendo todos los lotes en {output_path}...")
    writer = None
    for i, parquet_file in enumerate(tqdm(sorted(temp_dir.glob("batch_*.parquet")), desc="📚 Uniendo lotes")):
        table = pq.read_table(parquet_file)

        # Forzar tipo consistente en 'quality'
        if "quality" in table.schema.names:
            field_idx = table.schema.get_field_index("quality")
            current_type = table.schema.field(field_idx).type
            # Convertir a float64 si no lo es
            if current_type != pa.float64():
                quality_array = table.column("quality").to_pandas().astype("float64")
                table = table.set_column(field_idx, "quality", pa.array(quality_array, type=pa.float64()))
    
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)
    if writer:
        writer.close()

    print(f"[✅] Dataset parquet construido → {output_path}")

    # Limpiar carpeta temporal
    shutil.rmtree(temp_dir)
    print(f"🧹 Carpeta temporal eliminada: {temp_dir}")


