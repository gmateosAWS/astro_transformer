import pandas as pd
import os
from glob import glob
from tqdm import tqdm  # ⬅ Añadir

def read_and_merge_curves(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*/*.csv")
    files = glob(pattern)
    all_rows = []

    print(f"[📂] Procesando {len(files)} archivos de curvas...")
    
    for f in tqdm(files, desc="📊 Leyendo curvas"):
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
            all_rows.append(df)
        except Exception as e:
            print(f"❌ Error leyendo {f}: {e}")

    return pd.concat(all_rows, ignore_index=True)
