# script_3d_merge_gaia_with_coords.py

import pandas as pd
from pathlib import Path

# Paths de entrada y salida
GAIA_FILE = Path("data/processed/dataset_gaia_dr3_vsx_tic_labeled.parquet")
COORDS_FILE = Path("data/processed/dataset_vsx_tic_labeled_clean_fixed.parquet")
OUTPUT_FILE = Path("data/processed/dataset_gaia_dr3_vsx_tic_labeled_with_coords.parquet")


def main():
    print("📥 Cargando datasets...")
    df_gaia = pd.read_parquet(GAIA_FILE)
    df_coords = pd.read_parquet(COORDS_FILE)

    print(f"🔗 Realizando merge por id_objeto...")
    df_merged = df_gaia.merge(
        df_coords[["id_objeto", "tic_id", "ra", "dec"]],
        on="id_objeto",
        how="left"
    )


    print(f"🔎 Filas originales: {len(df_gaia)} | Tras merge: {len(df_merged)}")
    print(f"⚠️ Filas sin tic_id tras merge: {df_merged['tic_id'].isna().sum()}")

    df_merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Dataset combinado guardado en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
