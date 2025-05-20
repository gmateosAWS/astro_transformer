import pandas as pd
from pathlib import Path
from typing import Optional, List

class DatasetBuilder:
    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sources = []

    def add_source(self, name: str, df: pd.DataFrame, label_col: str, origin: str):
        """Añade una fuente etiquetada al conjunto consolidado."""
        df["origen_etiqueta"] = origin
        df["clase_variable"] = df[label_col]
        df = df.drop(columns=[label_col], errors="ignore")
        self.sources.append((name, df))

    def merge_sources(self) -> pd.DataFrame:
        """Unifica todas las fuentes en un único DataFrame."""
        all_dfs = [df for _, df in self.sources]
        return pd.concat(all_dfs, ignore_index=True)

    def save(self, df: pd.DataFrame, filename: str, format: str = "parquet"):
        """Guarda el dataset consolidado."""
        output_path = self.base_dir / f"{filename}.{format}"
        if format == "parquet":
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError("Formato no soportado: usa 'parquet' o 'csv'")
        print(f"[✅] Dataset guardado en: {output_path}")
