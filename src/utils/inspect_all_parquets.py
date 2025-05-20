# inspect_all_parquets.py (actualizado con exportación de resumen)

import pyarrow.dataset as ds
from pathlib import Path
import pandas as pd
import time

base_path = Path("data/processed")

parquets = [
    p for p in base_path.glob("dataset_*_labeled*_fixed.parquet")
    if "clean" not in p.name.lower()
] + [base_path / "dataset_gaia_dr3_vsx_tic_labeled.parquet"]
summary = []

print(f"📊 Inspección de {len(parquets)} datasets...\n")

def check_column(ds_obj, col):
    return col in ds_obj.schema.names

start = time.time()
for path in parquets:
    print(f"📂 Inspeccionando: {path.name}", flush=True)
    if not path.exists():
        summary.append({"dataset": path.name, "error": "❌ No encontrado"})
        continue
    try:
        dataset = ds.dataset(str(path), format="parquet")
        schema = dataset.schema
        row_count = dataset.count_rows()

        id_ok = False
        if "id_objeto" in schema.names:
            col = dataset.to_table(columns=["id_objeto"]).to_pandas()["id_objeto"].astype(str)
            id_ok = col.str.match(r"^[A-Z]+_\d+$").all()

        row = {
            "dataset": path.name,
            "rows": row_count,
            "id_objeto_ok": id_ok,
            "mision": check_column(dataset, "mision"),
            "ra": check_column(dataset, "ra"),
            "dec": check_column(dataset, "dec"),
            "clase_variable": check_column(dataset, "clase_variable"),
            "clase_variable_normalizada": check_column(dataset, "clase_variable_normalizada"),
            "origen_etiqueta": check_column(dataset, "origen_etiqueta")
        }
        summary.append(row)

    except Exception as e:
        summary.append({"dataset": path.name, "error": str(e)})

elapsed = time.time() - start

# Crear resumen
if summary:
    df_summary = pd.DataFrame(summary)
    cols = ["dataset", "rows", "id_objeto_ok", "mision", "ra", "dec",
            "clase_variable", "clase_variable_normalizada", "origen_etiqueta"]
    if "error" in df_summary.columns:
        cols.append("error")
    df_summary = df_summary[cols].fillna("—")

    # Mostrar y guardar
    print("\n⏱️ Tiempo total: {:.2f} segundos\n".format(elapsed))
    print(df_summary.to_markdown(index=False))

    out_path = base_path / "summary" / "summary_datasets_fixed.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(df_summary.to_markdown(index=False))
    print(f"\n📄 Markdown exportado: {out_path.relative_to(base_path.parent)}")
else:
    print("⚠️ No se generó resumen (posiblemente no se encontró ningún dataset)")
