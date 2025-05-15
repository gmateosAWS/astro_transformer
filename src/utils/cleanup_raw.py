import os
from pathlib import Path

def cleanup_raw_data(raw_dir: Path = Path("data/raw"), confirm: bool = True):
    """
    Elimina todos los archivos .csv en data/raw/*/*

    Parámetros:
        raw_dir (Path): Ruta base al directorio raw.
        confirm (bool): Solicita confirmación antes de borrar si es True.
    """
    files_deleted = 0
    if not raw_dir.exists():
        print(f"[⚠] El directorio {raw_dir} no existe.")
        return

    all_csvs = list(raw_dir.glob("*/*.csv"))
    if not all_csvs:
        print("[✓] No hay archivos .csv que eliminar.")
        return

    if confirm:
        respuesta = input(f"¿Estás seguro de que deseas eliminar {len(all_csvs)} archivos .csv? (s/N): ")
        if respuesta.strip().lower() != "s":
            print("[⏹] Cancelado por el usuario.")
            return

    for f in all_csvs:
        try:
            f.unlink()
            files_deleted += 1
        except Exception as e:
            print(f"❌ No se pudo eliminar {f}: {e}")

    print(f"[🧹] Archivos .csv eliminados: {files_deleted}")
