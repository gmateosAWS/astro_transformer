@echo off
setlocal enabledelayedexpansion

echo ========================================
echo 🧩 INICIO DE PROCESO DE COMPACTACION
echo ========================================
echo Directorio base: data\processed
echo Row group objetivo: 50000 filas
echo Compresion: ZSTD
echo ========================================
echo.

REM === FICHERO YA PROCESADO MANUALMENTE ===
REM duckdb -c "COPY (SELECT * FROM 'data/processed/dataset_eb_tess_labeled_fixed3-unified.parquet') TO 'data/processed/dataset_eb_tess_labeled_fixed3-compact.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
REM duckdb -c "SELECT COUNT(DISTINCT row_group_id) AS row_group_count FROM parquet_metadata('data/processed/dataset_eb_tess_labeled_fixed3-compact.parquet');"

REM === LISTA DE FICHEROS A PROCESAR ===

set FILTROS=^
dataset_eb_tess_labeled_fixed1-unified.parquet,^
dataset_eb_tess_labeled_fixed2-unified.parquet,^
dataset_vsx_tess_labeled_north-unified.parquet,^
dataset_vsx_tess_labeled_ampliado-unified.parquet

for %%F in (%FILTROS%) do (
    echo ----------------------------------------
    echo Procesando: %%F
    echo ----------------------------------------

    set "SRC=data/processed/%%F"
    set "DST=%%F"
    set "DST=!DST:-unified=-compact!"
    set "DST=data/processed/!DST!"

    echo Ejecutando compactacion...
    duckdb -c "COPY (SELECT * FROM '!SRC!') TO '!DST!' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 50000);"

    echo Verificando row groups generados:
    duckdb -c "SELECT COUNT(DISTINCT row_group_id) AS row_group_count FROM parquet_metadata('!DST!');"

    echo Procesado completo de: %%F
    echo.
)

echo ========================================
echo TODOS LOS FICHEROS PROCESADOS
echo ========================================
pause




rem COMPACTADO dataset_eb_tess_labeled_fixed3-unified.parquet: 18321 row groups, 18,142,465 filas => 118 row groups  (de 100k, con ZSP)
rem COMPACTADO dataset_eb_tess_labeled_fixed1-unified.parquet: 49998 row groups, 49,975,930 filas => 1000 row groups (de 50k, con SNAPPY)
rem COMPACTADO dataset_eb_tess_labeled_fixed2-unified.parquet: 52427 row groups, 52,403,954 filas => 1048 row groups (de 50k, con SNAPPY)
rem COMPACTADO dataset_vsx_tess_labeled_north-unified.parquet: 56781 row groups, 56,757,771 filas => 1135 row groups (de 50k, con SNAPPY)
rem COMPACTADO dataset_vsx_tess_labeled_ampliado-unified.parquet: 57476 row groups, 57,452,150 filas => 1149 row groups (de 50k, con SNAPPY)

rem DEMASIADO GRANDE dataset_eb_kepler_labeled_fixed-unified.parquet: 100,000 row groups, 100,000,000 filas => lo hemos compactado desde Python => 1434 row groups (de 50k, con SNAPPY)

