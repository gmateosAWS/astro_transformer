COLUMN_MAPPING = {
    "id": ["id", "source_id", "ID", "id_objeto", "oid"],
    "mission_id": ["id_mision"],
    "mission": ["mission", "mision"],
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
    "mag_err": ["mag_err", "magerr"],
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
    "tiempo": ["tiempo", "mjd", "HJD", "hjd", "time"],
    "time": ["time", "HJD", "MJD", "hjd", "mjd", "tiempo"],
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

ALT_TO_STD = {}
for std, alts in COLUMN_MAPPING.items():
    for alt in alts:
        ALT_TO_STD[alt] = std

def map_column_name(col):
    return ALT_TO_STD.get(col, col)

def find_column(schema_names, logical_name):
    # Busca el nombre lógico directamente
    if logical_name in schema_names:
        return logical_name
    # Busca en los alias definidos en COLUMN_MAPPING
    for alt in COLUMN_MAPPING.get(logical_name, []):
        if alt in schema_names:
            return alt
    # Busca si el nombre lógico es un alias de otro estándar
    for std, alts in COLUMN_MAPPING.items():
        if logical_name in alts and std in schema_names:
            return std
    return None
