NORMALIZATION_DICT = {
    # Eclipsing Binary
    "ECLIPSING BINARY": "Eclipsing Binary",
    "EA": "Eclipsing Binary",
    "EB": "Eclipsing Binary",
    "EW": "Eclipsing Binary",
    "ECL": "Eclipsing Binary",
    "E": "Eclipsing Binary",
    "ECLIPSING": "Eclipsing Binary",
    "DETACHED BINARY": "Eclipsing Binary",
    "SEMI-DETACHED BINARY": "Eclipsing Binary",
    "CONTACT BINARY": "Eclipsing Binary",
    "EC": "Eclipsing Binary",
    "ED": "Eclipsing Binary",
    "ESD": "Eclipsing Binary",
    "EP": "Eclipsing Binary",
    "EA:": "Eclipsing Binary",
    "EW/RS": "Eclipsing Binary",

    # RR Lyrae
    "RR LYRAE": "RR Lyrae",
    "RRAB": "RR Lyrae",
    "RRC": "RR Lyrae",
    "RRD": "RR Lyrae",
    "ARRD": "RR Lyrae",
    "RR": "RR Lyrae",
    "ACEP": "RR Lyrae",
    "RRC|EC": "RR Lyrae",
    "RR_LYRAE": "RR Lyrae",
    "RRAB/BL": "RR Lyrae",
    "RRAB/BL:": "RR Lyrae",
    "BL": "RR Lyrae",

    # Delta Scuti
    "DELTA SCUTI": "Delta Scuti",
    "DSCT": "Delta Scuti",
    "HADS": "Delta Scuti",
    "DSCT:": "Delta Scuti",
    "DSCT|GDOR|SXPHE": "Delta Scuti",
    "DELTA_SCUTI": "Delta Scuti",
    "DSCTC": "Delta Scuti",
    "DSCTR": "Delta Scuti",

    # Rotational
    "ROTATIONAL": "Rotational",
    "ROT": "Rotational",
    "BY": "Rotational",
    "BYDRA": "Rotational",
    "BY_DRA": "Rotational",
    "ACV": "Rotational",
    "ROAP": "Rotational",
    "ROAM": "Rotational",
    "ELL": "Rotational",
    "RSCVN": "Rotational",
    "RS_CVN": "Rotational",
    "RS": "Rotational",
    "SXARI": "Rotational",
    "SPOTTED ROTATOR": "Rotational",
    "RVA": "Rotational",


    # Irregular
    "IRREGULAR": "Irregular",
    "LB": "Irregular",
    "L": "Irregular",
    "Y": "Irregular",
    "SRC": "Irregular",
    "SRA": "Irregular",
    "SRB": "Irregular",
    "SR": "Irregular",
    "S": "Irregular",
    "MIRA": "Irregular",
    "M": "Irregular",
    "AP": "Irregular",
    "LC": "Irregular",
    "SRD": "Irregular",
    "LPV": "Irregular",
    "SRS": "Irregular",


    # Variable
    "VARIABLE": "Variable",
    "CEP": "Variable",
    "CEPII": "Variable",
    "VAR": "Variable",
    "VAR:": "Variable",
    "CEPH": "Variable",
    "CEPHIED": "Variable",
    "CEPHEID": "Variable",
    "CEPHEID VARIABLE": "Variable",
    "BETA_CEP": "Variable",
    "BETA CEP": "Variable",
    "CWFU": "Variable",
    "UV": "Variable",
    "MISC": "Variable",
    "DCEP": "Variable",
    "DCEPS": "Variable",
    "DDCEP": "Variable",
    "CWA": "Variable",
    "CWB": "Variable",
    "QP": "Variable",
    "P": "Variable",
    "BCEP": "Variable",
    "SPB": "Variable",
    "CW-FU": "Variable",
    "CW-FO": "Variable",

    # Cataclysmic
    "CATACLYSMIC": "Cataclysmic",
    "UG": "Cataclysmic",
    "UGSU": "Cataclysmic",
    "UGSS": "Cataclysmic",
    "UGZ": "Cataclysmic",
    "UGWZ": "Cataclysmic",
    "UGSU+E": "Cataclysmic",
    "NL": "Cataclysmic",
    "NOVA": "Cataclysmic",
    "CV": "Cataclysmic",
    "CV:": "Cataclysmic",
    "CV+E": "Cataclysmic",
    "CV:E": "Cataclysmic",
    "UGER": "Cataclysmic",
    "UGSU:E": "Cataclysmic",
    "UGSU:": "Cataclysmic",
    "AM": "Cataclysmic",
    "AM+E": "Cataclysmic",
    "AM:": "Cataclysmic",
    "DQ": "Cataclysmic",
    "DQ:": "Cataclysmic",

    # White Dwarf
    "WHITE DWARF": "White Dwarf",
    "DAV": "White Dwarf",
    "DBV": "White Dwarf",
    "WD": "White Dwarf",
    "ZZA": "White Dwarf",
    "ZZ_CETI": "White Dwarf",
    "ZZ CETI": "White Dwarf",
    "V1093HER": "White Dwarf",
    "V361HYA": "White Dwarf",
    "WHITE_DWARF": "White Dwarf",
    "ZZ": "White Dwarf",
    "ZZ:": "White Dwarf",
    "ZZB": "White Dwarf",
    "ZZLEP": "White Dwarf",
    "ZZLep": "White Dwarf",
    "ZZO": "White Dwarf",

    # Young Stellar Object
    "YOUNG STELLAR OBJECT": "Young Stellar Object",
    "YSO": "Young Stellar Object",
    "TTS": "Young Stellar Object",
    "T TAURI": "Young Stellar Object",

    # Other (controlado)
    "OTHER": "Other",
    "-----": "Unknown",
    "": "Unknown",
    "UNKNOWN": "Unknown"
}

VALID_CLASSES = {
    "Eclipsing Binary",
    "RR Lyrae",
    "Delta Scuti",
    "Rotational",
    "Irregular",
    "Cataclysmic",
    "White Dwarf",
    "Young Stellar Object",
    "Variable"
}

def normalize_label(label):
    if not isinstance(label, str) or label.strip() == "":
        return "Unknown"

    key = label.strip().upper().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())

    value = NORMALIZATION_DICT.get(key)
    if value and value in VALID_CLASSES:
        return value

    # División por separadores comunes
    separators = ['|', '/', '+', ':']
    for sep in separators:
        if sep in key:
            parts = key.split(sep)
            valid_candidates = []
            for part in parts:
                norm = normalize_label(part)
                if norm != "Unknown":
                    valid_candidates.append(norm)

            if valid_candidates:
                # Prioriza clases más frecuentes si hay varias
                for preferred in [
                    "Eclipsing Binary", "RR Lyrae", "Delta Scuti", "Rotational",
                    "Irregular", "Variable", "Cataclysmic", "White Dwarf", "Young Stellar Object"
                ]:
                    if preferred in valid_candidates:
                        return preferred
                return valid_candidates[0]

    key_nospecial = key.replace(" ", "").replace("-", "").replace("_", "")
    for dict_key, dict_val in NORMALIZATION_DICT.items():
        dict_key_nospecial = dict_key.replace(" ", "").replace("-", "").replace("_", "")
        if key_nospecial == dict_key_nospecial and dict_val in VALID_CLASSES:
            return dict_val

    return "Unknown"