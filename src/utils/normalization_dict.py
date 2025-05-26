NORMALIZATION_DICT = {
    # Eclipsing Binary
    "EA": "Eclipsing Binary",
    "EB": "Eclipsing Binary",
    "EW": "Eclipsing Binary",
    "ECL": "Eclipsing Binary",
    "Detached Binary": "Eclipsing Binary",
    "Semi-Detached Binary": "Eclipsing Binary",
    "Contact Binary": "Eclipsing Binary",
    
    # RR Lyrae
    "RRAB": "RR Lyrae",
    "RRC": "RR Lyrae",
    "RRD": "RR Lyrae",
    "ARRD": "RR Lyrae",
    "RR": "RR Lyrae",
    "ACEP": "RR Lyrae",
    "RRC|EC": "RR Lyrae",

    # Delta Scuti
    "DSCT": "Delta Scuti",
    "HADS": "Delta Scuti",
    "DSCT:": "Delta Scuti",
    "DSCT|GDOR|SXPHE": "Delta Scuti",

    # Rotational
    "ROT": "Rotational",
    "BY": "Rotational",
    "ACV": "Rotational",
    "ROAP": "Rotational",
    "ROAM": "Rotational",
    "ELL": "Rotational",
    "Spotted Rotator": "Rotational",

    # Irregular
    "LB": "Irregular",
    "L": "Irregular",
    "Y": "Irregular",
    "SRC": "Irregular",
    "SRA": "Irregular",
    "SRB": "Irregular",
    "SR": "Irregular",

    # Variable
    "VAR": "Variable",
    "VAR: ": "Variable",

    # White Dwarf
    "DAV": "White Dwarf",
    "DBV": "White Dwarf",
    "WD": "White Dwarf",
    "ZZA": "White Dwarf",
    "V1093HER": "White Dwarf",

    # Cataclysmic
    "UG": "Cataclysmic",
    "UGSU": "Cataclysmic",
    "UGSS": "Cataclysmic",
    "UGZ": "Cataclysmic",
    "NL": "Cataclysmic",
    "NOVA": "Cataclysmic",
    "CV": "Cataclysmic",

    # YSO
    "YSO": "Young Stellar Object",
    "TTS": "Young Stellar Object",
    "T Tauri": "Young Stellar Object",

    # Other (controlado)
    "Other": "Other"
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
    if not isinstance(label, str):
        return "Unknown"
    value = NORMALIZATION_DICT.get(label.strip().upper(), "Unknown")
    return value if value in VALID_CLASSES else "Unknown"