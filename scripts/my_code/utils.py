# =============================================================================
# This file contains utility functions that can be used across multiple 
# scripts in the EVBattery project.
# 
# The following functions are provided:
# - extract_pkl_data: Extracts the data array and metadata dictionary from a
#   pickle file, performing type and shape checks.
# =============================================================================

import pickle
import numpy as np

# ============================================================
# ESTRAZIONE DI DATI DA UN PKL
# ============================================================

# NOTE: Comment the checks to improve performance during batch processing. 
# Uncomment them if you want to enforce strict type and shape checks.
def extract_pkl_data(pkl_path):
    with open(pkl_path, "rb") as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        obj3 = pickle.load(f) # <-- the interesting one
        pickle.load(f)

    if not isinstance(obj3, tuple) or len(obj3) != 2:
        raise TypeError(
            f"La struttura del quarto oggetto non è la tuple attesa: {type(obj3)}"
        )

    data = obj3[0] # <-- the data array
    metadata = obj3[1] # <-- the metadata dictionary

    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"Il primo elemento della tuple non è un numpy.ndarray: {type(data)}"
        )

    if data.shape != (128, 8):
        raise ValueError(
            f"Shape inattesa in {pkl_path}: {data.shape}, attesa (128, 8)"
        )

    if not isinstance(metadata, dict):
        raise TypeError(
            f"Il secondo elemento della tuple non è un dict: {type(metadata)}"
        )

    return data, metadata
