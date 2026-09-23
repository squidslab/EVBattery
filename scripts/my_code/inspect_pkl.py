# =============================================================================
# This script is used to inspect the structure and contents of selected
# pickle files from the original EVBattery dataset.
#
# Expected structure of a pickle file:
#   object 0 -> int
#   object 1 -> int
#   object 2 -> dict
#   object 3 -> (data_array, metadata)
#   object 4 -> []
#
# The data array contains the time-series measurements of a charging snippet,
# while the metadata dictionary contains information such as the vehicle
# number, charge segment, mileage, label and capacity.
# 
# NOTE: The script assumes this file is located in the "scripts/my_code" 
#   directory of the EVBattery project.
# =============================================================================

import pickle
from pathlib import Path

# root directory of the project
PROJECT_DIR = Path(__file__).resolve().parents[2]

# directory containing the original EVBattery datasets
BASE_DIR = PROJECT_DIR / "original"

# EVBattery datasets to inspect
DATASETS = [
    "battery_dataset1",
    "battery_dataset2",
    "battery_dataset3",
]

# sample pickle files selected for inspection
PKLS = [
    "0.pkl",
    "100000.pkl",
    "100002.pkl",
]

for dataset in DATASETS:
    for pkl in PKLS:
        file_path = BASE_DIR / dataset / "data" / pkl

        with open(file_path, "rb") as f:
            objects = []

            # a pickle file may contain multiple objects written sequentially
            # keep loading objects until the end of the file is reached
            # otherwise, it would have loaded only the first object and ignored the rest
            while True:
                try:
                    objects.append(pickle.load(f))
                except EOFError:
                    break

        print(f"\n============ DATASET {dataset.replace('battery_dataset', '')}" \
               f" - PKL {pkl} ============")

        print(f"\n=== OGGETTI ===")
        for i, obj in enumerate(objects):
            if i == 3: continue  # Skip printing the data object
            print(f"Oggetto {i}: {type(obj)} -> {obj}")

        data = objects[3]   # tuple (array, dict)
        array = data[0]     # numPy array containing the time-series data
        metadata = data[1]  # dictionary containing metadata for the snippet
        
        print("\n=== DATI ===")
        print("Tipo:", type(array))
        print("Shape:", array.shape)
        print("Dtype:", array.dtype)

        print("\n=== VALORI ===")
        print("Prima riga:")
        print(array[0])
        print("Ultima riga:")
        print(array[-1])

        print("\n=== METADATI ===")
        for key, value in metadata.items():
            print(f"{key}: {value}")