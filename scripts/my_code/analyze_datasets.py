# =============================================================================
# This script analyzes the contents of the three original EVBattery datasets.
#
# For each dataset, the script loads the objects stored in each pickle file 
# and extracts the metadata from the main data object (object 3). Then it
# counts the total number of charging snippets, the number of distinct vehicles, 
# the number of vehicles with anomalous battery health, the occurrences of 
# each health label and the number of charging snippets with a valid capacity label.
#
# The results are printed separately for each of the three datasets.
#
# NOTE: The script assumes this file is located in the "scripts/my_code"
#   directory of the EVBattery project.
# =============================================================================

from pathlib import Path
from collections import Counter

from utils import extract_pkl_data

# root directory of the project
PROJECT_DIR = Path(__file__).resolve().parents[2]

# directory containing the original EVBattery datasets
BASE_DIR = PROJECT_DIR / "original"

# EVBattery datasets to analyze
DATASETS = [
    "battery_dataset1",
    "battery_dataset2",
    "battery_dataset3",
]

for dataset in DATASETS:
    data_dir = BASE_DIR / dataset / "data"

    label_counter = Counter()
    all_vehicles = set()
    vehicles_with_anomalies = set()
    capacity_labeled = 0
    total = 0

    print(f"\n\n====== {dataset.upper()} ======")

    for file_path in data_dir.glob("*.pkl"):
        data, metadata = extract_pkl_data(file_path)

        vehicle_id = metadata["car"]
        label = metadata["label"]
        capacity = metadata["capacity"]

        all_vehicles.add(vehicle_id)

        label_counter[label] += 1

        if label != "00":
            vehicles_with_anomalies.add(vehicle_id)

        if capacity > 0:
            capacity_labeled += 1

        total += 1

    print(f"\nNumero di charging snippets totali: {total}")

    print(f"Numero di veicoli totali: {len(all_vehicles)}")
    print(f"Numero di veicoli con salute della batteria anomala: {len(vehicles_with_anomalies)}")    

    print("\nLabel:")
    for label, count in sorted(label_counter.items()):
        print(f"  {label}: {count}")
    
    print(f"\nNumero di charging snippets di cui è stata rilevata la capacità della batteria: {capacity_labeled}")