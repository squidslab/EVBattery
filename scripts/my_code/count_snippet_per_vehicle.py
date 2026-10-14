# =============================================================================
# This script counts the number of charging snippets per vehicle.
#
# For each dataset, the script loads the objects stored in each pickle file 
# and extracts the metadata from the main data object. Then it counts the total
# number of charging snippets and the number of charging snippets with a valid 
# capacity label for each vehicle.
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

total_snippets_per_car = Counter()
total_labeled_snippets_per_car = Counter()

for dataset in DATASETS:
    data_dir = BASE_DIR / dataset / "data"
    
    # total_files = len(list(data_dir.glob('*.pkl')))
    # file_count = 0
    
    print(f"\n\n====== {dataset.upper()} ======")
    
    for file_path in data_dir.glob("*.pkl"):
        data, metadata = extract_pkl_data(file_path)

        total_snippets_per_car[metadata["car"]] += 1

        if metadata["capacity"] != 0:
            total_labeled_snippets_per_car[metadata["car"]] += 1

        # files_count += 1

        # if files_count % 10000 == 0:
        #     print(f"Processed {files_count} files out of {total_files} in {dataset}")

    # for vehicle, total_snippets in total_snippets_per_car.items():
    #     print(f"\nNumero di charging snippets totali per il veicolo {vehicle}: {total_snippets}")

    print(f"\nVeicolo con il maggior numero di charging snippets: {total_snippets_per_car.most_common(1)[0][0]} con {total_snippets_per_car.most_common(1)[0][1]} charging snippets")
    print(f"\nVeicolo con il maggior numero di charging snippets con capacity label valida: {total_labeled_snippets_per_car.most_common(1)[0][0]} con {total_labeled_snippets_per_car.most_common(1)[0][1]} charging snippets")

print("=" * 10)
print(f"\nVeicolo numero 51 -> validi / totali: {total_labeled_snippets_per_car[51]} / {total_snippets_per_car[51]}")
print(f"\nVeicolo numero 56 -> validi / totali: {total_labeled_snippets_per_car[56]} / {total_snippets_per_car[56]}")
print("=" * 10)
print(f"\nVeicolo numero 370 -> validi / totali: {total_labeled_snippets_per_car[370]} / {total_snippets_per_car[370]}")
print(f"\nVeicolo numero 431 -> validi / totali: {total_labeled_snippets_per_car[431]} / {total_snippets_per_car[431]}")
print("=" * 10)
print(f"\nVeicolo numero 500 -> validi / totali: {total_labeled_snippets_per_car[500]} / {total_snippets_per_car[500]}")
print(f"\nVeicolo numero 521 -> validi / totali: {total_labeled_snippets_per_car[521]} / {total_snippets_per_car[521]}")
print("=" * 10)