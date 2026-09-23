# =============================================================================
# This script performs a complete verification of the consolidated EVBattery
# datasets by comparing every original pickle (.pkl) file with its
# corresponding record in the processed dataset.
#
# For each dataset, the script retrieves both the original and 
# consolidated data and metadata, and performs a thorough comparison.
# Unlike the partial verification script, this script checks the complete
# dataset and can therefore be used as a final validation of the consolidation
# process.
#
# NOTE: The original PKL files and the processed dataset must use the same
#       numerical ordering. The same numeric sorting used during consolidation
#       is therefore applied here.
#
# NOTE: The script assumes this file is located in the "scripts/my_code"
#       directory of the EVBattery project.
# =============================================================================

from pathlib import Path
import pickle
import numpy as np

from utils import extract_pkl_data

# ============================================================
# CONFIGURAZIONE
# ============================================================

# root directory of the project
PROJECT_DIR = Path(__file__).resolve().parents[2]

# original EVBattery datasets to compare with the consolidated datasets
DATASETS = {
    "dataset1": PROJECT_DIR / "original/battery_dataset1",
    "dataset2": PROJECT_DIR / "original/battery_dataset2",
    "dataset3": PROJECT_DIR / "original/battery_dataset3",
}

# directory where the consolidated datasets are stored
PROCESSED_DIR = PROJECT_DIR / "processed"

# ============================================================
# VERIFICA DATASET
# ============================================================

def check_dataset(dataset_name, dataset_path):
    processed_path = PROCESSED_DIR / dataset_name
    data_path = processed_path / "data"
    metadata_path = processed_path / "metadata.pkl"

    # --------------------------------------------------------
    # LETTURA METADATI
    # --------------------------------------------------------

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    n = metadata["n_snippets"]

    # --------------------------------------------------------
    # APERTURA MEMMAP
    # --------------------------------------------------------

    # X contiene i dati delle serie temporali
    X = np.memmap(
        data_path / metadata["storage"]["X"],
        dtype=np.float32,
        mode="r",
        shape=(n, 128, 8),
    )

    # ID del veicolo
    vehicle = np.memmap(
        data_path / metadata["storage"]["vehicle"],
        dtype=np.int16,
        mode="r",
        shape=(n,),
    )

    # Chilometraggio
    mileage = np.memmap(
        data_path / metadata["storage"]["mileage"],
        dtype=np.float64,
        mode="r",
        shape=(n,),
    )

    # Numero del segmento di carica
    charge_segment = np.memmap(
        data_path / metadata["storage"]["charge_segment"],
        dtype=np.int16,
        mode="r",
        shape=(n,),
    )

    # Label dello stato di salute della batteria
    health_label = np.memmap(
        data_path / metadata["storage"]["health_label"],
        dtype=np.int8,
        mode="r",
        shape=(n,),
    )

    # Capacità della batteria (0 se sconosciuta)
    capacity = np.memmap(
        data_path / metadata["storage"]["capacity"],
        dtype=np.float64,
        mode="r",
        shape=(n,),
    )

    # --------------------------------------------------------
    # FILE PKL ORIGINALI
    # --------------------------------------------------------

    original_dir = dataset_path / "data"

    # L'ordinamento deve essere identico a quello utilizzato
    # durante la costruzione del dataset consolidato.
    pkl_files = sorted(
        original_dir.glob("*.pkl"),
        key=lambda path: int(path.stem)
    )

    # --------------------------------------------------------
    # VERIFICA NUMERO RECORD
    # --------------------------------------------------------

    if len(pkl_files) != n:
        return False

    # --------------------------------------------------------
    # VERIFICA DI TUTTI I RECORD
    # --------------------------------------------------------

    for i, pkl_path in enumerate(pkl_files):
        try:
            original_data, original_metadata = extract_pkl_data(pkl_path)
        except Exception:
            return False

        # ----------------------------------------------------
        # DATI X
        # ----------------------------------------------------

        # Nel dataset consolidato X è float32.
        # Converto quindi anche il dato originale
        # a float32 prima del confronto.
        data_ok = np.array_equal(
            original_data.astype(np.float32),
            X[i]
        )

        if not data_ok:
            return False

        # ----------------------------------------------------
        # METADATI
        # ----------------------------------------------------

        original_vehicle = int(
            original_metadata["car"]
        )

        original_mileage = float(
            original_metadata["mileage"]
        )

        original_charge_segment = int(
            original_metadata["charge_segment"]
        )

        original_capacity = float(
            original_metadata["capacity"]
        )

        original_label = str(
            original_metadata["label"]
        ).zfill(2)


        if original_label == "00":
            original_health_label = 0
        elif original_label == "10":
            original_health_label = 1
        else:
            return False

        # ----------------------------------------------------
        # CONFRONTO METADATI
        # ----------------------------------------------------

        vehicle_ok = (
            original_vehicle == vehicle[i]
        )
        mileage_ok = (
            original_mileage == mileage[i]
        )
        charge_segment_ok = (
            original_charge_segment == charge_segment[i]
        )
        capacity_ok = (
            original_capacity == capacity[i]
        )
        health_label_ok = (
            original_health_label == health_label[i]
        )

        # Se anche un solo campo non corrisponde,
        # il dataset viene considerato non corretto.
        if not (
            vehicle_ok
            and mileage_ok
            and charge_segment_ok
            and capacity_ok
            and health_label_ok
        ):
            return False

    # Tutti i PKL sono stati verificati correttamente.
    return True

# ============================================================
# MAIN
# ============================================================

def main():
    for dataset_name, dataset_path in DATASETS.items():
        try:
            correct = check_dataset(
                dataset_name,
                dataset_path
            )
            if correct:
                print(f"{dataset_name}: corretto")
            else:
                print(f"{dataset_name}: Non corretto")
        except Exception:
            print(f"{dataset_name}: Non corretto")

if __name__ == "__main__":
    main()