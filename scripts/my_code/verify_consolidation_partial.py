# =============================================================================
# This script performs a partial verification of the consolidated EVBattery
# datasets by comparing some original pickle (.pkl) file with its
# corresponding record in the processed dataset.
#
# For each dataset, the script retrieves both the original and 
# consolidated data and metadata, and performs a thorough comparison.
# The purpose of this script is not to validate every record in the dataset,
# but to provide a quick consistency check after the consolidation process.
#
# NOTE: The original PKL files and the processed datasets must use the same
#       numerical ordering. The same numeric sorting used during consolidation
#       is therefore applied when selecting the original PKL files.
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
    print("=" * 60)
    print(f"VERIFICA: {dataset_name}")
    print("=" * 60)

    processed_path = PROCESSED_DIR / dataset_name
    data_path = processed_path / "data"
    metadata_path = processed_path / "metadata.pkl"

    # --------------------------------------------------------
    # LETTURA METADATI
    # --------------------------------------------------------

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    n = metadata["n_snippets"]

    # print()
    # for key, value in metadata.items():
    #     print(f"{key}: {value}")
    print("\nMETADATI")
    print("-" * 40)
    print(f"Dataset:       {metadata['dataset_name']}")
    print(f"Numero record: {n}")
    print(f"Shape snippet: {metadata['snippet_shape']}")
    print("\nFeature:")
    for i, feature in enumerate(metadata["feature_names"]):
        print(f"  {i}: {feature}")

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
    # durante la costruzione del dataset processed
    pkl_files = sorted(
        original_dir.glob("*.pkl"),
        key=lambda path: int(path.stem)
    )

    print("\nCORRISPONDENZA RECORD / PKL")
    print("-" * 40)
    print(f"PKL originali trovati: {len(pkl_files)}")
    print(f"Record consolidati:    {n}")

    # Alcuni indici scelti per verificare la corrispondenza
    # tra posizione nel dataset processed e PKL originale
    test_indices = [0, 1, 2, 10, 100, 999]

    for i in test_indices:
        if i < n:
            print(f"record {i:4d} -> {pkl_files[i].name}")

    # --------------------------------------------------------
    # VERIFICA DEI RECORD
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VERIFICA DEI RECORD")
    print("=" * 60)

    for i in test_indices:
        if i >= n:
            continue

        pkl_path = pkl_files[i]
        original_data, original_metadata = extract_pkl_data(pkl_path)

        print()
        print("-" * 60)
        print(f"RECORD PROCESSED: {i}")
        print(f"PKL ORIGINALE:    {pkl_path.name}")
        print("-" * 60)

        # ----------------------------------------------------
        # DATI X
        # ----------------------------------------------------

        # Nel dataset processed X è float32.
        # Converto quindi anche il dato originale
        # a float32 prima del confronto.
        data_ok = np.array_equal(
            original_data.astype(np.float32),
            X[i]
        )

        print()
        print(f"Corrispondenza dati X:              {data_ok}")

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
            raise ValueError(
                f"Label inattesa nel file {pkl_path}: "
                f"{original_metadata['label']}"
            )

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

        print()
        print("Corrispondenza metadati:")
        print(
            f" vehicle             {vehicle_ok}    "
            # f"originale={original_vehicle!r} "
            # f"({type(original_vehicle).__name__}) | "
            # f"consolidato={vehicle[i]!r} "
            # f"({vehicle.dtype})"
        )

        print(
            f" mileage             {mileage_ok}    "
            # f"originale={original_mileage!r} "
            # f"({type(original_mileage).__name__}) | "
            # f"consolidato={mileage[i]!r} "
            # f"({mileage.dtype})"
        )

        print(
            f" charge_segment      {charge_segment_ok}    "
            # f"originale={original_charge_segment!r} "
            # f"({type(original_charge_segment).__name__}) | "
            # f"consolidato={charge_segment[i]!r} "
            # f"({charge_segment.dtype})"
        )

        print(
            f" capacity            {capacity_ok}    "
            # f"originale={original_capacity!r} "
            # f"({type(original_capacity).__name__}) | "
            # f"consolidato={capacity[i]!r} "
            # f"({capacity.dtype})"
        )

        print(
            f" health_label        {health_label_ok}    "
            # f"originale={original_health_label!r} "
            # f"({type(original_health_label).__name__}) | "
            # f"consolidato={health_label[i]!r} "
            # f"({health_label.dtype})"
        )

        # ----------------------------------------------------
        # RISULTATO
        # ----------------------------------------------------

        all_ok = (
            data_ok
            and vehicle_ok
            and mileage_ok
            and charge_segment_ok
            and capacity_ok
            and health_label_ok
        )

        print()

        if all_ok:
            print("RISULTATO: OK")
        else:
            print("RISULTATO: ERRORE")

    print()
    print("=" * 60)
    print("VERIFICA COMPLETATA")
    print("=" * 60)
    print("\n\n\n")

# ============================================================
# MAIN
# ============================================================

def main():
    for dataset_name, dataset_path in DATASETS.items():
        check_dataset(dataset_name, dataset_path)

if __name__ == "__main__":
    main()