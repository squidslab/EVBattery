# =============================================================================
# This script consolidates the original EVBattery pickle (.pkl) files into
# a smaller number of binary files using NumPy memory-mapped arrays (memmap).
#
# For each EVBattery dataset, the script reads and validates the
# structure of each pickle file. Then, it extracts the data and metadata 
# from each charging snippet and stores them in separate memmap files. 
# Finally, it saves a metadata file describing the structure and storage format of
# the consolidated dataset.
#
# NumPy memmap is used to avoid loading the entire consolidated dataset into
# RAM during creation and later access.
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

# EVBattery datasets to consolidate
DATASETS = {
    "dataset1": PROJECT_DIR / "original/battery_dataset1",
    "dataset2": PROJECT_DIR / "original/battery_dataset2",
    "dataset3": PROJECT_DIR / "original/battery_dataset3",
}

# the output directory for the consolidated datasets
OUTPUT_DIR = PROJECT_DIR / "processed"

# The maximum number of PKL files to process from each dataset.
# Set to None to process all available PKL files
MAX_SNIPPETS = None

## Feature names and descriptions for the time-series data
FEATURE_NAMES = [
    "average_cell_voltage",
    "charging_current",
    "state_of_charge",
    "maximum_cell_voltage",
    "minimum_cell_voltage",
    "maximum_cell_temperature",
    "minimum_cell_temperature",
    "timestamp",
]

# ============================================================
# COSTRUZIONE DATASET CON MEMMAP
# ============================================================

def build_dataset(dataset_name, dataset_path):
    data_dir = dataset_path / "data"

    # Ordino i PKL numericamente in base al numero contenuto
    # nel nome del file.
    #
    # In questo modo l'ordine sarà:
    # 0.pkl, 1.pkl, 2.pkl, ..., 10.pkl, ..., 100.pkl, ...
    #
    # Un normale sorted() avrebbe invece effettuato un ordinamento
    # alfabetico, mettendo per esempio 10.pkl prima di 2.pkl.
    pkl_files = sorted(
        data_dir.glob("*.pkl"),
        key=lambda path: int(path.stem)
    )

    if MAX_SNIPPETS is not None:
        pkl_files = pkl_files[:MAX_SNIPPETS]

    n = len(pkl_files)
    print("=" * 60)
    print(f"ELABORAZIONE: {dataset_name}")
    print("=" * 60)
    print(f"PKL da elaborare: {n}")

    if n == 0:
        print("Nessun PKL trovato.")
        return

    output_path = OUTPUT_DIR / dataset_name / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    # File di output
    x_path = output_path / "X.dat"
    vehicle_path = output_path / "vehicle.dat"
    mileage_path = output_path / "mileage.dat"
    charge_segment_path = output_path / "charge_segment.dat"
    health_label_path = output_path / "health_label.dat"
    capacity_path = output_path / "capacity.dat"

    # X contiene 128 punti temporali × 8 feature per ogni snippet.
    # float32 permette di ridurre sensibilmente l'occupazione
    # di memoria mantenendo una precisione adeguata per i dati
    # numerici del dataset
    X = np.memmap(
        x_path,
        dtype=np.float32,
        mode="w+",
        shape=(n, 128, 8),
    )

    # Gli identificativi dei veicoli presenti nel dataset rientrano
    # nell'intervallo rappresentabile da int16 (-32768, 32767)
    vehicle = np.memmap(
        vehicle_path,
        dtype=np.int16,
        mode="w+",
        shape=(n,),
    )

    # Il mileage viene mantenuto come float64 per non perdere
    # precisione rispetto al valore originale
    mileage = np.memmap(
        mileage_path,
        dtype=np.float64,
        mode="w+",
        shape=(n,),
    )

    # Il numero del charge segment è un valore intero che rientra
    # nell'intervallo rappresentabile da int16 (-32768, 32767)
    charge_segment = np.memmap(
        charge_segment_path,
        dtype=np.int16,
        mode="w+",
        shape=(n,),
    )

    # La health label originale usa le stringhe "00" e "10".
    # Viene convertita rispettivamente in 0 e 1, rappresentati in int8
    health_label = np.memmap(
        health_label_path,
        dtype=np.int8,
        mode="w+",
        shape=(n,),
    )

    # La capacity viene mantenuta come float64 per conservare
    # la precisione del valore originale
    capacity = np.memmap(
        capacity_path,
        dtype=np.float64,
        mode="w+",
        shape=(n,),
    )

    # --------------------------------------------------------
    # LETTURA E SCRITTURA PROGRESSIVA
    # --------------------------------------------------------

    # Elaborazione PKL
    for i, pkl_path in enumerate(pkl_files):
        data, metadata = extract_pkl_data(pkl_path)

        # Metadata
        car = int(metadata["car"])
        label = str(metadata["label"]).zfill(2)
        mileage_value = float(metadata["mileage"])
        charge_segment_value = int(metadata["charge_segment"])
        capacity_value = float(metadata["capacity"])

        # Conversione label:
        if label == "00":
            health_label[i] = 0
        elif label == "10":
            health_label[i] = 1
        else:
            raise ValueError(
                f"Label inattesa nel file {pkl_path}: {metadata['label']}"
            )

        # Dati
        X[i] = data.astype(np.float32, copy=False)

        # Metadati
        vehicle[i] = car
        mileage[i] = mileage_value
        charge_segment[i] = charge_segment_value
        capacity[i] = capacity_value

        # Progressione
        if (i + 1) % 10000 == 0 or i + 1 == n:
            print(f"Elaborati: {i + 1}/{n}")

    # Scrittura effettiva su disco
    X.flush()
    vehicle.flush()
    mileage.flush()
    charge_segment.flush()
    health_label.flush()
    capacity.flush()

    # --------------------------------------------------------
    # METADATI DEL DATASET
    # --------------------------------------------------------

    # Metadati del dataset consolidato
    metadata_path = OUTPUT_DIR / dataset_name / "metadata.pkl"

    dataset_metadata = {
        "dataset_name": dataset_name,
        "n_snippets": n,
        "snippet_shape": (128, 8),
        "feature_names": FEATURE_NAMES,
        "storage": {
            "X": x_path.name,
            "vehicle": vehicle_path.name,
            "mileage": mileage_path.name,
            "charge_segment": charge_segment_path.name,
            "health_label": health_label_path.name,
            "capacity": capacity_path.name,
        },

        # Tipi utilizzati nei file memmap.
        #
        # Sono riportati anche nei metadati in modo che,
        # quando il dataset verrà riaperto in futuro,
        # sia possibile sapere quale dtype utilizzare
        # per leggere correttamente ogni file
        "dtypes": {
            "X": "float32",
            "vehicle": "int16",
            "mileage": "float64",
            "charge_segment": "int16",
            "health_label": "int8",
            "capacity": "float64",
        },
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(dataset_metadata, f) # scrivo i metadati su disco

    # Rimuovo i riferimenti agli oggetti memmap per liberarli
    # dopo aver completato la scrittura su disco
    del X
    del vehicle
    del mileage
    del charge_segment
    del health_label
    del capacity

    print()
    print(f"Dataset completato: {dataset_name}")
    print(f"Metadati: {metadata_path}")
    print()

# ============================================================
# MAIN
# ============================================================

def main():
    for dataset_name, dataset_path in DATASETS.items():
        build_dataset(dataset_name, dataset_path)

if __name__ == "__main__":
    main()