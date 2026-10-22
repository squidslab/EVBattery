"""
Permutation Feature Importance per XGBoost per la stima della capacity EVBattery.

Esecuzione dalla cartella capacity_estimation:
    python analizing_feature_selection.py --fold_num 0

La procedura usa il training fold esterno e ne ricava una validation interna
separata per veicolo. Il test fold esterno non viene utilizzato per selezionare
le feature. XGBoost viene addestrato una sola volta; per ogni feature si valuta
l'RMSE dopo averne permutato i valori tra snippet.

Nota: gli snippet EVBattery hanno forma (128, 8). Per XGBoost vengono appiattiti
in 1024 colonne (128 timestep * 8 feature). Per stimare l'importanza di una
feature originale, la permutazione scambia l'intera sequenza temporale di quella
feature tra snippet, quindi permuta in blocco le 128 colonne corrispondenti.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from torch.utils.data import Subset

from capacity_dataset import CapacityDataset
from utils import Normalizer, PreprocessNormalizer

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
N_TIMESTEPS = 128
N_FEATURES = len(FEATURE_NAMES)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_car_ids(dataset):
    car_ids = []
    for i in range(len(dataset)):
        _, metadata = dataset[i]
        car_ids.append(int(metadata["car"]))
    return np.asarray(car_ids)


def split_indices_by_car(car_ids, validation_fraction=0.2, seed=42):
    unique_cars = np.unique(car_ids)
    if len(unique_cars) < 2:
        raise ValueError("Servono almeno due veicoli nel training fold.")

    rng = np.random.RandomState(seed)
    shuffled_cars = unique_cars.copy()
    rng.shuffle(shuffled_cars)

    n_val_cars = max(1, int(round(len(shuffled_cars) * validation_fraction)))
    n_val_cars = min(n_val_cars, len(shuffled_cars) - 1)
    validation_cars = set(shuffled_cars[:n_val_cars].tolist())

    train_indices = [
        i for i, car_id in enumerate(car_ids)
        if int(car_id) not in validation_cars
    ]
    val_indices = [
        i for i, car_id in enumerate(car_ids)
        if int(car_id) in validation_cars
    ]
    return train_indices, val_indices


def make_normalized_datasets(base_dataset, train_indices, val_indices):
    # Stesso approccio di normalizzazione di EVBattery
    n_reference = min(200, len(train_indices))
    if n_reference == 0:
        raise ValueError("Il training interno è vuoto.")

    reference_data = [base_dataset[train_indices[i]][0]
                      for i in range(n_reference)]
    normalizer = Normalizer(dfs=reference_data, variable_length=False)

    train_subset = Subset(base_dataset, train_indices)
    val_subset = Subset(base_dataset, val_indices)
    train_dataset = PreprocessNormalizer(
        train_subset, normalizer_fn=normalizer.norm_func
    )
    val_dataset = PreprocessNormalizer(
        val_subset, normalizer_fn=normalizer.norm_func
    )
    return train_dataset, val_dataset


def collect_flat_data(dataset, max_samples=0, seed=42):
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset vuoto.")

    indices = np.arange(n)
    if max_samples > 0 and n > max_samples:
        rng = np.random.RandomState(seed)
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))

    X = np.empty((len(indices), N_TIMESTEPS * N_FEATURES), dtype=np.float32)
    y = np.empty(len(indices), dtype=np.float32)

    for out_i, dataset_i in enumerate(indices):
        data, metadata = dataset[int(dataset_i)]
        arr = np.asarray(data, dtype=np.float32)

        if arr.shape != (N_TIMESTEPS, N_FEATURES):
            raise ValueError(
                "Forma snippet inattesa {}: attesa ({}, {}).".format(
                    arr.shape, N_TIMESTEPS, N_FEATURES
                )
            )

        X[out_i] = arr.reshape(-1)
        y[out_i] = float(metadata["capacity"])

    return X, y


def make_xgb_model(args):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method=args.tree_method,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def permutation_importance(model, X, y, n_repeats, seed):
    baseline_rmse = rmse(y, model.predict(X))
    rng = np.random.RandomState(seed)
    rows = []

    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        increases = []
        start = feature_idx
        columns = np.arange(start, N_TIMESTEPS * N_FEATURES, N_FEATURES)

        for _ in range(n_repeats):
            X_permuted = X.copy()
            permutation = rng.permutation(len(X))
            # Permuta la feature originale su tutti i 128 timestep in blocco
            X_3d = X_permuted.reshape(-1, N_TIMESTEPS, N_FEATURES)
            X_original_3d = X.reshape(-1, N_TIMESTEPS, N_FEATURES)
            X_3d[:, :, feature_idx] = X_original_3d[permutation, :, feature_idx]
            X_permuted = X_3d.reshape(len(X), -1)

            permuted_rmse = rmse(y, model.predict(X_permuted))
            increases.append(permuted_rmse - baseline_rmse)

        row = {
            "feature": feature_name,
            "baseline_rmse": baseline_rmse,
            "mean_rmse_increase": float(np.mean(increases)),
            "std_rmse_increase": float(np.std(increases, ddof=1))
                if len(increases) > 1 else 0.0,
            "repeats": n_repeats,
        }
        rows.append(row)
        print("{:<28} incremento RMSE: {:+.6f} (dev.std. {:.6f})".format(
            feature_name, row["mean_rmse_increase"], row["std_rmse_increase"]
        ))

    rows.sort(key=lambda row: row["mean_rmse_increase"], reverse=True)
    return baseline_rmse, rows


def main():
    parser = argparse.ArgumentParser(
        description="Permutation Feature Importance per XGBoost/EVBattery"
    )
    parser.add_argument("--fold_num", type=int, default=0, choices=range(5))
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--tree_method", type=str, default="hist",
                        choices=["auto", "exact", "approx", "hist"])
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--n_repeats", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Limite training interno; 0 = tutti.")
    parser.add_argument("--max_eval_samples", type=int, default=0,
                        help="Limite validation; 0 = tutti.")
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--output_dir", type=str,
                        default="results/feature_selection_xgboost")
    args = parser.parse_args()

    if args.n_estimators < 1 or args.n_repeats < 1:
        parser.error("--n_estimators e --n_repeats devono essere >= 1")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation_fraction deve essere compresa tra 0 e 1")

    set_seed(args.seed)
    print("Fold esterno:", args.fold_num)
    print("XGBoost:", xgb.__version__)

    base_train = CapacityDataset(train=True, fold_num=args.fold_num)
    print("Snippet nel training fold:", len(base_train))

    car_ids = get_car_ids(base_train)
    train_indices, val_indices = split_indices_by_car(
        car_ids, args.validation_fraction, args.seed
    )
    print("Snippet training interno:", len(train_indices))
    print("Snippet validation interna:", len(val_indices))

    train_dataset, val_dataset = make_normalized_datasets(
        base_train, train_indices, val_indices
    )
    X_train, y_train = collect_flat_data(
        train_dataset, args.max_train_samples, args.seed + 1
    )
    X_val, y_val = collect_flat_data(
        val_dataset, args.max_eval_samples, args.seed + 2
    )

    print("Matrice X train:", X_train.shape, X_train.dtype)
    print("Matrice X validation:", X_val.shape, X_val.dtype)
    print("Training XGBoost...")

    model = make_xgb_model(args)
    model.fit(X_train, y_train)

    baseline, rows = permutation_importance(
        model, X_val, y_val, args.n_repeats, args.seed + 3
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "feature_importance_fold_{}.csv".format(args.fold_num)
    txt_path = output_dir / "feature_importance_fold_{}.txt".format(args.fold_num)

    with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "feature", "baseline_rmse", "mean_rmse_increase",
            "std_rmse_increase", "repeats"
        ])
        writer.writeheader()
        writer.writerows(rows)

    with open(str(txt_path), "w", encoding="utf-8") as f:
        f.write("Permutation Feature Importance - XGBoost\n")
        f.write("Fold esterno: {}\n".format(args.fold_num))
        f.write("XGBoost version: {}\n".format(xgb.__version__))
        f.write("Training interno: {} snippet\n".format(len(X_train)))
        f.write("Validation valutata: {} snippet\n".format(len(y_val)))
        f.write("RMSE baseline: {:.6f}\n\n".format(baseline))
        f.write("Ranking (incremento RMSE decrescente):\n")
        for rank, row in enumerate(rows, start=1):
            f.write(
                "{}. {}: {:+.6f} (dev.std. {:.6f})\n".format(
                    rank, row["feature"], row["mean_rmse_increase"],
                    row["std_rmse_increase"]
                )
            )

    print("\nRMSE baseline: {:.6f}".format(baseline))
    print("Ranking salvato in:", txt_path)
    print("Risultati CSV salvati in:", csv_path)
    print("Importante: incremento RMSE maggiore = feature più utile in questo modello.")


if __name__ == "__main__":
    main()
