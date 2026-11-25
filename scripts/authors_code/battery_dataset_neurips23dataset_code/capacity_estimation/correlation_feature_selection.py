import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from capacity_dataset import CapacityDataset


FEATURE_NAMES = [
    "average_cell_voltage",
    "charging_current",
    "state_of_charge",
    "maximum_cell_voltage",
    "minimum_cell_voltage",
    "maximum_temperature",
    "minimum_temperature",
    "timestamp",
]

N_TIMESTEPS = 128
N_FEATURES = len(FEATURE_NAMES)


def load_training_data(fold_num):
    dataset = CapacityDataset(train=True, fold_num=fold_num)

    if len(dataset) == 0:
        raise ValueError("Il training fold {} è vuoto.".format(fold_num))

    indices = np.arange(len(dataset))

    X = np.empty((len(indices), N_TIMESTEPS, N_FEATURES), dtype=np.float32)
    y = np.empty(len(indices), dtype=np.float32)

    for out_idx, dataset_idx in enumerate(indices):
        data, metadata = dataset[int(dataset_idx)]
        values = np.asarray(data, dtype=np.float32)

        X[out_idx] = values
        y[out_idx] = float(metadata["capacity"])

    return X, y


def pearson_scores(X, y):
    X_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()

    numerator = np.sum(X_centered * y_centered[:, None, None], axis=0)
    denominator = np.sqrt(
        np.sum(X_centered ** 2, axis=0) * np.sum(y_centered ** 2)
    )

    correlations = np.zeros((N_TIMESTEPS, N_FEATURES), dtype=np.float64)
    np.divide(numerator, denominator, out=correlations, where=denominator > 0)
    correlations = np.clip(correlations, -1.0, 1.0)

    absolute = np.abs(correlations)
    return {
        "mean_abs": absolute.mean(axis=0),
        "median_abs": np.median(absolute, axis=0),
        "mean_signed": correlations.mean(axis=0),
    }


def mutual_information_scores(X, y, n_neighbors, seed):
    timestep_scores = np.zeros((N_TIMESTEPS, N_FEATURES), dtype=np.float64)

    for feature_idx in range(N_FEATURES):
        for timestep in range(N_TIMESTEPS):
            values = X[:, timestep, feature_idx]

            # Una variabile costante non contiene informazione utile sul target.
            if np.ptp(values) == 0:
                score = 0.0
            else:
                score = mutual_info_regression(
                    values.reshape(-1, 1),
                    y,
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                    random_state=seed + feature_idx * N_TIMESTEPS + timestep,
                )[0]

            timestep_scores[timestep, feature_idx] = float(score)

        print("Mutual Information: completata feature {} ({}/{})".format(
            FEATURE_NAMES[feature_idx], feature_idx + 1, N_FEATURES
        ))

    return {
        "mean": timestep_scores.mean(axis=0),
        "median": np.median(timestep_scores, axis=0),
    }


def save_fold_results(output_dir, fold_num, n_samples, pearson, mi):
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "feature_scores_fold_{}.txt".format(fold_num)

    rows = []
    for idx, name in enumerate(FEATURE_NAMES):
        rows.append({
            "feature": name,
            "pearson_mean_abs": pearson["mean_abs"][idx],
            "pearson_median_abs": pearson["median_abs"][idx],
            "pearson_mean_signed": pearson["mean_signed"][idx],
            "mutual_information_mean": mi["mean"][idx],
            "mutual_information_median": mi["median"][idx],
        })

    pearson_order = np.argsort(-pearson["mean_abs"])
    mi_order = np.argsort(-mi["mean"])

    with open(str(txt_path), "w", encoding="utf-8") as f:
        f.write("Pearson e Mutual Information - EVBattery\n")
        f.write("Fold: {}\nSnippet analizzati: {}\n\n".format(fold_num, n_samples))
        f.write("Ranking Pearson (media |r| sui 128 timestep):\n")
        for rank, idx in enumerate(pearson_order, 1):
            f.write("{}. {}: {:.8f} (mediana |r|: {:.8f})\n".format(
                rank, FEATURE_NAMES[idx], pearson["mean_abs"][idx],
                pearson["median_abs"][idx]
            ))

        f.write("\nRanking Mutual Information (media sui 128 timestep):\n")
        for rank, idx in enumerate(mi_order, 1):
            f.write("{}. {}: {:.8f} (mediana: {:.8f})\n".format(
                rank, FEATURE_NAMES[idx], mi["mean"][idx], mi["median"][idx]
            ))

    print("Ranking salvati in:", txt_path)

    return rows


def save_summary(output_dir, fold_results):
    txt_path = output_dir / "feature_scores_summary.txt"

    metric_names = [
        "pearson_mean_abs",
        "pearson_median_abs",
        "mutual_information_mean",
        "mutual_information_median",
    ]

    summary_rows = []
    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        row = {"feature": feature_name}
        for metric in metric_names:
            values = np.asarray([
                fold[feature_idx][metric] for fold in fold_results
            ], dtype=np.float64)
            row[metric + "_mean_folds"] = float(values.mean())
            row[metric + "_std_folds"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    pearson_order = sorted(
        range(N_FEATURES),
        key=lambda i: summary_rows[i]["pearson_mean_abs_mean_folds"],
        reverse=True,
    )
    mi_order = sorted(
        range(N_FEATURES),
        key=lambda i: summary_rows[i]["mutual_information_mean_mean_folds"],
        reverse=True,
    )

    with open(str(txt_path), "w", encoding="utf-8") as f:
        f.write("RIEPILOGO FEATURE SELECTION - {} fold\n\n".format(len(fold_results)))
        f.write("Ranking Pearson (media degli score sui fold):\n")
        for rank, idx in enumerate(pearson_order, 1):
            row = summary_rows[idx]
            f.write("{}. {}: {:.8f} +/- {:.8f}\n".format(
                rank, FEATURE_NAMES[idx], row["pearson_mean_abs_mean_folds"],
                row["pearson_mean_abs_std_folds"]
            ))

        f.write("\nRanking Mutual Information (media degli score sui fold):\n")
        for rank, idx in enumerate(mi_order, 1):
            row = summary_rows[idx]
            f.write("{}. {}: {:.8f} +/- {:.8f}\n".format(
                rank, FEATURE_NAMES[idx], row["mutual_information_mean_mean_folds"],
                row["mutual_information_mean_std_folds"]
            ))

    print("Ranking aggregati salvati in:", txt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold_num", type=int, default=0,
        help="Fold da analizzare (0-4); -1 esegue tutti i cinque fold. Default: 0."
    )
    parser.add_argument("--n_neighbors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--output_dir", type=str, default="results/feature_selection_filter/dataset1")
    args = parser.parse_args()

    if args.fold_num not in [-1, 0, 1, 2, 3, 4]:
        parser.error("--fold_num deve essere -1 oppure un valore tra 0 e 4")
    if args.n_neighbors < 1:
        parser.error("--n_neighbors deve essere >= 1")

    output_dir = Path(args.output_dir)
    folds = range(5) if args.fold_num == -1 else [args.fold_num]
    all_fold_rows = []

    for fold_num in folds:
        print("\n" + "=" * 60)
        print("Analisi fold {}".format(fold_num))
        print("=" * 60)

        X, y = load_training_data(fold_num)

        print("Snippet effettivamente analizzati:", len(y))
        print("Shape X:", X.shape)

        print("Calcolo Pearson...")
        pearson = pearson_scores(X, y)

        print("Calcolo Mutual Information...")
        mi = mutual_information_scores(X, y, args.n_neighbors, args.seed)

        rows = save_fold_results(output_dir, fold_num, len(y), pearson, mi)
        all_fold_rows.append(rows)

        del X, y

    if len(all_fold_rows) > 1:
        save_summary(output_dir, all_fold_rows)


if __name__ == "__main__":
    main()
