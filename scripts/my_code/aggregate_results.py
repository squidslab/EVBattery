from pathlib import Path
import re
import shutil
import statistics
import argparse
import math

FEATURES = [
    "average_cell_voltage", # 0
    "charging_current",     # 1
    "state_of_charge",      # 2
    "maximum_cell_voltage", 
    "minimum_cell_voltage", # 4
    "maximum_temperature",  
    "minimum_temperature",  # 6
    "timestamp"             
]

selected_features = [0, 1, 2, 4, 6] 

def read_test_rmse(file_path: Path, model: str) -> float:
    """Legge test RMSE o converte Test Loss in RMSE per LSTMNet."""
    text = file_path.read_text(encoding="utf-8")

    match = re.search(
        r"Test\s+RMSE\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(f"Valore 'Test RMSE' non trovato in {file_path.name}")

    return float(match.group(1))


def read_test_rmse(file_path: Path, model: str) -> float:
    """Legge Test RMSE oppure converte Test MSE in RMSE per LSTMNet."""
    text = file_path.read_text(encoding="utf-8")

    if model.lower() == "lstmnet":
        metric_name = "Test Loss"
    else:
        metric_name = "Test RMSE"

    match = re.search(
        rf"{re.escape(metric_name)}\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Valore '{metric_name}' non trovato in {file_path.name}"
        )

    value = float(match.group(1))

    # LSTMNet: converte MSE in RMSE
    if model.lower() == "lstmnet":
        return math.sqrt(value)

    return value

def main():
    parser = argparse.ArgumentParser(
        description="Aggrega i Test RMSE dei file results_fold_*.txt."
    )
    parser.add_argument(
        "--results-dir",
        default="LSTMNet/feature_sel_5/",
        help="Cartella contenente i file results_fold_*.txt",
    )
    parser.add_argument(
        "--model",
        default="LSTMNet",
        # default="XGBoost",
        # default="RandomForest",
        help="Nome del modello da riportare nel riepilogo",
    )
    parser.add_argument(
        "--dataset",
        default="ALL BATTERY DATASETS",
        # default="BATTERY DATASET 1",
        # default="BATTERY DATASET 2",
        help="Nome del dataset da riportare nel riepilogo",
    )
    parser.add_argument(
        "--sampling",
        # default="1%",
        # default="25%",
        # default="50%",
        default="100%",
        help="Percentuale di sampling da riportare nel riepilogo",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Percorso del file di output (default: results.txt nella cartella)",
    )
    args = parser.parse_args()

    source_dir = Path("../authors_code/battery_dataset_neurips23dataset_code/capacity_estimation/results") / args.results_dir
    
    if not source_dir.is_dir():
        raise SystemExit(f"Cartella non trovata: {source_dir}")

    # Il pattern evita di includere il file riepilogativo results_ds1_RF.txt.
    fold_files = sorted(
        source_dir.glob("results_fold_*.txt"),
        key=lambda p: int(re.search(r"results_fold_(\d+)", p.name).group(1))
        if re.search(r"results_fold_(\d+)", p.name)
        else p.name,
    )

    values = []
    errors = []

    for file_path in fold_files:
        try:
            values.append((file_path, read_test_rmse(file_path, args.model)))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    if errors:
        raise SystemExit("Impossibile leggere tutti i risultati:\n- " + "\n- ".join(errors))

    if not values:
        raise SystemExit("Nessun Test RMSE valido trovato.")

    rmse_values = [value for _, value in values]
    mean_rmse = statistics.mean(rmse_values)

    match args.dataset:
        case "BATTERY DATASET 1":
            results_dir = source_dir / "dataset1"
            name = "dset1"
        case "BATTERY DATASET 2":
            results_dir = source_dir / "dataset2"
            name = "dset2"
        case "ALL BATTERY DATASETS":
            results_dir = source_dir / "all_datasets"
            name = "all"
    
    results_dir.mkdir(parents=True, exist_ok=True)

    for file in source_dir.glob("results_fold_*.txt"):
        shutil.move(str(file), str(results_dir / file.name))

    # Deviazione standard della popolazione: divide per N, come nel riepilogo
    # RMSE ± deviazione standard comunemente calcolato sui fold.
    std_rmse = statistics.pstdev(rmse_values) if len(rmse_values) > 1 else 0.0

    lines = [
        "=" * 40,
        f"{args.dataset} - MODEL {args.model}",
        "=" * 40,
        "",
        f"Sampling: {args.sampling}",
        "",
        "Feature Selection:",
    ]

    for i, selected_feature in enumerate(selected_features):
        lines.append(f"    Feature {i}: {FEATURES[selected_feature]}")

    lines.append("-" * 40)
    lines.append("")

    for file_path, value in values:
        fold_match = re.search(r"results_fold_(\d+)", file_path.name)
        fold_label = fold_match.group(1) if fold_match else file_path.stem
        lines.append(f"Fold {fold_label} -> Test RMSE: {value:.4f}")

    lines.extend([
        "",
        "=" * 40, 
        f"RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}",
        "",
    ])

    args.model = "RF" if args.model == "RandomForest" else args.model
    file_name = f"results_{name}_{args.model}.txt"
    
    output_path = Path(args.output) if args.output else results_dir / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    main()
