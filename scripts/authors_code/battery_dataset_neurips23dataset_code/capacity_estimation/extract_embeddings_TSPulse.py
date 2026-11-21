import argparse
import gc
import json
from pathlib import Path

import numpy as np

FEATURES = [
    "average_cell_voltage",
    "charging_current",
    "state_of_charge",
    "maximum_cell_voltage",
    "minimum_cell_voltage",
    "maximum_cell_temperature",
    "minimum_cell_temperature",
    "timestamp",
]

N_FEATURES = len(FEATURES)
N_TIMESTEPS = 128
MODEL_CONTEXT = 512

MODEL_ID = "ibm-granite/granite-timeseries-tspulse-r1"
REVISION = "tspulse-block-dualhead-512-p16-r1"

def build_features(full, d_model, num_reg):
    """
    full: (N, F) embedding completo per sequenza (stesso layout di get_embeddings(mode="full")).
    Ritorna (N, D) con le parti richieste concatenate.
    """
    reg_size = num_reg * d_model

    if reg_size <= 0:
        raise ValueError("Il modello non ha register token.")
    
    # Estrae i register token e li organizza in: (N, num_reg, d_model)
    register_tokens = full[:, -reg_size:].reshape(full.shape[0], num_reg, d_model)

    # Media dei 5 register token (N, 5, 48) -> (N, 48)
    return register_tokens.mean(axis=1).astype(np.float32, copy=False)


def to_snippet_features(per_channel, batch_size, n_channels):
    """(B*C, D) -> (B, C*D), canale-maggiore."""
    return per_channel.reshape(batch_size, n_channels, -1).reshape(batch_size, -1)


def load_datasets(fold_num):
    from capacity_dataset import CapacityDataset
    from utils import Normalizer, PreprocessNormalizer

    train_pre = CapacityDataset(train=True, fold_num=fold_num)
    test_pre = CapacityDataset(train=False, fold_num=fold_num)

    normalizer = Normalizer(dfs=[train_pre[i][0] for i in range(200)], variable_length=False)

    train_dataset = PreprocessNormalizer(train_pre, normalizer_fn=normalizer.norm_func)
    test_dataset = PreprocessNormalizer(test_pre, normalizer_fn=normalizer.norm_func)

    return train_dataset, test_dataset


def extract_split(model, dataset, output_dir, split, batch_size, max_samples):
    import torch
    import torch.nn.functional as F
    from tsfm_public.models.tspulse.utils.helpers import get_embeddings

    n_samples = len(dataset)
    if max_samples > 0:
        n_samples = min(n_samples, max_samples)
    if n_samples == 0:
        raise ValueError(f"{split} dataset vuoto.")

    cfg = model.config
    d_model = cfg.d_model_layerwise[-1]
    num_reg = cfg.patch_register_tokens or 0

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"{split}_embeddings.npy"
    labels_path = output_dir / f"{split}_labels.npy"

    embeddings_file = None
    labels = np.empty(n_samples, dtype=np.float32)
    per_channel_dim = None

    model.eval()

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            items = [dataset[i] for i in range(start, end)]

            data = torch.stack([
                torch.as_tensor(it[0], dtype=torch.float32) for it in items
            ])
            labels[start:end] = [it[1]["capacity"] for it in items]

            b = data.shape[0]

            # (B, 128, 8) -> (B*8, 128, 1)
            data = data.permute(0, 2, 1).reshape(b * N_FEATURES, N_TIMESTEPS, 1)

            # (B*8, 128, 1) -> (B*8, 512, 1): il checkpoint vuole contesto 512
            data = F.interpolate(
                data.transpose(1, 2),
                size=MODEL_CONTEXT,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

            # Una sola chiamata al backbone; le parti si ricavano dal vettore
            full = get_embeddings(
                model=model,
                past_values=data,
                component="backbone",
                mode="full",
            )
            # (B*8, 1, F) -> (B*8, F)
            full = full.reshape(full.shape[0], -1).cpu().numpy()

            if embeddings_file is None:
                print(f"\nLayout embedding: F={full.shape[1]}, d_model={d_model}, register_tokens={num_reg}")

            per_channel = build_features(full, d_model, num_reg)
            feats = to_snippet_features(per_channel, b, N_FEATURES)

            if embeddings_file is None:
                per_channel_dim = per_channel.shape[1]
                embeddings_file = np.lib.format.open_memmap(
                    embeddings_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_samples, feats.shape[1]),
                )
                print(
                    f"{split}: {per_channel_dim} valori per canale, "
                    f"{feats.shape[1]} per snippet "
                    f"(originale: {N_TIMESTEPS * N_FEATURES})"
                )

            embeddings_file[start:end] = feats
            print(f"{split}: {end}/{n_samples}", end="\r")

            del items, data, full, per_channel, feats

    embeddings_file.flush()
    del embeddings_file
    np.save(labels_path, labels)
    gc.collect()

    print(f"\nSaved: {embeddings_path}")
    print(f"Saved: {labels_path}")

    return {
        "per_channel_dim": int(per_channel_dim),
        "d_model": int(d_model),
        "register_tokens": int(num_reg),
    }


def main():
    parser = argparse.ArgumentParser(description="Estrazione di embedding TSPulse compatti")
    parser.add_argument("--fold_num", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_root", type=str, default="embeddings")
    parser.add_argument("--dataset_name", type=str, default="all_datasets")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    print("Arguments:", args)

    import torch
    from tsfm_public.models.tspulse import TSPulseForClassification

    device = torch.device("cpu")

    print("\nLoading TSPulse...")
    model = TSPulseForClassification.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        disable_mask_in_classification_eval=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Input channels:", model.config.num_input_channels)
    if model.config.num_input_channels != 1:
        raise ValueError("Il checkpoint non e' configurato per un singolo canale.")

    print(f"\nLoading fold {args.fold_num}...")
    train_dataset, test_dataset = load_datasets(args.fold_num)

    output_dir = (
        Path(args.out_root)
        / f"{args.dataset_name}"
        / f"fold_{args.fold_num}"
    )

    info = None
    for split, ds in [("train", train_dataset), ("test", test_dataset)]:
        print(f"\nExtracting {split} embeddings...")
        info = extract_split(model, ds, output_dir, split, args.batch_size, args.max_samples)

    meta = {
        "channels": FEATURES,
        "layout": "channel-major: channel c = columns [c*D, (c+1)*D)",
        "interpolation": f"{N_TIMESTEPS} -> {MODEL_CONTEXT} (linear)",
        "model": MODEL_ID,
        "revision": REVISION,
        **info,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nFold completed.")


if __name__ == "__main__":
    main()