import argparse
import json
import os
import sys
import torch
from capacity_dataset import CapacityDataset
from utils import to_var, collate, Normalizer, PreprocessNormalizer
from model import LSTMNet, MLP, GatedCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import IncrementalPCA
from utils import build_loc_net, get_fc_graph_struc
import gc

# NOTE: The following instructions are added as thesis' work.
# ================================================================
from pathlib import Path

FEATURES = [
    "average_cell_voltage",
    "charging_current",
    "state_of_charge",
    "maximum_cell_voltage",
    "minimum_cell_voltage",
    "maximum_temperature",
    "minimum_temperature",
    "timestamp"
]

def save_results(fold_num, model_name, sampling, results, selected_features=[], pca=False):
    out_dir = Path("results") / model_name / ("pca" if pca else "no_pca") / f"sampling-{sampling}" / f"feature_sel_{len(selected_features)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename =  out_dir / f"results_fold_{fold_num}_{model_name}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Fold: {fold_num}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Sampling: {sampling}%\n")
        if len(selected_features) < len(FEATURES):
            f.write(f"Feature Selection:\n")
            for i, feature in enumerate(selected_features):
                f.write(f"  Feature {i}: {FEATURES[feature]}\n")
        f.write("=" * 40 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Results saved to {filename}")

class PCATransformDataset(torch.utils.data.Dataset):
    """
    Apply a pre-fitted PCA to each timestep of a snippet.

    Input per snippet:  (128, 8)
    Output per snippet: (128, n_components)
    """

    def __init__(self, dataset, pca):
        self.dataset = dataset
        self.pca = pca

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, metadata = self.dataset[index]

        # PCA wants 2D array of shape (n_samples, n_features)
        # rows = timestep, columns = feature
        data = np.asarray(data, dtype=np.float32)

        transformed_data = self.pca.transform(data)

        return transformed_data.astype(np.float32), metadata

def fit_incremental_pca(
    dataset,
    n_components,
    snippets_per_batch=256
):
    """
    IncrementalPCA is used to estimate PCA using all snippets in the dataset.

    Each snippet has shape (128, 8).
    PCA is estimated on the rows, i.e., on the timesteps.

    snippets_per_batch limits the amount of data loaded
    at once during the fit.
    """

    pca = IncrementalPCA(
        n_components=n_components,
        batch_size=128 * snippets_per_batch
    )

    n_snippets = len(dataset)

    for start in range(0, n_snippets, snippets_per_batch):
        end = min(start + snippets_per_batch, n_snippets)

        batch_snippets = []

        for index in range(start, end):
            data, _ = dataset[index]

            data = np.asarray(data, dtype=np.float32)
            batch_snippets.append(data)

        # (numero_snippet_batch * 128, 8)
        batch_data = np.concatenate(batch_snippets, axis=0)

        pca.partial_fit(batch_data)

        del batch_snippets
        del batch_data
        gc.collect()

        print(
            f"PCA fit: {end}/{n_snippets} snippets processed",
            end="\r"
        )

    print()

    explained_variance = np.sum(pca.explained_variance_ratio_)

    print(f"Number of PCA components: {n_components}")
    print(
        "Cumulative explained variance: "
        f"{explained_variance:.4%}"
    )

    return pca

# ================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch capacity estimation')
    parser.add_argument('--fold_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='LSTMNet')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--sampling', type=int, default=100)
    parser.add_argument('--feature_selection', type=bool, default=False)
    parser.add_argument('--pca', type=bool, default=True)
    parser.add_argument('--pca_components', type=int, default=3)
    parser.add_argument('--pca_batch_snippets', type=int, default=256)
    parser.add_argument('--load_saved_dataset', action='store_true')


    args = parser.parse_args()

    print("args", args)

    # load dataset
    # if args.load_saved_dataset:
    #     with open(f'saved_dataset/train_dataset_fold_{args.fold_num}.pkl', 'rb') as f:
    #         train_dataset = pickle.load(f)
    #     with open(f'saved_dataset/test_dataset_{args.fold_num}.pkl', 'rb') as f:
    #         test_dataset = pickle.load(f)
    # else:
    #     train_pre = CapacityDataset(train=True, fold_num=args.fold_num, sampling=args.sampling)
    #     test_pre = CapacityDataset(train=False, fold_num=args.fold_num)

    #     normalizer = Normalizer(dfs=[train_pre[i][0] for i in range(200)],
    #                                  variable_length=False)
    #     train_dataset = PreprocessNormalizer(train_pre, normalizer_fn=normalizer.norm_func)
    #     test_dataset = PreprocessNormalizer(test_pre, normalizer_fn=normalizer.norm_func)
    #     os.makedirs("saved_dataset", exist_ok=True)
    #     with open(f'saved_dataset/train_dataset_fold_{args.fold_num}.pkl', 'wb') as f:
    #         pickle.dump(train_dataset, f)
    #     with open(f'saved_dataset/test_dataset_{args.fold_num}.pkl', 'wb') as f:
    #         pickle.dump(test_dataset, f)

    #     if args.feature_selection:
    #         selected_features = [0, 1, 2, 4, 6]
    #     else:
    #         selected_features = list(range(len(FEATURES)))

    # ================================================================
    # Load dataset + optional PCA
    # ================================================================

    selected_features = list(range(len(FEATURES)))

    if args.feature_selection:
        selected_features = [0, 1, 2, 4, 6]

    if args.pca and args.feature_selection:
        raise ValueError(
            "Per questo esperimento PCA e feature selection "
            "devono essere usate separatamente."
        )

    if args.load_saved_dataset:
        with open(
            f"saved_dataset/train_dataset_fold_{args.fold_num}.pkl",
            "rb"
        ) as f:
            train_dataset = pickle.load(f)

        with open(
            f"saved_dataset/test_dataset_{args.fold_num}.pkl",
            "rb"
        ) as f:
            test_dataset = pickle.load(f)

    else:
        # Full training fold dataset, used to fit PCA before sampling.
        train_full_pre = CapacityDataset(
            train=True,
            fold_num=args.fold_num,
            sampling=100
        )

        # Training dataset, possibly sampled.
        train_pre = CapacityDataset(
            train=True,
            fold_num=args.fold_num,
            sampling=args.sampling
        )

        # Test dataset, always the full test fold.
        test_pre = CapacityDataset(
            train=False,
            fold_num=args.fold_num
        )

        # Normalizer fitted on the full training fold dataset.
        normalizer = Normalizer(
            dfs=[
                train_full_pre[i][0]
                for i in range(min(200, len(train_full_pre)))
            ],
            variable_length=False
        )

        train_full_normalized = PreprocessNormalizer(
            train_full_pre,
            normalizer_fn=normalizer.norm_func
        )

        train_dataset = PreprocessNormalizer(
            train_pre,
            normalizer_fn=normalizer.norm_func
        )

        test_dataset = PreprocessNormalizer(
            test_pre,
            normalizer_fn=normalizer.norm_func
        )

        if args.pca:
            pca = fit_incremental_pca(
                dataset=train_full_normalized,
                n_components=args.pca_components,
                snippets_per_batch=args.pca_batch_snippets
            )

            # Apply PCA to the training and test datasets.
            train_dataset = PCATransformDataset(
                train_dataset,
                pca
            )

            test_dataset = PCATransformDataset(
                test_dataset,
                pca
            )

            # Select the first n PCA components as features.
            selected_features = list(
                range(args.pca_components)
            )

            print(
                "PCA applicata. "
                f"Input shape: (128, {args.pca_components})"
            )

        del train_full_pre
        del train_full_normalized
        gc.collect()

        if not args.pca:
            os.makedirs("saved_dataset", exist_ok=True)

            with open(
                f"saved_dataset/train_dataset_fold_{args.fold_num}.pkl",
                "wb"
            ) as f:
                pickle.dump(train_dataset, f)

            with open(
                f"saved_dataset/test_dataset_{args.fold_num}.pkl",
                "wb"
            ) as f:
                pickle.dump(test_dataset, f)

    # LSTM
    if args.model == "LSTMNet":
        model = LSTMNet(input_dim=len(selected_features), hidden_dim=32, output_dim=1) # .cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        final_train_loss = None
        final_test_loss = None

        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0][:, :, selected_features].float())
                capacity = to_var(batch_data[1]['capacity'].float())
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss+=loss.item() * data.shape[0]

            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss/len(train_dataset)))
            final_train_loss = epoch_loss / len(train_dataset)

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0][:, :, selected_features].float())
                    capacity = to_var(batch_data[1]['capacity'].float())
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    test_loss+=loss.item() * data.shape[0]
            print(
                'Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, args.num_epochs, test_loss/len(test_dataset)))
            final_test_loss = test_loss / len(test_dataset)

        save_results(args.fold_num, args.model, args.sampling, {
            "Final Train Loss": final_train_loss,
            "Final Test Loss": final_test_loss
        }, selected_features, args.pca)

    # elif args.model == "XGBoost":
    #     # XGBoost
    #     train_data_list = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
    #     train_labels_list = [train_dataset.__getitem__(i)[1]['capacity'] for i in range(len(train_dataset))]
    #     train_data = np.array(train_data_list).reshape(len(train_data_list), -1)
    #     train_labels = np.array(train_labels_list)

    #     test_data_list = [test_dataset.__getitem__(i)[0] for i in range(len(test_dataset))]
    #     test_labels_list = [test_dataset.__getitem__(i)[1]['capacity'] for i in range(len(test_dataset))]
    #     test_data = np.array(test_data_list).reshape(len(test_data_list), -1)
    #     test_labels = np.array(test_labels_list)

    #     print(train_data.shape, test_data.shape)

    #     dtrain = xgb.DMatrix(train_data, label=train_labels)
    #     dtest = xgb.DMatrix(test_data, label=test_labels)

    #     params = {
    #         'objective': 'reg:squarederror',
    #         'eta': 0.1,
    #         'max_depth': 4,
    #         'eval_metric': 'rmse'
    #     }

    #     num_rounds = args.num_epochs
    #     model = xgb.train(params, dtrain, num_rounds)

    #     train_predictions = model.predict(dtrain)
    #     test_predictions = model.predict(dtest)

    #     print("Train RMSE:", np.sqrt(np.mean((train_predictions - train_labels) ** 2)))
    #     print("Test RMSE:", np.sqrt(np.mean((test_predictions - test_labels) ** 2)))

    #     save_results(args.fold_num, args.model, {
    #         "Train RMSE": np.sqrt(np.mean((train_predictions - train_labels) ** 2)),
    #         "Test RMSE": np.sqrt(np.mean((test_predictions - test_labels) ** 2))
    #     })

    elif args.model == "XGBoost":
        # XGBoost
        train_data = np.empty((len(train_dataset), 128 * len(selected_features)), dtype=np.float16)
        train_labels = np.empty(len(train_dataset), dtype=np.float32)

        for i in range(len(train_dataset)):
            data, metadata = train_dataset[i]

            train_data[i] = data[:, selected_features].reshape(-1)
            train_labels[i] = metadata['capacity']

        dtrain = xgb.DMatrix(train_data, label=train_labels)

        del train_data
        gc.collect()

        params = {
            'objective': 'reg:squarederror',
            'eta': 0.1,
            'max_depth': 4,
            'eval_metric': 'rmse',
            'tree_method': 'hist', 
            'max_bin': 64, 
            'nthread': 2
        }

        num_rounds = args.num_epochs
        model = xgb.train(params, dtrain, num_rounds)

        train_predictions = model.predict(dtrain)
        train_rmse = np.sqrt(np.mean((train_predictions - train_labels) ** 2))

        del dtrain
        del train_predictions
        del train_labels
        gc.collect()

        test_data = np.empty((len(test_dataset), 128 * len(selected_features)), dtype=np.float16)
        test_labels = np.empty(len(test_dataset), dtype=np.float32)

        for i in range(len(test_dataset)):
            data, metadata = test_dataset[i]

            test_data[i] = data[:, selected_features].reshape(-1)
            test_labels[i] = metadata['capacity']

        dtest = xgb.DMatrix(test_data, label=test_labels)

        del test_data
        gc.collect()

        test_predictions = model.predict(dtest)
        test_rmse = np.sqrt(np.mean((test_predictions - test_labels) ** 2))

        print("Train RMSE:", train_rmse)
        print("Test RMSE:", test_rmse)

        save_results(args.fold_num, args.model, args.sampling, {
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse
        }, selected_features, args.pca)

    elif args.model == "MEAN":
        train_data_list = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
        train_labels_list = [train_dataset.__getitem__(i)[1]['capacity'] for i in range(len(train_dataset))]
        train_data = np.array(train_data_list).reshape(len(train_data_list), -1)
        train_labels = np.array(train_labels_list)

        test_data_list = [test_dataset.__getitem__(i)[0] for i in range(len(test_dataset))]
        test_labels_list = [test_dataset.__getitem__(i)[1]['capacity'] for i in range(len(test_dataset))]
        test_data = np.array(test_data_list).reshape(len(test_data_list), -1)
        test_labels = np.array(test_labels_list)

        print(train_data.shape, test_data.shape)
        test_predictions = np.ones_like(test_labels)*np.mean(train_labels)

        print("Test RMSE:", np.sqrt(np.mean((test_predictions - test_labels) ** 2)))

    # elif args.model == "RandomForest":
    #     train_data_list = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
    #     train_labels_list = [train_dataset.__getitem__(i)[1]['capacity'] for i in range(len(train_dataset))]
    #     train_data = np.array(train_data_list).reshape(len(train_data_list), -1)
    #     train_labels = np.array(train_labels_list)

    #     test_data_list = [test_dataset.__getitem__(i)[0] for i in range(len(test_dataset))]
    #     test_labels_list = [test_dataset.__getitem__(i)[1]['capacity'] for i in range(len(test_dataset))]
    #     test_data = np.array(test_data_list).reshape(len(test_data_list), -1)
    #     test_labels = np.array(test_labels_list)

    #     model = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=10, max_depth=4)

    #     model.fit(train_data, train_labels)

    #     train_predictions = model.predict(train_data)
    #     train_rmse = np.sqrt(mean_squared_error(train_labels, train_predictions))
    #     print("Train RMSE:", train_rmse)

    #     test_predictions = model.predict(test_data)
    #     test_rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))
    #     print("Test RMSE:", test_rmse)

    #     save_results(args.fold_num, args.model, {
    #         "Train RMSE": train_rmse,
    #         "Test RMSE": test_rmse
    #     })

    elif args.model == "RandomForest":
        train_data = np.empty((len(train_dataset), 128 * len(selected_features)), dtype=np.float16)
        train_labels = np.empty(len(train_dataset), dtype=np.float32)

        for i in range(len(train_dataset)):
            data, metadata = train_dataset[i]

            train_data[i] = data[:, selected_features].reshape(-1)
            train_labels[i] = metadata['capacity']

        test_data = np.empty((len(test_dataset), 128 * len(selected_features)), dtype=np.float16)
        test_labels = np.empty(len(test_dataset), dtype=np.float32)

        for i in range(len(test_dataset)):
            data, metadata = test_dataset[i]

            test_data[i] = data[:, selected_features].reshape(-1)
            test_labels[i] = metadata['capacity']

        model = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=10, max_depth=4)

        model.fit(train_data, train_labels)

        train_predictions = model.predict(train_data)
        train_rmse = np.sqrt(mean_squared_error(train_labels, train_predictions))
        print("Train RMSE:", train_rmse)

        test_predictions = model.predict(test_data)
        test_rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))
        print("Test RMSE:", test_rmse)

        save_results(args.fold_num, args.model, args.sampling, {
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse
        }, selected_features, args.pca)

    elif args.model == "MLP":
        model = MLP(input_dim=128*8, hidden_dim=32, output_dim=1) # .cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float())
                capacity = to_var(batch_data[1]['capacity'].float())
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.shape[0]

            print(
                'Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss / len(train_dataset)))

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float())
                    capacity = to_var(batch_data[1]['capacity'].float())
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    test_loss += loss.item() * data.shape[0]
            print(
                'Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, args.num_epochs, test_loss / len(test_dataset)))

    elif args.model == "GatedCNN":
        model = GatedCNN(seq_len=128,
                         n_layers=4,
                         kernel=[5, 5],
                         out_chs=4,
                         res_block_count=2,
                         ans_size=1)
        optimizer = optim.Adam(model.parameters(), lr=0.001) # .cuda()
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float())
                capacity = to_var(batch_data[1]['capacity'].float())
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.shape[0]

            print(
                'Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss / len(train_dataset)))

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float())
                    capacity = to_var(batch_data[1]['capacity'].float())
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    test_loss += loss.item() * data.shape[0]
            print(
                'Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, args.num_epochs, test_loss / len(test_dataset)))

    else:
        raise NotImplementedError


