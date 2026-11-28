import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from capacity_embedding_model import CapacityEmbeddingRegressor


def load_datasets(fold_num):
    from capacity_dataset import CapacityDataset
    from utils import Normalizer, PreprocessNormalizer

    train_pre = CapacityDataset(train=True, fold_num=fold_num)
    test_pre = CapacityDataset(train=False, fold_num=fold_num)

    normalizer = Normalizer(dfs=[train_pre[i][0] for i in range(200)], 
                            variable_length=False)
    train_dataset = PreprocessNormalizer(train_pre, normalizer_fn=normalizer.norm_func)
    test_dataset = PreprocessNormalizer(test_pre, normalizer_fn=normalizer.norm_func)
    
    return train_dataset, test_dataset, train_pre.capacities


def get_batch(batch, device):
    # Default PyTorch collate returns (data_tensor, metadata_dict).
    x = batch[0].float().to(device)
    y = batch[1]['capacity'].float().view(-1, 1).to(device)
    return x, y


def evaluate(model, loader, device, target_mean, target_std, criterion=None):
    model.eval()
    squared_error = 0.0
    total = 0
    predictions = []
    targets = []
    loss_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            x, y_ah = get_batch(batch, device)
            y_norm = (y_ah - target_mean) / target_std
            pred_norm = model(x)
            if criterion is not None:
                loss_sum += criterion(pred_norm, y_norm).item() * x.size(0)

            pred_ah = pred_norm * target_std + target_mean
            squared_error += torch.sum((pred_ah - y_ah) ** 2).item()
            total += x.size(0)
            predictions.append(pred_ah.cpu().numpy().reshape(-1))
            targets.append(y_ah.cpu().numpy().reshape(-1))

    rmse = np.sqrt(squared_error / max(total, 1))
    if predictions:
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
    else:
        predictions = np.array([])
        targets = np.array([])
    average_loss = loss_sum / max(total, 1)
    return rmse, average_loss, predictions, targets



def export_embeddings(model, dataset, embeddings_path, labels_path, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting embeddings'):
            x, y = get_batch(batch, device)
            z = model.get_embedding(x)

            embeddings.append(z.cpu().numpy().astype(np.float32))
            labels.append(y.cpu().numpy().reshape(-1).astype(np.float32))

    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
    else:
        embeddings = np.empty((0, 128), dtype=np.float32)
        labels = np.empty((0,), dtype=np.float32)

    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)

    print('Embeddings saved:', embeddings_path, embeddings.shape)
    print('Labels saved:', labels_path, labels.shape)


def main():
    parser = argparse.ArgumentParser(
        description='Train an EVBattery-specific capacity embedding.')
    parser.add_argument('--fold_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='myEmbeddings/dataset1')
    parser.add_argument('--export_embeddings', action='store_true')
    args = parser.parse_args()

    if args.fold_num < 0 or args.fold_num > 4:
        parser.error('--fold_num deve essere compreso tra 0 e 4')
    if args.embedding_dim <= 0:
        parser.error('--embedding_dim deve essere positivo')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Arguments:', args)

    train_dataset, test_dataset, train_capacities = load_datasets(args.fold_num)
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError('Il training set o il test set è vuoto.')

    # Capacity target scaling is fitted on the training fold only.
    train_capacities = np.asarray(train_capacities, dtype=np.float32)
    target_mean_value = float(train_capacities.mean())
    target_std_value = float(train_capacities.std())
    if target_std_value < 1e-8:
        target_std_value = 1.0

    target_mean = torch.tensor(target_mean_value, dtype=torch.float32,
                               device=device)
    target_std = torch.tensor(target_std_value, dtype=torch.float32,
                              device=device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    model = CapacityEmbeddingRegressor(
        input_features=8, embedding_dim=args.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir) / f"fold_{args.fold_num}" 
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = output_dir / 'capacity_encoder.pt'
    results_path = output_dir / 'results.txt'

    final_train_rmse = None
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in tqdm(train_loader, desc='Epoch {}/{}'.format(
                epoch + 1, args.num_epochs)):
            x, y_ah = get_batch(batch, device)
            y_norm = (y_ah - target_mean) / target_std

            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y_norm)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_rmse, train_loss, _, _ = evaluate(
            model, train_loader, device, target_mean, target_std, criterion)
        final_train_rmse = train_rmse

        # Do not inspect the held-out test fold during training. It is evaluated
        # only once, after the fixed number of epochs, to avoid test-driven
        # epoch selection.
        print('Epoch [{}/{}] Train loss: {:.6f} | Train RMSE: {:.4f} Ah'.format(
            epoch + 1, args.num_epochs, train_loss, train_rmse))

    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': args.embedding_dim,
        'input_features': 8,
        'target_mean': target_mean_value,
        'target_std': target_std_value,
        'fold_num': args.fold_num,
        'num_epochs': args.num_epochs
    }, checkpoint_path)

    # Final metrics only. The test fold is not used for gradient updates.
    test_rmse, test_loss, _, _ = evaluate(
        model, test_loader, device, target_mean, target_std, criterion)
    with open(results_path, 'w') as f:
        f.write('Fold: {}\n'.format(args.fold_num))
        f.write('Embedding dimension: {}\n'.format(args.embedding_dim))
        f.write('Epochs: {}\n'.format(args.num_epochs))
        f.write('Train RMSE: {:.6f} Ah\n'.format(final_train_rmse))
        f.write('Test RMSE: {:.6f} Ah\n'.format(test_rmse))
        f.write('Test normalized MSE: {:.6f}\n'.format(test_loss))

    print('Checkpoint saved:', checkpoint_path)
    print('Results saved:', results_path)

    if args.export_embeddings:
        train_embeddings_path = output_dir / 'train_embeddings.npy'
        train_labels_path = output_dir / 'train_labels.npy'

        test_embeddings_path = output_dir / 'test_embeddings.npy'
        test_labels_path = output_dir / 'test_labels.npy'

        export_embeddings(model, train_dataset, train_embeddings_path, train_labels_path, args.batch_size, device)
        export_embeddings(model, test_dataset, test_embeddings_path, test_labels_path, args.batch_size, device)


if __name__ == '__main__':
    main()
