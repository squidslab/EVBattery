import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """A lightweight residual block for 1D time-series convolutions."""
    def __init__(self, channels, kernel_size=5):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class CapacityEncoder(nn.Module):
    """Maps (batch, 128, 8) snippets to (batch, embedding_dim)."""
    def __init__(self, input_features=8, embedding_dim=128):
        super(CapacityEncoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            ResidualBlock1D(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            ResidualBlock1D(64),
            # Keep a coarse temporal grid instead of averaging the whole
            # sequence into one value; order can matter for capacity.
            nn.AdaptiveAvgPool1d(8)
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Dataset format: (batch, timesteps, features).
        # Conv1d expects:   (batch, features, timesteps).
        x = x.transpose(1, 2)
        x = self.backbone(x)
        return self.projection(x)


class CapacityEmbeddingRegressor(nn.Module):
    """Jointly trains the encoder and a capacity-regression head."""
    def __init__(self, input_features=8, embedding_dim=128):
        super(CapacityEmbeddingRegressor, self).__init__()
        self.encoder = CapacityEncoder(input_features, embedding_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        capacity = self.regressor(embedding)
        return capacity

    def get_embedding(self, x):
        return self.encoder(x)
