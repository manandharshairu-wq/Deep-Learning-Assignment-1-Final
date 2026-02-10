# models/model2_cnn.py
import torch.nn as nn

class CNN(nn.Module):
    """
    Model 2: CNN
    - Tabular: Conv1D over features
    - Image: Conv2D
    """
    def __init__(self, in_shape, out_dim, dropout=0.2):
        super().__init__()

        if isinstance(in_shape, int):
            # Tabular as 1D signal: [B, D] -> [B, 1, D]
            self.feat = nn.Sequential(
                nn.Unflatten(1, (1, in_shape)),
                nn.Conv1d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten()
            )
            L = in_shape // 4  # pooled twice
            self.head = nn.Sequential(
                nn.Linear(32 * L, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, out_dim)
            )
        else:
            # Image CNN: [B, C, H, W]
            C, H, W = in_shape
            self.feat = nn.Sequential(
                nn.Conv2d(C, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            H2, W2 = H // 4, W // 4
            self.head = nn.Sequential(
                nn.Linear(64 * H2 * W2, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, out_dim)
            )

    def forward(self, x):
        return self.head(self.feat(x))
