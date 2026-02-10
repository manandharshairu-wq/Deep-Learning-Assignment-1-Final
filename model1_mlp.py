# models/model1_mlp.py
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    """
    Model 1: MLP
    Input -> 2 hidden layers (ReLU) -> Output
    Includes BatchNorm + Dropout
    Works for tabular and image (after flatten).
    """
    def __init__(self, in_shape, out_dim, hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        flat_size = in_shape if isinstance(in_shape, int) else int(np.prod(in_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),

            nn.Linear(hidden2, out_dim)
        )

    def forward(self, x):
        return self.net(x)
