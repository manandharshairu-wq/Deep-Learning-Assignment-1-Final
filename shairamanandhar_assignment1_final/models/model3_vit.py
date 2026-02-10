# models/model3_vit.py
import torch
import torch.nn as nn

class TabularAttention(nn.Module):
    """
    Model 3 (tabular): Transformer-style attention for Adult.
    Treat each feature as a token.
    """
    def __init__(self, num_features, out_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.feature_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x):
        # x: [B, D]
        x = x.unsqueeze(-1)              # [B, D, 1]
        x = self.feature_embed(x)        # [B, D, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)              # [B, D, d_model]
        x = self.norm(x).mean(dim=1)     # mean pool over tokens
        return self.head(x)

class TinyViT(nn.Module):
    """
    Model 3 (image): simple ViT-style encoder for CIFAR/PCam.
    No pretrained weights.
    """
    def __init__(self, in_shape, out_dim, patch=4, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        C, H, W = in_shape
        assert H % patch == 0 and W % patch == 0, "H and W must be divisible by patch size."
        n_patches = (H // patch) * (W // patch)

        self.patch_embed = nn.Conv2d(C, d_model, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, d_model))
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)          # [B, d, H/p, W/p]
        x = x.flatten(2).transpose(1, 2) # [B, tokens, d]
        B = x.size(0)

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d]
        x = torch.cat([cls, x], dim=1)          # [B, 1+tokens, d]
        x = self.drop(x + self.pos_embed[:, :x.size(1), :])

        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])            # CLS token
        return self.head(cls_out)
