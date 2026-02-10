# config.py
import torch

CFG = {
    "seed": 42,

    # Mac M2 uses MPS; Colab uses CUDA; fallback CPU
    "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),

    # Training (consistent across all models)
    "optimizer": "adam",
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 12,

    # Early stopping (consistent)
    "early_stopping": {
        "enabled": True,
        "monitor": "val_loss",
        "patience": 3,
        "min_delta": 0.0
    },

    # What to run
    "run_datasets": ["Adult", "CIFAR-100(0-9)"],
    "run_architectures": ["MLP", "CNN", "ViT"],

    # Splits
    "adult_test_size": 0.2,
    "adult_val_size": 0.15,
    "val_fraction": 0.15,

    "pcam_train_limit": 10000,
    "pcam_val_limit": 2000,
    "pcam_test_limit": 2000,

    # Architecture 3 configs
    "tabattn": {"d_model": 64, "heads": 4, "layers": 2, "dropout": 0.1},
    "vit": {"patch": 4, "dim": 128, "heads": 4, "layers": 2, "dropout": 0.1},

    # Bonus
    "bonus": {
        "make_learning_curve_comparison": True,
        "save_param_counts": True
    }
}
