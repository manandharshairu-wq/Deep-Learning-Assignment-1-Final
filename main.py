 # main.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CFG
from train import set_seed, train_eval, test_metrics, count_params, save_curves

from data_loaders.loaders import get_adult, get_cifar100_10class, get_pcam
from models.model1_mlp import MLP
from models.model2_cnn import CNN
from models.model3_vit import TabularAttention, TinyViT

DEVICE = CFG["device"]


def make_loaders(train_ds, val_ds, test_ds):
    # num_workers=0 is safest on Mac/MPS
    return (
        DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0),
    )


def load_dataset(name):
    if name == "Adult":
        return get_adult(CFG["seed"], CFG["adult_test_size"], CFG["adult_val_size"])
    if name == "CIFAR-100(0-9)":
        return get_cifar100_10class(CFG["val_fraction"], CFG["seed"])
    # default fallback: PCam
    return get_pcam(CFG["pcam_train_limit"], CFG["pcam_val_limit"], CFG["pcam_test_limit"])


def build_model(arch, dataset, in_shape, out_dim):
    if arch == "MLP":
        return MLP(in_shape, out_dim)

    if arch == "CNN":
        return CNN(in_shape, out_dim)

    if arch == "ViT":
        if dataset == "Adult":
            t = CFG["tabattn"]
            return TabularAttention(
                num_features=in_shape,
                out_dim=out_dim,
                d_model=t["d_model"],
                n_heads=t["heads"],
                n_layers=t["layers"],
                dropout=t["dropout"],
            )
        else:
            v = CFG["vit"]
            return TinyViT(
                in_shape,
                out_dim,
                patch=v["patch"],
                d_model=v["dim"],
                n_heads=v["heads"],
                n_layers=v["layers"],
                dropout=v["dropout"],
            )

    raise ValueError(f"Unknown architecture: {arch}")


def safe_name(s: str) -> str:
    # filesystem-safe name for png files
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def main():
    os.makedirs("results", exist_ok=True)

    set_seed(CFG["seed"])
    print("Using device:", DEVICE)

    results = []

    for dataset in CFG["run_datasets"]:
        train_ds, val_ds, test_ds, in_shape, out_dim, task = load_dataset(dataset)
        train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

        for arch in CFG["run_architectures"]:
            print(f"Running: {dataset} + {arch}")

            model = build_model(arch, dataset, in_shape, out_dim).to(DEVICE)

            history, best_epoch = train_eval(
                model, train_loader, val_loader, task, CFG, DEVICE
            )

            # ---- SAVE GRAPHS HERE ----
            if CFG.get("bonus", {}).get("make_learning_curve_comparison", False):
                metric_name = "F1" if task == "binary" else "Accuracy"
                run_id = safe_name(f"{dataset}_{arch}")
                save_curves(history, run_id, metric_name)

            acc, f1 = test_metrics(model, test_loader, task, DEVICE)

            shown_arch = "TabularAttention" if (arch == "ViT" and dataset == "Adult") else arch

            results.append({
                "Dataset": dataset,
                "Architecture": shown_arch,
                "Accuracy": round(acc, 4),
                "F1": round(f1, 4),
                "Params": count_params(model),
                "BestEpoch": best_epoch
            })

    df = pd.DataFrame(results)
    df.to_csv("results/final_metrics.csv", index=False)

    print("\nFinal Results:\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

