# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def criterion_for(task):
    return nn.BCEWithLogitsLoss() if task == "binary" else nn.CrossEntropyLoss()

def metric_from_logits(logits, y, task):
    if task == "binary":
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs >= 0.5).long().cpu().numpy()
        labels = y.view(-1).long().cpu().numpy()
        return f1_score(labels, preds, average="binary")
    else:
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = y.cpu().numpy()
        return accuracy_score(labels, preds)

@torch.no_grad()
def test_metrics(model, test_loader, task, device):
    model.eval()
    all_logits, all_y = [], []
    for x, y in test_loader:
        x = x.to(device)
        all_logits.append(model(x).cpu())
        all_y.append(y.cpu())

    logits = torch.cat(all_logits)
    y = torch.cat(all_y)

    if task == "binary":
        p = (torch.sigmoid(logits.view(-1)) >= 0.5).long().numpy()
        l = y.view(-1).long().numpy()
        return accuracy_score(l, p), f1_score(l, p, average="binary")
    else:
        p = logits.argmax(1).numpy()
        l = y.numpy()
        return accuracy_score(l, p), f1_score(l, p, average="weighted")

def save_curves(history, name, metric_name):
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{name} Loss")
    plt.legend()
    plt.savefig(f"results/{name}_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history["train_metric"], label=f"Train {metric_name}")
    plt.plot(history["val_metric"], label=f"Val {metric_name}")
    plt.title(f"{name} {metric_name}")
    plt.legend()
    plt.savefig(f"results/{name}_{metric_name.lower()}_curve.png")
    plt.close()

def train_eval(model, train_loader, val_loader, task, cfg, device):
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = criterion_for(task)

    es = cfg["early_stopping"]
    best_val = float("inf")
    best_state = None
    bad = 0
    best_epoch = 0

    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0
        tr_logits, tr_y = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)

            if task == "binary":
                loss = crit(logits.view(-1, 1), y.float().view(-1, 1))
            else:
                loss = crit(logits, y.long())

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            tr_logits.append(logits.detach().cpu())
            tr_y.append(y.detach().cpu())

        train_loss = train_loss_sum / len(train_loader)
        train_metric = metric_from_logits(torch.cat(tr_logits), torch.cat(tr_y), task)

        model.eval()
        val_loss_sum = 0.0
        va_logits, va_y = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)

                if task == "binary":
                    loss = crit(logits.view(-1, 1), y.float().view(-1, 1))
                else:
                    loss = crit(logits, y.long())

                val_loss_sum += loss.item()
                va_logits.append(logits.detach().cpu())
                va_y.append(y.detach().cpu())

        val_loss = val_loss_sum / len(val_loader)
        val_metric = metric_from_logits(torch.cat(va_logits), torch.cat(va_y), task)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metric"].append(train_metric)
        history["val_metric"].append(val_metric)

        improved = (best_val - val_loss) > es.get("min_delta", 0.0)
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
            best_epoch = epoch + 1
        else:
            bad += 1
            if es["enabled"] and bad >= es["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_epoch
