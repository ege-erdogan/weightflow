import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import WeightSpaceObject as WSO


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    val_loader: DataLoader | None = None,
    max_epochs: int = 10,
    max_iters: int = 1e99,
    log_weights_iter: int = 1e99,
    num_epochs_to_ignore: int = 0,
    track_test_loss_during_train: bool = False,
    loader_bar: bool = False,
    device: torch.device = None,
):
    """Train a regression/classification model with the option to collect the weights during training."""

    def get_val_loss(n_batches=8):
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for i, (x, y) in enumerate(val_loader):
                if i == n_batches:
                    break
                x, y = x.to(device), y.to(device)
                pred = model(x)
                pred_labels = pred.argmax(1)
                correct += pred_labels.eq(y).sum().item()
                total += y.size(0)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_acc = correct / total
        val_loss /= min(n_batches, len(val_loader))  # per batch

        model.train()
        return val_loss, val_acc

    results: dict = {
        "train_losses": [],
        "val_losses": [],
        "val_accs": [],
        "init_weights": [],
        "weights": [],
        "weights_y": [],
        "variances": [],
    }

    # save init weights as well
    init_weights, init_biases = model.get_weights()
    results["init_weights"].append(
        WSO.from_zipped(zip(init_weights, init_biases, strict=False))
    )

    epoch_pbar = (
        tqdm(range(max_epochs), desc="Epochs", unit="batch")
        if loader_bar
        else range(max_epochs)
    )

    total_iters, total_reached = 0, False
    for epoch in epoch_pbar:
        train_loss = 0.0
        if total_reached:
            break

        model.train()
        loader_iter = train_loader
        for i, batch in enumerate(loader_iter):
            optimizer.zero_grad()
            try:
                x, y = batch
                x, y = x.to(device), y.to(device)
            except:
                x = batch.to(device)
                y = batch.label
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            train_loss += loss.item()

            # log weights annotated with train losses
            # skip the first num_epochs_to_ignore epochs a
            if epoch >= num_epochs_to_ignore and i % log_weights_iter == 0:
                weights, biases = model.get_weights()
                results["weights"].append(
                    WSO.from_zipped(zip(weights, biases, strict=False))
                )
                results["weights_y"].append(loss.item())
                if track_test_loss_during_train and val_loader is not None:
                    val_loss, val_acc = get_val_loss()
                    results["val_losses"].append(val_loss)
                    results["val_accs"].append(val_acc)

            optimizer.step()
            total_iters += 1
            if total_iters == max_iters:
                total_reached = True
                break

        if not track_test_loss_during_train and val_loader is not None:
            val_loss, val_acc = get_val_loss()
            results["val_losses"].append(val_loss)
            results["val_accs"].append(val_acc)
        else:
            val_loss = -1

        train_loss /= len(train_loader)

        results["train_losses"].append(train_loss)

        if loader_bar:
            epoch_pbar.set_postfix({"train_loss": train_loss, "val_loss": val_loss})

    return results
