from itertools import product
from random import sample

import numpy as np
import ot
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from utils.data import Batch, WeightSpaceObject


class FlowEvaluator:
    def __init__(
        self,
        cfm,
        flow_sourceloader,
        flow_targetloader,
        nn_testloader,
        nn_model,
        layers,
        classification=True,
        y_scaler=None,
        noise_scale=0.0,
    ):
        self.cfm = cfm
        self.y_scaler = y_scaler
        self.noise_scale = noise_scale
        self.classification = classification
        self.flow_sourceloader = flow_sourceloader
        self.flow_targetloader = flow_targetloader
        self.nn_testloader = nn_testloader
        self.nn_model = nn_model
        self.layers = np.array(layers)
        self.source_losses, self.source_accs, self.target_losses, self.target_accs = (
            [],
            [],
            [],
            [],
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def convert_mapped_to_wsos(self, mapped, batch_size):
        """Convert a list of Batches to a list os WSOs."""
        mapped = [
            Batch.deflatten(m, batch_size, np.array(self.cfm.layer_layout), self.device)
            for m in mapped
        ]
        mapped_wsos = []
        for batch in mapped:
            w = WeightSpaceObject.from_batch(batch, self.layers)
            mapped_wsos += w
        return mapped_wsos

    def map(
        self, n_steps, n_batches=8, return_traj=False, loader=True
    ) -> list[WeightSpaceObject]:
        """Map `n_batches` batches from the sourceloader with the flow."""
        mapped = []
        iterator = (
            tqdm(self.flow_sourceloader, desc=f"Mapping {n_batches} batches")
            if loader
            else self.flow_sourceloader
        )
        for i, x in enumerate(iterator):
            if i == n_batches:
                break
            src = x.flatten(keep_n_batch=True)
            batch_size = src.size(0)
            pred = self.cfm.map(
                src,
                n_steps=n_steps,
                return_traj=return_traj,
                noise_scale=self.noise_scale,
            )
            if return_traj:
                wso_traj = self.convert_mapped_to_wsos(pred, batch_size)
                mapped += wso_traj
            else:
                mapped.append(pred)

        if return_traj:
            trajectories = [[] for _ in range(batch_size * n_batches)]
            for i, wso in enumerate(mapped):
                trajectories[i % (batch_size * n_batches)].append(wso)
            return trajectories
        else:
            return self.convert_mapped_to_wsos(mapped, batch_size)

    def get_losses_accs(self, mapped_wsos, n=1e99, loader=True):
        losses, accs = [], []
        iterator = (
            tqdm(mapped_wsos[:n], desc="eval mapped") if loader else mapped_wsos[:n]
        )
        for wso in iterator:
            loss, acc = get_loss_acc(
                wso,
                self.nn_model,
                self.nn_testloader,
                classification=self.classification,
                y_scaler=self.y_scaler,
            )
            losses.append(loss)
            accs.append(acc)
        if len(self.source_losses) == 0:
            source_wsos = sample(self.flow_sourceloader.dataset.objects, n)
            target_wsos = sample(self.flow_targetloader.dataset.objects, n)
            src_iterator = (
                tqdm(
                    zip(source_wsos, target_wsos, strict=False), desc="eval src/target"
                )
                if loader
                else zip(source_wsos, target_wsos, strict=False)
            )
            for source_wso, target_wso in src_iterator:
                source_loss, source_acc = get_loss_acc(
                    source_wso,
                    self.nn_model,
                    self.nn_testloader,
                    classification=self.classification,
                    y_scaler=self.y_scaler,
                )
                target_loss, target_acc = get_loss_acc(
                    target_wso,
                    self.nn_model,
                    self.nn_testloader,
                    classification=self.classification,
                    y_scaler=self.y_scaler,
                )
                self.source_losses.append(source_loss)
                self.source_accs.append(source_acc)
                self.target_losses.append(target_loss)
                self.target_accs.append(target_acc)
        return losses, accs

    def get_distances(self, mapped_wsos, n=32):
        results = dict()
        source_wsos = sample(self.flow_sourceloader.dataset.objects, n)
        target_wsos = sample(self.flow_targetloader.dataset.objects, n)
        mapped = mapped_wsos[:n]
        results["st"] = [
            F.mse_loss(s.flatten(), t.flatten()).item()
            for s, t in product(source_wsos, target_wsos)
        ]
        results["sm"] = [
            F.mse_loss(s.flatten(), t.flatten()).item()
            for s, t in product(source_wsos, mapped)
        ]
        results["tm"] = [
            F.mse_loss(s.flatten(), t.flatten()).item()
            for s, t in product(target_wsos, mapped)
        ]
        results["ss"] = [
            F.mse_loss(s.flatten(), t.flatten()).item()
            for s, t in product(source_wsos, source_wsos)
        ]
        results["tt"] = [
            F.mse_loss(s.flatten(), t.flatten()).item()
            for s, t in product(target_wsos, target_wsos)
        ]
        results["mm"] = [
            F.mse_loss(s.flatten(), t.flatten()).item()
            for s, t in product(mapped, mapped)
        ]
        return results


def get_wandb_runs_df(project_name, created_after=None):
    api = wandb.Api()

    ds = []
    runs = api.runs(f"{project_name}", per_page=10_000)
    runs = [run for run in runs if run.state == "finished"]
    if created_after is not None:
        runs = [run for run in runs if run.created_at >= created_after]

    for run in runs:
        run_dict = dict()
        run_dict["name"] = run.name

        run_dict.update(run.summary._json_dict)

        config_dict = {k: v for k, v in run.config.items() if not k.startswith("_")}
        run_dict.update(config_dict)

        ds.append(run_dict)

    return pd.DataFrame(ds)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    pred_fn,
    loss_fn,
    loader_bar=False,
    y_scaler=None,
    n_batches=1e99,
) -> dict:
    """
    Evaluate a classification model on a dataset.

    Args:
        model: The model to evaluate.
        loader: A DataLoader providing the data.
        device: The device on which to run the model.

    Returns:
        A dictionary containing the average loss and accuracy, as well as the predicted and ground-truth labels.

    """
    model.eval()
    loss, correct, n_samples = 0.0, 0.0, 0.0
    predicted, gt = [], []
    pbar = tqdm(loader, desc="Eval", leave=False) if loader_bar else loader
    n_batches = len(loader)
    for i, (x, y) in enumerate(pbar):
        if i == n_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        if y_scaler is not None:
            out = torch.from_numpy(y_scaler.inverse_transform(out.detach().cpu()))
            y = torch.from_numpy(y_scaler.inverse_transform(y.detach().cpu()))
        loss += loss_fn(out, y)
        n_samples += y.size(0)
        pred = pred_fn(out)
        correct += pred.eq(y).sum()
        predicted.append((x.cpu().numpy(), pred.cpu().numpy()))
        gt.extend(y.cpu().numpy().tolist())

    model.train()
    avg_loss = loss / n_batches
    avg_acc = correct / n_samples

    return dict(loss=avg_loss, accuracy=avg_acc, predicted=predicted, gt=gt)


def eval_classification(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loader_bar=False,
    n_batches=1e99,
) -> dict:
    return evaluate(
        model,
        loader,
        device,
        pred_fn=lambda x: x.argmax(1),
        loss_fn=F.cross_entropy,
        loader_bar=loader_bar,
        n_batches=n_batches,
    )


def eval_regression(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn=F.mse_loss,
    loader_bar=False,
    y_scaler=None,
    n_batches=1e99,
) -> dict:
    return evaluate(
        model,
        loader,
        device,
        pred_fn=lambda x: x,
        loss_fn=loss_fn,
        loader_bar=loader_bar,
        n_batches=n_batches,
        y_scaler=y_scaler,
    )


def get_loss(
    wso: WeightSpaceObject,
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    n_batches=1e99,
    device=None,
    y_scaler=None,
    classification=True,
) -> float:
    wso.apply_to(model)
    if classification:
        eval_results = eval_classification(
            model, testloader, device, n_batches=n_batches
        )
    else:
        eval_results = eval_regression(
            model, testloader, device, n_batches=n_batches, y_scaler=y_scaler
        )
    return eval_results["loss"]


@torch.no_grad()
def get_loss_acc(
    wso, model, testloader, classification=True, y_scaler=None, n_batches=1e99
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wso.apply_to(model)
    model = model.to(device)
    if classification:
        eval_results = eval_classification(
            model, testloader, device, n_batches=n_batches
        )
    else:
        eval_results = eval_regression(model, testloader, device, y_scaler=y_scaler)
    return (
        eval_results["loss"].item(),
        eval_results["accuracy"].item(),
    )


@torch.no_grad()
def get_interpolation_losses(
    wso_a,
    wso_b,
    testloader,
    model,
    device,
    n_points=25,
    n_batches=10,
    normalize=False,
):
    losses = []
    ts = np.linspace(0, 1, n_points)
    for t in ts:
        wso = wso_a * (1 - t) + wso_b * t
        if normalize:
            wso.normalize()
        loss = get_loss(wso, model, testloader, n_batches, device)
        losses.append(loss.detach().cpu())
    return losses


def wasserstein_distance(wsos_a, wsos_b):
    assert len(wsos_a) == len(wsos_b)
    n_samples = len(wsos_a)
    X = torch.stack([w.flatten() for w in wsos_a]).numpy()
    Y = torch.stack([w.flatten() for w in wsos_b]).numpy()
    cost_matrix = ot.dist(X, Y, metric="euclidean")
    a = np.ones((n_samples,)) / n_samples
    b = np.ones((n_samples,)) / n_samples
    distance = ot.emd2(a, b, cost_matrix)
    return distance


@torch.no_grad()
def get_pairwise_mid_loss(
    wsos,
    model,
    testloader,
    device,
    n_batches=10,
    relative=False,
    normalize=False,
):
    n = len(wsos)
    losses = np.zeros((n, n))
    for i, wso_a in enumerate(tqdm(wsos)):
        for j, wso_b in enumerate(wsos[i:]):
            mid_wso = (wso_a + wso_b) * 0.5
            if normalize:
                mid_wso.normalize()
            loss = get_loss(
                mid_wso, model, testloader, n_batches=n_batches, device=device
            )
            if relative:  # computes the barrier and not the absolute loss value
                loss_i = get_loss(
                    wso_a, model, testloader, n_batches=n_batches, device=device
                )
                loss_j = get_loss(
                    wso_b, model, testloader, n_batches=n_batches, device=device
                )
                loss -= (loss_i + loss_j) / 2
            losses[i, j + i] = losses[j + i, i] = loss
    return losses


def model_average(x, model, wsos, weights=None):
    preds = 0
    if weights is None:
        weights = torch.ones(len(wsos))
    n = len(wsos)
    for i, wso in enumerate(wsos):
        wso.apply_to(model)
        preds += weights[i] * model(x)
    avg = preds / weights.sum()
    return avg
