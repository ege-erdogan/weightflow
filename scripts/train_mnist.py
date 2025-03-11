# collect MNIST weights over SGD trajectories

import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn

from nn.mlp import MLP
from utils.data import (
    init_h5_dataset,
    load_mnist,
)
from utils.eval import eval_classification
from utils.parser import parse_args, set_logger
from utils.train import train

args = parse_args()

logger = set_logger(args.log)
logger.debug(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on {device}")

trainloader, testloader, valloader = load_mnist(batch_size=args.batch_size)

logger.info(f"Train: {len(trainloader.dataset)} samples, {len(trainloader)} batches")
logger.info(f"Test: {len(testloader.dataset)} samples, {len(testloader)} batches")
logger.info(f"Val: {len(valloader.dataset)} samples, {len(valloader)} batches")

init_weights = []
for r in range(args.reps):
    logger.info(f"rep: {r}")
    model = MLP(
        input_size=784,
        hidden_sizes=args.hidden_dims,
        output_size=10,
        weight_init="normal",
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    train_results = train(
        model,
        trainloader,
        optimizer,
        criterion,
        val_loader=valloader,
        max_epochs=args.epochs,
        log_weights_iter=args.log_weights_iter,
        num_epochs_to_ignore=args.ignore_epochs,
        loader_bar=True,
        device=device,
    )

    init_weights += train_results["init_weights"]
    weights = train_results["weights"]
    weights_y = train_results["weights_y"]

    if r == 0:
        samples_per_rep = len(weights)
        total_samples = args.reps * (samples_per_rep + (1 if args.save_init else 0))
        dim = weights[0].flatten().size(0)
        dset = init_h5_dataset(args.save_path, total_samples, dim)
        traj_ids = np.arange(args.reps).repeat(samples_per_rep)

    for i, wso in enumerate(weights):
        dset[r * samples_per_rep + i] = wso.flatten().cpu().numpy()

    if args.save_init and r == args.reps - 1:
        for i, wso in enumerate(init_weights):
            dset[samples_per_rep * args.reps + i] = wso.flatten().cpu().numpy()

    eval_results = eval_classification(model, testloader, device)

    logger.info(f'loss: {eval_results["loss"].item()}')
    logger.info(f'accuracy: {eval_results["accuracy"].item()}')
    logger.success(f"Rep {r} weights saved at {args.save_path}")
