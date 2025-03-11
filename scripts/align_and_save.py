# aligns all weights to a randomly chosen reference and saves the list in the same dir.

import sys

import h5py
from tqdm import tqdm

sys.path.insert(1, "..")

import numpy as np
import torch

from nn.mlp import MLP
from utils.data import WeightSpaceObject, align_to_ref
from utils.parser import parse_args, set_logger

args = parse_args()

logger = set_logger(args.log)
logger.debug(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on {device}")

model = MLP(
    input_size=args.input_dim,
    hidden_sizes=args.hidden_dims,
    output_size=args.output_dim,
    weight_init="normal",
    activation=args.activation,
).to(device)

f = h5py.File(args.data_path, "r")
dset = f["dataset"]
weights = []
layers = [args.input_dim] + args.hidden_dims + [args.output_dim]

iterator = dset if args.n_samples is None else dset[: args.n_samples]
for w in tqdm(iterator, desc="Loading data"):
    weights.append(
        WeightSpaceObject.from_flat(
            w, layers=np.array(layers), device=device, revert=True
        )
    )

ref = weights[256]  # pick a random not so early one

aligned_weights = align_to_ref(
    weights, ref, model, n_iters=args.n_iters, input_dim=args.input_dim
)

aligned_f = h5py.File(args.save_path, "a")
dim = aligned_weights[0].flatten().size(0)
n_samples = len(aligned_weights)

aligned_dset = aligned_f.create_dataset("dataset", (n_samples, dim))
for i, wso in enumerate(aligned_weights):
    aligned_dset[i] = wso.flatten().cpu().numpy()
logger.success(f"Saved {args.save_path}")
