import os
import sys

sys.path.insert(1, "..")

import time
from itertools import product

import yaml

from utils.parser import parse_args

args = parse_args()

# TODO set path to the sbatch file and the train script
sbatch_file_path = "path_to_sbatch"
script_path = "path_to_script"

with open(f"setups/{args.setup}.yaml") as f:
    setup = yaml.load(f, Loader=yaml.FullLoader)

configs_dict = {
    "euclidean": {
        "geometric": False,
        "normalize-pred": False,
        "normalize": False,
    },
    "euclidean-sphere": {
        "geometric": False,
        "normalize-pred": True,
        "normalize": True,
    },
    "geometric": {
        "geometric": True,
        "normalize-pred": True,
        "normalize": True,
    },
}

# sbatch
preamble = f"""#!/bin/bash
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time={setup['time']}
 
source ~/.bashrc

mamba activate weightflow
"""

for (
    data_path,
    lr,
    sigma,
    d_embed,
    batch_size,
    smm,
    heads,
    config,
    n_layers,
    fm_type,
    t_scale_loss,
    pos_embed,
    t_dist,
    backbone,
    n_samples,
) in product(
    setup["data_paths"],
    setup["lrs"],
    setup["sigmas"],
    setup["d_embeds"],
    setup["batch_sizes"],
    setup["source_mean_multiples"],
    setup["n_heads"],
    setup["configs"],
    setup["n_layers"],
    setup["fm_types"],
    setup["t_scale_loss"],
    setup["pos_embed_per_node"],
    setup["t_dist"],
    setup["model"],
    setup["n_samples"],
):
    # define command per parameters config
    command = f"srun --pty python {script_path} --data-path {data_path} --learning-rate {lr} --d-embed {d_embed} --in-proj-layers 3 --n-layers {n_layers} --dropout {setup['dropout']} --n-heads {heads} --n-iters {setup['n_iters']} --fm-type {fm_type} --fm-objective {setup['fm_objective']} --sigma {sigma} --gnn-backbone {backbone} --source-std {setup['source_std']} --source-mean-multiply {smm} --t-dist {t_dist}  --batch-size {batch_size} --save --wandb {setup['wandb']} --config {config} --hidden-dims {' '.join(map(str, setup['hidden_dims']))}"
    for k, v in configs_dict[config].items():
        command += f" --{k}" if v else ""
    command += f" --output-dim {10 if 'mnist' in data_path else 1}"
    command += f" --input-dim {784 if 'mnist' in data_path else -1}"
    command += f" --n-samples {n_samples}"
    command += " --aligned" if "aligned" in data_path else ""
    command += " --refine" if setup["refine"] else ""
    command += " --t-scale-loss" if t_scale_loss else ""
    command += " --learn-conditional-flow" if setup["conditional_flow"] else ""
    command += " --pos-embed-per-node" if pos_embed else ""

    # write preamble and command to sbatch file
    # opening with "w" clears the file first
    with open(sbatch_file_path, "w") as f:
        f.write(preamble)
        f.write(command)

    time.sleep(1)

    # submit job
    os.system(f"sbatch {sbatch_file_path}")
