import sys

# sys.path.insert(1, "/dss/dsshome1/0E/di38seq/weight-generation")
sys.path.insert(1, "..")
from pathlib import Path

import numpy as np
import torch
import wandb

import flow.flow_matching as fm
from nn.gnn import GNNForClassification
from nn.graph_constructor import GraphConstructor
from nn.relational_transformer import RelationalTransformer
from utils.data import (
    WeightDataset,
    avg_wsos,
    get_uci_loaders,
    load_weights,
    sample_gaussian_wsos,
)
from utils.parser import parse_args, set_logger

args = parse_args()

logger = set_logger(args.log)
logger.debug(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on {device}")

for rep in range(args.reps):
    if args.wandb is not None:
        wandb.login()
        wandb.init(
            project=args.wandb,
            config=vars(args),
            dir=Path(".").resolve(),
            reinit=True,
        )
        config = wandb.config

    task = "MNIST" if "mnist" in args.data_path else "REGRESSION"

    input_dim, output_dim = args.input_dim, args.output_dim

    if task != "MNIST":
        if "uci" in args.data_path:
            uci_id = int(args.data_path.split("_")[-2])
            _, _, _, input_dim, output_dim, _ = get_uci_loaders(id=uci_id)
            logger.warning(
                f"Task is regression. Inferred input dim {input_dim} from UCI dataset id {uci_id}."
            )
        else:
            input_dim = 1
            logger.warning("Task is regression. Setting input dim to 1 since not UCI.")

    weightloader, layer_layout = load_weights(
        filepath=args.data_path,
        batch_size=args.batch_size,
        layers=[input_dim] + args.hidden_dims + [output_dim],
        n_samples=args.n_samples,
    )
    target_weights = weightloader.dataset.objects
    n = len(target_weights)
    logger.info(layer_layout)

    mean_target_wso = avg_wsos(target_weights) * args.source_mean_multiply
    source_weights = sample_gaussian_wsos(mean_target_wso, std=args.source_std, n=n)

    if args.normalize:
        for wso in target_weights + source_weights:
            wso.normalize()
        target_dataset = WeightDataset(target_weights, labels=None)
        weightloader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        logger.info("Normalized NN layers")

    source_dataset = WeightDataset(source_weights, labels=None)
    sourceloader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    d_feat = args.d_feat  # weights/biases one per edge/node
    d_embed = args.d_embed

    graph_constructor = GraphConstructor(
        d_in=d_feat,
        d_edge_in=d_feat,
        d_node=d_embed,
        d_edge=d_embed,
        layer_layout=layer_layout,
        use_pos_embed=True,
        in_proj_layers=args.in_proj_layers,
        num_probe_features=0,
        rev_edge_features=False,
        stats=None,
        pos_embed_per_node=args.pos_embed_per_node,
    )

    extended_layout = [0] + layer_layout
    deg = torch.zeros(max(extended_layout) + 1, dtype=torch.long)
    for li in range(len(extended_layout) - 1):
        deg[extended_layout[li]] += extended_layout[li + 1]

    if args.gnn_backbone == "transformer":
        flow_model = RelationalTransformer(
            d_node=d_embed + 1,
            d_edge=d_embed + 1,
            d_attn_hid=d_embed,
            d_node_hid=d_embed,
            d_edge_hid=d_embed,
            d_out_hid=d_embed,
            d_out=1,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_batch=args.batch_size,
            layer_layout=layer_layout,
            graph_constructor=graph_constructor,
            dropout=args.dropout,
            node_update_type="rt",
            use_cls_token=False,
            pooling_method="cat",
            pooling_layer_idx="last",
            rev_edge_features=False,
            modulate_v=True,
            use_ln=True,
            tfixit_init=False,
            task_level="edge_node",
            device=device,
            refine=args.refine,
        )
    else:
        flow_model = GNNForClassification(
            d_hid=d_embed + 1,  # + 1 for time
            d_out=1,  # set d_out = 1 for regression
            n_batch=args.batch_size,
            graph_constructor=graph_constructor,
            backbone=args.gnn_backbone,
            deg=deg,
            n_layers=args.n_layers,
            device=device,
            layer_layout=layer_layout,
            rev_edge_features=False,
            pooling_method="cat",
            pooling_layer_idx="last",
            task_level="edge_node",
        ).to(device)

    flow_model.train()

    optimizer = torch.optim.Adam(flow_model.parameters(), lr=args.learning_rate)

    n_params = count_parameters(flow_model)

    cfm = fm.CFM(
        sourceloader=sourceloader,
        targetloader=weightloader,
        layer_layout=np.array(layer_layout),
        model=flow_model,
        fm_type=args.fm_type,
        mode=args.fm_objective,
        device=device,
        normalize_pred=args.normalize_pred,
        geometric=args.geometric,
        t_dist=args.t_dist,
    )

    cfm.train(
        args.n_iters,
        optimizer,
        sigma=args.sigma,
        patience=args.patience,
        log_freq=2,
        wandb=wandb,
    )

    if args.save:
        model_name = wandb.run.name
        cfm.save_model(f"{args.save_path}/{model_name}.pt")

    if args.wandb is not None:
        wandb.log({"n_params": n_params})
        wandb.finish()
