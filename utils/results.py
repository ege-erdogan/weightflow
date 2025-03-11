# functions to make scripts to get results shorter and simpler by abstracting away experimental details
from random import sample

import numpy as np
import torch
import wandb
from tqdm import tqdm

from flow.flow_matching import CFM
from nn.gnn import GNNForClassification
from nn.graph_constructor import GraphConstructor
from nn.mlp import MLP
from nn.relational_transformer import RelationalTransformer
from utils.data import get_1d_regression_dataset, load_mnist
from utils.train import train


def get_run_names(project_name: str, created_after: str):
    names = []
    api = wandb.Api()
    key_runs = api.runs(project_name, per_page=10000)
    names = [run.name for run in key_runs if run.created_at >= created_after]
    return names


def get_run_metadata(project_name: str, run_name: str):
    run_dict = dict()
    api = wandb.Api()
    key_runs = api.runs(project_name, per_page=10000)
    runs = [run for run in key_runs if run.name == run_name]
    if len(runs) == 0:
        raise "Run not found"
    else:
        run = runs[0]
        run_dict.update(run.summary._json_dict)
        run_dict.update(run.config)
        return run_dict


def get_cfm_from_wandb(
    name: str,
    batch_size: int,
    layer_layout: list[int],
    sourceloader: torch.utils.data.DataLoader,
    weightloader: torch.utils.data.DataLoader,
    gnn_backbone: str = "transformer",
    n_layers: int = 3,
    d_embed: int = 32,
    d_feat: int = 1,
    in_proj_layers: int = 3,
    n_heads: int = 2,
    fm_type: str = "vanilla",
    fm_objective: str = "target",
    geometric: bool = False,
    normalize_pred: bool = False,
    pos_embed_per_node=False,
    scale_loss=False,
    device=None,
    save_path: str = "../data/models",
) -> CFM:
    flow_graph_constructor = GraphConstructor(
        d_in=d_feat,
        d_edge_in=d_feat,
        d_node=d_embed,
        d_edge=d_embed,
        layer_layout=layer_layout,
        use_pos_embed=True,
        in_proj_layers=3,
        num_probe_features=0,
        rev_edge_features=False,
        stats=None,
        pos_embed_per_node=pos_embed_per_node,
    )

    extended_layout = [0] + layer_layout
    deg = torch.zeros(max(extended_layout) + 1, dtype=torch.long)
    for li in range(len(extended_layout) - 1):
        deg[extended_layout[li]] += extended_layout[li + 1]

    if gnn_backbone == "PNA":
        flow_model = GNNForClassification(
            d_hid=d_embed + 1,  # + 1 for time
            d_out=1,  # set d_out = 1 for regression
            n_batch=batch_size,
            graph_constructor=flow_graph_constructor,
            backbone="PNA",
            deg=deg,
            n_layers=n_layers,
            device=device,
            layer_layout=layer_layout,
            rev_edge_features=False,
            pooling_method="cat",
            pooling_layer_idx="last",
            task_level="edge_node",
        ).to(device)
        flow_model.eval()
    elif gnn_backbone == "transformer":
        flow_model = RelationalTransformer(
            d_node=d_embed + 1,
            d_edge=d_embed + 1,
            d_attn_hid=d_embed,
            d_node_hid=d_embed,
            d_edge_hid=d_embed,
            d_out_hid=d_embed,
            d_out=1,
            n_layers=n_layers,
            n_heads=n_heads,
            n_batch=batch_size,
            layer_layout=layer_layout,
            graph_constructor=flow_graph_constructor,
            dropout=0.0,
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
        )
        flow_model.eval()

    cfm = CFM(
        sourceloader=sourceloader,
        targetloader=weightloader,
        layer_layout=np.array(layer_layout),
        model=flow_model,
        fm_type=fm_type,
        mode=fm_objective,
        device=device,
        geometric=geometric,
        normalize_pred=normalize_pred,
    )

    cfm.load_model(f"{save_path}/{name}.pt")
    return cfm, flow_model


def get_taskloaders(
    task,
    n_train=2_000,
    n_test=200,
    batch_size=64,
    output_noise_scale=0.05,
    key="mnist",
    a=4,
    b=4.3,
):
    if task == "MNIST":
        trainloader, testloader, _ = load_mnist(batch_size=batch_size, key=key)
    else:
        trainloader, testloader, _ = get_1d_regression_dataset(
            n_train=n_train,
            n_test=n_test,
            batch_size=batch_size,
            output_noise_scale=output_noise_scale,
            a=a,
            b=b,
        )
    return trainloader, testloader


def get_sgd_traj_losses(
    layers,
    trainloader,
    testloader,
    model,
    opts=["adam", "sgd"],
    reps=4,
    device=None,
    epochs=25,
    activation="relu",
    init_wsos=None,
    loss_fn=torch.nn.MSELoss(),
):
    train_losses = dict()
    train_accs = dict()
    for opt in opts:
        sgd_accs, sgd_losses = [], []
        for rep in tqdm(range(reps)):
            model = MLP(
                input_size=layers[0],
                hidden_sizes=layers[1:-1],
                output_size=layers[-1],
                activation=activation,
                weight_init="normal",
            ).to(device)
            if init_wsos is not None:
                wso = sample(init_wsos, 1)[0]
                wso.apply_to(model)

            if opt == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            elif opt == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            else:
                optimizer = None

            train_results = train(
                model,
                trainloader,
                optimizer,
                loss_fn,
                val_loader=testloader,
                track_test_loss_during_train=True,
                max_epochs=epochs,
                log_weights_iter=5,
                num_epochs_to_ignore=0,
                device=device,
                loader_bar=True,
            )

            losses = train_results["val_losses"]
            accs = train_results["val_accs"]
            sgd_losses.append(losses)
            sgd_accs.append(accs)
        train_losses[opt] = sgd_losses
        train_accs[opt] = sgd_accs
    return train_losses, train_accs
