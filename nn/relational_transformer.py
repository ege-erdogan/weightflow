import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from nn.gnn import nn_to_edge_index
from nn.pooling import HomogeneousAggregator
from utils.data import Batch


class RelationalTransformer(nn.Module):
    def __init__(
        self,
        d_node,
        d_edge,
        d_attn_hid,
        d_node_hid,
        d_edge_hid,
        d_out_hid,
        d_out,
        n_layers,
        n_heads,
        n_batch,
        layer_layout,
        graph_constructor,
        refine=False,
        dropout=0.0,
        node_update_type="rt",
        disable_edge_updates=False,
        use_cls_token=True,
        pooling_method="cat",
        pooling_layer_idx="last",
        rev_edge_features=False,
        modulate_v=True,
        use_ln=True,
        task_level="edge_node",
        tfixit_init=False,
        device=None,
    ):
        super().__init__()
        assert use_cls_token == (pooling_method == "cls_token")
        self.refine = refine
        self.device = device
        self.pooling_method = pooling_method
        self.pooling_layer_idx = pooling_layer_idx
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
        self.construct_graph = graph_constructor.to(self.device)
        self.n_batch = n_batch
        self.task_level = task_level
        self.d_attn_hid = d_attn_hid
        self.d_node = d_node
        self.d_edge = d_edge

        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(d_node))

        self.layers = nn.ModuleList(
            [
                torch.jit.script(
                    RTLayer(
                        d_node,
                        d_edge,
                        d_attn_hid,
                        d_node_hid,
                        d_edge_hid,
                        n_heads,
                        dropout,
                        node_update_type=node_update_type,
                        disable_edge_updates=False,
                        modulate_v=modulate_v,
                        use_ln=use_ln,
                        tfixit_init=tfixit_init,
                        n_layers=n_layers,
                    ).to(self.device)
                )
                for i in range(n_layers)
            ]
        )

        if pooling_method != "cls_token":
            self.pool = HomogeneousAggregator(
                pooling_method,
                pooling_layer_idx,
                layer_layout,
            )

        num_graph_features = (
            layer_layout[-1] * d_node
            if pooling_method == "cat" and pooling_layer_idx == "last"
            else d_edge if pooling_method in ("mean_edge", "max_edge") else d_node
        )

        self.edge_index = nn_to_edge_index(
            self.nodes_per_layer, self.device, dtype=torch.long
        ).to(self.device)

        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_out_hid),
            nn.ReLU(),
            nn.Linear(d_out_hid, d_out_hid),
            nn.ReLU(),
            nn.Linear(d_out_hid, d_out),
        ).to(self.device)

        self.edge_proj_out = nn.Sequential(
            nn.Linear(d_edge, d_edge_hid),
            nn.ReLU(),
            nn.Linear(d_edge_hid, d_edge_hid),
            nn.ReLU(),
            nn.Linear(d_edge_hid, d_out),
        ).to(self.device)

        self.node_proj_out = nn.Sequential(
            nn.Linear(d_node, d_node_hid),
            nn.ReLU(),
            nn.Linear(d_node_hid, d_node_hid),
            nn.ReLU(),
            nn.Linear(d_node_hid, d_out),
        ).to(self.device)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor | None = None):
        inputs = Batch.deflatten(
            inputs,
            batch_size=self.n_batch,
            layer_layout=np.array(self.nodes_per_layer),
            device=self.device,
        )
        node_features, edge_features = self._incorporate_time(inputs, t)
        # node_features, edge_features = node_features.to(self.device), edge_features.to(self.device)
        n_nodes = node_features.size(1)
        n_input_nodes = self.nodes_per_layer[0]

        refinement_preds = []
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, None)
            if self.refine:
                refinement_preds.append(
                    self._get_final_pred(node_features, edge_features)
                )

        if self.task_level == "edge_node":
            # filter the existing edges and concat with node (biases) predictions
            final_pred = self._get_final_pred(node_features, edge_features)
            if self.refine:
                return refinement_preds, final_pred
            return final_pred
        else:
            raise Exception("Task level not supported")

    def _get_final_pred(self, node_features, edge_features):
        n_input_nodes = self.nodes_per_layer[0]
        edge_pred = self.edge_proj_out(edge_features)
        node_pred = self.node_proj_out(node_features[:, n_input_nodes:, :])
        edge_out = edge_pred[:, self.edge_index[0], self.edge_index[1], :]
        return torch.cat([edge_out, node_pred], dim=1)

    def _incorporate_time(self, inputs, t):
        node_features, edge_features, _ = self.construct_graph(inputs)
        node_features = node_features.to(self.device)
        edge_features = edge_features.to(self.device)

        n_batch = t.size(0)
        n_nodes, n_edges = node_features.size(1), self.edge_index.size(1)
        n_input_nodes = self.nodes_per_layer[0]
        if t is not None:
            times = torch.tensor([x[0] for x in t]).to(self.device)
            edge_times = (
                times.view(n_batch, 1, 1, 1)
                .expand((n_batch, n_nodes, n_nodes, 1))
                .to(self.device)
            )
            node_times = (
                times.view(n_batch, 1, 1).expand((n_batch, n_nodes, 1)).to(self.device)
            )
            edge_features = torch.cat((edge_features, edge_times), dim=-1)
            node_features = torch.cat((node_features, node_times), dim=-1)

            # set input nodes' and nonexistent edges' time embeddings to 1
            node_features[:, :n_input_nodes, -1] = 1
            edge_features[:, self.edge_index[0], self.edge_index[1], -1] = 1
        return node_features, edge_features

    def get_grad_norm(self) -> float:
        total = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total += param_norm.item() ** 2
        total = total**0.5
        return total


class RTLayer(nn.Module):
    def __init__(
        self,
        d_node,
        d_edge,
        d_attn_hid,
        d_node_hid,
        d_edge_hid,
        n_heads,
        dropout,
        node_update_type="rt",
        disable_edge_updates=False,
        modulate_v=True,
        use_ln=True,
        tfixit_init=False,
        n_layers=None,
    ):
        super().__init__()
        self.node_update_type = node_update_type
        self.disable_edge_updates = disable_edge_updates
        self.use_ln = use_ln
        self.n_layers = n_layers

        self.self_attn = torch.jit.script(
            RTAttention(
                d_node,
                d_edge,
                d_attn_hid,
                n_heads,
                modulate_v=modulate_v,
                use_ln=use_ln,
            )
        )
        # self.self_attn = RTAttention(d_hid, d_hid, d_hid, n_heads)
        self.lin0 = Linear(d_node, d_node)
        self.dropout0 = nn.Dropout(dropout)
        if use_ln:
            self.node_ln0 = nn.LayerNorm(d_node)
            self.node_ln1 = nn.LayerNorm(d_node)
        else:
            self.node_ln0 = nn.Identity()
            self.node_ln1 = nn.Identity()

        act_fn = nn.GELU

        self.node_mlp = nn.Sequential(
            Linear(d_node, d_node_hid, bias=False),
            act_fn(),
            Linear(d_node_hid, d_node),
            nn.Dropout(dropout),
        )

        if not self.disable_edge_updates:
            self.edge_updates = EdgeLayer(
                d_node=d_node,
                d_edge=d_edge,
                d_edge_hid=d_edge_hid,
                dropout=dropout,
                act_fn=act_fn,
                use_ln=use_ln,
            )
        else:
            self.edge_updates = NoEdgeLayer()

        if tfixit_init:
            self.fixit_init()

    def fixit_init(self):
        temp_state_dict = self.state_dict()
        n_layers = self.n_layers
        for name, param in self.named_parameters():
            if "weight" in name:
                if name.split(".")[0] in ["node_mlp", "edge_mlp0", "edge_mlp1"]:
                    temp_state_dict[name] = (0.67 * (n_layers) ** (-1.0 / 4.0)) * param
                elif name.split(".")[0] in ["self_attn"]:
                    temp_state_dict[name] = (0.67 * (n_layers) ** (-1.0 / 4.0)) * (
                        param * (2**0.5)
                    )

        self.load_state_dict(temp_state_dict)

    def node_updates(self, node_features, edge_features, mask):
        # attn_out = checkpoint(self.self_attn, node_features, edge_features, mask)
        node_features = self.node_ln0(
            node_features
            + self.dropout0(
                self.lin0(self.self_attn(node_features, edge_features, mask))
            )
        )
        node_features = self.node_ln1(node_features + self.node_mlp(node_features))

        return node_features

    def forward(self, node_features, edge_features, mask):
        node_features = self.node_updates(node_features, edge_features, mask)
        edge_features = self.edge_updates(node_features, edge_features, mask)

        return node_features, edge_features


class EdgeLayer(nn.Module):
    def __init__(
        self,
        *,
        d_node,
        d_edge,
        d_edge_hid,
        dropout,
        act_fn,
        use_ln=True,
    ) -> None:
        super().__init__()
        self.edge_mlp0 = EdgeMLP(
            d_edge=d_edge,
            d_node=d_node,
            d_edge_hid=d_edge_hid,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.edge_mlp1 = nn.Sequential(
            Linear(d_edge, d_edge_hid, bias=False),
            act_fn(),
            Linear(d_edge_hid, d_edge),
            nn.Dropout(dropout),
        )
        if use_ln:
            self.eln0 = nn.LayerNorm(d_edge)
            self.eln1 = nn.LayerNorm(d_edge)
        else:
            self.eln0 = nn.Identity()
            self.eln1 = nn.Identity()

    def forward(self, node_features, edge_features, mask):
        edge_features = self.eln0(
            edge_features + self.edge_mlp0(node_features, edge_features)
        )
        edge_features = self.eln1(edge_features + self.edge_mlp1(edge_features))
        return edge_features


class NoEdgeLayer(nn.Module):
    def forward(self, node_features, edge_features, mask):
        return edge_features


class EdgeMLP(nn.Module):
    def __init__(self, *, d_node, d_edge, d_edge_hid, act_fn, dropout):
        super().__init__()
        self.reverse_edge = Rearrange("b n m d -> b m n d")
        self.lin0_e = Linear(2 * d_edge, d_edge_hid)
        self.lin0_s = Linear(d_node, d_edge_hid)
        self.lin0_t = Linear(d_node, d_edge_hid)
        self.act = act_fn()
        self.lin1 = Linear(d_edge_hid, d_edge)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_features, edge_features):
        source_nodes = (
            self.lin0_s(node_features)
            .unsqueeze(-2)
            .expand(-1, -1, node_features.size(-2), -1)
        )
        target_nodes = (
            self.lin0_t(node_features)
            .unsqueeze(-3)
            .expand(-1, node_features.size(-2), -1, -1)
        )

        # reversed_edge_features = self.reverse_edge(edge_features)
        edge_features = self.lin0_e(
            torch.cat([edge_features, self.reverse_edge(edge_features)], dim=-1)
        )
        edge_features = edge_features + source_nodes + target_nodes
        edge_features = self.act(edge_features)
        edge_features = self.lin1(edge_features)
        edge_features = self.drop(edge_features)

        return edge_features


class RTAttention(nn.Module):
    def __init__(self, d_node, d_edge, d_hid, n_heads, modulate_v=None, use_ln=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_hid = d_hid
        self.use_ln = use_ln
        self.modulate_v = modulate_v
        self.scale = 1 / (d_hid**0.5)
        self.split_head_node = Rearrange("b n (h d) -> b h n d", h=n_heads)
        self.split_head_edge = Rearrange("b n m (h d) -> b h n m d", h=n_heads)
        self.cat_head_node = Rearrange("... h n d -> ... n (h d)", h=n_heads)

        self.qkv_node = Linear(d_node, 3 * d_hid, bias=False)
        self.edge_factor = 4 if modulate_v else 3
        self.qkv_edge = Linear(d_edge, self.edge_factor * d_hid, bias=False)
        self.proj_out = Linear(d_hid, d_node)

    def forward(self, node_features, edge_features, mask):
        qkv_node = self.qkv_node(node_features)
        # qkv_node = rearrange(qkv_node, "b n (h d) -> b h n d", h=self.n_heads)
        qkv_node = self.split_head_node(qkv_node)
        q_node, k_node, v_node = torch.chunk(qkv_node, 3, dim=-1)

        qkv_edge = self.qkv_edge(edge_features)
        # qkv_edge = rearrange(qkv_edge, "b n m (h d) -> b h n m d", h=self.n_heads)
        qkv_edge = self.split_head_edge(qkv_edge)
        qkv_edge = torch.chunk(qkv_edge, self.edge_factor, dim=-1)
        # q_edge, k_edge, v_edge, q_edge_b, k_edge_b, v_edge_b = torch.chunk(
        #     qkv_edge, 6, dim=-1
        # )
        # qkv_edge = [item.masked_fill(mask.unsqueeze(1) == 0, 0) for item in qkv_edge]

        q = q_node.unsqueeze(-2) + qkv_edge[0]  # + q_edge_b
        k = k_node.unsqueeze(-3) + qkv_edge[1]  # + k_edge_b
        if self.modulate_v:
            v = v_node.unsqueeze(-3) * qkv_edge[3] + qkv_edge[2]
        else:
            v = v_node.unsqueeze(-3) + qkv_edge[2]

        dots = self.scale * torch.einsum("b h i j d, b h i j d -> b h i j", q, k)
        # dots.masked_fill_(mask.unsqueeze(1).squeeze(-1) == 0, -1e-9)

        attn = F.softmax(dots, dim=-1)
        out = torch.einsum("b h i j, b h i j d -> b h i d", attn, v)
        out = self.cat_head_node(out)
        return self.proj_out(out)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)  # , gain=1 / math.sqrt(2))
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
