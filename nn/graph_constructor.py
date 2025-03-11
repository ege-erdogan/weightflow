import torch
import torch.nn as nn


def batch_to_graphs(
    weights,
    biases,
    labels,
    weights_mean=None,
    weights_std=None,
    biases_mean=None,
    biases_std=None,
):
    device = weights[0].device
    bsz = weights[0].shape[0]
    # num_nodes = weights[0].shape[1] + sum(w.shape[2] for w in weights)
    num_nodes = weights[0].shape[2] + sum(w.shape[1] for w in weights)

    node_features = torch.zeros(bsz, num_nodes, biases[0].shape[-1], device=device)
    edge_features = torch.zeros(
        bsz, num_nodes, num_nodes, weights[0].shape[-1], device=device
    )

    row_offset = 0
    col_offset = weights[0].shape[2]  # no edge to input nodes
    for i, w in enumerate(weights):
        # _, num_in, num_out, _ = w.shape
        _, num_out, num_in, _ = w.shape
        w_mean = weights_mean[i] if weights_mean is not None else 0
        w_std = weights_std[i] if weights_std is not None else 1
        edge_features[
            :, row_offset : row_offset + num_in, col_offset : col_offset + num_out
        ] = (w.reshape(bsz, num_in, num_out, 1) - w_mean) / w_std
        row_offset += num_in
        col_offset += num_out

    row_offset = weights[0].shape[2]  # no bias in input nodes
    for i, b in enumerate(biases):
        _, num_out, _ = b.shape
        b_mean = biases_mean[i] if biases_mean is not None else 0
        b_std = biases_std[i] if biases_std is not None else 1
        node_features[:, row_offset : row_offset + num_out] = (b - b_mean) / b_std
        row_offset += num_out

    return node_features, edge_features


class GraphConstructor(nn.Module):
    def __init__(
        self,
        d_in,
        d_edge_in,
        d_node,
        d_edge,
        layer_layout,
        rev_edge_features=False,
        in_proj_layers=1,  # no of layers in input projection from d_edge_in to d_edge
        use_pos_embed=True,
        num_probe_features=0,
        inr_model=None,
        stats=None,
        pos_embed_per_node=False,
    ):
        super().__init__()
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
        self.use_pos_embed = use_pos_embed
        self.stats = stats if stats is not None else {}
        self._d_node = d_node
        self._d_edge = d_edge

        # -- POSITIONAL EMBEDDINGS
        self.pos_embed_layout = (
            [1] * layer_layout[0] + layer_layout[1:-1] + [1] * layer_layout[-1]
        )
        if not pos_embed_per_node:
            self.pos_embed = nn.Parameter(
                torch.randn(len(self.pos_embed_layout), d_node)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(sum(self.pos_embed_layout), d_node)
            )

        proj_weight = []
        proj_weight.append(
            nn.Linear(d_edge_in + (2 * d_edge_in if rev_edge_features else 0), d_edge)
        )
        # --

        for i in range(in_proj_layers - 1):
            proj_weight.append(nn.SiLU())
            proj_weight.append(nn.Linear(d_edge, d_edge))

        self.proj_weight = nn.Sequential(*proj_weight)

        proj_bias = []
        proj_bias.append(nn.Linear(d_in, d_node))

        for _ in range(in_proj_layers - 1):
            proj_bias.append(nn.SiLU())
            proj_bias.append(nn.Linear(d_node, d_node))

        self.proj_bias = nn.Sequential(*proj_bias)

        self.proj_node_in = nn.Linear(d_node, d_node)
        self.proj_edge_in = nn.Linear(d_edge, d_edge)

        self.gpf = None

    def forward(self, inputs):
        node_features, edge_features = batch_to_graphs(*inputs, **self.stats)
        mask = edge_features.sum(dim=-1, keepdim=True) != 0

        if self.rev_edge_features:
            rev_edge_features = edge_features.transpose(-2, -3)
            edge_features = torch.cat(
                [edge_features, rev_edge_features, edge_features + rev_edge_features],
                dim=-1,
            )
            mask = mask | mask.transpose(-3, -2)

        edge_features = self.proj_weight(edge_features)
        node_features = self.proj_bias(node_features)

        if self.gpf is not None:
            probe_features = self.gpf(*inputs)
            node_features = node_features + probe_features

        node_features = self.proj_node_in(node_features)
        edge_features = self.proj_edge_in(edge_features)

        if self.use_pos_embed:
            pos_embed = torch.cat(
                [
                    # repeat(self.pos_embed[i], "d -> 1 n d", n=n)
                    self.pos_embed[i].unsqueeze(0).expand(1, n, -1)
                    for i, n in enumerate(self.pos_embed_layout)
                ],
                dim=1,
            )
            node_features = node_features + pos_embed
        return node_features, edge_features, mask
