from collections import OrderedDict

import torch

ACTIVATIONS = {
    "relu": torch.nn.ReLU(),
    "sigmoid": torch.nn.Sigmoid(),
}


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        weight_init: str = "uniform",
        source_std: float = 0.1,
        activation="relu",
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = ACTIVATIONS[activation]
        self.source_std = source_std
        self.is_deep = len(hidden_sizes) > 1

        # Create the input layer
        self.input_layer = torch.nn.Linear(input_size, hidden_sizes[0])

        # Create the hidden layers
        # self.hidden_layers = torch.nn.ModuleList()
        # for i in range(len(hidden_sizes) - 1):
        #     hidden_layer = torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
        #     self.hidden_layers.append(hidden_layer)
        keys, layers = [], []
        for i in range(len(hidden_sizes) - 1):
            keys.append(f"fc{i}")
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            keys.append(f"act{i}")
            layers.append(ACTIVATIONS[activation])  # TODO config
        self.hidden = torch.nn.Sequential(
            OrderedDict([(k, v) for k, v in zip(keys, layers, strict=False)])
        )

        # Create the output layer
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], output_size)

        if weight_init == "normal":
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.source_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = self.activation(self.input_layer(x))
        if self.is_deep:
            # for hidden_layer in self.hidden_layers:
            # x = self.activation(hidden_layer(x))
            x = self.hidden(x)
        x = self.output_layer(x)
        return x

    @torch.no_grad()
    def set_weights(self, wso):
        wb = wso.get_wb()
        will_transpose = [False for _ in wb]
        for i, param in enumerate(self.parameters()):
            if i % 2 == 0 and wb[i].dim() == 1:
                wb[i] = wb[i].unsqueeze(dim=-1)
            if wb[i].size() != param.size():
                will_transpose[i] = True
        for i, param in enumerate(self.parameters()):
            if i % 2 == 0:  # weights
                if will_transpose[i]:
                    param.data = torch.nn.Parameter(wb[i].T)
                else:
                    param.data = torch.nn.Parameter(wb[i].T)
            else:  # biases
                if will_transpose[i]:
                    param.data = torch.nn.Parameter(wb[i].unsqueeze(dim=-1))
                else:
                    param.data = torch.nn.Parameter(wb[i])

    def get_weights(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        weights = [
            p.detach().clone().T.unsqueeze(-1)
            for i, p in enumerate(self.parameters())
            if i % 2 == 0
        ]
        biases = [
            p.detach().clone().unsqueeze(-1)
            for i, p in enumerate(self.parameters())
            if i % 2 == 1
        ]
        return weights, biases
