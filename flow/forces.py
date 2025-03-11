import numpy as np
import torch

from utils.data import Batch, WeightSpaceObject


def get_force_field(
    model,
    layer_layout,
    trainloader,
    batch_size,
    device,
    n_eval_batch=1,
    loss_fn=torch.nn.MSELoss(),
):
    """
    Get the 'force field' to guide the integration of the ODE.

    Args:
        model (_type_): base model (e.g. MLP)
        layer_layout (_type_): array containing layer dims for the base model
        trainloader (_type_): loader for the base model's training data
        batch_size (_type_): batch size for the batch of WSOs
        device (_type_): device
        n_eval_batch (int, optional): how many batches to evaluate the base task gradient with. Defaults to 1.
        loss_fn (_type_, optional): loss used for the base task. Defaults to torch.nn.MSELoss().

    """

    def force_field(batch: torch.Tensor) -> torch.Tensor:
        wsos = WeightSpaceObject.from_batch(
            Batch.deflatten(
                batch,
                batch_size=batch_size,
                layer_layout=np.array(layer_layout),
                device=device,
            ),
            layers=np.array(layer_layout),
        )
        forces = []
        for wso in wsos:
            wso.apply_to(model)
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)

                # flatten into correct shape and return
                weight_flats, bias_flats = [], []
                for i, x in enumerate(grad):
                    if i % 2 == 0:
                        weight_flats.append(x.flatten())
                    else:
                        bias_flats.append(x.flatten())
                grads = torch.concat(weight_flats + bias_flats)
                forces.append(-grads)
                break

        return torch.stack(forces), loss

    return force_field
