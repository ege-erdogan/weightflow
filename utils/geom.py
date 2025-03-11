import numpy as np
import torch
from geoopt.manifolds import Sphere
from torch import Tensor

from utils.data import WeightSpaceObject

sphere = Sphere()


def _extract_wb(
    wso0: WeightSpaceObject, wso1: None | WeightSpaceObject = None
) -> tuple:
    """Get the weights and biases as lists in order from two WeightSpaceObjects."""
    w0, w1, b0, b1 = [], [], [], []
    for i, (x0, x1) in enumerate(zip(wso0.weights, wso1.weights, strict=False)):
        if i == len(wso0.weights) - 1:
            w0.append(x0.T.flatten().unsqueeze(dim=0))
            w1.append(x1.T.flatten().unsqueeze(dim=0))
        else:
            w0.append(x0.T)
            w1.append(x1.T)
    if wso1 is not None:
        for x0, x1 in zip(wso0.biases, wso1.biases, strict=False):
            b0.append(x0)
            b1.append(x1)
        return w0, w1, b0, b1
    return w0, b0


def sphere_point_dist(
    wso_a: Tensor,
    wso_b: Tensor,
) -> Tensor:
    """Geodesic distance between two points on the sphere."""
    w_a, w_b, b_a, b_b = _extract_wb(wso_a, wso_b)
    weight_loss = torch.stack(
        [(sphere.dist(a, b) ** 2).mean() for a, b in zip(w_a, w_b, strict=False)]
    ).mean()
    bias_loss = torch.nn.functional.mse_loss(
        torch.concat(b_a).flatten(), torch.concat(b_b).flatten()
    )
    return weight_loss + bias_loss


def sphere_squared_vec_dist(
    wsot: Tensor,
    wso1: Tensor,
    vt: Tensor,
    ut: Tensor,
    layer_layout: np.ndarray,
    device: torch.device,
) -> Tensor:
    """Squared distance between two vectors on the sphere, computed via its inner product."""
    wt, w1, bt, b1 = _extract_wb(wsot, wso1)
    vt_wso = WeightSpaceObject.from_flat(vt, layer_layout, device)
    ut_wso = WeightSpaceObject.from_flat(ut, layer_layout, device)
    vt_w, ut_w, vt_b, ut_b = _extract_wb(vt_wso, ut_wso)

    vt_w_tr = vt_w

    weight_loss = torch.stack(
        [
            (sphere.inner(wt[i], v - u, v - u) ** 2).mean()
            for i, (v, u) in enumerate(zip(vt_w_tr, ut_w, strict=False))
        ]
    ).mean()
    bias_loss = torch.nn.functional.mse_loss(
        torch.concat(vt_b).flatten(), torch.concat(ut_b).flatten()
    )

    return weight_loss + bias_loss


def spherical_interpolation(p: Tensor, q: Tensor, t: Tensor) -> Tensor:
    """Interpolate between two points on the sphere, wtih t in [0,1]."""
    return sphere.expmap(p, t * sphere.logmap(p, q))


def weight_vector_field(xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Compute the weight component of the ground truth vector field from xt at time t to x1."""
    return sphere.logmap(xt, x1) / (1 - t)


def bias_vector_field(xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Compute the bias component of the ground truth vector field from xt at time t to x1."""
    return (x1 - xt) / (1 - t.unsqueeze(dim=-1))


def interpolate(
    t: Tensor, wso0: WeightSpaceObject, wso1: WeightSpaceObject
) -> WeightSpaceObject:
    """
    Interpolate between two WeightSpaceObjects with t in [0,1].

    Interpolates over the spheres for the weights and in Euclidean space for the biases.
    """
    w0, w1, b0, b1 = _extract_wb(wso0, wso1)
    wt = []
    for i, (x0, x1) in enumerate(zip(w0, w1, strict=False)):
        xt = spherical_interpolation(x0, x1, t).T
        if i == len(w0) - 1:
            wt.append(xt.view(-1, wt[-1].size(1)).T)
        else:
            wt.append(xt)

    # wt = [spherical_interpolation(x0, x1, t).T for x0, x1 in zip(w0, w1)]
    bt = [(1 - t) * x0 + t * x1 for x0, x1 in zip(b0, b1, strict=False)]
    return WeightSpaceObject(wt, bt)


def vector_field(
    t: Tensor, wsot: WeightSpaceObject, wso1: WeightSpaceObject
) -> torch.Tensor:
    """Compute the ground truth vector field from wso_t to wso_1 at time t, combining the vector fields for the weight/bias components."""
    wt, w1, bt, b1 = _extract_wb(wsot, wso1)
    uw_t = [
        weight_vector_field(xt, x1, t).T.flatten()
        for xt, x1 in zip(wt, w1, strict=True)
    ]
    ub_t = [
        bias_vector_field(xt, x1, t).flatten() for xt, x1 in zip(bt, b1, strict=True)
    ]
    return torch.concat(uw_t + ub_t, dim=0)


def euler_step(
    xt: Tensor,
    vt: Tensor,
    dt: Tensor,
    layer_layout: None | np.ndarray,
    device: None | torch.device,
) -> Tensor:
    """Perform one Euler step in the product geometry."""
    vt_wso = WeightSpaceObject.from_flat(vt, layer_layout, device)
    xt_wso = WeightSpaceObject.from_flat(xt, layer_layout, device)
    vt_w, xt_w, vt_b, xt_b = _extract_wb(vt_wso, xt_wso)

    w_new = []
    for i, (x, v) in enumerate(zip(xt_w, vt_w, strict=True)):
        x_new = sphere.expmap(x, v * dt).T
        if i == len(vt_w) - 1:
            w_new.append(x_new.view(-1, w_new[-1].size(-1)).T)
        else:
            w_new.append(x_new)

    b_new = [b_t + dt * vb_t for b_t, vb_t in zip(xt_b, vt_b, strict=True)]

    return WeightSpaceObject(w_new, b_new)
