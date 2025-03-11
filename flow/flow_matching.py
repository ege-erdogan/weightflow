import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchcfm import OTPlanSampler
from tqdm import tqdm, trange

import utils.geom as geom
from utils.data import Bunch, sample_from_loader
from utils.data import WeightSpaceObject as WSO


class CFM:
    """Conditional Flow Matching object to handle training of and mapping with flow models."""

    def __init__(
        self,
        sourceloader: DataLoader,
        targetloader: DataLoader,
        model: nn.Module,
        layer_layout: None | np.ndarray = None,
        fm_type: str = "vanilla",
        geometric: bool = False,
        mode: str = "velocity",
        normalize_pred: bool = False,
        t_dist: str = "uniform",
        device: None | torch.device = None,
    ):
        """
        Initialize CFM objects.

        Args:
            sourceloader (DataLoader): source data loader (x_0)
            targetloader (DataLoader): target data loader (x_1)
            model (nn.Module): velocity model
            layer_layout (Optional[np.array], optional): layer layout for the base MLP. Defaults to None.
            fm_type (str, optional):
                - "vanilla": (default). independent coupling
                - "ot": mini-batch optimal transport couplings
            geometric (bool, optional): Flow is defined over the hypersphere product geometry if True. Defaults to False.
            mode (str, optional):
                - "velocity": (Default) use flow model to predict velocity v_t
                - "target":  use flow model to predict target point x_1
            normalize_pred (bool, optional): normalize (map to product geometry) flow model predictions. Defaults to False.
            t_dist (str, optional):
                - "uniform": (Default) sample time uniformy in [0,1]
                - "beta": sample time from Beta(1,2)
            device (torch.device, optional): device

        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_dist = t_dist
        self.sourceloader = sourceloader
        self.targetloader = targetloader
        self.model = model
        self.fm_type = fm_type
        self.device = device
        self.layer_layout = layer_layout
        self.mode = mode
        self.normalize_pred = normalize_pred
        self.geometric = geometric
        self.lambda_f = 0
        self.force_field = lambda x: x  # supress linter warnings
        if self.fm_type == "ot":
            self.ot_sampler = OTPlanSampler(method="exact")

    def map(
        self,
        x0: torch.Tensor,
        n_steps: int = 5,
        return_traj: bool = False,
        noise_scale: float = 0.0,
    ) -> torch.Tensor | list:
        """
        Map x_0 from t=0 to t=1 with the flow model using an Euler ODE solver.

        Args:
            x0 (torch.Tensor): Starting point.
            n_steps (int, optional): Number of Euler steps. Defaults to 5.
            return_traj (bool, optional): Return the whole trajectory. Defaults to False.
            noise_scale (float, optional): Noise level for stochastic sampling (only at t >= 0.8). Defaults to 0.0.

        Returns:
            xt: Mapped point at t=1.

        """
        traj = []
        times = torch.linspace(0, 1, n_steps).to(self.device)
        dt = times[1]  # since times[0] = 0
        n_batch = x0.size(0)

        xt = x0.clone()
        for t in times[:-1]:  # don't push forward at t=1
            xt = xt.to(self.device)
            if return_traj:
                traj.append(xt.detach().clone())
            if t > 0.8:
                # add noise before getting vector field
                xt += torch.randn_like(xt).to(self.device) * noise_scale
            with torch.no_grad():
                vt = self.vector_field(xt, t, x0)
            if self.lambda_f > 0 and self.force_field is not None:
                force, loss = self.force_field(xt)
                vt = vt + self.lambda_f * force

            if self.geometric:
                xt = torch.stack(
                    [
                        geom.euler_step(
                            xt_i, vt_i, dt, self.layer_layout, self.device
                        ).flatten()
                        for xt_i, vt_i in zip(xt, vt, strict=True)
                    ]
                )
            else:
                xt += vt.view(n_batch, vt.size(1)) * dt

        if return_traj:
            traj.append(xt.detach().clone())
            return traj
        return xt

    def forward(self, flow: Bunch) -> tuple[None, torch.Tensor]:
        """
        Do a forward pass through the model.

        Args:
            flow (Bunch): Object containing the flow attributes

        Returns:
            tuple[None, torch.Tensor]: Flow model prediction

        """
        flow_pred = self.model(inputs=flow.xt, t=flow.t)
        if self.normalize_pred:
            flow_pred = WSO.project_to_sphere(
                flow_pred.view(flow.xt.size()), self.layer_layout, self.device
            )
        return None, flow_pred

    def vector_field(
        self, xt: torch.Tensor, t: torch.Tensor, x0: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the predicted vector field at time t and point xt.

        Args:
            xt (torch.Tensor): current point x_t
            t (torch.Tensor): time
            x0 (torch.Tensor, optional): Optional starting point x_0. Defaults to None.

        Returns:
            torch.Tensor: velocity at time t

        """
        n_batch, d = xt.size(0), xt.size(1)
        t_rep = t.repeat(n_batch).unsqueeze(dim=1).repeat(1, d)
        _, pred = self.forward(Bunch(xt=xt, t=t_rep))

        if self.mode == "velocity":
            vt = pred
        elif self.mode == "target":
            xt = xt.view(pred.size())
            if self.geometric:
                t_step = t  # if t.dim() == 0 else t[i]
                pred_wsos = [
                    WSO.from_flat(x, self.layer_layout, self.device) for x in pred
                ]
                wsos_t = [WSO.from_flat(x, self.layer_layout, self.device) for x in xt]
                vt = torch.stack(
                    [
                        geom.vector_field(t_step, wso_t, pred_wso)
                        for i, (wso_t, pred_wso) in enumerate(
                            zip(wsos_t, pred_wsos, strict=False)
                        )
                    ]
                )
            vt = pred - x0.view(pred.size())
        vt = vt.view(n_batch, -1)
        return vt

    def sample_time_and_flow(self):
        """
        Sample time, start and end points, and intermediate x_t.

        Returns:
            Object with time (t), starting point (x0), intermediate point (xt), end point (x1), and target velocity (ut)

        """
        x0 = sample_from_loader(self.sourceloader).flatten(keep_n_batch=True)
        x1 = sample_from_loader(self.targetloader).flatten(keep_n_batch=True)
        x0, x1 = x0.to(self.device), x1.to(self.device)

        if self.t_dist == "uniform":
            t = torch.rand(x0.size(0)).to(self.device)
        elif self.t_dist == "beta":
            t = torch.distributions.Beta(1, 2).sample((x0.size(0),)).to(self.device)
        t_pad = t.reshape(-1, *([1] * (x0.dim() - 1)))

        if self.fm_type == "ot":
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        if self.geometric:
            wsos_0 = [WSO.from_flat(x, self.layer_layout, self.device) for x in x0]
            wsos_1 = [WSO.from_flat(x, self.layer_layout, self.device) for x in x1]
            wsos_t = [
                geom.interpolate(t[i], wso0, wso1)
                for i, (wso0, wso1) in enumerate(zip(wsos_0, wsos_1, strict=True))
            ]
            ut = [
                geom.vector_field(t[i], wsot, wso1)
                for i, (wsot, wso1) in enumerate(zip(wsos_t, wsos_1, strict=True))
            ]
            xt = torch.stack([wso.flatten() for wso in wsos_t])
            ut = torch.stack(ut)
        else:
            mu_t = (1 - t_pad) * x0 + t_pad * x1
            sigma_pad = (
                torch.tensor(self.sigma)
                .reshape(-1, *([1] * (x0.dim() - 1)))
                .to(self.device)
            )
            xt = mu_t + sigma_pad * torch.randn_like(x0).to(self.device)
            ut = x1 - x0

        t, xt, ut = t.to(self.device), xt.to(self.device), ut.to(self.device)

        t = t.unsqueeze(dim=-1).repeat(1, xt.size(-1))

        return Bunch(t=t, x0=x0, xt=xt, x1=x1, ut=ut, eps=0, lambda_t=0)

    def loss_fn(
        self,
        flow_pred: torch.Tensor,
        flow: Bunch,
    ) -> torch.Tensor:
        """
        Compute the loss to train the flow model.

        Args:
            flow_pred (torch.Tensor): Predicted flow object
            flow (Bunch): Ground truth flow object

        Returns:
            torch.Tensor: Loss

        """
        if self.mode == "target":
            if self.geometric:
                wsos_pred = [
                    WSO.from_flat(x, self.layer_layout, self.device) for x in flow_pred
                ]
                wsos_true = [
                    WSO.from_flat(x, self.layer_layout, self.device) for x in flow.x1
                ]
                l_flow = torch.stack(
                    [
                        geom.sphere_point_dist(wso_pred, wso_true)
                        for wso_pred, wso_true in zip(wsos_pred, wsos_true, strict=True)
                    ]
                )
                l_flow = l_flow.mean()
            else:
                l_flow = torch.mean((flow_pred.squeeze() - flow.x1) ** 2)
        elif self.mode == "velocity":
            if self.geometric:
                wsos_t = [
                    WSO.from_flat(x, self.layer_layout, self.device) for x in flow.xt
                ]
                wsos_1 = [
                    WSO.from_flat(x, self.layer_layout, self.device) for x in flow.x1
                ]
                l_flow = torch.stack(
                    [
                        geom.sphere_squared_vec_dist(
                            wso_t,
                            wso_1,
                            flow_pred[i],
                            flow.ut[i],
                            self.layer_layout,
                            self.device,
                        )
                        for i, (wso_t, wso_1) in enumerate(
                            zip(wsos_t, wsos_1, strict=True)
                        )
                    ]
                )
                l_flow = l_flow.mean()
            else:
                l_flow = torch.mean((flow_pred.squeeze() - flow.ut) ** 2)
        return None, l_flow

    def train(
        self,
        n_iters: int = 10,
        optimizer: None | torch.optim.Optimizer = None,
        sigma: float = 0.001,
        patience: int = 1e99,
        log_freq: int = 5,
        wandb=None,
    ):
        """
        Train the CFM model.

        Args:
            n_iters (int, optional): Number of training steps. Defaults to 10.
            optimizer (None | torch.optim.Optimizer, optional): Optimizer. Defaults to None.
            sigma (float, optional): Sigma for Gaussian probability paths. Defaults to 0.001.
            patience (int, optional): Stop training if loss doesn't decrease for this many steps. Defaults to 1e99.
            log_freq (int, optional): Log loss and other metrics every this many steps. Defaults to 5.
            wandb (_type_, optional): wandb object to log losses etc to. Defaults to None.

        """
        self.sigma = sigma
        self.wandb = wandb
        self.metrics: dict[str, list[float]] = {
            "train_loss": [],
            "time": [],
            "grad_norm": [],
            "flow_norm": [],
            "true_norm": [],
        }
        last_loss = 1e99
        patience_count = 0

        pbar = trange(n_iters, desc="Training steps")
        for i in pbar:
            optimizer.zero_grad()

            flow = self.sample_time_and_flow()
            _, flow_pred = self.forward(flow)
            _, loss = self.loss_fn(flow_pred, flow)
            loss.backward()
            optimizer.step()

            # early stopping wrt the train loss
            if loss.item() > last_loss:
                patience_count += 1
                if patience_count >= patience:
                    break
            last_loss = loss.item()

            if i % log_freq == 0:
                train_loss_val = loss.item()

                true_tensor = flow.ut if self.mode == "velocity" else flow.x1
                grad_norm = self.model.get_grad_norm()
                self.log_metric("train_loss", train_loss_val)
                self.log_metric("flow_norm", flow_pred.norm(p=2, dim=1).mean().item())
                self.log_metric("time", flow.t.mean().item())
                self.log_metric("true_norm", true_tensor.norm(p=2, dim=1).mean().item())
                self.log_metric("grad_norm", grad_norm)

                pbar.set_description(
                    f"Iters [loss {train_loss_val:.6f}, ∇ norm {grad_norm:.6f}]"
                )

    def estimate_jacobian_trace(self, xt, t, x0, nfe=25):
        """
        Estimate the Jacobian trace with Hutchinson's estimator.

        Args:
            xt (_type_): input point
            t (_type_): time to evalaute the vector field at
            x0 (_type_): starting point
            nfe (int, optional): number of function evaluations. Defaults to 25.

        """

        def functional_v(xt):
            return self.vector_field(xt, t, x0)

        mean = 0
        for _ in range(nfe):
            eps = torch.randn_like(xt)
            dx, jvp = torch.autograd.functional.jvp(functional_v, inputs=xt, v=eps)
            est = torch.einsum("bi,bi->b", eps, jvp)
            mean += est
        est_trace = mean / nfe
        return dx, est_trace

    def get_ode(self, x0: torch.Tensor, trace="exact", nfe=25):
        """
        Get the function representing the ODE learned by the flow model.

        Args:
            x0 (torch.Tensor): Starting point
            trace (str, optional):
                - "exact": Compute the Jacobian trace exactly (computationally intensive in high dim)
                - "hutchinson": Estimate Jacobian trace using Hutchinson's estimator. Significantly faster.
            nfe (int, optional): Number of function evaluations for the Hutchinson estimator. Defaults to 25.

        """

        def ode(
            t: torch.Tensor,
            x: torch.Tensor,
        ) -> torch.Tensor:
            """Pass x through the flow and compute the change in x and the neg. log det. (/divergence/trace)."""
            torch.cuda.empty_cache()
            xt, _ = x
            if not xt.requires_grad:
                xt.requires_grad = True
            n_batch, d = xt.size(0), xt.size(1)

            if trace == "exact":
                divergence = torch.zeros(n_batch).to(xt)
                dx = self.vector_field(xt, t, x0)
                for i in tqdm(range(d)):
                    # for each dimension of the vector field
                    # xt of size (n_batch,dim)
                    grad_dx_i = torch.autograd.grad(
                        outputs=dx[:, i],  # ith output dim for each batch
                        inputs=xt,
                        grad_outputs=torch.ones_like(dx[:, i]).to(self.device),
                        retain_graph=(i < d - 1),
                        create_graph=False,
                    )[0]
                    # grad_dx_i of size (n_batch, d)
                    divergence += grad_dx_i[:, i]
            elif trace == "hutchinson":
                dx, divergence = self.estimate_jacobian_trace(xt, t, x0, nfe=nfe)

            return dx, divergence.view(-1, 1)

        return ode

    def save_model(self, path: str):
        """Save the flow model at path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load flow model from path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def log_metric(self, key, value):
        """Log the metrics saved during training to wandb."""
        self.metrics[key].append(value)
        if self.wandb is not None:
            self.wandb.log({key: value})

    def plot_metrics(self):
        """Plot the metrics saved during training."""
        labels = list(self.metrics.keys())
        lists = list(self.metrics.values())
        n = len(lists)

        fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
        for i, (label, lst) in enumerate(zip(labels, lists, strict=True)):
            axs[i].plot(lst)
            axs[i].grid()
            axs[i].title.set_text(label)
            if label == "train_loss":
                axs[i].set_yscale("log")
        plt.tight_layout()
        plt.show()
