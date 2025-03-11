import collections
import io
import pickle
from copy import copy, deepcopy
from random import sample
from typing import NamedTuple

import h5py
import numpy as np
import torch
from rebasin import RebasinNet
from rebasin.loss import DistL2Loss, MidLoss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def get_uci_loaders(id=None, trainrate=0.8, batch_size=64, random_state=42):
    repo = fetch_ucirepo(id=id)
    classification = id == 17
    if classification:
        label_encoder = LabelEncoder()
        X = repo.data.features
        y = label_encoder.fit_transform(repo.data.targets["Diagnosis"])
        Xs = torch.from_numpy(X.to_numpy()).to(torch.float32)
        ys = torch.from_numpy(y)
        scaler_y = None
    else:
        X, y = repo.data.features, repo.data.targets
        Xs = torch.from_numpy(X.to_numpy()).to(torch.float32)
        ys = torch.from_numpy(y.to_numpy()).to(torch.float32)
        scaler_y = StandardScaler()
        ys = torch.from_numpy(scaler_y.fit_transform(y.to_numpy().reshape(-1, 1))).to(
            torch.float32
        )
    scaler_X = StandardScaler()
    Xs = torch.from_numpy(scaler_X.fit_transform(X.to_numpy())).to(torch.float32)
    X_train, Xs_temp, y_train, ys_temp = train_test_split(
        Xs, ys, train_size=trainrate, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        Xs_temp, ys_temp, test_size=0.5, random_state=random_state
    )
    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)
    valset = TensorDataset(X_val, y_val)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    in_dim = X_train.size(1)
    out_dim = (
        len(repo.data.targets["Diagnosis"].unique())
        if classification
        else y_train.size(1)
    )
    return trainloader, testloader, valloader, in_dim, out_dim, scaler_y


def init_h5_dataset(path: str, n: int, dim: int):
    f = h5py.File(path, "a")
    dset = f.create_dataset("dataset", (n, dim))
    return dset


def wandb_log_metrics(wandb, metrics):
    """Log a dict with each value a list of the same length to wandb."""
    keys = list(metrics.keys())
    for i in range(len(metrics[keys[0]])):
        log_dict = {}
        for k in keys:
            log_dict[k] = metrics[k][i]
        print(log_dict)
        wandb.log(log_dict)


def align_to_ref(
    wsos,
    model,
    loss="l2",
    n_iters=5,
    bar=True,
    dataloader=None,
    input_dim: int = 784,
):
    """
    Align all wsos to one reference.

    Args:
        wsos (_type_): List of WSOs to align, ref: wsos[0].
        model (_type_): base model
        loss (str, optional): Loss for rebasin. Defaults to "l2".
        n_iters (int, optional): Number of iterations to align. Defaults to 5.
        bar (bool, optional): Show progress bar. Defaults to True.
        dataloader (_type_, optional): Test data loader (for "mid" loss). Defaults to None.
        input_dim (int, optional): Dimensionality of the inputs to the model. Defaults to 784.

    Returns:
        _type_: List of aligned WSOs.

    """
    if bar:
        lst = tqdm(wsos, desc="Aligning")
    else:
        lst = wsos
    return [
        wso.align(
            wsos[0],
            model,
            loss=loss,
            n_est_batch=25,
            input_dim=input_dim,
            testloader=dataloader,
            n_iters=n_iters,
            inplace=False,
        )
        for wso in lst
    ]


def augment(wso_a, wso_b, n=10):
    """Compute n equally-spaced WSOs by interpolating between wso_a and wso_b."""
    ts = np.linspace(0, 1, 10 + 2)[1:-1]
    wsos = [wso_a * (1 - t) + wso_b * t for t in ts]
    return wsos


def pickle_dump(path, obj):
    with open(path, "wb+") as f:
        pickle.dump(obj, f)


def pickle_load(path, cpu=False):
    if cpu:

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                else:
                    return super().find_class(module, name)

    with open(path, "rb") as f:
        return CPU_Unpickler(f).load() if cpu else pickle.load(f)


def sample_from_loader(loader: DataLoader):
    """Sample iteratively from a dataloader, circling back after the end."""
    if loader._iterator is None:
        loader._iterator = iter(loader)
    try:
        return next(loader._iterator)
    except StopIteration:
        # StopIteration thrown when all the data is iterated over
        # Data is reshuffled by default for repeats
        loader._iterator = iter(loader)
        return next(loader._iterator)


def load_weights(
    filepath: str,
    batch_size: int,
    layers: list[int],
    n_samples: int | None = None,
    map_fn=None,
) -> tuple[torch.utils.data.DataLoader, torch.Tensor]:
    """
    Load NN weights from a collection of h5 files.

    Args:
        filepath (str): Path to the h5 file
        batch_size (int): Batch size for the returned loaders
        layers (List[int]): List of integers with the layer widths of the MLP.
        n_samples (Optional[int], optional): Number of samples to subsample, no subsampling if None. Defaults to None.
        map_fn (_type_, optional): Optional function to apply over the loaded WSOs. Defaults to None.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.Tensor]: _description_

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = h5py.File(filepath, "r")
    dset = f["dataset"]
    weights = []
    # iterator = dset if n_samples is None else dset[:n_samples]
    iterator = dset
    for w in tqdm(iterator, desc="Loading data"):
        weights.append(
            WeightSpaceObject.from_flat(
                w, layers=np.array(layers), device=device, revert=True
            )
        )

    if map_fn is not None:
        weights = [w.map(map_fn) for w in weights]

    if n_samples is not None:
        weights = sample(weights, n_samples)

    # create the layer layout
    s = weights[0]
    weight_shapes = [w.shape for w in s.weights]
    bias_shapes = [b.shape for b in s.biases]
    # if layers[0] != 1:
    if "mnist" in filepath:
        layer_layout = [s[0] for s in weight_shapes] + [weight_shapes[-1][0]]
    else:
        layer_layout = (
            [weight_shapes[0][0]]
            + [s[1] for s in weight_shapes[:-1]]
            + [bias_shapes[-1][0]]
        )

    # create data loader
    weight_dataset = WeightDataset(weights, labels=None)
    weightloader = torch.utils.data.DataLoader(
        weight_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return weightloader, layer_layout


def load_mnist(batch_size: int, data_dir: str = "../data", key="mnist"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if key == "mnist":
        train_set = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_val_set = datasets.MNIST(data_dir, train=False, transform=transform)
    elif key == "f-mnist":
        train_set = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_val_set = datasets.FashionMNIST(data_dir, train=False, transform=transform)
    else:
        raise "Dataset key not valid"

    test_set, val_set = random_split(
        test_val_set, [len(test_val_set) // 2, len(test_val_set) // 2]
    )

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, valloader


def load_cifar(batch_size=64, val_split=0.1):
    tr_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    tr_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load full training set
    full_trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=tr_train
    )

    # Calculate split sizes
    val_size = int(len(full_trainset) * val_split)
    train_size = len(full_trainset) - val_size

    # Split into train and validation sets
    trainset, valset = torch.utils.data.random_split(
        full_trainset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Override transform for validation set (no augmentation)
    valset.dataset.transform = tr_test

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=tr_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, valloader


class WeightSpaceObject(collections.abc.Sequence):
    """Object to represent point in weight-space. Currently only supports MLPs."""

    def __init__(self, weights, biases):
        if isinstance(weights, list):
            weights = tuple(weights)
        if isinstance(biases, list):
            biases = tuple(biases)
        if weights[0].dim() == 1:
            for i, w in enumerate(weights):
                weights[i] = w.unsqueeze(dim=-1 if i == 0 else 0)
        self.weights = weights
        self.biases = biases

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        return zip(self.weights, self.biases, strict=True)

    def __getitem__(self, idx):
        return (self.weights[idx], self.biases[idx])

    def __add__(self, other):
        """Add each corresponding component of two WSOs."""
        out_weights = tuple(
            w1 + w2.view(w1.size())
            for w1, w2 in zip(self.weights, other.weights, strict=True)
        )
        out_biases = tuple(
            b1 + b2.view(b1.size())
            for b1, b2 in zip(self.biases, other.biases, strict=True)
        )
        return WeightSpaceObject(out_weights, out_biases)

    def __mul__(self, other):
        """Multiply each component with the corresponding component in another WSO or by a constant."""
        if isinstance(other, WeightSpaceObject):
            weights = tuple(
                w1 * w2 for w1, w2 in zip(self.weights, other.weights, strict=True)
            )
            biases = tuple(
                b1 * b2 for b1, b2 in zip(self.biases, other.biases, strict=True)
            )
            return WeightSpaceObject(weights, biases)
        return self.map(lambda x: x * other)

    def unsqueeze(self):
        self.weights = tuple(
            w.unsqueeze(dim=-1 if i == 0 else 0) for i, w in enumerate(self.weights)
        )

    def size(self):
        sizes = []
        for w, b in zip(self.weights, self.biases, strict=False):
            sizes.append(w.size())
            sizes.append(b.size())
        return tuple(sizes)

    def update_wb(self, wb):
        weights, biases = [], []
        for i, w in enumerate(wb):
            if i % 2 == 0:
                weights.append(w)
            else:
                biases.append(w)
        self.weights = tuple(weights)
        self.biases = tuple(biases)

    def normalize(self, skip_last=False, scale_bias=True):
        """
        For ReLU MLP, normalize incoming weights to each neuron including the biases, and the final layer as a whole.

        Preserves the functions the intermediate neurons compute.
        The predicted label is preserved for classification tasks if the last layer is scaled.
        """
        wb = self.get_wb()
        for i in range(len(wb[:-2])):
            w = wb[i]
            if i % 2 == 0:
                norms = w.norm(dim=0)
                wb[i] = wb[i] / norms
                wb[i + 2] = (norms * wb[i + 2].T).T
                if scale_bias:
                    wb[i + 1] = (1 / norms) * wb[i + 1]  # incoming bias
        if not skip_last:  # final layer
            n_labels = 1
            out_norm = wb[-2].norm()
            wb[-2] = (np.sqrt(n_labels) / out_norm) * wb[-2]
            wb[-1] = (np.sqrt(n_labels) / out_norm.flatten()) * wb[-1]
        self.update_wb(wb)

    def dist(self, other):
        """Euclidean distance between two wsos"""
        weight_dist = [
            ((w1 - w2) ** 2).sum()
            for w1, w2 in zip(self.weights, other.weights, strict=True)
        ]
        bias_dist = [
            ((b1 - b2) ** 2).sum()
            for b1, b2 in zip(self.biases, other.biases, strict=True)
        ]
        return torch.sqrt(sum(weight_dist) + sum(bias_dist))

    def detach(self):
        """Get a copy with detached tensors."""
        return WeightSpaceObject(
            tuple(w.detach() for w in self.weights),
            tuple(b.detach() for b in self.biases),
        )

    def flatten(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Get the WSO as a flat vector."""
        return torch.cat(
            [x.flatten() for x in self.weights] + [x.flatten() for x in self.biases],
            dim=0,
        ).to(device)

    def map(self, func):
        """Apply a function to each weight and bias tensor."""
        return WeightSpaceObject(
            tuple(func(w) for w in self.weights), tuple(func(b) for b in self.biases)
        )

    def align(
        self,
        other,
        model,
        loss="l2",
        testloader=None,
        n_est_batch=10,
        input_dim=784,
        n_iters=1_000,
        inplace=True,
    ):
        """Aligns this wso to the `other` wso on the given model, works in-place."""
        batch_size = 4 if testloader is None else testloader.batch_size
        device = self.weights[0].device
        model_b = deepcopy(model)
        other.apply_to(model_b)
        self.apply_to(model)
        pi_model = RebasinNet(model, input_shape=(batch_size, input_dim)).to(device)
        if loss == "l2":
            criterion = DistL2Loss(model_b)
        elif loss == "mid":
            criterion = MidLoss(model_b, criterion=torch.nn.CrossEntropyLoss())
        optimizer = torch.optim.Adam(pi_model.p.parameters(), lr=0.1)
        for i in range(n_iters):
            if loss == "l2":
                rebased_model = pi_model()
                loss_training = criterion(rebased_model)
                optimizer.zero_grad()
                loss_training.backward()
                optimizer.step()
            elif loss == "mid":
                for i, (x, y) in enumerate(testloader):
                    if i == n_est_batch:
                        break
                    rebased_model = pi_model()
                    loss_training = criterion(rebased_model, x.to(device), y.to(device))
                    optimizer.zero_grad()
                    loss_training.backward()
                    optimizer.step()
        pi_model.eval()

        permuted_model = deepcopy(pi_model())
        weights, biases = permuted_model.get_weights()
        weights = [w.view(self.weights[i].size()) for i, w in enumerate(weights)]
        biases = [b.view(self.biases[i].size()) for i, b in enumerate(biases)]

        if inplace:
            self.weights = weights
            self.biases = biases
        else:
            return WeightSpaceObject.from_zipped(zip(weights, biases, strict=True))

    def apply_to(self, model: torch.nn.Module):
        """Apply the weights/biases to the model inplace."""
        model.set_weights(self)

    def get_wb(self):
        """Get weights and biases in order to apply to a model."""
        result = []
        for w, b in self:
            result += [
                (w.squeeze(dim=-1).T if w.dim() > 2 else w),
                (b.squeeze(dim=-1) if b.dim() != 1 else b),
            ]
        return result

    def to(self, device):
        """Move all tensors to device."""
        return WeightSpaceObject(
            tuple(w.to(device) for w in self.weights),
            tuple(b.to(device) for b in self.biases),
        )

    def distance(self, other):
        dists = [
            ((x - y) ** 2).mean().sqrt()
            for x, y in zip(self.get_wb(), other.get_wb(), strict=False)
        ]
        return sum(dists).item()

    @classmethod
    def from_flat(cls, flat, layers, device, revert=True):
        is_deep = len(layers) > 3
        sizes = [0] + list(layers[0:-1] * layers[1:]) + list(layers[1:])
        csum_sizes = np.cumsum(sizes)
        if type(flat) is torch.Tensor:
            parts = [flat[s : s + sizes[i + 1]] for i, s in enumerate(csum_sizes[:-1])]
        else:
            parts = [
                torch.from_numpy(flat[s : s + sizes[i + 1]])
                for i, s in enumerate(csum_sizes[:-1])
            ]
        mid = int(len(parts) / 2)

        weights = []
        for i, w in enumerate(parts[:mid]):
            if is_deep and layers[i] == layers[i + 1]:
                if revert:
                    weights.append(w.view(layers[i], layers[i + 1]))
                else:
                    weights.append(w.view(layers[i + 1], layers[i]).T)
            else:
                if revert:
                    weights.append(w.view(layers[i], layers[i + 1]))
                else:
                    weights.append(w.view(layers[i + 1], layers[i]).T)

        weights = tuple(weights)
        biases = tuple(parts[mid:])
        return cls(weights, biases).to(device)

    @classmethod
    def project_to_sphere(cls, ws, layers, device):
        """
        Project each flattened weight in the list `ws` onto the product geometry of hyperspheres by normalizing the layers.

        Only uses WSO.normalize method but inputs and outputs are flat vectors which is useful in many places.
        """
        n_batch = ws.size(0)
        wsos = [
            WeightSpaceObject.from_flat(ws[i], layers=layers, device=device)
            for i in range(n_batch)
        ]
        proj = []
        for wso in wsos:
            wso.normalize()
            proj.append(wso.flatten())
        return torch.stack(proj).to(device)

    @classmethod
    def from_zipped(cls, weight_and_biases):
        """Convert a list of (weights, biases) to a WeightSpaceObject."""
        weights, biases = zip(*weight_and_biases, strict=False)
        return cls(weights, biases)

    @classmethod
    def from_batch(cls, batch, layers):
        batch_flat = batch.flatten(keep_n_batch=True)
        wsos = [cls.from_flat(flat, layers, device=flat.device) for flat in batch_flat]
        return wsos


def avg_wsos(lst: list[WeightSpaceObject]) -> WeightSpaceObject:
    total = lst[0]
    for w in lst[1:]:
        total = total + w
    return total * (1 / len(lst))


def sample_gaussian_wsos(
    mean: WeightSpaceObject, std: int, n=1_000
) -> list[WeightSpaceObject]:
    """Sample a list of weights from an isotropic gaussian centered at `mean`."""
    weights = [
        mean.map(
            lambda x: x
            + torch.normal(torch.zeros_like(x).to(x), torch.tensor(float(std)).to(x))
        )
        for _ in range(n)
    ]
    return weights


class Batch(NamedTuple):
    """Object to represent batches of WeightSpaceObjects while making some operations more convenient."""

    weights: list
    biases: list
    label: torch.Tensor

    def _assert_same_len(self):
        assert len({len(t) for t in self}) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """Move batch to device."""
        return self.__class__(
            weights=[w.to(device) for w in self.weights],
            biases=[w.to(device) for w in self.biases],
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])

    def flatten(
        self, keep_n_batch: bool = False, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        if keep_n_batch:
            n_batch = self.weights[0].size(0)
            flats = [
                torch.cat(
                    [x[i].flatten() for x in self.weights]
                    + [x[i].flatten() for x in self.biases],
                    dim=0,
                )
                for i in range(n_batch)
            ]
            return torch.stack(flats)
        else:
            return torch.cat(
                [x.flatten() for x in self.weights]
                + [x.flatten() for x in self.biases],
                dim=0,
            ).to(device)

    def concat(self, other, inplace=False):
        """Create a new batch with the node/edge features of the two batches concatenated."""
        weights, biases = [[] for _ in self.weights], [[] for _ in self.biases]
        for i in range(len(self.weights)):
            weights[i] = torch.cat((self.weights[i], other.weights[i]), dim=-1)
            biases[i] = torch.cat((self.biases[i], other.biases[i]), dim=-1)
        if inplace:
            for i in range(len(self.weights)):
                self.weights[i] = weights[i]
                self.biases[i] = biases[i]
        else:
            return self.__class__(weights=weights, biases=biases, label=self.label)

    @classmethod
    def deflatten(
        cls,
        flat: torch.Tensor,
        batch_size: int,
        layer_layout: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ):
        is_deep = len(layer_layout) > 3
        w_dims = layer_layout[:-1] * layer_layout[1:]
        b_dims = layer_layout[1:]
        dims = np.concatenate((np.array([0]), w_dims, b_dims))
        starts = np.cumsum(dims)
        ends = starts[1:]
        n_weights = len(layer_layout) - 1

        batch_weights, batch_biases = [], []
        final_weights, final_biases = [], []
        for i in range(batch_size):
            parts = [flat[i][si:ei] for si, ei in zip(starts, ends, strict=False)]

            weight_parts = parts[: len(parts) // 2]
            bias_parts = parts[len(parts) // 2 :]

            weights, biases = [], []
            for j, (wp, bp) in enumerate(zip(weight_parts, bias_parts, strict=False)):
                if layer_layout[0] == 1:
                    if is_deep and layer_layout[j] == layer_layout[j + 1]:
                        weights.append(
                            wp.reshape(1, layer_layout[j], layer_layout[j + 1], 1)
                        )
                    else:
                        weights.append(
                            wp.reshape(1, layer_layout[j], layer_layout[j + 1], 1).T
                        )
                else:
                    weights.append(
                        wp.reshape(1, layer_layout[j + 1], layer_layout[j], 1)
                    )
                biases.append(bp.reshape(1, layer_layout[j + 1], 1))

            batch_weights += weights
            batch_biases += biases

        for n in range(n_weights):
            final_weights.append(
                torch.cat(
                    [batch_weights[n + b * n_weights] for b in range(batch_size)], dim=0
                )
            )
            final_biases.append(
                torch.cat(
                    [batch_biases[n + b * n_weights] for b in range(batch_size)], dim=0
                )
            )

        return cls(list(final_weights), list(final_biases), torch.randn(batch_size)).to(
            device
        )


class WeightDataset(torch.utils.data.Dataset):
    def __init__(self, objects: list[WeightSpaceObject], labels: list | None = None):
        self.objects = objects
        self.labels = labels

    def shuffle(self, indices: None | list[int] = None):
        will_return = False
        if indices is None:
            will_return = True
            indices = np.arange(len(self.objects))
            np.random.shuffle(indices)
        objects = copy(self.objects)
        for dest, src in tqdm(enumerate(indices), desc="Shuffling"):
            self.objects[dest] = objects[src]
        if will_return:
            return indices

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        if self.labels is not None:
            return Batch(
                weights=self.objects[idx].weights,
                biases=self.objects[idx].biases,
                label=self.labels[idx],
            )
        else:
            return Batch(
                weights=self.objects[idx].weights,
                biases=self.objects[idx].biases,
                label=0,
            )


def get_1d_x(n_points_bin, dataset="sines"):
    return np.hstack(
        [
            np.random.uniform(-2, -1.4, n_points_bin),
            np.random.uniform(2.0, 2.8, n_points_bin),
        ]
    )


def get_1d_y(x, output_noise_scale=0.05, a=4, b=4.3):
    y = np.sin(a * (x - b))
    return y + (output_noise_scale * np.random.normal(size=y.shape))


def tensorize(x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor


def get_1d_dataloader(x, y, batch_size=64):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_1d_regression_dataset(
    n_train: int,
    n_test: int,
    batch_size: int,
    output_noise_scale: float = 0.05,
    a: float = 4,
    b: float = 4.3,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    n_points_bin_train = int(n_train / 2)
    n_points_bin_test = int(n_test / 2)
    x_train = get_1d_x(n_points_bin=n_points_bin_train)
    x_test = get_1d_x(n_points_bin=n_points_bin_test)
    x_val = get_1d_x(n_points_bin=n_points_bin_test)

    y_train = get_1d_y(x_train, output_noise_scale, a=a, b=b)
    y_test = get_1d_y(x_test, output_noise_scale, a=a, b=b)
    y_val = get_1d_y(x_val, output_noise_scale, a=a, b=b)

    x_train = x_train.reshape((x_train.size, 1))
    x_test = x_test.reshape((x_test.size, 1))
    x_val = x_val.reshape((x_val.size, 1))

    y_train = y_train.reshape((x_train.size, 1))
    y_test = y_test.reshape((x_test.size, 1))
    y_val = y_val.reshape((x_val.size, 1))

    x_train_tensor, y_train_tensor = tensorize(x_train, y_train)
    x_test_tensor, y_test_tensor = tensorize(x_test, y_test)
    x_val_tensor, y_val_tensor = tensorize(x_val, y_val)

    trainloader = get_1d_dataloader(
        x_train_tensor, y_train_tensor, batch_size=batch_size
    )
    testloader = get_1d_dataloader(x_test_tensor, y_test_tensor, batch_size=batch_size)
    valloader = get_1d_dataloader(x_val_tensor, y_val_tensor, batch_size=batch_size)

    return trainloader, testloader, valloader


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
