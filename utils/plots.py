import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def prep_for_lineplot(ys: list[list]):
    x = np.tile(np.arange(len(ys[0])), len(ys))
    y = np.array(ys).flatten()
    return x, y


def plot_regression_estimates(
    x_true,
    y_true,
    x_pred,
    y_preds,
    size=(4, 3),
    legend=True,
    title=None,
):
    sns.set_theme()
    plt.figure(figsize=size)
    sns.scatterplot(x=x_true, y=y_true, label="True", color="green", alpha=0.5, sizes=1)
    for y_pred in y_preds:
        sns.lineplot(
            x=x_pred.detach().cpu(), y=y_pred, label="Pred", color="red", alpha=0.1
        )
    plt.xlabel(None)
    plt.ylabel(None)
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend(["True", "Pred"])
    else:
        plt.legend("", frameon=False)
    plt.show()


def plot_regression_means_stds(
    x_true,
    y_true,
    x_preds,
    y_preds,
    labels=None,
    size=(4, 3),
    legend=True,
    title=None,
):
    plt.figure(figsize=size)
    sns.scatterplot(
        x=x_true.flatten(), y=y_true.flatten(), label="True", color="green", alpha=0.5
    )

    for i, ys in enumerate(y_preds):
        sns.lineplot(
            x=x_preds.flatten(),
            y=ys.flatten(),
            err_style="band",
            errorbar="sd",
            label=labels[i] if labels is not None else None,
        )

    plt.xlabel(None)
    plt.ylabel(None)
    plt.grid(alpha=0.25)
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend()
    else:
        plt.legend("", frameon=False)
    plt.show()


def plot_train_val_loss(train_loss: list[float], val_loss: list[float] = None):
    plt.figure(figsize=(3, 3))
    plt.grid()
    try:
        plt.plot(train_loss, label="train")
        plt.plot(val_loss, label="val")
    except:
        pass
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def visualize_adj(edge_index: torch.tensor, max_nodes: int):
    adj = torch.zeros(max_nodes, max_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    plt.imshow(adj)
    plt.show()


def plot_hists(
    lists: list,
    labels: list,
    size: tuple,
    bins: int | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    density=False,
    log_x=False,
    log_y=False,
    save=None,
):
    fig = plt.figure(figsize=size)
    plt.grid(alpha=0.5)
    for lst, lab in zip(lists, labels, strict=False):
        plt.hist(lst, density=density, alpha=0.5, bins=bins, label=lab)
        # plt.plot(sorted(lst), label=lab)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.legend()
    if save == None:
        plt.show()
    else:
        plt.savefig(save)


def plot_hists_grid(
    lists: list,
    labels: list,
    size: tuple,
    bins: int | None = "auto",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    density=False,
    ax=None,
    save=None,
    show=True,
):
    # fig = plt.figure(figsize=size)
    plt.grid(alpha=0.5)
    for lst, lab in zip(lists, labels, strict=False):
        sns.histplot(
            data=lst,
            alpha=0.5,
            log_scale=False,
            bins=bins,
            stat="probability",
            label=lab,
            kde=True,
            ax=ax,
        )

    plt.legend(labels)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if save == None and show:
        plt.show()
    if save is not None:
        plt.savefig(save)


def get_dim_weights(lst, dim):
    weights = [w[dim].item() for w in lst]
    return np.array(weights)


def plot_dim_kde(target_samples, mapped_samples, dim_x, dim_y, ax, height=4):
    x_target = get_dim_weights(target_samples, dim_x)
    y_target = get_dim_weights(target_samples, dim_y)
    x_mapped = get_dim_weights(mapped_samples, dim_x)
    y_mapped = get_dim_weights(mapped_samples, dim_y)

    target = np.column_stack((x_target, y_target))
    mapped = np.column_stack((x_mapped, y_mapped))

    combined = np.vstack((target, mapped))
    labels = np.array(["Target"] * len(target) + ["Mapped"] * len(mapped))
    x_data = combined[:, 0]
    y_data = combined[:, 1]

    sns.set_theme()
    sns.kdeplot(x=x_data, y=y_data, hue=labels, palette="tab10", ax=ax)
    ax.set(title=f"{dim_x}-{dim_y}")


def plot_dim_kdes(target_wsos, mapped_wsos, dim_xs, dim_ys):
    assert len(dim_xs) == len(dim_ys)
    n = len(dim_xs)
    target_samples = [wso.flatten() for wso in target_wsos]
    mapped_samples = [wso.flatten() for wso in mapped_wsos]

    fig, axs = plt.subplots(ncols=len(dim_xs), nrows=1, figsize=((n + 1) * 3, 3))
    for i, (dim_x, dim_y) in enumerate(zip(dim_xs, dim_ys, strict=False)):
        plot_dim_kde(target_samples, mapped_samples, dim_x, dim_y, ax=axs.flatten()[i])


def plot_dim_kde_single(wsos, dim_xs, dim_ys, color="blue"):
    flats = [wso.flatten() for wso in wsos]
    n = len(dim_xs)

    fig, axs = plt.subplots(ncols=len(dim_xs), nrows=1, figsize=((n + 1) * 3, 3))
    for i, (dim_x, dim_y) in enumerate(zip(dim_xs, dim_ys, strict=False)):
        xs = get_dim_weights(flats, dim_x)
        ys = get_dim_weights(flats, dim_y)
        vals = np.column_stack((xs, ys))
        sns.set_theme()
        sns.kdeplot(x=vals[:, 0], y=vals[:, 1], ax=axs.flatten()[i], color=color)
        axs.flatten()[i].set(title=f"{dim_x}-{dim_y}")
    plt.show()
