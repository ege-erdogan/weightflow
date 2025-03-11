import sys
from argparse import ArgumentParser

from loguru import logger


def set_logger(level):
    logger.remove()
    logger.add(sys.stderr, level=level)
    return logger


def parse_args(args=sys.argv[1:]):
    parser = ArgumentParser()

    # save/load data
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-path", type=str)
    parser.add_argument(
        "--data-path",
        type=str,
    )
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--setup", type=str)

    parser.add_argument("--n-samples", type=int)

    # experiment setup, reps etc.
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--task", type=str, default="MNIST")
    parser.add_argument("--log-weights-iter", type=int, default=10)
    parser.add_argument("--ignore-epochs", type=int, default=100)
    parser.add_argument("--save-init", action="store_true")
    parser.add_argument("--wandb", type=str)
    parser.add_argument("--config", type=str)

    # general training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--patience", type=float, default=1e99)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--activation", type=str, default="relu")

    # gnn args
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--d-feat", type=int, default=1)
    parser.add_argument("--d-embed", type=int, default=16)
    parser.add_argument("--in-proj-layers", type=int, default=3)
    parser.add_argument("--gnn-backbone", type=str, default="PNA")
    parser.add_argument("--pos-embed-per-node", action="store_true")

    # flow args
    parser.add_argument("--source-std", type=float, default=0.01)
    parser.add_argument("--source-mean-multiply", type=float, default=1.0)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1e-4)
    parser.add_argument("--fm-type", type=str, default="vanilla")
    parser.add_argument("--fm-objective", type=str, default="velocity")
    parser.add_argument("--geometric", action="store_true")
    parser.add_argument("--normalize-pred", action="store_true")
    parser.add_argument("--t-dist", type=str, choices=["uniform", "beta"])

    # target model setup
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[10])

    # logging etc
    parser.add_argument("--log", type=str, default="INFO")

    args = parser.parse_args(args)
    return args
