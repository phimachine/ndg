import pickle
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np

from global_params import *
from models import LatentModel, MNISTSeq1, MNISTSeq2


def array_graph_path(experiment_name, time_stamp=None):
    if time_stamp:
        return save_dir / (experiment_name + "_" + time_stamp + "_ag.pkl")
    else:
        return save_dir / (experiment_name + "_ag.pkl")


def graph_path(experiment_name, likelihood):
    return save_dir / f"{experiment_name}_{likelihood}_graph.pkl"


def model_path(experiment_name):
    return save_dir / (experiment_name + "_model.pkl")


def latest_model_path(experiment_name, seed):
    best_epoch = -1
    for i in save_dir.iterdir():
        if i.is_file():
            name = i.name
            if experiment_name in name:
                try:
                    epoch = int(name.split("_")[-2])
                    if epoch > best_epoch:
                        best_epoch = epoch
                except ValueError:
                    pass
    if best_epoch == -1:
        raise ValueError("This model is not found")
    return model_path(f"{experiment_name}_{seed}_{best_epoch}")


def model_factory(model_type=None, num_predicates=32):
    if model_type == "mnist":
        enc = MNISTSeq1(num_predicates)
        dec = MNISTSeq2(num_predicates)
    else:
        raise ValueError
    model = LatentModel(enc, dec, num_predicates)
    return model


def load_model(dataset, experiment_name, num_predicates, latest_seed=False, model_type=None):
    # if dataset == "mnist":
    #     enc = MNISTEncoder(num_predicates)
    # elif dataset == "cifar":
    #     enc = MobileNetEncoder(num_predicates)
    # else:
    #     raise
    # dec = Decoder(num_predicates)
    # model = Model(enc, dec, num_predicates)
    model_type = dataset if model_type is None else model_type
    model = model_factory(model_type, num_predicates)
    if latest_seed:
        mp = latest_model_path(experiment_name, latest_seed)
    else:
        mp = model_path(experiment_name)
    print(f"Loading model at {mp}")
    with open(mp, "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def save_model(model, experiment_name):
    mp = model_path(experiment_name)
    with open(mp, "wb") as f:
        torch.save(model.state_dict(), f)
    print(f"Saved at {mp}")


def save_arguments(fn):
    """
    Save the parameters, use them as cache
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        pickle.dump((args, kwargs), (Path("debug_cache") / fn.__name__).open("wb"))
        return fn(*args, **kwargs)

    return wrapped


def load_last_arguments(fn):
    @wraps(fn)
    def wrapped():
        args, kwargs = pickle.load((Path("debug_cache") / fn.__name__).open("rb"))
        return fn(*args, **kwargs)

    return wrapped


# this function will run if no cache is found, and load from disk if cache is found
def cached_return_value_for_debug(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        path = cache_dir / "debug_cache" / fn.__name__
        if path.exists():
            res = pickle.load(path.open("rb"))
            return res
        else:
            res = fn(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(res, path.open("wb"))
            return res

    return wrapped


def plot_image_tensor(tensor):
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    plt.imshow(array, cmap='gray')
    plt.show()
    # plt.close()


def discretize(predicates):
    p = predicates.clone().detach()
    p[p > 0.5] = 1
    p[p <= 0.5] = 0
    return p


def binary_accuracy(sigmoid_output, target):
    ret = sigmoid_output * target + (1 - sigmoid_output) * (1 - target)
    ret = ret.mean()
    return ret.item()


def accuracy(output, target):
    if len(output.shape) == 1:
        pred = output > 0.5  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).float().mean().item()
        return correct
    else:
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).float().mean().item()
        return correct


def validate_graph(train_graph, valid_graph):
    """
    Train graph is used.
    Valid data is used, graph ignored.
    """
    precisions = {}
    for edge in train_graph.g.edges:
        a, b = edge
        idx = 0
        if not b[0]:
            idx += 1
        if not a[0]:
            idx += 2
        a_true_b_true = valid_graph.count[a[1], b[1], idx]
        a_true = valid_graph.count[a[1], a[1], 0 if a[0] else 3]
        # if a_true < 3:
        #     # rule of three
        #     continue
        if a_true == 0:
            valid_precision = None
        else:
            valid_precision = a_true_b_true / a_true
        precisions[edge] = (train_graph.get_precision(a, b), valid_precision)
    if len(precisions) > 0:
        train_acc = sum(v[0] for v in precisions.values()) / len(precisions)
        valid_acc = sum(v[1] for v in precisions.values() if v[1] is not None) / len(
            list(v[1] for v in precisions.values() if v[1] is not None))
    else:
        train_acc, valid_acc = 0, 0
    return precisions, len(train_graph.g.edges), train_acc, valid_acc


def change_names(df):
    s = {
        "mnist": "MNIST",
        "mnist2": "MNIST even",
        "sst2": "SST2",
        "allnli": "AllNLI",
        "cub200": "CUB200",
        "code": "Devign"
    }
    return df.rename(index=s)
