import multiprocessing
import random
from copy import deepcopy
from typing import List

import matplotlib
import numpy as np
import ray
from tqdm import tqdm
from transformers import BatchEncoding
import torch.nn.functional as F

from array_graph import ArrayGraph
from train_models import InfoMan, get_parser, get_configs, get_best_timestamp
from ds import *
import torch.nn as nn
from models import CallbackLatentHook, multi_layer_hack
from global_params import save_dir

from logger import MultiAverageMeter, Logger
import torch


class ZipDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        ds_len = len(self.datasets[0])
        for ds in self.datasets:
            assert len(ds) == ds_len
        self.ds_len = ds_len

    def __len__(self):
        return self.ds_len

    def __getitem__(self, item):
        ret = []
        for ds in self.datasets:
            ret.append(ds[item])
        return tuple(ret)


class CounterFactualDataset(Dataset):
    def __init__(self, cf_latents, cf_targets):
        super(CounterFactualDataset, self).__init__()
        self.cf_latents = cf_latents
        self.cf_targets = cf_targets
        assert len(self.cf_targets) == len(self.cf_latents)

    def __len__(self):
        return len(self.cf_targets)

    def __getitem__(self, item):
        return self.cf_latents[item].astype("float32"), self.cf_targets[item]


def reset_params(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class Alignment:
    def __init__(self, info: InfoMan):
        self.info: InfoMan = info
        # the main info model
        self.model = info.model
        self.array_graph = None
        self.drop_last = False

        if not info.no_model:
            if not info.inter_layer and not info.inter_model:
                self.hook = CallbackLatentHook(self.model)
                self.model.eval()
            elif info.inter_layer:
                try:
                    self.hook = MultiHook([
                        CallbackLatentHook(self.model,
                                           layer=self.model.multi_latent[i],
                                           latent_to_bool=self.model.multi_latent_to_bool[i]) for i in range(2)])
                except AttributeError:
                    ml, mltb = multi_layer_hack(self.model)
                    self.hook = MultiHook([
                        CallbackLatentHook(self.model,
                                           layer=ml[i],
                                           latent_to_bool=mltb[i]) for i in range(2)])
                self.model.eval()
            elif info.inter_model:
                raise NotImplementedError("Use MultiAlignment")

    def build_graph(self, verbose=True, id=None, count_array=None, num_latent=None, num_labels=None):
        if count_array is None:
            loader = self.info.make_dataloader(self.info.split, shuffle=False, drop_last=self.drop_last)
            device = self.info.device
            max_i = self.info.max_vector_input // self.info.batch_size
            max_i = min(max_i, len(loader))
            self.array_graph = None
            if self.info.trivial_random_model:
                self.model.apply(reset_params)
            with torch.no_grad():
                pbar = tqdm(loader, total=max_i)
                refs = []
                for i, batch in enumerate(pbar):
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    if self.info.trivial_random_dataset:
                        if isinstance(data, BatchEncoding):
                            data["input_ids"] = torch.randint_like(data["input_ids"],
                                                                   high=data["input_ids"].max()) * data[
                                                    "attention_mask"]
                        else:
                            data.uniform_()
                    logits = self.model(data)
                    preds = torch.argmax(logits, dim=-1)
                    num_labels = logits.shape[-1]
                    preds = F.one_hot(preds, num_labels).cpu()

                    boo = self.hook.get_boo()

                    num_latent = boo.shape[-1]

                    if self.array_graph is None:
                        self.array_graph = ArrayGraph(num_latent, labels=num_labels,
                                                      alpha=self.info.alpha,
                                                      criterion=self.info.criterion)

                    nodes = torch.cat((preds, boo), dim=1)
                    ref = self.array_graph.parallel_vector_input(nodes)
                    refs.append(ref)
                    if len(refs) > multiprocessing.cpu_count() * 2:
                        ray.wait(refs, num_returns=i - multiprocessing.cpu_count() * 2 + 1)
                    if i > max_i:
                        break

            self.array_graph.get_parallel_count()
        else:
            self.array_graph = ArrayGraph(num_latent, labels=num_labels,
                                          alpha=self.info.alpha,
                                          criterion=self.info.criterion)
            self.array_graph.count = count_array
        self.array_graph.get_graph(with_negation=True, verbose=verbose)
        with self.get_graph_pickle_path(id).open('wb') as f:
            pickle.dump(self.array_graph, f)
        print(f"Criterion used : {self.array_graph.criterion}")
        return self.array_graph

    def inter_layer_edges_shapes(self):
        loader = self.info.make_dataloader(self.info.split, shuffle=False, drop_last=self.drop_last)
        device = self.info.device
        max_i = self.info.max_vector_input // self.info.batch_size
        max_i = min(max_i, len(loader))
        self.array_graph = None
        if self.info.trivial_random_model:
            self.model.apply(reset_params)
        with torch.no_grad():
            pbar = tqdm(loader, total=max_i)
            refs = []
            for i, batch in enumerate(pbar):
                data, target = batch
                data, target = data.to(device), target.to(device)
                if self.info.trivial_random_dataset:
                    if isinstance(data, BatchEncoding):
                        data["input_ids"] = torch.randint_like(data["input_ids"],
                                                               high=data["input_ids"].max()) * data["attention_mask"]
                    else:
                        data.uniform_()
                logits = self.model(data)

                shapes = []
                for hook in self.hook.hooks:
                    shapes.append(hook.saved_output.shape)
                return shapes

    def build_graph_reuse_counts(self, id=None, verbose=True):
        with self.get_graph_pickle_path(id).open('rb') as f:
            array_graph = pickle.load(f)
        new_array_graph = deepcopy(array_graph)
        new_array_graph.alpha = self.info.alpha
        new_array_graph.criterion = self.info.criterion
        new_array_graph.get_graph(with_negation=True, verbose=verbose)
        with self.get_graph_pickle_path(id).open('wb') as f:
            pickle.dump(self.array_graph, f)
        print(f"Criterion used : {self.array_graph.criterion}")

    def load_graph(self, id=None) -> ArrayGraph:
        try:
            path = self.get_graph_pickle_path(id)
            print("Loading graph from: " + str(path))
            with path.open('rb') as f:
                graph = pickle.load(f)
            print(f"Criterion used : {graph.criterion}")
            return graph
        except FileNotFoundError:
            return self.build_graph(id=id)

    def get_graph_pickle_path(self, id=None):
        name = f"{self.info.split}_{self.info.criterion}_{self.info.alpha}"
        if id:
            name += f"_{id}"
        if self.info.trivial_random_model:
            name += "_trivial_random_model"
        if self.info.trivial_random_dataset:
            name += "_trivial_random_dataset"
        if self.info.inter_layer:
            name += "_inter_layer"
        if self.info.inter_model:
            name += "inter_model"
        name += "_graph.pkl"
        return self.info.logger.cache_dir / name

    def get_counterfactual_dataset_path(self):
        return self.info.logger.cache_dir / f"{self.info.split}_counterfactual_dataset.pkl"

    def interchange_intervention(self, graph=None, type=0, max_i=float('inf')):
        graph: ArrayGraph = graph or self.load_graph()
        reverse_graph = graph.g.reverse()
        dataloader = self.info.make_dataloader(self.info.split, shuffle=False)

        mam = MultiAverageMeter()
        counterfactual_dataset = []

        necessary_cache = {}
        num_labels = graph.labels
        with torch.no_grad():
            for i, batch_data in tqdm(enumerate(dataloader), total=min(max_i, len(dataloader))):
                if i > max_i:
                    break

                batch_data, batch_target = batch_data
                cf_targets = []

                original_pred = self.info.model(batch_data)
                batch_latent = self.hook.get_latent()
                if self.info.debug and i == 10:
                    break
                negs = []
                poss = []
                bl = batch_latent.to("cpu")
                bt = batch_target.to("cpu")
                contra = 0
                for target, latent in zip(bt, bl):
                    target_node = (True, target.item())

                    alternative_target = list(range(0, graph.labels))
                    alternative_target.remove(target)
                    alternative_target = random.choice(alternative_target)
                    cf_targets.append(alternative_target)
                    alternative_node = (True, alternative_target)

                    desc_contra = set()
                    if type == 0:
                        # as seen in the body of the paper, the Q function
                        for n in (alternative_node,):
                            if n in necessary_cache:
                                necessary = necessary_cache[n]
                            else:
                                ancestors, _, _ = \
                                    graph.bidirectional_propagate_all({n}, self.info.bidi_recall_thres, reverse_graph)
                                necessary, _, _ = \
                                    graph.bidirectional_propagate_all(ancestors, self.info.bidi_recall_thres)
                                necessary_cache[n] = necessary
                            desc_contra.update(necessary)
                    elif type == 1:
                        # definition G.1: the first alternative
                        # the minimal set for logically reasonable
                        desc_contra = set()
                        # for n in (alternative_node, (False, target_node[1])):
                        for n in (alternative_node,):
                            if n in necessary_cache:
                                necessary = necessary_cache[n]
                            else:
                                necessary, _, _ = \
                                    graph.bidirectional_propagate_all({n}, self.info.bidi_recall_thres)
                                necessary_cache[n] = necessary
                            desc_contra.update(necessary)

                        # strong ii turns off necessary conditions of target
                        # strong ii turn on all sufficient conditions of alternative targets
                        negs = set()
                        # for n in ((False, alternative_target), (True, target_node[1])):
                        for n in ((True, target_node[1]),):
                            if n in necessary_cache:
                                necessary = necessary_cache[n]
                            else:
                                necessary, _, _ = \
                                    graph.bidirectional_propagate_all({n}, self.info.bidi_recall_thres)
                                necessary_cache[n] = necessary
                            negs.update(necessary)
                        for n in negs:
                            desc_contra.add((not n[0], n[1]))
                    elif type == 2:
                        # Definition G.2: the second alternative
                        for n in (alternative_node,):
                            if n in necessary_cache:
                                necessary = necessary_cache[n]
                            else:
                                necessary, _, _ = \
                                    graph.bidirectional_propagate_all({n}, self.info.bidi_recall_thres)
                                necessary_cache[n] = necessary
                            desc_contra.update(necessary)
                    else:
                        raise NotImplementedError

                    desc = set()
                    for n in desc_contra:
                        if (not n[0], n[1]) in desc_contra:
                            contra += 1
                        else:
                            desc.add(n)

                    val, mask = necessary_of_target_to_val_mask(desc, latent, num_labels)
                    poss.append((val, mask))

                contra /= len(bt)
                contra /= 2
                d = torch.stack([n[0] for n in poss], dim=0)
                replace_mask = torch.stack([m[1] for m in poss], dim=0)

                batch_counterfactuals = self.modify_bool_latent_tensor(bl, d, replace_mask)
                batch_counterfactuals = batch_counterfactuals.float().detach().to(self.info.device)

                for cf, at in zip(batch_counterfactuals, cf_targets):
                    counterfactual_dataset.append((cf, at))
                cf = self.model.bool_to_latent(batch_counterfactuals)
                counter_output = self.model.latent_to_pred(cf).cpu()
                cf_pred = counter_output.max(dim=1)[1]
                original_pred = original_pred.cpu().float()
                original_pred = original_pred.max(dim=1)[1]
                cf_targets = torch.Tensor(cf_targets).long()
                bt = bt.cpu()
                original_correct = original_pred == bt
                counterfactual_aligned = (cf_pred == cf_targets) * original_correct
                counterfactual_changed_unaligned = (cf_pred != bt) * (
                        cf_pred != cf_targets) * original_correct
                counterfactual_unchanged = (cf_pred == bt) * original_correct

                mam.update(counterfactual_aligned=counterfactual_aligned.float().mean().item(),
                           counterfactual_changed_unaligned=counterfactual_changed_unaligned.float().mean().item(),
                           counterfactual_unchanged=counterfactual_unchanged.float().mean().item(),
                           original_correct=original_correct.float().mean().item(),
                           contradictions=contra)
        avgs = mam.get()
        avgs.update({"dataset_size": len(self.info.datasets[self.info.split])})
        inputs, targets = zip(*counterfactual_dataset)
        inputs = [i.cpu().numpy() for i in inputs]
        inputs = np.stack(inputs, axis=0)
        targets = np.array(targets)
        counterfactual_dataset = CounterFactualDataset(inputs, targets)
        with self.get_counterfactual_dataset_path().open('wb') as f:
            pickle.dump(counterfactual_dataset, f)
        return avgs, counterfactual_dataset

    def get_counterfactual_datasets(self):
        old_split = self.info.split
        datasets = {s: None for s in self.info.splits}
        for s in datasets:
            self.info.split = s
            with self.get_counterfactual_dataset_path().open('rb') as f:
                datasets[s] = pickle.load(f)
        self.info.split = old_split
        return datasets

    def negate_nodes(self, nodes):
        new_nodes = []
        for node in nodes:
            new_nodes.append((not node[0], node[1]))
        return new_nodes

    def modify_bool_latent(self, latent, true_nodes, num_labels):
        latent = latent.detach().clone()
        for node in true_nodes:
            if node[1] > num_labels:
                latent[node[1] - num_labels] = node[0]
        return latent

    def modify_bool_latent_tensor(self, latent: torch.BoolTensor, val: torch.BoolTensor, replace_mask: torch.BoolTensor,
                                  negate=False):
        latent = latent.detach().clone()
        if negate:
            val = ~val

        # erase replaced
        latent = latent * ~ replace_mask

        # add the replaced values
        latent = latent + replace_mask * val
        return latent

    def make_counterfactual_dataset(self) -> CounterFactualDataset:
        pass

    def mix_counterfactual_dataset(self, cf_dataset: CounterFactualDataset):
        pass


def necessary_of_target_to_val_mask(necessary_of_target, latent, num_labels):
    val = torch.zeros_like(latent)
    mask = torch.zeros_like(latent)
    for node in necessary_of_target:
        if node[1] > num_labels:
            idx = node[1] - num_labels
            val[idx] = node[0]
            mask[idx] = 1
    return val, mask


def batch_encoding_slice(be, f, t):
    be = BatchEncoding({"input_ids": be.data['input_ids'][f:t],
                        "attention_mask": be.data['attention_mask'][f:t]},
                       be.encodings[f:t])
    return be


class MultiModel(nn.Module):
    def __init__(self, models, different_inputs=False):
        super(MultiModel, self).__init__()
        self.different_inputs = different_inputs
        self.models = models
        self.random_input = None

    def forward(self, input):
        if not self.random_input:
            outputs = []
            if self.different_inputs:
                if isinstance(input, BatchEncoding):
                    bs = input["input_ids"].shape[0]
                    for i, m in enumerate(self.models):
                        outputs.append(m(batch_encoding_slice(input, i * bs // 2, (i + 1) * bs // 2)))
                else:
                    bs = input.shape[0]
                    for i, m in enumerate(self.models):
                        outputs.append(m(input[i * bs // 2: (i + 1) * bs // 2]))
            else:
                for m in self.models:
                    outputs.append(m(input))
            return torch.cat(outputs, dim=1)
        else:
            outputs = []
            for idx, m in enumerate(self.models):
                if idx != 0:
                    if isinstance(input, BatchEncoding):
                        input["input_ids"] = torch.randint_like(input["input_ids"],
                                                                high=input["input_ids"].max()) * input["attention_mask"]
                    else:
                        input = input.clone().detach()
                        input.uniform_()
                outputs.append(m(input))
            return torch.cat(outputs, dim=1)

    def set_random(self, *args):
        self.random_input = args


class MultiHook:
    def __init__(self, hooks: List[CallbackLatentHook]):
        self.hooks = hooks

    def get_boo(self):
        lats = []
        for hook in self.hooks:
            lats.append(hook.get_boo())
        lats = torch.cat(lats, dim=-1)
        return lats


class MultiAlignment(Alignment):
    def __init__(self, infos: List[InfoMan], different_inputs=False):
        self.infos = infos
        self.info = self.infos[-1]
        self.model = MultiModel([i.model for i in infos], different_inputs)
        self.drop_last = True

        if not self.info.no_model:
            self.hook = MultiHook([CallbackLatentHook(self.model.models[0]),
                                   CallbackLatentHook(self.model.models[1])])
            self.model.eval()

    def get_graph_pickle_path(self, id=None):
        name = f"{self.info.split}_{self.info.criterion}_{self.info.alpha}"
        if id:
            name += f"_{id}"
        if self.info.trivial_random_model:
            name += "_trivial_random_model"
        if self.info.trivial_random_dataset:
            name += "_trivial_random_dataset"
        if self.info.inter_layer:
            name += "_inter_layer"
        if self.info.inter_model:
            name += "inter_model"
        name += "_graph_mam.pkl"
        return self.info.logger.cache_dir / name

    def set_random(self, *args):
        assert len(args) == len(self.infos)
        self.model.set_random(*args)


def align(args=None, align_on_split="train", uni=True, build_valid=False, build_test=False,
          interchange_intervention=True, reuse_counts=False, type=0, max_i=float('inf')):
    if args:
        args = get_parser().parse_args(args.split())
    else:
        args = get_parser().parse_args()
    args.shuffle = False
    if uni:
        args.bidi_recall_thres = 1
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.load_checkpoint()
    align = Alignment(info)
    if not reuse_counts:
        train_graph = align.build_graph()
    else:
        try:
            train_graph = align.build_graph_reuse_counts()
        except:
            train_graph = align.build_graph()
    if build_test:
        info.split = "test"
        align = Alignment(info)
        if not reuse_counts:
            align.build_graph()
        else:
            try:
                align.build_graph_reuse_counts()
            except:
                align.build_graph()
        info.split = "train"
    if build_valid:
        info.split = "valid"
        align = Alignment(info)
        if not reuse_counts:
            align.build_graph()
        else:
            try:
                align.build_graph_reuse_counts()
            except:
                align.build_graph()
        info.split = "train"
    info.split = align_on_split
    if interchange_intervention:
        stats, ds = align.interchange_intervention(train_graph, type=type, max_i=max_i)
        return stats, ds, args


def align_load_graph(args, on_split="train", uni=False, interchange_intervention=True, type=0, max_i=float('inf')):
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    if uni:
        args.bidi_recall_thres = 1
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.load_checkpoint()
    align = Alignment(info)
    try:
        graph = align.load_graph()
    except FileNotFoundError:
        graph = align.build_graph()
    info.split = on_split
    if interchange_intervention:
        stats, ds = align.interchange_intervention(graph, type=type, max_i=max_i)
    return stats, ds, args


def all_alignment_results(enable=None, load_graph=True, on_split="train", uni=True,
                          interchange_intervention=True, type=0, max_i=float('inf'), version="alignment_results8",
                          build_valid=True, build_test=True, reuse_counts=True):
    logger = Logger(save_dir, -1, "general", version, master_log=True)
    configs = get_configs()
    if enable:
        configs = {k: w for k, w in configs.items() if k in enable}
    for ds, args in configs.items():
        if load_graph:
            stats, _, args = align_load_graph(args, on_split, uni=uni,
                                              interchange_intervention=interchange_intervention, type=type, max_i=max_i)
        else:
            stats, _, args = align(args, on_split, uni=uni, build_valid=build_valid, build_test=build_test,
                                   interchange_intervention=interchange_intervention, type=type, max_i=max_i,
                                   reuse_counts=True)
        stats.update({"dataset": ds})
        stats.update(vars(args))
        logger.auto_log("alignment_stats", unidirectional=uni, validation=on_split, type=type, max_i=max_i, **stats)


def plot_color_map():
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    # https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
    cmaps = {}

    gradient = np.linspace(0, 1, 101)
    gradient = np.vstack((gradient, gradient))
    cmap_list = ["plasma_r"]
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh + 0.35))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.05, right=0.97)
    axs[0].set_title(f'sample proportion (percentage)', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        # ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
        #         transform=ax.transAxes)
    axs[1].set_axis_off()
    axs[0].get_yaxis().set_visible(False)
    axs[0].get_xaxis().set_ticks([0, 20, 40, 60, 80, 100])

    save_dir = project_root
    plot_path = save_dir / "figures"
    plt.savefig(plot_path / "colormap.png")
    plt.show()


def plot_color_map_2():
    import matplotlib.pyplot as plt
    for i in [3, 3.5, 4, 6, 8, 10]:
        a = np.array([[0, 1]])
        fig = plt.figure(figsize=(1.1, i))
        plt.subplots_adjust(top=0.975, bottom=0.025,
                            left=0.05, right=0.4)
        img = plt.imshow(a, cmap="plasma_r")
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.025, 0.2, 0.95])
        cax.tick_params(labelsize=15)
        bar = plt.colorbar(orientation="vertical", cax=cax)
        bar.ax.set_ylabel("sample proportion", rotation=270, labelpad=25, fontsize=23)
        plt.savefig(project_root / "figures" / f"colorbar_v_{i}.png", bbox_inches='tight')
        plt.show()


def plot_graph(dataset, omit_equiv=False, prior_color=False, load_graph=True):
    configs = get_configs()
    args = configs[dataset]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.load_checkpoint()
    align = Alignment(info)
    if load_graph:
        graph = align.load_graph()
    else:
        graph = align.build_graph()
    save_dir = project_root
    plot_path = save_dir / "figures" / (dataset + ".png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    dot, g = graph.nice_plot(graph.g, plot_path, save="png", omit_equiv=omit_equiv)
    # for edge in dot.get_edges():
    #     edge.set_penwidth(0.1)
    for node in dot.get_nodes():
        if node.get_color() == "red":
            node.set_fillcolor("#f79186")
            node.set_style('filled')
    plot_path = save_dir / "figures" / (dataset + ".eps")
    dot.write(plot_path, format='eps')
    if prior_color:
        from matplotlib import cm
        max_p = 0
        min_p = 1
        # for node in dot.get_nodes():
        #     p = float(node.obj_dict["attributes"]["prior"])
        #     if p > max_p:
        #         max_p = p
        #     if p < min_p:
        #         min_p = p
        col = cm.get_cmap("plasma")
        for node in dot.get_nodes():
            p = float(node.obj_dict["attributes"]["prior"])
            deg = (p - min_p) / (max_p - min_p)
            c = col(deg)
            hex = matplotlib.colors.to_hex(c)
            node.set_color(hex)
            node.set_penwidth(5)

        plot_path = save_dir / "figures" / (dataset + "_prior.eps")
        dot.write(plot_path, format='eps')
        print(f"{max_p=}, {min_p=}")
    print("Plotted " + dataset)


def plot_all_graphs(**kwargs):
    configs = get_configs()
    for dataset in configs:
        if dataset in ("mnist", "mnist2", "allnli", "code"):
            plot_graph(dataset, omit_equiv=False, **kwargs)
        else:
            pass
            # plot_graph(dataset, omit_equiv=True)


def build_all_graphs(arg_str="", reuse_counts=False):
    configs = get_configs()
    for ds, args in configs.items():
        align(args + " " + arg_str + " ", uni=False, build_valid=True, build_test=True,
              interchange_intervention=False, reuse_counts=reuse_counts)


def inter_layer(dataset="mnist"):
    configs = get_configs()
    args = configs[dataset]
    args = get_parser().parse_args(args.split())
    args.shuffle = False
    best_timestamp = get_best_timestamp(args)
    info = InfoMan(args, timestamp=best_timestamp)
    info.split = "train"
    info.args["inter_layer"] = True
    info.args["max_vector_input"] = 1e5
    info.load_checkpoint()
    align = Alignment(info)
    graph = align.build_graph()
    info.split = "test"
    graph = align.build_graph()


def all_inter_layer(enable=None):
    for dataset in get_configs():
        if enable and dataset not in enable:
            continue
        inter_layer(dataset)


def inter_layer_num_preds():
    shapes = {}
    for dataset in get_configs():
        configs = get_configs()
        args = configs[dataset]
        args = get_parser().parse_args(args.split())
        args.shuffle = False
        best_timestamp = get_best_timestamp(args)
        info = InfoMan(args, timestamp=best_timestamp)
        info.split = "train"
        info.args["inter_layer"] = True
        info.args["max_vector_input"] = 1e5
        info.load_checkpoint()
        align = Alignment(info)
        shape = align.inter_layer_edges_shapes()
        shapes[dataset] = shape
    for ds in shapes:
        shapes[ds] = (shapes[ds][0][-1], shapes[ds][1][-1])
    print(shapes)
    return shapes


if __name__ == '__main__':
    """
    Select the stage you want to run
    """
    all_alignment_results(enable=None, load_graph=False, max_i=float('inf'), build_test=True)
    # all_alignment_results(uni=True, load_graph=True)
    # plot_all_graphs(prior_color=True, load_graph=False)
    # plot_color_map_2()

    # for type in (0, 1, 2):
    #     for split in ("train", "test"):
    #         all_alignment_results(enable=None, load_graph=False, on_split=split, uni=True, max_i=float('inf'),
    #                               build_test=True, type=type, version="rerun3")
    # all_alignment_results(load_graph=True)
