import math
import time
from datetime import datetime

from TransFG.utils.scheduler import WarmupCosineSchedule
from models import *
from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import argparse
import numpy as np

from ds import *
from global_params import save_dir
from helpers import accuracy
from logger import Logger, MultiAverageMeter, MasterLog
import torch
from sentence_transformers import SentenceTransformer


class InfoMan:
    def __init__(self, args: Namespace, timestamp=None, no_model=False, no_dataset=False):
        self.no_model = no_model
        self.no_dataset = no_dataset
        self.args = vars(args)
        if self.secondary:
            self.seed = self.seed * 11
            if "_secondary" not in self.version:
                self.version = self.version + "_secondary"
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.args["experiment_name"] = self.dataset_name
        self.device = torch.device(self.local_rank if self.local_rank != -1 else 0)
        self.timestamp = timestamp
        self.datasets = None
        self.dataloaders = None
        self.split = None
        self.model = None
        self.logger = None
        self.loss_fn = None
        self.batch_size = None
        self.splits = {"train", "valid", "test"}
        self.epoch = None
        self.iteration = None
        self.always_logged = {}
        self.start_time = None
        self.max_performance = -float('inf')
        self.is_counterfactual = False
        self.max_batches = float('inf')

        if self.dataset_name == "mnist":
            self.mnist()
        elif self.dataset_name == "cub200":
            self.cub200()
        elif self.dataset_name == "nli":
            self.nli()
        elif self.dataset_name == "allnli":
            self.allnli()
        elif self.dataset_name == "mqnli":
            self.mqnli()
        elif self.dataset_name == "code":
            self.code()
        elif self.dataset_name == "sst2":
            self.sst2()
        elif self.dataset_name == "mnist2":
            self.parity()
        else:
            raise NotImplementedError
        if not self.no_model:
            self.model = self.model.to(self.device)

    def common_init(self, named=False):
        self.logger = Logger(save_dir, self.local_rank, self.experiment_name, self.version, self.timestamp,
                             master_log=True, disabled=self.disable_logger)
        if not self.no_model:
            if self.optimizer == "adamw":
                if hasattr(self.model, "new_params") and self.model.new_params():
                    freeze_pretrained = []
                    s = set(self.model.new_params())
                    for n, p in self.model.named_parameters():
                        if p not in s:
                            freeze_pretrained.append(p)
                    groups = [{
                        'params': list(self.model.new_params()),
                        'lr': (self.learning_rate or 1e-3) * self.freeze_pretrained_factor
                    }, {
                        'params': freeze_pretrained,
                        'lr': (self.learning_rate or 1e-3) / self.freeze_pretrained_factor
                    }]
                    self.optimizer = torch.optim.AdamW(groups)
                else:
                    self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                                       lr=self.learning_rate or 1e-3)
            else:
                if hasattr(self.model, "new_params") and self.model.new_params():
                    freeze_pretrained = []
                    s = set(self.model.new_params())
                    for n, p in self.model.named_parameters():
                        if p not in s:
                            freeze_pretrained.append(p)
                    groups = [{
                        'params': list(self.model.new_params()),
                        'lr': (self.learning_rate or 3e-2) * self.freeze_pretrained_factor
                    }, {
                        'params': freeze_pretrained,
                        'lr': (self.learning_rate or 3e-2) / self.freeze_pretrained_factor
                    }]
                    self.optimizer = torch.optim.AdamW(groups)
                else:
                    self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                     lr=self.learning_rate or 3e-2,
                                                     momentum=0.9,
                                                     weight_decay=0)
        if not self.no_dataset:
            self.dataloaders = {s: DataLoader(self.datasets[s], num_workers=self.num_workers,
                                              batch_size=self.batch_size, shuffle=self.shuffle,
                                              collate_fn=self.collate_fn) for s in self.splits}
        warmup = self.total_epochs * self.warmup + 1
        if not self.no_model:

            if self.scheduler == "triangular":
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda step: max((self.total_epochs - step) / (self.total_epochs - warmup),
                                               0) if step + 1 > warmup or warmup == 0 else (step + 1) / warmup
                )
            elif self.scheduler == "cosine":
                self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=warmup, t_total=self.total_epochs)
            elif self.scheduler == "warmup_linear":
                warmup_steps = math.ceil(
                    len(self.dataloaders["train"]) * self.total_epochs * 0.1)  # 10% of train data for warm-up
                self.scheduler = SentenceTransformer._get_scheduler(self.optimizer, scheduler="WarmupLinear",
                                                                    warmup_steps=warmup_steps,
                                                                    t_total=warmup_steps * 10)

        config = {"batch_size": self.batch_size, }
        config.update(self.args)
        self.logger.master_log.collection_write("master", "config", config)
        self.logger.log_print(f"Running {self.dataset_name}")
        self.timestamp = self.logger.timestamp
        self.start_time = time.time()

    def make_dataloader(self, split, **kwargs):
        k = {"num_workers": self.num_workers,
             "batch_size": self.batch_size,
             "shuffle": self.shuffle,
             "collate_fn": self.collate_fn}
        k.update(kwargs)
        if split != "train":
            k["shuffle"] = False
        dl = DataLoader(self.datasets[split], **k)
        return dl

    def reset_dataloaders(self, **kwargs):
        for split in self.splits:
            self.dataloaders[split] = self.make_dataloader(split, **kwargs)

    def mnist(self):
        if not self.no_dataset:
            self.datasets = {s: MNIST(s) for s in self.splits}
        if not self.no_model:
            self.model = MNISTM(32)
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor

        def collate_fn(batch):
            data, target = default_collate(batch)
            data, target = data.to(self.device), target.to(self.device)
            return data, target

        self.collate_fn = collate_fn
        self.common_init()

    def parity(self):
        if not self.no_dataset:
            self.datasets = {s: MNISTParity(s) for s in self.splits}
        if not self.no_model:
            self.model = MNISTParityM(32)
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor
        self.total_epochs = 2

        def collate_fn(batch):
            data, target = default_collate(batch)
            data, target = data.to(self.device), target.to(self.device)
            return data, target

        self.collate_fn = collate_fn
        self.common_init()

    def cub200(self):
        if not self.no_dataset:
            self.datasets = {s: CUB200(s) for s in self.splits}
        if not self.no_model:
            self.model = CUB200Model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor // 4
        self.total_epochs //= 3

        def collate_fn(batch):
            data, target = default_collate(batch)
            target = target.long()
            data, target = data.to(self.device), target.to(self.device)
            return data, target

        self.collate_fn = collate_fn
        self.common_init()

    def sst2(self):
        if not self.no_dataset:
            self.datasets = {s: SST2(s) for s in self.splits}
        if not self.no_model:
            self.model = Sentiment()
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor

        def collate_fn(batch):
            t = [b['text'] for b in batch]
            data = self.model.tokenizer(t,
                                        return_tensors='pt', padding='max_length',
                                        max_length=self.max_length)
            target = [b["label"] for b in batch]
            target = torch.tensor(target, dtype=torch.long).to(
                self.device)
            data = data.to(self.device)
            return data, target

        self.collate_fn = collate_fn
        self.common_init()

    def mqnli(self):
        if not self.no_dataset:
            self.datasets = {s: MQNLI(s) for s in self.splits}
        if not self.no_model:
            self.model = NLIModel()
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor // 2

        def collate_fn(batch):
            a, b, l = zip(*batch)
            data = self.model.tokenizer(a, b,
                                        return_tensors='pt', padding='max_length',
                                        max_length=self.max_length)
            target = l
            new_target = []
            for t in target:
                if t == "neutral":
                    new_target.append(1)
                elif t == "entailment":
                    new_target.append(0)
                elif t == "contradiction":
                    new_target.append(2)
                else:
                    raise NotImplementedError
            target = torch.Tensor(new_target).long()
            data, target = data.to(self.device), target.to(self.device)
            return data, target

        self.collate_fn = collate_fn
        self.common_init()

    def nli(self):
        if not self.no_dataset:
            self.datasets = {s: NLI(s) for s in self.splits}
        if not self.no_model:
            self.model = NLIModel()
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor * 2
        self.total_epochs //= 4

        def collate_fn(batch):
            texts = [[b["premise"] for b in batch], [b["hypothesis"] for b in batch]]
            labels = [b["label"] for b in batch]
            data = self.model.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt",
                                        max_length=self.max_length)
            target = torch.tensor(labels, dtype=torch.long).to(
                self.device)
            data, target = data.to(self.device), target.to(self.device)
            return data, target

        self.collate_fn = collate_fn
        self.common_init()

    def allnli(self):
        if not self.no_dataset:
            self.datasets = {s: AllNLI(s) for s in self.splits}
        if not self.no_model:
            self.model = ALLNLIModel()
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor
        self.total_epochs = 4

        def collate_fn(batch):
            texts = [[b["premise"] for b in batch], [b["hypothesis"] for b in batch]]
            labels = [b["label"] for b in batch]
            tokenized = self.model.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt",
                                             max_length=self.max_length)
            labels = torch.tensor(labels, dtype=torch.long).to(
                self.device)

            for name in tokenized:
                tokenized[name] = tokenized[name].to(self.device)
            return tokenized, labels

        self.collate_fn = collate_fn
        self.common_init(named=True)

    def code(self):
        if not self.no_dataset:
            self.datasets = {s: Code(s) for s in self.splits}
        if not self.no_model:
            self.model = CodeModel()
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor // 2
        self.total_epochs //= 2

        def collate_fn(batch):
            f, t = [b["func"] for b in batch], [b["target"] for b in batch]
            data = self.model.tokenizer.batch_encode_plus(f,
                                                          return_tensors='pt', padding=True,
                                                          max_length=self.max_length, truncation=True)
            target = torch.tensor(t, dtype=torch.long).to(self.device)
            data = data.to(self.device)
            return data, target

        self.collate_fn = collate_fn
        self.common_init()

    def counterfactual(self, alignment, counterfactual_datasets):
        """
        Train the pipe from the latent to the prediction

        Replace model to be the last layers
        Replace dataset to be the counter_facutal datasets
        Loggers are not reused, but the old timestamp should be logged to associate them
        """
        self.datasets = counterfactual_datasets
        self.model = alignment.model.latent_to_pred
        self.always_logged.update({"cf_main_timestamp": alignment.info.timestamp})
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = self.batch_size_factor * 16
        self.is_counterfactual = True

        def collate_fn(batch):
            data, target = zip(*batch)
            data = torch.Tensor(data)
            target = torch.Tensor(target).long()
            data, target = data.to(self.device), target.to(self.device)
            return data, target

        self.collate_fn = collate_fn
        self.common_init()

    def auto_log(self, **kwargs):
        kwargs.update({
            "dataset": self.dataset_name,
            "eta": self.get_eta(),
        })
        master_col_vals = {"experiment": self.logger.exp_name,
                           "version": self.logger.version,
                           "timestamp": self.logger.timestamp,
                           "savedir": str(self.logger.save_dir)}
        master_col_vals.update(self.always_logged)
        self.logger.auto_log(prepend=self.split, time_key="iter", master_col_vals=master_col_vals, **kwargs)

    def get_eta(self):
        elapsed = time.time() - self.start_time
        iters_so_far = len(self.dataloaders["train"]) * self.epoch + self.iteration
        total_iters = len(self.dataloaders["train"]) * self.total_epochs
        if iters_so_far == 0:
            return f"hr:min"

        sec_per_iter = elapsed / iters_so_far
        remaining = sec_per_iter * (total_iters - iters_so_far)
        minutes = int(remaining // 60)
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}:{minutes}"

    def save_checkpoint(self, performance=None):
        # save model, dump log to json
        assert self.split == "valid"
        if performance > self.max_performance:
            self.max_performance = performance
            is_best = True
        else:
            is_best = False
        payload = {"model_state_dict": self.model.state_dict(),
                   "optimizer_state_dict": self.optimizer.state_dict(),
                   "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None}
        self.logger.save_pickle(payload,
                                epoch=self.epoch, iteration=self.iteration, is_best=is_best)

    def load_checkpoint(self):
        payload, epoch, iteration = self.logger.load_pickle()
        self.epoch, self.iteration = epoch, iteration
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])

    @property
    def dataloader(self):
        return self.dataloaders[self.split]

    def train(self):
        self.model.train()
        self.split = "train"

    def eval(self):
        self.model.eval()
        self.split = "valid"

    def test(self):
        self.model.eval()
        self.split = "test"

    def __getattr__(self, item):
        if item in self.args:
            return self.args[item]
        else:
            raise AttributeError


def get_best_timestamp(args, return_document=False):
    dataset_name, version = args.dataset_name, args.version
    master = MasterLog(dataset_name, version)
    col = master.collection
    res = col.find({"prepend": "valid"})
    res = res.sort("acc", -1)
    res = list(iter(res))
    for r in res:
        timestamp = r["timestamp"]
        if not timestamp_banned(timestamp):
            best_acc = r
            break
    if return_document:
        return timestamp, best_acc
    else:
        return timestamp


def timestamp_banned(timestamp):
    ban = ["12_05_21_37_26"]
    if timestamp in ban:
        return True
    earliest = datetime(2021, 12, 6)
    # parse time
    parsed = datetime.strptime(timestamp, "%m_%d_%H_%M_%S")
    parsed = parsed.replace(year=2021)
    if parsed > earliest:
        return False
    else:
        return True


def run_one_epoch(info: InfoMan):
    i = 0
    mam = MultiAverageMeter()
    info.optimizer.zero_grad()
    for batch in info.dataloader:
        data, target = batch
        if info.debug:
            if i > 10:
                break
        out = info.model(data)
        loss = info.loss_fn(out, target)
        acc = accuracy(out, target)
        if info.split == "train":
            info.iteration = i
            loss.backward()
            if info.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(info.model.parameters(), info.max_grad_norm)
            info.optimizer.step()
            info.optimizer.zero_grad()
            if info.scheduler and info.batch_scheduler_step:
                info.scheduler.step()
            if i > info.max_batches:
                break
        mam.update(loss=loss.item(), acc=acc)
        if i % info.logging_interval == 0 and info.split == 'train':
            info.auto_log(epoch=info.epoch, iter=i, **mam.get())
        i += 1
    if info.split == "train" and info.scheduler and not info.batch_scheduler_step:
        info.scheduler.step()
    info.auto_log(epoch=info.epoch, iter=i, **mam.get())
    return mam.get()["acc"]


def train_one_epoch(info):
    info.train()
    run_one_epoch(info)


def valid_one_epoch(info):
    info.eval()
    with torch.no_grad():
        acc = run_one_epoch(info)
    info.save_checkpoint(performance=acc)


def infer_one_epoch(info):
    info.test()
    with torch.no_grad():
        run_one_epoch(info)


def run_all_epochs(info, epochs=None):
    for epoch in range(epochs or info.total_epochs):
        info.epoch = epoch
        train_one_epoch(info)
        if epoch % info.eval_interval == 0:
            valid_one_epoch(info)
        if info.debug and epoch > 1:
            break
    valid_one_epoch(info)
    infer_one_epoch(info)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    """
    Args will be the entry to modify all args
    Rest of the args will be combined and dumped to disk as json

    """
    from transformers import logging
    logging.set_verbosity_error()
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = ArgumentParser("Neural dependency graph causal abstraction experiments")
    parser.add_argument('--dataset_name', type=str,
                        choices=['mnist', 'cub200', 'code', "mnist2",
                                 'mqnli', 'sst2', 'allnli'],
                        default='mnist',
                        help='Name of the dataset.')
    parser.add_argument("--debug", type=str2bool, default=False, help="activate debug mode")
    parser.add_argument("--total_epochs", type=int, default=20, help="total epochs for training")
    parser.add_argument("--eval_interval", type=int, default=10, help="evaluate every k epoch ")
    parser.add_argument("--learning_rate", type=float, default=0, help="learning rate")
    # parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--batch_size_factor", type=int, default=32, help="rough batch size, changed dynamically")
    parser.add_argument("--warmup", type=float, default=0.1, help="fraction of epochs for warmup")
    parser.add_argument("--seed", type=int, default=49, help="seed")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="shuffle dataset")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler type")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm. 0 to turn off")
    parser.add_argument("--extreme", type=str2bool, default=False, help="Make the logits extreme and far from zero")
    parser.add_argument("--batch_scheduler_step", type=str2bool, default=False, help="Step scheduler after batch")
    parser.add_argument("--freeze_pretrained_factor", type=float, default=1, help="")

    # nlp
    parser.add_argument("--max_length", type=int, default=512, help="dataloader workers")

    # exp management
    parser.add_argument("--timestamp", type=str, default=None, help="load the timestamped data")
    parser.add_argument('--experiment_name', type=str, default="default", help="Name the experiment. For logging")
    parser.add_argument('--version', type=str, default="v1", help="Name the version. For logging")
    parser.add_argument('--logging_interval', type=int, default=100, help="Logging interval")
    parser.add_argument('--disable_logger', type=str2bool, default=False, help="Disable logger")

    # distributed
    parser.add_argument('--master_addr', type=str, default="localhost", help="Distributed training: master address")
    parser.add_argument('--master_port', type=str, default="8886", help="Distributed training: master port")
    parser.add_argument('--gpus_per_node', type=int, default=4, help="Distributed training: gpus per node")
    parser.add_argument('--local_rank', type=int, default=-1, help="Distributed training: local rank")
    parser.add_argument('--node_rank', type=int, default=0, help="Distributed training: node rank")

    # ndg
    parser.add_argument('--alpha', type=float, default=20, help="Threshold parameter")
    parser.add_argument('--max_vector_input', type=int, default=1e5,
                        help="The number of data points needed to make the graph")  # this is around one minute
    parser.add_argument('--bidi_recall_thres', type=float, default=0.99,
                        help="Using bidrectional propagation, recall threshold, 1 for strongly connected, "
                             "0 for weakly connected"
                             "beta in the paper")  # this is around one minute
    parser.add_argument('--criterion', type=int, default=4, help="NDG criterion")

    # triviality
    parser.add_argument('--trivial_random_model', type=str2bool, default=False,
                        help="Triviality experiment: random model")
    parser.add_argument('--trivial_random_dataset', type=str2bool, default=False,
                        help="Triviality experiment: random dataset")

    # inter
    parser.add_argument('--inter_layer', type=str2bool, default=False,
                        help="Inter layer")
    parser.add_argument('--inter_model', type=str2bool, default=False,
                        help="Inter model")
    parser.add_argument('--secondary', type=str2bool, default=False,
                        help="Secondary model. Trained with different seeds")
    return parser


def get_configs():
    disabled = {"mqnli": "--dataset_name mqnli --optimizer sgd --scheduler cosine",
                "nli": "--dataset_name nli --optimizer sgd --scheduler cosine",
                }
    configs = {"mnist": "--dataset_name mnist --optimizer adamw --alpha 100 --total_epochs 10 ",
               "mnist2": "--dataset_name mnist2 --optimizer adamw --alpha 100 ",
               "sst2": "--dataset_name sst2 --optimizer adamw --learning_rate 1e-5 "
                       "--total_epochs 50 --max_length 128 --eval_interval 50 --alpha 100 ",
               "allnli": "--dataset_name allnli --version v2 --scheduler warmup_linear --optimizer adamw "
                         "--learning_rate 2e-5 --batch_scheduler_step True ",
               "cub200": "--dataset_name cub200 --version v2 --scheduler warmup_linear --optimizer adamw "
                         "--learning_rate 2e-5 --batch_scheduler_step True --freeze_pretrained_factor 3 "
                         "--alpha 3  ",
               "code": "--dataset_name code --version v2 --scheduler warmup_linear --optimizer adamw "
                       "--learning_rate 2e-5 --batch_scheduler_step True "}
    # configs = {k: v for k, v in configs.items() if k in ("sst2", "cub200")}
    for k, v in configs.items():
        configs[k] += " --max_num_batches_main_epoch 5000 "
    return configs


def train_all_models():
    configs = get_configs()
    ##
    enabled = None # ["cub200"]
    ##
    for dataset in configs:
        if enabled and dataset not in enabled:
            continue
        args = get_parser().parse_args(args=configs[dataset].split())
        info = InfoMan(args)
        run_all_epochs(info)


def train_secondary():
    configs = get_configs()
    enabled = None # ["cub200"]
    for dataset in configs:
        if enabled and dataset not in enabled:
            continue
        args = get_parser().parse_args(args=configs[dataset].split())
        args.secondary = True
        info = InfoMan(args)
        run_all_epochs(info)


def main():
    train_all_models()


if __name__ == '__main__':
    train_all_models()
    train_secondary()

"""
Time budget:
MNIST: 5 minutes
SST2: 1 hour
ALLNLI: 2 hours
CUB200: 0.5 hour
Code: 1 hour
"""
