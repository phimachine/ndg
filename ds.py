"""
Five datasets are provided with a unified interface
MNIST, CUB200, NLI, MQNLI, Code
"""
import csv
import gzip
import json
import os
import pickle
import sys

import requests
import torch.utils.data.dataset
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import datasets as tvdatasets
from torchvision.transforms import transforms
from global_params import project_root, data_source_path
from TransFG.utils.data_utils import get_loader, CUB
import datasets
from datasets import load_dataset


class DS(Dataset):
    split_ratio = 10

    def __init__(self):
        super(DS, self).__init__()
        self.ds = None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        return self.ds[item]

    @staticmethod
    def collate(*args):
        return args


class MNIST(DS):
    def __init__(self, split="train"):
        super(MNIST, self).__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if split in ("train", "valid"):
            self.ds = tvdatasets.MNIST(data_source_path, train=True, download=True,
                                       transform=transform)
            l = len(self.ds)
            train, valid = data.random_split(self.ds, [l - l // DS.split_ratio, l // DS.split_ratio])
            self.ds = train if split == "train" else valid
        else:
            self.ds = tvdatasets.MNIST(data_source_path, train=False,
                                       transform=transform)


class MNISTParity(DS):
    def __init__(self, split="train"):
        super(MNISTParity, self).__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if split in ("train", "valid"):
            self.ds = tvdatasets.MNIST(data_source_path, train=True, download=True,
                                       transform=transform)
            l = len(self.ds)
            train, valid = data.random_split(self.ds, [l - l // DS.split_ratio, l // DS.split_ratio])
            self.ds = train if split == "train" else valid
        else:
            self.ds = tvdatasets.MNIST(data_source_path, train=False,
                                       transform=transform)

    def __getitem__(self, item):
        i, o = self.ds[item]
        o = o % 2
        return i, o


class NLI(DS):
    def __init__(self, split="train"):
        super(NLI, self).__init__()

        if split == 'valid':
            split = 'validation'

        self.ds: datasets.Dataset = load_dataset("snli")[split]
        self.ds = self.ds.filter(lambda d: d['label'] != -1)


class MQNLI(DS):
    def __init__(self, split="train", ratio=0.25):
        super(MQNLI, self).__init__()
        assert ratio in [0, 0.0625, 0.125, 0.25, 0.5, 0.75]
        self.data_dir = project_root / "cauabs/mqnli"
        if split == "valid":
            split = "val"
        self.data_file = self.data_dir / f"{ratio}gendata.{split}"
        self.ds = self.read_data()

    def read_data(self):
        ds = []
        with self.data_file.open('r') as f:
            for line in f:
                d = json.loads(line)
                ds.append(tuple(d.values()))
        return ds


class Code(DS):
    def __init__(self, split="train"):
        super(Code, self).__init__()
        if split == 'valid':
            split = 'validation'
        self.ds = load_dataset("code_x_glue_cc_defect_detection")[split]


class CUB200(DS):
    def __init__(self, split="train"):
        super(CUB200, self).__init__()
        self.split = split
        self.data_root = data_source_path / "CUB_200_2011"
        cp = self.get_cache_path()
        try:
            with cp.open('rb') as f:
                self.ds = pickle.load(f)
        except FileNotFoundError:
            self.build_cache()
        if isinstance(self.ds, torch.utils.data.dataset.Subset):
            if not hasattr(self.ds.dataset.transform.transforms[0], "max_size"):
                self.build_cache()
        else:
            if not hasattr(self.ds.transform.transforms[0], "max_size"):
                self.build_cache()

    def build_cache(self):
        if self.split in ("train", "valid"):
            train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                                  transforms.RandomCrop((448, 448)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.ds = CUB(root=self.data_root, is_train=True, transform=train_transform)
            l = len(self.ds)
            train, valid = data.random_split(self.ds, [l - l // DS.split_ratio, l // DS.split_ratio])
            self.ds = train if self.split == "train" else valid
        elif self.split == "test":
            test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                                 transforms.CenterCrop((448, 448)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.ds = CUB(root=self.data_root, is_train=False, transform=test_transform)
        cp = self.get_cache_path()
        with cp.open('wb') as f:
            pickle.dump(self.ds, f)

    def get_cache_path(self):
        p = self.data_root / f"CUB200.{self.split}"
        return p


class SST2(DS):
    def __init__(self, split="train"):
        super(SST2, self).__init__()
        if split == 'valid':
            split = 'validation'
        self.ds = load_dataset("gpt3mix/sst2")[split]


class AllNLI(DS):
    train_samples = None
    dev_samples = None

    def __init__(self, split='train'):
        super(AllNLI, self).__init__()
        nli_dataset_path = data_source_path / 'datasets/AllNLI.tsv.gz'

        if not nli_dataset_path.exists():
            http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
        if AllNLI.train_samples is None:
            label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
            train_samples = []
            dev_samples = []
            with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
                reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
                for row in reader:
                    label_id = label2int[row['label']]
                    dat = {"premise": row['sentence1'], "hypothesis": row['sentence2'], "label": label_id}
                    if row['split'] == 'train':
                        train_samples.append(dat)
                    else:
                        dev_samples.append(dat)
            AllNLI.train_samples = train_samples
            AllNLI.dev_samples = dev_samples

        if split in ("train", "valid"):
            self.ds = AllNLI.train_samples
            l = len(self.ds)
            train, valid = data.random_split(self.ds, [l - l // DS.split_ratio, l // DS.split_ratio])
            self.ds = train if split == "train" else valid
        else:
            self.ds = AllNLI.dev_samples


def http_get(url, path):
    """
    Downloads a URL to a given path on disk
    """
    path = str(path)
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


if __name__ == '__main__':
    cub = CUB200()
    print(cub[20])
