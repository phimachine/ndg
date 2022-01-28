import time

from train_models import get_parser, InfoMan, run_all_epochs
import pytest
import torch


@pytest.fixture(autouse=True)
def run_around_tests():
    time.sleep(5)
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()
    time.sleep(5)


def test_mnist():
    args = get_parser().parse_args(args="")
    args.debug = True
    info = InfoMan(args)
    run_all_epochs(info)


def test_cub200():
    args = get_parser().parse_args(args="")
    args.dataset_name = "cub200"
    args.debug = True
    info = InfoMan(args)
    run_all_epochs(info)


def test_allnli():
    args = get_parser().parse_args(args="")
    args.dataset_name = "allnli"
    args.debug = True
    info = InfoMan(args)
    run_all_epochs(info)

def test_code():
    args = get_parser().parse_args(args="")
    args.dataset_name = "code"
    args.debug = True
    info = InfoMan(args)
    run_all_epochs(info)


def test_sst2():
    args = get_parser().parse_args(args="")
    args.dataset_name = "sst2"
    args.debug = True
    info = InfoMan(args)
    run_all_epochs(info)
