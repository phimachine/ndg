import pytest

from train_models import get_parser, InfoMan, get_configs
from align import Alignment
import torch
import time


@pytest.fixture(autouse=True)
def run_around_tests():
    time.sleep(2)
    yield
    torch.cuda.empty_cache()
    time.sleep(2)


@pytest.mark.parametrize("dataset", get_configs().keys())
def test_hook(dataset):
    configs = get_configs()
    args = configs[dataset]
    args = get_parser().parse_args(args=args.split())
    args.debug = True
    info = InfoMan(args)
    align = Alignment(info)
    dataloader = align.info.dataloaders["train"]

    for i, batch_data in enumerate(dataloader):
        batch_data, batch_target = batch_data
        pred = align.info.model(batch_data.to(align.info.device))
        batch_latent = align.hook.get_latent().float()
        original_latent_to_pred = align.model.latent_to_pred(batch_latent)
        pred_cls = pred.max(dim=1)[1]
        lat_pred_cls = original_latent_to_pred.max(dim=1)[1]
        # assert (pred_cls == lat_pred_cls).all()
        if i == 10:
            break


@pytest.mark.parametrize("dataset", get_configs().keys())
def test_intervention(dataset):
    configs = get_configs()
    args = configs[dataset]
    args = get_parser().parse_args(args=args.split())
    args.debug = True
    info = InfoMan(args)
    info.split = "train"
    align = Alignment(info)
    align.build_graph()
    stats, ds = align.interchange_intervention()
