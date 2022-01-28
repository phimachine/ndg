# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
from pathlib import Path

import numpy as np
import time

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from logger import Logger
from TransFG.models.modeling import VisionTransformer, CONFIGS
from TransFG.models.actv_modeling import VisionTransformerACTV, VisionTransformerACTV2
from TransFG.utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from TransFG.utils.data_utils import get_loader
from TransFG.utils.dist_util import get_world_size
import torch.multiprocessing as mp
from global_params import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def save_model(args, model, epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    logger = args.logger
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    # torch.save(checkpoint, model_checkpoint)
    # logger.log_print("Saved model checkpoint to [DIR: %s]", args.output_dir)
    logger.save_pickle(checkpoint, epoch)


def setup(args, head_hidden_size=None):
    # Prepare model
    config = CONFIGS[args.model_type]
    if head_hidden_size is not None:
        config.head_hidden_size = head_hidden_size
    else:
        config.head_hidden_size = config.hidden_size
    config.split = args.split
    config.slide_step = args.slide_step
    if "logger" in args:
        logger = args.logger
    else:
        logger = None

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    else:
        raise

    if args.actv == 2:
        model = VisionTransformerACTV2(config, args.img_size, zero_head=True, num_classes=num_classes,
                                       smoothing_value=args.smoothing_value)
    elif args.actv:
        model = VisionTransformerACTV(config, args.img_size, zero_head=True, num_classes=num_classes,
                                      smoothing_value=args.smoothing_value)
    else:
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,
                                  smoothing_value=args.smoothing_value)

    # model.load_from(np.load(args.pretrained_dir))
    model.load_from(np.load(Path(args.pretrained_dir) / f"{args.model_type}.npz"))  # ['model']
    # model.load_state_dict(pretrained_model)
    args.device = 0
    model.to(args.device)
    num_params = count_parameters(model)
    if logger:
        logger.log_print("{}".format(config))
        logger.log_print("Training parameters %s", args)
        logger.log_print("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.world_size > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger = args.logger
    logger.log_print("***** Running Validation *****")
    logger.log_print(f"  Num steps = {len(test_loader)}", )
    logger.log_print(f"  Batch size = {args.eval_batch_size}", )

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0],
                          position=0,
                          leave=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            y = y.long()
            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    if args.local_rank != -1:
        dist.barrier()
    val_accuracy = reduce_mean(accuracy, args.world_size)
    val_accuracy = val_accuracy.detach().cpu().numpy()

    # logger.log_print("\n")
    logger.auto_log("validation", global_step=global_step, loss=eval_losses.avg, accuracy=val_accuracy)
    # logger.log_print("Validation Results")
    # logger.log_print("Global Steps: %d" % global_step)
    # logger.log_print("Valid Loss: %2.5f" % eval_losses.avg)
    # logger.log_print("Valid Accuracy: %2.5f" % val_accuracy)

    return val_accuracy


def train(args, model):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    scaler = GradScaler()
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank])

    # Train!
    logger = args.logger
    logger.log_print("***** Running training *****")
    logger.log_print("  Total optimization steps = %d", args.num_steps)
    logger.log_print("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.log_print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                     args.train_batch_size * args.gradient_accumulation_steps * (
                         torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.log_print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    epoch = 0
    # valid(args, model, test_loader, global_step)
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            y = y.long()

            loss, logits = model(x, y)
            loss = loss.mean()

            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            # loss = scaler.scale(loss)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            # scaler.step(optimizer)
            # scaler.update()

            losses.update(loss.item() * args.gradient_accumulation_steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )
            # if args.local_rank in [-1, 0]:
            #     writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
            #     writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
            if global_step % args.eval_every == args.eval_every - 1:
                with torch.no_grad():
                    accuracy = valid(args, model, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model, epoch)
                        best_acc = accuracy
                    logger.auto_log("best", global_step=global_step, loss=losses.val, acc=best_acc)
                model.train()

            if global_step % t_total == 0:
                break
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)
        if args.local_rank != -1:
            dist.barrier()
        train_accuracy = reduce_mean(accuracy, args.nprocs)
        train_accuracy = train_accuracy.detach().cpu().numpy()
        logger.auto_log("training", global_step=global_step, loss=losses.val, acc=train_accuracy)
        losses.reset()
        epoch += 1
        if global_step % t_total == 0:
            break

    logger.log_print("Best Accuracy: \t%f" % best_acc)
    logger.log_print("End Training!")
    end_time = time.time()
    logger.log_print("Total Training Time: \t%f" % ((end_time - start_time) / 3600))


def train2(args, info):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    # train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.learning_rate,
    #                             momentum=0.9,
    #                             weight_decay=args.weight_decay)
    scaler = GradScaler()
    t_total = args.num_steps
    # if args.decay_type == "cosine":
    #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # else:
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    scheduler = info.scheduler
    optimizer = info.optimizer
    train_loader, test_loader = info.dataloaders["train"], info.dataloaders["test"]
    model = info.model
    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank])

    # Train!
    logger = args.logger
    logger.log_print("***** Running training *****")
    logger.log_print("  Total optimization steps = %d", args.num_steps)
    logger.log_print("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.log_print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                     args.train_batch_size * args.gradient_accumulation_steps * (
                         torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.log_print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    epoch = 0
    # valid(args, model, test_loader, global_step)
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            y = y.long()

            loss, logits = model(x, y)
            loss = loss.mean()

            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            # loss = scaler.scale(loss)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            # scaler.step(optimizer)
            # scaler.update()

            losses.update(loss.item() * args.gradient_accumulation_steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )
            # if args.local_rank in [-1, 0]:
            #     writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
            #     writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
            if global_step % args.eval_every == args.eval_every - 1:
                with torch.no_grad():
                    accuracy = valid(args, model, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model, epoch)
                        best_acc = accuracy
                    logger.auto_log("best", global_step=global_step, loss=losses.val, acc=best_acc)
                model.train()

            if global_step % t_total == 0:
                break
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)
        if args.local_rank != -1:
            dist.barrier()
        train_accuracy = reduce_mean(accuracy, args.nprocs)
        train_accuracy = train_accuracy.detach().cpu().numpy()
        logger.auto_log("training", global_step=global_step, loss=losses.val, acc=train_accuracy)
        losses.reset()
        epoch += 1
        if global_step % t_total == 0:
            break

    logger.log_print("Best Accuracy: \t%f" % best_acc)
    logger.log_print("End Training!")
    end_time = time.time()
    logger.log_print("Total Training Time: \t%f" % ((end_time - start_time) / 3600))


def get_parser():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--set_name", default="transfg",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--version", default="first",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"],
                        default="CUB_200_2011",
                        help="Which dataset.")
    data_root = Path(data_source_path)
    parser.add_argument('--data_root', type=str, default=data_root)
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default=data_root / "vit",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    # parser.add_argument("--output_dir", default="D:\Git\cache\\transfg", type=str,
    #                     help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--gpus_per_node", type=int, default=1,
                        help="gpus per node for distributed training on gpus")
    parser.add_argument("--world_size", type=int, default=1,
                        help="world_size for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=49,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--actv", action="store_true")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args


def main(local_rank=-1, args=None, actv=False, version="default", head_hidden_size=None):
    if args is None:
        args = get_args()
    args.local_rank = local_rank

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='gloo',
                                             timeout=timedelta(minutes=60),
                                             rank=args.local_rank,
                                             world_size=args.world_size)
    args.device = device
    # I'm disabling my 1080, only 3090
    # args.nprocs = torch.cuda.device_count()
    args.nprocs = 1
    args.actv = actv
    args.version = version
    # Setup logging
    from global_params import save_dir
    logger = Logger(save_dir, args.local_rank, args.set_name, args.version, disabled=args.local_rank not in (0, -1))
    logger.log_print("Process rank: %s, device: %s, world_size: %s, distributed training: %s, 16-bits training: %s" %
                     (args.local_rank, args.device, args.world_size, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)
    args.logger = logger

    # Model & Tokenizer Setup
    args, model = setup(args, head_hidden_size)
    # Training
    train(args, model)


def spawn(actv=True, version="actv"):
    """
    For non-command-line entry

    Parameters
    ----------
    args

    Returns
    -------

    """
    args = get_args()
    args.gpus_per_node = 1
    args.nodes = 1
    args.world_size = args.gpus_per_node * args.nodes  #
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "50011"
    # context = mp.get_context("spawn")
    # ret_q = context.Queue()
    mp.start_processes(main, nprocs=args.gpus_per_node,
                       args=(args, actv, version),
                       start_method="spawn")  #
    # for _ in range(args.gpus_per_node):
    #     ret = ret_q.get()  # only get one timestamp
    # ret_q.close()
    # return ret


if __name__ == "__main__":
    spawn(actv=True, version="actv")
