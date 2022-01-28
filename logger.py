import csv
import datetime
import logging
import shutil
import subprocess
from pathlib import Path

import torch.distributed as dist
import torch
from tensorboardX import SummaryWriter

import os
from collections import OrderedDict


class Logger:
    """
    Use the logger to output csv of training statistics, save/load models
    The logger creates a unique directory so you can run experiments in parallel.
    """

    def __init__(self, save_dir, local_rank=-1, experiment_name="default", version="v0",
                 timestamp=None,
                 disabled=False,
                 tensorboard=True, master_log=False):
        """

        :param experiment_name:
        :param version:
        :param disabled: disable if rank is not 0 or -1
        :param tensorboard:
        """
        self.exp_name = experiment_name
        self.version = version
        self.local_rank = local_rank
        if master_log:
            self.master_log: MasterLog = MasterLog(experiment_name, version)
        else:
            self.master_log = None
        if local_rank != -1:
            self.world_size = dist.get_world_size()
        else:
            self.world_size = None
        self.disabled = disabled or local_rank not in (0, -1)
        self.logger = None

        # the overall saves directory is parallel to the project directory
        # this is so that my IDE does not reindex the files

        # take a look at the file structure defined here
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists() and not self.disabled:
            self.save_dir.mkdir()

        self.set_dir = self.save_dir / self.exp_name
        if not self.set_dir.exists() and not self.disabled:
            self.set_dir.mkdir()

        self.version_dir = self.set_dir / self.version
        if not self.version_dir.exists() and not self.disabled:
            self.version_dir.mkdir()

        if timestamp is None:
            self.timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        else:
            self.timestamp = timestamp
        self.exp_dir = self.version_dir / self.timestamp
        if not self.exp_dir.exists() and not self.disabled:
            self.exp_dir.mkdir()

        self.model_check_points_dir = self.exp_dir / "checkpoints"
        if not self.model_check_points_dir.exists() and not self.disabled:
            self.model_check_points_dir.mkdir()

        self.cache_dir = self.exp_dir / "cache"
        if not self.cache_dir.exists() and not self.disabled:
            self.cache_dir.mkdir()

        self.log_dir = self.exp_dir / "log"
        if not self.log_dir.exists() and not self.disabled:
            self.log_dir.mkdir()

        self.csv_dir = self.exp_dir / "csv"
        if not self.csv_dir.exists() and not self.disabled:
            self.csv_dir.mkdir()

        self.log_file = self.new_log_file_path()
        self.csv_field_names = {}
        if tensorboard and not self.disabled:
            self.make_writer()
        else:
            self.writer: SummaryWriter = None

        self.last_prepend = None

    @property
    def path(self):
        return self.exp_dir

    def make_writer(self):
        self.writer = SummaryWriter(self.path / "tensorboard")
        return self.writer

    def use_py_logger(self, py_logger):
        self.logger = py_logger

    def auto_log(self, prepend, time_key=None, csv_name=None, replace=True, tensorboard=True, master_col_vals=None,
                 **column_values):
        """
        Overwrites the log
        Unlikely though, since the directories are timestamped
        :param prepend:
        :param csv_name:
        :param replace:
        :param: the column, values
        :return:
            The distributed reduced logging_dict
        """

        if not self.disabled:
            # csv
            if csv_name is None:
                csv_name = self.auto_csv_name(prepend)

            columns, values = column_values.keys(), column_values.values()
            column_value_dict = column_values

            if csv_name not in self.csv_field_names:
                self.new_csv(csv_name, columns, replace=replace)

            self.csv_write_row(csv_name, column_value_dict)

            # mongodb
            if self.master_log:
                cv = column_values.copy()
                cv["timestamp"] = self.timestamp
                if master_col_vals:
                    cv.update(master_col_vals)
                self.master_log.write(prepend, time_key, **cv)

        if prepend != self.last_prepend:
            self.log_print(f"{'======' + prepend + '======':^60}".upper())
            self.last_prepend = prepend

        headers = "|"
        values = "|"
        for i, (col, val) in enumerate(column_values.items()):
            wid = len(col) if len(col) > 8 else 8
            if isinstance(val, int):
                val_str = f"{val:{wid}}|"
            else:
                if isinstance(val, float):
                    pval = val
                elif isinstance(val, torch.Tensor):
                    val = val.clone().detach()
                    if self.local_rank != -1:
                        dist.all_reduce(val)
                        val = val / self.world_size
                        column_values[col] = val
                    val = val.item()
                    pval = val
                elif isinstance(val, AverageMeter):
                    val = val.avg
                    pval = val
                elif isinstance(val, str):
                    pval = val
                else:
                    try:
                        val = float(val)
                        pval = val
                    except TypeError:
                        val = str(val)
                        pval = val

                if isinstance(pval, float) or isinstance(pval, int):
                    if pval < 1e-3 or pval > 1e3:
                        val_str = f"{pval:{wid}.2E}|"
                    else:
                        val_str = f"{pval:{wid}.4f}|"
                else:
                    val_str = f"{pval:>{wid}}|"

            if tensorboard and not self.disabled:
                if col != time_key:
                    if isinstance(val, float) or isinstance(val, int):
                        if time_key is None:
                            self.writer.add_scalar(prepend + '/' + col, val)
                        else:
                            self.writer.add_scalar(prepend + '/' + col, val, column_values[time_key])
                    else:
                        if time_key is None:
                            self.writer.add_text(prepend + '/' + col, val)
                        else:
                            self.writer.add_text(prepend + '/' + col, val, column_values[time_key])

                    self.writer.flush()

            values += val_str
            headers += f"{col:>{wid}}|"

            if (i + 1) % 8 == 0:
                self.log_print(headers)
                self.log_print(values)
                headers = "|"
                values = "|"
                # headers = f"{prepend}||"
                # values = f"{'':{len(prepend)}}||"

        if len(column_values) % 8 != 0:
            self.log_print(headers)
            self.log_print(values)

        return column_values

    def load_pickle(self, timestamp=None, starting_epoch=None, starting_iteration=None, map_location=None,
                    save_dir=None):
        """
        Do not load individual models. All optimizers/models/auxiliary objects need to be saved/loaded at
        the same time.
        :param starting_epoch: You can specify the epoch/iteration of model you want to load.
        :param starting_iteration:
        :return:
        """
        timestamp = timestamp or self.timestamp
        highest_epoch = 0
        highest_iter = 0
        if save_dir is not None:
            time_stamp_dir = Path(save_dir) / timestamp / "checkpoints"
        else:
            time_stamp_dir = self.version_dir / timestamp / "checkpoints"

        for child in time_stamp_dir.iterdir():
            if "_".join(child.name.split("_")[:-2]) == self.version:
                try:
                    epoch = child.name.split("_")[-2]
                    iteration = child.name.split("_")[-1].split('.')[0]
                except IndexError:
                    print(str(child))
                    raise
                iteration = int(iteration)
                epoch = int(epoch)
                # some files are open but not written to yet.
                if child.stat().st_size > 128:
                    if epoch > highest_epoch or (iteration > highest_iter and epoch == highest_epoch):
                        highest_epoch = epoch
                        highest_iter = iteration
        # if highest_epoch == 0 and highest_iter == 0:
        #     print("nothing to load")
        #     return None

        if starting_epoch is None and starting_iteration is None:
            # load the highest epoch, iteration
            pickle_file = time_stamp_dir / (
                    self.version + "_" + str(highest_epoch) + "_" + str(highest_iter) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                payload = torch.load(pickle_file, map_location=map_location)
            print('Loaded model at epoch ', highest_epoch, 'iteration', highest_iter)
        else:
            if starting_iteration is None:
                starting_iteration = 0
            pickle_file = time_stamp_dir / (
                    self.version + "_" + str(starting_epoch) + "_" + str(starting_iteration) + ".pkl")
            if not pickle_file.exists():
                raise FileNotFoundError("The model checkpoint does not exist")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                payload = torch.load(pickle_file, map_location=map_location)
            print('Loaded model at epoch ', starting_epoch, 'iteration', starting_iteration)

        return payload, highest_epoch, highest_iter

    def save_pickle(self, payload, epoch, iteration=0, is_best=False, prefix=""):
        if not self.disabled:
            epoch = int(epoch)
            pickle_file = self.model_check_points_dir / (
                    prefix + self.version + "_" + str(epoch) + "_" + str(iteration) + ".pkl")
            with pickle_file.open('wb') as fhand:
                torch.save(payload, fhand)
            if is_best:
                pickle_file = self.model_check_points_dir / (
                        prefix + "best_" + self.version + "_" + str(epoch) + "_" + str(iteration) + ".pkl")
                with pickle_file.open('wb') as fhand:
                    torch.save(payload, fhand)

            self.log_print(f"Saved model set {self.exp_name} version {self.version} at {pickle_file}")

    def new_log_file_path(self):
        """
        Does not make the file.
        :return:
        """
        log_file = self.log_dir / "log.txt"
        self.log_file = log_file
        return log_file

    def log_print(self, string, print_it=True):
        if self.logger is not None:
            self.logger.info(string)
        else:
            if not self.disabled:
                string = str(string)
                if self.log_file is not None and self.log_file is not None:
                    with open(self.log_file, 'a') as handle:
                        handle.write(string + '\n')
            if print_it:
                print(string)

    def new_csv(self, csv_name, field_names, replace=False):
        """
        :param csv_name: the name of the csv
        :param field_names:
        :param replace: replace the csv file
        :return:
        """
        # safety checks
        csv_path = self.csv_dir / csv_name
        if not csv_path.suffix == ".csv":
            raise ValueError("use .csv file names")
        if csv_path.exists():
            if not replace:
                raise FileExistsError("the csv file already exists")
            else:
                os.remove(csv_path)

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()

        self.csv_field_names[csv_name] = field_names

    def csv_write_row(self, csv_name, column_values_dict):
        """
        :param csv_name:
        :param column_values_dict: a dictionary column : value
        :return:
        """

        csv_path = self.csv_dir / csv_name
        if not csv_path.suffix == ".csv":
            raise ValueError("use .csv file names")
        if not csv_path.exists():
            raise FileNotFoundError("The csv you are writing to is not registered")

        try:
            field_names = self.csv_field_names[csv_name]
            for key in column_values_dict.keys():
                assert key in field_names
            for fn in field_names:
                assert fn in column_values_dict
        except AssertionError:
            return

        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writerow(column_values_dict)

    def auto_csv_name(self, prepend):
        # make csv_name automatically
        words = prepend.split()
        csv_name = ""
        for word in words:
            csv_name += word[:4].lower()
        return csv_name + ".csv"

    def close(self):
        if self.writer:
            self.writer.close()


def print_file(filename):
    with open(filename, "r") as handle:
        for line in handle:
            print(line)


class MultiAverageMeter:
    def __init__(self, reset_after_get=False, local_rank=-1):
        self.dict = OrderedDict()
        self.reset_after_get = reset_after_get
        self.local_rank = local_rank
        if self.local_rank != -1:
            self.world_size = dist.get_world_size()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    value = value.clone().detach()
                if self.local_rank != -1:
                    dist.all_reduce(value)
                    value = value / self.world_size
            if key not in self.dict:
                self.dict[key] = AverageMeter()
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass
            am = self.dict[key]
            am.update(value)

    def get(self):
        ret = OrderedDict()
        for key, val in self.dict.items():
            ret[key] = val.avg
        if self.reset_after_get:
            self.reset()
            return ret
        else:
            return ret

    def reset(self):
        self.dict = OrderedDict()


class AverageMeter:
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.M2 = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        delta = val - self.avg
        self.avg = self.sum / self.count
        delta2 = val - self.avg
        self.M2 += delta * delta2

    def __float__(self):
        return self.avg

    def avg(self):
        return self.avg

    @property
    def var(self):
        return self.M2 / (self.count - 1)


from pymongo import MongoClient

"""
Explicitly made to manage experiments
"""


class MasterLog:
    def __init__(self, experiment_name, version, addr="localhost:27017"):
        self.addr = addr
        self.connection = MongoClient(addr)
        self.db = self.connection[experiment_name]
        self.collection = self.db[version]

    def write(self, prepend, time_key=None, **column_values):
        document = column_values.copy()
        document.update({"prepend": prepend,
                         "log_time": datetime.datetime.now()})
        self.collection.insert_one(document)

    def collection_write(self, db, collection, document):
        self.connection[db][collection].insert_one(document)
