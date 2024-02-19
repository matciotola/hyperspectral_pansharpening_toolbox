import os.path
import torch
from torch.utils.data import Dataset
import yaml
from recordclass import recordclass

from Utils.load_save_tools import open_mat


def normalize(tensor):
    return tensor / (2 ** 16)


def denormalize(tensor):
    return tensor * (2 ** 16)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def open_config(file_path):
    yaml_file = read_yaml(file_path)
    return recordclass('config', yaml_file.keys())(*yaml_file.values())


def generate_paths(root, dataset, type, resolution):

    ds_paths = []
    names = sorted(next(os.walk(os.path.join(root, dataset, type, resolution)))[2])

    for name in names:
        ds_paths.append(os.path.join(root, dataset, type, resolution, name))

    return ds_paths


class TrainingDatasetRR(Dataset):
    def __init__(self, img_paths, norm):
        super(TrainingDatasetRR, self).__init__()

        pan = []
        ms_lr = []
        ms = []
        gt = []

        for i in range(len(img_paths)):
            pan_single, ms_lr_single, ms_single, gt_single, _, _ = open_mat(img_paths[i])
            pan.append(pan_single.float())
            ms_lr.append(ms_lr_single.float())
            ms.append(ms_single.float())
            gt.append(gt_single.float())

        pan = torch.cat(pan, 0)
        ms_lr = torch.cat(ms_lr, 0)
        ms = torch.cat(ms, 0)
        gt = torch.cat(gt, 0)

        pan = norm(pan)
        ms_lr = norm(ms_lr)
        ms = norm(ms)
        gt = norm(gt)

        self.pan = pan
        self.ms_lr = ms_lr
        self.ms = ms
        self.gt = gt

    def __len__(self):
        return self.pan.shape[0]

    def __getitem__(self, index):
        return self.pan[index], self.ms_lr[index], self.ms[index], self.gt[index]


class TrainingDatasetFR(Dataset):
    def __init__(self, img_paths, norm):
        super(TrainingDatasetFR, self).__init__()

        pan = []
        ms_lr = []
        ms = []

        for i in range(len(img_paths)):
            pan_single, ms_lr_single, ms_single, _, _, _ = open_mat(img_paths[i])
            pan.append(pan_single.float())
            ms_lr.append(ms_lr_single.float())
            ms.append(ms_single.float())

        pan = torch.cat(pan, 0)
        ms_lr = torch.cat(ms_lr, 0)
        ms = torch.cat(ms, 0)

        pan = norm(pan)
        ms_lr = norm(ms_lr)
        ms = norm(ms)

        self.pan = pan
        self.ms_lr = ms_lr
        self.ms = ms

    def __len__(self):
        return self.pan.shape[0]

    def __getitem__(self, index):
        return self.pan[index], self.ms_lr[index], self.ms[index]

