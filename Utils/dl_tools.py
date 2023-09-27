import os.path
import torch
from torch.utils.data import Dataset
import yaml
from recordclass import recordclass

from Utils.load_save_tools import open_mat


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def open_config(file_path):
    yaml_file = read_yaml(file_path)
    return recordclass('config', yaml_file.keys())(*yaml_file.values())


def generate_paths(root, dataset):

    ds_paths = []
    names = sorted(next(os.walk(os.path.join(root, dataset)))[2])

    for name in names:
        ds_paths.append(os.path.join(root, dataset, name))

    return ds_paths


class TrainingDataset20m(Dataset):
    def __init__(self, pan_paths, ms_lr_paths, norm, input_prepro, get_patches, ratio=2, patches_size_lr=33, patch_size_hr=33):
        super(TrainingDataset20m, self).__init__()

        ms_lr = []
        pan = []

        for i in range(len(pan_paths)):
            pan.append(open_mat(pan_paths[i]))
            ms_lr.append(open_mat(ms_lr_paths[i]))

        pan = torch.cat(pan, 0)
        ms_lr = torch.cat(ms_lr, 0)

        pan_downsampled, ms_downsampled, ms_lr = input_prepro(pan, ms_lr, ratio)

        pan_downsampled = norm(pan_downsampled)
        ms_downsampled = norm(ms_downsampled)
        ms_lr = norm(ms_lr)

        self.patches_high_lr = get_patches(pan_downsampled, patch_size_hr)
        self.patches_low_lr = get_patches(ms_downsampled, patches_size_lr)
        self.patches_low = get_patches(ms_lr, patch_size_hr)

    def __len__(self):
        return self.patches_high_lr.shape[0]

    def __getitem__(self, index):
        return self.patches_high_lr[index], self.patches_low_lr[index], self.patches_low[index]


class TrainingDataset60m(Dataset):
    def __init__(self, pan_paths, bands_intermediate_lr_paths, ms_lr_paths, norm, input_prepro,
                 get_patches, ratio=2, patches_size=33):
        super(TrainingDataset60m, self).__init__()

        ms_lr = []
        bands_intermediate_lr = []
        pan = []

        for i in range(len(pan_paths)):
            pan.append(open_mat(pan_paths[i]))
            bands_intermediate_lr.append(open_mat(bands_intermediate_lr_paths[i]))
            ms_lr.append(open_mat(ms_lr_paths[i]))

        pan = torch.cat(pan, 0)
        bands_intermediate_lr = torch.cat(bands_intermediate_lr, 0)
        ms_lr = torch.cat(ms_lr, 0)

        pan_downsampled, bands_intermediate_downsampled, ms_downsampled, ms_lr = input_prepro(
            pan, bands_intermediate_lr, ms_lr, ratio)

        pan_downsampled = norm(pan_downsampled)
        bands_intermediate_downsampled = norm(bands_intermediate_downsampled)
        ms_downsampled = norm(ms_downsampled)
        ms_lr = norm(ms_lr)

        self.patches_high_lr = get_patches(pan_downsampled, patches_size)
        self.patches_intermediate_lr = get_patches(bands_intermediate_downsampled, patches_size)
        self.patches_low_lr = get_patches(ms_downsampled, patches_size)
        self.patches_low = get_patches(ms_lr, patches_size)

    def __len__(self):
        return self.patches_high_lr.shape[0]

    def __getitem__(self, index):
        return self.patches_high_lr[index], self.patches_intermediate_lr, self.patches_low_lr[index], self.patches_low[
            index]

"""
def get_patches(bands, patches_size=33):
    print('list_bands for patches: ' + str(len(bands)))
    patches = []

    _, b, c, r = bands.shape
    cont = 0
    for i in range(0, r - patches_size, patches_size):
        for j in range(0, c - patches_size, patches_size):
            p = bands[:, :, i:i + patches_size, j:j + patches_size]

            patches.append(p)

            cont += 1
    patches = torch.cat(patches, dim=0)
    return patches
"""

def get_patches(bands, patch_size=33):

    patches = []
    for i in range(bands.shape[2] // patch_size):
        for j in range(bands.shape[3] // patch_size):
            patches.append(bands[:, :, patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1)])

    patches = torch.cat(patches, dim=0)
    return patches