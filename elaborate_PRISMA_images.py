import os
import argparse
import gc
import random
import numpy as np
import torch
import h5py
import inspect
from scipy import io
from torch.nn import functional as func
from Utils.spectral_tools import mtf, mtf_pan
from Utils.interpolator_tools import ideal_interpolator

# Constants
random.seed(3)
LIST_BANDS = np.concatenate([list(range(7, 63)), list(range(71, 86)), list(range(89, 107)), list(range(120, 150)), list(range(176, 216))])

TEST_FILES = ['PRS_L2D_STD_20230908173127_20230908173131_0001.he5',
              'PRS_L2D_STD_20230824100356_20230824100400_0001.he5',
              'PRS_L2D_STD_20220905101901_20220905101905_0001.he5',
              'PRS_L2D_STD_20231120102229_20231120102233_0001.he5']

ZONES = {'PRS_L2D_STD_20230908173127_20230908173131_0001.he5': 'Kansas',
         'PRS_L2D_STD_20230824100356_20230824100400_0001.he5': 'Udine',
         'PRS_L2D_STD_20220905101901_20220905101905_0001.he5': 'Cagliari',
         'PRS_L2D_STD_20231120102229_20231120102233_0001.he5': 'Tabasco'}

INIT_ROWS = {'PRS_L2D_STD_20230908173127_20230908173131_0001.he5': 500,
             'PRS_L2D_STD_20230824100356_20230824100400_0001.he5': 428,
             'PRS_L2D_STD_20220905101901_20220905101905_0001.he5': 800,
             'PRS_L2D_STD_20231120102229_20231120102233_0001.he5': 338}

INIT_COLS = {'PRS_L2D_STD_20230908173127_20230908173131_0001.he5': 384,
             'PRS_L2D_STD_20230824100356_20230824100400_0001.he5': 692,
             'PRS_L2D_STD_20220905101901_20220905101905_0001.he5': 120,
             'PRS_L2D_STD_20231120102229_20231120102233_0001.he5': 216}

INIT_ROWS_RR = {'PRS_L2D_STD_20230908173127_20230908173131_0001.he5': 30,
                'PRS_L2D_STD_20230824100356_20230824100400_0001.he5': 38,
                'PRS_L2D_STD_20220905101901_20220905101905_0001.he5': 68,
                'PRS_L2D_STD_20231120102229_20231120102233_0001.he5': 50}

INIT_COLS_RR = {'PRS_L2D_STD_20230908173127_20230908173131_0001.he5': 40,
                'PRS_L2D_STD_20230824100356_20230824100400_0001.he5': 82,
                'PRS_L2D_STD_20220905101901_20220905101905_0001.he5': 30,
                'PRS_L2D_STD_20231120102229_20231120102233_0001.he5': 36}


def he5_to_mat(filename, list_bands):
    with h5py.File(filename, 'r') as h5f:
        SWIR = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'][()], dtype=np.uint16)
        VNIR = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'][()], dtype=np.uint16)
        pan = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube'][()], dtype=np.uint16)

        VNIR = np.flip(np.moveaxis(VNIR, 1, 2), axis=-1)
        SWIR = np.flip(np.moveaxis(SWIR, 1, 2), axis=-1)

        wl_VNIR = np.linspace(400, 1010, 66).astype(np.float32)
        wl_SWIR = np.linspace(920, 2505, 173).astype(np.float32)
        wl = np.concatenate((wl_VNIR, wl_SWIR))

        overlap = np.max([i for i, value in enumerate(wl[list_bands]) if value < 701])

        hs = np.concatenate((VNIR, SWIR), axis=2)
        return hs[:, :, list_bands], pan, wl[list_bands], overlap


def interpolate_hs(hs, ratio):
    hs = np.moveaxis(hs, -1, 0)[None, :, :, :].astype(np.float32)
    hs = torch.from_numpy(hs).float()
    hs_exp = np.zeros((1, hs.size()[1], hs.size()[2] * ratio, hs.size()[3] * ratio), dtype=np.uint16)

    for i in range(hs.size()[1]):
        hs_exp[:, i, :, :] = torch.round(torch.clip(ideal_interpolator(hs[:, i:i + 1, :, :].double(), ratio), 0, 2**16)).numpy().astype(np.uint16)

    hs_exp = np.squeeze(hs_exp)
    hs_exp = np.moveaxis(hs_exp, 0, -1)
    return hs_exp


def downsample_hs_pan(hs, pan, ratio):
    hs = np.moveaxis(hs, -1, 0)[None, :, :, :].astype(np.float32)
    pan = pan[None, None, :, :].astype(np.float32)

    hs = torch.from_numpy(hs).float()
    pan = torch.from_numpy(pan).float()

    hs_lp = mtf(hs, 'PRISMA', ratio)
    pan_lp = mtf_pan(pan, 'PRISMA', ratio)

    hs_lr = func.interpolate(hs_lp, scale_factor=1 / ratio, mode='nearest-exact')
    pan_lr = func.interpolate(pan_lp, scale_factor=1 / ratio, mode='nearest-exact')
    hs_lr_exp = torch.round(torch.clip(ideal_interpolator(hs_lr.double(), ratio), 0, 2**16))

    hs_lr = np.squeeze(hs_lr.numpy()).astype(np.uint16)
    pan_lr = np.squeeze(pan_lr.numpy()).astype(np.uint16)
    hs_lr_exp = np.squeeze(hs_lr_exp.numpy()).astype(np.uint16)

    hs_lr = np.moveaxis(hs_lr, 0, -1)
    hs_lr_exp = np.moveaxis(hs_lr_exp, 0, -1)
    return hs_lr, pan_lr, hs_lr_exp


def create_patches(hs, pan, hs_exp, hs_gt=None, patch_size=32, ratio=6):
    num_hor_patches = hs.shape[0] // patch_size
    num_ver_patches = hs.shape[1] // patch_size

    hs_patches = []
    hs_exp_patches = []
    hs_gt_patches = []
    pan_patches = []

    for i in range(num_ver_patches):
        for j in range(num_hor_patches):
            hs_patches.append(hs[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :])
            pan_patches.append(pan[i * patch_size * ratio:(i + 1) * patch_size * ratio, j * patch_size * ratio:(j + 1) * patch_size * ratio])
            hs_exp_patches.append(hs_exp[i * patch_size * ratio:(i + 1) * patch_size * ratio, j * patch_size * ratio:(j + 1) * patch_size * ratio, :])
            if hs_gt is not None:
                hs_gt_patches.append(hs_gt[i * patch_size * ratio:(i + 1) * patch_size * ratio, j * patch_size * ratio:(j + 1) * patch_size * ratio, :])

    return hs_patches, pan_patches, hs_exp_patches, hs_gt_patches


def filter_patches(pan_patches):
    return [i for i, patch in enumerate(pan_patches) if np.all(patch != 0)]


def create_dataset(raw_dataset_dir, save_dir, create_train=True, create_test=True):
    ratio = 6
    patch_size_lr = 32
    patch_size = patch_size_lr * ratio
    test_patch_size = 200
    test_patch_size_rr = 100
    num_val_patches = 2

    dirs = {
        'val': os.path.join(save_dir, "Validation"),
        'train': os.path.join(save_dir, "Training"),
        'test': os.path.join(save_dir, "Test"),
        'fr_val': os.path.join(save_dir, "Validation", "Full_Resolution"),
        'rr_val': os.path.join(save_dir, "Validation", "Reduced_Resolution"),
        'fr_train': os.path.join(save_dir, "Training", "Full_Resolution"),
        'rr_train': os.path.join(save_dir, "Training", "Reduced_Resolution"),
        'fr_test': os.path.join(save_dir, "Test", "Full_Resolution"),
        'rr_test': os.path.join(save_dir, "Test", "Reduced_Resolution")
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    files = [f.path for f in os.scandir(raw_dataset_dir) if f.is_file()]
    test_names = TEST_FILES
    i = 0

    for filename in files:
        i = i + 1
        name = filename.split('/')[-1]
        print(f"Processing {str(i).zfill(2)} / {str(len(files))}: {name}")
        if (name not in test_names) and create_train:
            hs, pan, wl, overlap = he5_to_mat(filename, LIST_BANDS)
            hs_lr, pan_lr, hs_lr_exp = downsample_hs_pan(hs, pan, ratio)
            hs_exp = interpolate_hs(hs, ratio)

            hs_patches, pan_patches, hs_exp_patches, _ = create_patches(hs, pan, hs_exp, patch_size=patch_size, ratio=ratio)
            hs_lr_patches, pan_lr_patches, hs_lr_exp_patches, hs_gt_patches = create_patches(hs_lr, pan_lr, hs_lr_exp, hs, patch_size=patch_size_lr, ratio=ratio)

            good_patches = filter_patches(pan_patches)

            hs_patches = [hs_patches[i] for i in good_patches]
            hs_exp_patches = [hs_exp_patches[i] for i in good_patches]
            pan_patches = [pan_patches[i] for i in good_patches]

            hs_lr_patches = [hs_lr_patches[i] for i in good_patches]
            hs_lr_exp_patches = [hs_lr_exp_patches[i] for i in good_patches]
            hs_gt_patches = [hs_gt_patches[i] for i in good_patches]
            pan_lr_patches = [pan_lr_patches[i] for i in good_patches]

            for k in range(num_val_patches):
                val_index = random.randint(0, len(hs_patches) - 1)

                io.savemat(os.path.join(dirs['fr_val'], f"{name[:-4]}_{str(k+1).zfill(2)}.mat"), {"I_MS_LR": hs_patches[val_index], "I_PAN": pan_patches[val_index], "I_MS": hs_exp_patches[val_index], 'wavelengths': wl, 'overlap': overlap})
                io.savemat(os.path.join(dirs['rr_val'], f"{name[:-4]}_{str(k+1).zfill(2)}.mat"), {"I_MS_LR": hs_lr_patches[val_index], "I_PAN": pan_lr_patches[val_index], "I_MS": hs_lr_exp_patches[val_index], "I_GT": hs_gt_patches[val_index], 'wavelengths': wl, 'overlap': overlap})

                hs_patches.pop(val_index)
                pan_patches.pop(val_index)
                hs_exp_patches.pop(val_index)

                hs_lr_patches.pop(val_index)
                pan_lr_patches.pop(val_index)
                hs_lr_exp_patches.pop(val_index)
                hs_gt_patches.pop(val_index)

            for k, (hs_patch, pan_patch, hs_exp_patch) in enumerate(zip(hs_patches, pan_patches, hs_exp_patches)):
                io.savemat(os.path.join(dirs['fr_train'], f"{name[:-4]}_{str(k+1).zfill(2)}.mat"), {"I_MS_LR": hs_patch, "I_PAN": pan_patch, "I_MS": hs_exp_patch, 'wavelengths': wl, 'overlap': overlap})

            for k, (hs_lr_patch, pan_lr_patch, hs_lr_exp_patch, hs_gt_patch) in enumerate(zip(hs_lr_patches, pan_lr_patches, hs_lr_exp_patches, hs_gt_patches)):
                io.savemat(os.path.join(dirs['rr_train'], f"{name[:-4]}_{str(k+1).zfill(2)}.mat"), {"I_MS_LR": hs_lr_patch, "I_PAN": pan_lr_patch, "I_MS": hs_lr_exp_patch, "I_GT": hs_gt_patch, 'wavelengths': wl, 'overlap': overlap})

            del hs_patches, pan_patches, hs_exp_patches, hs_lr_patches, pan_lr_patches, hs_lr_exp_patches, hs_gt_patches
            gc.collect()

        elif (name in test_names) and create_test:
            # for test_file in TEST_FILES:
            filename = os.path.join(raw_dataset_dir, name)
            zone = ZONES[name]


            hs, pan, wl, overlap = he5_to_mat(filename, LIST_BANDS)
            hs_lr, pan_lr, hs_lr_exp = downsample_hs_pan(hs, pan, ratio)
            hs_exp = interpolate_hs(hs, ratio)

            io.savemat(os.path.join(dirs['fr_test'], f"{name[:-4]}_{zone.upper()}_FR.mat"), {
                "I_MS_LR": hs[INIT_ROWS[name]:INIT_ROWS[name]+test_patch_size, INIT_COLS[name]:INIT_COLS[name]+test_patch_size, :],
                "I_PAN": pan[INIT_ROWS[name]*ratio:(INIT_ROWS[name]+test_patch_size)*ratio, INIT_COLS[name]*ratio:(INIT_COLS[name]+test_patch_size)*ratio],
                "I_MS": hs_exp[INIT_ROWS[name]*ratio:(INIT_ROWS[name]+test_patch_size)*ratio, INIT_COLS[name]*ratio:(INIT_COLS[name]+test_patch_size)*ratio, :],
                'wavelengths': wl, 'overlap': overlap
            })
            io.savemat(os.path.join(dirs['rr_test'], f"{name[:-4]}_{zone.upper()}_RR.mat"), {
                "I_MS_LR": hs_lr[INIT_ROWS_RR[name]:INIT_ROWS_RR[name]+test_patch_size_rr, INIT_COLS_RR[name]:INIT_COLS_RR[name]+test_patch_size_rr, :],
                "I_PAN": pan_lr[INIT_ROWS_RR[name] * ratio:(INIT_ROWS_RR[name]+test_patch_size_rr)*ratio, INIT_COLS_RR[name]*ratio:(INIT_COLS_RR[name]+test_patch_size_rr)*ratio],
                "I_MS": hs_lr_exp[INIT_ROWS_RR[name] * ratio:(INIT_ROWS_RR[name]+test_patch_size_rr)*ratio, INIT_COLS_RR[name]*ratio:(INIT_COLS_RR[name]+test_patch_size_rr)*ratio, :],
                "I_GT": hs[INIT_ROWS_RR[name] * ratio:(INIT_ROWS_RR[name]+test_patch_size_rr)*ratio, INIT_COLS_RR[name]*ratio:(INIT_COLS_RR[name]+test_patch_size_rr)*ratio, :],
                'wavelengths': wl, 'overlap': overlap
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create PRISMA dataset')
    parser.add_argument('-i', '--input_dir', type=str, help='Input directory with PRISMA images')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory for the dataset', default=os.path.join(os.path.dirname(inspect.getfile(filter_patches)), 'Dataset'))
    parser.add_argument('--no_train', action='store_false', default=True, help='Do not create training dataset')
    parser.add_argument('--no_test', action='store_false', default=True, help='Do not create test dataset')
    args = parser.parse_args()

    create_dataset(args.input_dir, args.output_dir, create_train=args.no_train, create_test=args.no_test)
