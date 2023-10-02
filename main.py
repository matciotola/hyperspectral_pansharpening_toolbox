import os
import torch
import numpy as np

from osgeo import gdal

import gc
from recordclass import recordclass

import csv

from CS.PRACS import PRACS
from CS.Brovey import BT_H
from CS.BDSD import BDSD
from CS.GS import GS, GSA
from MRA.GLP import MTF_GLP, MTF_GLP_FS, MTF_GLP_HPM, MTF_GLP_HPM_H, MTF_GLP_HPM_R
from MRA.AWLP import AWLP
from MRA.MF import MF

from Metrics.evaluation import evaluation_rr, evaluation_fr

from Utils.dl_tools import generate_paths
from Utils.load_save_tools import open_mat
from Utils import load_save_tools as ut


gdal.UseExceptions()

pansharpening_algorithm_dict = {'BDSD': BDSD, 'GS': GS, 'GSA': GSA, 'BT-H': BT_H, 'PRACS': PRACS,  # Component substitution
                                'AWLP': AWLP, 'MTF-GLP': MTF_GLP, 'MTF-GLP-FS': MTF_GLP_FS,        # Multi-Resolution analysis
                                'MTF-GLP-HPM': MTF_GLP_HPM, 'MTF-GLP-HPM-H': MTF_GLP_HPM_H,        # Multi-Resolution analysis
                                'MTF-GLP-HPM-R': MTF_GLP_HPM_R, 'MF': MF}                          # Multi-Resolution analysis

fieldnames_rr = ['Method', 'ERGAS', 'SAM', 'Q', 'Q2n']
fieldnames_fr = ['Method', 'R-ERGAS', 'R-SAM', 'R-Q', 'D_lambda', 'D_s', 'D_sR', 'D_rho']


if __name__ == '__main__':
    from Utils.dl_tools import open_config

    config_path = 'preambol.yaml'
    config = open_config(config_path)

    for dataset in config.datasets:

        ds_paths = generate_paths(config.ds_root, dataset)

        for path in ds_paths:
            print(path)
            name = path.split(os.sep)[-1].split('.')[0]


            metrics_rr = []
            metrics_fr = []



            for algorithm in config.pansharpening_based_algorithms:

                pan, ms_lr, ms, gt = open_mat(path)

                exp_info = {'ratio': pan.shape[-2] // ms_lr.shape[-2]}
                exp_info['ms_lr'] = ms_lr
                exp_info['ms'] = ms
                exp_info['pan'] = pan
                exp_info['dataset'] = dataset

                exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())

                if gt is None:
                    experiment_type = 'FR'
                else:
                    experiment_type = 'RR'

                print('Running algorithm: ' + algorithm)

                method = pansharpening_algorithm_dict[algorithm]
                with torch.no_grad():
                    fused = method(exp_input)

                from scipy import io
                import numpy as np

                mat_root_path = '/home/matteo/Desktop/GRSM_FR_Code/Outputs_images/WV3/Adelaide/Adelaide_1_zoom/'
                mat_counterpart = io.loadmat(os.path.join(mat_root_path, algorithm + '.mat'))['I_MS']
                mat_counterpart = torch.from_numpy(np.moveaxis(mat_counterpart, -1, 0)[None, :, :, :])
                error = torch.abs(torch.clip(fused,0,2048) - torch.clip(mat_counterpart,0,2048))
                print(error.max())

                if experiment_type == 'RR':
                    metrics_values_rr = list(evaluation_rr(fused, gt, ratio=exp_info['ratio']))
                    metrics_values_rr.insert(0, algorithm)
                    metrics_values_rr_dict = dict(zip(fieldnames_rr, metrics_values_rr))
                    print(metrics_values_rr_dict)
                    metrics_rr.append(metrics_values_rr_dict)
                else:
                    metrics_values_fr = list(evaluation_fr(fused, pan, ms_lr, ratio=exp_info['ratio'], dataset=exp_info['dataset']))
                    metrics_values_fr.insert(0, algorithm)
                    metrics_values_fr_dict = dict(zip(fieldnames_fr, metrics_values_fr))
                    print(metrics_values_fr_dict)
                    metrics_fr.append(metrics_values_fr_dict)
                if config.save_results:
                    save_root = os.path.join(config.save_root, dataset, name)
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    ut.save_mat(np.squeeze(fused.numpy(), axis=0), os.path.join(save_root, algorithm + '.mat'))

                del fused
                gc.collect()

            if not os.path.exists(os.path.join(save_root, experiment_type)):
                os.makedirs(os.path.join(save_root, experiment_type))

            if experiment_type == 'RR':
                with open(os.path.join(save_root, experiment_type, 'Evaluation_RR.csv'), 'w', encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames_rr)
                    writer.writeheader()
                    writer.writerows(metrics_rr)
            else:
                with open(os.path.join(save_root, experiment_type, 'Evaluation_FR.csv'), 'w', encoding='UTF8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames_fr)
                    writer.writeheader()
                    writer.writerows(metrics_fr)


