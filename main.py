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

from HySURE.HySURE import HySURE
from HyperPNN.HyperPNN import HyperPNN
from HSpeNet.HSpeNet import HSpeNet
from R_PNN.R_PNN import R_PNN
from PCA_Z_PNN.PCA_Z_PNN import PCA_Z_PNN

from Metrics.evaluation import evaluation_rr, evaluation_fr

from Utils.dl_tools import generate_paths
from Utils.load_save_tools import open_mat
from Utils import load_save_tools as ut


gdal.UseExceptions()

pansharpening_algorithm_dict = {'BDSD': BDSD, 'GS': GS, 'GSA': GSA, 'BT-H': BT_H, 'PRACS': PRACS,  # Component substitution
                                'AWLP': AWLP, 'MTF-GLP': MTF_GLP, 'MTF-GLP-FS': MTF_GLP_FS,  # Multi-Resolution analysis
                                'MTF-GLP-HPM': MTF_GLP_HPM, 'MTF-GLP-HPM-H': MTF_GLP_HPM_H,  # Multi-Resolution analysis
                                'MTF-GLP-HPM-R': MTF_GLP_HPM_R, 'MF': MF,  # Multi-Resolution analysis
                                'HySURE': HySURE, 'HyperPNN': HyperPNN, 'HSpeNet': HSpeNet,  # Ad hoc
                                'R-PNN': R_PNN, 'PCA-Z-PNN': PCA_Z_PNN
                                }

fieldnames_rr = ['Method', 'ERGAS', 'SAM', 'Q', 'Q2n']
fieldnames_fr = ['Method', 'R-ERGAS', 'R-SAM', 'R-Q', 'D_lambda', 'D_s', 'D_sR', 'D_rho']


if __name__ == '__main__':
    from Utils.dl_tools import open_config

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = 'preambol.yaml'
    config = open_config(config_path)

    for dataset in config.datasets:

        ds_paths = generate_paths(config.ds_root, dataset, 'Test')

        for path in ds_paths:
            print(path)
            name = path.split(os.sep)[-1].split('.')[0]
            pan, ms_lr, ms, gt, wavelenghts = open_mat(path)

            exp_info = {'ratio': pan.shape[-2] // ms_lr.shape[-2]}
            exp_info['ms_lr'] = ms_lr
            exp_info['ms'] = ms
            exp_info['pan'] = pan
            exp_info['wavelenghts'] = wavelenghts
            exp_info['dataset'] = dataset
            exp_info['name'] = name
            exp_info['root'] = config.ds_root

            exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())

            metrics_rr = []
            metrics_fr = []

            for algorithm in config.pansharpening_based_algorithms:

                if gt is None:
                    experiment_type = 'FR'
                else:
                    experiment_type = 'RR'

                print('Running algorithm: ' + algorithm)

                method = pansharpening_algorithm_dict[algorithm]

                fused = method(exp_input)

                if experiment_type == 'RR':
                    metrics_values_rr = list(evaluation_rr(fused.to(device), torch.clone(gt).to(device), ratio=exp_info['ratio']))
                    metrics_values_rr.insert(0, algorithm)
                    metrics_values_rr_dict = dict(zip(fieldnames_rr, metrics_values_rr))
                    print(metrics_values_rr_dict)
                    metrics_rr.append(metrics_values_rr_dict)
                else:
                    metrics_values_fr = list(evaluation_fr(fused.to(device), torch.clone(pan).to(device), torch.clone(ms_lr).to(device), ratio=exp_info['ratio'], dataset=exp_info['dataset']))
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


