import os
import torch
import numpy as np

import gc
from recordclass import recordclass

import csv

from CS.PRACS import PRACS
from CS.Brovey import BT_H
from CS.BDSD import BDSD, BDSD_PC
from CS.GS import GS, GSA
from MRA.GLP import MTF_GLP, MTF_GLP_FS, MTF_GLP_HPM, MTF_GLP_HPM_H, MTF_GLP_HPM_R
from MRA.AWLP import AWLP
from MRA.MF import MF

from ModelBasedOptimization.SR_D import SR_D
from ModelBasedOptimization.TV import TV
from ModelBasedOptimization.Bayesian import BayesianNaive, BayesianSparse
from ModelBasedOptimization.HySURE import HySURE
from HyperPNN.HyperPNN import HyperPNN
from HSpeNet.HSpeNet import HSpeNet
from R_PNN.R_PNN import R_PNN
from PCA_Z_PNN.PCA_Z_PNN import PCA_Z_PNN
from DIPHyperKite.DIP_HyperKite import DIP_HyperKite
from HyperDSNet.HyperDSNet import HyperDSNet
from DHPDarn.DHPDarn import DHP_Darn

from Metrics.evaluation import evaluation_rr, evaluation_fr

from Utils.dl_tools import generate_paths
from Utils.load_save_tools import open_mat
from Utils import load_save_tools as ut


pansharpening_algorithm_dict = {'BDSD': BDSD, 'BDSD-PC': BDSD_PC,'GS': GS, 'GSA': GSA, 'BT-H': BT_H, 'PRACS': PRACS,  # Component substitution
                                'AWLP': AWLP, 'MTF-GLP': MTF_GLP, 'MTF-GLP-FS': MTF_GLP_FS,  # Multi-Resolution analysis
                                'MTF-GLP-HPM': MTF_GLP_HPM, 'MTF-GLP-HPM-H': MTF_GLP_HPM_H,  # Multi-Resolution analysis
                                'MTF-GLP-HPM-R': MTF_GLP_HPM_R, 'MF': MF,  # Multi-Resolution analysis
                                'SR-D': SR_D, 'TV': TV,  # Model-Based Optimization
                                'BayesianNaive': BayesianNaive, 'BayesianSparse': BayesianSparse, 'HySURE': HySURE, # Model-Based Optimization
                                'HyperPNN': HyperPNN, 'HSpeNet': HSpeNet,  # Deep Learning Supervised
                                'DIP-HyperKite': DIP_HyperKite, 'Hyper-DSNet': HyperDSNet, 'DHP-DARN': DHP_Darn,  # Deep Learning Supervised
                                'R-PNN': R_PNN, 'PCA-Z-PNN': PCA_Z_PNN # Deep Learning Unsupervised
                                }

fieldnames_rr = ['Method', 'ERGAS', 'SAM', 'Q', 'Q2n']
fieldnames_fr = ['Method', 'R-ERGAS', 'R-SAM', 'D_lambda_V', 'D_lambda_K', 'D_s', 'D_sR', 'D_rho']


if __name__ == '__main__':
    from Utils.dl_tools import open_config

    config_path = 'preambol.yaml'
    config = open_config(config_path)

    for dataset in config.datasets:
        ds_paths = []
        for experiment_folder in config.experiment_folders:
            ds_paths += generate_paths(config.ds_root, dataset, 'Test', experiment_folder)


        for path in ds_paths:
            print(path)
            name = path.split(os.sep)[-1].split('.')[0]
            pan, ms_lr, ms, gt, wavelenghts, overlap = open_mat(path)
            save_root = os.path.join(config.save_root, dataset, name)

            exp_info = {'ratio': pan.shape[-2] // ms_lr.shape[-2]}
            exp_info['ms_lr'] = ms_lr
            exp_info['ms'] = ms
            exp_info['pan'] = pan
            exp_info['wavelenghts'] = wavelenghts
            exp_info['overlap'] = overlap
            exp_info['dataset'] = dataset
            exp_info['sensor'] = config.sensor
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
                    metrics_values_rr = list(evaluation_rr(fused, torch.clone(gt), ratio=exp_info['ratio']))
                    metrics_values_rr.insert(0, algorithm)
                    metrics_values_rr_dict = dict(zip(fieldnames_rr, metrics_values_rr))
                    print(metrics_values_rr_dict)
                    metrics_rr.append(metrics_values_rr_dict)
                else:
                    metrics_values_fr = list(evaluation_fr(fused, torch.clone(pan), torch.clone(ms_lr), ratio=exp_info['ratio'], sensor=exp_info['sensor']))
                    metrics_values_fr.insert(0, algorithm)
                    metrics_values_fr_dict = dict(zip(fieldnames_fr, metrics_values_fr))
                    print(metrics_values_fr_dict)
                    metrics_fr.append(metrics_values_fr_dict)
                if config.save_results:
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    ut.save_mat(np.round(np.squeeze(fused.numpy(), axis=0)).astype(np.uint16), os.path.join(save_root, algorithm + '.mat'))

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


