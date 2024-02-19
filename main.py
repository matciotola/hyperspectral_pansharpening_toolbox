import os
import torch
import numpy as np
import time

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
from ReRaNet.ReRaNet import ReRaNet
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
                                'ReRaNet': ReRaNet, 'R-PNN': R_PNN, 'PCA-Z-PNN': PCA_Z_PNN # Deep Learning Unsupervised
                                }

fieldnames_rr = ['Method', 'ERGAS', 'SAM', 'Q', 'Q2n', 'Elapsed_time']
fieldnames_fr = ['Method', 'R-ERGAS', 'R-SAM', 'D_lambda_V', 'D_lambda_K', 'D_s', 'D_sR', 'D_rho', 'Overlapped_D_s', 'Overlapped_D_sR', 'Overlapped_D_rho', 'Not_Overlapped_D_s', 'Not_Overlapped_D_sR', 'Not_Overlapped_D_rho', 'Elapsed_time']


if __name__ == '__main__':
    from Utils.dl_tools import open_config

    config_path = 'preambol.yaml'
    config = open_config(config_path)

    for dataset in config.datasets:
        ds_paths = []
        for experiment_folder in config.experiment_folders:
            ds_paths += generate_paths(config.ds_root, dataset, 'Test', experiment_folder)


        for i, path in enumerate(ds_paths):
            print(path)
            name = path.split(os.sep)[-1].split('.')[0]
            pan, ms_lr, ms, gt, wavelenghts, overlap = open_mat(path)

            if gt is None:
                experiment_type = 'FR'
            else:
                experiment_type = 'RR'

            save_assessment = os.path.join(config.save_assessment, dataset)
            save_root = os.path.join(config.save_root, dataset, name)

            if not os.path.exists(os.path.join(save_assessment, experiment_type)):
                os.makedirs(os.path.join(save_assessment, experiment_type))

            if experiment_type == 'RR':
                if not os.path.exists(os.path.join(save_assessment, experiment_type, name + '_RR.csv')):
                    with open(os.path.join(save_assessment, experiment_type, name + '_RR.csv'), 'w', encoding='UTF8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames_rr)
                        writer.writeheader()
                    f.close()

            else:
                if not os.path.exists(os.path.join(save_assessment, experiment_type, name + '_FR.csv')):
                    with open(os.path.join(save_assessment, experiment_type, name + '_FR.csv'), 'w', encoding='UTF8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames_fr)
                        writer.writeheader()
                    f.close()


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
            exp_info['img_number'] = i

            exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())

            metrics_rr = []
            metrics_fr = []

            for algorithm in config.pansharpening_based_algorithms:


                print('Running algorithm: ' + algorithm)

                method = pansharpening_algorithm_dict[algorithm]
                start_time = time.time()
                fused = method(exp_input)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Elapsed time for executing the algorithm: ' + str(elapsed_time))
                with torch.no_grad():
                    if experiment_type == 'RR':
                        metrics_values_rr = list(evaluation_rr(fused, torch.clone(gt), ratio=exp_info['ratio']))
                        metrics_values_rr.insert(0, algorithm)
                        metrics_values_rr.append(elapsed_time)
                        metrics_values_rr_dict = dict(zip(fieldnames_rr, metrics_values_rr))
                        print(metrics_values_rr_dict)
                        metrics_rr.append(metrics_values_rr_dict)
                        with open(os.path.join(save_assessment, experiment_type, name + '_RR.csv'), 'a', encoding='UTF8', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames_rr)
                            writer.writerow(metrics_values_rr_dict)
                    else:
                        metrics_values_fr = list(evaluation_fr(fused, torch.clone(pan), torch.clone(ms_lr), torch.clone(ms), ratio=exp_info['ratio'], sensor=exp_info['sensor'], overlap=torch.clone(overlap)))
                        metrics_values_fr.insert(0, algorithm)
                        metrics_values_fr.append(elapsed_time)
                        metrics_values_fr_dict = dict(zip(fieldnames_fr, metrics_values_fr))
                        print(metrics_values_fr_dict)
                        metrics_fr.append(metrics_values_fr_dict)
                        with open(os.path.join(save_assessment, experiment_type, name + '_FR.csv'), 'a', encoding='UTF8', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames_fr)
                            writer.writerow(metrics_values_fr_dict)
                if config.save_results:
                    if not os.path.exists(save_root):
                        os.makedirs(save_root)
                    ut.save_mat(np.round(np.squeeze(fused.numpy(), axis=0)).astype(np.uint16), os.path.join(save_root, algorithm + '.mat'))

                del fused
                torch.cuda.empty_cache()
                gc.collect()




