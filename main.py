import os
import torch
import numpy as np
import time

import gc
from recordclass import recordclass

import csv

from Utils.EXP import EXP
from CS.PRACS import PRACS
from CS.Brovey import BT_H
from CS.BDSD import BDSD_PC
from CS.GS import GSA
from MRA.GLP import MTF_GLP_FS, MTF_GLP_HPM, MTF_GLP_HPM_R
from MRA.AWLP import AWLP
from MRA.MF import MF

from ModelBasedOptimization.SR_D import SR_D
from ModelBasedOptimization.TV import TV
from ModelBasedOptimization.HySURE import HySURE
from HyperPNN.HyperPNN import HyperPNN
from HSpeNet.HSpeNet import HSpeNet
from R_PNN.R_PNN import R_PNN
from PCA_Z_PNN.PCA_Z_PNN import PCA_Z_PNN
from DIPHyperKite.DIP_HyperKite import DIP_HyperKite
from HyperDSNet.HyperDSNet import HyperDSNet
from DHPDarn.DHPDarn import DHP_Darn

from Metrics.evaluation import evaluation_rr, evaluation_fr

from Utils.dl_tools import generate_paths, open_config
from Utils.load_save_tools import open_mat
from Utils import load_save_tools as ut


algorithm_dict = {
                    'EXP': EXP,  # Baseline
                    'BDSD-PC': BDSD_PC, 'GSA': GSA, 'BT-H': BT_H, 'PRACS': PRACS, # Component substitution
                    'AWLP': AWLP, 'MTF-GLP-FS': MTF_GLP_FS, 'MTF-GLP-HPM': MTF_GLP_HPM, # Multi-Resolution analysis
                    'MTF-GLP-HPM-R': MTF_GLP_HPM_R, 'MF': MF, # Multi-Resolution analysis
                    'SR-D': SR_D, 'TV': TV, # Model-Based Optimization
                    'HySURE': HySURE, # Model-Based Optimization
                    'HyperPNN': HyperPNN, 'HSpeNet': HSpeNet, # Deep Learning Supervised
                    'DIP-HyperKite': DIP_HyperKite, 'Hyper-DSNet': HyperDSNet, 'DHP-DARN': DHP_Darn, # Deep Learning - Supervised
                    'R-PNN': R_PNN, 'PCA-Z-PNN': PCA_Z_PNN # Deep Learning - Unsupervised
                  }

fieldnames_rr = ['Method', 'ERGAS', 'SAM', 'Q2n', 'Elapsed_time']
fieldnames_fr = ['Method', 'D_lambda', 'D_sR', 'QNR', 'Elapsed_time']


if __name__ == '__main__':
    config_path = 'preambol.yaml'
    config = open_config(config_path)

    for dataset in config.datasets:
        ds_paths = [path for exp_folder in config.experiment_folders
                    for path in generate_paths(config.ds_root, dataset, 'Test', exp_folder)]

        for i, path in enumerate(ds_paths):
            print(f'{i + 1} / {len(ds_paths)}:', path)
            name = os.path.basename(path).split('.')[0]
            pan, ms_lr, ms, gt, wavelenghts, overlap = open_mat(path)
            experiment_type = 'FR' if gt is None else 'RR'
            save_assessment = os.path.join(config.save_assessment, dataset, experiment_type)
            save_root = os.path.join(config.save_root, dataset, name)

            os.makedirs(save_assessment, exist_ok=True)
            file_suffix = '_FR.csv' if experiment_type == 'FR' else '_RR.csv'
            ut.create_csv_if_not_exists(os.path.join(save_assessment, name + file_suffix), fieldnames_fr if experiment_type == 'FR' else fieldnames_rr)

            exp_info = {
                'ratio': pan.shape[-2] // ms_lr.shape[-2],
                'ms_lr': ms_lr, 'ms': ms, 'pan': pan, 'wavelenghts': wavelenghts,
                'overlap': overlap, 'dataset': dataset, 'sensor': config.sensor,
                'name': name, 'root': config.ds_root, 'img_number': i
            }
            exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())

            for algorithm in config.algorithms:
                print(f'Running algorithm: {algorithm}')
                method = algorithm_dict[algorithm]
                start_time = time.time()
                fused = method(exp_input)
                elapsed_time = time.time() - start_time
                print(f'Elapsed time for executing the algorithm: {elapsed_time}')

                with torch.no_grad():
                    if experiment_type == 'RR':
                        metrics_values = evaluation_rr(fused, gt.clone(), ratio=exp_info['ratio'])
                        fieldnames, metrics_dict = fieldnames_rr, dict(zip(fieldnames_rr, [algorithm, *metrics_values, elapsed_time]))
                    else:
                        metrics_values = evaluation_fr(fused, pan.clone(), ms_lr.clone(), ms.clone(), ratio=exp_info['ratio'], sensor=exp_info['sensor'], overlap=overlap.clone())
                        fieldnames, metrics_dict = fieldnames_fr, dict(zip(fieldnames_fr, [algorithm, *metrics_values, elapsed_time]))

                    print(metrics_dict)
                    csv_path = os.path.join(save_assessment, name + file_suffix)
                    with open(csv_path, 'a', encoding='UTF8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(metrics_dict)

                if config.save_results:
                    os.makedirs(save_root, exist_ok=True)
                    result_path = os.path.join(save_root, f'{algorithm}.mat')
                    ut.save_mat(np.round(np.clip(fused.permute(0, 2, 3, 1).numpy().squeeze(0), 0, 2**16 - 1)).astype(np.uint16), result_path)

                del fused
                torch.cuda.empty_cache()
                gc.collect()
