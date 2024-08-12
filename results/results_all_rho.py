import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

data = pd.read_csv("E:/DICDNet/results/metrics.txt", sep="\s+")
data_LI = pd.read_csv("E:/DICDNet/results/LI_metrics.txt", sep="\s+")
data_ma = pd.read_csv("E:/DICDNet/results/ma_metrics.txt", sep="\s+")

RHO = data['rho'].unique()
METRICS = ['PSNR', 'SSIM', 'NRMSE']
MASKS = data['mask'].unique()
PHI = data['phi'].unique()
PREFIXES = ['pred', 'LI', 'ma']
LABELS = ['DICDNet', 'ЛИ', 'АСПВ']
PHI_ours = [PHI[1], PHI[4], PHI[7]]

mkdir('E:/DICDNet/results/all_rho')

for phi in PHI_ours:
    for mask in MASKS:
        for metric in METRICS:
            with sns.axes_style('whitegrid'):
                plt.figure()
            
            for prefix in PREFIXES:
                if prefix == PREFIXES[0]:
                    phi_frame = data[data['phi'] == phi]
                if prefix == PREFIXES[1]:
                    phi_frame = data_LI[data_LI['phi'] == phi]
                if prefix == PREFIXES[2]:
                    phi_frame = data_ma[data_ma['phi'] == phi]

                mask_frame = phi_frame[phi_frame['mask'] == mask]
                
                mtrcs = []
            
                for rho in RHO:
                    rho_frame = mask_frame[mask_frame['rho'] == rho]
                    mtrcs.append(rho_frame[metric].mean())
                
                if prefix == PREFIXES[0]:
                    plt.plot(RHO, mtrcs, marker='o', color='#000000')
                if prefix == PREFIXES[1]:
                    plt.plot(RHO, mtrcs, marker='v', color='#00cccc')
                if prefix == PREFIXES[2]:
                    plt.plot(RHO, mtrcs, marker='s', color='#bbbbbb')
            
            plt.grid()
            plt.xlabel("rho", fontsize=20)
            plt.ylabel(metric, fontsize=20)
            # plt.title(f"phi = {np.round(phi, decimals=2)} " + "mask " + str(mask) + " " + metric + " vs rho", fontsize=22)
            plt.title(metric + " vs rho", fontsize=22)
            plt.legend(labels=[label for label in LABELS], fontsize=14)
            cur_path = 'E:/DICDNet/results/all_rho/mask' + str(mask) + '/'
            mkdir(cur_path)
            plt.savefig(cur_path + 'mask' + str(mask) + '_' + metric + '_phi' + str(phi) + '_rho.png', bbox_inches='tight', pad_inches=0)
            plt.clf()