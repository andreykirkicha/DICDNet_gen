import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

phi = PHI[2]

mkdir('E:/DICDNet/results/all_rho')

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
                plt.plot(RHO, mtrcs, marker='o', color='#0000ff')
            if prefix == PREFIXES[1]:
                plt.plot(RHO, mtrcs, marker='v', color='#00ff00')
            if prefix == PREFIXES[2]:
                plt.plot(RHO, mtrcs, marker='s', color='#ff0000')
        
        plt.grid()
        plt.xlabel("rho")
        plt.ylabel(metric)
        plt.title("mask " + str(mask) + " " + metric + " vs rho")
        plt.legend(labels=[prefix for prefix in PREFIXES])
        cur_path = 'E:/DICDNet/results/all_rho/mask' + str(mask) + '/'
        mkdir(cur_path)
        plt.savefig(cur_path + 'mask' + str(mask) + '_' + metric + '_rho.png', dpi=100)
        plt.clf()