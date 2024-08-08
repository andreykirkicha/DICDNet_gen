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

rho = RHO[1]

mkdir('E:/DICDNet/results/all_phi')

for mask in MASKS:
    for metric in METRICS:
        with sns.axes_style('whitegrid'):
            plt.figure()
        
        for prefix in PREFIXES:
            if prefix == PREFIXES[0]:
                rho_frame = data[data['rho'] == rho]
            if prefix == PREFIXES[1]:
                rho_frame = data_LI[data_LI['rho'] == rho]
            if prefix == PREFIXES[2]:
                rho_frame = data_ma[data_ma['rho'] == rho]

            mask_frame = rho_frame[rho_frame['mask'] == mask]
            
            mtrcs = []
        
            for phi in PHI:
                phi_frame = mask_frame[mask_frame['phi'] == phi]
                mtrcs.append(phi_frame[metric].mean())
            
            if prefix == PREFIXES[0]:
                plt.plot(PHI, mtrcs, marker='o', color='#0000ff')
            if prefix == PREFIXES[1]:
                plt.plot(PHI, mtrcs, marker='v', color='#00ff00')
            if prefix == PREFIXES[2]:
                plt.plot(PHI, mtrcs, marker='s', color='#ff0000')
        
        plt.grid()
        plt.xlabel("phi")
        plt.ylabel(metric)
        plt.title("mask " + str(mask) + " " + metric + " vs phi")
        plt.legend(labels=[prefix for prefix in PREFIXES])
        cur_path = 'E:/DICDNet/results/all_phi/mask' + str(mask) + '/'
        mkdir(cur_path)
        plt.savefig(cur_path + 'mask' + str(mask) + '_' + metric + '_phi.png', dpi=100)
        plt.clf()