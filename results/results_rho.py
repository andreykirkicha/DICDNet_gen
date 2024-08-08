import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

prefix = 'ma_'

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

data = pd.read_csv("E:/DICDNet/results/" + prefix + "metrics.txt", sep="\s+")   

RHO = data['rho'].unique()                                      
METRICS = ['PSNR', 'SSIM', 'NRMSE']
MASKS = data['mask'].unique()
PHI = data['phi'].unique()

phi = PHI[2]
phi_frame = data[data['phi'] == phi]

mkdir('E:/DICDNet/results/' + prefix + 'metric_rho')

for mask in MASKS:
    mask_frame = phi_frame[phi_frame['mask'] == mask]

    for metric in METRICS:
        with sns.axes_style('whitegrid'):
            plt.figure()

        mtrcs = []
        
        for rho in RHO:
            rho_frame = mask_frame[mask_frame['rho'] == rho]

            mtrcs.append(rho_frame[metric].mean())
        
        plt.plot(RHO, mtrcs, 'ro-')
        plt.grid()
        plt.xlabel("rho")
        plt.ylabel(metric)
        plt.title(prefix + "mask " + str(mask) + " " + metric + " vs rho")
        plt.legend(labels=[f"phi = {phi}"])
        cur_path = 'E:/DICDNet/results/' + prefix + 'metric_rho/mask' + str(mask) + '/'
        mkdir(cur_path)
        plt.savefig(cur_path + prefix + 'mask' + str(mask) + '_' + metric + '_rho.png', dpi=100)
        plt.clf()