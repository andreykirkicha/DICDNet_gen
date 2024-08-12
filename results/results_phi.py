import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from results_rho import mkdir

#  '', 'LI_', 'ma_'
prefix = ''

data = pd.read_csv("E:/DICDNet/results/" + prefix + "metrics.txt", sep="\s+")

RHO = data['rho'].unique()
METRICS = ['PSNR', 'SSIM', 'NRMSE']
MASKS = data['mask'].unique()

mkdir('E:/DICDNet/results/' + prefix + 'metric_phi')

for mask in MASKS:
    mask_frame = data[data['mask'] == mask]

    for metric in METRICS:
        with sns.axes_style('whitegrid'):
            plt.figure()

        for rho in RHO:
            rho_frame = mask_frame[mask_frame['rho'] == rho]
            PHI = rho_frame['phi'].unique()

            avgs = []

            for phi in PHI:
                phi_frame = rho_frame[rho_frame['phi'] == phi]
                
                avgs.append(phi_frame[metric].mean())

            if rho == RHO[0]:
                plt.plot(PHI, avgs, marker='o', color='#44aaff')
            elif rho == RHO[1]:
                plt.plot(PHI, avgs, marker='v', color='#4488ff')
            elif rho == RHO[2]:
                plt.plot(PHI, avgs, marker='s', color='#4466ff')
            elif rho == RHO[3]:
                plt.plot(PHI, avgs, marker='d', color='#5544ff') 
            elif rho == RHO[4]:
                plt.plot(PHI, avgs, marker='.', color='#6622dd')
            elif rho == RHO[5]:
                plt.plot(PHI, avgs, marker='1', color='#8800bb')
            elif rho == RHO[6]:
                plt.plot(PHI, avgs, marker='*', color='#aa0099')
            elif rho == RHO[7]:
                plt.plot(PHI, avgs, marker='x', color='#cc0077')
            elif rho == RHO[8]:
                plt.plot(PHI, avgs, marker='|', color='#ee0055')
            elif rho == RHO[9]:
                plt.plot(PHI, avgs, marker='h', color='#cc0033')
            elif rho == RHO[10]:
                plt.plot(PHI, avgs, marker='X', color='#aa0011')
            else:
                plt.plot(PHI, avgs, marker='P', color='#880000')

        plt.grid()
        plt.xlabel("phi")
        plt.ylabel(metric)
        plt.title(prefix + "mask " + str(mask) + " " + metric + " vs phi for different rho values")
        plt.legend(labels=[f"rho = {rho}" for rho in RHO])
        cur_path = 'E:/DICDNet/results/' + prefix + 'metric_phi/mask' + str(mask) + '/'
        mkdir(cur_path)
        plt.savefig(cur_path + prefix + 'mask' + str(mask) + '_' + metric + '_phi.png', dpi=100)
        plt.clf()