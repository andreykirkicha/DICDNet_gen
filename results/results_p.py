import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("results/metrics.txt", sep="\s+")            # reading from file

idxs = data[data['phi'] == 20.0].index
data = data.drop(index=idxs)                            # dropping failed values

RHO = data['rho'].unique()                           # getting 'rho' values
METRICS = ['PSNR', 'SSIM', 'NRMSE']
MASKS = [5, 7]

for mask in MASKS:
    mask_frame = data[data['mask'] == mask]

    for metric in METRICS:
        with sns.axes_style('whitegrid'):
            plt.figure()
        
        for rho in RHO:
            rho_frame = mask_frame[mask_frame['rho'] == rho]
            PHI = rho_frame['phi'].unique()              # getting values of 'phi' for current 'rho'

            mtrcs = []

            for phi in PHI:
                phi_frame = rho_frame[rho_frame['phi'] == phi]
                
                mtrcs.append(phi_frame[metric].mean())

            if rho == RHO[0]:
                plt.plot(PHI, mtrcs, 'ro-', alpha=0.6)
            elif rho == RHO[1]:
                plt.plot(PHI, mtrcs, 'bv-', alpha=0.6)
            elif rho == RHO[2]:
                plt.plot(PHI, mtrcs, 'gs-', alpha=0.6)
            else:
                plt.plot(PHI, mtrcs, 'md-', alpha=0.6)    

        plt.grid()
        plt.xlabel("phi")
        plt.ylabel(metric)
        plt.title("mask " + str(mask) + " " + metric + " vs phi for different rho values")
        plt.legend(labels=[f"rho = {rho}" for rho in RHO])
        plt.savefig('results/mask_' + str(mask) + '_' + metric + '_p.png', dpi=100)
        plt.clf()