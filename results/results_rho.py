import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("results/metrics.txt", sep="\s+")            # reading from file

idxs = data[data['phi'] == 20.0].index
data = data.drop(index=idxs)                            # dropping failed values

RHO = data['rho'].unique()                           # getting 'rho' values
METRICS = ['PSNR', 'SSIM', 'NRMSE']
MASKS = [0, 1]
phi = 10.0       

phi_frame = data[data['phi'] == phi]

for mask in MASKS:
    mask_frame = phi_frame[phi_frame['mask'] == mask]

    for metric in METRICS:
        with sns.axes_style('whitegrid'):
            plt.figure()

        mtrcs = []
        
        for rho in RHO:
            rho_frame = mask_frame[mask_frame['rho'] == rho]

            mtrcs.append(rho_frame[metric].mean())
        
        plt.plot(RHO, mtrcs, 'ro-', alpha=0.6)
        plt.grid()
        plt.xlabel("rho")
        plt.ylabel(metric)
        plt.title("mask " + str(mask) + " " + metric + " vs rho")
        plt.legend(labels=[f"phi = {phi}"])
        plt.savefig('results/mask_' + str(mask) + '_' + metric + '_rho.png', dpi=100)
        plt.clf()