import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("results/metrics.txt", sep="\s+")            # reading from file

idxs = data[data['phi'] == 20.0].index
data = data.drop(index=idxs)                            # dropping failed values

RHO = data['rho'].unique()                           # getting 'rho' values

METRICS = ['PSNR', 'SSIM', 'NRMSE']

for metric in METRICS:
    with sns.axes_style('whitegrid'):
        plt.figure()
    
    for rho in RHO:
        rho_frame = data[data['rho'] == rho]
        PHI = rho_frame['phi'].unique()              # getting values of 'phi' for current 'rho'

        avgs = []

        for phi in PHI:
            phi_frame = rho_frame[rho_frame['phi'] == phi]
            
            avgs.append(phi_frame[metric].mean())

        if rho == RHO[0]:
            plt.plot(PHI, avgs, 'o-')
        elif rho == RHO[1]:
            plt.plot(PHI, avgs, 'v-')
        elif rho == RHO[2]:
            plt.plot(PHI, avgs, 's-')
        else:
            plt.plot(PHI, avgs, 'd-')    

    plt.grid()
    plt.xlabel("phi")
    plt.ylabel(metric)
    plt.title(metric + " vs phi for different rho values")
    plt.legend(labels=[f"rho = {rho}" for rho in RHO])
    plt.savefig('results/' + metric + '.png', dpi=100)
    plt.clf()