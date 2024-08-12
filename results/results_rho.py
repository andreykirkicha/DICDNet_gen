import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# '', 'LI_', 'ma_'
prefix = ''

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

data = pd.read_csv("E:/DICDNet/results/" + prefix + "metrics.txt", sep="\s+")   

RHO = data['rho'].unique()                                      
METRICS = ['PSNR', 'SSIM', 'NRMSE']
MASKS = data['mask'].unique()
PHI = data['phi'].unique()
PHI_ours = [PHI[1], PHI[4], PHI[7]]

mkdir('E:/DICDNet/results/' + prefix + 'metric_rho')

for phi in PHI_ours:
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
            
            plt.plot(RHO, mtrcs, 'o-', color='#00cccc')
            plt.grid()
            plt.xlabel("rho")
            plt.ylabel(metric)
            plt.title(prefix + "mask " + str(mask) + " " + metric + " vs rho")
            plt.legend(labels=[f"phi = {phi}"])
            cur_path = 'E:/DICDNet/results/' + prefix + 'metric_rho/mask' + str(mask) + '/'
            mkdir(cur_path)
            plt.savefig(cur_path + prefix + 'mask' + str(mask) + '_' + metric + '_phi' + str(phi) + '_rho.png', dpi=100)
            plt.clf()