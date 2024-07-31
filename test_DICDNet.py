import os
import os.path
import argparse
import numpy as np
import torch
import time
import shutil
import csv
from data.preprocess.preprocessing import clinic_input_data, save_as_image
from data.preprocess.utils import get_config
from data.preprocess.generate_config import mkdir
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from dicdnet import DICDNet

parser = argparse.ArgumentParser(description="ACDNet_Test")
parser.add_argument("--model_dir", type=str, default="pretrained_model/DICDNet_latest.pt", help='path to model file')
parser.add_argument("--data_path", type=str, default="data/test/", help='path to test data')
parser.add_argument("--mask_path", type=str, default="data/mask/", help='path to masks')
parser.add_argument("--gen_path", type=str, default="data/generated/", help='path to generated data')
parser.add_argument("--config_path", type=str, default="data/preprocess/configs/", help='path to configuration files')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--save_path", type=str, default="results/", help='path to testing results')
parser.add_argument('--num_M', type=int, default=32, help='the number of feature maps')
parser.add_argument('--num_Q', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--T', type=int, default=3, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='Stage number')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='stepsize for updating X')
parser.add_argument('--batchSize', type=int, default=1, help='testing input batch size')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

mask_thre = 2500 / 1000 * 0.192 + 0.192

def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (maxX - minX)
    return X

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def image_get_minmax():
    return 0.0, 1.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def tohu(X):           # display window as [-175HU, 275HU]
    CT = (X - 0.192) * 1000 / 0.192
    CT_win = CT.clamp_(-175, 275)
    CT_winnorm = (CT_win +175) / (275+175)
    return CT_winnorm

def test_image(allXma, allXgt, allXLI, allM, allSma, allSLI, allTr, mask_idx):
    Xma = allXma[mask_idx]
    Xgt = allXgt[mask_idx]
    XLI = allXLI[mask_idx]
    M = allM[mask_idx]
    Sma = allSma[mask_idx]
    SLI = allSLI[mask_idx]
    Tr = allTr[mask_idx]

    Xma = normalize(Xma, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())

    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)    
    non_mask = 1 - Mask
    
    return torch.Tensor(Xma).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(non_mask).cuda()

def main():
    # Build model
    print('Loading model ...')
    model = DICDNet(opt).cuda()
    model.load_state_dict(torch.load(opt.model_dir))
    model.eval()

    time_test = 0
    count = 0

    for dir in os.listdir(opt.gen_path):
        shutil.rmtree(os.path.join(opt.gen_path, dir))

    print('Load data for DICDNet ...\n')
    for config_dir in os.listdir(opt.config_path):
        cur_conf = opt.config_path + config_dir

        for config_name in os.listdir(cur_conf):
            config = get_config(os.path.join(cur_conf, config_name))
            CTpara = config['CTpara']

            print(50*'*', 'PARAMETERS', f"k = {CTpara['k']:.2f}, phi = {CTpara['phi']:.2f}", 50*'*')

            allM, allSLI, allSma, allTr, allXgt, allXLI, allXma = clinic_input_data(opt.data_path, 'data/generated', 
                                                                                    opt.mask_path, os.path.join(cur_conf, config_name))
            
            print('\nTesting network ...')

            for img_idx in range(len(allXma)):
                print(10*'=', 'IMAGE ', img_idx,  10*'=')

                for mask_idx in range(len(allXma[img_idx])):
                    print(6*' ', 3*'=', 'Mask  ', mask_idx, 3*'=')

                    Xma, Xgt, XLI, M = test_image(allXma[img_idx], allXgt[img_idx], allXLI[img_idx], allM[img_idx], allSma[img_idx], allSLI[img_idx], allTr[img_idx], mask_idx)
                    
                    with torch.no_grad():
                        if opt.use_GPU:
                            torch.cuda.synchronize()
                        start_time = time.time()
                        X0, ListX, ListA = model(Xma, XLI, M)
                        end_time = time.time()
                        dur_time = end_time - start_time
                        time_test += dur_time

                    Xoutclip = torch.clip(ListX[-1] / 255.0, 0, 1)
                    # Xoutnorm = Xoutclip / 0.5
                    # Xouthu = tohu(Xoutclip)

                    Xpred_out = Xoutclip.data.cpu().numpy().squeeze()
                    Xgt_out = Xgt.data.cpu().numpy().squeeze()

                    [row, col] = np.where(Xgt_out > mask_thre)
                    Xgt_out[row, col] = 1
                    Xpred_out[row, col] = 1
                    
                    save_as_image(Xpred_out, img_idx, mask_idx, opt.save_path, CTpara, 'Xpred')

                    metrics = {'k' : CTpara['k'], 'phi' : CTpara['phi'],
                               'PNSR' : psnr(Xpred_out, Xgt_out),
                               'SSIM' : ssim(Xpred_out, Xgt_out),
                               'NRMSE' : nrmse(Xgt_out, Xpred_out, normalization='mean') /  nrmse(Xgt_out, np.zeros_like(Xgt_out), normalization='mean')
                               }

                    print('PNSR :\t{:.4f}'.format(metrics['PNSR']))
                    print('SSIM :\t{:.4f}'.format(metrics['SSIM']))
                    print('NRMSE:\t{:.4f}'.format(metrics['NRMSE']))
                    print('')

if __name__ == "__main__":
    main()