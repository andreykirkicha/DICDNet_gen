import os
import os.path
import argparse
import numpy as np
import skimage.metrics
import torch
import time
import skimage
from utils import utils_image
from data.preprocess_clinic.preprocessing_clinic import clinic_input_data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
import PIL
from PIL import Image
import utils.save_image as save_img
from dicdnet import DICDNet

parser = argparse.ArgumentParser(description="ACDNet_Test")
parser.add_argument("--model_dir", type=str, default="pretrained_model/DICDNet_latest.pt", help='path to model file')
parser.add_argument("--data_path", type=str, default="data/test/", help='path to test data')
parser.add_argument("--mask_path", type=str, default="data/mask/", help='path to masks')
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

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

pred_path = opt.save_path +'/Xmar/'
mkdir(pred_path)

gt_path = opt.save_path +'/Xgt/'
mkdir(gt_path)

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

def test_image(allXma, allXgt, allXLI, allM, allSma, allSLI, allTr, vol_idx):
    Xma = allXma[vol_idx]
    Xgt = allXgt[vol_idx]
    XLI = allXLI[vol_idx]
    M = allM[vol_idx]
    Sma = allSma[vol_idx]
    SLI = allSLI[vol_idx]
    Tr = allTr[vol_idx]

    Xma = normalize(Xma, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())

    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)    
    non_mask = 1 - Mask
    
    return torch.Tensor(Xma).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(non_mask).cuda()

def main():
    # Build model
    print('Loading model ...\n')
    model = DICDNet(opt).cuda()
    model.load_state_dict(torch.load(opt.model_dir))
    model.eval()

    time_test = 0
    count = 0

    names = ['Xgt', 'Xmar']

    for name in names:
        current_path = os.path.join(opt.save_path, name)
        files = os.listdir(current_path)
        for file in files:
            os.remove(os.path.join(current_path, file))

    print('load data for DICDNet ...')
    allXma, allXgt, allXLI, allM, allSma, allSLI, allTr, allfilename = clinic_input_data(opt.data_path, 'data/generated', opt.mask_path)
    print('\ntesting DICDNet ...')
   
    for vol_idx in range(len(allXma)):
        print("imag_idx:", vol_idx)
      
        Xma, Xgt, XLI, M = test_image(allXma, allXgt, allXLI, allM, allSma, allSLI, allTr, vol_idx)
        
        with torch.no_grad():
            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()
            X0, ListX, ListA = model(Xma, XLI, M)
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time

        Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
        Xoutnorm = Xoutclip / 0.5
        # Xouthu = tohu(Xoutclip)

        Xpred_out = Xoutnorm.data.cpu().numpy().squeeze()
        Xgt_out = Xgt.data.cpu().numpy().squeeze()
        XLI_out = XLI.data.cpu().numpy().squeeze()
        
        image = Image.fromarray(Xpred_out)
        image.save(pred_path + 'pred_' + str(vol_idx) + '.tif')
        print('image pred_' + str(vol_idx) + '.tif saved')
        
        image_gt = Image.fromarray(Xgt_out)
        image_gt.save(gt_path + 'gt_' + str(vol_idx) + '.tif')   
        print('image gt_' + str(vol_idx) + '.tif saved')

        print('PNSR\t metric: {:.4f}'.format(psnr(Xpred_out, Xgt_out)))
        print('SSIM\t metric: {:.4f}'.format(ssim(Xpred_out, Xgt_out)))
        # print('L2_diff/L2_gt   : {:.4f}'.format(np.sqrt(np.mean((Xpred_out - Xgt_out) ** 2) / np.mean(Xgt_out ** 2))))
        print('L2_diff/L2_gt  : {:.4f}'.format(nrmse(Xgt_out, Xpred_out, normalization='mean') /  nrmse(Xgt_out, np.zeros_like(Xgt_out), normalization='mean')))

        print('Times: {:.4f}'.format(dur_time))
        count += 1
        print(100*'*')

if __name__ == "__main__":
    main()