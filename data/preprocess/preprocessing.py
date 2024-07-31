# Given clinical Xma, generate data, including: XLI, M, Sma, SLI, Tr for infering InDuDoNet
import numpy as np
import os
import shutil
from scipy.interpolate import interp1d, RegularGridInterpolator
from .utils import get_config
from .build_geometry import initialization, imaging_geo
from .generate_config import mkdir
import PIL
from PIL import Image

def generation(test_path, res_path, mask_path, config_name):
    config = get_config(config_name)
    CTpara = config['CTpara']                       # CT imaging parameters

    mask_thre = 2500 / 1000 * 0.192 + 0.192 + 0.3         # taking 2500HU as a thresholding to segment the metal region

    param = initialization(CTpara)
    ray_trafo, FBPOper = imaging_geo(param)         # CT imaging geometry, ray_trafo is fp, FBPoper is fbp

    img_num = 0
    mat = []
    spec = []

    with open("data/mar_bh_gen/material.txt", "r+") as f:
        # Reading from a file
        for line in f:
            a, b = [float(x) for x in line[:-1].split(' ')]
            mat.append([a*1000, b])
    with open("data/mar_bh_gen/spectra.txt", "r+") as f:
        # Reading from a file
        for line in f:
            a, b = [float(x) for x in line[:-1].split(' ')]
            spec.append([a, b])

    mat  = np.array(mat)
    spec = np.array(spec)
    mat_interp = RegularGridInterpolator((mat[:,0],), mat[:,1])
    mat_grid = np.array([mat_interp([x]) for x in list(spec[:,0])])

    for file_name in os.listdir(test_path):
        file_path = os.path.join(test_path, file_name)

        # for DeepLesion
        # image = (open_image(file_path, 'float32').astype(np.int32) - 32768).astype(np.float32)
        # image = image / 1000 * 0.192 + 0.192
        # image = np.maximum(image, 0)

        # for CLINIC-metal
        image = open_image(file_path, 'float32', CTpara['imPixNum'], CTpara['imPixNum'])

        M = np.zeros((CTpara['imPixNum'], CTpara['imPixNum']), dtype='float32')
        Xgt = np.zeros_like(M)
        Xma = np.zeros_like(M)
        XLI = np.zeros_like(M)
        Tr = np.zeros((CTpara['sinogram_size_x'], CTpara['sinogram_size_y']), dtype='float32')
        Sgt = np.zeros_like(Tr)
        Sma = np.zeros_like(Tr)
        SLI = np.zeros_like(Tr)

        mask_num = 0

        for mask_name in os.listdir(mask_path):
            print(">>> Generation of img ", img_num, " with mask ", mask_num, " ...")

            cur_mask_path = os.path.join(mask_path, mask_name)
            mask = open_image(cur_mask_path, 'float32', CTpara['imPixNum'], CTpara['imPixNum'])

            # put metal into slice
            Xgt = np.copy(image)
            [x, y] = np.where(mask != 0)
            Xgt[x, y] = mask[x, y]                                      
            
            # segment metal region
            [rowindex, colindex] = np.where(Xgt > mask_thre)
            M.fill(0.0)
            M[rowindex, colindex] = Xgt[rowindex, colindex]              
            
            Pmetal_kev = np.asarray(ray_trafo(M))         
            Tr = Pmetal_kev > 0
            
            # polychromatic radiation
            Sgt = np.asarray(ray_trafo(Xgt))

            rho = CTpara['rho']        # max sinogram value to be about 3-5
            total_sum = 0
            for i in range(mat_grid.shape[0]):
                total_sum += spec[i, 1]
                lin = Tr * mat_grid[i] / rho + Sgt
                Sma += spec[i, 1] * np.exp(-lin)
            Sma = -np.log(Sma / total_sum)
            Xma = np.asarray(FBPOper(Sma))

            # to match metal region of gt and ma images
            [row, col] = np.where(Xma > mask_thre)
            Xgt[row, col] = Xma[row, col]
            
            # linear interpolation
            SLI = interpolate_projection(Sma, Tr)
            XLI = np.asarray(FBPOper(SLI))

            # visualization
            save_as_image(Xma, img_num, mask_num, res_path, CTpara, 'Xma')
            save_as_image(Xgt, img_num, mask_num, res_path, CTpara, 'Xgt')
            save_as_image(M, img_num, mask_num, res_path, CTpara, 'M')
            save_as_image(Tr, img_num, mask_num, res_path, CTpara, 'Tr')
            save_as_image(Sma, img_num, mask_num, res_path, CTpara, 'Sma')
            save_as_image(SLI, img_num, mask_num, res_path, CTpara, 'SLI')
            save_as_image(XLI, img_num, mask_num, res_path, CTpara, 'XLI')

            mask_num += 1
        
        img_num += 1


# save all the to-the-tested volumes
def clinic_input_data(test_path, res_path, mask_path, config_name):
    config = get_config(config_name)
    CTpara = config['CTpara']

    allXma = []
    allXgt = []
    allXLI = []
    allM = []
    allSma = []
    allSLI = []
    allTr = []

    all = [allM, allSLI, allSma, allTr, allXgt, allXLI, allXma]

    # comment if you do not need generation to execute
    generation(test_path, res_path, mask_path, config_name)

    dir_idx = 0
    for dir in os.listdir(res_path):
        path = os.path.join(res_path, dir, f"rho={CTpara['rho']:.2f}", f"phi={CTpara['phi']:.2f}")

        img_idx = 0
        for img_dir in os.listdir(path):
            cur_img = os.path.join(path, img_dir)
            all[dir_idx].append([])

            for mask in os.listdir(cur_img):
                array = open_image(os.path.join(cur_img, mask), 'float32', CTpara['imPixNum'], CTpara['imPixNum'])
                (all[dir_idx])[img_idx].append(array)

            img_idx += 1

        dir_idx += 1

    return allM, allSLI, allSma, allTr, allXgt, allXLI, allXma

def open_image(file_path, d_type, x_size, y_size):
    img = np.array(Image.open(file_path), dtype=d_type)      # ndarray
    return np.array(Image.fromarray(img).resize((x_size, y_size), PIL.Image.Resampling.BILINEAR))     # resize image

def save_as_image(array, img_num, mask_num, res_path, conf, name):
    for_image = array.astype(np.float32)
    image = Image.fromarray(for_image)
    
    cur_path = os.path.join(res_path, name)
    mkdir(cur_path)

    cur_path = os.path.join(cur_path, f"rho={conf['rho']:.2f}")
    mkdir(cur_path)

    cur_path = os.path.join(cur_path, f"phi={conf['phi']:.2f}")
    mkdir(cur_path)

    cur_path = os.path.join(cur_path, 'img' + str(img_num))
    mkdir(cur_path)
    
    image.save(os.path.join(cur_path, f"rho{conf['rho']:.2f}" + f"_phi{conf['phi']:.2f}" + 
                            '_img' + str(img_num) + '_mask' + str(mask_num) + '.tif'))
    # print(name + '\t image saved as ' + 'img' + str(img_num) + '_mask' + str(mask_num) + '.tif')

def interpolate_projection(proj, metalTrace):
    # projection linear interpolation
    # Input:
    # proj:         uncorrected projection
    # metalTrace:   metal trace in projection domain (binary image)
    # Output:
    # Pinterp:      linear interpolation corrected projection
    Pinterp = proj.copy()
    for i in range(Pinterp.shape[0]):
        mslice = metalTrace[i]
        pslice = Pinterp[i]

        metalpos = np.nonzero(mslice==1)[0]
        nonmetalpos = np.nonzero(mslice==0)[0]
        pnonmetal = pslice[nonmetalpos]
        pslice[metalpos] = interp1d(nonmetalpos,pnonmetal)(metalpos)
        Pinterp[i] = pslice

    return Pinterp

if __name__ == '__main__':
    test_path = 'data/test/'
    res_path  = 'data/generated/'
    mask_path = 'data/mask/'
    config_path = 'data/preprocess/configs/'

    for dir in os.listdir(res_path):
        shutil.rmtree(os.path.join(res_path, dir))
    
    for config_dir in os.listdir(config_path):
        cur_conf = config_path + config_dir
        for config_name in os.listdir(cur_conf):
            clinic_input_data(test_path, res_path, mask_path, os.path.join(cur_conf, config_name))
            # clinic_input_data(test_path, res_path, mask_path, 'data/preprocess/config.yaml')
