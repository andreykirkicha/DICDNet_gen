# Given clinical Xma, generate data, including: XLI, M, Sma, SLI, Tr for infering InDuDoNet
import numpy as np
import os
from scipy.interpolate import interp1d, RegularGridInterpolator
from .utils import get_config
from .build_gemotry import initialization, imaging_geo
import PIL
from PIL import Image

config = get_config('data/preprocess_clinic/dataset_py_640geo.yaml')
CTpara = config['CTpara']                       # CT imaging parameters

mask_thre = 2500 / 1000 * 0.192 + 0.192         # taking 2500HU as a thresholding to segment the metal region

param = initialization()
ray_trafo, FBPOper = imaging_geo(param)         # CT imaging geometry, ray_trafo is fp, FBPoper is fbp

allXma = []
allXLI = []
allM = []
allSma = []
allSLI = []
allTr = []
allaffine = []
allfilename = []

# process and save all the to-the-tested volumes
def clinic_input_data(test_path, res_path, mask_path):
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

    names = ['M', 'SLI', 'Sma', 'Tr', 'XLI', 'Xma']

    for name in names:
        current_path = os.path.join(res_path, name)
        files = os.listdir(current_path)
        for file in files:
            os.remove(os.path.join(current_path, file))

    for file_name in os.listdir(test_path):
        file_path = os.path.join(test_path, file_name)

        # for DeepLesion
        # image = (open_image(file_path, 'float32').astype(np.int32) - 32768).astype(np.float32)
        # image = image / 1000 * 0.192 + 0.192
        # image = np.maximum(image, 0)

        # for CLINIC-metal
        image = open_image(file_path, 'float32')

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
            cur_mask_path = os.path.join(mask_path, mask_name)
            mask = open_image(cur_mask_path, 'float32')

            # put metal into slice
            Xgt = np.copy(image)
            [x, y] = np.where(mask != 0)
            Xgt[x, y] = mask[x, y]                                      
            
            # segment metal region
            [rowindex, colindex] = np.where(Xgt > mask_thre)
            M.fill(0.0)
            M[rowindex, colindex] = 1                 
            
            Pmetal_kev = np.asarray(ray_trafo(M))         
            Tr = Pmetal_kev > 0      
            
            # polychromatic radiation
            Sgt = np.asarray(ray_trafo(Xgt))

            rho = 0.4505
            total_sum = 0
            for i in range(mat_grid.shape[0]):
                total_sum += spec[i, 1]
                lin = Tr * mat_grid[i] / rho + Sgt
                Sma += spec[i, 1] * np.exp(-lin)
            Sma = -np.log(Sma/total_sum)
            Xma = np.asarray(FBPOper(Sma))

            # to match metal region of gt and ma images
            [row, col] = np.where(Xma > mask_thre)
            Xgt[row, col] = Xma[row, col]
            
            # linear interpolation
            SLI = interpolate_projection(Sma, Tr)
            XLI = np.asarray(FBPOper(SLI))

            # visualization
            print("\n======================== ...saving... ========================\n")
            save_as_image(Xma, img_num, mask_num, res_path, 'Xma')
            save_as_image(Xgt, img_num, mask_num, 'results/', 'Xgt')
            save_as_image(M, img_num, mask_num, res_path, 'M')
            save_as_image(Tr, img_num, mask_num, res_path, 'Tr')
            save_as_image(Sma, img_num, mask_num, res_path, 'Sma')
            save_as_image(SLI, img_num, mask_num, res_path, 'SLI')
            save_as_image(XLI, img_num, mask_num, res_path, 'XLI')

            allXma.append(Xma)
            allXLI.append(XLI)
            allM.append(M)
            allSma.append(Sma)
            allSLI.append(SLI)
            allTr.append(Tr)
            allfilename.append(file_name)

            mask_num += 1

        img_num += 1

    return allXma, allXLI, allM, allSma, allSLI, allTr, allfilename

def open_image(file_path, d_type='float32'):
    img = np.array(Image.open(file_path), dtype=d_type)      # ndarray
    return np.array(Image.fromarray(img).resize((CTpara['imPixNum'], CTpara['imPixNum']), PIL.Image.BILINEAR))     # resize image

def save_as_image(array, img_num, mask_num, res_path, name):
    for_image = array.astype(np.float32)
    image = Image.fromarray(for_image)
    
    cur_path = os.path.join(res_path, name)
    
    image.save(os.path.join(cur_path, name + '_img' + str(img_num) + '_mask' + str(mask_num) + '.tif'))
    print(name + '\t image saved as ' + name + '_img' + str(img_num) + '_mask' + str(mask_num) + '.tif')

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

    names = ['M', 'SLI', 'Sma', 'Tr', 'XLI', 'Xma']

    for name in names:
        current_path = os.path.join(res_path, name)
        files = os.listdir(current_path)
        for file in files:
            os.remove(os.path.join(current_path, file))
    
    clinic_input_data(test_path, res_path, mask_path)
