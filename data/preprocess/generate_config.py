import yaml
import numpy as np
import os
import shutil

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def config_gen(path, dict_file):
    with open(path, 'w') as output:
        yaml.dump(dict_file, output)

if __name__ == '__main__':
    detPixNum = 640
    imPixNum = 416
    voxSize = 512 / 416 * 0.03
    sx = voxSize * imPixNum
    sy = voxSize * imPixNum
    k = 2

    RHO = np.linspace(0.01, 3, 12)
    PHI = np.linspace(0.2, 50, 8)
    
    configs_path = 'data/preprocess/configs/'

    dirs = os.listdir(configs_path)
    for dir in dirs:
        shutil.rmtree(os.path.join(configs_path, dir))

    for rho in RHO:
        for phi in PHI:
            detSize = 2 * np.sqrt(sx ** 2 + sy ** 2) * k / 2
            pixSize = detSize / detPixNum
            phi_rad = phi * np.pi / 180
            SDD = detSize / np.abs(np.tan(phi_rad))
            SOD = SDD / k

            dict_file = {'CTpara' : {'imPixNum' : int(imPixNum),
                                     'angSize' : 0.05,
                                     'linSize' : 1.8536,
                                     'pixSize' : float(pixSize),
                                     'voxSize' : float(voxSize),
                                     'detPixNum' : int(detPixNum),
                                     'detSize' : float(detSize),
                                     'angNum' : 640,
                                     'SDD' : float(SDD),
                                     'SOD' : float(SOD),
                                     'sinogram_size_x' : 640,
                                     'sinogram_size_y' : 641,
                                     'window' : '[-175, 275] / 1000 * 0.192 + 0.192',
                                     'k' : float(k),
                                     'rho' : float(rho),
                                     'phi' : float(phi),
                                     'sx' : float(sx),
                                     'sy' : float(sy)}}
            cur_dir_name = configs_path + 'rho=' + f'{rho:.2f}'
            mkdir(cur_dir_name)
            config_gen(cur_dir_name + '/phi=' + f'{phi:.2f}' + '.yaml', dict_file)