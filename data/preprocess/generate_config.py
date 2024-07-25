import yaml
import numpy as np
import os
import shutil

SDD = 2000

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def config_gen(path, dict_file):
    with open(path, 'w') as output:
        yaml.dump(dict_file, output)

if __name__ == '__main__':
    K   = np.linspace(1.2, 5, 3)
    PHI = np.linspace(0.1, 15, 3)
    configs_path = 'data/preprocess/configs/'

    dirs = os.listdir(configs_path)
    for dir in dirs:
        shutil.rmtree(os.path.join(configs_path, dir))

    for k in K:
        for phi in PHI:
            SOD = SDD / k
            dict_file = {'CTpara' : {'imPixNum' : 416,
                                     'angSize' : 0.05,
                                     'linSize' : 1.8536,
                                     'angNum' : 640,
                                     'SOD' : float(SOD),
                                     'imPixScale' : '512 / 416 * 0.03',
                                     'sinogram_size_x' : 640,
                                     'sinogram_size_y' : 641,
                                     'window' : '[-175, 275] / 1000 * 0.192 + 0.192',
                                     'k' : float(k),
                                     'phi' : float(phi)}}
            cur_dir_name = configs_path + 'k=' + f'{k:.2f}'
            mkdir(cur_dir_name)
            config_gen(cur_dir_name + '/phi=' + f'{phi:.2f}' + '.yaml', dict_file)