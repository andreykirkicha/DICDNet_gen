import yaml
import numpy as np
import os

SDD = 2000

def config_gen(path, dict_file):
    with open(path, 'w') as output:
        yaml.dump(dict_file, output)

if __name__ == '__main__':
    K   = np.linspace(1.2, 5, 5)
    PHI = np.linspace(0.1, 15, 5)
    configs_path = 'data/preprocess/configs/'

    files = os.listdir(configs_path)
    for file in files:
        os.remove(os.path.join(configs_path, file))

    num_k = 0

    for k in K:
        num_phi = 0

        for phi in PHI:
            SOD = SDD / k
            dict_file = {'CTPara' : {'imPixNum' : 416,
                                     'angSize' : 0.05,
                                     'linSize' : 1.8536,
                                     'angNum' : 640,
                                     'SOD' : float(SOD),
                                     'imPixScale' : '512 / 416 * 0.03',
                                     'sinogram_size_x' : 640,
                                     'sinogram_size_y' : 641,
                                     'window' : '[-175, 275] / 1000 * 0.192 + 0.192'}}
            config_gen(configs_path + 'config_k' + str(num_k) + '_phi' + str(num_phi) + '.yaml', dict_file)

            num_phi += 1
        
        num_k += 1