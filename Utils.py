import scipy.io
import os
import h5py
import numpy as np
import pretty_errors

import BayerUnifyAug

SIDD_PATH = '/home/ruianhe/siddplus/train/SIDD_Medium_Raw'

NOISY_PATH = ['_NOISY_RAW_010.MAT','_NOISY_RAW_011.MAT']

MODEL_BAYER = {'GP':'BGGR','IP':'RGGB','S6':'GRBG','N6':'BGGR','G4':'BGGR'}

PATCH_SIZE = 256
TARGET_PATTERN = 'BGGR'
UNIFY_MODE = 'crop'

def meta_read(info):
    info = info.split('_')
    scene_instance_number       = info[0]
    scene_number                = info[1]
    smartphone_code             = info[2]
    ISO_level                   = info[3]
    shutter_speed               = info[4]
    illuminant_temperature      = info[5]
    illuminant_brightness_code  = info[6]

    return scene_instance_number,scene_number,MODEL_BAYER[smartphone_code]

def get_file_list():
    file_lists = []
    folder_names = os.listdir(os.path.join(SIDD_PATH,'Data'))

    for folder_name in folder_names:
        scene_instance_number,_,_ = meta_read(folder_name)
        for path in NOISY_PATH:
            file_lists.append(os.path.join(SIDD_PATH,'Data',folder_name,scene_instance_number+path))

    return file_lists

def h5py_loadmat(file_path):
    with h5py.File(file_path, 'r') as f:
        return np.array(f.get('x'))

def get_sample_from_file(file_path):
    noisy_path = file_path
    gt_path = file_path.replace('NOISY', 'GT')
    _,_,bayer_pattern = meta_read(file_path.split('/')[-2])

    gt = h5py_loadmat(noisy_path)
    gt = BayerUnifyAug.bayer_unify(gt,bayer_pattern,TARGET_PATTERN,UNIFY_MODE)

    noisy = h5py_loadmat(gt_path)
    noisy = BayerUnifyAug.bayer_unify(noisy,bayer_pattern,TARGET_PATTERN,UNIFY_MODE)

    augment = np.random.choice(1,3)
    gt = BayerUnifyAug.bayer_aug(gt,augment[0],augment[1],augment[2],TARGET_PATTERN)
    noisy = BayerUnifyAug.bayer_aug(noisy,augment[0],augment[1],augment[2],TARGET_PATTERN)

    w, h = gt.shape
    
    s_x = (np.random.random_integers(0, w - PATCH_SIZE*2)//2)*2
    s_y = (np.random.random_integers(0, h - PATCH_SIZE*2)//2)*2
    e_x = s_x + PATCH_SIZE*2
    e_y = s_y + PATCH_SIZE*2
    
    gt = gt[s_x:e_x, s_y:e_y]
    noisy = noisy[s_x:e_x, s_y:e_y]

    gt_4ch = np.empty([PATCH_SIZE, PATCH_SIZE, 4])
    noisy_4ch = np.empty([PATCH_SIZE, PATCH_SIZE, 4])
    
    gt_4ch[:,:,0] = gt[0::2, 0::2]
    gt_4ch[:,:,1] = gt[0::2, 1::2]
    gt_4ch[:,:,2] = gt[1::2, 0::2]
    gt_4ch[:,:,3] = gt[1::2, 1::2]
    
    noisy_4ch[:,:,0] = noisy[0::2, 0::2]
    noisy_4ch[:,:,1] = noisy[0::2, 1::2]
    noisy_4ch[:,:,2] = noisy[1::2, 0::2]
    noisy_4ch[:,:,3] = noisy[1::2, 1::2]

    return noisy_4ch, gt_4ch

def main():
    pass

if __name__ == '__main__':
    main()