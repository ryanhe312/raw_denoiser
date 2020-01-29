import scipy.io
import os
import h5py
import numpy as np
import BayerUnifyAug

SIDD_PATH = '/home/ruianhe/siddplus/train/SIDD_Medium_Raw'

NOISY_PATH = ['_NOISY_RAW_010.MAT','_NOISY_RAW_011.MAT']
GT_PATH = ['_GT_RAW_010.MAT','_GT_RAW_011.MAT']

MODEL_BAYER = {'GP':'BGGR','IP':'RGGB','S6':'GRBG','N6':'BGGR','G4':'BGGR'}

PATCH_SIZE = 256
BATCH_SIZE = 4
TARGET_PATTERN = 'BGGR'
UNIFY_MODE = 'crop'

def meta_read(infomation):
    infomation = infomation.split('_')
    scene_instance_number       = int(infomation[0])
    scene_number                = int(infomation[1])
    smartphone_code             = infomation[2]
    ISO_level                   = int(infomation[3])
    shutter_speed               = int(infomation[4])
    illuminant_temperature      = int(infomation[5])
    illuminant_brightness_code  = infomation[6]

    return scene_instance_number,MODEL_BAYER[smartphone_code]

def get_file_list():
    file_tuples = []
    folder_names = os.listdir(os.path.join(SIDD_PATH,'Data'))
    
    for folder_name in folder_names:
        scene_instance_number,bayer_pattern = meta_read(folder_name)
        folder_path = os.path.join(SIDD_PATH,'Data',folder_name)

        noisy_path_1 = os.path.join(folder_path,scene_instance_number+NOISY_PATH[1])
        gt_path_1 = os.path.join(folder_path,scene_instance_number+GT_PATH[1])
        file_tuples.append((noisy_path_1,gt_path_1,bayer_pattern))
        
        noisy_path_2 = os.path.join(folder_path,scene_instance_number+NOISY_PATH[2])
        gt_path_2 = os.path.join(folder_path,scene_instance_number+GT_PATH[2])
        file_tuples.append((noisy_path_2,gt_path_2,bayer_pattern))

    return file_tuples

def h5py_loadmat(file_path):
    with h5py.File(file_path, 'r') as f:
        return np.array(f.get('x'))

def get_sample_from_file(file_tuple):
    noisy_path,gt_path,bayer_pattern = file_tuple

    gt = h5py_loadmat(noisy_path)
    gt = BayerUnifyAug.bayer_unify(gt,bayer_pattern,TARGET_PATTERN,UNIFY_MODE)

    noisy = h5py_loadmat(gt_path)
    noisy = BayerUnifyAug.bayer_unify(noisy,bayer_pattern,TARGET_PATTERN,UNIFY_MODE)

    augment = np.random.choice(1,3)
    gt = BayerUnifyAug.bayer_aug(gt,augment[0],augment[1],augment[2],TARGET_PATTERN)
    noisy = BayerUnifyAug.bayer_aug(noisy,augment[0],augment[1],augment[2],TARGET_PATTERN)

    h, w = gt.shape
    
    s_x = (np.random.random_integers(0, w - PATCH_SIZE*2)//2)*2
    s_y = (np.random.random_integers(0, h - PATCH_SIZE*2)//2)*2
    e_x = s_x + PATCH_SIZE*2
    e_y = s_y + PATCH_SIZE*2
    
    gt = gt[s_x:e_x,s_y:e_y,:]
    noisy = noisy[s_x:e_x,s_y:e_y,:]

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
