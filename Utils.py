import scipy.io
import os
import h5py
import numpy as np

import BayerUnifyAug

SIDD_PATH = '/home/ruianhe/siddplus/train/SIDD_Medium_Raw'

NOISY_PATH = ['_NOISY_RAW_010.MAT','_NOISY_RAW_011.MAT']

MODEL_BAYER = {'GP':'BGGR','IP':'RGGB','S6':'GRBG','N6':'BGGR','G4':'BGGR'}

PATCH_SIZE = 128
BATCH_LENGTH = 3
TARGET_PATTERN = 'BGGR'
UNIFY_MODE = 'crop'

DATA_TYPE = ['train','test']
TEST_SCENE = '001'

def meta_read(info:str):
    info = info.split('_')
    scene_instance_number       = info[0]
    scene_number                = info[1]
    smartphone_code             = info[2]
    ISO_level                   = info[3]
    shutter_speed               = info[4]
    illuminant_temperature      = info[5]
    illuminant_brightness_code  = info[6]

    return scene_instance_number,scene_number,MODEL_BAYER[smartphone_code]

def get_file_list(data_type:str):
    if data_type not in DATA_TYPE:
        return None

    file_lists = []
    folder_names = os.listdir(os.path.join(SIDD_PATH,'Data'))

    for folder_name in folder_names:
        scene_instance_number,scene_number,_ = meta_read(folder_name)
        if data_type == 'train' and  scene_number == TEST_SCENE:
            continue
        if data_type == 'test'  and  scene_number != TEST_SCENE:
            continue
        for path in NOISY_PATH:
            file_lists.append(os.path.join(SIDD_PATH,'Data',folder_name,scene_instance_number+path))

    return file_lists

def h5py_loadmat(file_path:str):
    with h5py.File(file_path, 'r') as f:
        return np.array(f.get('x'),dtype=np.float32)

def get_sample_from_file(file_path:str):
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
    
    s_x = (np.random.random_integers(0, w - BATCH_LENGTH*PATCH_SIZE*2)//2)*2
    s_y = (np.random.random_integers(0, h - BATCH_LENGTH*PATCH_SIZE*2)//2)*2
    
    gt_batch    = np.empty([BATCH_LENGTH**2, PATCH_SIZE*2, PATCH_SIZE*2])
    noisy_batch = np.empty([BATCH_LENGTH**2, PATCH_SIZE*2, PATCH_SIZE*2])

    for x in range(BATCH_LENGTH):
        for y in range(BATCH_LENGTH):
            e_x = s_x+x*PATCH_SIZE*2
            e_y = s_y+y*PATCH_SIZE*2
            gt_batch    [y+x*BATCH_LENGTH,:,:] = gt     [e_x:e_x+PATCH_SIZE*2,e_y:e_y+PATCH_SIZE*2]
            noisy_batch [y+x*BATCH_LENGTH,:,:] = noisy  [e_x:e_x+PATCH_SIZE*2,e_y:e_y+PATCH_SIZE*2]

    gt_4ch    = np.empty([BATCH_LENGTH**2, PATCH_SIZE, PATCH_SIZE, 4])
    noisy_4ch = np.empty([BATCH_LENGTH**2, PATCH_SIZE, PATCH_SIZE, 4])
    
    gt_4ch[:,:,:,0] = gt_batch[:,0::2, 0::2]
    gt_4ch[:,:,:,1] = gt_batch[:,0::2, 1::2]
    gt_4ch[:,:,:,2] = gt_batch[:,1::2, 0::2]
    gt_4ch[:,:,:,3] = gt_batch[:,1::2, 1::2]
    
    noisy_4ch[:,:,:,0] = noisy_batch[:,0::2, 0::2]
    noisy_4ch[:,:,:,1] = noisy_batch[:,0::2, 1::2]
    noisy_4ch[:,:,:,2] = noisy_batch[:,1::2, 0::2]
    noisy_4ch[:,:,:,3] = noisy_batch[:,1::2, 1::2]

    return noisy_4ch, gt_4ch

from keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, data:list):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        file_path = self.data[idx]
        return get_sample_from_file(file_path)
    def on_epoch_end(self):
        np.random.shuffle(self.data)

def psnr(y_true, y_pred):
    rmse = np.mean(np.power(y_true.flatten() - y_pred.flatten(), 2))
    return 10 * np.log(1.0 / rmse)/np.log(10.)

def ssim(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def main():
    test = DataGenerator(get_file_list('test'))
    x,y = test.__getitem__(1)
    print('psnr:',psnr(x,y))
    print('ssim:',ssim(x,y))

if __name__ == '__main__':
    main()