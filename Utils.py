import scipy.io
import os
import h5py
import numpy as np

import BayerUnifyAug

SIDD_PATH = '/home/ruianhe/siddplus/train/SIDD_Medium_Raw'

NOISY_PATH = ['_NOISY_RAW_010.MAT','_NOISY_RAW_011.MAT']

MODEL_BAYER = {'GP':'BGGR','IP':'RGGB','S6':'GRBG','N6':'BGGR','G4':'BGGR'}

PATCH_SIZE = 128
BATCH_MULTI = 3
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

def mat_seg(mat:np.array,multi:int):
    if len(mat.shape) != 2:
        return None

    mat_batch    = np.empty([multi*multi, mat.shape[0]//multi, mat.shape[1]//multi])

    for x in range(multi):
        for y in range(multi):

            s_x = x*(mat.shape[0]//multi)
            e_x = (x+1)*(mat.shape[0]//multi)

            s_y = y*(mat.shape[1]//multi)
            e_y = (y+1)*(mat.shape[1]//multi)

            mat_batch[y+x*multi,:,:] = mat[s_x:e_x,s_y:e_y]
    return mat_batch

def mat_comb(mat:np.array,multi:int):
    if len(mat.shape) != 3 or mat.shape[0] != multi*multi:
        return None

    mat_comb    = np.empty([mat.shape[1]*multi, mat.shape[2]*multi])

    for x in range(multi):
        for y in range(multi):

            s_x = x*(mat.shape[1])
            e_x = (x+1)*(mat.shape[1])

            s_y = y*(mat.shape[2])
            e_y = (y+1)*(mat.shape[2])

            mat_comb[s_x:e_x,s_y:e_y] = mat[y+x*multi,:,:]
    return mat_comb

def bayer_to_4ch(mat:np.array):
    if len(mat.shape) == 3:

        mat_4ch = np.empty([mat.shape[0], mat.shape[1]//2, mat.shape[2]//2, 4])
        mat_4ch[:,:,:,0] = mat[:,0::2, 0::2]
        mat_4ch[:,:,:,1] = mat[:,0::2, 1::2]
        mat_4ch[:,:,:,2] = mat[:,1::2, 0::2]
        mat_4ch[:,:,:,3] = mat[:,1::2, 1::2]

        return mat_4ch

    if len(mat.shape) == 4:

        mat_bayer = np.empty([mat.shape[0], mat.shape[1]*2, mat.shape[2]*2])
        mat_bayer[:,0::2, 0::2] = mat[:,:,:,0]
        mat_bayer[:,0::2, 1::2] = mat[:,:,:,1]
        mat_bayer[:,1::2, 0::2] = mat[:,:,:,2]
        mat_bayer[:,1::2, 1::2] = mat[:,:,:,3]

        return mat_bayer
    
    return None


def get_sample_from_file(file_path:str):
    noisy_path = file_path
    gt_path = file_path.replace('NOISY', 'GT')
    _,_,bayer_pattern = meta_read(file_path.split('/')[-2])

    gt    = h5py_loadmat(gt_path)
    gt    = BayerUnifyAug.bayer_unify(gt,bayer_pattern,TARGET_PATTERN,UNIFY_MODE)

    noisy = h5py_loadmat(noisy_path)
    noisy = BayerUnifyAug.bayer_unify(noisy,bayer_pattern,TARGET_PATTERN,UNIFY_MODE)

    augment = np.random.choice(1,3)
    gt    = BayerUnifyAug.bayer_aug(gt,   augment[0],augment[1],augment[2],TARGET_PATTERN)
    noisy = BayerUnifyAug.bayer_aug(noisy,augment[0],augment[1],augment[2],TARGET_PATTERN)

    w, h = gt.shape
    
    s_x = (np.random.random_integers(0, w - BATCH_MULTI*PATCH_SIZE*2)//2)*2
    s_y = (np.random.random_integers(0, h - BATCH_MULTI*PATCH_SIZE*2)//2)*2
    
    gt_clip    = gt     [s_x:s_x+BATCH_MULTI*PATCH_SIZE*2,s_y:s_y+BATCH_MULTI*PATCH_SIZE*2]
    noisy_clip = noisy  [s_x:s_x+BATCH_MULTI*PATCH_SIZE*2,s_y:s_y+BATCH_MULTI*PATCH_SIZE*2]

    gt_batch    =   mat_seg(gt_clip,    BATCH_MULTI) 
    noisy_batch =   mat_seg(noisy_clip, BATCH_MULTI)

    gt_4ch    = bayer_to_4ch(gt_batch)
    noisy_4ch = bayer_to_4ch(noisy_batch)

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

from keras.models import load_model
from cv2 import cvtColor,COLOR_BAYER_RG2BGR,imwrite
def test(noisy,target):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"    
    model = load_model('./model-resnet/model-128.mdl',compile=False)
    model.load_weights("./model-resnet/ckpt-128-mae-adam-0.0002.ckpt")
    denoised = model.predict(noisy)
    
    noisy    = mat_comb(bayer_to_4ch(noisy),BATCH_MULTI)
    noisy    = cvtColor(np.array(noisy*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite('noisy.jpg',noisy)

    denoised = mat_comb(bayer_to_4ch(denoised),BATCH_MULTI)
    denoised = cvtColor(np.array(denoised*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite('denoised.jpg',denoised)
    
    target   = mat_comb(bayer_to_4ch(target),BATCH_MULTI)
    target   = cvtColor(np.array(target*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite('target.jpg',target)

def main():
    data = DataGenerator(get_file_list('test'))
    noisy,target = data.__getitem__(10)
    print(noisy.shape)
    print('psnr:',psnr(noisy,target))
    print('ssim:',ssim(noisy,target))


if __name__ == '__main__':
    main()