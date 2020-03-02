import scipy.io
import os
import h5py
import numpy as np

import BayerUnifyAug

SIDD_PATH = '/home/ruianhe/siddplus/train/SIDD_Medium_Raw'

NOISY_PATH = ['_NOISY_RAW_010.MAT','_NOISY_RAW_011.MAT']

MODEL_BAYER = {'GP':'BGGR','IP':'RGGB','S6':'GRBG','N6':'BGGR','G4':'BGGR'}

TARGET_PATTERN = 'BGGR'
UNIFY_MODE = 'crop'

DATA_TYPE = ['train','test']
TEST_SCENE = '001'

def meta_read(info:str):
    info = info.split('_')
    scene_instance_number       = info[0]
    scene_number                = info[1]
    smartphone_code             = info[2]
    #ISO_level                   = info[3]
    #shutter_speed               = info[4]
    #illuminant_temperature      = info[5]
    #illuminant_brightness_code  = info[6]

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

    mat_batch = np.concatenate([np.hsplit(m,multi) for m in np.vsplit(mat,multi)])

    return mat_batch

def mat_comb(mat:np.array,multi:int):
    if len(mat.shape) != 3 or mat.shape[0] != multi*multi:
        return None

    mat_comb  = np.concatenate([np.concatenate(np.vsplit(m,multi),axis=-1) for m in np.vsplit(mat,multi)],axis=1)[0]

    return mat_comb

def bayer_to_4ch(mat:np.array):
    if len(mat.shape) == 3:

        b,h,w = mat.shape
        m1 = mat[:,np.repeat(np.arange(0,h,2),w//2),np.tile(np.arange(0,w,2),h//2)]
        m2 = mat[:,np.repeat(np.arange(0,h,2),w//2),np.tile(np.arange(1,w,2),h//2)]
        m3 = mat[:,np.repeat(np.arange(1,h,2),w//2),np.tile(np.arange(0,w,2),h//2)]
        m4 = mat[:,np.repeat(np.arange(1,h,2),w//2),np.tile(np.arange(1,w,2),h//2)]

        return np.stack([m1,m2,m3,m4],-1).reshape(b,h//2,w//2,4)

    if len(mat.shape) == 4:

        mat_bayer = np.empty([mat.shape[0], mat.shape[1]*2, mat.shape[2]*2])
        mat_bayer[:,0::2, 0::2] = mat[:,:,:,0]
        mat_bayer[:,0::2, 1::2] = mat[:,:,:,1]
        mat_bayer[:,1::2, 0::2] = mat[:,:,:,2]
        mat_bayer[:,1::2, 1::2] = mat[:,:,:,3]

        return mat_bayer
    
    return None


def get_sample_from_file(file_path:str,patch_size:int,batch_multi:int):
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
    
    s_x = (np.random.random_integers(0, w - batch_multi*patch_size*2)//2)*2
    s_y = (np.random.random_integers(0, h - batch_multi*patch_size*2)//2)*2
    
    gt_clip    = gt     [s_x:s_x+batch_multi*patch_size*2,s_y:s_y+batch_multi*patch_size*2]
    noisy_clip = noisy  [s_x:s_x+batch_multi*patch_size*2,s_y:s_y+batch_multi*patch_size*2]

    gt_batch    =   mat_seg(gt_clip,    batch_multi) 
    noisy_batch =   mat_seg(noisy_clip, batch_multi)

    gt_4ch    = bayer_to_4ch(gt_batch)
    noisy_4ch = bayer_to_4ch(noisy_batch)

    return noisy_4ch, gt_4ch

from keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, data:list,patch_size:int,batch_multi:int):
        self.data = data
        self.patch_size = patch_size
        self.batch_multi = batch_multi
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return get_sample_from_file(self.data[idx],self.patch_size,self.batch_multi)
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

from keras.utils import multi_gpu_model
from keras.models import load_model
from cv2 import cvtColor,COLOR_BAYER_RG2BGR,imwrite
def test(noisy,target,batch_multi):
    MODEL_PATH = 'model-resnet/model-64.mdl'
    CKPT_PATH  = "model-resnet/multickpt1-64-adam-0.0001-mae.ckpt"
    JPG_PATH   = 'model-resnet/multickpt1-64-adam-0.0001-mae'

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"   

    model = load_model(MODEL_PATH,compile=False)
    model = multi_gpu_model(model,gpus=2)
    model.load_weights(CKPT_PATH)
    denoised = model.predict(noisy)

    print('psnr:',psnr(noisy,target),psnr(denoised,target))
    print('ssim:',ssim(noisy,target),ssim(denoised,target))
    
    noisy    = mat_comb(bayer_to_4ch(noisy),batch_multi)
    noisy    = cvtColor(np.array(noisy*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite(os.path.join(JPG_PATH,'noisy.jpg'),noisy)

    denoised = mat_comb(bayer_to_4ch(denoised),batch_multi)
    denoised = cvtColor(np.array(denoised*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite(os.path.join(JPG_PATH,'denoised.jpg'),denoised)
    
    target   = mat_comb(bayer_to_4ch(target),batch_multi)
    target   = cvtColor(np.array(target*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite(os.path.join(JPG_PATH,'target.jpg'),target)

def main():
    batch_multi = 4
    data = DataGenerator(get_file_list('test'),64,batch_multi)
    noisy,target = data.__getitem__(10)
    test(noisy,target,batch_multi)
    #print('psnr:',psnr(noisy,target))
    #print('ssim:',ssim(noisy,target))


if __name__ == '__main__':
    main()