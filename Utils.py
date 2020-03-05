import scipy.io
import os
import h5py
import numpy as np

import BayerUnifyAug

SIDD_PATH = '/home/ruianhe/siddplus/train/SIDD_Medium_Raw'

NOISY_PATH = ['_NOISY_RAW_010.MAT','_NOISY_RAW_011.MAT']

MODEL_BAYER = {'GP':'BGGR','IP':'RGGB','S6':'GRBG','N6':'BGGR','G4':'BGGR'}

TARGET_PATTERN = 'BGGR'

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

def mat_seg_comb(mat:np.array,multi:int,mode):
    if mode == 'seg':
        if len(mat.shape) != 2:
            return None

        mat_batch = np.concatenate([np.hsplit(m,multi) for m in np.vsplit(mat,multi)])

        return mat_batch

    if mode == 'comb':
        if len(mat.shape) != 3 or mat.shape[0] != multi*multi:
            return None

        mat_comb  = np.concatenate([np.concatenate(np.vsplit(m,multi),axis=-1) for m in np.vsplit(mat,multi)],axis=1)[0]

        return mat_comb
    return None

def bayer_to_4ch(mat:np.array,mode):
    if len(mat.shape) == 3 and mode == 'bayer':

        b,h,w = mat.shape
        m1 = mat[:,np.repeat(np.arange(0,h,2),w//2),np.tile(np.arange(0,w,2),h//2)]
        m2 = mat[:,np.repeat(np.arange(0,h,2),w//2),np.tile(np.arange(1,w,2),h//2)]
        m3 = mat[:,np.repeat(np.arange(1,h,2),w//2),np.tile(np.arange(0,w,2),h//2)]
        m4 = mat[:,np.repeat(np.arange(1,h,2),w//2),np.tile(np.arange(1,w,2),h//2)]

        return np.stack([m1,m2,m3,m4],-1).reshape(b,h//2,w//2,4)

    if len(mat.shape) == 4 and mode == '4ch':

        mat_bayer = np.empty([mat.shape[0], mat.shape[1]*2, mat.shape[2]*2])
        mat_bayer[:,0::2, 0::2] = mat[:,:,:,0]
        mat_bayer[:,0::2, 1::2] = mat[:,:,:,1]
        mat_bayer[:,1::2, 0::2] = mat[:,:,:,2]
        mat_bayer[:,1::2, 1::2] = mat[:,:,:,3]

        return mat_bayer
    
    return None


def get_mat_from_file(file_path:str):
    noisy_path = file_path
    gt_path = file_path.replace('NOISY', 'GT')
    _,_,bayer_pattern = meta_read(file_path.split('/')[-2])

    return h5py_loadmat(gt_path),h5py_loadmat(noisy_path),bayer_pattern

def aug_unify(img,aug_seed:int,bayer_pattern:str,target_pattern:str,unify_mode:str):
    augment = [aug_seed%2, (aug_seed//2)%2, (aug_seed//4)%2]

    img    = BayerUnifyAug.bayer_unify(img,bayer_pattern,target_pattern,unify_mode)
    img    = BayerUnifyAug.bayer_aug  (img,   augment[0],augment[1],augment[2],target_pattern)

    return img

def random_clip(img,aug_seed:int,patch_size:int,batch_multi:int):
    np.random.seed(aug_seed)
    w, h = img.shape
    
    s_x = (np.random.randint(0, w - batch_multi*patch_size*2)//2)*2
    s_y = (np.random.randint(0, h - batch_multi*patch_size*2)//2)*2
    
    img_clip    = img     [s_x:s_x+batch_multi*patch_size*2,s_y:s_y+batch_multi*patch_size*2]
    img_batch   = mat_seg_comb(img_clip,    batch_multi,'seg') 

    return img_batch

def self_ensemble(img:np.array,mode,bayer_pattern):
    if mode == 'ensemble':
        output = [img]*8

        for i in range(1,8):
            output[i] = aug_unify(output[i],i,bayer_pattern,TARGET_PATTERN,'pad')

        output = np.concatenate(output,axis=0)
        return output

    if mode == 'deensemble':
        output = np.vsplit(img,4)

        for i in range(1,8):
            output[i] = aug_unify(output[i],i,TARGET_PATTERN,bayer_pattern,'pad')

        output = np.mean(output,axis=0)
        return output

    return None

from keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, data:list,patch_size:int,batch_multi:int):
        self.aug_seed = 0
        self.data = data
        self.patch_size = patch_size
        self.batch_multi = batch_multi
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        noisy,gt,bayer = get_mat_from_file(self.data[idx])

        random_seed = np.random.random_integers(65535)
        noisy = aug_unify(noisy,random_seed,bayer,TARGET_PATTERN,'crop')
        noisy = random_clip(noisy,random_seed,self.patch_size,self.batch_multi)
        noisy = bayer_to_4ch(noisy,'bayer')

        gt = aug_unify(gt,random_seed,bayer,TARGET_PATTERN,'crop')
        gt = random_clip(gt,random_seed,self.patch_size,self.batch_multi)
        gt = bayer_to_4ch(gt,'bayer')

        return noisy,gt
    def on_epoch_end(self):
        #self.aug_seed += 1
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
def predict(noisy):
    MODEL_PATH = 'model-resnet/model-64.mdl'
    CKPT_PATH  = "model-resnet/multickpt1-64-adam-0.0001-mae.ckpt"

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"   

    model = load_model(MODEL_PATH,compile=False)
    #model = multi_gpu_model(model,gpus=2)
    model.load_weights(CKPT_PATH)
    denoised = model.predict(noisy)

    return denoised

from cv2 import cvtColor,COLOR_BAYER_RG2BGR,imwrite
def evaluate(noisy,denoised,target):
    JPG_PATH   = 'model-resnet/multickpt1-64-adam-0.0001-mae'

    print('psnr:',psnr(noisy,target),psnr(denoised,target))
    print('ssim:',ssim(noisy,target),ssim(denoised,target))
    
    noisy    = cvtColor(np.array(noisy*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite(os.path.join(JPG_PATH,'noisy.jpg'),noisy)

    denoised = cvtColor(np.array(denoised*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite(os.path.join(JPG_PATH,'denoised.jpg'),denoised)
    
    target   = cvtColor(np.array(target*1024,dtype=np.uint16),COLOR_BAYER_RG2BGR)
    imwrite(os.path.join(JPG_PATH,'target.jpg'),target)

def test():
    test_data = get_file_list('test')
    noisy,target,bayer = get_mat_from_file(test_data[0])
    noisy  = random_clip(noisy ,0,128,1)
    target = random_clip(target,0,128,1)

    noisy_ensem = self_ensemble(noisy,'ensemble',bayer)
    noisy_ensem = bayer_to_4ch(noisy_ensem,'bayer')
    
    denoised_ensem = predict(noisy_ensem)

    noisy_ensem = bayer_to_4ch(noisy_ensem,'4ch')
    denoised = self_ensemble(denoised_ensem,'deensemble',bayer)

    evaluate(noisy,denoised,target)


def main():
    batch_multi = 3
    data = DataGenerator(get_file_list('test'),128,batch_multi)
    noisy,target = data.__getitem__(10)
    print(noisy.shape)
    print('psnr:',psnr(noisy,target))
    print('ssim:',ssim(noisy,target))


if __name__ == '__main__':
    main()