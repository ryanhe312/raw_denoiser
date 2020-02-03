#!/usr/bin/env python

import numpy as np
import os
import os.path
import shutil
from scipy.io.matlab.mio import savemat, loadmat

from keras.models import load_model
import Model

CKPT_PATH = './ckpt/ckpt.ckpt'
MODEL_PATH = './ckpt/model.mdl' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = load_model(MODEL_PATH)
model.load_weights(CKPT_PATH)

def denoiser(noisy):
    denoised = model.predict(noisy)
    return denoised


# TODO: Download noisy images from:
#  https://competitions.codalab.org/my/datasets/download/4d26bd29-ab8b-4fe7-8fa1-7a33b32154c7

# TODO: update your working directory; it should contain the .mat file containing noisy images
work_dir = '/home/ruianhe/siddplus/valid'

# load noisy images
noisy_fn = 'siddplus_valid_noisy_raw.mat'
noisy_key = 'siddplus_valid_noisy_raw'
noisy_mat = loadmat(os.path.join(work_dir, noisy_fn))[noisy_key]

# denoise
n_im, h, w = noisy_mat.shape
#results = noisy_mat.copy()
#for i in range(n_im):
#    noisy = np.reshape(noisy_mat[i, :, :], (h, w))
#    denoised = denoiser(noisy)
#    results[i, :, :] = denoised
noisy_4ch = np.empty([n_im,h//2, w//2, 4])
noisy_4ch[:,:,:,0] = noisy_mat[:,0::2, 0::2]
noisy_4ch[:,:,:,1] = noisy_mat[:,0::2, 1::2]
noisy_4ch[:,:,:,2] = noisy_mat[:,1::2, 0::2]
noisy_4ch[:,:,:,3] = noisy_mat[:,1::2, 1::2]
results = denoiser(noisy_4ch)

# create results directory
res_dir = 'res_dir'
os.makedirs(os.path.join(work_dir, res_dir), exist_ok=True)

# save denoised images in a .mat file with dictionary key "results"
res_fn = os.path.join(work_dir, res_dir, 'results.mat')
res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
savemat(res_fn, {res_key: results})

# submission indormation
# TODO: update the values below; the evaluation code will parse them
runtime = 0.0  # seconds / megapixel
cpu_or_gpu = 0  # 0: GPU, 1: CPU
use_metadata = 0  # 0: no use of metadata, 1: metadata used
other = ''

# prepare and save readme file
readme_fn = os.path.join(work_dir, res_dir, 'readme.txt')  # Note: do not change 'readme.txt'
with open(readme_fn, 'w') as readme_file:
    readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
    readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
    readme_file.write('Metadata[1] / No Metadata[0]: %s\n' % str(use_metadata))
    readme_file.write('Other description: %s\n' % str(other))

# compress results directory
res_zip_fn = 'results_dir'
shutil.make_archive(os.path.join(work_dir, res_zip_fn), 'zip', os.path.join(work_dir, res_dir))

#  TODO: upload the compressed .zip file here:
#  https://competitions.codalab.org/competitions/22230#participate-submit_results
