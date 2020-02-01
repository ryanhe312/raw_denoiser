from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from math import ceil
import os
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

from Model import get_unet
import Utils

BATCH_SIZE = 4
EPOCHS = 1000
CKPT_PATH = "./ckpt/ckpt.mdl"
LOG_PATH = "./log"
FIG_PATH = "./log/loss.jpg"

def generate_data(data:list):
    while True:
        inputs = []
        outputs = []
        for file_path in data:
            x,y = Utils.get_sample_from_file(file_path)
            inputs.append(x)
            outputs.append(y)
            if len(inputs) == BATCH_SIZE:
                x = np.array(inputs)
                y = np.array(outputs)
                yield ({'input': x}, {'output': y})
                inputs = []
                outputs = []
        x = np.array(inputs)
        y = np.array(outputs)
        yield ({'input': x}, {'output': y})

def train(model:Model):
    train_data = []
    for file_path in Utils.get_file_list():
        _,scene_number,_ = Utils.meta_read(file_path.split('/')[-2])
        if scene_number != '001':
           train_data.append(file_path) 

    checkpoint = ModelCheckpoint(CKPT_PATH, monitor='loss',verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=LOG_PATH)
    history = model.fit_generator(generate_data(train_data),\
                        steps_per_epoch = ceil(len(train_data)/BATCH_SIZE),\
                        epochs = EPOCHS,\
                        callbacks=[checkpoint,tensorboard])
    return history

def test(model:Model):
    test_data = []
    for file_path in Utils.get_file_list():
        _,scene_number,_ = Utils.meta_read(file_path.split('/')[-2])
        if scene_number == '001':
           test_data.append(file_path) 

    results = model.evaluate_generator(generate_data(test_data),\
                                       steps_per_epoch = ceil(len(test_data)/BATCH_SIZE))
    return results

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    model = get_unet()
    model.summary()

    history = train(model)
    plt.figure()
    plt.plot(history.history)
    plt.savefig(FIG_PATH)
    #results = test(model)
    #print(zip(model.metrics_names,results))

if __name__=='__main__':
    main()
