from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
import os
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

from Model import get_unet
from Utils import get_file_list, DataGenerator

BATCH_SIZE = 4
EPOCHS = 300
CKPT_PATH = "./ckpt/ckpt.mdl"
LOG_PATH = "./log"
FIG_PATH = "./loss.jpg"
        

def train(model:Model):
    train_data = get_file_list('train')

    checkpoint = ModelCheckpoint(CKPT_PATH, monitor='loss',verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=LOG_PATH)
    history = model.fit_generator(DataGenerator(train_data,BATCH_SIZE),\
                                steps_per_epoch = int(np.ceil(len(train_data)/BATCH_SIZE)),\
                                epochs = EPOCHS,\
                                verbose = 1,\
                                callbacks=[checkpoint,tensorboard])
    #validation_data = DataGenerator(val_data,BATCH_SIZE)
    #validation_steps = int(np.ceil(len(val_data)/BATCH_SIZE))
    return history

def test(model:Model):
    test_data = get_file_list('test')

    results = model.evaluate_generator(DataGenerator(test_data,BATCH_SIZE),\
                                       steps = int(np.ceil(len(test_data)/BATCH_SIZE)),\
                                       verbose = 1)
    return results

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = get_unet()
    model.load_weights(CKPT_PATH)
    history = train(model)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.savefig(FIG_PATH)

    #results = test(model)
    #print(dict(zip(model.metrics_names,results)))

if __name__=='__main__':
    main()