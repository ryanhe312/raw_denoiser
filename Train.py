from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
from math import ceil
import numpy as np
import keras.backend as K
import pretty_errors

import Model
import Utils

BATCH_SIZE = 4
EPOCHS = 1000
CKPT_PATH = "./ckpt/ckpt.hdf5"
LOG_PATH = "./log"

def generate_data(data):
    while True:
        inputs = []
        outputs = []
        for file_path in data:
            x,y = Utils.get_sample_from_file(file_path)
            inputs.append(x)
            outputs.append(y)
            if len(inputs) == BATCH_SIZE:
                x = K.tf.convert_to_tensor(np.array(inputs))
                y = K.tf.convert_to_tensor(np.array(outputs))
                yield ({'input': x}, {'output': y})
                inputs = []
                outputs = []
        x = K.tf.convert_to_tensor(np.array(inputs))
        y = K.tf.convert_to_tensor(np.array(outputs))
        yield ({'input': x}, {'output': y})

def train(model):
    train_data = []
    for file_path in Utils.get_file_list():
        _,scene_number,_ = Utils.meta_read(file_path.split('/')[-2])
        if scene_number != '001':
           train_data.append(file_path) 

    model_checkpoint = ModelCheckpoint(CKPT_PATH, monitor='loss',verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=LOG_PATH)
    model.fit_generator(generate_data(train_data),\
                        steps_per_epoch = ceil(len(train_data)/BATCH_SIZE),\
                        epochs = EPOCHS,\
                        callbacks=[model_checkpoint,tensorboard])

def test():
    # TODO: complete all test
    pass

def main():
    pass


if __name__=='__main__':
    main()
