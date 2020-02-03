from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from math import ceil
from os import environ

import Model 
import Utils

BATCH_SIZE = 4
EPOCHS = 150

CKPT_PATH = "./ckpt/ckpt.ckpt"
MODEL_PATH = './ckpt/model.mdl'

LOG_PATH = "./log"
LOSS_PATH = "./loss.txt"

environ["CUDA_VISIBLE_DEVICES"] = "0"       

def train(model):
    train_data = Utils.get_file_list('train')

    checkpoint = ModelCheckpoint(CKPT_PATH, monitor='loss',verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=LOG_PATH)
    history = model.fit_generator(Utils.DataGenerator(train_data,BATCH_SIZE),\
                                steps_per_epoch = int(ceil(len(train_data)/BATCH_SIZE)),\
                                epochs = EPOCHS,\
                                verbose = 1,\
                                callbacks=[checkpoint,tensorboard])
    #validation_data = Utils.DataGenerator(val_data,BATCH_SIZE)
    #validation_steps = int(np.ceil(len(val_data)/BATCH_SIZE))
    return history

def test(model):
    test_data = Utils.get_file_list('test')

    results = model.evaluate_generator(Utils.DataGenerator(test_data,BATCH_SIZE),\
                                       steps = int(ceil(len(test_data)/BATCH_SIZE)),\
                                       verbose = 1)
    return results

def main():
    model = load_model(MODEL_PATH)
    #model.load_weights(CKPT_PATH)
    
    history = train(model)
    loss = open(LOSS_PATH,'a')
    loss.writelines('\n'.join(history.history['loss'])+'\n')
    loss.close()

    #results = test(model)
    #print(dict(zip(model.metrics_names,results)))

if __name__=='__main__':
    main()