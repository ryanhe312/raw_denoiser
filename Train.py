from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import mse
from math import ceil
from os import environ
import keras.backend as K

import Model 
import Utils

BATCH_SIZE = 4
EPOCHS = 100

CKPT_PATH = "./ckpt/ckpt.ckpt"
MODEL_PATH = './ckpt/model-128.mdl'

LOG_PATH = "./log"
LOSS_PATH = "./loss.txt"

environ["CUDA_VISIBLE_DEVICES"] = "0"      

def psnr(y_true, y_pred):
    rmse = K.mean(K.pow(K.flatten(y_true - y_pred), 2))
    return 10 * K.log(1.0 / rmse)/K.log(10.)

def ssim(y_true , y_pred):
    u_true = K.mean(y_true)
    u_pred = K.mean(y_pred)
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = K.square(0.01*7)
    c2 = K.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def train(model):
    train_data = Utils.get_file_list('train')

    checkpoint = ModelCheckpoint(CKPT_PATH, monitor='loss',verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=LOG_PATH)
    
    model.compile(optimizer=Adam(lr=2e-4,decay=2e-5,amsgrad=True), loss=mse, metrics=[psnr,ssim])
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
    model = load_model(MODEL_PATH,compile=False)
    #model.load_weights(CKPT_PATH)
    
    history = train(model)
    log = open(LOSS_PATH,'a')
    log.writelines([str(loss)+'\n' for loss in history.history['loss']])
    log.close()

    #results = test(model)
    #print(dict(zip(model.metrics_names,results)))

if __name__=='__main__':
    main()