from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.losses import mse,mae
from math import ceil
from os import environ,makedirs,path
import keras.backend as K

import Model 
import Utils

EPOCHS = 10 

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

def train(model,ckpt_path,log_path):
    train_data = Utils.get_file_list('train')
    val_data = Utils.get_file_list('test')

    makedirs(path.dirname(ckpt_path),exist_ok=True)
    makedirs(log_path,exist_ok=True)

    checkpoint = ModelCheckpoint(ckpt_path, monitor='loss',verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=log_path)
    
    history = model.fit_generator(Utils.DataGenerator(train_data),\
                                steps_per_epoch = len(train_data),\
                                epochs = EPOCHS,\
                                verbose = 2,\
                                callbacks=[checkpoint,tensorboard],\
                                validation_data = Utils.DataGenerator(val_data),\
                                validation_steps = len(val_data))
    return history

def test(model):
    test_data = Utils.get_file_list('test')

    results = model.evaluate_generator(Utils.DataGenerator(test_data),\
                                       steps = len(test_data),\
                                       verbose = 0)
    return results

def main():
    environ["CUDA_VISIBLE_DEVICES"] = "2"    
    size, opt, lr, loss= 128, 'adam', 1e-4, 'mae'

    model = load_model('./model-resnet/model-128.mdl',compile=False)
    #model.load_weights('./model-resnet/ckpt-128-mae-adam-0.0002.ckpt')
    model.compile(optimizer=Adam(lr=lr), loss=mae, metrics=[psnr,ssim])

    model_name = 'ckpt-'+str(size)+'-'+str(opt)+'-'+str(lr)+'-'+str(loss)
    history = train(model,'./model-resnet/'+model_name+'.ckpt','./log/'+model_name)

    #results = test(model)
    #print(dict(zip(model.metrics_names,results)))

if __name__=='__main__':
    main()