from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.losses import mse,mae,msle
from keras.utils import multi_gpu_model
from math import ceil
from os import environ,makedirs,path
import keras.backend as K

import Model 
import Utils

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

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

def train(model,patch_size,batch_multi,epochs,ckpt_path,log_path):
    train_data = Utils.get_file_list('train')
    val_data = Utils.get_file_list('test')

    makedirs(path.dirname(ckpt_path),exist_ok=True)
    makedirs(log_path,exist_ok=True)

    checkpoint = ModelCheckpoint(ckpt_path, monitor='loss',verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=log_path)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    
    history = model.fit_generator(Utils.DataGenerator(train_data,patch_size,batch_multi),\
                                epochs = epochs,\
                                verbose = 2,\
                                shuffle = True,\
                                callbacks=[checkpoint,tensorboard],\
                                validation_data = Utils.DataGenerator(val_data,patch_size,batch_multi))
    return history

def test(model,patch_size,batch_multi):
    test_data = Utils.get_file_list('test')

    results = model.evaluate_generator(Utils.DataGenerator(test_data,patch_size,batch_multi),\
                                       steps = len(test_data),\
                                       verbose = 0)
    return results

def main():
    environ["CUDA_VISIBLE_DEVICES"] = "2,3"    
    idx, patch_size, opt, lr, loss, batch_multi, epochs, multi_gpu=3, 256, 'adam', 1e-6, 'mae' , 2, 300, 'multi'

    model_name = str(multi_gpu)+'ckpt'+str(idx)+'-'+str(patch_size)+'-'+str(opt)+'-'+str(lr)+'-'+str(loss)
    ckpt_path  = 'model-resnet/'+model_name
    model_path = 'model-resnet/model-'+str(patch_size)+'.mdl'

    model = load_model(model_path,compile=False)
    model = multi_gpu_model(model,gpus=2)
    model.load_weights('model-resnet/multickpt2-256-adam-1e-05-mae.ckpt')

    model.compile(optimizer=Adam(lr=lr), loss=mae, metrics=[psnr,ssim])
    history = train(model,patch_size,batch_multi,epochs,ckpt_path+'.ckpt',ckpt_path)

    #results = test(model,patch_size,batch_multi)
    #print(dict(zip(model.metrics_names,results)))

if __name__=='__main__':
    main()