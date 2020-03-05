from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.losses import mse,mae,msle
from keras.utils import multi_gpu_model
from math import ceil,sqrt
from os import environ,makedirs,path
import keras.backend as K
 
from Utils import get_file_list,DataGenerator

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

def lr_schedule(epoch):
    lr = 3e-4
    if epoch > 100:
        lr *= 0.1
    elif epoch > 200:
        lr *= 0.1
    print('Learning rate: ', lr)
    return lr

def train(model,patch_size,batch_multi,epochs,ckpt_path,log_path):
    train_data = get_file_list('train')
    val_data = get_file_list('test')

    makedirs(path.dirname(ckpt_path),exist_ok=True)
    makedirs(log_path,exist_ok=True)

    checkpoint = ModelCheckpoint(ckpt_path, monitor='loss',verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=log_path)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=sqrt(0.1),
                                   monitor='loss',
                                   patience=16,
                                   min_lr=1e-6)
    
    history = model.fit_generator(DataGenerator(train_data,patch_size,batch_multi),\
                                epochs = epochs,\
                                verbose = 2,\
                                shuffle = True,\
                                callbacks=[checkpoint,tensorboard,lr_reducer],\
                                validation_data = DataGenerator(val_data,patch_size,batch_multi))
    return history

def main():
    environ["CUDA_VISIBLE_DEVICES"] = "1"    
    idx, patch_size, opt, lr, loss, batch_multi, epochs, multi_gpu=1, 128, 'adam', 3e-4, 'mae' , 2, 100, ''

    model_name = str(multi_gpu)+'ckpt'+str(idx)+'-'+str(patch_size)+'-'+str(opt)+'-'+str(lr)+'-'+str(loss)
    ckpt_path  = 'model-grdn/'+model_name
    model_path = 'model-grdn/model-'+str(patch_size)+'.mdl'

    model = load_model(model_path,compile=False)
    #model = multi_gpu_model(model,gpus=2)
    #model.load_weights('')

    model.compile(optimizer=Adam(lr=lr), loss=mae, metrics=[psnr,ssim])
    history = train(model,patch_size,batch_multi,epochs,ckpt_path+'.ckpt',ckpt_path)

    #results = test(model,patch_size,batch_multi)
    #print(dict(zip(model.metrics_names,results)))

if __name__=='__main__':
    main()