from keras.models import Model,Sequential
from keras.layers.merge import add
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
import keras.backend as K

from Utils import PATCH_SIZE

LAYER_CONFIG = {"activation":"relu", "padding":"same", "kernel_initializer":"he_normal"}

def psnr(y_true, y_pred):
    rmse = K.mean(K.pow(K.flatten(y_true - y_pred), 2))
    return 10 * K.log(1.0 / rmse)/K.log(10)

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

def get_unet():
    Inputs = Input((PATCH_SIZE, PATCH_SIZE, 4),name='input')

    EncConv1_1 = Conv2D(32, (3, 3), **LAYER_CONFIG)(Inputs)
    EncConv1_2 = Conv2D(32, (3, 3), **LAYER_CONFIG)(EncConv1_1)
    Pool1 = MaxPooling2D(pool_size=(2, 2))(EncConv1_2)

    EncConv2_1 = Conv2D(64, (3, 3), **LAYER_CONFIG)(Pool1)
    EncConv2_2 = Conv2D(64, (3, 3), **LAYER_CONFIG)(EncConv2_1)
    Pool2 = MaxPooling2D(pool_size=(2, 2))(EncConv2_2)

    EncConv3_1 = Conv2D(128, (3, 3), **LAYER_CONFIG)(Pool2)
    EncConv3_2 = Conv2D(128, (3, 3), **LAYER_CONFIG)(EncConv3_1)
    Pool3 = MaxPooling2D(pool_size=(2, 2))(EncConv3_2)

    EncConv4_1 = Conv2D(256, (3, 3), **LAYER_CONFIG)(Pool3)
    EncConv4_2 = Conv2D(256, (3, 3), **LAYER_CONFIG)(EncConv4_1)
    Pool4 = MaxPooling2D(pool_size=(2, 2))(EncConv4_2)

    EncConv5_1 = Conv2D(512, (3, 3), **LAYER_CONFIG)(Pool4)
    EncConv5_2 = Conv2D(512, (3, 3), **LAYER_CONFIG)(EncConv5_1)
    Deconv4 = Conv2D(256, (3, 3), **LAYER_CONFIG)(UpSampling2D(size=(2,2))(EncConv5_2))

    Add4 = add([Deconv4,EncConv4_2])
    DecConv4_1 = Conv2D(256, (3, 3), **LAYER_CONFIG)(Add4)
    DecConv4_2 = Conv2D(256, (3, 3), **LAYER_CONFIG)(DecConv4_1)
    Deconv3 = Conv2D(128, (3, 3), **LAYER_CONFIG)(UpSampling2D(size=(2,2))(DecConv4_2))

    Add3 = add([Deconv3,EncConv3_2])
    DecConv3_1 = Conv2D(128, (3, 3), **LAYER_CONFIG)(Add3)
    DecConv3_2 = Conv2D(128, (3, 3), **LAYER_CONFIG)(DecConv3_1)
    Deconv2 = Conv2D(64, (3, 3), **LAYER_CONFIG)(UpSampling2D(size=(2,2))(DecConv3_2))

    Add2 = add([Deconv2,EncConv2_2])
    DecConv2_1 = Conv2D(64, (3, 3), **LAYER_CONFIG)(Add2)
    DecConv2_2 = Conv2D(64, (3, 3), **LAYER_CONFIG)(DecConv2_1)
    Deconv1 = Conv2D(32, (3, 3), **LAYER_CONFIG)(UpSampling2D(size=(2,2))(DecConv2_2))

    Add1 = add([Deconv1,EncConv1_2])
    DecConv1_1 = Conv2D(32, (3, 3), **LAYER_CONFIG)(Add1)
    DecConv1_2 = Conv2D(32, (3, 3), **LAYER_CONFIG)(DecConv1_1)
    DecConv1_3 = Conv2D(4, (3, 3), **LAYER_CONFIG)(DecConv1_2)

    Add0 = add([DecConv1_3,Inputs],name='output')

    model = Model(inputs=Inputs, outputs=Add0)
    model.compile(optimizer=Adam(lr=2e-4,decay=2e-5), loss=mean_absolute_error, metrics=[psnr,ssim])

    return model


def main():
    model=get_unet()
    model.summary()

if __name__=='__main__':
    main()
