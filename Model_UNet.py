from keras.models import Model,load_model
from keras.layers.merge import add
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, PReLU,Activation, Conv2DTranspose
from os import environ

def ReLUDeconv(num_filters):
    def func(x):
        x = Conv2DTranspose(num_filters,kernel_size=(3,3),strides=(2, 2) ,padding='same',kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        return x
    return func

def ReLUConv(num_filters):
    def func(x):
        x = Conv2D(num_filters,kernel_size=(3,3),padding='same',kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        return x
    return func

def get_unet(patch_size):
    Inputs = Input((patch_size, patch_size, 4),name='input')

    EncConv1_1 = ReLUConv(32)(Inputs)
    EncConv1_2 = ReLUConv(32)(EncConv1_1)
    Pool1 = MaxPooling2D(pool_size=(2, 2))(EncConv1_2)

    EncConv2_1 = ReLUConv(64)(Pool1)
    EncConv2_2 = ReLUConv(64)(EncConv2_1)
    Pool2 = MaxPooling2D(pool_size=(2, 2))(EncConv2_2)

    EncConv3_1 = ReLUConv(128)(Pool2)
    EncConv3_2 = ReLUConv(128)(EncConv3_1)
    Pool3 = MaxPooling2D(pool_size=(2, 2))(EncConv3_2)

    EncConv4_1 = ReLUConv(256)(Pool3)
    EncConv4_2 = ReLUConv(256)(EncConv4_1)
    Pool4 = MaxPooling2D(pool_size=(2, 2))(EncConv4_2)

    EncConv5_1 = ReLUConv(512)(Pool4)
    EncConv5_2 = ReLUConv(512)(EncConv5_1)
    Deconv4 = ReLUDeconv(256)(EncConv5_2)

    Add4 = add([Deconv4,EncConv4_2])
    DecConv4_1 = ReLUConv(256)(Add4)
    DecConv4_2 = ReLUConv(256)(DecConv4_1)
    Deconv3 = ReLUDeconv(128)(DecConv4_2)

    Add3 = add([Deconv3,EncConv3_2])
    DecConv3_1 = ReLUConv(128)(Add3)
    DecConv3_2 = ReLUConv(128)(DecConv3_1)
    Deconv2 = ReLUDeconv(64)(DecConv3_2)

    Add2 = add([Deconv2,EncConv2_2])
    DecConv2_1 = ReLUConv(64)(Add2)
    DecConv2_2 = ReLUConv(64)(DecConv2_1)
    Deconv1 = ReLUDeconv(32)(DecConv2_2)

    Add1 = add([Deconv1,EncConv1_2])
    DecConv1_1 = ReLUConv(32)(Add1)
    DecConv1_2 = ReLUConv(32)(DecConv1_1)
    DecConv1_3 = ReLUConv(4)(DecConv1_2)

    Add0 = add([DecConv1_3,Inputs],name='output')

    model = Model(inputs=Inputs, outputs=Add0)

    return model

def main():
    environ["CUDA_VISIBLE_DEVICES"] = "2"
    patch_size = 256
    model=get_unet(patch_size)
    model.summary()
    #model.save('./model-unet/model-'+str(patch_size)+'.mdl')

if __name__=='__main__':
    main()
