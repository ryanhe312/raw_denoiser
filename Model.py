from keras.models import Model,load_model
from keras.layers.merge import add
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout,PReLU
from os import environ

PATCH_SIZE = 128

def PReLUConv(num_filters):
    def func(x):
        x = Conv2D(num_filters,kernel_size=(3,3),padding='same',kernel_initializer='he_normal')(x)
        x = PReLU()(x)
        return x
    return func

def ResBlock(num_filters):
    def func(x):
        conv1 = Conv2D(num_filters,kernel_size=(3,3),padding='same',kernel_initializer='he_normal')(x)
        prelu1 = PReLU()(conv1)
        conv2 = Conv2D(num_filters,kernel_size=(3,3),padding='same',kernel_initializer='he_normal')(prelu1)
        
        conv  = Conv2D(num_filters,kernel_size=(1,1),padding='same',kernel_initializer='he_normal')(x)
        merge = add([conv,conv2])
        prelu2 = PReLU()(merge)
        return prelu2
    return func

def get_resnet():
    Inputs = Input((PATCH_SIZE, PATCH_SIZE, 4),name='input')

    EncBlock1 = ResBlock(256)(Inputs)
    Pool1 = MaxPooling2D(pool_size=(2, 2))(EncBlock1)

    EncBlock2 = ResBlock(512)(Pool1)
    Pool2 = MaxPooling2D(pool_size=(2, 2))(EncBlock2)

    EncBlock3 = ResBlock(512)(Pool2)
    Pool3 = MaxPooling2D(pool_size=(2, 2))(EncBlock3)

    EncBlock4 = ResBlock(512)(Pool3)
    Pool4 = MaxPooling2D(pool_size=(2, 2))(EncBlock4)

    EncBlock5 = ResBlock(512)(Pool4)
    Deconv4 = PReLUConv(512)(UpSampling2D(size=(2,2))(EncBlock5))

    Add4 = add([Deconv4,EncBlock4])
    DecBlock4 = ResBlock(512)(Add4)
    Deconv3 = PReLUConv(512)(UpSampling2D(size=(2,2))(DecBlock4))

    Add3 = add([Deconv3,EncBlock3])
    DecBlock3 = ResBlock(512)(Add3)
    Deconv2 = PReLUConv(512)(UpSampling2D(size=(2,2))(DecBlock3))

    Add2 = add([Deconv2,EncBlock2])
    DecBlock2 = ResBlock(512)(Add2)
    Deconv1 = PReLUConv(256)(UpSampling2D(size=(2,2))(DecBlock2))

    Add1 = add([Deconv1,EncBlock1])
    DecBlock1 = ResBlock(256)(Add1)
    DecConv = PReLUConv(4)(DecBlock1)

    Add0 = add([DecConv,Inputs],name='output')

    model = Model(inputs=Inputs, outputs=Add0)

    return model

def get_unet():
    Inputs = Input((PATCH_SIZE, PATCH_SIZE, 4),name='input')

    EncConv1_1 = PReLUConv(32)(Inputs)
    EncConv1_2 = PReLUConv(32)(EncConv1_1)
    Pool1 = MaxPooling2D(pool_size=(2, 2))(EncConv1_2)

    EncConv2_1 = PReLUConv(64)(Pool1)
    EncConv2_2 = PReLUConv(64)(EncConv2_1)
    Pool2 = MaxPooling2D(pool_size=(2, 2))(EncConv2_2)

    EncConv3_1 = PReLUConv(128)(Pool2)
    EncConv3_2 = PReLUConv(128)(EncConv3_1)
    Pool3 = MaxPooling2D(pool_size=(2, 2))(EncConv3_2)

    EncConv4_1 = PReLUConv(256)(Pool3)
    EncConv4_2 = PReLUConv(256)(EncConv4_1)
    Pool4 = MaxPooling2D(pool_size=(2, 2))(EncConv4_2)

    EncConv5_1 = PReLUConv(512)(Pool4)
    EncConv5_2 = PReLUConv(512)(EncConv5_1)
    Deconv4 = PReLUConv(256)(UpSampling2D(size=(2,2))(EncConv5_2))

    Add4 = add([Deconv4,EncConv4_2])
    DecConv4_1 = PReLUConv(256)(Add4)
    DecConv4_2 = PReLUConv(256)(DecConv4_1)
    Deconv3 = PReLUConv(128)(UpSampling2D(size=(2,2))(DecConv4_2))

    Add3 = add([Deconv3,EncConv3_2])
    DecConv3_1 = PReLUConv(128)(Add3)
    DecConv3_2 = PReLUConv(128)(DecConv3_1)
    Deconv2 = PReLUConv(64)(UpSampling2D(size=(2,2))(DecConv3_2))

    Add2 = add([Deconv2,EncConv2_2])
    DecConv2_1 = PReLUConv(64)(Add2)
    DecConv2_2 = PReLUConv(64)(DecConv2_1)
    Deconv1 = PReLUConv(32)(UpSampling2D(size=(2,2))(DecConv2_2))

    Add1 = add([Deconv1,EncConv1_2])
    DecConv1_1 = PReLUConv(32)(Add1)
    DecConv1_2 = PReLUConv(32)(DecConv1_1)
    DecConv1_3 = PReLUConv(4)(DecConv1_2)

    Add0 = add([DecConv1_3,Inputs],name='output')

    model = Model(inputs=Inputs, outputs=Add0)

    return model


def main():
    environ["CUDA_VISIBLE_DEVICES"] = "2"
    model=get_resnet()
    model.summary()
    model.save('./model-resnet/model-128.mdl')

if __name__=='__main__':
    main()
