from keras.models import Model,load_model
from keras.layers.merge import add
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, PReLU,Activation, Conv2DTranspose
from os import environ

def PReLUConv(num_filters):
    def func(x):
        x = Conv2D(num_filters,kernel_size=(3,3),padding='same',kernel_initializer='he_normal')(x)
        x = PReLU(shared_axes=[1,2])(x)
        return x
    return func

def ResBlock(num_filters):
    def func(x):
        conv1 = Conv2D(num_filters,kernel_size=(3,3),padding='same',kernel_initializer='he_normal')(x)
        prelu1 = PReLU(shared_axes=[1,2])(conv1)
        conv2 = Conv2D(num_filters,kernel_size=(3,3),padding='same',kernel_initializer='he_normal')(prelu1)
        
        conv  = Conv2D(num_filters,kernel_size=(1,1),padding='same',kernel_initializer='he_normal')(x)
        merge = add([conv,conv2])
        prelu2 = PReLU(shared_axes=[1,2])(merge)
        return prelu2
    return func

def get_resnet(patch_size):
    Inputs = Input((patch_size, patch_size, 4),name='input')

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


def main():
    environ["CUDA_VISIBLE_DEVICES"] = "2"
    patch_size = 256
    model=get_resnet(patch_size)
    model.summary()
    #model.save('./model-resnet/model-'+str(patch_size)+'.mdl')

if __name__=='__main__':
    main()
