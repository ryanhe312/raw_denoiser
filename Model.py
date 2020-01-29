import pretty_errors

from keras.models import Model
from keras.layers.merge import add
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.losses import mean_absolute_error
from keras.metrics import mse,mae

PATCH_SIZE = 256

def get_unet():
    Inputs = Input((PATCH_SIZE, PATCH_SIZE, 4))
    EncConv1_1 = Conv2D(32, (3, 3), activation="relu", padding="same")(Inputs)
    EncConv1_2 = Conv2D(32, (3, 3), activation="relu", padding="same")(EncConv1_1)
    Pool1 = MaxPooling2D(pool_size=(2, 2))(EncConv1_2)

    EncConv2_1 = Conv2D(64, (3, 3), activation="relu", padding="same")(Pool1)
    EncConv2_2 = Conv2D(64, (3, 3), activation="relu", padding="same")(EncConv2_1)
    Pool2 = MaxPooling2D(pool_size=(2, 2))(EncConv2_2)

    EncConv3_1 = Conv2D(128, (3, 3), activation="relu", padding="same")(Pool2)
    EncConv3_2 = Conv2D(128, (3, 3), activation="relu", padding="same")(EncConv3_1)
    Pool3 = MaxPooling2D(pool_size=(2, 2))(EncConv3_2)

    EncConv4_1 = Conv2D(256, (3, 3), activation="relu", padding="same")(Pool3)
    EncConv4_2 = Conv2D(256, (3, 3), activation="relu", padding="same")(EncConv4_1)
    Pool4 = MaxPooling2D(pool_size=(2, 2))(EncConv4_2)

    EncConv5_1 = Conv2D(512, (3, 3), activation="relu", padding="same")(Pool4)
    EncConv5_2 = Conv2D(512, (3, 3), activation="relu", padding="same")(EncConv5_1)
    Deconv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(UpSampling2D(size=(2,2))(EncConv5_2))

    Add4 = add([Deconv4,EncConv4_2])
    DecConv4_1 = Conv2D(256, (3, 3), activation="relu", padding="same")(Add4)
    DecConv4_2 = Conv2D(256, (3, 3), activation="relu", padding="same")(DecConv4_1)
    Deconv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(UpSampling2D(size=(2,2))(DecConv4_2))

    Add3 = add([Deconv3,EncConv3_2])
    DecConv3_1 = Conv2D(128, (3, 3), activation="relu", padding="same")(Add3)
    DecConv3_2 = Conv2D(128, (3, 3), activation="relu", padding="same")(DecConv3_1)
    Deconv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(UpSampling2D(size=(2,2))(DecConv3_2))

    Add2 = add([Deconv2,EncConv2_2])
    DecConv2_1 = Conv2D(64, (3, 3), activation="relu", padding="same")(Add2)
    DecConv2_2 = Conv2D(64, (3, 3), activation="relu", padding="same")(DecConv2_1)
    Deconv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(UpSampling2D(size=(2,2))(DecConv2_2))

    Add1 = add([Deconv1,EncConv1_2])
    DecConv1_1 = Conv2D(32, (3, 3), activation="relu", padding="same")(Add1)
    DecConv1_2 = Conv2D(32, (3, 3), activation="relu", padding="same")(DecConv1_1)
    DecConv1_3 = Conv2D(4, (3, 3), activation="relu", padding="same")(DecConv1_2)

    Add0 = add([DecConv1_3,Inputs])

    model = Model(input=Inputs, output=Add0)

    model.compile(optimizer=Adam(lr=2e-4,decay=2e-5), loss=mean_absolute_error, metrics=[mse,mae])

    return model


def main():
    model=get_unet()
    model.summary()

if __name__=='__main__':
    main()
