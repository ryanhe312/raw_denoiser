from keras.models import Model,load_model
from keras.layers.merge import add
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, PReLU,Activation, Conv2DTranspose,GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Add, Activation, Lambda
from os import environ

def RDBlocks(self,x,name , count = 6 , g=32):
    ## 6 layers of RDB block
    ## this thing need to be in a damn loop for more customisability
    li = [x]
    pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)

    for i in range(2 , count+1):
        li.append(pas)
        out =  Concatenate(axis = self.channel_axis)(li) # conctenated out put
        pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)

    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis = self.channel_axis)(li)
    feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)

    feat = Add()([feat , x])
    return feat

def RDBBlock():
    pass


def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature._keras_shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def get_grdn(patch_size):
    Inputs = Input((patch_size, patch_size, 4),name='input')

    model = Model(inputs=Inputs, outputs=Inputs)

    return model


def main():
    environ["CUDA_VISIBLE_DEVICES"] = "2"
    patch_size = 256
    model=get_resnet(patch_size)
    model.summary()
    #model.save('./model-resnet/model-'+str(patch_size)+'.mdl')

if __name__=='__main__':
    main()
