from keras.models import Model,load_model
from keras.layers.merge import add
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, PReLU,Activation, Conv2DTranspose,GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Add, Activation, Lambda, Multiply
from os import environ

def Dense(filters):
    def func(x):
        out = Conv2D(filters = filters, \
					kernel_size=(3,3), \
					strides=(1, 1), \
					padding='same' , \
					kernel_initializer='he_normal',\
					activation='relu')(x)
        out = Concatenate(axis = -1)([x, out])
        return out
	return func

def RDBlock(filters, growth_rate, dense_count = 3):
	def func(x):
		out = x
		
		for i in range(dense_count):
            out = Dense(filters = growth_rate)(out)

		out = Conv2D(filters = filters, \
				     kernel_size=(1,1), \
					 strides=(1, 1), \
					 padding='same', \
					 kernel_initializer='he_normal',\
					 activation='relu')(concate)
		return feat
	return func

def GRDBlock(filters, growth_rate, rdb_count = 4, dense_count=3):
    def func(x):
		out = x
		outputlist = []

		for i in range(rdb_count):
			output = RDBlock(filters = filters,\
							 growth_rate = growth_rate, \
							 dense_count = dense_count)(out)
			share.append(output)
			out = output
		
		concate = Concatenate(axis = -1)(share)
		conv = Conv2D(filters = filters, \
					  kernel_size=(1,1), \
					  strides=(1, 1), \
					  padding='same', \
					  kernel_initializer='he_normal',\
					  activation='relu')(concate)

		out = Add()([conv , x])
		return out

	return func

def GGRDBlock(filters, growth_rate, grdb_count = 4, rdb_count = 4, dense_count=3):
    def func(x):
        output = x
        for i in range(grdb_count):
            output = GRDBlock(filters = filters, \
							  growth_rate = growth_rate, \
							  rdb_count = rdb_count,\
							  dense_count = dense_count)(output)

		out = Add()([output , x])
        return out

	return func

def ChannelAttention(ratio=8):
	def func(x):
		channel = x._keras_shape[-1]
		
		shared_layer_one = Dense(channel//ratio,\
								activation='relu',\
								kernel_initializer='he_normal',\
								use_bias=True,\
								bias_initializer='zeros')
		shared_layer_two = Dense(channel,\
								kernel_initializer='he_normal',\
								use_bias=True,\
								bias_initializer='zeros')
		
		avg_pool = GlobalAveragePooling2D()(x)    
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
		
		out = Multiply()([x, cbam_feature])
		return out
	return func

def SpatialAttention():
	def func(x):
		channel = x._keras_shape[-1]
		cbam_feature = x
		
		avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
		assert avg_pool._keras_shape[-1] == 1
		max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
		assert max_pool._keras_shape[-1] == 1
		concat = Concatenate(axis=3)([avg_pool, max_pool])
		assert concat._keras_shape[-1] == 2
		cbam_feature = Conv2D(filters = 1,\
						kernel_size=7,\
						strides=1,\
						padding='same',\
						activation='sigmoid',\
						kernel_initializer='he_normal',\
						use_bias=False)(concat)	
		assert cbam_feature._keras_shape[-1] == 1

		out = Multiply()([x, cbam_feature])
		return out
	return func			

def CBAMBlock(ratio=8):
	def func(x):
		out = x 
		out = ChannelAttention(ratio)(out)
		out = SpatialAttention()(out)
		return out
	return func

def get_grdn(patch_size):
    Inputs = Input((patch_size, patch_size, 4),name='input')

	Conv1 = Conv2D(filters = 64, \
					 kernel_size=(3,3), \
					 strides=(1, 1), \
					 padding='same' , \
					 kernel_initializer='he_normal',\
					 activation='relu')(Inputs)

	Conv2 = Conv2D(filters = 64, \
					 kernel_size=(4,4), \
					 strides=(2, 2), \
					 padding='same' , \
					 kernel_initializer='he_normal')(Conv1)

	Grdbs = GGRDBlock(128,128)(Conv2)

	Deconv2 = Conv2DTranspose(filters = 64, \,
							kernel_size=(4,4), \
							strides=(2, 2), \
							padding='same' , \
							kernel_initializer='he_normal')(Grdbs)

	Cbam = CBAMBlock()(Deconv2)

	Conv3 = Conv2D(filters = 4, \
					 kernel_size=(3,3), \
					 strides=(1, 1), \
					 padding='same' , \
					 kernel_initializer='he_normal',\
					 activation='relu')(Cbam)

	Add = Add()([Inputs,Conv3])

    model = Model(inputs=Inputs, outputs=Add)

    return model


def main():
    environ["CUDA_VISIBLE_DEVICES"] = "2"
    patch_size = 256
    model=get_grdn(patch_size)
    model.summary()
    #model.save('./model-resnet/model-'+str(patch_size)+'.mdl')

if __name__=='__main__':
    main()
