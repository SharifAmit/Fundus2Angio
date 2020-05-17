import tensorflow as tf
import keras
from keras.layers import Layer, InputSpec, Reshape
from keras.layers import Input, Add, Concatenate, Lambda
from keras.layers import LeakyReLU
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, Dropout
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
from keras.initializers import RandomNormal

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        if type(padding) == int:
            padding = (padding, padding)
        self.padding = padding
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def novel_residual_block(X_input, filters):
    X = X_input
    X = ReflectionPadding2D((1,1))(X)
    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='valid')(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.2)(X)


    X = ReflectionPadding2D((1,1))(X)
    X = SeparableConv2D(filters, kernel_size=(3,3), strides=(1,1), padding='valid')(X)
    X = BatchNormalization()(X)

    X = Add()([X_input, X])
    return X


def coarse_generator(img_shape=(256, 256, 3),ncf=64,
                           n_downsampling=2, n_blocks=9, n_channels=1):
    X_input = Input(shape=img_shape)
    

    X = ReflectionPadding2D((3,3))(X_input)
    X = Conv2D(ncf, kernel_size=(7,7), strides=(1,1), padding='valid',kernel_initializer=RandomNormal(stddev=0.02))(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.2)(X)

    # Downsampling layers
    for i in range(n_downsampling):
        down_filters = ncf * pow(2,i) * 2
        X = Conv2D(down_filters, kernel_size=(3,3), strides=(2,2), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)

    # Novel Residual Blocks
    res_filters = pow(2,n_downsampling)
    for i in range(n_blocks):
        X = novel_residual_block(X, ncf*res_filters)

    # Upsampling layers
    for i in range(n_downsampling):
        up_filters  =int(ncf * pow(2,(n_downsampling - i)) / 2) 
        X = Conv2DTranspose(filters=up_filters, kernel_size=(3,3), strides=(2,2), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X)
        
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)

    feature_out = X

    X = ReflectionPadding2D((3,3))(X)
    X = Conv2D(n_channels, kernel_size=(7,7), strides=(1,1), padding='valid',kernel_initializer=RandomNormal(stddev=0.02))(X)
    X = Activation('tanh')(X)

    model = Model(inputs=X_input, outputs=[X,feature_out],name='G_Coarse')
    model.compile(loss=['mse',None], optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))

    model.summary()
    return model