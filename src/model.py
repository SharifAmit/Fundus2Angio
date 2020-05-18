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


def coarse_generator(img_shape=(256, 256, 3),ncf=64, n_downsampling=2, n_blocks=9, n_channels=1):
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


def fine_generator(x_coarse_shape=(256,256,64),input_shape=(512, 512, 3), nff=32, n_blocks=3, n_coarse_gen=1,n_channels = 1):

    
    X_input = Input(shape=input_shape)
    X_coarse = Input(shape=x_coarse_shape)

    for i in range(1, n_coarse_gen+1):
        down_filters = nff * (2**(n_coarse_gen-i))
        
        # Downsampling layers
        X = ReflectionPadding2D((3,3))(X_input)
        X = Conv2D(down_filters, kernel_size=(7,7), strides=(1,1), padding='valid')(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = Conv2D(down_filters*2, kernel_size=(3,3), strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)

        # Connection from Coarse Generator
        
        X = Add()([X_coarse,X])

        for j in range(n_blocks):
            res_filters = nff * (2**(n_coarse_gen-i)) * 2
            X = novel_residual_block(X, res_filters)

        # Upsampling layers
        up_filters = nff * (2**(n_coarse_gen-i))
        X = Conv2DTranspose(filters=up_filters, kernel_size=(3,3), strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)

    X = ReflectionPadding2D((3,3))(X)
    X = Conv2D(n_channels, kernel_size=(7,7), strides=(1,1), padding='valid')(X)
    X = Activation('tanh')(X)

    model = Model(inputs=[X_input,X_coarse], outputs=X, name='G_Fine')
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))

    model.summary()
    return model


def discriminator(input_shape_fundus=(512, 512, 3),
                        input_shape_angio=(512, 512, 1),
                        ndf=32, n_layers=3, activation='linear',
                        n_downsampling=1, name='Discriminator'):
    X_input_fundus = Input(shape=input_shape_fundus)
    X_input_angio = Input(shape=input_shape_angio)


    X = Concatenate(axis=-1)([X_input_fundus, X_input_angio])
    for i in range(n_downsampling):
        X = AveragePooling2D((3,3), strides=(2,2), padding='same')(X)

    X = Conv2D(ndf, kernel_size=(4,4), strides=(2,2), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X)
    X = LeakyReLU(alpha=0.2)(X)


    for i in range(1, n_layers):
        down_filters = min(ndf * 2, 512)
        X = Conv2D(down_filters, kernel_size=(4,4), strides=(2,2), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)


    nf = min(ndf * 2, 512)
    X = Conv2D(nf, kernel_size=(4,4), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.2)(X)


    X = Conv2D(1, kernel_size=(4,4), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X)
    X = Activation(activation)(X)


    model = Model(inputs=[X_input_fundus, X_input_angio], outputs=X , name=name)
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))
    return model

def fundus2angio_gan(g_model_fine,g_model_coarse, d_model1, d_model2, d_model3, d_model4,image_shape_fine,image_shape_coarse,image_shape_x_coarse):
    # Discriminator NOT trainable
    d_model1.trainable = False
    d_model2.trainable = False
    d_model3.trainable = False
    d_model4.trainable = False

    in_fine= Input(shape=image_shape_fine)
    in_coarse = Input(shape=image_shape_coarse)
    in_x_coarse = Input(shape=image_shape_x_coarse)

    # Generators
    gen_out_coarse, _ = g_model_coarse(in_coarse)
    gen_out_fine = g_model_fine([in_fine,in_x_coarse])

    # Discriminators Fine
    dis_out_1 = d_model1([in_fine, gen_out_fine])
    dis_out_2 = d_model2([in_fine, gen_out_fine])

    # Discriminators Coarse
    dis_out_3 = d_model3([in_coarse, gen_out_coarse])
    dis_out_4 = d_model4([in_coarse, gen_out_coarse])

    model = Model([in_fine,in_coarse,in_x_coarse], [dis_out_1,dis_out_2,dis_out_3,dis_out_4,gen_out_coarse,gen_out_fine])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mse','mse','mse','mse','mse'], optimizer=opt,loss_weights=[1,1,1,1,10,10])
    model.summary()
    return model