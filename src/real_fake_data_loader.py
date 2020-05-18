import numpy as np
from numpy import load
import tensorflow as tf
def load_real_data(filename):

    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']

    # normalize from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

def generate_real_data(data, random_samples, patch_shape):

    trainA, trainB = data

    id = np.random.randint(0, trainA.shape[0], random_samples)

    X1, X2 = trainA[id], trainB[id]

    # generate 'real' class labels (1)
    y1 = np.ones((random_samples, patch_shape[0], patch_shape[0], 1))
    y2 = np.ones((random_samples, patch_shape[1], patch_shape[1], 1))
    y3 = np.ones((random_samples, patch_shape[2], patch_shape[2], 1))
    return [X1, X2], [y1,y2,y3]

def generate_fake_data_fine(g_model, batch_data, x_global, patch_shape):
    # generate fake fine data
    X = g_model.predict([batch_data,x_global])

    # create 'fake' class labels (0)
    y1 = np.zeros((len(X), patch_shape[0], patch_shape[0], 1))
    y2 = np.zeros((len(X), patch_shape[1], patch_shape[1], 1))
    return X, [y1,y2]

def generate_fake_data_coarse(g_model, batch_data, patch_shape):
    # generate fake coarse data
    X, X_global = g_model.predict(batch_data)

    # create 'fake' class labels (0)
    y1 = np.zeros((len(X), patch_shape[1], patch_shape[1], 1))
    y2 = np.zeros((len(X), patch_shape[2], patch_shape[2], 1))
    return [X,X_global], [y1,y2]

def resize(X_realA,X_realB,out_shape):
    X_realA = tf.image.resize(X_realA, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    X_realA = np.array(X_realA)
    
    X_realB = tf.image.resize(X_realB, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    X_realB = np.array(X_realB)
    
    return [X_realA,X_realB]