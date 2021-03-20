from keras import backend as K
import tensorflow as tf

def NMAE(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)/K.mean(y_true))

def NMAE_metric(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)/K.mean(y_true))


def MARE(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred)/K.mean(y_true))