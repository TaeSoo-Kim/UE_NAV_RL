import keras
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Lambda
)
from keras.layers.convolutional import (
    Convolution3D,
    ZeroPadding3D,
)
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adagrad
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

import pdb 


def conv3D_bn(x, nb_filter, nb_row, nb_col, nb_dep,
              name='',border_mode='same', subsample=(1, 1, 1),batch_norm=0,
              weight_decay=0, dim_ordering='th'):
  '''Utility function to apply to a tensor a module conv + BN
  with optional weight decay (L2 weight regularization).
  '''
  if weight_decay:
    W_regularizer = l2(weight_decay)
    b_regularizer = l2(weight_decay)
  else:
    W_regularizer = None
    b_regularizer = None
  x = Convolution3D(name=name,
                    nb_filter=nb_filter, 
                    kernel_dim1=nb_row, kernel_dim2=nb_col, kernel_dim3=nb_dep,
                    subsample=subsample,
                    border_mode=border_mode,
                    init='glorot_uniform', # == 'xavier' in caffe
                    W_regularizer=W_regularizer,
                    b_regularizer=b_regularizer,
                    dim_ordering=dim_ordering)(x)
  if batch_norm:
    x = BatchNormalization()(x)
  x = Activation("relu")(x)
  return x