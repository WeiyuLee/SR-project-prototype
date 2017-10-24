# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:03:19 2017

@author: Weiyu_Lee
"""

"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import h5py
import numpy as np
import random

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_data(path):  
    """
    Read h5 format data file
      
    Args:
        path: file path of desired file
        data: '.h5' file format that contains train data values
        label: '.h5' file format that contains train label values
    """
    
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('input'))
        label = np.array(hf.get('label'))
        #freq = np.array(hf.get('freq'))
        freq = []

    return data, label, freq

def batch_shuffle(data, label, batch_size):
    """
    Shuffle the batch data
    """
    # Shuffle the batch data
#    shuffled_data = list(zip(data, *label))
#    random.shuffle(shuffled_data)
#    tmp = list(zip(*shuffled_data))
#    
#    data_shuffled = tmp[0]
#    label_shuffled = tmp[1:]

    shuffled_data = list(zip(data, label))
    random.shuffle(shuffled_data)
    data_shuffled, label_shuffled = zip(*shuffled_data)
    
    return data_shuffled, label_shuffled

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
