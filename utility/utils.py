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
import scipy.misc as misc

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

    return data, label

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

def batch_shuffle_rndc(data, label, scale, subimage_size, index, batch_size):

        crop_data = []
        crop_label = []
        i = index
        while i < index+batch_size:
            
            lridx = random_crop(scale, data[i].shape, subimage_size)
            crop_lr = data[i][lridx[0]:lridx[0]+subimage_size,
                              lridx[1]:lridx[1]+subimage_size,
                              :]
            crop_hr = label[i][lridx[0]*scale:lridx[0]*scale+scale*subimage_size,
                              lridx[1]*scale:lridx[1]*scale+scale*subimage_size,
                              :]
            #Reject small spatial gradient patch                  
            patch_grad = np.mean(np.gradient(crop_hr))

            if patch_grad > 40:
                crop_data.append(crop_lr)
                crop_label.append(crop_hr) 
                i+=1   

        return crop_data, crop_label




def random_crop(scale, LR_image_size, subimage_size):

    x_list = list(range(0,LR_image_size[0]-subimage_size-1))
    y_list = list(range(0,LR_image_size[1]-subimage_size-1))
    random.shuffle(x_list)
    random.shuffle(y_list)

    return (x_list[0], y_list[0])

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
