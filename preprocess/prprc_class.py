# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:54:46 2017

@author: Weiyu_Lee
"""

import os
import glob
import scipy.misc
import numpy as np
import h5py

class PRPRC(object):
    
    def __init__(self, 
                 mode,
                 scale,
                 image_size,
                 label_size,
                 color_dim,
                 train_extract_stride,
                 test_extract_stride,
                 train_dir,
                 test_dir,
                 output_dir,
                 train_h5_name,
                 test_h5_name):
        """
        Initial function
        """
        
        self.mode = mode
        self.scale = scale
        self.image_size = image_size
        self.label_size = label_size
        self.color_dim = color_dim
        self.train_extract_stride = train_extract_stride
        self.test_extract_stride = test_extract_stride
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.train_h5_name = train_h5_name
        self.test_h5_name = test_h5_name
        
    def prepare_data(self):
        """
        Prepare image data
        """
        
        print("Preparing training dataset...")
        sub_HF_count, sub_LF_count = self.input_image_setup(self.train_dir, self.train_extract_stride, self.train_h5_name)
        print("Sub-images number = [{}], HF sub-images number = [{}], LF sub-images number = [{}], Done.".format(sub_HF_count+sub_LF_count, sub_HF_count, sub_LF_count))
        
        print("Preparing testing dataset...")
        sub_HF_count, sub_LF_count = self.input_image_setup(self.test_dir, self.test_extract_stride, self.test_h5_name)
        print("Sub-images number = [{}], HF sub-images number = [{}], LF sub-images number = [{}], Done.".format(sub_HF_count+sub_LF_count, sub_HF_count, sub_LF_count))
        
    def input_image_setup(self, data_dir, extract_stride, h5_name):
        """
        Setup the input images
        """
        input_data_path, label_data_path = self.get_file_path(data_dir)
        
#        if self.color_dim is 1:
#            is_grayscale = True
#        else:
#            is_grayscale = False

        sub_input_sequence = []
        sub_label_sequence = []
        sub_freq_label_sequence = []
        
        sub_HF_count = 0
        sub_LF_count = 0
        
        for i in range(len(input_data_path)):
            input_data = self.imread(input_data_path[i])
            label_data = self.imread(label_data_path[i])

            if len(input_data.shape) == 3:
                h, w, _ = input_data.shape
            else:
                h, w = input_data.shape
            
            for x in range(0, h-self.image_size+1, extract_stride):
                for y in range(0, w-self.image_size+1, extract_stride):
                    
                    sub_input = input_data[x:x+self.image_size, y:y+self.image_size]
                    sub_label = label_data[x:x+self.image_size, y:y+self.image_size]

                    # Preprocess
                    sub_input_preprocessd, sub_label_preprocessd, sub_freq_label = self.preprocess(sub_input, sub_label, h5_name)                    

                    for j in range(len(sub_input_preprocessd)):
                        tmp_input = sub_input_preprocessd[j]
                        tmp_label = sub_label_preprocessd[j]
                        tmp_freq_label = sub_freq_label[j]
                        
                        # Make channel value
                        tmp_input = tmp_input.reshape([self.image_size, self.image_size, 1])  
                        tmp_label = tmp_label.reshape([self.image_size, self.image_size, 1])  
                        
                        # Save the sub-images & freq. labels
                        sub_input_sequence.append(tmp_input)
                        sub_label_sequence.append(tmp_label)  
                        sub_freq_label_sequence.append(tmp_freq_label)
                        
                    sub_HF_count += sub_freq_label.count("HF")
                    sub_LF_count += sub_freq_label.count("LF")
        
        arr_data = np.asarray(sub_input_sequence) 
        arr_label = np.asarray(sub_label_sequence) 
        arr_freq_label = np.asarray(sub_freq_label_sequence) 
        
        h5_name = h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.image_size)
        save_dir = os.path.join(os.getcwd(), self.output_dir, h5_name)
        self.make_h5data(arr_data, arr_label, arr_freq_label, save_dir)

        return sub_HF_count, sub_LF_count
        
    def get_file_path(self, data_dir):
        """
        Get the dataset file path.
        According to the scale, choose the correct folder that made by Matlab preprocessing code.
        
        Args:
            dataset: choose train dataset or test dataset
            
            For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
        """
        
        # Define the folder name
        preprocessed_folder = "preprocessed_scale_{}".format(self.scale)
        
        # Define the preprocessed ext.              
        input_data_ext = "*_bicubic_scale_{}_input.bmp".format(self.scale)
        label_data_ext = "*_label.bmp".format(self.scale)
    
        data_dir = os.path.join(os.getcwd(), data_dir, preprocessed_folder)
               
        input_data_path = glob.glob(os.path.join(data_dir, input_data_ext))
        label_data_path = glob.glob(os.path.join(data_dir, label_data_ext))
        
        input_data_path = sorted(input_data_path)
        label_data_path = sorted(label_data_path)   
               
        return input_data_path, label_data_path    
    
    def imread(self, data_path, is_grayscale=True):

        if is_grayscale:
            return scipy.misc.imread(data_path, flatten=True, mode='YCbCr').astype(np.float64)
        else:
            return scipy.misc.imread(data_path, mode='YCbCr').astype(np.float64)        
    
    def preprocess(self, input_data, label_data, h5_name):      
        preprocessed_input_data = []
        preprocessed_label_data = []
        freq_label = []
        
        # Normalization
        input_data = input_data / 255.
        label_data = label_data / 255.
        
        preprocessed_input_data.append(input_data)
        preprocessed_label_data.append(label_data)
        
        tmp_var = np.var(input_data)
        if tmp_var >= 0.0325/2:
            freq_label.append("HF")
            
            if self.mode == "freq" and h5_name == self.train_h5_name:                      
                # Flip
                flipr_input_data = np.fliplr(input_data)
                flipr_label_data = np.fliplr(label_data)
                preprocessed_input_data.append(flipr_input_data)
                preprocessed_label_data.append(flipr_label_data)
                freq_label.append("HF")
                
                # Rotation
                rot90_input_data = np.rot90(input_data)
                rot90_label_data = np.rot90(label_data)
                preprocessed_input_data.append(rot90_input_data)        
                preprocessed_label_data.append(rot90_label_data)        
                freq_label.append("HF")            
        else:
            freq_label.append("LF")                       
               
        return preprocessed_input_data, preprocessed_label_data, freq_label
        
    def make_h5data(self, input, label, freq_label, save_dir):
        """
        Make input data as h5 file format
        """
        
        savepath = os.path.join(os.getcwd(), save_dir)
        
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('input', data=input)
            hf.create_dataset('label', data=label)
            #hf.create_dataset('freq', data=freq_label)
           