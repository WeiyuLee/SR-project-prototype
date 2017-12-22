# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:20:02 2017

@author: Weiyu_Lee
"""

import os
import sys
sys.path.append('./utility')

import tensorflow as tf
from tqdm import tqdm
import numpy as np

import model_zoo
import scipy.misc as misc
import random

import time

from utils import (
  read_data, 
  batch_shuffle_rndc,
  batch_shuffle,
  log10,
  laplacian_filter
)




class MODEL(object):
    def __init__(self, 
                 sess, 
                 mode=None,
                 epoch=10,
                 batch_size=128,
                 image_size=32,
                 label_size=20, 
                 learning_rate=1e-4,
                 color_dim=1, 
                 scale=4,
                 train_extract_stride=14,
                 test_extract_stride=20,
                 checkpoint_dir=None, 
                 log_dir=None,
                 output_dir=None,
                 train_dir=None,
                 test_dir=None,
                 h5_dir=None,
                 train_h5_name=None,
                 test_h5_name=None,
                 ckpt_name=None,
                 is_train=True,
                 model_ticket=None):                 
        """
        Initial function
          
        Args:
            image_size: training or testing input image size. 
                        (if scale=3, image size is [33x33].)
            label_size: label image size. 
                        (if scale=3, image size is [21x21].)
            batch_size: batch size
            color_dim: color dimension number. (only Y channel, color_dim=1)
            checkpoint_dir: checkpoint directory
            output_dir: output directory
        """  
        
        self.sess = sess
        
        self.mode = mode

        self.epoch = epoch

        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size      

        self.learning_rate = learning_rate
        self.color_dim = color_dim
        self.is_grayscale = (color_dim == 1)        
        self.scale = scale
    
        self.train_extract_stride = train_extract_stride
        self.test_extract_stride = test_extract_stride
    
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.output_dir = output_dir
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        
        self.h5_dir = h5_dir
        self.train_h5_name = train_h5_name
        self.test_h5_name = test_h5_name
        
        self.ckpt_name = ckpt_name
        
        self.is_train = is_train      
        
        self.model_ticket = model_ticket
        
        self.model_list = ["googleLeNet_v1", "resNet_v1", "srcnn_v1", "grr_srcnn_v1", 
                           "grr_grid_srcnn_v1", "espcn_v1", "edsr_v1","edsr_v2","edsr_attention_v1",
                           "edsr_1X1_v1", "edsr_local_att_v1", "edsr_attention_v2", "edsr_v2_dual",
                           "edsr_local_att_v2_upsample", "edsr_lsgan","edsr_lsgan_up","edsr_lsgan_dis_large"]
        
        self.build_model()        
    
    def build_model(self):###              
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, "build_" + self.model_ticket)
            model = fn()
            return model    
        
    def train(self):
        if self.model_ticket not in self.model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, "train_" + self.model_ticket)
            function = fn()
            return function                 

    def build_srcnn_v1(self):###
        """
        Build srcnn_v1 model
        """   
        # Define input and label images
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        self.pred = mz.build_model()
                         
        # Define loss function (MSE) 
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
            
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.loss, collections=['train'])
            self.merged_summary_train = tf.summary.merge_all('train')   

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.loss, collections=['test'])
            self.merged_summary_test = tf.summary.merge_all('test')             
                    
        self.saver = tf.train.Saver()     

    def build_grr_srcnn_v1(self):###
        """
        Build grr_srcnn_v1 model
        """        
        # Define input and label images
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.stg1_labels = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='stg1_labels')
        self.stg2_labels = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='stg2_labels')
        self.stg3_labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.color_dim], name='stg3_labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.images, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        self.stg1_pred, self.stg2_pred, self.stg3_pred = mz.build_model()
                  
        padding = 6
        
        # Define loss function (MSE) 
        ## Stage 1 loss:
        self.stg1_loss = tf.reduce_mean(tf.square(self.stg1_labels[:, padding:-padding, padding:-padding, :] - self.stg1_pred[:, padding:-padding, padding:-padding, :]))
        ## Stage 2 loss:
        self.stg2_loss = tf.reduce_mean(tf.square(self.stg2_labels[:, padding:-padding, padding:-padding, :] - self.stg2_pred[:, padding:-padding, padding:-padding, :]))    
        ## Stage 3 loss:
        self.stg3_loss = tf.reduce_mean(tf.square(self.stg3_labels - self.stg3_pred))
    
        self.all_stg_loss = tf.add(tf.add(self.stg1_loss, self.stg2_loss), self.stg3_loss)

        with tf.name_scope('train_summary'):
            tf.summary.scalar("Stg1 loss", self.stg1_loss, collections=['train'])
            tf.summary.scalar("Stg2 loss", self.stg2_loss, collections=['train'])
            tf.summary.scalar("Stg3 loss", self.stg3_loss, collections=['train'])
            self.merged_summary_train = tf.summary.merge_all('train')                        

        with tf.name_scope('test_summary'):
            tf.summary.scalar("Stg1 loss", self.stg1_loss, collections=['test'])
            tf.summary.scalar("Stg2 loss", self.stg2_loss, collections=['test'])
            tf.summary.scalar("Stg3 loss", self.stg3_loss, collections=['test'])
            self.merged_summary_test = tf.summary.merge_all('test')         
            
        self.saver = tf.train.Saver()
    
    def build_grr_grid_srcnn_v1(self):###
        """
        Build grr_grid_srcnn_v1 model
        """        
        # Define input and label images
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.stg1_labels = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='stg1_labels')
        self.stg2_labels = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='stg2_labels')
        self.stg3_labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.color_dim], name='stg3_labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        
        self.inputs = self.images
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.inputs, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        self.stg_pred, self.HFLF_pred, self.HFLF_idx, self.TV_stg3_output = mz.build_model()
                         
        padding = 6
           
        # Define loss function (MSE) 
        ## Stage 1 loss:
        self.stg1_loss = tf.reduce_mean(tf.square(self.stg1_labels[:, padding:-padding, padding:-padding, :] - self.stg_pred[0][:, padding:-padding, padding:-padding, :]))
        ## Stage 2 loss:
        self.stg2_loss = tf.reduce_mean(tf.square(self.stg2_labels[:, padding:-padding, padding:-padding, :] - self.stg_pred[1][:, padding:-padding, padding:-padding, :]))    
        ## Stage 3 loss:
        self.stg3_loss = tf.reduce_mean(tf.square(self.stg3_labels - self.stg_pred[2]))
    
        self.all_stg_loss = tf.add(tf.add(self.stg1_loss, self.stg2_loss), self.stg3_loss)

        ## HF loss
        self.HF_labels = tf.squeeze(tf.gather(self.stg3_labels, self.HFLF_idx[0]), 1)
        self.HF_loss = tf.reduce_mean(tf.square(self.HF_labels - self.HFLF_pred[0]))

        self.before_HF_pred = tf.squeeze(tf.gather(self.stg_pred[2], self.HFLF_idx[0]), 1)
        self.before_HF_loss = tf.reduce_mean(tf.square(self.HF_labels - self.before_HF_pred))       

        ## LF loss
        self.LF_labels = tf.squeeze(tf.gather(self.stg3_labels, self.HFLF_idx[1]), 1)
        self.LF_loss = tf.reduce_mean(tf.square(self.LF_labels - self.HFLF_pred[1]))
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("Stg1 loss", self.stg1_loss, collections=['train'])
            tf.summary.scalar("Stg2 loss", self.stg2_loss, collections=['train'])
            tf.summary.scalar("Stg3 loss", self.stg3_loss, collections=['train'])
            tf.summary.scalar("HF loss", self.HF_loss, collections=['train'])
            tf.summary.scalar("LF loss", self.LF_loss, collections=['train'])
            tf.summary.scalar("Final loss", self.stg3_loss, collections=['train'])
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("Stg1 loss", self.stg1_loss, collections=['test'])
            tf.summary.scalar("Stg2 loss", self.stg2_loss, collections=['test'])
            tf.summary.scalar("Stg3 loss", self.stg3_loss, collections=['test'])
            tf.summary.scalar("HF loss", self.HF_loss, collections=['test'])
            tf.summary.scalar("LF loss", self.LF_loss, collections=['test'])
            tf.summary.scalar("Final loss", self.stg3_loss, collections=['test'])
            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()        

    def train_srcnn_v1(self):
        """
        Training process.
        """     
        print("Training...")

        stage_size = 3

        # Define dataset path
        self.train_h5_name = self.train_h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.image_size)
        self.test_h5_name = self.test_h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.image_size)
        
        train_data_dir = os.path.join('./{}'.format(self.h5_dir), self.train_h5_name)
        test_data_dir = os.path.join('./{}'.format(self.h5_dir), self.test_h5_name)
        
        # Read data from .h5 file
        train_data, train_label = read_data(train_data_dir)
        test_data, test_label = read_data(test_data_dir)

        # Stochastic gradient descent with the standard backpropagation       
        ## Stage loss 
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        summary_writer = tf.summary.FileWriter('log', self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        # Define iteration counter, timer and average loss
        itera_counter = 0
        avg_500_loss = 0
        avg_loss = 0
        
        # Load checkpoint 
        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
             
        batch_labels = [None]*stage_size  

        train_batch_num = len(train_data) // self.batch_size
        
        padding = (self.image_size - self.label_size) // 2 # 6

        # Prerpare validation data       
        val_images = test_data;
        val_labels = test_label[:, padding:-padding, padding:-padding, :]

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            train_data, train_label = batch_shuffle(train_data, train_label, self.batch_size)

            epoch_pbar.set_description("Epoch: [%2d]" % ((ep+1)))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, train_batch_num), desc="Batch: [0]")
            for idx in batch_pbar:                
                itera_counter += 1
                  
                # Get the training data
                batch_images = train_data[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_images = np.array(batch_images)

                batch_labels = np.array(train_label[idx*self.batch_size : (idx+1)*self.batch_size])
                batch_labels = batch_labels[:, padding:-padding, padding:-padding, :]          
                
                # Run the model
                train_sum, _, train_err = self.sess.run([self.merged_summary_train,
                                                         self.train_op, 
                                                         self.loss],
                                                         feed_dict={
                                                                     self.images: batch_images, 
                                                                     self.labels: batch_labels,
                                                                     self.dropout: 1.
                                                                   })   

                avg_loss += train_err
                avg_500_loss += train_err
                   
                batch_pbar.set_description("Batch: [%2d]" % (idx+1))
            
            if ep % 5 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                # Validation
                ## Run the test images
                test_sum, val_err = self.sess.run([self.merged_summary_test,
                                                   self.loss] ,
                                                   feed_dict={
                                                               self.images: val_images, 
                                                               self.labels: val_labels,
                                                               self.dropout: 1.
                                                             })
                     
                avg_500_loss /= (train_batch_num*5)
                
                print("Epoch: [%2d], Average train loss: 5 ep loss: [%.8f], all loss: [%.8f], Test stg loss: [%.8f]\n" \
                     % ((ep+1), avg_500_loss, avg_loss/itera_counter, val_err))       
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)
                
                avg_500_loss = 0   

    def train_grr_srcnn_v1(self):
        """
        Training process.
        """     
        print("Training...")

        stage_size = 3

        # Define dataset path
        self.train_h5_name = self.train_h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.image_size)
        self.test_h5_name = self.test_h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.image_size)
        
        train_data_dir = os.path.join('./{}'.format(self.h5_dir), self.train_h5_name)
        test_data_dir = os.path.join('./{}'.format(self.h5_dir), self.test_h5_name)
        
        # Read data from .h5 file
        train_data, train_label = read_data(train_data_dir)
        test_data, test_label = read_data(test_data_dir)
    
        # Stochastic gradient descent with the standard backpropagation       
        ## Stage loss 
        self.stg_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.all_stg_loss)
        #self.stg1_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg1_loss)
        #self.stg2_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg2_loss)
        #self.stg3_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg3_loss)

        summary_writer = tf.summary.FileWriter('log', self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        # Define iteration counter, timer and average loss
        itera_counter = 0
        avg_500_loss = [0]*(stage_size+1)    # 3 stage + 1 total loss
        avg_final_loss = 0
        
        # Load checkpoint 
        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
             
        batch_labels = [None]*stage_size  

        train_batch_num = len(train_data) // self.batch_size
        
        padding = (self.image_size - self.label_size) // 2 # 6

        # Prerpare validation data
        val_label = [None]*stage_size
        
        val_images = test_data;
        val_label[0] = test_label
        val_label[1] = test_label
        val_label[2] = test_label[:, padding:-padding, padding:-padding, :]

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            train_data, train_label = batch_shuffle(train_data, train_label, self.batch_size)

            epoch_pbar.set_description("Epoch: [%2d]" % ((ep+1)))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, train_batch_num), desc="Batch: [0]")
            for idx in batch_pbar:                
                itera_counter += 1
                  
                # Get the training data
                batch_images = train_data[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_images = np.array(batch_images)

                batch_labels[0] = (train_label[idx*self.batch_size : (idx+1)*self.batch_size])
                batch_labels[1] = (train_label[idx*self.batch_size : (idx+1)*self.batch_size])
                batch_labels[2] = np.array(train_label[idx*self.batch_size : (idx+1)*self.batch_size])
                batch_labels[2] = batch_labels[2][:, padding:-padding, padding:-padding, :]          
                
                # Run the model
                train_sum, _, stg_err = self.sess.run([   self.merged_summary_train,
                                                          self.stg_train_op, 
                                                          self.stg3_loss, 
                                                       ], 
                                                       feed_dict={
                                                                   self.images: batch_images, 
                                                                   self.stg1_labels: batch_labels[0],
                                                                   self.stg2_labels: batch_labels[1],
                                                                   self.stg3_labels: batch_labels[2],
                                                                   self.dropout: 1.
                                                                 })   
   
                avg_500_loss[0] += stg_err
    
                batch_pbar.set_description("Batch: [%2d]" % (idx+1))
                #batch_pbar.refresh()
            
            if ep % 5 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                # Validation
                ## Run the test images
                test_sum, val_stg3_err = self.sess.run([  self.merged_summary_test,
                                                          self.stg3_loss,
                                                       ], 
                                                       feed_dict={
                                                                    self.images: val_images, 
                                                                    self.stg1_labels: val_label[0],
                                                                    self.stg2_labels: val_label[1],
                                                                    self.stg3_labels: val_label[2],
                                                                    self.dropout: 1.
                                                                  })
 
                for i in range(len(avg_500_loss)):
                    avg_500_loss[i] /= (train_batch_num*5)

                avg_final_loss /= (train_batch_num*5)

                print("Epoch: [%2d], Average train loss of 5 epoches: stg3 loss: [%.8f], Test stg loss: [%.8f]\n" \
                     % ((ep+1), avg_500_loss[0], val_stg3_err))                            
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)
                
                avg_500_loss = [0]*(stage_size+1)


    def build_espcn_v1(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.color_dim], name='labels')
        #mean_x = tf.reduce_mean(self.input)
        #image_input =  self.input - mean_x
        #mean_y = tf.reduce_mean(self.image_target)
        #taget  =  self.image_target - mean_y
        image_input = self.input
        target = self.image_target
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        #self.logits = mz.build_model(scale=4,feature_size = 32)
        self.logits = mz.build_model()
        self.l1_loss = tf.reduce_mean(tf.losses.absolute_difference(target,self.logits ))
        mse = tf.reduce_mean(tf.square(target - self.logits))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(mse)
        
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target, collections=['train'])
            tf.summary.image("output_image",self.logits, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target , collections=['test'])
            tf.summary.image("output_image",self.logits, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()        
        
    
                
    def train_grr_grid_srcnn_v1(self):

        """
        Training process.
        """     
        print("Training...")

        stage_size = 3

        # Define dataset path
        self.train_h5_name = self.train_h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.image_size)
        self.test_h5_name = self.test_h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.image_size)
        
        train_data_dir = os.path.join('./{}'.format(self.h5_dir), self.train_h5_name)
        test_data_dir = os.path.join('./{}'.format(self.h5_dir), self.test_h5_name)
        
        # Read data from .h5 file
        train_data, train_label = read_data(train_data_dir)
        test_data, test_label = read_data(test_data_dir)
   
        # Stochastic gradient descent with the standard backpropagation       
        ## Stage loss 
        self.stg_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.all_stg_loss)
        #self.stg1_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg1_loss)
        #self.stg2_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg2_loss)
        #self.stg3_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg3_loss)
        
        self.HF_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.HF_loss)


        summary_writer = tf.summary.FileWriter('log', self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        # Define iteration counter, timer and average loss
        itera_counter = 0
        avg_500_loss = [0]*5    # 5 temp var. for debuging
        
        # Load checkpoint 

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
             
        batch_labels = [None]*stage_size  


        train_batch_num = len(train_data) // self.batch_size
        
        padding = (self.image_size - self.label_size) // 2 # 6

        # Prerpare validation data
        val_label = [None]*stage_size
        
        val_images = test_data
        val_label[0] = test_label
        val_label[1] = test_label
        val_label[2] = test_label[:, padding:-padding, padding:-padding, :]

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            shuffled_train_data, shuffled_train_label = batch_shuffle(train_data, train_label, self.batch_size)

            epoch_pbar.set_description("Epoch: [%2d]" % ((ep+1)))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, train_batch_num), desc="Batch: [0]")
            for idx in batch_pbar:                
                itera_counter += 1
                  
                # Get the training data
                batch_images = shuffled_train_data[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_images = np.array(batch_images)

                batch_labels[0] = (shuffled_train_label[idx*self.batch_size : (idx+1)*self.batch_size])
                batch_labels[1] = (shuffled_train_label[idx*self.batch_size : (idx+1)*self.batch_size])
                batch_labels[2] = np.array(shuffled_train_label[idx*self.batch_size : (idx+1)*self.batch_size])
                batch_labels[2] = batch_labels[2][:, padding:-padding, padding:-padding, :]          
                
                # Run the model
                train_sum, _, _, stg3_err, bHF_err, HF_err, LF_err, HFLF_idx, TV = self.sess.run([            self.merged_summary_train,
                                                                                          self.stg_train_op,
                                                                                          self.HF_train_op,
                                                                                          self.stg3_loss,
                                                                                          self.before_HF_loss,
                                                                                          self.HF_loss,
                                                                                          self.LF_loss,
                                                                                          self.HFLF_idx,
                                                                                          self.TV_stg3_output
                                                                                          ], 
                                                                                          feed_dict={
                                                                                                      self.images: batch_images, 
                                                                                                      self.stg1_labels: batch_labels[0],
                                                                                                      self.stg2_labels: batch_labels[1],
                                                                                                      self.stg3_labels: batch_labels[2],
                                                                                                      self.dropout: 1.
                                                                                                    })   

                final_err = (HF_err * (HFLF_idx[0].size) + LF_err * (HFLF_idx[1].size)) / (HFLF_idx[0].size + HFLF_idx[1].size)   
                
                avg_500_loss[0] += stg3_err
                avg_500_loss[1] += bHF_err    
                avg_500_loss[2] += HF_err    
                avg_500_loss[3] += LF_err    
                avg_500_loss[4] += final_err    
    
                batch_pbar.set_description("Batch: [%2d]" % (idx+1))
                #batch_pbar.refresh()
            

            if ep % 5 == 0:

                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                # Validation
                ## Run the test images
                test_sum, val_stg3_err, val_bHF_err, val_HF_err, val_LF_err, val_HFLF_idx = self.sess.run([  self.merged_summary_test,
                                                                                                             self.stg3_loss,
                                                                                                             self.before_HF_loss,
                                                                                                             self.HF_loss,
                                                                                                             self.LF_loss,                                                                                                             
                                                                                                             self.HFLF_idx,
                                                                                                             ], 
                                                                                                             feed_dict={
                                                                                                                         self.images: val_images, 
                                                                                                                         self.stg1_labels: val_label[0],
                                                                                                                         self.stg2_labels: val_label[1],
                                                                                                                         self.stg3_labels: val_label[2],
                                                                                                                         self.dropout: 1.
                                                                                                                       })
 
                val_final_loss = (val_HF_err * (val_HFLF_idx[0].size) + val_LF_err * (val_HFLF_idx[1].size)) / (val_HFLF_idx[0].size + val_HFLF_idx[1].size)
                #val_final_loss = (val_HF_err*test_HF_size + val_LF_err*test_LF_size) / test_data_size
                                                                                                                       
                #print("Val. HF num: ", (val_HFLF_idx[0].size), "Val. LF num", (val_HFLF_idx[1].size))

                for i in range(len(avg_500_loss)):
                    avg_500_loss[i] /= (train_batch_num*5)
               
                print("Epoch: [%2d], Average train loss of 5 epoches: stg3 loss: [%.8f], HF loss: [%.8f]->[%.8f], LF loss: [%.8f], Final loss: [%.8f]" \
                     % ((ep+1), avg_500_loss[0], avg_500_loss[1], avg_500_loss[2], avg_500_loss[3], avg_500_loss[4]))             
                print("Epoch: [%2d], Test stg loss: [%.8f], HF loss: [%.8f]->[%.8f], LF loss: [%.8f], Final loss: [%.8f]\n"\
                     % ((ep+1), val_stg3_err, val_bHF_err, val_HF_err, val_LF_err, val_final_loss))
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)
                


    def build_edsr_v1(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        
        mean_x = tf.reduce_mean(self.input)
        image_input  =self.input - mean_x
        mean_y = tf.reduce_mean(self.image_target)
        target = self.image_target - mean_y
        
        #image_input = self.input
        #target = self.image_target
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        self.logits = mz.build_model({"scale":self.scale,"feature_size" :64})
        self.l1_loss = tf.reduce_mean(tf.losses.absolute_difference(target,self.logits))
        mse = tf.reduce_mean(tf.squared_difference(target,self.logits))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/(mse)
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss)
        
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target, collections=['train'])
            tf.summary.image("output_image",self.logits, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['test'])
            tf.summary.scalar("PSNR",PSNR,  collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target , collections=['test'])
            tf.summary.image("output_image",self.logits, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_v1(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

        #mean of DIV2K
        train_mean = np.zeros((1,1,3))
        train_mean[0][0][0] = 113.9427
        train_mean[0][0][1] = 111.3509
        train_mean[0][0][2] = 103.1092

        #mean of set5
        test_mean = np.zeros((1,1,3))
        test_mean[0][0][0] = 140.6670
        test_mean[0][0][1] = 112.9228
        test_mean[0][0][2] = 85.2956


        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/preprocessed_scale_2", test_mean,type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K/", train_mean)

        

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
       
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(400, self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d]" % ((ep+1)))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
            print("learning_rate: ", learning_rate)
            if ep%4000 == 0 and ep != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                
               
                
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 1.,
                                                                                        self.lr:learning_rate})
                                                           
    
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
              
            if ep % 5 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum, train_loss = self.sess.run([self.merged_summary_train, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_images, 
                                                                                                    self.image_target: batch_labels,
                                                                                                    self.dropout: 1.
                                                                                                                   })
                batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                test_sum, test_loss = self.sess.run([self.merged_summary_test, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_test_images, 
                                                                                                    self.image_target: batch_test_labels,
                                                                                                    self.dropout: 1.})
                                                                                                                       

               
                    
                print("Epoch: [{}], Train_loss: {}".format((ep+1), train_loss))     
                print("Epoch: [{}], Test_loss: {}".format((ep+1), test_loss))  
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)




    def build_edsr_v2(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, None, None, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        """
        mean_x = tf.reduce_mean(self.input)
        image_input  = self.input - mean_x
        mean_y = tf.reduce_mean(self.image_target)
        target = self.image_target - mean_y
        """
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        logits2, logits4 = mz.build_model({"scale":self.scale,"feature_size" :64})

        lap_logit2 = laplacian_filter(logits2)
        lap_logit4 = laplacian_filter(logits4)

        self.l1_loss2 = tf.reduce_mean(tf.losses.absolute_difference(target,logits2))
        self.l1_loss4 = tf.reduce_mean(tf.losses.absolute_difference(target,logits4))
  
        self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss2)
        self.train_op4 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss4)

        if self.scale == 2:

            self.logits = logits2
            self.l1_loss = self.l1_loss2
            self.train_op = self.train_op2
        
        elif self.scale == 4:

            self.logits = logits4
            self.l1_loss = self.l1_loss4
            self.train_op = self.train_op4

        mse = tf.reduce_mean(tf.squared_difference(target*255.,self.logits*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",self.logits*255, collections=['train'])
            tf.summary.scalar("lap2",lap_logit2, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['test'])
            tf.summary.scalar("PSNR",PSNR,  collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target*255 , collections=['test'])
            tf.summary.image("output_image",self.logits*255, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_v2(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

    
        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/fake_preprocessed_scale_"+str(self.scale),type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K_fake/", lrtype='bicubic', type='train')

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
       
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
           
            if ep%4000 == 0 and ep != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                
                
                #print(batch_images, batch_labels)
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 1.,
                                                                                        self.lr:learning_rate})
                                                           
                
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
              
            if ep % 50 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum, train_loss = self.sess.run([self.merged_summary_train, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_images, 
                                                                                                    self.image_target: batch_labels,
                                                                                                    self.dropout: 1.
                                                                                                                   })
                #batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                test_sum, test_loss = self.sess.run([self.merged_summary_test, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: test_data, 
                                                                                                    self.image_target: test_label,
                                                                                                    self.dropout: 1.})
                                                                                                                       

               
                    
                print("Epoch: [{}], Train_loss: {}".format((ep+1), train_loss))     
                print("Epoch: [{}], Test_loss: {}".format((ep+1), test_loss))  
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)
        
              
    def build_edsr_attention_v1(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, None, None, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        logits2, logits4, self.testw = mz.build_model({"scale":self.scale,"feature_size" :64,"dropout":self.dropout, 
                                            "is_training":self.is_training})

        
        self.l1_loss2 = tf.reduce_mean(tf.losses.absolute_difference(target,logits2))
        self.l1_loss4 = tf.reduce_mean(tf.losses.absolute_difference(target,logits4))
  
        self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss2)
        self.train_op4 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss4)

        if self.scale == 2:

            self.logits = logits2
            self.l1_loss = self.l1_loss2
            self.train_op = self.train_op2
        
        elif self.scale == 4:

            self.logits = logits4
            self.l1_loss = self.l1_loss4
            self.train_op = self.train_op4

        mse = tf.reduce_mean(tf.squared_difference(target*255.,self.logits*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",self.logits*255, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['test'])
            tf.summary.scalar("PSNR",PSNR,  collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target*255 , collections=['test'])
            tf.summary.image("output_image",self.logits*255, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_attention_v1(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

    
        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/preprocessed_scale_"+str(self.scale),type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K/")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
       
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
           
            if ep%8000 == 0 and ep != 0 and learning_rate >= 2.5e-5:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                
                
                #print(batch_images, batch_labels)
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 0.6,
                                                                                        self.is_training:True,
                                                                                        self.lr:learning_rate})

    
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
              
            if ep % 50 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum, train_loss = self.sess.run([self.merged_summary_train, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_images, 
                                                                                                    self.image_target: batch_labels,
                                                                                                    self.dropout: 1.,
                                                                                                    self.is_training:False})

                batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                test_sum, test_loss = self.sess.run([self.merged_summary_test, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_test_images, 
                                                                                                    self.image_target: batch_test_labels,
                                                                                                    self.dropout: 1.,
                                                                                                    self.is_training:False})
                                                                                                                       

               
                    
                print("Epoch: [{}], Train_loss: {}".format((ep+1), train_loss))     
                print("Epoch: [{}], Test_loss: {}".format((ep+1), test_loss))  
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)

    def build_edsr_1X1_v1(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, None, None, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        logits2, logits4 = mz.build_model({"scale":self.scale,"feature_size" :64,"dropout":self.dropout, 
                                            "is_training":self.is_training})

        
        self.l1_loss2 = tf.reduce_mean(tf.losses.absolute_difference(target,logits2))
        self.l1_loss4 = tf.reduce_mean(tf.losses.absolute_difference(target,logits4))
  
        self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss2)
        self.train_op4 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss4)

        if self.scale == 2:

            self.logits = logits2
            self.l1_loss = self.l1_loss2
            self.train_op = self.train_op2
        
        elif self.scale == 4:

            self.logits = logits4
            self.l1_loss = self.l1_loss4
            self.train_op = self.train_op4

        mse = tf.reduce_mean(tf.squared_difference(target*255.,self.logits*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",self.logits*255, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['test'])
            tf.summary.scalar("PSNR",PSNR,  collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target*255 , collections=['test'])
            tf.summary.image("output_image",self.logits*255, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_1X1_v1(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

    
        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/preprocessed_scale_"+str(self.scale),type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K/")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
       
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
           
            if ep%8000 == 0 and ep != 0 and learning_rate >= 2.5e-5:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                
                
                #print(batch_images, batch_labels)
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 0.6,
                                                                                        self.is_training:True,
                                                                                        self.lr:learning_rate})

    
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
              
            if ep % 50 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum, train_loss = self.sess.run([self.merged_summary_train, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_images, 
                                                                                                    self.image_target: batch_labels,
                                                                                                    self.dropout: 1.,
                                                                                                    self.is_training:False})

                batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                test_sum, test_loss = self.sess.run([self.merged_summary_test, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_test_images, 
                                                                                                    self.image_target: batch_test_labels,
                                                                                                    self.dropout: 1.,
                                                                                                    self.is_training:False})
                                                                                                                       

               
                    
                print("Epoch: [{}], Train_loss: {}".format((ep+1), train_loss))     
                print("Epoch: [{}], Test_loss: {}".format((ep+1), test_loss))  
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)
                
    def save_ckpt(self, checkpoint_dir, ckpt_name, step):
        """
        Save the checkpoint. 
        According to the scale, use different folder to save the models.
        """          
        
        print(" [*] Saving checkpoints...step: [{}]".format(step))
        model_name = ckpt_name
        
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
        
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_ckpt(self, checkpoint_dir, ckpt_name=""):
        """
        Load the checkpoint. 
        According to the scale, read different folder to load the models.
        """     
        
        print(" [*] Reading checkpoints...")
        if ckpt_name == "":
            model_dir = "%s_%s_%s" % ("srcnn", "scale", self.scale)
        else:
            model_dir = ckpt_name
            
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt  and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        
            return True
        
        else:
            return False          


    def load_divk(self, dataset_path, mean=0, lrtype='bicubic', type='train'):

        #dataset_path = "/home/ubuntu/dataset/SuperResolution/DIV2K/"

        if type == "train":

            sub_path = "DIV2K_train"

            lr_subpath = []
            lr_imgs = []

            if lrtype == 'bicubic':
                lr_subpath.append(sub_path + "_LR_bicubic/" + "X" + str(self.scale))
            elif lrtype == 'unknown':
                lr_subpath.append(sub_path + "_LR_unknown/" + "X" + str(self.scale))
            elif lrtype == 'all':
                lr_subpath.append(sub_path + "_LR_bicubic/" + "X" + str(self.scale))
                lr_subpath.append(sub_path + "_LR_unknown/" + "X" + str(self.scale))
            else:
                #print("lrtype error: [{}]".format(lrtype))
                return  0

            HR_path = os.path.join(dataset_path, sub_path + "_HR")
            hr_imgs = os.listdir(HR_path)
            hr_imgs = [os.path.join(HR_path, hr_imgs[i]) for i in range(len(hr_imgs))]

            for path_idx in range(len(lr_subpath)):
                LR_path = os.path.join(dataset_path, lr_subpath[path_idx])
                file_name = [os.path.basename(hr_imgs[i]) for i in range(len(hr_imgs))]
                lr_imgs.append([os.path.join(LR_path, file_name[i].split(".")[0] + 'x' + str(self.scale)+'.' + file_name[i].split(".")[1]) for i in range(len(hr_imgs))])
            


        elif type == "test":

            lr_imgs = []
            images = os.listdir(dataset_path)
            #for i in range(len(images)//2):
            lr_imgs.append([os.path.join(dataset_path, "img_"+str(i+1)+"_SRF_" + str(self.scale)+"_LR.png") for i in range(len(images)//2)])
            hr_imgs = ([os.path.join(dataset_path, "img_"+str(i+1)+"_SRF_"+ str(self.scale)+"_HR.png")  for i in range(len(images)//2)])
        
           
        hr_list = []
        lr_list = []
        lr_list2 = []

        for i in range(len(hr_imgs)):

           sys.stdout.write("Load data:{}/{}".format(i,len(hr_imgs))+'\r')
           sys.stdout.flush()
           hr_list.append(misc.imread(hr_imgs[i]))            
           lr_list.append(misc.imread(lr_imgs[0][i]))
           if lrtype == 'all':
            lr_list2.append(misc.imread(lr_imgs[1][i]))
           #if lrtype == 'bicubic' and i > 20: break

        print("[load_divk] type: [{}], lrtype: [{}]".format(type, lrtype))
        print("[load_divk] HR images number: [{}]".format(len(hr_list)))
        print("[load_divk] LR bicubic images number: [{}]".format(len(lr_list)))

        if lrtype == 'all':
            print("[load_divk] LR unknown images number: [{}]".format(len(lr_list2)))
            
        if lrtype == 'all':
            return list(zip(lr_list + lr_list2, hr_list + hr_list))
        else:
            return list(zip(lr_list, hr_list))


    def build_edsr_local_att_v1(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, None, None, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        """
        mean_x = tf.reduce_mean(self.input)
        image_input  = self.input - mean_x
        mean_y = tf.reduce_mean(self.image_target)
        target = self.image_target - mean_y
        """
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        logits2, logits4 = mz.build_model({"scale":self.scale,"feature_size" :64})

        lap_logit2 = laplacian_filter(logits2)
        lap_logit4 = laplacian_filter(logits4)

        self.l1_loss2 = tf.reduce_mean(tf.losses.absolute_difference(target,logits2))
        self.l1_loss4 = tf.reduce_mean(tf.losses.absolute_difference(target,logits4))
  
        self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss2)
        self.train_op4 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss4)

        if self.scale == 2:

            self.logits = logits2
            self.l1_loss = self.l1_loss2
            self.train_op = self.train_op2
        
        elif self.scale == 4:

            self.logits = logits4
            self.l1_loss = self.l1_loss4
            self.train_op = self.train_op4

        mse = tf.reduce_mean(tf.squared_difference(target*255.,self.logits*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",self.logits*255, collections=['train'])
            tf.summary.scalar("lap2",lap_logit2, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['test'])
            tf.summary.scalar("PSNR",PSNR,  collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target*255 , collections=['test'])
            tf.summary.image("output_image",self.logits*255, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_local_att_v1(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

    
        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/preprocessed_scale_"+str(self.scale),type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K/")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
       
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
           
            if ep%4000 == 0 and ep != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                
                
                #print(batch_images, batch_labels)
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 1.,
                                                                                        self.lr:learning_rate})
                                                           
                
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
              
            if ep % 50 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum, train_loss = self.sess.run([self.merged_summary_train, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_images, 
                                                                                                    self.image_target: batch_labels,
                                                                                                    self.dropout: 1.
                                                                                                                   })
                batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                test_sum, test_loss = self.sess.run([self.merged_summary_test, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_test_images, 
                                                                                                    self.image_target: batch_test_labels,
                                                                                                    self.dropout: 1.})
                                                                                                                       

               
                    
                print("Epoch: [{}], Train_loss: {}".format((ep+1), train_loss))     
                print("Epoch: [{}], Test_loss: {}".format((ep+1), test_loss))  
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)


    def build_edsr_local_att_v2_upsample(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, None, None, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
     
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        logits2, logits4 = mz.build_model({"scale":self.scale,"feature_size" :64, "kernel_size":3})

        def regional_l1loss(target, logits):

            bsize, a, b, c = logits.get_shape().as_list()

            logits = tf.reshape(logits,(tf.shape(logits)[0],16,a//16,16,b//16,c))
            target = tf.reshape(target,(tf.shape(logits)[0],16,a//16,16,b//16,c))
            
            logits_loss = tf.abs(target - logits)       
            logits_loss = tf.reduce_sum(logits_loss, axis=[1,3])
            logits_loss = logits_loss/256.
            logits_loss = tf.reduce_sum(logits_loss)

            return logits_loss

        #self.l1_loss2 = tf.reduce_mean(tf.losses.absolute_difference(target,logits2))
        #self.l1_loss4 = tf.reduce_mean(tf.losses.absolute_difference(target,logits4))
        self.l1_loss2 = regional_l1loss(target,logits2)
        self.l1_loss4 = regional_l1loss(target,logits4)
  
        self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss2)
        self.train_op4 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss4)

        if self.scale == 2:

            self.logits = logits2
            self.l1_loss = self.l1_loss2
            self.train_op = self.train_op2
        
        elif self.scale == 4:

            self.logits = logits4
            self.l1_loss = self.l1_loss4
            self.train_op = self.train_op4

        mse = tf.reduce_mean(tf.squared_difference(target*255.,self.logits*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",self.logits*255, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['test'])
            tf.summary.scalar("PSNR",PSNR,  collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target*255 , collections=['test'])
            tf.summary.image("output_image",self.logits*255, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_local_att_v2_upsample(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

        print("/home/ubuntu/dataset/SuperResolution/Set5/validation_scale_"+str(self.scale)+"/")
        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/validation_scale_"+str(self.scale)+"/",type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K/",lrtype="all", type="train")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
       
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(5745, self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
           
            if ep%4000 == 0 and ep != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                
                
                #print(batch_images, batch_labels)
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 1.,
                                                                                        self.lr:learning_rate})
                                                           
                
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
              
            if ep % 50 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum, train_loss = self.sess.run([self.merged_summary_train, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_images, 
                                                                                                    self.image_target: batch_labels,
                                                                                                    self.dropout: 1.
                                                                                                                   })
                #batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                test_sum, test_loss = self.sess.run([self.merged_summary_test, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: test_data, 
                                                                                                    self.image_target: test_label,
                                                                                                    self.dropout: 1.})
                                                                                                                       

               
                    
                print("Epoch: [{}], Train_loss: {}".format((ep+1), train_loss))     
                print("Epoch: [{}], Test_loss: {}".format((ep+1), test_loss))  
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)


    def build_edsr_attention_v2(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, None, None, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
     
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        logits2, logits4 = mz.build_model({"scale":self.scale,"feature_size" :64, "kernel_size":3})

        def regional_l1loss(target, logits):

            bsize, a, b, c = logits.get_shape().as_list()

            center_logits = tf.slice(logits, [0,7,7,0],[-1, 8, 8, c])
            center_target = tf.slice(target, [0,7,7,0],[-1, 8, 8, c])

            bsize, a, b, c = center_logits.get_shape().as_list()

            #logits = tf.reshape(center_logits,(tf.shape(center_logits)[0],8,a//8,8,b//8,c))
            #target = tf.reshape(center_target,(tf.shape(center_target)[0],8,a//8,8,b//8,c))
            #print(logits)

            logits = center_logits
            target = center_target
            
            logits_loss = tf.abs(target - logits)       
            logits_loss = tf.reduce_sum(logits_loss, axis=[1,3])
            logits_loss = logits_loss/64.
            logits_loss = tf.reduce_sum(logits_loss)

            return logits_loss

        #self.l1_loss2 = tf.reduce_mean(tf.losses.absolute_difference(target,logits2))
        #self.l1_loss4 = tf.reduce_mean(tf.losses.absolute_difference(target,logits4))
        self.l1_loss2 = regional_l1loss(target,logits2)
        self.l1_loss4 = regional_l1loss(target,logits4)

        self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss2)
        self.train_op4 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss4)

        if self.scale == 2:

            self.logits = logits2
            self.l1_loss = self.l1_loss2
            self.train_op = self.train_op2
        
        elif self.scale == 4:

            self.logits = logits4
            self.l1_loss = self.l1_loss4
            self.train_op = self.train_op4

        mse = tf.reduce_mean(tf.squared_difference(target*255.,self.logits*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",self.logits*255, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss, collections=['test'])
            tf.summary.scalar("PSNR",PSNR,  collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target*255 , collections=['test'])
            tf.summary.image("output_image",self.logits*255, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_attention_v2(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

        print("/home/ubuntu/dataset/SuperResolution/Set5/validation_scale_"+str(self.scale)+"/")
        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/validation_24_scale_"+str(self.scale)+"/",type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K/",lrtype="all", type="train")

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
       
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
           
            if ep%4000 == 0 and ep != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                
                
                #print(batch_images, batch_labels)
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 1.,
                                                                                        self.lr:learning_rate})
                                                           
                
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
              
            if ep % 50 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum, train_loss = self.sess.run([self.merged_summary_train, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_images, 
                                                                                                    self.image_target: batch_labels,
                                                                                                    self.dropout: 1.
                                                                                                                   })
                #batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                test_sum, test_loss = self.sess.run([self.merged_summary_test, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: test_data, 
                                                                                                    self.image_target: test_label,
                                                                                                    self.dropout: 1.})
                                                                                                                       

               
                    
                print("Epoch: [{}], Train_loss: {}".format((ep+1), train_loss))     
                print("Epoch: [{}], Test_loss: {}".format((ep+1), test_loss))  
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)

    def build_edsr_v2_dual(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, None, None, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.inverse = tf.placeholder(tf.bool, name='inverse')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        """
        mean_x = tf.reduce_mean(self.input)
        image_input  = self.input - mean_x
        mean_y = tf.reduce_mean(self.image_target)
        target = self.image_target - mean_y
        """
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        logits2, logits4 = mz.build_model({"scale":self.scale,"feature_size" :64})

        self.l1_loss2 = tf.reduce_sum(tf.abs(target - logits2), axis=[1,2,3])
        self.l1_loss4 = tf.reduce_sum(tf.abs(target - logits4), axis=[1,2,3])

        self.l2_boolmask = tf.less(self.l1_loss2, self.l1_loss4)
        self.l4_boolmask = tf.less(self.l1_loss4, self.l1_loss2)

        def inv_ture():
            self.l2_boolmask = tf.logical_not(self.l2_boolmask)
            self.l4_boolmask = tf.logical_not(self.l4_boolmask)
            return self.l2_boolmask,  self.l4_boolmask
        def inv_false():
            self.l2_boolmask = tf.less(self.l1_loss2, self.l1_loss4)
            self.l4_boolmask = tf.less(self.l1_loss4, self.l1_loss2)
            return self.l2_boolmask,  self.l4_boolmask

        self.tl2_boolmask,  self.tl4_boolmask  = tf.cond(self.inverse , inv_ture, inv_false)

        self.l1_loss2_train = tf.boolean_mask(self.l1_loss2, self.tl2_boolmask, "boolmask2")
        self.l1_loss4_train = tf.boolean_mask(self.l1_loss4, self.tl4_boolmask, "boolmask4")
       
        self.l1_loss2_train = tf.reduce_mean(self.l1_loss2_train)
        self.l1_loss4_train = tf.reduce_mean(self.l1_loss4_train)

        self.train_op2 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss2_train)
        self.train_op4 = tf.train.AdamOptimizer(self.lr).minimize(self.l1_loss4_train)

        self.l1_loss2_sum = tf.reduce_mean(self.l1_loss2)
        self.l1_loss4_sum = tf.reduce_mean(self.l1_loss4)
    

        mse = tf.reduce_mean(tf.squared_difference(target*255.,logits2*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        mse4 = tf.reduce_mean(tf.squared_difference(target*255.,logits4*255.))    
        PSNR4 = tf.constant(255**2,dtype=tf.float32)/mse4
        PSNR4 = tf.constant(10,dtype=tf.float32)*log10(PSNR4)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("loss1", self.l1_loss2_sum, collections=['train'])
            tf.summary.scalar("MSE1", mse, collections=['train'])
            tf.summary.scalar("PSNR1",PSNR, collections=['train'])
            tf.summary.image("input_image1",self.input , collections=['train'])
            tf.summary.image("target_image1",target*255, collections=['train'])
            tf.summary.image("output_image1",logits2*255, collections=['train'])

            tf.summary.scalar("loss2", self.l1_loss4_sum, collections=['train'])
            tf.summary.scalar("MSE2", mse4, collections=['train'])
            tf.summary.scalar("PSNR2",PSNR4, collections=['train'])
            tf.summary.image("output_image2",logits4*255, collections=['train'])
            
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.l1_loss2_sum, collections=['test'])
            tf.summary.scalar("PSNR",PSNR,  collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.image("input_image",self.input, collections=['test'])
            tf.summary.image("target_image",target*255 , collections=['test'])
            tf.summary.image("output_image",logits2*255, collections=['test'])

            tf.summary.scalar("loss2", self.l1_loss4_sum, collections=['test'])
            tf.summary.scalar("MSE2", mse4, collections=['test'])
            tf.summary.scalar("PSNR2",PSNR4, collections=['test'])
            tf.summary.image("input_image2",self.input , collections=['test'])
            tf.summary.image("target_image2",target*255, collections=['test'])
            tf.summary.image("output_image2",logits4*255, collections=['test'])

            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_v2_dual(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

    
        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/fake_preprocessed_scale_"+str(self.scale),type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K_fake/", lrtype='bicubic', type='train')

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
       
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
           
            if ep%4000 == 0 and ep != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                
                
                #print(batch_images, batch_labels)
                
                # Run the model

                if ep < 5000: 
                    inv_bool = np.random.choice([True, False])
                else:
                    inv_bool = False
                _,_,l2,l4 = self.sess.run([self.train_op2, self.train_op4, self.tl2_boolmask,  self.tl4_boolmask],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 1.,
                                                                                        self.lr:learning_rate,
                                                                                        self.inverse:inv_bool})
                print(l2)
                print(l4)
                print(inv_bool)                    
                batch_pbar.set_description("Batch: [%2d]" % (idx+1))
                #batch_pbar.refresh()
              
            if ep % 50 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                        feed_dict={
                                                                            self.input: batch_images, 
                                                                            self.image_target: batch_labels,
                                                                            self.dropout: 1.,
                                                                            self.inverse: False})
                #batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                test_sum = self.sess.run(self.merged_summary_test, 
                                                                        feed_dict={
                                                                            self.input: test_data, 
                                                                            self.image_target: test_label,
                                                                            self.dropout: 1.,
                                                                            self.inverse: False})
                                                                                                                       

               
                    
                print("Epoch: [{}]".format((ep+1)))     
                print("Epoch: [{}]".format((ep+1)))  
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)

    def build_edsr_lsgan(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='images')
        #self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        """
        mean_x = tf.reduce_mean(self.input)
        image_input  = self.input - mean_x
        mean_y = tf.reduce_mean(self.image_target)
        target = self.image_target - mean_y
        """
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        
        gen_f, dis_f = mz.build_model({"d_inputs":None,"d_target":self.target,"scale":self.scale,"feature_size" :64, "reuse":False, "is_training":True,"is_generate":True})
        _, dis_t = mz.build_model({"d_inputs":self.target,"d_target":self.target,"scale":self.scale,"feature_size" :64,"reuse":True, "is_training":True,"is_generate":False})

        

        #Calculate gradient panalty
        self.epsilon = epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.target + (1. - epsilon) * (gen_f)
        _, d_hat = mz.build_model({"d_inputs":x_hat,"d_target":self.target,"scale":self.scale,"feature_size" :64, "reuse":True, "is_training":True,"is_generate":False})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=[1,2,3]))
        d_gp = tf.reduce_mean(tf.square(d_gp - 1.0)) * 10

        self.disc_ture_loss = disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)

        reconstucted_weight = 1.0  #StarGAN is 10
        self.d_loss =   disc_fake_loss - disc_ture_loss + d_gp
        self.g_l1loss = tf.reduce_mean(tf.losses.absolute_difference(target,gen_f))
        self.g_loss =  -1.0*disc_fake_loss + reconstucted_weight*self.g_l1loss
        
        """
        #Genric GAN Loss
        self.g_l1loss = tf.reduce_mean(tf.losses.absolute_difference(target,gen_f))
        self.d_loss = -tf.reduce_mean(tf.log(dis_t) + tf.log(1.0 - dis_f))
        self.g_loss = -tf.reduce_mean(tf.log(dis_f))
        """

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("EDSR_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("EDSR_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=generator_variables)
        self.train_l1 = tf.train.AdamOptimizer(self.lr,beta1=0.5, beta2=0.999).minimize(self.g_l1loss, var_list=generator_variables)
        

        """
        train_variables = tf.trainable_variables()
        discriminator_variables = [v for v in train_variables if v.name.startswith("EDSR_dis")]
        self.clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for
                                var in discriminator_variables]

        """
        """
        alpha = 0.00005
        optimizer = tf.train.RMSPropOptimizer(self.lr)
        gvs_d = optimizer.compute_gradients(self.d_loss)
        gvs_g = optimizer.compute_gradients(self.g_loss)
        gvs_l1 = optimizer.compute_gradients(self.g_l1loss)

        wgvs_d = [(grad*alpha, var) for grad, var in gvs_d]
        wgvs_g = [(grad*alpha, var) for grad, var in gvs_g]

        self.train_d = optimizer.apply_gradients(wgvs_d)
        self.train_g = optimizer.apply_gradients(wgvs_g)
        self.train_l1 = optimizer.apply_gradients(gvs_l1)
        """
        #calculate discriminator accuracy


        mse = tf.reduce_mean(tf.squared_difference(target*255.,gen_f*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        mse_ref = tf.reduce_mean(tf.squared_difference(target*255.,self.image_input*255.))    
        PSNR_ref = tf.constant(255**2,dtype=tf.float32)/mse_ref
        PSNR_ref = tf.constant(10,dtype=tf.float32)*log10(PSNR_ref)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.g_l1loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d_true_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("d_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("grad_loss", d_gp, collections=['train'])
            tf.summary.scalar("dis_f_mean", tf.reduce_mean(dis_f), collections=['train'])
            tf.summary.scalar("dis_t_mean", tf.reduce_mean(dis_t), collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",gen_f*255, collections=['train'])
            tf.summary.image("enhence_img",(2.0*gen_f-target)*255, collections=['train'])
            tf.summary.image("dis_f_img",dis_f*255, collections=['train'])
            tf.summary.image("dis_t_img",dis_t*255, collections=['train'])
            tf.summary.image("dis_diff",10*tf.abs(dis_t-dis_f)*255, collections=['train'])
            tf.summary.histogram("d_false", dis_f, collections=['train'])
            tf.summary.histogram("d_true", dis_t, collections=['train'])
    
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.g_l1loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", disc_fake_loss, collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.scalar("PSNR_ref",PSNR_ref, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("target_image",target*255, collections=['test'])
            tf.summary.image("output_image",gen_f*255, collections=['test'])
        
            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_lsgan(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

        #Single image    
        #test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/fake_preprocessed_scale_"+str(self.scale),type="test")
        #dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K_fake/", lrtype='bicubic', type='train')

        #96X96
        #test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/validation96_scale_"+"2"+"/",type="test")
        #dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K_base/", lrtype='bicubic', type='train')

        #gcloud96X96
        test_dataset = self.load_divk("//home/moxalab/data/SuperResolution/Set5/validation96_scale_"+"2"+"/",type="test")
        dataset = self.load_divk("/home/moxalab/data/SuperResolution/DIV2K_base/", lrtype='bicubic', type='train')

        #48X48
        #test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/validation_scale_"+ str(self.scale),type="test")
        #dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K/", lrtype='all', type='train')
       

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
        
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(0,self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
            action = 0
            cycle_times = 100000
            current_cycle = ep%cycle_times

            if current_cycle < 500:
                action = 2
            elif current_cycle >= 1000 and current_cycle < 2000:
                action = 2
            elif current_cycle >= 2000:
                action = 2

            itr_per_epoch = len(train_data)//self.batch_size 
            if (ep*itr_per_epoch)%300000 == 0 and ep*itr_per_epoch != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                batch_pbar.set_description("Batch: [%2d], Action: [%d]" % ((idx+1) ,action))
                itera_counter += 1
                batch_index = idx*self.batch_size 
                #batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                #start_time = time.time()
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, 1, self.image_size*2,batch_index, self.batch_size)
                #elapse = time.time() - start_time
                #print("batch elapse:", elapse)
                # Select different action each cycle time

                #start_time = time.time()
                if action == 0:
                    
                    self.sess.run(self.train_l1,
                                             feed_dict={self.input: batch_images,
                                                        self.image_target: batch_labels,
                                                        self.dropout: 1.,
                                                        self.lr:learning_rate})

                elif action == 1:

                    
                    self.sess.run([self.train_d, self.d_loss],
                                                 feed_dict={self.input: batch_images,
                                                            self.image_target: batch_labels,
                                                            self.dropout: 1.,
                                                            self.lr:1e-4})
                    #self.sess.run(self.clip_discriminator_var_op)
                          

                elif action == 2:

                    
                    if idx%5 == 0:
                    
                        t = self.sess.run([self.train_g],
                                                 feed_dict={self.input: batch_images,
                                                            self.image_target: batch_labels,
                                                            self.dropout: 1.,
                                                            self.lr:learning_rate})
                       
                    
                    _, loss = self.sess.run([self.train_d, self.d_loss],
                                             feed_dict={self.input: batch_images,
                                                        self.image_target: batch_labels,
                                                        self.dropout: 1.,
                                                        self.lr:learning_rate})
                    #self.sess.run(self.clip_discriminator_var_op)

                #elapse = time.time() - start_time 
                #print("training elapse:", elapse)           
                #batch_pbar.refresh()
              
            if ep % 50 == 1:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                        feed_dict={
                                                                            self.input: batch_images, 
                                                                            self.image_target: batch_labels,
                                                                            self.dropout: 1.
                                                                           })
                #batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                
                test_sum = self.sess.run(self.merged_summary_test, 
                                                                        feed_dict={
                                                                            self.input: test_data, 
                                                                            self.image_target: test_label,
                                                                            self.dropout: 1.,
                                                                            })
                                                                                                                       
                
               
                    
                print("Epoch: [{}]".format((ep+1)))       
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)
        
    def build_edsr_lsgan_up(self):
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        _, dis_t = mz.build_model({"d_inputs":self.target,"d_target":self.target,"scale":self.scale,"feature_size" :64,"reuse":False, "is_training":True,"is_generate":False})
        gen_f, dis_f = mz.build_model({"d_inputs":None,"d_target":self.target,"scale":self.scale,"feature_size" :64, "reuse":True, "is_training":True,"is_generate":True})

        #Calculate gradient panalty
        self.epsilon = epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.target + (1. - epsilon) * (gen_f)
        _, d_hat = mz.build_model({"d_inputs":x_hat,"d_target":self.target,"scale":self.scale,"feature_size" :64, "reuse":True, "is_training":True,"is_generate":False})
        
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=[1,2,3]))
        d_gp = tf.reduce_mean(tf.square(d_gp - 1.0)) * 10

        self.disc_ture_loss = disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)

        reconstucted_weight = 1.0  #StarGAN is 10
        self.d_loss =   disc_fake_loss - disc_ture_loss + d_gp
        self.g_l1loss = tf.reduce_mean(tf.losses.absolute_difference(target,gen_f))
        self.g_loss =  -1.0*disc_fake_loss + reconstucted_weight*self.g_l1loss
        
       

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("EDSR_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("EDSR_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=generator_variables)
        self.train_l1 = tf.train.AdamOptimizer(self.lr,beta1=0.5, beta2=0.999).minimize(self.g_l1loss, var_list=generator_variables)
        

        #calculate discriminator accuracy


        mse = tf.reduce_mean(tf.squared_difference(target*255.,gen_f*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.g_l1loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d_true_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("d_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("grad_loss", d_gp, collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",gen_f*255, collections=['train'])
            tf.summary.image("enhence_img",(2.0*gen_f-target)*255, collections=['train'])
            tf.summary.image("dis_f_img",10*dis_f*255, collections=['train'])
            tf.summary.image("dis_t_img",10*dis_t*255, collections=['train'])
            tf.summary.image("dis_diff",tf.abs(dis_t-dis_t)*255, collections=['train'])
            tf.summary.histogram("d_false", dis_f, collections=['train'])
            tf.summary.histogram("d_true", dis_t, collections=['train'])

    
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):
            tf.summary.scalar("loss", self.g_l1loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", disc_fake_loss, collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("target_image",target*255, collections=['test'])
            tf.summary.image("output_image",gen_f*255, collections=['test'])
        
            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_lsgan_up(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

        #Single image    
        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/fake_preprocessed_scale_"+str(self.scale),type="test")
        dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K_fake/", lrtype='bicubic', type='train')


        #48X48
        #test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/validation_scale_"+ str(self.scale),type="test")
        #dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/DIV2K/", lrtype='all', type='train')
       

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
        
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
            action = 0
            cycle_times = 10000
            current_cycle = ep%cycle_times

            if current_cycle < 1000:
                action = 2
            elif current_cycle >= 1000 and current_cycle < 5000:
                action = 1
            elif current_cycle >= 5000:
                action = 2

            itr_per_epoch = len(train_data)//self.batch_size 
            if (ep*itr_per_epoch)%300000 == 0 and ep*itr_per_epoch != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                batch_pbar.set_description("Batch: [%2d], Action: [%d]" % ((idx+1) ,action))
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
               
                # Select different action each cycle time

                if action == 0:
                    
                    self.sess.run(self.train_l1,
                                             feed_dict={self.input: batch_images,
                                                        self.image_target: batch_labels,
                                                        self.dropout: 1.,
                                                        self.lr:learning_rate})

                elif action == 1:

                    
                    self.sess.run([self.train_d, self.d_loss],
                                                 feed_dict={self.input: batch_images,
                                                            self.image_target: batch_labels,
                                                            self.dropout: 1.,
                                                            self.lr:learning_rate})
                    #self.sess.run(self.clip_discriminator_var_op)
                          

                elif action == 2:

                    
                    if idx%5 == 0:
                    
                        t = self.sess.run([self.train_g],
                                                 feed_dict={self.input: batch_images,
                                                            self.image_target: batch_labels,
                                                            self.dropout: 1.,
                                                            self.lr:learning_rate})
                       
                    
                    _, loss = self.sess.run([self.train_d, self.d_loss],
                                             feed_dict={self.input: batch_images,
                                                        self.image_target: batch_labels,
                                                        self.dropout: 1.,
                                                        self.lr:learning_rate})
                    #self.sess.run(self.clip_discriminator_var_op)

                              
                #batch_pbar.refresh()
              
            if ep % 50 == 1:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                        feed_dict={
                                                                            self.input: batch_images, 
                                                                            self.image_target: batch_labels,
                                                                            self.dropout: 1.
                                                                           })
                #batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                
                test_sum = self.sess.run(self.merged_summary_test, 
                                                                        feed_dict={
                                                                            self.input: test_data, 
                                                                            self.image_target: test_label,
                                                                            self.dropout: 1.,
                                                                            })
                                                                                                                       
                
               
                    
                print("Epoch: [{}]".format((ep+1)))       
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)



    def build_edsr_lsgan_dis_large(self):###
        """
        Build SRCNN model
        """        
        # Define input and label images
        self.input = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='images')
        #self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.color_dim], name='images')
        self.image_target = tf.placeholder(tf.float32, [None, self.image_size*2, self.image_size*2, self.color_dim], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        
        """
        mean_x = tf.reduce_mean(self.input)
        image_input  = self.input - mean_x
        mean_y = tf.reduce_mean(self.image_target)
        target = self.image_target - mean_y
        """
        self.image_input = self.input/255.
        self.target = target = self.image_target/255.
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(self.image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        with tf.variable_scope("wgan") as scope:
            gen_f, dis_f = mz.build_model({"d_inputs":None,"d_target":self.target,"scale":self.scale,"feature_size" :64, "reuse":False, "is_training":True,"is_generate":True})
            scope.reuse_variables()
            _, dis_t = mz.build_model({"d_inputs":self.target,"d_target":self.target,"scale":self.scale,"feature_size" :64,"reuse":True, "is_training":True,"is_generate":False})

            scope.reuse_variables()

            #Calculate gradient panalty
            self.epsilon = epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * self.target + (1. - epsilon) * (gen_f)
            _, d_hat = mz.build_model({"d_inputs":x_hat,"d_target":self.target,"scale":self.scale,"feature_size" :64, "reuse":True, "is_training":True,"is_generate":False})
            
        d_gp = tf.gradients(d_hat, [x_hat])[0]
        d_gp = tf.sqrt(tf.reduce_sum(tf.square(d_gp), axis=[1,2,3]))
        d_gp = tf.reduce_mean(tf.square(d_gp - 1.0)) * 10

        self.disc_ture_loss = disc_ture_loss = tf.reduce_mean(dis_t)
        disc_fake_loss = tf.reduce_mean(dis_f)

        reconstucted_weight = 1.0  #StarGAN is 10
        self.d_loss =   disc_fake_loss - disc_ture_loss + d_gp
        self.g_l1loss = tf.reduce_mean(tf.losses.absolute_difference(target,gen_f))
        self.g_loss =  -1.0*disc_fake_loss + reconstucted_weight*self.g_l1loss
        
       

        train_variables = tf.trainable_variables()
        generator_variables = [v for v in train_variables if v.name.startswith("wgan/EDSR_gen")]
        discriminator_variables = [v for v in train_variables if v.name.startswith("wgan/EDSR_dis")]
        self.train_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=discriminator_variables)
        self.train_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=generator_variables)
        self.train_l1 = tf.train.AdamOptimizer(self.lr,beta1=0.5, beta2=0.999).minimize(self.g_l1loss, var_list=generator_variables)


        mse = tf.reduce_mean(tf.squared_difference(target*255.,gen_f*255.))    
        PSNR = tf.constant(255**2,dtype=tf.float32)/mse
        PSNR = tf.constant(10,dtype=tf.float32)*log10(PSNR)

        mse_ref = tf.reduce_mean(tf.squared_difference(target*255.,self.image_input*255.))    
        PSNR_ref = tf.constant(255**2,dtype=tf.float32)/mse_ref
        PSNR_ref = tf.constant(10,dtype=tf.float32)*log10(PSNR_ref)

        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("l1_loss", self.g_l1loss, collections=['train'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['train'])
            tf.summary.scalar("d_true_loss", disc_ture_loss, collections=['train'])
            tf.summary.scalar("d_fake_loss", disc_fake_loss, collections=['train'])
            tf.summary.scalar("grad_loss", d_gp, collections=['train'])
            tf.summary.scalar("dis_f_mean", tf.reduce_mean(dis_f), collections=['train'])
            tf.summary.scalar("dis_t_mean", tf.reduce_mean(dis_t), collections=['train'])
            tf.summary.scalar("MSE", mse, collections=['train'])
            tf.summary.scalar("PSNR",PSNR, collections=['train'])
            tf.summary.image("input_image",self.input , collections=['train'])
            tf.summary.image("target_image",target*255, collections=['train'])
            tf.summary.image("output_image",gen_f*255, collections=['train'])
            tf.summary.image("enhence_img",(2.0*gen_f-target)*255, collections=['train'])
            tf.summary.image("dis_f_img",dis_f*255, collections=['train'])
            tf.summary.image("dis_t_img",dis_t*255, collections=['train'])
            tf.summary.image("dis_diff",10*tf.abs(dis_t-dis_f)*255, collections=['train'])
            tf.summary.histogram("d_false", dis_f, collections=['train'])
            tf.summary.histogram("d_true", dis_t, collections=['train'])
    
            self.merged_summary_train = tf.summary.merge_all('train')          

        with tf.name_scope('test_summary'):

            tf.summary.scalar("loss", self.g_l1loss, collections=['test'])
            tf.summary.scalar("d_loss", self.d_loss, collections=['test'])
            tf.summary.scalar("g_loss", disc_fake_loss, collections=['test'])
            tf.summary.scalar("MSE", mse, collections=['test'])
            tf.summary.scalar("PSNR",PSNR, collections=['test'])
            tf.summary.scalar("PSNR_ref",PSNR_ref, collections=['test'])
            tf.summary.image("input_image",self.input , collections=['test'])
            tf.summary.image("target_image",target*255, collections=['test'])
            tf.summary.image("output_image",gen_f*255, collections=['test'])
        
            self.merged_summary_test = tf.summary.merge_all('test')                 
        
        self.saver = tf.train.Saver()

        
    def train_edsr_lsgan_dis_large(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path

        #gcloud96X96
        test_dataset = self.load_divk("//home/moxalab/data/SuperResolution/Set5/validation96_scale_"+"2"+"/",type="test")
        dataset = self.load_divk("/home/moxalab/data/SuperResolution/DIV2K_base/", lrtype='bicubic', type='train')
        

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        
        
        # Define iteration counter, timer and average loss
        itera_counter = 0
        learning_rate = 1e-4
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(1210,self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d], lr:%f" % ((ep+1), learning_rate))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            
            action = 0
            cycle_times = 100000
            current_cycle = ep%cycle_times

            if current_cycle < 500:
                action = 2
            elif current_cycle >= 1000 and current_cycle < 2000:
                action = 2
            elif current_cycle >= 2000:
                action = 2

            itr_per_epoch = len(train_data)//self.batch_size 
            if (ep*itr_per_epoch)%300000 == 0 and ep*itr_per_epoch != 0:learning_rate = learning_rate/2

            for idx in batch_pbar:                
                batch_pbar.set_description("Batch: [%2d], Action: [%d]" % ((idx+1) ,action))
                itera_counter += 1
                batch_index = idx*self.batch_size 
                #batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                #start_time = time.time()
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, 1, self.image_size*2,batch_index, self.batch_size)
                #elapse = time.time() - start_time
                #print("batch elapse:", elapse)
                # Select different action each cycle time

                #start_time = time.time()
                if action == 0:
                    
                    self.sess.run(self.train_l1,
                                             feed_dict={self.input: batch_images,
                                                        self.image_target: batch_labels,
                                                        self.dropout: 1.,
                                                        self.lr:learning_rate})

                elif action == 1:

                    
                    self.sess.run([self.train_d, self.d_loss],
                                                 feed_dict={self.input: batch_images,
                                                            self.image_target: batch_labels,
                                                            self.dropout: 1.,
                                                            self.lr:1e-4})
                    #self.sess.run(self.clip_discriminator_var_op)
                          

                elif action == 2:

                    
                    if idx%5 == 0:
                    
                        t = self.sess.run([self.train_g],
                                                 feed_dict={self.input: batch_images,
                                                            self.image_target: batch_labels,
                                                            self.dropout: 1.,
                                                            self.lr:learning_rate})
                       
                    
                    _, loss = self.sess.run([self.train_d, self.d_loss],
                                             feed_dict={self.input: batch_images,
                                                        self.image_target: batch_labels,
                                                        self.dropout: 1.,
                                                        self.lr:learning_rate})
                    #self.sess.run(self.clip_discriminator_var_op)

                #elapse = time.time() - start_time 
                #print("training elapse:", elapse)           
                #batch_pbar.refresh()
              
            if ep % 50 == 1:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                train_sum = self.sess.run(self.merged_summary_train, 
                                                                        feed_dict={
                                                                            self.input: batch_images, 
                                                                            self.image_target: batch_labels,
                                                                            self.dropout: 1.
                                                                           })
                #batch_test_images, batch_test_labels = batch_shuffle_rndc(test_data, test_label, self.scale, self.image_size, 0, 5)
                
                test_sum = self.sess.run(self.merged_summary_test, 
                                                                        feed_dict={
                                                                            self.input: test_data, 
                                                                            self.image_target: test_label,
                                                                            self.dropout: 1.,
                                                                            })
                                                                                                                       
                
               
                    
                print("Epoch: [{}]".format((ep+1)))       
                
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)
