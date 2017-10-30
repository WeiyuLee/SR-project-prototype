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

from utils import (
  read_data, 
  batch_shuffle_rndc,
  batch_shuffle,
  log10
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
        
        self.model_list = ["googleLeNet_v1", "resNet_v1", "srcnn_v1", "grr_srcnn_v1", "grr_grid_srcnn_v1", "espcn_v1", "edsr_v1"]
        
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
    
    def build_grr_grid_srcnn_v1(self):###
        """
        Build SRCNN model
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
        self.stg_pred, self.grid_pred, self.pred, idx_shuffled = mz.build_model()
           
        # Define loss function (MSE) 
        ## Stage 1 loss:
        self.stg1_loss = tf.reduce_mean(tf.square(self.stg1_labels - self.stg_pred[0]))
        ## Stage 2 loss:
        self.stg2_loss = tf.reduce_mean(tf.square(self.stg2_labels - self.stg_pred[1]))    
        ## Stage 3 loss:
        self.stg3_loss = tf.reduce_mean(tf.square(self.stg3_labels - self.stg_pred[2]))
    
        self.all_stg_loss = tf.add(tf.add(self.stg1_loss, self.stg2_loss), self.stg3_loss)

        stg3_label_size = self.stg3_labels.get_shape()
        grid_h = stg3_label_size[1] // 2
        grid_w = stg3_label_size[2] // 2

        grid_label = [None]*4
        grid_label[0] = self.stg3_labels[:, 0:grid_h,                                 0:grid_w,                                 :]
        grid_label[1] = self.stg3_labels[:, 0:grid_h,                                 grid_w:,                                  :]
        grid_label[2] = self.stg3_labels[:, grid_h:,                                  0:grid_w,                                 :]
        grid_label[3] = self.stg3_labels[:, grid_h:,                                  grid_w:,                                  :]
        self.grid_label_reorder = tf.gather(grid_label, idx_shuffled)

        ## Grid 1 loss:
        self.grid1_loss = tf.reduce_mean(tf.square(self.grid_label_reorder[0] - self.grid_pred[0]))
        ## Grid 2 loss:
        self.grid2_loss = tf.reduce_mean(tf.square(self.grid_label_reorder[1] - self.grid_pred[1]))
        ## Grid 3 loss:
        self.grid3_loss = tf.reduce_mean(tf.square(self.grid_label_reorder[2] - self.grid_pred[2]))
        ## Grid 4 loss:
        self.grid4_loss = tf.reduce_mean(tf.square(self.grid_label_reorder[3] - self.grid_pred[3]))        
        
        ## Final total loss
        self.final_loss = tf.add(tf.add(tf.add(self.grid1_loss, self.grid2_loss), self.grid3_loss), self.grid4_loss) / 4
        
        with tf.name_scope('train_summary'):
            tf.summary.scalar("Stg1 loss", self.stg1_loss, collections=['train'])
            tf.summary.scalar("Stg2 loss", self.stg2_loss, collections=['train'])
            tf.summary.scalar("Stg3 loss", self.stg3_loss, collections=['train'])
            tf.summary.scalar("Grid1 loss", self.grid1_loss, collections=['train'])
            tf.summary.scalar("Grid2 loss", self.grid2_loss, collections=['train'])
            tf.summary.scalar("Grid3 loss", self.grid3_loss, collections=['train'])
            tf.summary.scalar("Grid4 loss", self.grid4_loss, collections=['train'])
            self.merged_summary_train = tf.summary.merge_all('train')          
            
        with tf.name_scope('test_summary'):
            tf.summary.scalar("Stg1 loss", self.stg1_loss, collections=['test'])
            tf.summary.scalar("Stg2 loss", self.stg2_loss, collections=['test'])
            tf.summary.scalar("Stg3 loss", self.stg3_loss, collections=['test'])
            tf.summary.scalar("Grid1 loss", self.grid1_loss, collections=['test'])
            tf.summary.scalar("Grid2 loss", self.grid2_loss, collections=['test'])
            tf.summary.scalar("Grid3 loss", self.grid3_loss, collections=['test'])
            tf.summary.scalar("Grid4 loss", self.grid4_loss, collections=['test'])
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
        train_data, train_label, freq_label = read_data(train_data_dir)
        test_data, test_label, freq_label = read_data(test_data_dir)
        
    
        # Get the trainable variables
        vars = tf.trainable_variables()

        grid1 = [v for v in vars if v.name.startswith("grid1_conv1")]            
        grid1 += [v for v in vars if v.name.startswith("grid1_conv2")]            
        grid1 += [v for v in vars if v.name.startswith("grid1_conv3")]            

        grid2 = [v for v in vars if v.name.startswith("grid2_conv1")]            
        grid2 += [v for v in vars if v.name.startswith("grid2_conv2")]            
        grid2 += [v for v in vars if v.name.startswith("grid2_conv3")]  
        
        grid3 = [v for v in vars if v.name.startswith("grid3_conv1")]            
        grid3 += [v for v in vars if v.name.startswith("grid3_conv2")]            
        grid3 += [v for v in vars if v.name.startswith("grid3_conv3")]  

        grid4 = [v for v in vars if v.name.startswith("grid4_conv1")]            
        grid4 += [v for v in vars if v.name.startswith("grid4_conv2")]            
        grid4 += [v for v in vars if v.name.startswith("grid4_conv3")]  
    
        # Stochastic gradient descent with the standard backpropagation       
        ## Stage loss 
        self.stg_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.all_stg_loss)
        self.stg1_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg1_loss)
        self.stg2_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg2_loss)
        self.stg3_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.stg3_loss)
        
        ## Grid loss
        self.grid1_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.grid1_loss, var_list=grid1)
        self.grid2_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.grid2_loss, var_list=grid2)
        self.grid3_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.grid3_loss, var_list=grid3)
        self.grid4_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.grid4_loss, var_list=grid4)

#        stg1_conv3 = tf.get_default_graph().get_tensor_by_name("stg1_conv3/weights/read:0")
#        stg2_conv3 = tf.get_default_graph().get_tensor_by_name("stg2_conv3/weights/read:0")
#        stg3_conv3 = tf.get_default_graph().get_tensor_by_name("stg3_conv3/weights/read:0")

        summary_writer = tf.summary.FileWriter('log', self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        # Define iteration counter, timer and average loss
        itera_counter = 0
        avg_500_loss = [0]*(stage_size+1)    # 3 stage + 1 total loss
        avg_gird_loss = [0]*4
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
                train_sum, _, stg1_err, stg2_err, stg3_err, total_stg_err,\
                grid1_err, grid2_err, grid3_err, grid4_err, total_err = self.sess.run([   self.merged_summary_train,
                                                                                          self.stg_train_op,
                                                                                          #self.stg3_train_op, 
                                                                                          #self.grid1_train_op,
                                                                                          #self.grid2_train_op,
                                                                                          #self.grid3_train_op,
                                                                                          #self.grid4_train_op,
                                                                                          self.stg1_loss, 
                                                                                          self.stg2_loss, 
                                                                                          self.stg3_loss,
                                                                                          self.all_stg_loss,
                                                                                          self.grid1_loss,
                                                                                          self.grid2_loss,
                                                                                          self.grid3_loss,
                                                                                          self.grid4_loss,
                                                                                          self.final_loss], 
                                                                                          feed_dict={
                                                                                                      self.images: batch_images, 
                                                                                                      self.stg1_labels: batch_labels[0],
                                                                                                      self.stg2_labels: batch_labels[1],
                                                                                                      self.stg3_labels: batch_labels[2],
                                                                                                      self.dropout: 1.
                                                                                                    })   

                avg_500_loss[0] += stg1_err
                avg_500_loss[1] += stg2_err
                avg_500_loss[2] += stg3_err
                avg_500_loss[3] += total_stg_err
                
                avg_gird_loss[0] += grid1_err
                avg_gird_loss[1] += grid2_err
                avg_gird_loss[2] += grid3_err
                avg_gird_loss[3] += grid4_err
                
                avg_final_loss += ((grid1_err+grid2_err+grid3_err+grid4_err)/4)
    
                batch_pbar.set_description("Batch: [%2d]" % (idx+1))
                #batch_pbar.refresh()
            
            if ep % 5 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                # Validation
                ## Run the test images
                test_sum, val_stg1_err, val_stg2_err, val_stg3_err, val_total_stg_err,\
                val_grid1_err, val_grid2_err, val_grid3_err, val_grid4_err, val_total_err = self.sess.run([  self.merged_summary_test,
                                                                                                             self.stg1_loss,
                                                                                                             self.stg2_loss,
                                                                                                             self.stg3_loss,
                                                                                                             self.all_stg_loss,
                                                                                                             self.grid1_loss,
                                                                                                             self.grid2_loss,
                                                                                                             self.grid3_loss,
                                                                                                             self.grid4_loss,
                                                                                                             self.final_loss], 
                                                                                                             feed_dict={
                                                                                                                         self.images: val_images, 
                                                                                                                         self.stg1_labels: val_label[0],
                                                                                                                         self.stg2_labels: val_label[1],
                                                                                                                         self.stg3_labels: val_label[2],
                                                                                                                         self.dropout: 1.
                                                                                                                       })
 
                for i in range(len(avg_500_loss)):
                    avg_500_loss[i] /= (train_batch_num*5)
                    avg_gird_loss[i] /= (train_batch_num*5)
                avg_final_loss /= (train_batch_num*5)
                    
                print("Epoch: [%2d], Average train loss of 5 epoches: stg loss: [%.8f, %.8f, %.8f], total stg loss: [%.8f]" \
                     % ((ep+1), avg_500_loss[0], avg_500_loss[1], avg_500_loss[2], avg_500_loss[3]))     
                print("Epoch: [%2d], Average train loss of 5 epoches: grid loss: [%.8f, %.8f, %.8f, %.8f], total loss: [%.8f]" \
                     % ((ep+1), avg_gird_loss[0], avg_gird_loss[1], avg_gird_loss[2], avg_gird_loss[3], avg_final_loss))     
                print("Epoch: [%2d], Test stg loss: [%.8f, %.8f, %.8f], total stg loss: [%.8f]" \
                      % ((ep+1), val_stg1_err, val_stg2_err, val_stg3_err, val_total_stg_err))       
                print("Epoch: [%2d], Test grid loss: [%.8f, %.8f, %.8f, %.8f], total loss: [%.8f]\n" \
                      % ((ep+1), val_grid1_err, val_grid2_err, val_grid3_err, val_grid4_err, val_total_err))       
                
                #print(stg1_conv3_weight[0][0])
                #print(stg2_conv3_weight[0][0])
                #print(stg3_conv3_weight[0][0])
                
                summary_writer.add_summary(train_sum, ep)
                summary_writer.add_summary(test_sum, ep)
                
                avg_500_loss = [0]*(stage_size+1)
                avg_gird_loss = [0]*4
                avg_final_loss = 0     

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
        
    def train_espcn_v1(self):
        """
        Training process.
        """     
        print("Training...")

        # Define dataset path
        self.train_h5_name = self.train_h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.label_size)
        self.test_h5_name = self.test_h5_name + "_[{}]_scale_{}_size_{}.h5".format(self.mode, self.scale, self.label_size)
        
        train_data_dir = os.path.join(self.h5_dir,self.train_h5_name)
        test_data_dir = os.path.join(self.h5_dir,self.test_h5_name)
        
        # Read data from .h5 file
        train_data, train_label, freq_label = read_data(train_data_dir)
        test_data, test_label, freq_label = read_data(test_data_dir)

        log_dir = os.path.join(self.log_dir, self.ckpt_name, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)    
    
        self.sess.run(tf.global_variables_initializer())

        if self.load_ckpt(self.checkpoint_dir, self.ckpt_name):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        train_data = self.sess.run(tf.image.resize_images(train_data, [self.image_size, self.image_size]))
        test_data = self.sess.run(tf.image.resize_images(test_data, [self.image_size, self.image_size]))

        # Define iteration counter, timer and average loss
        itera_counter = 0
        train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            train_data, train_label = batch_shuffle(train_data, train_label, self.batch_size)
            test_data, test_label = batch_shuffle(test_data, test_label, self.batch_size)

            epoch_pbar.set_description("Epoch: [%2d]" % ((ep+1)))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, train_batch_num), desc="Batch: [0]")
            for idx in batch_pbar:                
                itera_counter += 1
                  
                # Get the training data
                batch_images = train_data[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_labels = train_label[idx*self.batch_size : (idx+1)*self.batch_size]     
                batch_test_images =  test_data[:self.batch_size] 
                batch_test_labels = test_label[:self.batch_size]
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 1.})
                                                           
    
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
            
            if ep % 100 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                _,  train_sum, train_loss = self.sess.run([self.train_op, self.merged_summary_train, self.l1_loss], 
                                                                                                    feed_dict={
                                                                                                        self.input: batch_images, 
                                                                                                        self.image_target: batch_labels,
                                                                                                        self.dropout: 1.
                                                                                                                       })
                test_sum, test_loss = self.sess.run([self.merged_summary_test, self.l1_loss], 
                                                                                                feed_dict={
                                                                                                    self.input: batch_test_images, 
                                                                                                    self.image_target: batch_test_labels,
                                                                                                    self.dropout: 1.})
                                                                                                                       

               
                    
                print("Epoch: [{}], Train_loss: {}".format((ep+1), train_loss))     
                print("Epoch: [{}], Test_loss: {}".format((ep+1), test_loss))  
                
                
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
        
        """
        mean_x = tf.reduce_mean(self.input)
        image_input  =self.input - mean_x
        mean_y = tf.reduce_mean(self.image_target)
        target = self.image_target - mean_y
        """
        #image_input = self.input
        #target = self.image_target
        
        # Initial model_zoo
        mz = model_zoo.model_zoo(image_input, self.dropout, self.is_train, self.model_ticket)
        
        # Build model
        self.logits = mz.build_model(scale=self.scale,feature_size = 256)
        self.l1_loss = tf.reduce_mean(tf.losses.absolute_difference(target,self.logits))
        mse = tf.reduce_mean(tf.square(target - self.logits))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l1_loss)
        
        
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


        test_dataset = self.load_divk("/home/ubuntu/dataset/SuperResolution/Set5/image_SRF_4", test_mean,type="test")
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
        #train_batch_num = len(train_data) // self.batch_size

        epoch_pbar = tqdm(range(self.epoch))
        for ep in epoch_pbar:            
            # Run by batch images
            random.shuffle(dataset) 
            train_data, train_label  = zip(*dataset)
            test_data, test_label  = zip(*test_dataset)

            epoch_pbar.set_description("Epoch: [%2d]" % ((ep+1)))
            epoch_pbar.refresh()
        
            batch_pbar = tqdm(range(0, len(train_data)//self.batch_size), desc="Batch: [0]")
            

            for idx in batch_pbar:                
                        
                itera_counter += 1
                batch_index = idx*self.batch_size 
                batch_images, batch_labels = batch_shuffle_rndc(train_data, train_label, self.scale, self.image_size,batch_index, self.batch_size)
                  
                
                # Run the model
                _, train_loss = self.sess.run([self.train_op, self.l1_loss],
                                                                             feed_dict={self.input: batch_images, 
                                                                                        self.image_target: batch_labels,
                                                                                        self.dropout: 1.})
                                                           
    
                batch_pbar.set_description("Batch: [%2d], L1:%.2f" % (idx+1, train_loss))
                #batch_pbar.refresh()
              
            if ep % 5 == 0:
                self.save_ckpt(self.checkpoint_dir, self.ckpt_name, itera_counter)
                
                _,  train_sum, train_loss = self.sess.run([self.train_op, self.merged_summary_train, self.l1_loss], 
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


    def load_divk(self, dataset_path, mean, lrtype='bicubic', type='train'):

        #dataset_path = "/home/ubuntu/dataset/SuperResolution/DIV2K/"

        if type == "train":
            sub_path = "DIV2K_train"


            if lrtype == 'bicubic':
                lr_subpath = sub_path + "_LR_bicubic/" + "X" + str(self.scale)
            else:
                lr_subpath = sub_path + "_LR_unkown/" + "X" + str(self.scale)
            LR_path = os.path.join(dataset_path, lr_subpath)
            HR_path = os.path.join(dataset_path, sub_path + "_HR")

            hr_imgs = os.listdir(HR_path)
            lr_imgs = [os.path.join(LR_path,hr_imgs[i].split(".")[0] + 'x' + str(self.scale)+'.' + hr_imgs[i].split(".")[1]) for i in range(len(hr_imgs))]
            hr_imgs = [os.path.join(HR_path, hr_imgs[i]) for i in range(len(hr_imgs))]
        

        if type == "test":
            lr_imgs = []
            hr_imgs = []
            images = os.listdir(dataset_path)
            for i in range(len(images)//2):
                lr_imgs.append(os.path.join(dataset_path, "img_00"+str(i+1)+"_SRF_" + str(self.scale)+"_LR.png"))
                hr_imgs.append(os.path.join(dataset_path, "img_00"+str(i+1)+"_SRF_"+ str(self.scale)+"_HR.png"))

        hr_list = []
        lr_list = []

        for i in range(len(lr_imgs)):
            print("read:", i)
            
             
            hr_list.append(7))
            lr_list.append(misc.imread(lr_imgs[i]))
            
            
        return list(zip(lr_list, hr_list))
