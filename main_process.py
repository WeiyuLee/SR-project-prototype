# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:18:05 2017

@author: Weiyu_Lee
"""

from model import MODEL

import tensorflow as tf

import pprint
import os

flags = tf.app.flags

flags.DEFINE_string("mode", "normal", "operation mode: normal or freq [normal]")

flags.DEFINE_integer("epoch", 10, "Number of epoch [10]")

flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 32, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 20, "The size of label to produce [21]")

flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("color_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 4, "The size of scale factor for preprocessing input image [3]")

flags.DEFINE_integer("train_extract_stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_integer("test_extract_stride", flags.FLAGS.label_size, "The size of stride to apply input image [14]")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("output_dir", "output", "Name of sample directory [output]")
flags.DEFINE_string("train_dir", "Train", "Name of train dataset directory")
flags.DEFINE_string("test_dir", "Test/Set5", "Name of test dataset directory [Test/Set5]")

flags.DEFINE_string("h5_dir", "preprocess/output", "Name of train dataset .h5 file")
flags.DEFINE_string("train_h5_name", "train", "Name of train dataset .h5 file")
flags.DEFINE_string("test_h5_name", "test", "Name of test dataset .h5 file")

flags.DEFINE_string("ckpt_name", "", "Name of checkpoints")

flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")

flags.DEFINE_string("model_ticket", "grr_grid_srcnn_v1", "Name of checkpoints")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    with tf.Session() as sess:
        srcnn = MODEL(sess, 
                      mode=FLAGS.mode,
                      epoch=FLAGS.epoch,
                      batch_size=FLAGS.batch_size,
                      image_size=FLAGS.image_size, 
                      label_size=FLAGS.label_size, 
                      learning_rate=FLAGS.learning_rate,
                      color_dim=FLAGS.color_dim, 
                      scale=FLAGS.scale,
                      train_extract_stride=FLAGS.train_extract_stride,
                      test_extract_stride=FLAGS.test_extract_stride,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      output_dir=FLAGS.output_dir,
                      train_dir=FLAGS.train_dir,
                      test_dir=FLAGS.test_dir,
                      h5_dir=FLAGS.h5_dir,
                      train_h5_name=FLAGS.train_h5_name,
                      test_h5_name=FLAGS.test_h5_name,
                      ckpt_name=FLAGS.ckpt_name,
                      is_train=FLAGS.is_train,
                      model_ticket=FLAGS.model_ticket)

        if FLAGS.is_train:
            srcnn.train()
    
if __name__ == '__main__':
  tf.app.run()