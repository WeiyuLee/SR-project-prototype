# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:47:06 2017

@author: Weiyu_Lee
"""
from prprc_class import PRPRC

import tensorflow as tf

import pprint
import os

flags = tf.app.flags

flags.DEFINE_string("mode", "small", "operation mode: normal or freq [normal]")

flags.DEFINE_integer("scale", 4, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("image_size", 64, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 64, "The size of label to produce [21]")
flags.DEFINE_integer("color_dim", 1, "Dimension of image color. [1]")

flags.DEFINE_integer("train_extract_stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_integer("test_extract_stride", flags.FLAGS.label_size, "The size of stride to apply input image [14]")

flags.DEFINE_string("train_dir", "Train", "Name of train dataset directory")
flags.DEFINE_string("test_dir", "Test/Set5", "Name of test dataset directory [Test/Set5]")
flags.DEFINE_string("output_dir", "output", "Name of sample directory [output]")

flags.DEFINE_string("train_h5_name", "train", "Name of train dataset .h5 file")
flags.DEFINE_string("test_h5_name", "test", "Name of test dataset .h5 file")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    prprc = PRPRC(mode=FLAGS.mode, 
                  scale=FLAGS.scale, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  color_dim=FLAGS.color_dim, 
                  train_extract_stride=FLAGS.train_extract_stride,
                  test_extract_stride=FLAGS.test_extract_stride,
                  train_dir=FLAGS.train_dir,
                  test_dir=FLAGS.test_dir,
                  output_dir=FLAGS.output_dir,
                  train_h5_name=FLAGS.train_h5_name,
                  test_h5_name=FLAGS.test_h5_name)

    prprc.prepare_data()
   
if __name__ == '__main__':
  tf.app.run()