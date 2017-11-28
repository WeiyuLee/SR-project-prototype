import tensorflow as tf
from utility import netfactory as nf
import numpy as np

image_input = tf.placeholder(tf.float32, [None, 48, 48, 3], name='images')
att_net = nf.convolution_layer(image_input, [3,3,64], [1,1,1,1],name="conv1-1")
att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv1-2")
att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv2-1")
att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv3-1")
att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-1")
att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv5-1")
print(att_net)