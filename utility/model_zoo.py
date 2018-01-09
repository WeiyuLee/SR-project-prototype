import tensorflow as tf
import netfactory as nf
import numpy as np

class model_zoo:
    
    def __init__(self, inputs, dropout, is_training, model_ticket):
        
        self.model_ticket = model_ticket
        self.inputs = inputs
        self.dropout = dropout
        self.is_training = is_training
        
    
        
    def googleLeNet_v1(self):
        
        model_params = {
        
            "conv1": [5,5, 64],
            "conv2": [3,3,128],
            "inception_1":{                 
                    "1x1":64,
                    "3x3":{ "1x1":96,
                            "3x3":128
                            },
                    "5x5":{ "1x1":16,
                            "5x5":32
                            },
                    "s1x1":32
                    },
            "inception_2":{                 
                    "1x1":128,
                    "3x3":{ "1x1":128,
                            "3x3":192
                            },
                    "5x5":{ "1x1":32,
                            "5x5":96
                            },
                    "s1x1":64
                    },
            "fc3": 10,
                     
        }
                
        
        with tf.name_scope("googleLeNet_v1"):
            net = nf.convolution_layer(self.inputs, model_params["conv1"], [1,2,2,1],name="conv1")
            net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.local_response_normalization(net, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='LocalResponseNormalization')
            net = nf.convolution_layer(net, model_params["conv2"], [1,1,1,1],name="conv2", flatten=False)
            net = tf.nn.local_response_normalization(net, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='LocalResponseNormalization')
            net = nf.inception_v1(net, model_params, name= "inception_1", flatten=False)
            net = nf.inception_v1(net, model_params, name= "inception_2", flatten=False)
            net = tf.nn.avg_pool (net, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding='VALID')
            net = tf.reshape(net, [-1, int(np.prod(net.get_shape()[1:]))])
            
            net = tf.layers.dropout(net, rate=self.dropout, training=self.is_training, name='dropout2')
            logits = nf.fc_layer(net, model_params["fc3"], name="logits", activat_fn=None)

            
        return logits
    
    
    def resNet_v1(self):
        
        model_params = {
        
            "conv1": [5,5, 64],
            "rb1_1": [3,3,64],
            "rb1_2": [3,3,64],
            "rb2_1": [3,3,128],
            "rb2_2": [3,3,128],
            "fc3": 10,
                     
        }
                
        
        with tf.name_scope("resNet_v1"):
            net = nf.convolution_layer(self.inputs, model_params["conv1"], [1,2,2,1],name="conv1")
            id_rb1 = tf.nn.max_pool(net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
            
            net = nf.convolution_layer(id_rb1, model_params["rb1_1"], [1,1,1,1],name="rb1_1")
            id_rb2 = nf.convolution_layer(net, model_params["rb1_2"], [1,1,1,1],name="rb1_2")
            
            id_rb2 = nf.shortcut(id_rb2,id_rb1, name="rb1")
            
            net = nf.convolution_layer(id_rb2, model_params["rb2_1"], [1,2,2,1],padding="SAME",name="rb2_1")
            id_rb3 = nf.convolution_layer(net, model_params["rb2_2"], [1,1,1,1],name="rb2_2")
            
            id_rb3 = nf.shortcut(id_rb3,id_rb2, name="rb2")
            
            net  = nf.global_avg_pooling(id_rb3, flatten=True)
            
            net = tf.layers.dropout(net, rate=self.dropout, training=self.is_training, name='dropout2')
            logits = nf.fc_layer(net, model_params["fc3"], name="logits", activat_fn=None)

            
        return logits
    
    
    def srcnn_v1(self):

        model_params = {
        
            "conv1": [9, 9, 64],
            "conv2": [1, 1, 32],
            "conv3": [5, 5, 1],                       

        }                
        
        with tf.name_scope("srcnn_v1"):           
 
            init = tf.random_normal_initializer(mean=0, stddev=1e-3)
            
            layer1_output = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", padding="VALID", initializer=init)
            
            layer2_output = nf.convolution_layer(layer1_output, model_params["conv2"], [1,1,1,1], name="conv2", padding="VALID", initializer=init)

            layer3_output = nf.convolution_layer(layer2_output, model_params["conv3"], [1,1,1,1], name="conv3", padding="VALID", initializer=init, activat_fn=None)

            
        return layer3_output

   
    def grr_srcnn_v1(self):

        model_params = {    
            
            # Stage 1
            "stg1_conv1": [9, 9, 64],
            "stg1_conv2": [1, 1, 32],
            "stg1_conv3": [5, 5, 1],                       

            # Stage 2
            "stg2_conv1": [9, 9, 64],
            "stg2_conv2": [1, 1, 32],
            "stg2_conv3": [5, 5, 1], 
            
            # Stage 3
            "stg3_conv1": [9, 9, 64],
            "stg3_conv2": [1, 1, 32],
            "stg3_conv3": [5, 5, 1],              
        }                
        
        with tf.name_scope("grr_srcnn_v1"):           
 
            init = tf.random_normal_initializer(mean=0, stddev=1e-3)
            padding = 6
            #print(self.inputs.shape)

            # Stage 1            
            stg1_layer1_output = nf.convolution_layer(self.inputs,        model_params["stg1_conv1"], [1,1,1,1], name="stg1_conv1", padding="SAME", initializer=init)           
            stg1_layer2_output = nf.convolution_layer(stg1_layer1_output, model_params["stg1_conv2"], [1,1,1,1], name="stg1_conv2", padding="SAME", initializer=init)
            stg1_layer3_output = nf.convolution_layer(stg1_layer2_output, model_params["stg1_conv3"], [1,1,1,1], name="stg1_conv3", padding="SAME", initializer=init, activat_fn=None)
            
            stg1_layer3_output = tf.add(self.inputs, stg1_layer3_output) ## multi-stg
            
            # Stage 2            
            stg2_layer1_output = nf.convolution_layer(stg1_layer3_output, model_params["stg2_conv1"], [1,1,1,1], name="stg2_conv1", padding="SAME", initializer=init)           
            stg2_layer2_output = nf.convolution_layer(stg2_layer1_output, model_params["stg2_conv2"], [1,1,1,1], name="stg2_conv2", padding="SAME", initializer=init)
            stg2_layer3_output = nf.convolution_layer(stg2_layer2_output, model_params["stg2_conv3"], [1,1,1,1], name="stg2_conv3", padding="SAME", initializer=init, activat_fn=None)
            
            stg2_layer3_output = tf.add(stg1_layer3_output, stg2_layer3_output) ## multi-stg_3
            
            # Stage 3
            stg3_layer1_output = nf.convolution_layer(stg2_layer3_output, model_params["stg3_conv1"], [1,1,1,1], name="stg3_conv1", padding="VALID", initializer=init)           
            stg3_layer2_output = nf.convolution_layer(stg3_layer1_output, model_params["stg3_conv2"], [1,1,1,1], name="stg3_conv2", padding="VALID", initializer=init)
            stg3_layer3_output = nf.convolution_layer(stg3_layer2_output, model_params["stg3_conv3"], [1,1,1,1], name="stg3_conv3", padding="VALID", initializer=init, activat_fn=None)

            stg3_layer3_output = tf.add(stg2_layer3_output[:,padding:-padding,padding:-padding,:], stg3_layer3_output) ## multi-stg_3
           
        return stg1_layer3_output, stg2_layer3_output, stg3_layer3_output    
    
    def grr_grid_srcnn_v1(self):

        model_params = {    
            
            # Stage 1
            "stg1_conv1": [9, 9, 64],
            "stg1_conv2": [1, 1, 32],
            "stg1_conv3": [5, 5, 1],                       

            # Stage 2
            "stg2_conv1": [9, 9, 64],
            "stg2_conv2": [1, 1, 32],
            "stg2_conv3": [5, 5, 1], 
            
            # Stage 3
            "stg3_conv1": [9, 9, 64],
            "stg3_conv2": [1, 1, 32],
            "stg3_conv3": [5, 5, 1],              
            
            # HF Stage 
            "HF_conv1": [9, 9, 64],
            "HF_conv2": [1, 1, 32],
            "HF_conv3": [5, 5, 1],   

            # LF Stage 
            "LF_conv1": [9, 9, 64],
            "LF_conv2": [1, 1, 32],
            "LF_conv3": [5, 5, 1],                 
        }                
        
        with tf.name_scope("grr_grid_srcnn_v1"):           
 
            init = tf.random_normal_initializer(mean=0, stddev=1e-3)
            padding = 6

            # Stage 1                        
            stg1_input = self.inputs
            
            stg1_layer1_output = nf.convolution_layer(stg1_input,         model_params["stg1_conv1"], [1,1,1,1], name="stg1_conv1", padding="SAME", initializer=init)           
            stg1_layer2_output = nf.convolution_layer(stg1_layer1_output, model_params["stg1_conv2"], [1,1,1,1], name="stg1_conv2", padding="SAME", initializer=init)
            stg1_layer3_output = nf.convolution_layer(stg1_layer2_output, model_params["stg1_conv3"], [1,1,1,1], name="stg1_conv3", padding="SAME", initializer=init, activat_fn=None)

            stg1_layer3_output = tf.add(stg1_input, stg1_layer3_output) ## multi-stg_1
                       
            #stg1_output = tf.stop_gradient(stg1_layer3_output)
            stg1_output = stg1_layer3_output

            # Stage 2            
            stg2_input = stg1_output
            
            stg2_layer1_output = nf.convolution_layer(stg2_input,         model_params["stg2_conv1"], [1,1,1,1], name="stg2_conv1", padding="SAME", initializer=init)           
            stg2_layer2_output = nf.convolution_layer(stg2_layer1_output, model_params["stg2_conv2"], [1,1,1,1], name="stg2_conv2", padding="SAME", initializer=init)
            stg2_layer3_output = nf.convolution_layer(stg2_layer2_output, model_params["stg2_conv3"], [1,1,1,1], name="stg2_conv3", padding="SAME", initializer=init, activat_fn=None)
            
            stg2_layer3_output = tf.add(stg1_output, stg2_layer3_output) ## multi-stg_2
            
            #stg2_output = tf.stop_gradient(stg2_layer3_output)
            stg2_output = stg2_layer3_output

            # Stage 3
            stg3_input = stg2_output
            
            stg3_layer1_output = nf.convolution_layer(stg3_input,         model_params["stg3_conv1"], [1,1,1,1], name="stg3_conv1", padding="VALID", initializer=init)           
            stg3_layer2_output = nf.convolution_layer(stg3_layer1_output, model_params["stg3_conv2"], [1,1,1,1], name="stg3_conv2", padding="VALID", initializer=init)
            stg3_layer3_output = nf.convolution_layer(stg3_layer2_output, model_params["stg3_conv3"], [1,1,1,1], name="stg3_conv3", padding="VALID", initializer=init, activat_fn=None)

            stg3_layer3_output = tf.add(stg2_output[:, padding:-padding, padding:-padding,:], stg3_layer3_output) ## multi-stg_3
            
            stg3_output = tf.stop_gradient(stg3_layer3_output)
            #stg3_output = stg3_layer3_output
            
            HF_thrld = tf.constant(13, dtype=tf.float32)
            TV_stg3_output = tf.image.total_variation(stg3_output)
            
            HF_cond = tf.greater_equal(TV_stg3_output, HF_thrld)
            LF_cond = tf.less(TV_stg3_output, HF_thrld)
            
            HF_indices = tf.where(HF_cond)
            LF_indices = tf.where(LF_cond)
            
            # HF Stage
            HF_input = tf.squeeze(tf.gather(stg3_output, HF_indices), 1)
            #HF_input = tf.pad(HF_input, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")
            HF_input = tf.pad(HF_input, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
            
            HF_layer1_output = nf.convolution_layer(HF_input,         model_params["HF_conv1"], [1,1,1,1], name="HF_conv1", padding="VALID", initializer=init)           
            HF_layer2_output = nf.convolution_layer(HF_layer1_output, model_params["HF_conv2"], [1,1,1,1], name="HF_conv2", padding="VALID", initializer=init)
            HF_layer3_output = nf.convolution_layer(HF_layer2_output, model_params["HF_conv3"], [1,1,1,1], name="HF_conv3", padding="VALID", initializer=init, activat_fn=None)            
            
            # Get LF
            LF_output = tf.squeeze(tf.gather(stg3_output, LF_indices), 1)
            
            # Get Final output
            final_output = tf.concat([LF_output, HF_layer3_output], 0)
            
            print(final_output.shape)
            
        return (stg1_layer3_output, stg2_layer3_output, stg3_layer3_output), (HF_layer3_output, LF_output), (HF_indices, LF_indices), TV_stg3_output
#               (HF_input, LF_input)
    
    def edsr_v1(self, kwargs):

        scale = kwargs["scale"]
        feature_size = kwargs["feature_size"]
        scaling_factor = 1
        num_resblock = 16
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3]
                        }
        with tf.name_scope("EDSR_v1"):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_1 = x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2")
                    x += conv_1

            with tf.name_scope("upsample"):
                    x = nf.upsample(x, scale, feature_size, True,None)
                    network = nf.convolution_layer(x, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)
        """
        with tf.name_scope("EDSR_v1"):       

            
            net = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")
            
            
            shortcut1 = net

            with tf.name_scope("resblock"): 
                resblock = []
                [resblock.append(model_params['resblock']) for i in range(num_resblock)]
                net = nf.edsr_resblock(net, resblock, repeations = num_resblock, scale = scaling_factor, name="resblock")
            
            net = nf.convolution_layer(net, model_params["conv2"], [1,1,1,1], name="conv2")
            
            net = net + shortcut1
            
            with tf.name_scope("upsample"):
                netowrk = nf.upsample(net, scale, feature_size, True,None)
        """
        return network

    
    def edsr_v2(self, kwargs):

        scale = kwargs["scale"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 16
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3]
                        }
        with tf.name_scope("EDSR_v1"):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_1 = x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2")
                    x += conv_1

            with tf.name_scope("upsamplex2"):
                    upsample2 = nf.upsample(x, 2, feature_size, 3,None)
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)
           
            with tf.name_scope("upsamplex4"):
                    upsample4 = nf.upsample(upsample2, 2, feature_size, 3,None)
                    network2 = nf.convolution_layer(upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)
                    
            
        return [network, network2]


    def edsr_v2_dual(self, kwargs):

        scale = kwargs["scale"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 16
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3]
                        }

        with tf.name_scope("EDSR_1"):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_1 = x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2")
                    x += conv_1

            with tf.name_scope("upsamplex2"):
                    upsample2 = nf.upsample(x, 2, feature_size, 3,None)
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)

        with tf.name_scope("EDSR_2"):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_1 = x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2")
                    x += conv_1

            with tf.name_scope("upsamplex2"):
                    upsample2 = nf.upsample(x, 2, feature_size, 3,None)
                    network2 = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)
           
                  
        return [network, network2]


    def attention_network(self, image_input, scale, layers, channels ,dropout, is_training):

        with tf.variable_scope("attention"):
                    
                    att_net = nf.convolution_layer(image_input, [3,3,64], [1,2,2,1],name="conv1-1")
                    att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv1-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv2-1")
                    #att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv3-1")
                    #att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv4-1")
                    #att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = tf.reshape(att_net, [-1, int(np.prod(att_net.get_shape()[1:]))]) 
                    att_net = nf.fc_layer(att_net, 2048, name="fc1")
                    att_net = nf.fc_layer(att_net, 2048, name="fc2")
                    att_net = tf.layers.dropout(att_net, rate=dropout, training=is_training, name='dropout1')
                    logits = nf.fc_layer(att_net, channels*(scale**2)*layers, name="logits", activat_fn=None)
                    
                    bsize = tf.shape(logits)[0]
                    logits = tf.reshape(logits, (bsize,1,1,channels*(scale**2), layers))
                    weighting = tf.nn.softmax(logits)
                    
                    """
                    max_index = tf.argmax(tf.nn.softmax(logits),4) 
                    weighting = tf.one_hot(max_index, 
                                        depth=layers, 
                                        on_value=1.0,
                                        axis = -1)
                    """
                  

        return weighting

    def edsr_1X1_v1(self, kwargs):

        scale = kwargs["scale"]
        feature_size = kwargs["feature_size"]
        dropout = kwargs["dropout"]
        is_training = kwargs["is_training"]

        scaling_factor = 1
        num_resblock = 16
        att_layers = 15
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'conv4': [3,3,3*att_layers],
                        'conv5': [3,3,3*att_layers]
                        }

        with tf.name_scope("attention_x2"):
                att_weight_x2 = self.attention_network(self.inputs, 1, att_layers,3, dropout, is_training)
        with tf.name_scope("attention_x4"):
                att_weight_x4 = self.attention_network(self.inputs, 1, att_layers,3, dropout, is_training)

        with tf.name_scope("EDSR_v1"):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_1 = x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2")
                    x += conv_1

            with tf.name_scope("upsamplex2"):
                    upsample2 = nf.upsample(x, 2, feature_size, 3,None)
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3_1", activat_fn=None)
                    network = nf.convolution_layer(network, model_params["conv4"], [1,1,1,1], name="conv3_2", activat_fn=None)
                    network = nf.convolution_layer(network, model_params["conv5"], [1,1,1,1], name="conv3_3", activat_fn=None)

                    netshape = tf.shape(network)
                    network = tf.reshape(network, (netshape[0],netshape[1],netshape[2],3, att_layers))
                    network = tf.multiply(network, att_weight_x2)
                    network = tf.reduce_sum(network,4) 

            with tf.name_scope("upsamplex4"):
                    upsample4 = nf.upsample(upsample2, 2, feature_size, 3,None)
                    network2 = nf.convolution_layer(upsample4, model_params["conv3"], [1,1,1,1], name="conv4_1", activat_fn=None)
                    network2 = nf.convolution_layer(network2, model_params["conv4"], [1,1,1,1], name="conv4_2", activat_fn=None)
                    network2 = nf.convolution_layer(network2, model_params["conv5"], [1,1,1,1], name="conv4_3", activat_fn=None)
                    
                    netshape = tf.shape(network2)
                    network2 = tf.reshape(network2, (netshape[0],netshape[1],netshape[2],3, att_layers))
                    network2 = tf.multiply(network2, att_weight_x2)
                    network2 = tf.reduce_sum(network2,4) 

        return [network, network2]

    def edsr_attention_v1(self, kwargs):


        def attention_network(image_input, scale, layers, channels ,dropout, is_training):

            with tf.variable_scope("attention"):
                    
                    att_net = nf.convolution_layer(image_input, [3,3,64], [1,2,2,1],name="conv1-1")
                    att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv1-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-1")
                    att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-1")
                    att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-1")
                    #att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = tf.reshape(att_net, [-1, int(np.prod(att_net.get_shape()[1:]))]) 
                    att_net = nf.fc_layer(att_net, 2048, name="fc1")
                    att_net = nf.fc_layer(att_net, 2048, name="fc2")
                    att_net = tf.layers.dropout(att_net, rate=dropout, training=is_training, name='dropout1')
                    logits = nf.fc_layer(att_net, channels*(scale**2)*layers, name="logits", activat_fn=None)
                    
                    bsize = tf.shape(logits)[0]
                    logits = tf.reshape(logits, (bsize,1,1,channels*(scale**2), layers))
                    weighting = tf.nn.softmax(logits)
                    
                    """
                    max_index = tf.argmax(tf.nn.softmax(logits),4) 
                    weighting = tf.one_hot(max_index, 
                                        depth=layers, 
                                        on_value=1.0,
                                        axis = -1)
                    """
                  

            return weighting



        scale = kwargs["scale"]
        feature_size = kwargs["feature_size"]
        dropout = kwargs["dropout"]
        is_training = kwargs["is_training"]

        scaling_factor = 1
        num_resblock = 16
        att_layers = 15
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3]
                        }

        with tf.name_scope("attention_x2"):
                arr = att_weight_x2 = attention_network(self.inputs, scale, att_layers,3, dropout, is_training)
        with tf.name_scope("attention_x4"):
                att_weight_x4 = attention_network(self.inputs, scale, att_layers,3, dropout, is_training)

        with tf.name_scope("EDSR_v1"):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_1 = x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2")
                    x += conv_1
        
            with tf.name_scope("upsamplex2"):
                    upsample2 = nf.upsample_attention(x, att_weight_x2, 2, feature_size, channels = 3,  attentions = att_layers, activation=None)
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)

            with tf.name_scope("upsamplex4"):
                    upsample4 = nf.upsample_attention(upsample2, att_weight_x4, 2, feature_size, channels = 3,  attentions = att_layers, activation=None)
                    network2 = nf.convolution_layer(upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)
       
        return [network, network2, att_weight_x2]


    def espcn_v1(self):

    
        model_params = {

                        'conv1': [5,5,64],
                        'conv2': [3,3,32]
                        }

        with tf.name_scope("espcn_v1"):       

            
            net = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1") 
            net = nf.convolution_layer(net, model_params["conv2"], [1,1,1,1], name="conv2")
            
            with tf.name_scope("upsample"):
                netowrk = nf.upsample_ESPCN(net, 4, 0, False,None)

        return netowrk


    def edsr_local_att_v1(self, kwargs):


        scale = kwargs["scale"]
        feature_size = 256
        scaling_factor = 1
        num_resblock = 16
        att_layers = 15

        def local_attention(scope_name, inputs, scale, channels = 3,att_layers = att_layers):

            with tf.variable_scope(scope_name):
                        
            
                att_net = nf.convolution_layer(inputs, [3,3,64], [1,1,1,1],name="conv1-1")
                att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv1-2")
                #att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv2-1")
                att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv3-1")
                att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-1")
                #att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                att_net = nf.convolution_layer(att_net, [3,3,channels*(scale**2)*att_layers], [1,1,1,1],name="conv5-1",activat_fn=None)

                weighting = tf.reshape(att_net, (-1,24,1,24,1,channels*(scale**2), att_layers))
                weighting = tf.nn.softmax(weighting)
            
                    
            return weighting   
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [1,1,feature_size],
                        'conv2': [1,1,feature_size],
                        'conv3': [1,1,3]
                        }
        with tf.name_scope("EDSR_v1"):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_1 = x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2")
                    x += conv_1


            with tf.name_scope("upsamplex2"):
                    weight2 = local_attention("attention_2", self.inputs, 2)
                    network = nf.upsample_local_attention(x, weight2, attentions = att_layers, activation=None)
                    #network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)
                    
            with tf.name_scope("upsamplex4"):
                    weight4 = local_attention("attention_4", network, 2)
                    network2 = nf.upsample_local_attention(network, weight4, attentions = att_layers, activation=None)
                    #network2 = nf.convolution_layer(upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)
                   
           
        return [network, network2]



    def edsr_attention_v2(self, kwargs):


        scale = kwargs["scale"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 16
        att_layers = 30

        def local_attention(scope_name, inputs, scale, channels = 3,att_layers = att_layers):
            att_ksize = 8
            num_resblock = 3       
            bsize, a, b, c = inputs.get_shape().as_list()
            model_params = {

                        'conv1': [att_ksize,att_ksize,feature_size],
                        'resblock': [att_ksize,att_ksize,feature_size],
                        'conv2': [att_ksize,att_ksize,feature_size],
                        'conv3': [att_ksize,att_ksize,feature_size],
                        'att': [att_ksize,att_ksize,channels*(scale**2)*att_layers],
                        }

            with tf.variable_scope(scope_name):

                x = nf.convolution_layer(inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None)
                conv_1 = x
                with tf.variable_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None)
                    x += conv_1

                att_net = nf.convolution_layer(x, model_params['att'], [1,1,1,1],name="att", activat_fn=None)
                weighting = tf.reshape(att_net, (-1,a,b,channels*(scale**2), att_layers))
                weighting = tf.nn.softmax(weighting)

            return weighting
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3]
                        }
        with tf.name_scope("EDSR_att_v2"):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None)
            conv_1 = x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None)
                    x += conv_1


            with tf.name_scope("upsamplex2"):
                    weight2 = local_attention("attention_2", self.inputs, 2)
                    network = nf.upsample_attention(x, weight2, attentions = att_layers, activation=None)
                    #network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)
                    
            with tf.name_scope("upsamplex4"):
                    weight4 = local_attention("attention_4", network, 2)
                    network2 = nf.upsample_attention(network, weight4, attentions = att_layers, activation=None)
                    #network2 = nf.convolution_layer(upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)
                   
           
        return [network, network2]




    def edsr_local_att_v2_upsample(self, kwargs):


        scale = kwargs["scale"]
        kernel_size = kwargs["kernel_size"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 16
        att_layers = 15
        portion = 8

        def local_attention(scope_name, inputs, scale, channels = 3,att_layers = att_layers):
            att_ksize = kernel_size + 2 
            num_resblock = 8       
            bsize, a, b, c = inputs.get_shape().as_list()
            model_params = {

                        'conv1': [att_ksize,att_ksize,feature_size],
                        'resblock': [att_ksize,att_ksize,feature_size],
                        'conv2': [att_ksize,att_ksize,feature_size],
                        'conv3': [att_ksize,att_ksize,feature_size],
                        'att': [a//portion,b//portion,channels*(scale**2)*att_layers],
                        }

            with tf.variable_scope(scope_name):

                x = nf.convolution_layer(inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None)
                conv_1 = x
                with tf.variable_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None)
                    x += conv_1

                network = nf.convolution_layer(x, model_params['conv3'], [1,1,1,1],name="conv3", activat_fn=None)

                #weighting = tf.reshape(att_net, (-1,48,1,48,1,64, att_layers))
                
                h_split = tf.split(network, portion, 1)
                v_split = []

                for i in range(portion):
                    v_split.append(tf.split(h_split[i],portion,2))

                cv_split = []
                for i in range(portion):
                    tmp_conv = []
                    for j in range(portion):
                        name = "att_.format_{}_{}".format(i,j)
                        tmp =  nf.convolution_layer(v_split[i][j], model_params['att'], [1,1,1,1],name=name, padding='VALID', activat_fn=None)
                        tmp_conv.append(tmp)

                        
                    cv_split.append(tmp_conv)

                v_merge = []
                for i in range(portion):
                    cv_tmp = tf.concat(cv_split[i],2)
                    v_merge.append(cv_tmp)
                
                weighting = tf.concat(v_merge,1)
                weighting = tf.reshape(weighting, (-1,portion,1,portion,1,channels*(scale**2), att_layers))
                weighting = tf.nn.softmax(weighting)
                    
            return weighting   
            
        model_params = {

                        'conv1': [kernel_size,kernel_size,feature_size],
                        'resblock': [kernel_size,kernel_size,feature_size],
                        'conv2': [kernel_size,kernel_size,feature_size],
                        'conv3': [kernel_size,kernel_size,3]
                        }


        inputs = self.inputs


        with tf.variable_scope("EDSR_v1"):     
            x = nf.convolution_layer(inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None)
            conv_1 = x
            with tf.variable_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2",  activat_fn=None)
                    x += conv_1


            with tf.name_scope("upsamplex2"):
                    weight2 = local_attention("attention_2", self.inputs, 2)
                    network2 = nf.upsample_local_attention_v2(x, weight2,kernel_size, portion = portion,attentions = att_layers, activation=None)
                   
            """   
            with tf.name_scope("upsamplex4"):
                    weight4 = local_attention("attention_4", network2, 2)
                    network4 = nf.upsample_local_attention_v2(network2, weight4, attentions = att_layers, activation=None)
            """    
           
                    

        return network2, network2



    def edsr_lsgan(self, kwargs):

        ###Generator
        init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32)
        scale = kwargs["scale"]
        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 8
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3]
                        }
        
        with tf.variable_scope("EDSR_gen", reuse=reuse):     
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
            conv_1 = x
            with tf.variable_scope("resblock",reuse=reuse): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                    x += conv_1

            with tf.variable_scope("upsamplex2", reuse=reuse):
                    #upsample2 = nf.upsample(x, 2, feature_size, 3,None, initializer=init)
                    upsample2 = x
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None, initializer=init)
       
        ###Discriminator
        num_resblock = 0

        if is_training:

            if d_inputs == None: 

                #d_inputs = network  + 3.0*(network - d_target)
                #Test 
                res_gen = network - d_target
                input_gan = tf.concat([network,res_gen], axis=3)
                #d_inputs = network
            else:
                res_gen = d_inputs - d_target
                input_gan = tf.concat([d_inputs, res_gen], axis=3)
            
            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     
                x = nf.convolution_layer( input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, is_bn=True,initializer=init)
                conv_1 = x
                with tf.variable_scope("resblock", reuse=reuse): 
                
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, is_bn=True, reuse=reuse, idx = i, activation_fn=nf.lrelu, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2",activat_fn=nf.lrelu,  is_bn=True, initializer=init)
                        x += conv_1
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, is_bn=True, initializer=init)
                d_logits = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)

        else:
            d_logits = network
  
        return [network, d_logits]



    def edsr_lsgan_up(self, kwargs):

        ###Generator
        init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32)
        scale = kwargs["scale"]
        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        is_generate = kwargs["is_generate"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 4
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3]
                        }

        if is_generate:
            with tf.variable_scope("EDSR_gen", reuse=reuse):     
                x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
                conv_1 = x
                with tf.variable_scope("resblock",reuse=reuse): 
                
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                        x += conv_1

                with tf.variable_scope("upsamplex2", reuse=reuse):
                        upsample2 = nf.upsample(x, 2, feature_size, 3,None, initializer=init)
                        network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None, initializer=init)
        else:
            network = d_inputs
           
        ###Discriminator

        

        if is_training:

            if d_inputs == None: 
                #d_inputs = network  + 2.0*(network - d_target)
                d_inputs = network
            #input_gan = tf.concat([d_inputs, d_target], axis=3)
            input_gan = d_inputs

            with tf.variable_scope("EDSR_dis", reuse=reuse):     
                x = nf.convolution_layer( input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, is_bn=True,initializer=init)
                conv_1 = x
                with tf.variable_scope("resblock", reuse=reuse): 
                
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, is_bn=True, reuse=reuse, idx = i, activation_fn=nf.lrelu, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2",activat_fn=nf.lrelu,  is_bn=True, initializer=init)
                        x += conv_1
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, is_bn=True, initializer=init)
                d_logits = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)
               
            
                #Enable for GAN Loss
                #d_logits = nf.fc_layer(d_logits,2048,"fc1",activat_fn=nf.lrelu, initializer=init)
                #d_logits = nf.fc_layer(d_logits,1,"fc2",activat_fn=None, initializer=init)
                #d_logits = tf.sigmoid(d_logits)

            
        else:
            d_logits = network
  
        return [network, d_logits]


    def edsr_wgan_encode(self, kwargs):

        ###Generator
        init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32)
        scale = kwargs["scale"]
        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 4

        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,2*feature_size],
                        'conv2_2': [3,3,4*feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3]
                        }
            
        
        
        with tf.variable_scope("EDSR_gen", reuse=reuse):
            with tf.variable_scope("encoder", reuse=reuse):     
                x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1_1", activat_fn=None, initializer=init)
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv1_2", activat_fn=None, initializer=init)
                x = tf.nn.max_pool(x, ksize=[1,2,2,1],strides = [1,2,2,1], padding='VALID', name="pool1")
                x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=nf.lrelu, initializer=init)
                x = tf.nn.max_pool(x, ksize=[1,2,2,1],strides = [1,2,2,1], padding='VALID', name="pool2")
                x = nf.convolution_layer(x, model_params["conv2_2"], [1,1,1,1], name="conv3", activat_fn=nf.lrelu, initializer=init)
            
            with tf.variable_scope("decoder", reuse=reuse):     
                x = nf.convolution_layer(x, model_params["conv2_2"], [1,1,1,1], name="d_conv3", activat_fn=nf.lrelu, initializer=init)
                W_d_conv1 = tf.get_variable("w_d_conv1", [3,3,256,256], tf.float32, initializer= init)

                output_shape_d_conv1 = tf.stack([tf.shape(x)[0], 48, 48, 256])
                x = tf.nn.conv2d_transpose(x, W_d_conv1, output_shape_d_conv1, [1, 2, 2, 1], name="upsample1")
                print(output_shape_d_conv1)
                
                x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], pre_shape = 256, name="d_conv2", activat_fn=nf.lrelu, initializer=init)
                
                W_d_conv2 = tf.get_variable("w_d_conv2", [3,3,64,128], tf.float32, initializer= init)
                output_shape_d_conv2 = tf.stack([tf.shape(x)[0], 96, 96, 64])
                x = tf.nn.conv2d_transpose(x, W_d_conv2, output_shape_d_conv2, [1, 2, 2, 1], name="upsample2")
                
                #x = nf.upsample(x, 2, feature_size, 3,None, initializer=init)
                network = nf.convolution_layer(x, model_params["conv3"], [1,1,1,1], pre_shape = 64, name="d_conv1", activat_fn=None, initializer=init)
               
            
       
        ###Discriminator
        num_resblock = 0

        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3]
                        }

        if is_training:

            if d_inputs == None: 

                #d_inputs = network  + 3.0*(network - d_target)
                #Test 
                res_gen = network - d_target
                input_gan = tf.concat([network,res_gen], axis=3)
                #d_inputs = network
            else:
                res_gen = d_inputs - d_target
                input_gan = tf.concat([d_inputs, res_gen], axis=3)
            
            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     
                x = nf.convolution_layer( input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, is_bn=True,initializer=init)
                conv_1 = x
                with tf.variable_scope("resblock", reuse=reuse): 
                
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, is_bn=True, reuse=reuse, idx = i, activation_fn=nf.lrelu, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2",activat_fn=nf.lrelu,  is_bn=True, initializer=init)
                        x += conv_1
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, is_bn=True, initializer=init)
                d_logits = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)

        else:
            d_logits = network
  
        return [network, d_logits]


    def edsr_lsgan_recursive(self, kwargs):

        ###Generator
        init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32)
        scale = kwargs["scale"]
        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        aux_input = kwargs["aux_input"]
        is_training = kwargs["is_training"]
        is_generate = kwargs["is_generate"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 4


            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3],

                        'conv1_wgan': [5,5,feature_size],
                        'conv2_wgan': [5,5,feature_size*2],
                        'conv3_wgan': [5,5,feature_size*4],
                        'd_output_wgan': [5,5,3],                        
                        'maxpool_wgan': [1, 3, 3, 1],
                        }
        
        with tf.variable_scope("EDSR_gen", reuse=reuse): 
            concat_input = tf.concat([self.inputs, aux_input], axis=3)   
            x = nf.convolution_layer(concat_input, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
            conv_1 = x
            with tf.variable_scope("resblock",reuse=reuse): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                    x += conv_1

            with tf.variable_scope("upsamplex2", reuse=reuse):
                    #upsample2 = nf.upsample(x, 2, feature_size, 3,None, initializer=init)
                    upsample2 = x
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None, initializer=init)
       
        ###Discriminator
        """
        DEPTH = 64
        model_params = {

                        'conv1_wgan': [5,5,DEPTH],
                        'conv2_wgan': [5,5,DEPTH*2],
                        'conv3_wgan': [5,5,DEPTH*4],
                        'd_output_wgan': [5,5,3],                        
                        'maxpool_wgan': [1, 3, 3, 1],
                        }
        """
        num_resblock = 0

        if is_training:

            if d_inputs == None: 

                #Test 
                input_gan = network
            else:
                input_gan = d_inputs
            
            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     
                x = nf.convolution_layer( input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, is_bn=True,initializer=init)
                
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, is_bn=True, initializer=init)
                mid_out = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)
                
                #Down sample
                x = nf.convolution_layer(mid_out,   model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan",     activat_fn=nf.lrelu, initializer=init)

                pool1 = nf.max_pool_layer(x, [1, 2, 2, 1], [1, 2, 2, 1], name="conv1_wgan_mp")
                pool1_ = nf.max_pool_layer(-x, [1, 2, 2, 1], [1, 2, 2, 1], name="conv1_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool1_), pool1), tf.float32)
                plus_mask = tf.cast(tf.greater(pool1, tf.abs(pool1_)), tf.float32)
                pool1 = plus_mask*pool1 + minus_mask*(-pool1_)
                    
                x = nf.convolution_layer(pool1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan",     activat_fn=nf.lrelu, initializer=init)

                pool2 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                pool2_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool2_), pool2), tf.float32)
                plus_mask = tf.cast(tf.greater(pool2, tf.abs(pool2_)), tf.float32)
                pool2 = plus_mask*pool2 + minus_mask*(-pool2_)

                x = nf.convolution_layer(pool2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan",     activat_fn=nf.lrelu, initializer=init)
                    
                pool3 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                pool3_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool3_), pool3), tf.float32)
                plus_mask = tf.cast(tf.greater(pool3, tf.abs(pool3_)), tf.float32)
                pool3 = plus_mask*pool3 + minus_mask*(-pool3_)    
                x = nf.convolution_layer(pool3,           model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan",  activat_fn=nf.lrelu, initializer=init)
                d_logits = x
        else:
            d_logits = network
  
        return [network, d_logits, mid_out]


    

    def edsr_lsgan_lap(self, kwargs):

        ###Generator
        init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32)
        scale = kwargs["scale"]
        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        is_generate = kwargs["is_generate"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 4

        def laplacian_filter(image_target):

            with tf.name_scope("laplacian_operation"):

                laplacian_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
                laplacian_kernel = np.reshape(laplacian_kernel,[3,3,1,1])

        

                gt = tf.split(image_target, 3, axis=3)
                

                for i in range(3):

                    gt[i] = tf.nn.conv2d(gt[i], laplacian_kernel, [1,1,1,1], padding='VALID')
                  
                lap_results = tf.concat(gt, axis=3)    
            

            return lap_results
    
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3],

                        'conv1_wgan': [5,5,feature_size],
                        'conv2_wgan': [5,5,feature_size*2],
                        'conv3_wgan': [5,5,feature_size*4],
                        'd_output_wgan': [5,5,3],                        
                        'maxpool_wgan': [1, 3, 3, 1],
                        }
        
        with tf.variable_scope("EDSR_gen", reuse=reuse): 
            
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
            conv_1 = x
            with tf.variable_scope("resblock",reuse=reuse): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                    x += conv_1

            with tf.variable_scope("upsamplex2", reuse=reuse):
                    #upsample2 = nf.upsample(x, 2, feature_size, 3,None, initializer=init)
                    upsample2 = x
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None, initializer=init)
       
        ###Discriminator

        num_resblock = 0

        if is_training:

            if d_inputs == None: 

                #Test 
                input_gan = network
            else:
                input_gan = d_inputs

            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     
                x = nf.convolution_layer( input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, is_bn=True,initializer=init)
                
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, is_bn=True, initializer=init)
                mid_out = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)
                
                #Down sample
                x = nf.convolution_layer(mid_out,   model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan",     activat_fn=nf.lrelu, initializer=init)

                pool1 = nf.max_pool_layer(x, [1, 2, 2, 1], [1, 2, 2, 1], name="conv1_wgan_mp")
                pool1_ = nf.max_pool_layer(-x, [1, 2, 2, 1], [1, 2, 2, 1], name="conv1_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool1_), pool1), tf.float32)
                plus_mask = tf.cast(tf.greater(pool1, tf.abs(pool1_)), tf.float32)
                pool1 = plus_mask*pool1 + minus_mask*(-pool1_)
                    
                x = nf.convolution_layer(pool1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan",     activat_fn=nf.lrelu, initializer=init)

                pool2 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                pool2_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool2_), pool2), tf.float32)
                plus_mask = tf.cast(tf.greater(pool2, tf.abs(pool2_)), tf.float32)
                pool2 = plus_mask*pool2 + minus_mask*(-pool2_)

                x = nf.convolution_layer(pool2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan",     activat_fn=nf.lrelu, initializer=init)
                    
                pool3 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                pool3_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool3_), pool3), tf.float32)
                plus_mask = tf.cast(tf.greater(pool3, tf.abs(pool3_)), tf.float32)
                pool3 = plus_mask*pool3 + minus_mask*(-pool3_)    
                x = nf.convolution_layer(pool3,           model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan",  activat_fn=nf.lrelu, initializer=init)
                d_logits = x
        else:
            d_logits = network
  
        return [network, d_logits, mid_out, laplacian_filter(network), laplacian_filter(d_target)]


    def edsr_lsgan_lap_v2(self, kwargs):

        ###Generator
        init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32)
        scale = kwargs["scale"]
        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        is_generate = kwargs["is_generate"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 4

        def laplacian_filter(image_target):

            with tf.name_scope("laplacian_operation"):

                laplacian_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
                laplacian_kernel = np.reshape(laplacian_kernel,[3,3,1,1])

        

                gt = tf.split(image_target, 3, axis=3)
                

                for i in range(3):

                    gt[i] = tf.nn.conv2d(gt[i], laplacian_kernel, [1,1,1,1], padding='VALID')
                  
                lap_results = tf.concat(gt, axis=3)    
            

            return lap_results
    
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3],

                        'conv1_wgan': [5,5,feature_size],
                        'conv2_wgan': [5,5,feature_size*2],
                        'conv3_wgan': [5,5,feature_size*4],
                        'd_output_wgan': [5,5,3],                        
                        'maxpool_wgan': [1, 3, 3, 1],
                        }
        
        with tf.variable_scope("EDSR_gen", reuse=reuse): 
            
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
            conv_1 = x
            with tf.variable_scope("resblock",reuse=reuse): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                    x += conv_1

            with tf.variable_scope("upsamplex2", reuse=reuse):
                    #upsample2 = nf.upsample(x, 2, feature_size, 3,None, initializer=init)
                    upsample2 = x
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None, initializer=init)
       
        ###Discriminator

        num_resblock = 0
        if is_training:

            if d_inputs == None: 

                #Test 
                input_gan = laplacian_filter(network)
            else:
                input_gan = laplacian_filter(d_inputs)

            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     
                x = nf.convolution_layer( input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, is_bn=True,initializer=init)
                
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, is_bn=True, initializer=init)
                mid_out = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)
                
                #Down sample
                x = nf.convolution_layer(mid_out,   model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan",     activat_fn=nf.lrelu, initializer=init)

                pool1 = nf.max_pool_layer(x, [1, 2, 2, 1], [1, 2, 2, 1], name="conv1_wgan_mp")
                pool1_ = nf.max_pool_layer(-x, [1, 2, 2, 1], [1, 2, 2, 1], name="conv1_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool1_), pool1), tf.float32)
                plus_mask = tf.cast(tf.greater(pool1, tf.abs(pool1_)), tf.float32)
                pool1 = plus_mask*pool1 + minus_mask*(-pool1_)
                    
                x = nf.convolution_layer(pool1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan",     activat_fn=nf.lrelu, initializer=init)

                pool2 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                pool2_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool2_), pool2), tf.float32)
                plus_mask = tf.cast(tf.greater(pool2, tf.abs(pool2_)), tf.float32)
                pool2 = plus_mask*pool2 + minus_mask*(-pool2_)

                x = nf.convolution_layer(pool2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan",     activat_fn=nf.lrelu, initializer=init)
                    
                pool3 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                pool3_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool3_), pool3), tf.float32)
                plus_mask = tf.cast(tf.greater(pool3, tf.abs(pool3_)), tf.float32)
                pool3 = plus_mask*pool3 + minus_mask*(-pool3_)    
                x = nf.convolution_layer(pool3,           model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan",  activat_fn=nf.lrelu, initializer=init)
                d_logits = x
        else:
            d_logits = network
  
        return [network, d_logits, mid_out]


    def edsr_lsgan_lap_att_v2(self, kwargs):

        ###Generator
        init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32)
        scale = kwargs["scale"]
        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        is_generate = kwargs["is_generate"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 4
        

        def attention_network(image_input, scale, layers, channels, is_training):

            with tf.variable_scope("attention"):
                    
                    att_net = nf.convolution_layer(image_input, [3,3,64], [1,2,2,1],name="conv1-1")
                    att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv1-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-1")
                    att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-1")
                    att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-1")
                    #att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = tf.reshape(att_net, [-1, int(np.prod(att_net.get_shape()[1:]))]) 
                    att_net = nf.fc_layer(att_net, 2048, name="fc1")
                    att_net = nf.fc_layer(att_net, 2048, name="fc2")
                    #att_net = tf.layers.dropout(att_net, rate=dropout, training=is_training, name='dropout1')
                    logits = nf.fc_layer(att_net, channels*layers, name="logits", activat_fn=None)
                    
                    bsize = tf.shape(logits)[0]
                    #logits = tf.reshape(logits, (bsize,1,1,channels*layers))
                    logits = tf.reshape(logits, (bsize,1,1,channels, layers))
                    weighting = tf.nn.softmax(logits)
                    
                    """
                    max_index = tf.argmax(tf.nn.softmax(logits),4) 
                    weighting = tf.one_hot(max_index, 
                                        depth=layers, 
                                        on_value=1.0,
                                        axis = -1)
                    """
                  

            return weighting

        def laplacian_filter(image_target):

            with tf.name_scope("laplacian_operation"):

                laplacian_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
                laplacian_kernel = np.reshape(laplacian_kernel,[3,3,1,1])

        

                gt = tf.split(image_target, 3, axis=3)
                

                for i in range(3):

                    gt[i] = tf.nn.conv2d(gt[i], laplacian_kernel, [1,1,1,1], padding='VALID')
                  
                lap_results = tf.concat(gt, axis=3)    
            

            return lap_results
    
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,384],
                        'g_output': [3,3,3],
                        'd_output': [3,3,3],

                        'conv1_wgan': [5,5,feature_size],
                        'conv2_wgan': [5,5,feature_size*2],
                        'conv3_wgan': [5,5,feature_size*4],
                        'd_output_wgan': [5,5,3],                        
                        'maxpool_wgan': [1, 3, 3, 1],
                        }

        
        
        with tf.variable_scope("EDSR_gen", reuse=reuse): 

            with tf.name_scope("attention_x2"):
                att_layers = 128
                arr = att_weight_x2 = attention_network(self.inputs, scale, att_layers,3, is_training)
            
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
            conv_1 = x
            with tf.variable_scope("resblock",reuse=reuse): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                    x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                    x += conv_1

            with tf.variable_scope("upsamplex2", reuse=reuse):
                    #upsample2 = nf.upsample(x, 2, feature_size, 3,None, initializer=init)
                    upsample2 = x
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None, initializer=init)
                    bsize, a, b, c = network.get_shape().as_list()
                    network = tf.reshape(network, (-1, a, b, 3, att_layers))
                    network = tf.multiply(network, arr)
                    #network = nf.convolution_layer(network, model_params["g_output"], [1,1,1,1], name="g_output", activat_fn=None, initializer=init)
                    network = tf.reduce_sum(network,4)
                    print(network) 
        ###Discriminator

        num_resblock = 0
        if is_training:

            if d_inputs == None: 

                #Test 
                #input_gan = laplacian_filter(network)
                input_gan = network
            else:
                #input_gan = laplacian_filter(d_inputs)
                input_gan = d_inputs
            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     
                x = nf.convolution_layer( input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, is_bn=True,initializer=init)
                
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, is_bn=True, initializer=init)
                mid_out = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)
                
                #Down sample
                x = nf.convolution_layer(mid_out,   model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan",     activat_fn=nf.lrelu, initializer=init)

                pool1 = nf.max_pool_layer(x, [1, 2, 2, 1], [1, 2, 2, 1], name="conv1_wgan_mp")
                pool1_ = nf.max_pool_layer(-x, [1, 2, 2, 1], [1, 2, 2, 1], name="conv1_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool1_), pool1), tf.float32)
                plus_mask = tf.cast(tf.greater(pool1, tf.abs(pool1_)), tf.float32)
                pool1 = plus_mask*pool1 + minus_mask*(-pool1_)
                    
                x = nf.convolution_layer(pool1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan",     activat_fn=nf.lrelu, initializer=init)

                pool2 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                pool2_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool2_), pool2), tf.float32)
                plus_mask = tf.cast(tf.greater(pool2, tf.abs(pool2_)), tf.float32)
                pool2 = plus_mask*pool2 + minus_mask*(-pool2_)

                x = nf.convolution_layer(pool2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan",     activat_fn=nf.lrelu, initializer=init)
                    
                pool3 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                pool3_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                minus_mask = tf.cast(tf.greater(tf.abs(pool3_), pool3), tf.float32)
                plus_mask = tf.cast(tf.greater(pool3, tf.abs(pool3_)), tf.float32)
                pool3 = plus_mask*pool3 + minus_mask*(-pool3_)    
                x = nf.convolution_layer(pool3,           model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan",  activat_fn=nf.lrelu, initializer=init)
                d_logits = x
        else:
            d_logits = network
  
        return [network, d_logits, mid_out]



    def edsr_lsgan_dis_large(self, kwargs):


        ###Generator
        init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None, dtype=tf.float32)
        scale = kwargs["scale"]
        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        is_generate = kwargs["is_generate"]
        feature_size = 64
        scaling_factor = 1
        num_resblock = 2
            
        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3]
                        }
        network = d_inputs

        if is_generate:
                with tf.device('/gpu:0'):
                        with tf.variable_scope("EDSR_gen", reuse=reuse):     
                            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
                            conv_1 = x

                            with tf.variable_scope("resblock",reuse=reuse): 
                
                        #Add the residual blocks to the model
                                for i in range(num_resblock):
                                    x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                                x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                                x += conv_1

                            with tf.variable_scope("upsamplex2", reuse=reuse):
                        #upsample2 = nf.upsample(x, 2, feature_size, 3,None, initializer=init)
                                upsample2 = x
                                network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None, initializer=init)
           
        ###Discriminator

        num_resblock = 2

        if is_training:

            if d_inputs == None: 
                d_inputs = network 
                #d_inputs = network

            input_gan = d_inputs
            #input_gan = tf.concat([d_inputs, d_target], axis=3)
            with tf.device('/gpu:1'):
                with tf.variable_scope("EDSR_dis", reuse=reuse):

                    input_gan =  nf.upsample(input_gan, 2, feature_size, 3,None, initializer=init)  
                    x = nf.convolution_layer( input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, is_bn=True,initializer=init)
                    conv_1 = x
                    with tf.variable_scope("resblock", reuse=reuse): 
                    
                            #Add the residual blocks to the model
                            for i in range(num_resblock):
                                x = nf.resBlock(x,feature_size,scale=scaling_factor, is_bn=True, reuse=reuse, idx = i, activation_fn=nf.lrelu, initializer=init)
                            x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2",activat_fn=nf.lrelu,  is_bn=True, initializer=init)
                            x += conv_1
                    x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, is_bn=True, initializer=init)
                    d_logits = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)
               
            

        else:
            d_logits = network
  
        return [network, d_logits]


    def EDSR_WGAN_att(self, kwargs):


        def attention_network(image_input, layers, channels, is_training):

            with tf.variable_scope("attention"):
                    
                    att_net = nf.convolution_layer(image_input, [3,3,64], [1,2,2,1],name="conv1-1")
                    att_net = nf.convolution_layer(att_net, [3,3,64], [1,1,1,1],name="conv1-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-1")
                    att_net = nf.convolution_layer(att_net, [3,3,128], [1,1,1,1],name="conv2-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-1")
                    att_net = nf.convolution_layer(att_net, [3,3,256], [1,1,1,1],name="conv3-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-1")
                    #att_net = nf.convolution_layer(att_net, [3,3,512], [1,1,1,1],name="conv4-2")
                    att_net = tf.nn.max_pool(att_net, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
                    att_net = tf.reshape(att_net, [-1, int(np.prod(att_net.get_shape()[1:]))]) 
                    att_net = nf.fc_layer(att_net, 2048, name="fc1")
                    att_net = nf.fc_layer(att_net, 2048, name="fc2")
                    #att_net = tf.layers.dropout(att_net, rate=dropout, training=is_training, name='dropout1')
                    logits = nf.fc_layer(att_net, channels*layers, name="logits", activat_fn=None)
                    
                    bsize = tf.shape(logits)[0]
                    #logits = tf.reshape(logits, (bsize,1,1,channels*layers))
                    logits = tf.reshape(logits, (bsize,1,1,channels, layers))
                    weighting = tf.nn.softmax(logits)
                    
                    """
                    max_index = tf.argmax(tf.nn.softmax(logits),4) 
                    weighting = tf.one_hot(max_index, 
                                        depth=layers, 
                                        on_value=1.0,
                                        axis = -1)
                    """
                  

            return weighting

        reuse = kwargs["reuse"]
        d_inputs = kwargs["d_inputs"]
        d_target = kwargs["d_target"]
        is_training = kwargs["is_training"]
        net = kwargs["net"]
        
        init = tf.random_normal_initializer(stddev=0.01)

        feature_size = 64
        scaling_factor = 1

        DEPTH = 28
#        DEPTH = 32

        model_params = {

                        'conv1': [3,3,feature_size],
                        'resblock': [3,3,feature_size],
                        'conv2': [3,3,feature_size],
                        'conv3': [3,3,3],
                        'd_output': [3,3,3*feature_size],
                        
                        'conv1_wgan-gp': [5,5,DEPTH],
                        'conv2_wgan-gp': [5,5,DEPTH*2],
                        'conv3_wgan-gp': [5,5,DEPTH*4],
                        'd_output_wgan-gp': [5,5,3],
                        
                        # v5-0
                        'conv1_wgan': [5,5,DEPTH],
                        'conv2_wgan': [5,5,DEPTH*2],
                        'conv3_wgan': [5,5,DEPTH*4],
                        'd_output_wgan': [5,5,3],                        
                        'maxpool_wgan': [1, 2, 2, 1],
#                        
#                        # v5-1
#                        'conv1_wgan': [3,3,DEPTH],
#                        'conv2_wgan': [3,3,DEPTH*2],
#                        'conv3_wgan': [3,3,DEPTH*4],
#                        'd_output_wgan': [3,3,3],                       
#                        'maxpool_wgan': [1, 3, 3, 1],
                        
#                        # v5-3
#                        'conv1_wgan': [9,9,DEPTH],
#                        'conv2_wgan': [9,9,DEPTH*2],
#                        'conv3_wgan': [9,9,DEPTH*4],
#                        'd_output_wgan': [9,9,3],                       
#                        'maxpool_wgan': [1, 2, 2, 1],                        
                        
#                        # v5-4
#                        'conv1_wgan': [5,5,DEPTH],
#                        'conv2_wgan': [7,7,DEPTH*2],
#                        'conv3_wgan': [9,9,DEPTH*4],
#                        'd_output_wgan': [5,5,3],                       
#                        'maxpool_wgan': [1, 2, 2, 1],    


                        }

        if net is "Gen":
        
            ### Generator
            num_resblock = 16
                       
            g_input = self.inputs
            
            with tf.variable_scope("EDSR_gen", reuse=reuse):  

                with tf.name_scope("attention_x2"):
                    att_layers = feature_size
                    arr = att_weight_x2 = attention_network(self.inputs, att_layers,3, is_training)
               
                x = nf.convolution_layer(g_input, model_params["conv1"], [1,1,1,1], name="conv1", activat_fn=None, initializer=init)
                conv_1 = x
                with tf.variable_scope("resblock",reuse=reuse): 
                
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2", activat_fn=None, initializer=init)
                        x += conv_1
                x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=None, initializer=init)
                g_network = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=None, initializer=init)
                
                #Attention
                bsize, a, b, c = g_network.get_shape().as_list()
                g_network = tf.reshape(g_network, (-1, a, b, 3, att_layers))
                g_network = tf.multiply(g_network, arr)
                g_network = tf.reduce_sum(g_network,4)

                g_output = tf.nn.sigmoid(g_network)
                           
            return g_output

        elif net is "Dis":
            d_model = kwargs["d_model"]            
            
            ### Discriminator
            num_resblock = 2
            
            input_gan = d_inputs 
            
            with tf.variable_scope("EDSR_dis", reuse=reuse):     
                if d_model is "EDSR":
                    
                    x = nf.convolution_layer(input_gan, model_params["conv1"], [1,1,1,1], name="conv1",  activat_fn=nf.lrelu, initializer=init)
                    conv_1 = x
                    with tf.variable_scope("resblock", reuse=reuse):                   
                        #Add the residual blocks to the model
                        for i in range(num_resblock):
                            x = nf.resBlock(x,feature_size,scale=scaling_factor, reuse=reuse, idx = i, activation_fn=nf.lrelu, initializer=init)
                        x = nf.convolution_layer(x, model_params["conv2"], [1,1,1,1], name="conv2",activat_fn=nf.lrelu, initializer=init)
                        x += conv_1
                        
                    x = nf.convolution_layer(x, model_params["conv1"], [1,1,1,1], name="conv3",  activat_fn=nf.lrelu, initializer=init)
                    d_logits = nf.convolution_layer(x, model_params["d_output"], [1,1,1,1], name="conv4", activat_fn=nf.lrelu, flatten=False, initializer=init)
                    
                elif d_model is "WGAN-GP":
                    
                    x = nf.convolution_layer(input_gan, model_params["conv1_wgan-gp"],    [1,1,1,1], name="conv1_wgan-gp",     activat_fn=nf.lrelu, initializer=init)
                    x = nf.convolution_layer(x,         model_params["conv2_wgan-gp"],    [1,1,1,1], name="conv2_wgan-gp",     activat_fn=nf.lrelu, initializer=init)
                    x = nf.convolution_layer(x,         model_params["conv3_wgan-gp"],    [1,1,1,1], name="conv3_wgan-gp",     activat_fn=nf.lrelu, initializer=init)
                    x = nf.convolution_layer(x,         model_params["d_output_wgan-gp"], [1,1,1,1], name="d_output_wgan-gp",  activat_fn=nf.lrelu, initializer=init)
                    d_logits = x
                
                elif d_model is "PatchWGAN":    

                    x = nf.convolution_layer(input_gan,   model_params["conv1_wgan"],    [1,1,1,1], name="conv1_wgan",     activat_fn=nf.lrelu, initializer=init)
                    
                    pool1 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv1_wgan_mp")
                    pool1_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv1_wgan_mp")
                    minus_mask = tf.cast(tf.greater(tf.abs(pool1_), pool1), tf.float32)
                    plus_mask = tf.cast(tf.greater(pool1, tf.abs(pool1_)), tf.float32)
                    pool1 = plus_mask*pool1 + minus_mask*(-pool1_)
                    
                    x = nf.convolution_layer(pool1,       model_params["conv2_wgan"],    [1,1,1,1], name="conv2_wgan",     activat_fn=nf.lrelu, initializer=init)

                    pool2 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                    pool2_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv2_wgan_mp")
                    minus_mask = tf.cast(tf.greater(tf.abs(pool2_), pool2), tf.float32)
                    plus_mask = tf.cast(tf.greater(pool2, tf.abs(pool2_)), tf.float32)
                    pool2 = plus_mask*pool2 + minus_mask*(-pool2_)

                    x = nf.convolution_layer(pool2,       model_params["conv3_wgan"],    [1,1,1,1], name="conv3_wgan",     activat_fn=nf.lrelu, initializer=init)
                    
                    pool3 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                    pool3_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv3_wgan_mp")
                    minus_mask = tf.cast(tf.greater(tf.abs(pool3_), pool3), tf.float32)
                    plus_mask = tf.cast(tf.greater(pool3, tf.abs(pool3_)), tf.float32)
                    pool3 = plus_mask*pool3 + minus_mask*(-pool3_)
                    
                    x = nf.convolution_layer(pool3,           model_params["d_output_wgan"], [1,1,1,1], name="d_output_wgan",  activat_fn=nf.lrelu, initializer=init)

                    ### v4
#                    pool4 = nf.max_pool_layer(x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv4_wgan_mp")
#                    pool4_ = nf.max_pool_layer(-x, model_params["maxpool_wgan"], [1, 2, 2, 1], name="conv4_wgan_mp")
#                    minus_mask = tf.cast(tf.greater(tf.abs(pool4_), pool4), tf.float32)
#                    plus_mask = tf.cast(tf.greater(pool4, tf.abs(pool4_)), tf.float32)
#                    x = plus_mask*pool4 + minus_mask*(-pool4_)

                    d_logits = x

                elif d_model is "PatchWGAN_GP":    

                    patch_size = 16
                    _, image_h, image_w, image_c = input_gan.get_shape().as_list()
                    
                    d_patch_list = []
                    for i in range(0, image_h//patch_size):
                        for j in range(0, image_w//patch_size):    
                            input_patch = input_gan[:, i:i+patch_size, j:j+patch_size, :] 
                            
                            x = nf.convolution_layer(input_patch, model_params["conv1_wgan-gp"],    [1,1,1,1], name="conv1_wgan-gp",     activat_fn=nf.lrelu, initializer=init)
                            x = nf.convolution_layer(x,           model_params["conv2_wgan-gp"],    [1,1,1,1], name="conv2_wgan-gp",     activat_fn=nf.lrelu, initializer=init)        
                            x = nf.convolution_layer(x,           model_params["conv3_wgan-gp"],    [1,1,1,1], name="conv3_wgan-gp",     activat_fn=nf.lrelu, initializer=init)        
                            x = nf.convolution_layer(x,           model_params["d_output_wgan-gp"], [1,1,1,1], name="d_output_wgan-gp",  activat_fn=nf.lrelu, initializer=init)        

                            d_curr_patch = x
                            d_curr_patch = tf.reduce_mean(d_curr_patch, axis=[1,2,3])
                            d_patch_list.append(d_curr_patch)
                            
                    d_patch_stack = tf.stack([d_patch_list[i] for i in range((image_h//patch_size)*(image_w//patch_size))], axis=1)
                    d_patch_weight = d_patch_stack / tf.reduce_sum(tf.abs(d_patch_stack), axis=1, keep_dims=True)
                    d_patch = d_patch_weight*d_patch_stack

                    d_logits = d_patch
                    
            return d_logits


    def build_model(self, kwargs = {}):

        model_list = ["googleLeNet_v1", "resNet_v1", "srcnn_v1", "grr_srcnn_v1",
                      "grr_grid_srcnn_v1","edsr_v1", "espcn_v1","edsr_v2",
                      "edsr_attention_v1", "edsr_1X1_v1", "edsr_local_att_v1",
                      "edsr_local_att_v2_upsample", "edsr_attention_v2", "edsr_v2_dual",
                      "edsr_lsgan", "edsr_lsgan_up", "edsr_lsgan_dis_large"
                      , "edsr_wgan_encode", "edsr_lsgan_recursive", "edsr_lsgan_lap",
                      "edsr_lsgan_lap_v2", "edsr_lsgan_lap_att_v2", "EDSR_WGAN_att"]
        
        if self.model_ticket not in model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
           
            fn = getattr(self,self.model_ticket)
            
            if kwargs == {}:
                netowrk = fn()
            else:
                netowrk = fn(kwargs)
            return netowrk
        
        
def unit_test():

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    mz = model_zoo(x, dropout, is_training,"resNet_v1")
    return mz.build_model()
    

#m = unit_test()