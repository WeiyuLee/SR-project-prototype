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

            with tf.name_scope("upsamplex2"):
                    upsample2 = nf.upsample(x, 2, feature_size, True,None)
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)

            with tf.name_scope("upsamplex4"):
                    upsample4 = nf.upsample(upsample2, 2, feature_size, True,None)
                    network2 = nf.convolution_layer(upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)
       
        return [network, network2]


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

    def grr_edsr_v2(self, kwargs):

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
        
        with tf.name_scope("EDSR_v1_stg1"):    
            stg1_input = self.inputs
            
            stg1_x = nf.convolution_layer(stg1_input, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_stg1_1 = stg1_x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        stg1_x = nf.resBlock(stg1_x,feature_size,scale=scaling_factor)
                    stg1_x = nf.convolution_layer(stg1_x, model_params["conv2"], [1,1,1,1], name="conv2")
                    stg1_output = stg1_x + conv_stg1_1

#            with tf.name_scope("upsamplex2"):
#                    stg1_upsample2 = nf.upsample(stg1_output, 2, feature_size, True,None)
#                    stg1_network = nf.convolution_layer(stg1_upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)
#
#            with tf.name_scope("upsamplex4"):
#                    stg1_upsample4 = nf.upsample(stg1_upsample2, 2, feature_size, True,None)
#                    stg1_network2 = nf.convolution_layer(stg1_upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)

        with tf.name_scope("EDSR_v1_stg2"):     
            stg2_input = nf.shortcut(self.inputs, stg1_output, "stg2_input")

            stg2_x = nf.convolution_layer(stg2_input, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_stg2_1 = stg2_x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        stg2_x = nf.resBlock(stg2_x,feature_size,scale=scaling_factor)
                    stg2_x = nf.convolution_layer(stg2_x, model_params["conv2"], [1,1,1,1], name="conv2")
                    stg2_output = stg2_x + conv_stg2_1

#            with tf.name_scope("upsamplex2"):
#                    stg2_upsample2 = nf.upsample(stg2_output, 2, feature_size, True,None)
#                    stg2_network = nf.convolution_layer(stg2_upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)
#
#            with tf.name_scope("upsamplex4"):
#                    stg2_upsample4 = nf.upsample(stg2_upsample2, 2, feature_size, True,None)
#                    stg2_network2 = nf.convolution_layer(stg2_upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)
       
        with tf.name_scope("EDSR_v1_stg3"):     
            stg3_input = nf.shortcut(stg2_input, stg2_output, "stg3_input")

            stg3_x = nf.convolution_layer(stg3_input, model_params["conv1"], [1,1,1,1], name="conv1")
            conv_stg3_1 = stg3_x
            with tf.name_scope("resblock"): 
            
                    #Add the residual blocks to the model
                    for i in range(num_resblock):
                        stg3_x = nf.resBlock(stg3_x,feature_size,scale=scaling_factor)
                    stg3_x = nf.convolution_layer(stg3_x, model_params["conv2"], [1,1,1,1], name="conv2")
                    stg3_output = stg3_x + conv_stg3_1

            with tf.name_scope("upsamplex2"):
                    stg3_upsample2 = nf.upsample(stg3_output, 2, feature_size, True,None)
                    stg3_network = nf.convolution_layer(stg3_upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)

            with tf.name_scope("upsamplex4"):
                    stg3_upsample4 = nf.upsample(stg3_upsample2, 2, feature_size, True,None)
                    stg3_network2 = nf.convolution_layer(stg3_upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)
        
        return [stg3_network, stg3_network2]

    def GoogLeNet_edsr_v1(self, kwargs):

        scale = kwargs["scale"]
        feature_size = kwargs["feature_size"]
        scaling_factor = 1
        num_resblock = 16
           
        model_params = {
                        'conv1': [9,9,feature_size],

                        "inception_1":{                 
                            'gn_1x1_conv1': [1,1,feature_size/2],
                            'gn_1x1_resblock': [1,1,feature_size/2],
                            'gn_1x1_conv2': [1,1,feature_size/2],

                            'gn_3x3_conv1': [3,3,feature_size*2],
                            'gn_3x3_resblock': [3,3,feature_size*2],
                            'gn_3x3_conv2': [3,3,feature_size*2],
    
                            'gn_5x5_conv1': [5,5,feature_size*2],
                            'gn_5x5_resblock': [5,5,feature_size*2],
                            'gn_5x5_conv2': [5,5,feature_size*2],  
                            
                            'gn_s1x1_conv': [1,1,feature_size/4],
                        },
                                
                        'conv2': [5,5,feature_size],                    
                        
                        'conv3': [3,3,3],    
                        
                        }
                        
        with tf.name_scope("GN_EDSR_v1"):     
                            
            x = nf.convolution_layer(self.inputs, model_params["conv1"], [1,1,1,1], name="conv1")

            net = nf.gn_edsr_inception_v1(x, model_params, num_resblock, name= "inception_1", flatten=False)

            x = nf.convolution_layer(net, model_params["conv2"], [1,1,1,1], name="conv2")

            with tf.name_scope("upsamplex2"):
                    upsample2 = nf.upsample(x, 2, feature_size, True,None)
                    network = nf.convolution_layer(upsample2, model_params["conv3"], [1,1,1,1], name="conv3", activat_fn=None)

            with tf.name_scope("upsamplex4"):
                    upsample4 = nf.upsample(upsample2, 2, feature_size, True,None)
                    network2 = nf.convolution_layer(upsample4, model_params["conv3"], [1,1,1,1], name="conv4", activat_fn=None)
       
        return [network, network2]

    def build_model(self, kwargs = {}):

        model_list = ["googleLeNet_v1", "resNet_v1", "srcnn_v1", "grr_srcnn_v1", "grr_grid_srcnn_v1", "edsr_v1", "espcn_v1", "edsr_v2", "grr_edsr_v2", "GoogLeNet_edsr_v1"]
        
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