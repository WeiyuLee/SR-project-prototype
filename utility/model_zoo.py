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
            
            # Grid 
            "grid_conv1": [7, 7, 64],
            "grid_conv2": [1, 1, 32],
            "grid_conv3": [3, 3, 1],                   
        }                
        
        with tf.name_scope("grr_grid_srcnn_v1"):           
 
            init = tf.random_normal_initializer(mean=0, stddev=1e-3)
            padding = 6
            padding_grid = 4
            #print(self.inputs.shape)

            # Stage 1                        
            stg1_input = self.inputs
            
            stg1_layer1_output = nf.convolution_layer(stg1_input,         model_params["stg1_conv1"], [1,1,1,1], name="stg1_conv1", padding="SAME", initializer=init)           
            stg1_layer2_output = nf.convolution_layer(stg1_layer1_output, model_params["stg1_conv2"], [1,1,1,1], name="stg1_conv2", padding="SAME", initializer=init)
            stg1_layer3_output = nf.convolution_layer(stg1_layer2_output, model_params["stg1_conv3"], [1,1,1,1], name="stg1_conv3", padding="SAME", initializer=init, activat_fn=None)

            #stg1_layer3_output = tf.pad(stg1_layer3_output, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
                       
            stg1_layer3_output = tf.add(self.inputs, stg1_layer3_output) ## multi-stg_1
            
            stg1_output = stg1_layer3_output
            #stg1_output = tf.stop_gradient(stg1_layer3_output)

            # Stage 2            
            stg2_input = stg1_output
            
            stg2_layer1_output = nf.convolution_layer(stg2_input,         model_params["stg2_conv1"], [1,1,1,1], name="stg2_conv1", padding="SAME", initializer=init)           
            stg2_layer2_output = nf.convolution_layer(stg2_layer1_output, model_params["stg2_conv2"], [1,1,1,1], name="stg2_conv2", padding="SAME", initializer=init)
            stg2_layer3_output = nf.convolution_layer(stg2_layer2_output, model_params["stg2_conv3"], [1,1,1,1], name="stg2_conv3", padding="SAME", initializer=init, activat_fn=None)
            
            #stg2_layer3_output = tf.pad(stg2_layer3_output, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
            
            stg2_layer3_output = tf.add(stg1_output, stg2_layer3_output) ## multi-stg_2
            
            stg2_output = stg2_layer3_output
            #stg2_output = tf.stop_gradient(stg2_layer3_output)

            # Stage 3
            stg3_input = stg2_output
            
            stg3_layer1_output = nf.convolution_layer(stg3_input,         model_params["stg3_conv1"], [1,1,1,1], name="stg3_conv1", padding="VALID", initializer=init)           
            stg3_layer2_output = nf.convolution_layer(stg3_layer1_output, model_params["stg3_conv2"], [1,1,1,1], name="stg3_conv2", padding="VALID", initializer=init)
            stg3_layer3_output = nf.convolution_layer(stg3_layer2_output, model_params["stg3_conv3"], [1,1,1,1], name="stg3_conv3", padding="VALID", initializer=init, activat_fn=None)

            stg3_layer3_output = tf.add(stg2_output[:, padding:-padding, padding:-padding,:], stg3_layer3_output) ## multi-stg_3
            
            stg3_output = tf.stop_gradient(stg3_layer3_output)
            
            #stg3_layer3_output_pad = tf.pad(stg3_layer3_output, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")
            #print(stg3_layer3_output_pad.get_shape)
            
            stg3_layer3_size = stg3_layer3_output.get_shape()
            grid_h = stg3_layer3_size[1] // 2
            grid_w = stg3_layer3_size[2] // 2
            
            grid1_input = stg3_output[:, 0:grid_h,    0:grid_w,   :]
            grid2_input = stg3_output[:, 0:grid_h,    grid_w:,    :]
            grid3_input = stg3_output[:, grid_h:,     0:grid_w,   :]
            grid4_input = stg3_output[:, grid_h:,     grid_w:,    :]
            
            grid1_input = tf.pad(grid1_input, [[0, 0], [padding_grid, padding_grid], [padding_grid, padding_grid], [0, 0]], "SYMMETRIC")
            grid2_input = tf.pad(grid2_input, [[0, 0], [padding_grid, padding_grid], [padding_grid, padding_grid], [0, 0]], "SYMMETRIC")
            grid3_input = tf.pad(grid3_input, [[0, 0], [padding_grid, padding_grid], [padding_grid, padding_grid], [0, 0]], "SYMMETRIC")
            grid4_input = tf.pad(grid4_input, [[0, 0], [padding_grid, padding_grid], [padding_grid, padding_grid], [0, 0]], "SYMMETRIC")
            
            #print(grid1_input.get_shape())
            #print(grid2_input.get_shape())
            #print(grid3_input.get_shape())
            #print(grid4_input.get_shape())
            
            grid_input = [grid1_input, grid2_input, grid3_input, grid4_input]
            
            #FFT Grids
            #FFT = [[],[],[]],[]
            
            #if FFT > 0:
                #network1
            
            idx = tf.constant([0, 1, 2, 3])
            idx_shuffled = tf.random_shuffle(idx)
            grid_input_shuffled = tf.gather(grid_input, idx_shuffled)
            
            grid_output = [None]*4
            
            # Grid 1
            grid1_layer1_output = nf.convolution_layer(grid_input_shuffled[0], model_params["grid_conv1"], [1,1,1,1], name="grid1_conv1", padding="VALID", initializer=init)           
            grid1_layer2_output = nf.convolution_layer(grid1_layer1_output, model_params["grid_conv2"], [1,1,1,1], name="grid1_conv2", padding="VALID", initializer=init)
            grid1_layer3_output = nf.convolution_layer(grid1_layer2_output, model_params["grid_conv3"], [1,1,1,1], name="grid1_conv3", padding="VALID", initializer=init, activat_fn=None)
            grid_output[0] = grid1_layer3_output
            
            # Grid 2
            grid2_layer1_output = nf.convolution_layer(grid_input_shuffled[1], model_params["grid_conv1"], [1,1,1,1], name="grid2_conv1", padding="VALID", initializer=init)           
            grid2_layer2_output = nf.convolution_layer(grid2_layer1_output, model_params["grid_conv2"], [1,1,1,1], name="grid2_conv2", padding="VALID", initializer=init)
            grid2_layer3_output = nf.convolution_layer(grid2_layer2_output, model_params["grid_conv3"], [1,1,1,1], name="grid2_conv3", padding="VALID", initializer=init, activat_fn=None)
            grid_output[1] = grid2_layer3_output
            
            # Grid 3
            grid3_layer1_output = nf.convolution_layer(grid_input_shuffled[2], model_params["grid_conv1"], [1,1,1,1], name="grid3_conv1", padding="VALID", initializer=init)           
            grid3_layer2_output = nf.convolution_layer(grid3_layer1_output, model_params["grid_conv2"], [1,1,1,1], name="grid3_conv2", padding="VALID", initializer=init)
            grid3_layer3_output = nf.convolution_layer(grid3_layer2_output, model_params["grid_conv3"], [1,1,1,1], name="grid3_conv3", padding="VALID", initializer=init, activat_fn=None)
            grid_output[2] = grid3_layer3_output

            # Grid 4
            grid4_layer1_output = nf.convolution_layer(grid_input_shuffled[3], model_params["grid_conv1"], [1,1,1,1], name="grid4_conv1", padding="VALID", initializer=init)           
            grid4_layer2_output = nf.convolution_layer(grid4_layer1_output, model_params["grid_conv2"], [1,1,1,1], name="grid4_conv2", padding="VALID", initializer=init)
            grid4_layer3_output = nf.convolution_layer(grid4_layer2_output, model_params["grid_conv3"], [1,1,1,1], name="grid4_conv3", padding="VALID", initializer=init, activat_fn=None)            
            grid_output[3] = grid4_layer3_output  
            
#            idx_shuffled = idx_shuffled.eval()
#            print(idx_shuffled)
            
            idx_grid_dict = {idx_shuffled[i]:grid_output[i] for i in range(4)}
            grid_output_reorder = tf.gather(grid_output, tf.nn.top_k(-idx_shuffled, k=4).indices)
            
            #print(grid1_layer3_output.get_shape())
            #print(grid2_layer3_output.get_shape())
            #print(idx_grid_dict)
            
            tmp_output1 = tf.concat([grid_output_reorder[0], grid_output_reorder[1]], 2)
            tmp_output2 = tf.concat([grid_output_reorder[2], grid_output_reorder[3]], 2)
            final_output = tf.concat([tmp_output1, tmp_output2], 1)
            
            #print(tmp_output1.get_shape())
            #print(tmp_output2.get_shape())
            #print(final_output.get_shape())
            
        return (stg1_layer3_output, stg2_layer3_output, stg3_layer3_output),\
               (grid1_layer3_output, grid2_layer3_output, grid3_layer3_output, grid4_layer3_output),\
               final_output,\
               idx_shuffled      
    
    def build_model(self):
        model_list = ["googleLeNet_v1", "resNet_v1", "srcnn_v1", "grr_srcnn_v1", "grr_grid_srcnn_v1"]
        
        if self.model_ticket not in model_list:
            print("sorry, wrong ticket!")
            return 0
        
        else:
            fn = getattr(self, self.model_ticket)
            netowrk = fn()
            return netowrk
        
        
def unit_test():

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
    is_training = tf.placeholder(tf.bool, name='is_training')
    dropout = tf.placeholder(tf.float32, name='dropout')
    
    mz = model_zoo(x, dropout, is_training,"resNet_v1")
    return mz.build_model()
    

#m = unit_test()