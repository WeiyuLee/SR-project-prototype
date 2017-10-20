import cv2
import numpy as np
import os
import argparse
import tensorflow as tf
import sys
import scipy.misc

sys.path.append('./utility')
import utils as ut
import model_zoo  
import config

def resize(inputs, target_shape, interpolation = cv2.INTER_CUBIC):
    
    resize_img = cv2.resize(inputs, target_shape,  interpolation = interpolation)
    
    if inputs.shape[-1] == 1: resize_img = np.expand_dims(resize_img, axis = 2)
    
    return resize_img

def GaussianBlur(inputs, guassian_kernel_size, std):

    blur_img = cv2.GaussianBlur( inputs, guassian_kernel_size,std)
    if inputs.shape[-1] == 1: blur_img = np.expand_dims(blur_img, axis = 2)
    return blur_img

def imread(data_path, is_grayscale=False):

        if is_grayscale:
            img = scipy.misc.imread(data_path, flatten=True, mode='YCbCr').astype(np.float64)
            img = np.expand_dims(img, axis=2)
            return img
        else:
            return scipy.misc.imread(data_path, mode='YCbCr').astype(np.float64)        

def split_img(imgname,img, padding_size, subimg_size):

    """
    split a image into sub-images
    img: image for splitting
    padding_size: Size of padding
    subimg_size: Size of each subimage
    """

    ori_size = img.shape
    assert len(ori_size) == 3, "the dimensions of image shall be (height, width, channel)! " 

    #Calculate image size without padding
    padded_size = [    ori_size[0] - 2*padding_size[0],
                    ori_size[1] - 2*padding_size[1],
                    ori_size[2]]

    strides = subimg_size
    sub_imgs = {}

    for r in range(padded_size[0]//subimg_size[0]):
        for c in range(padded_size[1]//subimg_size[1]):

            grid_r = padding_size[0] + r*strides[0] 
            grid_c = padding_size[1] + c*strides[1] 

            sub_img = img[    grid_r - padding_size[0] : grid_r + strides[0] + padding_size[0],
                            grid_c - padding_size[1] : grid_c + strides[1] + padding_size[1],
                            :]

            # insert sub image to dictionary with key = [imagename]_[row_index]_[col_index]
            sub_imgs[imgname + "_"+ str(grid_r) + "_" + str(grid_c)] = sub_img

    return padded_size, sub_imgs



def merge_img(img_size, sub_images, padding_size,subimg_size, scale=2, down_scale_by_model=False):

    # Create an empty array for merging image

    padded_size = [    img_size[0],
                    img_size[1],
                    img_size[2]]

    merged_image = np.zeros([    (padded_size[0]//subimg_size[0])*subimg_size[0],
                                (padded_size[1]//subimg_size[1])*subimg_size[1],
                                 padded_size[2]])
    for k in sub_images:

        key = k.split("_")

        grid_r = int(key[1])*scale - padding_size[0]
        grid_c = int(key[2])*scale - padding_size[1]

        if down_scale_by_model == True: 
            padding_size_rescale = [0,0]
        else:
            padding_size_rescale = padding_size

        merged_image[grid_r:grid_r+subimg_size[0],
                        grid_c:grid_c+subimg_size[1],
                        :] = sub_images[k][padding_size_rescale[0]:padding_size_rescale[0]+subimg_size[0],
                                            padding_size_rescale[1]:padding_size_rescale[1]+subimg_size[1],
                                            :]
    return merged_image

class becnchmark:

    def run(self, inputs, target, psnr_mode='YCbCr'):

        self.input = inputs
        self.target = target

        assert inputs.shape[-1] == 3 or inputs.shape[-1] == 1, "Only allow inputs with channel 1 or 3!" 

        psnr = self._psnr(mode = psnr_mode)
        ssim = self._ssim()

        return psnr, ssim


    def _psnr(self, mode='YCbCr'):
        
        _input = self.input
        _target = self.target    

        if mode == 'YCbCr':
            if _input.shape[-1] == 1:
                _input = _input
                _target = _target
            else:
                _input = _input[:,:,0]
                _target = _target[:,:,0]
        else:
            _input = _input
            _target = _target


        mse = np.square(_input - _target)
        mse = mse.mean()
        psnr_val = 20*np.log10(255/(np.sqrt(mse)))    


        return psnr_val

    def _ssim(self, param = [0.01,0.03], L=255,guassian_kernel_size = (11,11), std = 1.5):

        

        C1 = np.power(param[0]*L,2)
        C2 = np.power(param[1]*L,2)

        _input = self.input
        _target = self.target

        clipping = [guassian_kernel_size[0]//2, _input.shape[0] - guassian_kernel_size[0]//2,
                    guassian_kernel_size[1]//2, _input.shape[1] - guassian_kernel_size[0]//2]


        
        blur_input = GaussianBlur( _input, guassian_kernel_size,std)[clipping[0]:clipping[1], clipping[2]:clipping[3],:]
        blur_target = GaussianBlur( _target, guassian_kernel_size,std)[clipping[0]:clipping[1], clipping[2]:clipping[3],:]
        
        
        mu1_sq = np.multiply(blur_input, blur_input)
        mu2_sq = np.multiply(blur_target, blur_target)
        mu1_mu2 = np.multiply(blur_input, blur_target)

        sigma1_sq = np.subtract(GaussianBlur(np.multiply(_input,_input),guassian_kernel_size,std)[clipping[0]:clipping[1], clipping[2]:clipping[3],:], mu1_sq)
        sigma2_sq = np.subtract(GaussianBlur(np.multiply(_target,_target),guassian_kernel_size,std)[clipping[0]:clipping[1], clipping[2]:clipping[3],:], mu2_sq)
        sigma12 = np.subtract(GaussianBlur(np.multiply(_input,_target),guassian_kernel_size,std)[clipping[0]:clipping[1], clipping[2]:clipping[3],:], mu1_mu2)

        
        upper_part = np.multiply(2*mu1_mu2 + C1, 2*sigma12+C2)
        down_part =  np.multiply(mu1_sq+mu2_sq+C1, sigma1_sq+sigma2_sq+C2)

        ssim_map = np.divide(upper_part, down_part)

        ssim_mean = np.mean(ssim_map)

        return ssim_mean

    def ifc(self):
        return



def print_progress(type_name, current_progress, total_progress, percentage=True):

    
    if percentage == True:
        progress_pr = int(current_progress/total_progress*20)
        sys.stdout.write(type_name +" Progress: " + str(current_progress) + '/' + str(total_progress)+ ' |'+'#'*progress_pr + '-'*(20-progress_pr) + '\r')
    else:
        sys.stdout.write(type_name + " Progress: " + str(current_progress) + '/' + str(total_progress)+ ' |'+'#'*current_progress + '-'*(total_progress-current_progress) + '\r')
    
    
    if current_progress == total_progress:sys.stdout.write('\n')
    else: sys.stdout.flush()    

class evaluation:

    def __init__(self, data_root, dataset_list, scale_list, subimg_size, padding_size, channel = 1):

        """
            data_root: root path for all datasets in dataset list
            dataset_list: datasets for evaluation
            scale_list: image's scales for evaluation
        """
        self.data_root = data_root
        self.dataset_list = dataset_list
        self.scale_list = scale_list
        self.becnchmark = becnchmark()
        self.subimg_size = subimg_size
        self.padding_size = padding_size
        self.inputs = tf.placeholder(tf.float32, [	None,
        											subimg_size[0]+padding_size[0]*2, 
        											subimg_size[1]+padding_size[1]*2,
        											channel])

    def dataset_setup(self):
        """
        set up dataset for evaluation
        return : dictionary of data path dictionary[dataset][scale][img_id] = { HR: High resolution image path,
                                                                                LR:    Low resolution image path            
                                                                                }
        """

        datasets = self.dataset_list
        scales = self.scale_list
        self.dataset_pair = {}

        for d in datasets:

            self.dataset_pair[d] = {}

            for scale in scales:

                self.dataset_pair[d][scale] = {}

                dataset_path = os.path.join(self.data_root, d, 'preprocessed_scale_'+str(scale))
                
                filelist = sorted(os.listdir(dataset_path))

                assert len(filelist)%3 == 0, "Some data is missing"
                
                for i in range(len(filelist)//3):
                    LR_fname = filelist[i*3]
                    Small_LR_fname = filelist[i*3 + 1]
                    HR_fname = filelist[i*3 + 2]
                    self.dataset_pair[d][scale][LR_fname.split("_")[0]] = {
                                                                            'HR':os.path.join(dataset_path, HR_fname),
                                                                            'LR':os.path.join(dataset_path, LR_fname),
                                                                            'small':os.path.join(dataset_path, Small_LR_fname)
                                                                        }

        return self.dataset_pair

    def input_setup(self, input_path, target_path, padding_size = [3,3], subimg_size = [30,30], scale = 2):

    
        inputs = imread(input_path)
        targets = imread(target_path)


        _, sout = split_img("input",inputs, padding_size, subimg_size)

        grid_out = []
        grid_out_key = []
        for k in sout:
            grid_out_key.append(k)
            grid_out.append(sout[k])
        grid_stack = np.stack(grid_out)

        padding_size = [padding_size[0]*scale, padding_size[1]*scale]
        subimg_size = [subimg_size[0]*scale, subimg_size[1]*scale]
        padded_size, target_split = split_img("target",targets, padding_size, subimg_size)
        target_merge = merge_img(padded_size, target_split, padding_size, subimg_size, 1)

        input_pair = {
                        'inputs':grid_stack,
                        'input_key': grid_out_key,
                        'target': target_merge
                    }

        return input_pair

    def load_model(self, model_ticket, ckpt_file):

        mz = model_zoo.model_zoo(self.inputs, None, False, model_ticket)    
        model_prediction = mz.build_model()
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)

        return sess, model_prediction

    def prediction(self, image, sess,model_prediction):

        """
        run model in model_ticket_list and return prediction
        """
            
        predicted,_,_,_= sess.run(model_prediction, feed_dict = {self.inputs:image})
            

        return predicted[2]



    def run_evaluation(self, benchmark_type = ["bicubic", "model"], model_dict = {}):

        """
        Run Evaluation and return benchmark value in benchmark_type
        """
        
        dataset_pair = self.dataset_setup()
        becnchmark = self.becnchmark
        bechmark_val = {}
        padding_size = self.padding_size
        subimg_size = self.subimg_size

        for set_key in dataset_pair:
            dataset = dataset_pair[set_key]
            bechmark_val[set_key] = {}

            sys.stdout.write("Evaluating dataset: " + set_key + '\n')
            
            for scale_key in dataset:
                dataset_scale = dataset[scale_key]
                bechmark_val[set_key][scale_key] = {}

                sys.stdout.write("Evaluating scales: " + str(scale_key) + '\n')

                progress = 0

                for input_key in dataset_scale:

                    progress = progress + 1


                    if "bicubic" in benchmark_type:

                        if "bicubic" not in bechmark_val[set_key][scale_key]: bechmark_val[set_key][scale_key]["bicubic"] = {}
                        bechmark_val[set_key][scale_key]["bicubic"][input_key] = {}

                        scale =  int(scale_key)
                        HR_img = dataset_scale[input_key]['HR']
                        LR_img = dataset_scale[input_key]['small']
                        input_pair = self.input_setup(LR_img, HR_img, padding_size = padding_size, subimg_size = subimg_size, scale = scale)


                        test_input = input_pair['inputs']
                        key =  input_pair['input_key']

                        output_stack = {}

                        up_padding_size = [padding_size[0]*scale, padding_size[1]*scale]
                        up_subimg_size = [subimg_size[0]*scale, subimg_size[1]*scale]
                
                        for l in range(len(test_input)):
                            resize_img = resize(test_input[l], (up_subimg_size[1]+2*up_padding_size[1],up_subimg_size[0]+2*up_padding_size[0]),  interpolation = cv2.INTER_CUBIC)
                            output_stack[key[l]] = resize_img

                        bicubic_output = merge_img(input_pair['target'].shape, output_stack, up_padding_size,up_subimg_size, scale)

                        results = becnchmark.run(np.expand_dims(bicubic_output[:,:,0], axis=2), np.expand_dims(input_pair['target'][:,:,0],axis=2))
                        bechmark_val[set_key][scale_key]["bicubic"][input_key]["psnr"] = results[0]
                        bechmark_val[set_key][scale_key]["bicubic"][input_key]["SSIM"] = results[1]


                        print_progress("bicubic", progress, len(dataset_scale))
                        

                    if "model" in benchmark_type and len(model_dict) > 0:


                            for mkey in model_dict:

                                if mkey not in bechmark_val[set_key][scale_key]: bechmark_val[set_key][scale_key][mkey] = {}
                                bechmark_val[set_key][scale_key][mkey][input_key] = {}

                                if model_dict[mkey]["upsample"] == False:
                                    HR_img = dataset_scale[input_key]['HR']
                                    LR_img = dataset_scale[input_key]['LR']
                                    scale = 1  
                                else:
                                    HR_img = dataset_scale[input_key]['HR']
                                    LR_img = dataset_scale[input_key]['small']
                                    scale =  int(scale_key)

                            input_pair = self.input_setup(LR_img, HR_img, padding_size = padding_size, subimg_size = subimg_size, scale = scale)
                            test_input = input_pair['inputs']
                            key =  input_pair['input_key']

        
                            output_stack = {}

                            up_padding_size = [padding_size[0]*scale, padding_size[1]*scale]
                            up_subimg_size = [subimg_size[0]*scale, subimg_size[1]*scale]

                            sess, model_prediction = self.load_model(mkey,model_dict[mkey]["ckpt_file"])

                            for l in range(len(test_input)):
                                if model_dict[mkey]["isGray"] == True: 
                                    testimg = np.expand_dims(test_input[l][:,:,0], axis = 2)
                                    if l==0: targetimg = np.expand_dims(input_pair['target'][:,:,0], axis=2)
                                else:
                                    testimg = test_input[l]
                                    targetimg = input_pair['target']

                                if model_dict[mkey]["isNormallized"] == True: 
                                    testimg = testimg/255.
                                    if l==0: targetimg = targetimg/255.
                                
                                m_out = self.prediction([testimg], sess, model_prediction)                    
                                output_stack[key[l]] = np.squeeze(m_out,axis=0)
                                
                            model_out = merge_img(targetimg.shape, output_stack, up_padding_size,up_subimg_size, scale, down_scale_by_model=True)                            
                            
                            if model_dict[mkey]["isNormallized"] == True: 
                                model_out = model_out*255
                                targetimg = targetimg*255


                            results = becnchmark.run(model_out, targetimg)
                            bechmark_val[set_key][scale_key][mkey][input_key]["psnr"] = results[0]
                            bechmark_val[set_key][scale_key][mkey][input_key]["SSIM"] = results[1]

                            print_progress(mkey, progress, len(dataset_scale))
                            
                            sess.close()


        return bechmark_val


def summary_result(filename, results):

	with open(filename, 'w') as f:

	   for i in range(len(results)):
	       
	       
	       
	       for setk in results[i]:
	            for scale_key in results[i][setk]:
	                for mkey in results[i][setk][scale_key]:
	                    
	                    mPSNR = 0
	                    mSSIM = 0
	                    
	                    print("Dataset: {}, scale: {}, model: {}".format(setk, scale_key, mkey),file=f)
	                    print("{:<15} {:<20} {:<30}".format("", "PSNR","SSIM"),file=f)
	                    
	                    for input_key in results[i][setk][scale_key][mkey]:    
	                        print("{:<15} {:<20} {:<30}".format( input_key, 
	                                                              results[i][setk][scale_key][mkey][input_key]["psnr"], 
	                                                              results[i][setk][scale_key][mkey][input_key]["SSIM"]), file=f)
	                        
	                        mPSNR = mPSNR + results[i][setk][scale_key][mkey][input_key]["psnr"]
	                        mSSIM = mSSIM + results[i][setk][scale_key][mkey][input_key]["SSIM"]
	                    
	                    mPSNR = mPSNR/len(results[i][setk][scale_key][mkey])
	                    mSSIM = mSSIM/len(results[i][setk][scale_key][mkey])
	                    
	                    print("="*60,file=f)
	                   
	                    print("mPSNR: {}, mSSIM: {}".format(mPSNR, mSSIM),file=f)
    

	   f.close()

def main_process():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="example",help="Configuration name")
    args = parser.parse_args()

    eval_result = []

    conf = config.config(args.config).config

    
    for midx in range(len(conf["evaluation"]["models"])):

        mkeys = list(conf["evaluation"]["models"][midx])[0]

        eval = evaluation(conf["evaluation"]["dataroot"], 
                          conf["evaluation"]["test_set"], 
                          conf["evaluation"]["models"][midx][mkeys]["scale"],
                          conf["evaluation"]["models"][midx][mkeys]["subimages"], 
                          conf["evaluation"]["models"][midx][mkeys]["padding"])


        results = eval.run_evaluation(benchmark_type = ['model'], model_dict = conf["evaluation"]["models"][midx])
        eval_result.append(results)
    
    results = eval.run_evaluation(benchmark_type = ['bicubic'])
    eval_result.append(results)
    summary_result(conf["evaluation"]["summary_file"], eval_result)

    return eval_result

if __name__ == '__main__':

    print("Start Evaluation")
    eval_result = main_process()
    print("Done Evaluation")
   