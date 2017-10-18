import cv2
import numpy as np
import os
import tensorflow as tf
import sys
import scipy.misc
sys.path.append('./utility')
import utils as ut
import model_zoo  



class becnchmark:

	def run(self, inputs, target, psnr_mode='YCbCr'):
		self.input = inputs
		self.target = target

		psnr = self._psnr(mode = psnr_mode)
		ssim = self._ssim()

		return psnr, ssim


	def _psnr(self, mode='YCbCr'):

		#input_uint8 = cv2.convertScaleAbs(self.input)
		#target_uint8 = cv2.convertScaleAbs(self.target)

		input_uint8 = self.input
		target_uint8 = self.target

		

		if mode == 'YCbCr':
			inputs = input_uint8
			target = target_uint8

		else:
			inputs = input_uint8
			target = target_uint8

		mse = np.square(inputs - target)
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



		blur_input = np.expand_dims(cv2.GaussianBlur( _input, guassian_kernel_size,std), axis=2)[clipping[0]:clipping[1], clipping[2]:clipping[3],:]
		blur_target = np.expand_dims(cv2.GaussianBlur( _target, guassian_kernel_size,std), axis=2)[clipping[0]:clipping[1], clipping[2]:clipping[3],:]
		
		
		mu1_sq = np.multiply(blur_input, blur_input)
		mu2_sq = np.multiply(blur_target, blur_target)
		mu1_mu2 = np.multiply(blur_input, blur_target)

		sigma1_sq = np.subtract(np.expand_dims(cv2.GaussianBlur(np.multiply(_input,_input),guassian_kernel_size,std), axis=2)[clipping[0]:clipping[1], clipping[2]:clipping[3],:], mu1_sq)
		sigma2_sq = np.subtract(np.expand_dims(cv2.GaussianBlur(np.multiply(_target,_target),guassian_kernel_size,std), axis=2)[clipping[0]:clipping[1], clipping[2]:clipping[3],:], mu2_sq)
		sigma12 = np.subtract(np.expand_dims(cv2.GaussianBlur(np.multiply(_input,_target),guassian_kernel_size,std), axis=2)[clipping[0]:clipping[1], clipping[2]:clipping[3],:], mu1_mu2)

		
		upper_part = np.multiply(2*mu1_mu2 + C1, 2*sigma12+C2)
		down_part =  np.multiply(mu1_sq+mu2_sq+C1, sigma1_sq+sigma2_sq+C2)

		ssim_map = np.divide(upper_part, down_part)

		ssim_mean = np.mean(ssim_map)

		return ssim_mean

	def ifc(self):
		return

def imread(data_path, is_grayscale=True):

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
	padded_size = [	ori_size[0] - 2*padding_size[0],
					ori_size[1] - 2*padding_size[1],
					ori_size[2]]

	strides = subimg_size
	sub_imgs = {}

	for r in range(padded_size[0]//subimg_size[0]):
		for c in range(padded_size[1]//subimg_size[1]):

			grid_r = padding_size[0] + r*strides[0] 
			grid_c = padding_size[1] + c*strides[1] 

			sub_img = img[	grid_r - padding_size[0] : grid_r + strides[0] + padding_size[0],
							grid_c - padding_size[1] : grid_c + strides[1] + padding_size[1],
							:]

			# insert sub image to dictionary with key = [imagename]_[row_index]_[col_index]
			sub_imgs[imgname + "_"+ str(grid_r) + "_" + str(grid_c)] = sub_img


	return sub_imgs



def merge_img(img_size, sub_images, padding_size,subimg_size, scale=2, down_scale_by_model=False):

	# Create an empty array for merging image

	padded_size = [	img_size[0] - 2*padding_size[0],
					img_size[1] - 2*padding_size[1],
					img_size[2]]

	
	merged_image = np.zeros([	(padded_size[0]//subimg_size[0])*subimg_size[0],
								(padded_size[1]//subimg_size[1])*subimg_size[1],
								 padded_size[2]])
	for k in sub_images:

		key = k.split("_")

		grid_r = int(key[1])*scale - padding_size[0]
		grid_c = int(key[2])*scale - padding_size[1]

		#print(k, grid_r, grid_c,  merged_image.shape,sub_images[k].shape,padding_size, subimg_size[1])
		
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


def input_setup(input_path, target_path, padding_size = [3,3], subimg_size = [30,30], scale = 2):

	
	inputs = imread(input_path,True)
	targets = imread(target_path,True)
	
	sout = split_img("input",inputs, padding_size, subimg_size)

	grid_out = []
	grid_out_key = []
	for k in sout:
		grid_out_key.append(k)
		grid_out.append(sout[k])
	grid_stack = np.stack(grid_out)

	padding_size = [padding_size[0]*scale, padding_size[1]*scale]
	subimg_size = [subimg_size[0]*scale, subimg_size[1]*scale]
	target_split = split_img("target",targets, padding_size, subimg_size)
	target_merge = merge_img(targets.shape, target_split, padding_size, subimg_size, 1)

	input_pair = {
					'inputs':grid_stack,
					'input_key': grid_out_key,
					'target': target_merge
				}

	return input_pair


def input_setup_test(padding_size = [3,3], subimg_size = [30,30], scale = 2):

	small_out = {}

	img_path = '/home/ubuntu/dataset/SuperResolution/Set5/image_SRF_1/img_002_SRF_1_HR.bmp'
	img_path2 = '/home/ubuntu/dataset/SuperResolution/Set5/image_SRF_1/img_002_SRF_1_LR.bmp'
	targets = cv2.imread(img_path)
	inputs = cv2.imread(img_path2)

	sout = split_img("input",inputs, padding_size, subimg_size)
	padding_size = [padding_size[0]*scale, padding_size[1]*scale]
	
	for k in sout:

		small_out[k] = cv2.resize(sout[k], (subimg_size[1]*scale+2*padding_size[1],subimg_size[0]*scale+2*padding_size[0]))


	subimg_size = [subimg_size[0]*scale, subimg_size[1]*scale]
	target_out = split_img("target",targets, padding_size, subimg_size)
	

	ms = merge_img(targets.shape, small_out, padding_size,subimg_size,scale)
	ts = merge_img(targets.shape, target_out, padding_size, subimg_size, 1)
	
	cv2.imwrite("small.jpg", ms)
	cv2.imwrite("target.jpg", ts)



def dataset_setup(data_root = '/home/ubuntu/dataset/SuperResolution', dataset = ["Set5"], scales = [2,3,4]):

	dataset_pair = {}

	for d in dataset:

		dataset_pair[d] = {}

		for scale in scales:

			dataset_pair[d][scale] = {}

			dataset_path = os.path.join(data_root, d, 'image_SRF_'+str(scale))
			Ntestfiles = len(os.listdir(dataset_path))//2

			for i in range(Ntestfiles):
				HR_fname = "img_" + '{0:03}'.format(i+1) + '_SRF_' + str(scale) + '_HR.png'
				LR_fname = "img_" + '{0:03}'.format(i+1) + '_SRF_' + str(scale) + '_LR.png'

				dataset_pair[d][scale]['{0:03}'.format(i+1)] = {
														'HR':os.path.join(dataset_path, HR_fname),
														'LR':os.path.join(dataset_path, LR_fname),
														}

	return dataset_pair



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
		self.inputs = tf.placeholder(tf.float32, [None, 32, 32, 1])

	def dataset_setup(self):
		"""
		set up dataset for evaluation
		return : dictionary of data path dictionary[dataset][scale][img_id] = { HR: High resolution image path,
																				LR:	Low resolution image path			
																				}
		"""

		datasets = self.dataset_list
		scales = self.scale_list
		self.dataset_pair = {}

		for d in datasets:

			self.dataset_pair[d] = {}

			for scale in scales:

				self.dataset_pair[d][scale] = {}

				dataset_path = os.path.join(self.data_root, d, 'image_SRF_'+str(scale))
				Ntestfiles = len(os.listdir(dataset_path))//2

				for i in range(Ntestfiles):
					HR_fname = "img_" + '{0:03}'.format(i+1) + '_SRF_' + str(scale) + '_HR.bmp'
					LR_fname = "img_" + '{0:03}'.format(i+1) + '_SRF_' + str(scale) + '_LR.bmp'

					self.dataset_pair[d][scale]['{0:03}'.format(i+1)] = {
																			'HR':os.path.join(dataset_path, HR_fname),
																			'LR':os.path.join(dataset_path, LR_fname),
																		}

		return self.dataset_pair

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

	def run_evaluation(self, benchmark_type = ["bicubic", "model"], model_ticket_list = [], ckpt_file_list = []):

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


					bechmark_val[set_key][scale_key][input_key] = {}

					scale =  int(scale_key)
					HR_img = dataset_scale[input_key]['HR']
					LR_img = dataset_scale[input_key]['LR']

					input_pair = input_setup(LR_img, HR_img, padding_size = padding_size, subimg_size = subimg_size, scale = scale)

					test_input = input_pair['inputs']
					key =  input_pair['input_key']

					if "bicubic" in benchmark_type:
						
						bechmark_val[set_key][scale_key][input_key]["bicubic"] = {}
						output_stack = {}

						up_padding_size = [padding_size[0]*scale, padding_size[1]*scale]
						up_subimg_size = [subimg_size[0]*scale, subimg_size[1]*scale]
				
						for l in range(len(test_input)):
							tmp = cv2.resize(test_input[l], (up_subimg_size[1]+2*up_padding_size[1],up_subimg_size[0]+2*up_padding_size[0]),  interpolation = cv2.INTER_CUBIC)
							tmp = np.expand_dims(tmp, axis = 2)
							output_stack[key[l]] = tmp

						bicubic_output = merge_img(imread(HR_img).shape, output_stack, up_padding_size,up_subimg_size, scale)

						progress_pr = int((progress/len(dataset_scale)*10)) 
						sys.stdout.write("Progress: " + str(progress) + '/' + str(len(dataset_scale))+ ' |'+'#'*progress_pr + '-'*(10-progress_pr) + '\r')
						if (progress) == len(dataset_scale):sys.stdout.write('\n')
						else: sys.stdout.flush()	

						results = becnchmark.run(bicubic_output, input_pair['target'])
						bechmark_val[set_key][scale_key][input_key]["bicubic"]["psnr"] = results[0]
						bechmark_val[set_key][scale_key][input_key]["bicubic"]["SSIM"] = results[1]

						print(results[0])

					if "model" in benchmark_type and len(model_ticket_list) > 0:
						bechmark_val[set_key][scale_key][input_key]["srcnn"] = {}
						output_stack = {}
						for midx in range (len(model_ticket_list)):

							sess, model_prediction = self.load_model(model_ticket_list[midx],ckpt_file_list[midx])

							for l in range(len(test_input)):
								testimg = np.expand_dims(test_input[l], axis=0) 
								testimg = testimg/255.

								m_out = self.prediction(testimg, sess, model_prediction)					
								output_stack[key[l]] = np.squeeze(m_out,axis=0)
								

							model_out = merge_img(imread(HR_img).shape, output_stack, up_padding_size,up_subimg_size, scale, down_scale_by_model=True)
							
							#model_out = np.squeeze(model_out)
							
							results = becnchmark.run(model_out*255, input_pair['target'])
							bechmark_val[set_key][scale_key][input_key]["srcnn"]["psnr"] = results[0]
							bechmark_val[set_key][scale_key][input_key]["srcnn"]["SSIM"] = results[1]
							
							print(results[0])
							
							image = model_out*255
							output_image = scipy.misc.toimage(np.squeeze(image), high=np.max(image), low=np.min(image), mode='L')
							output_image.save("small.jpg")
							
							cv2.imwrite("target.jpg", bicubic_output)
							

							sess.close()


		return bechmark_val

eval = evaluation('/home/ubuntu/dataset/SuperResolution', ["Set5"], [1], [20,20], [6,6])
results = eval.run_evaluation(model_ticket_list = ["grr_grid_srcnn_v1"], ckpt_file_list = ["/home/ubuntu/model/model/SR_project/SRCNN/SRCNN.model-309672"])
print(results)