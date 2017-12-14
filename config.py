class config:

	def __init__(self, configuration):
		
		self.configuration = configuration
		self.config = {
						"common":{},
						"train":{},
						"evaluation":{
								"dataroot":None,
								"test_set":["Set5", "Set14", "BSD100"],
								"models":{},
								
							}
						}
		self.get_config()


	def get_config(self):

		try:
			conf = getattr(self, self.configuration)
			conf()

		except: 
			print("Can not find configuration")
			raise
			
			flags.DEFINE_string("mode", "normal", "operation mode: normal or freq [normal]")


	def example(self):

		train_config = self.config["train"]

		train_config["mode"] = "normal" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 10000  # Number of epoch [10000]
		train_config["batch_size"] = 128 # The size of batch images [128]
		train_config["image_size"] = 32 # The size of image to use [33]
		train_config["label_size"] = 20 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 1 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["test_extract_stride"] # The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "checkpoint" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "log"
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "preprocess/output" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "grr_grid_srcnn_v1" # Name of checkpoints

		def srcnn(self):
						
			mconfig = {}
			mconfig["grr_grid_srcnn_v1"] = {

										"scale":[4],
										"subimages":[20,20],
										"padding":[6,6],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/SRCNN/SRCNN.model-309672",
										"isGray": True,
										"isNormallized":True,
										"upsample": False
										}
			return mconfig


		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = './preprocess/Test'
		eval_config["test_set"] = ["Set5", "Set14", "BSD100"]
		eval_config["models"] = [srcnn(self)]
		eval_config["summary_file"] = "example_summary.txt"

	def jesse_edsr(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 80000  # Number of epoch [10]
		train_config["batch_size"] = 1 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "edsrv2_dual_single" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_v2_dual" # Name of checkpoints

		def edsrv1(self):
						
			mconfig = {}
			mconfig["edsr_v2_dual"] = {

										"scale":[2],
										"subimages":[76,76],
										"padding":[8,8],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/edsrv2_single/edsrv2_single-79951",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"scale":2,"feature_size" : 64}
										}
			return mconfig


		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsrv1(self)]
		eval_config["summary_file"] = "example_summary.txt"


	def jesse_espcn(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 20000  # Number of epoch [10]
		train_config["batch_size"] = 32 # The size of batch images [128]
		train_config["image_size"] = 17 # The size of image to use [33]
		train_config["label_size"] = 68 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 1 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/EDSR_new_v1/model/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/EDSR_new_v1/model/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "espcn_s4" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "espcn_v1" # Name of checkpoints

		def espcn_v1(self):
						
			mconfig = {}
			mconfig["espcn_v1"] = {

										"scale":[2],
										"subimages":[96,96],
										"padding":[16,16],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/EDSR_new_v1/model/EDSR_base_1X1_v1_s2/EDSR_base_1X1_v1_s2-529256",
										"isGray": True,
										"isNormallized":False,
										"upsample": True,
										"sub_mean":False,
										"model_config" :{}
										}
			return mconfig


		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = './preprocess/Test'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [espcn_v1(self)]
		eval_config["summary_file"] = "example_summary.txt"


	def jesse_edsr_att(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 20000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 36 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "edsr_base_attention_v2_p36l30" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_attention_v1" # Name of checkpoints

		def edsr_attention_v1(self):
						
			mconfig = {}
			
			mconfig["edsr_attention_v1"] = {

										"scale":[2],
										"subimages":[96,96],
										"padding":[16,16],
										#"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2_oh/edsr_base_attention_v2_oh-719656",
										"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2/edsr_base_attention_v2-1117256",
										"isGray": False,
										"isNormallized":True,
										"upsample": True,
										"sub_mean":False,
										"model_config" :{"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False}
										}
			
			
			return mconfig

		def edsr_v2(self):
			mconfig = {}
			mconfig["edsr_v2"] = {

										"scale":[2],
										"subimages":[80,80],
										"padding":[8,8],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/EDSR_base_v3_s2/EDSR_base_v3_s4-559384",
										#"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2_24/edsr_base_attention_v2_24-1117256",
										"isGray": False,
										"isNormallized":True,
										"upsample": True,
										"sub_mean":False,
										"model_config" :{"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False}
										}

			return mconfig

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsr_attention_v1(self)]
		eval_config["summary_file"] = "example_summary.txt"


	def jesse_edsr_att1x1(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 20000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "edsr_base_attention3x3_v2_l15" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_1X1_v1" # Name of checkpoints

		def edsr_attention_v1(self):
						
			mconfig = {}
			
			mconfig["edsr_1X1_v1"] = {

										"scale":[2],
										"subimages":[80,80],
										"padding":[8,8],
										#"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2_oh/edsr_base_attention_v2_oh-719656",
										"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention3x3_v2_l15/edsr_local_att_v1-411656",
										"isGray": False,
										"isNormallized":True,
										"upsample": True,
										"sub_mean":False,
										"model_config" :{"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False}
										}
			
			
			return mconfig

		

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsr_attention_v1(self)]
		eval_config["summary_file"] = "example_summary.txt"




	def edsr_local_att_v2_upsample(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 20000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "edsr_local_att_v5_upsample" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_local_att_v2_upsample" # Name of checkpoints

		#V2 "model_config" :{"scale":2,"feature_size" :64, "kernel_size":1}
		#V3 "model_config" :{"scale":2,"feature_size" :64, "kernel_size":3}
		#V4 "model_config" :{"scale":2,"feature_size" :64, "kernel_size":3} add unknown dataset
		#V5 "model_config" :{"scale":2,"feature_size" :64, "kernel_size":3} 
							#add unknown dataset and regional_l1loss with portion 6

		def edsrv1(self):
						
			mconfig = {}
			mconfig["edsr_local_att_v2_upsample"] = {

										"scale":[2],
										"subimages":[80,80],
										"padding":[8,8],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_local_att_v5_upsample/edsr_local_att_v5_upsample-638512",
										"isGray": False,
										"isNormallized":True,
										"upsample": True,
										"sub_mean":False,
										"model_config" :{"scale":2,"feature_size" :64, "kernel_size":3}
										}
			return mconfig


		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsrv1(self)]
		eval_config["summary_file"] = "example_summary.txt"

	def jesse_edsr_local_att_v1(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 20000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "edsr_local_att_v1_1x1" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_local_att_v1" # Name of checkpoints

		def edsrv1(self):
						
			mconfig = {}
			mconfig["edsr_local_att_v1"] = {

										"scale":[2],
										"subimages":[80,80],
										"padding":[8,8],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_local_att_v1_1x1/edsr_local_att_v1_1x1-394856",
										"isGray": False,
										"isNormallized":True,
										"upsample": True,
										"sub_mean":False,
										"model_config" :{"scale":2,"feature_size" : 64}
										}
			return mconfig


		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsrv1(self)]
		eval_config["summary_file"] = "example_summary.txt"

	def jesse_edsr_att_v2(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 20000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 24 # The size of image to use [33]
		train_config["label_size"] = 48 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "edsr_attention_v2_local_loss" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_attention_v2" # Name of checkpoints

		def edsr_attention_v2(self):
						
			mconfig = {}
			
			mconfig["edsr_attention_v2"] = {

										"scale":[2],
										"subimages":[96,96],
										"padding":[16,16],
										#"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2_oh/edsr_base_attention_v2_oh-719656",
										"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2/edsr_base_attention_v2-1117256",
										"isGray": False,
										"isNormallized":True,
										"upsample": True,
										"sub_mean":False,
										"model_config" :{"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False}
										}
			
			
			return mconfig

		
		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsr_attention_v2(self)]
		eval_config["summary_file"] = "example_summary.txt"


	def jesse_edsr_lsgan(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 100000  # Number of epoch [10]
		train_config["batch_size"] = 16 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "edsr_ls_gan_res4" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_lsgan" # Name of checkpoints

		def edsr_lsgan(self):
						
			mconfig = {}
			
			mconfig["edsr_lsgan"] = {

										"scale":[2],
										"subimages":[80,80],
										"padding":[8,8],
										#"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2_oh/edsr_base_attention_v2_oh-719656",
										"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_ls_gan/edsr_ls_gan-694512",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None,"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False, "reuse":False}
										}
			
			
			return mconfig

		

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsr_lsgan(self)]
		eval_config["summary_file"] = "example_summary.txt"

	def edsr_lsgan_up(self):

		train_config = self.config["train"]

		train_config["mode"] = "small" # Operation mode: normal or freq [normal]
		train_config["epoch"] = 100000  # Number of epoch [10]
		train_config["batch_size"] = 1 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 96 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 2 # The size of scale factor for preprocessing input image [3]
		train_config["train_extract_stride"] = 14 #The size of stride to apply input image [14]
		train_config["test_extract_stride"] = train_config["label_size"] #The size of stride to apply input image [14]
		train_config["checkpoint_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["log_dir"] = "/home/ubuntu/model/model/SR_project/" #Name of checkpoint directory [checkpoint]
		train_config["output_dir"] = "output" # Name of sample directory [output]
		train_config["train_dir"] =  "Train" # Name of train dataset directory
		train_config["test_dir"] = "Test/Set5" # Name of test dataset directory [Test/Set5]
		train_config["h5_dir"] = "/home/ubuntu/dataset/SuperResolution/train" # Name of train dataset .h5 file
		train_config["train_h5_name"] = "train" # Name of train dataset .h5 file
		train_config["test_h5_name"] = "test" # Name of test dataset .h5 file
		train_config["ckpt_name"] = "edsr_lsgan_up_aux_v2" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_lsgan_up" # Name of checkpoints

		def edsr_lsgan_up(self):
						
			mconfig = {}
			
			mconfig["edsr_lsgan_up"] = {

										"scale":[2],
										"subimages":[80,80],
										"padding":[8,8],
										#"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_base_attention_v2_oh/edsr_base_attention_v2_oh-719656",
										"ckpt_file":"/home/ubuntu/model/model/SR_project/edsr_lsgan_up_aux/edsr_lsgan_up_aux-64952",
										"isGray": False,
										"isNormallized":True,
										"upsample": False,
										"sub_mean":False,
										"model_config" :{"d_inputs":None, "d_target":None,"scale":2,"feature_size" : 64,"dropout" : 1.0,"feature_size" : 64, "is_training":False, "reuse":False}
										}
			
			
			return mconfig

		

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution/'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [edsr_lsgan_up(self)]
		eval_config["summary_file"] = "example_summary.txt"