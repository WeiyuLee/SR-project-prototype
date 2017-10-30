
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
		train_config["epoch"] = 20000  # Number of epoch [10]
		train_config["batch_size"] = 10 # The size of batch images [128]
		train_config["image_size"] = 48 # The size of image to use [33]
		train_config["label_size"] = 192 # The size of label to produce [21]
		train_config["learning_rate"] = 1e-4 #The learning rate of gradient descent algorithm [1e-4]
		train_config["color_dim"] = 3 # Dimension of image color. [1]
		train_config["scale"] = 4 # The size of scale factor for preprocessing input image [3]
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
		train_config["ckpt_name"] = "EDSR_v2_s4" # Name of checkpoints
		train_config["is_train"] = True # True for training, False for testing [True]
		train_config["model_ticket"] = "edsr_v1" # Name of checkpoints

		def edsrv1(self):
						
			mconfig = {}
			mconfig["edsr_v1"] = {

										"scale":[4],
										"subimages":[43,43],
										"padding":[4,4],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/EDSR_new_v1/model/edsr_0_s4/edsr_0_s4-81991",
										"isGray": True,
										"isNormallized":False,
										"upsample": True,
										"sub_mean":True,
										"model_config" :{"scale":4,"feature_size" : 64}
										}
			return mconfig


		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = './preprocess/Test'
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

										"scale":[4],
										"subimages":[128,128],
										"padding":[4,4],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/EDSR_new_v1/model/edsr_0_s4/edsr_0_s4-81991",
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




