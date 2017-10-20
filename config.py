
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


		def srcnn(self):
						
			mconfig = {}
			mconfig["grr_grid_srcnn_v1"] = {

										"scale":[1],
										"subimages":[20,20],
										"padding":[6,6],
										"ckpt_file":"/home/ubuntu/model/model/SR_project/SRCNN/SRCNN.model-309672",
										"isGray": True,
										"isNormallized":True
										}
			return mconfig

		train_config = self.config["train"]

		train_config["mode"] = "normal"
		train_config["epoch"] = 10
		train_config["batch_size"] = 128
		train_config["image_size"] = 32
		train_config["label_size"] = 20
		train_config["learning_rate"] = 1e-4
		train_config["color_dim"] = 1
		train_config["scale"] = 4
		train_config["train_extract_stride"] = 14
		train_config["test_extract_stride"] = train_config["label_size"]
		train_config["test_extract_stride"]
		train_config["checkpoint_dir"] = "checkpoint"
		train_config["output_dir"] = "output"
		train_config["train_dir"] =  "Train"
		train_config["test_dir"] = "Test/Set5"
		train_config["h5_dir"] = "preprocess/output"
		train_config["train_h5_name"] = "train"
		train_config["test_h5_name"] = "test"
		train_config["ckpt_name"] = ""
		train_config["is_train"] = True
		train_config["model_ticket"] = "grr_grid_srcnn_v1" 




		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [srcnn(self)]

		

