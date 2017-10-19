
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

		eval_config = self.config["evaluation"]
		eval_config["dataroot"] = '/home/ubuntu/dataset/SuperResolution'
		eval_config["test_set"] = ["Set5"]
		eval_config["models"] = [srcnn(self)]

		

