from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger
from Utils.utilsFunctions import pickAPlatform, acceleratorWarning

class PlatformContext():

    def __init__(self):
        """
        Initialize the Device Context of the system, creates the concrete strategies
        """
        self.__packageDownloadManager = None
        self.__configurationManager = None
        self.__runnerModule = None
        self.__statsModule = None
        self.__platform = pickAPlatform()

        match self.__platform:

            case "generic":

                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerGeneric
                from ConfigurationModule.configurationManager import ConfigManagerGeneric
                
                self.__configurationManager = ConfigManagerGeneric(self.__platform)
                self.__packageDownloadManager = PackageDownloadManagerGeneric()
                # self.__runnerModule = RunnerModuleGeneric()
                # self.__statsModule = StatsModuleGeneric()

            case "coral":
                
                # --- Coral Imports ---
                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerCoral
                from ConfigurationModule.configurationManager import ConfigManagerCoral

                acceleratorWarning()
                             
                self.__configurationManager = ConfigManagerCoral(self.__platform)
                self.__packageDownloadManager = PackageDownloadManagerCoral()
                # self.__runnerModule = RunnerModuleCoral()
                # self.__statsModule = StatsModuleCoral() 
                
            case "fusion":


                # --- Fusion Imports ---
                acceleratorWarning()
                

                from PackageDownloadModule.packageDownloadManager import PackageDownloadManagerFusion
                from ConfigurationModule.configurationManager import ConfigManagerFusion


                self.__configurationManager = ConfigManagerFusion(self.__platform)
                self.__packageDownloadManager = PackageDownloadManagerFusion()
                # self.__runnerModule = RunnerModuleFusion()
                # self.__statsModule = StatsModuleFusion()

    def test(self):

        config = {
            "models": [
                {
                    "module": "torchvision.models",
                    "model_name": "mobilenet_v2",
                    "native": False,
                    "weights_path": "ModelData/Weights/mobilenet_v2_best.pth",
                    "device": "cpu",
                    "class_name": "mobilenet_v2",
                    "weights_class": "MobileNet_V2_Weights.DEFAULT", 
                    "image_size": 224, 
                    "num_classes": 2,
                    "task": "classification",
                    "description": "Mobilenet V2 from torchvision"
                }, 
                {
                "module": "torchvision.models",
                "model_name": "efficientnet", 
                "native": False,
                "weights_path": "./ModelData/Weights/efficientnet_b0_best.pth",
                "device": "cpu",
                "class_name": "efficientnet_b0",
                "weights_class": "EfficientNet_B0_Weights.DEFAULT", 
                "image_size": 224,
                "num_classes": 2,
                "task": "classification",
                "description": "EfficientNet from torchvision"
                },
                {
                    'module': 'torchvision.models',
                    'model_name': "mnasnet1_0",
                    'native': False,
                    'weights_path': "ModelData/Weights/mnasnet1_0_best.pth",
                    'device': "cpu",
                    'class_name': 'mnasnet1_0',
                    'weights_class': 'MNASNet1_0_Weights.DEFAULT',
                    'image_size': 224,
                    'num_classes': 2,
                    "task": "classification",
                    'description': 'mnasnet_v2 from torchvision'
                }
            ],
            "optimizations": {
                "Pruning": {
                    "method": "LnStructured", 
                    "amount": 0.15,
                    "n": 1
                },
                "Distillation":{
                    "method": True,
                    "distilled_paths": {}
                }
            },
            "dataset": {
                "data_dir": "ModelData/Dataset/casting_data",
                "batch_size": 32
            },
            "repetitions": 2

        }



        hash_val = self.__configurationManager.createConfigFile(config)
        print(hash_val)
        self.__packageDownloadManager.checkDownloadedDependencies()



if __name__ == "__main__":

    ctx = PlatformContext()


    ctx.test()
