from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

from json import load
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from Utils.utilsFunctions import cleanCaches

#For Example Purposes
from ProbeHardwareModule.probeHardwareManager import ProbeHardwareManager
from PackageDownloadModule.packageDownloadManager import PackageDownloadManager

class DoE():
    _instance = None #for Singleton

    #Singleton Management
    def __new__(cls, *args, **kwargs) -> object:
        """
        Returns a new istance if isn't created.

        Input: 
            -cls: the class

        Output: 
            -object: the created object

        """
        if cls._instance is None:
            cls._instance=super().__new__(cls)
        return cls._instance


    def __init__(self, config: dict, config_id: str):
        if not hasattr(self, "created"):
            self.created=True #After that the first object is initialized, we'll not initialize another one. 
            self.__initialized=False
            self.__config_id = config_id
            self.__model_info=config["models"]
            self.__optimizations_info=config["optimizations"]
            self.__dataset_info = config["dataset"]
            self.__models=self.__initializeListOfModels(config["models"])
            self.__optimizations=self.__initializeListOfOptimizations(config["optimizations"])
            self.__dataset=self.__initializeDataset()
            self.__inference_loaders={}
        #metrics?

    def __initializeListOfModels(self, models: list) -> list:
        """ 
        Initializes the models list of the DoE class, starting from the config given in input. 

        Input:
            - models: list of dicts taken from the config.

        Output:
            - aiModelList: list of AIModel objects.

        """

        ai_models_list=[]

        for aiModelDict in models:
            logger.info(f"CREATING THE {aiModelDict['model_name']} MODEL...\n")
            ai_models_list.append(aiModel.AIModel(aiModelDict))

        logger.info(f"MODELS CREATED!\n")


        return ai_models_list


    def __initializeListOfOptimizations(self, optimizations: dict) -> list:
        """ 
        Initializes the optimization list of the DoE class, starting from the config given in input. 

        Input:
            - optimizations: dict taken from the config

        Output:
            - optimization_object_list: list of Optimization objects.

        """
        
        optimization_object_list=[]

        try:
        
            for optimization_name in optimizations.keys():
                full_class_name=f"{optimization_name}Optimization"
                target_class = getattr(optimization, full_class_name)

                optimization_object = target_class(optimizations[optimization_name])
                optimization_object_list.append(optimization_object)
                logger.info(f"{full_class_name} ADDED!")

        except (FileNotFoundError,Exception) as e:
            logger.error(f"Encountered a generic problem initializing the list of optimizations.\nThe specific error is: {e}.")
  

        return optimization_object_list
        


    def __initializeDataset(self) -> object:
        """
            Initializes the DataWrapper object. 

            Input:
                None

            Output:
                - dataset_wrapper: the data wrapper object.

        """

        dataset_wrapper = datawrapper.DataWrapper()

        return dataset_wrapper


    def initializeDoE(self):
        """
            This function initializes DoE, applying optimizations to the models 
            and loading the dataset.

            Input:
                - None

            Output:
                - None
        """

        optimized_models = []
        

        for model in self.__models: #Iterating the base models. 
            dataset = datawrapper.DataWrapper() #creating the dedicated dataWrapper for the model
            dataset.loadInferenceData(model_info=model.getAllInfo(), dataset_info=self.__dataset_info)
            inference_loader = dataset.getLoader()
            self.__inference_loaders[model.getInfo("model_name")]=inference_loader #N Base Models = N Loaders
            model.createOnnxModel(inference_loader, self.__config_id)

            #Optimized Model...
            for optimizator in self.__optimizations:
                optimizator.setAIModel(model)
                optimized_model = optimizator.applyOptimization(inference_loader, self.__config_id)

                if not optimized_model.getInfo("model_name").endswith("quantized"):
                    optimized_model.createOnnxModel(inference_loader, self.__config_id)

                optimized_models.append(optimized_model)
                self.__inference_loaders[optimized_model.getInfo("model_name")] = inference_loader

        self.__models.extend(optimized_models)
        self.__initialized=True



     

    def run(self):
        assert self.__initialized, "The DoE should be initialized in order to run."

    
        for model in self.__models:
            #for inferece_loader_name, inferece_loader in self.__inference_loaders.items():
            inference_loader = self.__inference_loaders[model.getInfo("model_name")]
            logger.info(f"MODEL NAME: {model.getInfo('model_name')} - INFERENCE LOADER {inference_loader}")
            cleanCaches()
            model.runInference(inference_loader, self.__config_id)
                    





if __name__ == "__main__":

    config = {
        "models": [
            {
                "module": "torchvision.models",
                "model_name": "mobilenet_v2",
                "native": False,
                "distilled": False,
                "weights_path": "ModelData/Weights/mobilenet_v2.pth",
                "device": "cpu",
                "class_name": "mobilenet_v2",
                "weights_class": "MobileNet_V2_Weights.DEFAULT", 
                "image_size": 224, 
                "num_classes": 2,
                "task": "classification",
                "description": "Mobilenet V2 from torchvision"
            }, 
            {
                "model_name": "efficientnet",
                "native": True
            }
        ],
        "optimizations": {
            "Quantization": {
                "method": "QInt8",
                "type": "static"
            }, 
            "Pruning": {
                "method": "Random", 
                "amount": 0.3
            },
            "Distillation":{
                "method": True,
                "distilled_paths": {}
            }
        },
        "dataset": {
            "data_dir": "ModelData/Dataset/casting_data",
            "batch_size": 32
        }
    }

    probe = ProbeHardwareManager()

    there_is_gpu, gpu_type, sys_arch = probe.checkSystem()

    pdm = PackageDownloadManager()

    pdm.checkDownloadedDependencies(there_is_gpu)


    #local scripts
    import BenchmarkingFactory.aiModel as aiModel
    import BenchmarkingFactory.optimization as optimization
    import BenchmarkingFactory.dataWrapper as datawrapper
    from ConfigurationModule.configurationManager import optimizations_library_path, ConfigManager


    cm = ConfigManager(there_is_gpu=there_is_gpu, arch=sys_arch)

    config_id = cm.createConfigFile(config)

    doe = DoE(config, config_id)

    doe.initializeDoE()
    doe.run()



    #doe.getString()


    

    







