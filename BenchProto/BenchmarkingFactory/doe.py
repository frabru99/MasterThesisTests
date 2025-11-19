from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)



from json import load
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

#local scripts
import BenchmarkingFactory.aiModel as aiModel
import BenchmarkingFactory.optimization as optimization
import BenchmarkingFactory.dataWrapper as datawrapper
from ConfigurationModule.configurationManager import optimizations_library_path, ConfigManager





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
        if not hasattr(self, "initialized"):
            self.initialized=True #After that the first object is initialized, we'll not initialize another one. 
            self.__config_id = config_id
            self.__model_info=config["models"]
            self.__optimizations_info=config["optimizations"]
            self.__dataset_info = config["dataset"]
            self.__models=self.__initializeListOfModels(config["models"])
            self.__optimizations=self.__initializeListOfOptimizations(config["optimizations"])
            self.__dataset=self.__initializeDataset()
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

        #dataset_wrapper.loadInferenceData(dataset, self.__model_info[0])

        return dataset_wrapper
     

    def run():
        pass








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
                "module": "torchvision.models",
                "model_name": "efficientnet", 
                "native": False,
                "distilled": False,
                "weights_path": "ModelData/Weights/casting_efficientnet_b0.pth",
                "device": "cpu",
                "class_name": "efficientnet_b0",
                "weights_class": "EfficientNet_B0_Weights.DEFAULT", 
                "image_size": 224,
                "num_classes": 2,
                "task": "classification",
                "description": "EfficientNet from Custom Models"
            }
        ],
        "optimizations": {
            "Pruning": {
                "method": "L1Unstructured",
                "amount": 0.7
            }
        },
        "dataset": {
            "data_dir": "ModelData/Dataset/dataset_name",
            "batch_size": 32
        }
    }

    cm = ConfigManager()

    config_id = cm.createConfigFile(config)

    doe = DoE(config, config_id)


    #doe.getString()


    

    








