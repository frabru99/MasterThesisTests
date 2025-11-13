from jsonschema import validate, ValidationError
from json import load, dump
from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
from rich.pretty import pprint
from os.path import exists
from os import listdir
from pathlib import Path
from numpy import delete
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = getLogger(__name__)
config.dictConfig(TEST_LOGGING_CONFIG)
configPath=str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "config.json")
modelsLibraryPath = str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "models_library.json")
optimizationsLibraryPath = str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "optimizations_library.json")

errorDatasetPathMessage =""" 
BenchProto/
├── ModelData/
│   └── Dataset/
│       └── dataset_name/
│           └── test/
│               ├── class1/
│               │   └── image1...
│               └── class2/
│                   └── image1...

"""

class ConfigManager:

    def __init__(self, configSchemaPath="./ConfigurationModule/ConfigFiles/configScheme.json"):
        """
        Creates the ConfigManger object, loading the JSON Schema which the configuration have
        to be complaiant with.
        
        """
        try:
            with open(configSchemaPath, "r") as configSchemaFile:
                self.__schema = load(configSchemaFile)
        except (FileNotFoundError, Exception) as e:
            logger.error(f"Encountered a problem loading the config schema file.\nThe specific error is: {e}.")



    def __printConfigFile(self, input: any, topic: str) -> None:
        """
        Prints the input with pprint with a specified topic.

        Input: 
            - input: the element to print on the screen
            - topic: the specific topic of that element
        """
        print("\n" +"-"*10 + topic + "-"*10)
        pprint(input, expand_all=True)
        print("-"*10 + "-"*len(topic)+"-"*10+"\n")


    
    def __checkModels(self, models: list) -> bool:
        """
        Checks the availability of models wrote in config file. 

        Input: 
            - models: the list of chosen models from the configuration file.
        Output:
            - result: bool
        """
        try:
            models_library = None
            changed=False
            idx_to_del=[]

            try:
                with open(modelsLibraryPath, "r") as modelsLibraryFile:
                    models_library = load(modelsLibraryFile)
            except (FileNotFoundError, Exception) as e:
                logger.error(f"The library file was not found or not loaded in the correct way.\nThe specific error is {e}.")
                return False


            #make a set of value in order to improve the performance in searching.
            models_library_sets = {
                key: set(value) for key, value in models_library.items()
            }


            for idx, model in enumerate(models):
                if model["native"]:

                    if model["model_name"] not in models_library_sets:
                        logger.error(f"The model {model['model_name']} is not present in the model library. Removing it from the config...\n")
                        idx_to_del.append(idx)
                        changed=True
                        continue

                    logger.info(f"CHANGING CONFIG TO NATIVE {model['model_name']} MODEL...")
                    models[idx]=models_library[model["model_name"]]
                    changed= True

                else:  #checks the custom model
                    logger.info(f"CHECKING CONFIG FOR CUSTOM {model['model_name']} MODEL...")

                    if exists(model["weights_path"]):
                        logger.info(f"CHECKING CONFIG WEIGHT PATH FILE FOR {model['model_name']}...")
                    else:
                        logger.error(f"There are no weights file for {model['model_name']}. Try to provide it in ./ModelData/Weights/ dir or check the weights path in config files.\n")
                        return False


            if changed:
                models = delete(models, idx_to_del).tolist() #Deleting the "native" models not present in models_library
                if len(models)==0:
                    logger.info("NO MODEL PRESENT IN THE CONFIGURATION. EXITING....")
                    exit(0)       
                logger.info(f"SHOWING NEW MODELS CONFIGURATION...")
                self.__printConfigFile(models, " MODELS SECTION ")
            else:
                logger.info(f"CONFIGURATION NOT CHANGED...")


        except (Exception) as e:
            logger.error(f"Encountered a generic problem in model check.\nThe specific error is: {e}.\n")
            return False
        return True



    def __checkOptimizations(self, optimizations: dict) -> bool:
        """
        Checks the availability of optimization methods wrote in config file. 

        Input: 
            - optimizations: the list of chosen optimizations from the configuration file.
        Output:
            - result: bool
        """

        logger.info("CHECKING OPTIMIZATION METHODS...")
        try:
            optimizations_library = None

            try:
                with open(optimizationsLibraryPath, "r") as optimizationsLibraryFile:
                    optimizations_library = load(optimizationsLibraryFile)
            except (FileNotFoundError, Exception) as e:
                logger.error(f"The library file was not found or not loaded in the correct way.\nThe specific error is {e}.")
                return False

            #make a set of value in order to improve the performance in searching.
            optimizations_library_sets = {
                key: set(value) for key, value in optimizations_library.items()
            }

            for optimization_name,value in list(optimizations.items()):
                if optimization_name not in optimizations_library_sets:
                    logger.info(f"THE OPTIMIZATION {optimization_name} IS NOT AVAILABLE. REMOVING IT FROM CONFIG FILE...")
                    optimizations.pop(optimization_name, None)
                    continue
                elif optimizations[optimization_name] not in optimizations_library_sets[optimization_name]:
                        logger.error(f"THE OPTIMIZATION {optimization_name} - {optimizations[optimization_name]} DOESN'T EXISTS. REMOVING IT FROM CONFIG FILE...")
                        optimizations.pop(optimization_name, None)
                        continue
                else:
                    logger.info(f"OPTIMIZATION {optimization_name} - {optimizations[optimization_name]} RECOGNISED!")

            if len(optimizations) == 0:
                logger.info("NO OPTIMIZATIONS PRESENT IN THE CONFIGURATION. EXITING....")
                exit(0)       
            
            self.__printConfigFile(optimizations, " OPTIMIZATIONS SECTION ")

        except (Exception) as e:
            logger.error(f"Encountered a generic problem in optimization check.\nThe specific error is: {e}")

        return True

    def __checkDataset(self, dataset: dict) -> bool:
        """
        Checks if the dataset path specified contains at least one file. The validity of the dataset will be checked later. 

        Input: 
            - dataset: the dict of dataset section in config file
        Output:
            - result: bool
        """

        dataset_path = dataset["dataset_path"] + "/test"

        logger.info(f"CHECKING DATASET PATH...")
        if exists(dataset_path) and len(listdir(dataset_path))>1:
            logger.info(f"DATASET PATH RECOGNISED!")
            self.__printConfigFile(dataset, " DATASET SECTION ")
            return True
        
        logger.error(f"Dataset path not recognised! You should have this path configuration (with at least two classes):")
        print(errorDatasetPathMessage)

        return False
        

    def loadConfigFile(self, path=configPath) -> dict:
        """
        Loads the configuration from a JSON file. 

        Input: 
            -path: path of the configuration file. It should be a JSON file. 
        Output: 
            -config: the dict that contains the configuration. 
        """

        config = ""
        try:
            with open(configPath, "r") as configFile:
                config = load(configFile)
                logger.info("LOADING CONFIGURATON...")

        
            logger.info("VALIDATING LOADED CONFIGURATION...")
            validate(instance=config, schema=self.__schema)

    
        except (ValidationError, Exception) as e:
            logger.error(f"Encountered a problem validating the config file. Check if the fields provided are correct. \n The specific error is: {e}.\n")
            return None

    
        logger.info("CONFIGURATION FILE CORRECTLY VALIDATED! \n")
        self.__printConfigFile(config, " INITIAL CONF. FILE ")

        logger.info("CHECKING THE MODELS...")
        
        if self.__checkModels(config["models"]) and self.__checkDataset(config["dataset"]) and self__checkOptimizations(config["optimizations"]):
            logger.info("DONE!")
            self.__printConfigFile(config, " FINAL CONF. FILE ")
            return config
        else:
            logger.info(f"EXITING...\n")



    def createConfigFile(self, config: dict):
        """
        Creates the configuration file from a constructed dict created by the interactive CLI session.

        Input: 
            - config: the config dict generated from interactive session
        
        """

        # It's an useless check, but we'll never know!
        try:
            logger.info("VALIDATING CREATED CONFIGURATON...")
            validate(instance=config, schema=self.__schema)
        except (ValidationError, Exception) as e:
            logger.error(f"Encountered a problem validating the config file. Check if the fields provided are correct. \n The specific error is: {e}.\n")
            return

        logger.info("CONFIGURATION FILE CORRECTLY VALIDATED! \n")

        if self.__checkModels(config["models"]) and self.__checkDataset(config["dataset"]) and self.__checkOptimizations(config["optimizations"]):
            logger.info("DONE!")
            self.__printConfigFile(config, " FINAL CONF. FILE ")
            logger.info(f"SAVING IT INTO {configPath}...")

            with open(configPath, "w") as configFile:
                dump(config, configFile, indent=4)

            logger.info(f"SAVED!")

        else:
            logger.info(f"EXITING...\n")

        


if __name__ == "__main__":

    configTest = {
        "models": [
            {
                "model_name": "mobilenet_v2", 
                "native": True
            },
            {
                "module": "torchvision.models",
                "model_name": "efficientnet", 
                "native": False,
                "distilled": False,
                "weights_path": "./ModelData/Weights/efficientnet.pth",
                "device": "cpu",
                "class_name": "efficientnet_b0",
                "weights_class": "EfficientNet_B0_Weights", 
                "image_size": 224,
                "num_classes": 1000,
                "task": "classification",
                "description": "EfficientNet from Custom Models"
            }
        ],
        "optimizations": {"Quantization": "8int", "Pruning": "RandomUnstructured"},
        "dataset": {
            "dataset_path": "./ModelData/Dataset/dataset_name",
            "batch_size": 32
        }
    }


    configManager = ConfigManager()
    #configFile = configManager.loadConfigFile()
    configManager.createConfigFile(configTest)



    
