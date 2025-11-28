from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger


from jsonschema import validate, ValidationError
from json import load, dump, decoder
from rich.pretty import pprint
from os.path import exists
from os import listdir
from pathlib import Path
from numpy import delete
from pathlib import Path
from hashlib import sha224
from Utils.utilsFunctions import getLongestSubString, getFilenameList


PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path=str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "config.json") #config file path
config_schema_path=str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "configScheme.json") #configSchema file path
config_history_path=str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "configHistory.json") #configHistory file path
models_library_path= str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "models_library.json") #models_library file path
optimizations_library_path = str(PROJECT_ROOT / "ConfigurationModule" / "ConfigFiles" / "optimizations_library.json") #optimizations_library file path
models_weights_path = str(PROJECT_ROOT / "ModelData" / "Weights") #weights of embedded models in the framework
VALID_CHOICES = {'y','n'} #Choices for CPU Usage
OPTIMIZATIONS_NEED_ARCH = {"Quantization"} #Optimizations that needs the arch type of the system.

error_dataset_path_message =""" 
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

    def __init__(self, arch, there_is_gpu, schema_path=config_schema_path):
        """
        Creates the ConfigManger object, loading the JSON Schema which the configuration have
        to be complaiant with.
        
        """
        try:
            with open(schema_path, "r") as config_schema_file:
                self.__schema = load(config_schema_file)
            self.__arch = arch
            self.__there_is_gpu = there_is_gpu
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
        Checks the availability of models wrote in config file and applies the needed changes.

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
                with open(models_library_path, "r") as models_library_file:
                    models_library = load(models_library_file)
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

                if models[idx]['device'] == "gpu" and not self.__there_is_gpu:
                    logger.warning(f"THE GPU IS NOT PRESENT, CHANGING THE DEVICE TO 'CPU' for {model['model_name']}...")
                    models[idx]['device'] = "cpu"

            if changed:

                models = delete(models, idx_to_del).tolist() #Deleting the "native" models not present in models_library
                if len(models)==0:
                    logger.info("NO MODEL PRESENT IN THE CONFIGURATION. EXITING....")
                    exit(0)

                logger.info(f"SHOWING NEW MODELS CONFIGURATION...")
                self.__printConfigFile(models, " MODELS SECTION ")

            else:
                logger.info(f"CONFIGURATION NOT CHANGED...")
                self.__printConfigFile(models, " MODELS SECTION ")



        except (Exception) as e:
            logger.error(f"Encountered a generic problem in model check.\nThe specific error is: {e}.\n")
            return False
        return True



    def __checkOptimizations(self, optimizations: dict, model_dicts) -> bool:
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
            opt_to_remove=[]

            try:
                with open(optimizations_library_path, "r") as optimizations_library_file:
                    optimizations_library = load(optimizations_library_file)
            except (FileNotFoundError, Exception) as e:
                logger.error(f"The library file was not found or not loaded in the correct way.\nThe specific error is {e}.")
                return False

            #make a set of value in order to improve the performance in searching.
            optimizations_library_sets = {
                key: set(value) for key, value in optimizations_library.items()
            }

            for optimization_name in optimizations.keys():

                if optimization_name not in optimizations_library_sets:
                    logger.info(f"THE OPTIMIZATION {optimization_name} IS NOT AVAILABLE. REMOVING IT FROM CONFIG FILE...")
                    opt_to_remove.append(optimization_name)
                    continue

                elif optimizations[optimization_name]["method"] not in optimizations_library_sets[optimization_name]:
                    logger.error(f"THE OPTIMIZATION {optimization_name} - {optimizations[optimization_name]['method']} DOESN'T EXISTS. REMOVING IT FROM CONFIG FILE...")
                    opt_to_remove.append(optimization_name)
                    continue

                else:
                    logger.info(f"OPTIMIZATION {optimization_name} - {optimizations[optimization_name]['method']} RECOGNISED!")


            if len(opt_to_remove) > 0:
                for name in opt_to_remove:
                    optimizations.pop(name, None)
                    

            if len(optimizations) == 0:
                logger.info("NO OPTIMIZATIONS PRESENT IN THE CONFIGURATION. EXITING....")
                exit(0)   

            if "Distillation" in optimizations.keys():
                if optimizations['Distillation']['method']:
                    self.__createDistilledPaths(optimizations, model_dicts)
            
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

        dataset_path = dataset["data_dir"] + "/test"

        logger.info(f"CHECKING DATASET PATH...")
        if exists(dataset_path) and len(listdir(dataset_path))>1:
            logger.info(f"DATASET PATH RECOGNISED!")
            self.__printConfigFile(dataset, " DATASET SECTION ")
            return True
        
        logger.error(f"Dataset path not recognised! You should have this path configuration (with at least two classes):")
        print(error_dataset_path_message)

        return False

    
    def __updateConfigHistory(self, config: dict, hash_value: str) -> None:

        """
        This function asks to the user if the loaded/created configurations has to be added to the historyConfig.json file. 
        If the hash_value (key) is already present, the function returns.

        Input:
            - config: the created/loaded configuration
        
        """
    

        history_dict = {}
        with open(config_history_path, "r") as config_history_file:
            try:
                history_dict = load(config_history_file)
                if hash_value in history_dict.keys():
                    logger.info("The configuration is already present in the history!")
                    return

            except decoder.JSONDecodeError as e:
                logger.info(f"THE HISTORY FILE WAS EMPTY!")



        while True:
            choice = input(f"Do you want save the config into the history? (y/n): ").lower()

            if choice in VALID_CHOICES:
                if choice == 'y':
                    try:

                        history_dict[f"{hash_value}"] = config

                        with open(config_history_path, "w") as config_history_file:
                            dump(history_dict, config_history_file, indent=4)

                        logger.info("CONFIG ADDED CORRECTLY TO THE HISTORY!")

                    except (FileNotFoundError,Exception) as e:
                        logger.error(f"Encountered a problem saving the config in the history.\nThe specific error is: {e}.\n")
            else:
                print("Invalid Input. Please enter 'y' or 'n'.")
                continue
            break
        print("\n")

        

    def loadConfigFile(self, path=config_path) -> (dict, str):
        """
        Loads the configuration from a JSON file. 

        Input: 
            -path: path of the configuration file. It should be a JSON file. 
        Output: 
            -config: the dict that contains the configuration. 
            -hash_value: the hash value generated on config file
        """

        config = ""
        try:
            with open(path, "r") as config_file:
                config = load(config_file)
                logger.info("LOADING CONFIGURATON...")

        
            logger.info("VALIDATING LOADED CONFIGURATION...")
            validate(instance=config, schema=self.__schema)

        except (ValidationError, Exception) as e:
            logger.error(f"Encountered a problem validating the config file. Check if the fields provided are correct.\nThe specific error is: {e}.\n")
            return None

    
        logger.info("CONFIGURATION FILE CORRECTLY VALIDATED! \n")
        self.__printConfigFile(config, " INITIAL CONF. FILE ")

        logger.info("CHECKING THE MODELS...")
        
        if self.__checkModels(config["models"]) and self.__checkDataset(config["dataset"]) and self.__checkOptimizations(config["optimizations"], config["models"]):
            logger.info("DONE!")
            self.__printConfigFile(config, " FINAL CONF. FILE ")
            hash_value = sha224(str(config).encode("utf-8")).hexdigest()
            self.__updateConfigHistory(config, hash_value)
            #Arch for Quantization Optimization
            self.__addArchType(config)
            return config, hash_value
        else:
            logger.info(f"EXITING...\n")
            return None, None

    def __addArchType(self, config: dict):

        for optimization in OPTIMIZATIONS_NEED_ARCH:
                if optimization in config["optimizations"]:
                    config["optimizations"][optimization]["arch"] = self.__arch

    
    def createConfigFile(self, config: dict) -> str:
        """
        Creates the configuration file from a constructed dict created by the interactive CLI session.

        Input: 
            - config: the config dict generated from interactive session

        Output:
            - hash_value: returns the hash_value of the config for id purposes
        
        """

        # It's an useless check, but we'll never know!
        try:
            logger.info("VALIDATING CREATED CONFIGURATON...")
            validate(instance=config, schema=self.__schema)
        except (ValidationError, Exception) as e:
            logger.error(f"Encountered a problem validating the config file. Check if the fields provided are correct. \n The specific error is: {e}.\n")
            return

        logger.info("CONFIGURATION FILE CORRECTLY VALIDATED! \n")

        if self.__checkModels(config["models"]) and self.__checkDataset(config["dataset"]) and self.__checkOptimizations(config["optimizations"], config['models']):
            logger.info("DONE!")
            self.__printConfigFile(config, " FINAL CONF. FILE ")
            logger.info(f"SAVING IT INTO {config_path}...")

            with open(config_path, "w") as config_file:
                dump(config, config_file, indent=4)

            logger.info(f"SAVED!")

            hash_value = sha224(str(config).encode("utf-8")).hexdigest()
            self.__updateConfigHistory(config, hash_value)
            self.__addArchType(config)
            return hash_value
        else:
            logger.info(f"EXITING...\n")

    def __createDistilledPaths(self, optimizations: dict, model_dicts):
        """
        Create the distilled paths for loading the distilled version of chosen models

        Input:
            - optimizations: dict of optimizations
            - model_dicts: dict of all models choosen
        """

        file_name_list = getFilenameList(models_weights_path)
        file_name_list = [file_name for file_name in file_name_list if "_distilled" in file_name]

        for model_dict in model_dicts:

            best_candidate_for_model, best_file_name = "", ""
            for file_name in file_name_list:

                file_name_without_pth = file_name.removesuffix("_distilled.pth")

                best_candidate_for_name = getLongestSubString(model_dict['model_name'], file_name_without_pth)
                best_candidate_for_path = getLongestSubString(model_dict['weights_path'].split("/")[-1].removesuffix(".pth"), file_name_without_pth)

                best_candidate_between_name_path = max(best_candidate_for_name, best_candidate_for_path, key=len)

                if len(best_candidate_between_name_path) > len(best_candidate_for_model):
                    best_candidate_for_model = best_candidate_between_name_path
                    best_file_name = file_name

                # Perfect Match, found the distilled weights
                if len(file_name_without_pth) == len(best_candidate_for_model):
                    break

            if len(best_file_name.removesuffix("_distilled.pth")) == len(best_candidate_for_model):
                logger.info(f"MODEL: {model_dict['model_name']} | YOU FOUND THE CORRECT DISTILLED MODEL {best_file_name}")
            elif len(best_file_name) == 0:
                logger.error(f"MODEL: {model_dict['model_name']} | NO MATCH WITH NONE FILE FOR DISTILLED MODEL")
                exit(0)
            else:
                logger.warning(f"MODEL: {model_dict['model_name']} | YOU FOUND A PARTIAL MATHC FOR A DISTILLED MODEL: {best_file_name}")

            optimizations['Distillation']['distilled_paths'][model_dict['model_name']] =  f"{models_weights_path}/{best_file_name}"              

    

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
                "weights_path": "./ModelData/Weights/casting_efficientnet_b0.pth",
                "device": "cpu",
                "class_name": "efficientnet_b0",
                "weights_class": "EfficientNet_B0_Weights", 
                "image_size": 224,
                "num_classes": 1000,
                "task": "classification",
                "description": "EfficientNet from Custom Models"
            }
        ],
        "optimizations": {
            "Quantization": {
                "method": "QInt8",
                "type":"static" 
            },
            "Pruning": {
                "method": "L1Unstructured",
                "amount": 0.7
            },
            "Distillation": {
                'method': True,
                'distilled_paths': {}
            }
        },
        "dataset": {
            "data_dir": "./ModelData/Dataset/casting_data",
            "batch_size": 32
        }
    }


    configManager = ConfigManager(arch="x86", there_is_gpu=False)
    #configFile, hash_value = configManager.loadConfigFile()

    hash_value = configManager.createConfigFile(configTest)





    
