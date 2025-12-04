from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

import pandas as pd
import os
import torch
import json
import gc
import traceback
from pathlib import Path
from statsmodels.formula.api import ols
from itertools import product
from pandas import DataFrame
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from multiprocessing import Process, SimpleQueue, set_start_method
from Utils.utilsFunctions import subRun, cleanCaches, initialPrint
from rich.pretty import pprint


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from Utils.utilsFunctions import cleanCaches

#For Example Purposes
from ProbeHardwareModule.probeHardwareManager import ProbeHardwareManager
from PackageDownloadModule.packageDownloadManager import PackageDownloadManager


torch.multiprocessing.set_sharing_strategy('file_system')


class DoE():
    _instance = None #for Singleton

    #Singleton Management
    def __new__(cls, *args, **kwargs) -> object:
        """
        Returns a new istance if isn't created.s

        Input: 
            -cls: the class

        Output: 
            -object: the created object

        """
        if cls._instance is None:
            cls._instance=super().__new__(cls)
        return cls._instance

    ###################################################################

    # TODO
    # SOME DATA IN THE DOE CONFIG MUST BE ADDED TO THE GENERARL CONFIG

    ###################################################################


    def __init__(self, config: dict, config_id: str):
        if not hasattr(self, "created"):
            self.created=True #After that the first object is initialized, we'll not initialize another one. 
            self.__initialized=False
            self.__ran=False
            self.__config_id = config_id
            self.__repetitions = config["repetitions"] # Must be added to the general config
            self.__model_info=config["models"]
            self.__optimizations_info=config["optimizations"]
            self.__dataset_info = config["dataset"]
            self.__models=self.__initializeListOfModels(config["models"])
            self.__optimizations=self.__initializeListOfOptimizations(config["optimizations"])
            self.__dataset=self.__initializeDataset()
            self.__inference_loaders={}
            self.__design = self.__initializeDesign()
            self.__current_stats = None
        #metrics?

    def __initializeListOfModels(self, models: list) -> list:
        """ 
        Initializes the models list of the DoE class, starting from the config given in input. 

        Input:
            - models: list of dicts taken from the config.

        Output:
            - aiModelList: list of AIModel objects.

        """

        ai_models_dict={}

        for aiModelDict in models:
            logger.info(f"CREATING THE {aiModelDict['model_name']} MODEL...\n")
            ai_models_dict[aiModelDict['model_name']] = {}   
            ai_models_dict[aiModelDict['model_name']]['Base'] = aiModel.AIModel(aiModelDict)

        logger.info(f"MODELS CREATED!\n")


        return ai_models_dict


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
                optimization_object_list.append((optimization_object, optimization_name))
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

    def __initializeDesign(self) -> dict:
        """
            Initializes the Design of experiment combinations. 

            Input:
                None

            Output:
                - list of all possible combination in the experiments

        """

        initialPrint("DESIGN OF EXPERIMENTS\n")
        models_list = []
        for model_name in self.__models.keys():
            models_list.append(model_name)

        optimization_list = []
        for opt_obj, opt_name in self.__optimizations:
            optimization_list.append(opt_name)

        design = list(product(models_list, optimization_list))

        self.__printDesign(design)
        full_design = []
        for _ in range(self.__repetitions):
            full_design.extend(design)

        return full_design

    def __printDesign(self, design:list):
        print('\x1b[32m' +f"{'MODEL NAME':<20}\t{'OPTIMIZATION':<20}\tREPETITIONS" + '\x1b[37m')
        for model, optimization in design:
            print(f"{model:<20}\t{optimization:<20}\t{self.__repetitions}")

        print("\n")



    def initializeDoE(self):
        """
            This function initializes DoE, applying optimizations to the models 
            and loading the dataset.

            Input:
                - None

            Output:
                - None
        """

        initialPrint("OPTIMIZATIONS APPLY")
        optimized_models = []
        

        for model_key, model_dict in self.__models.items(): #Iterating the base models. 
            dataset = self.__dataset #creating the dedicated dataWrapper for the model
            dataset.loadInferenceData(model_info=model_dict['Base'].getAllInfo(), dataset_info=self.__dataset_info)
            inference_loader = dataset.getLoader()
            self.__inference_loaders[model_dict['Base'].getInfo("model_name")]=inference_loader #N Base Models = N Loaders
            model_dict['Base'].createOnnxModel(inference_loader, self.__config_id)

            #Optimized Model...
            for optimizator, op_name in self.__optimizations:
                optimizator.setAIModel(model_dict['Base'])
                optimized_model = optimizator.applyOptimization(inference_loader, self.__config_id)

                if not optimized_model.getInfo("model_name").endswith("quantized"):
                    optimized_model.createOnnxModel(inference_loader, self.__config_id)

                #optimized_models.append(optimized_model)
                model_dict[op_name] = optimized_model
                self.__inference_loaders[optimized_model.getInfo("model_name")] = inference_loader

        #self.__models.extend(optimized_models)
        self.__initialized=True


    def run(self):
        assert self.__initialized, "The DoE should be initialized in order to run."

        initialPrint("INFERENCES")

        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass # Context already set

        temp_dir = PROJECT_ROOT / "temp_results"
        temp_dir.mkdir(exist_ok=True)
    
        results_list = []

        for i, (mod_name, opt_name) in enumerate(self.__design):

            print("\n\t"+ "\x1b[36m" + f" RUN {i+1} / {len(self.__design)}, : {mod_name} | {opt_name}" + "\x1b[37m" + "\n")

            cleanCaches()

            try:
                aimodel = self.__models[mod_name][opt_name]
            except KeyError:
                logger.error(f"Model {mod_name} with Optimization {opt_name} not found!")
                continue

            internal_name = aimodel.getInfo('model_name')
            inference_loader = self.__inference_loaders[internal_name]

            # Temp path creation
            temp_file_path = temp_dir / f"run_{i}.json"

            # Clean up if it exists from a previous crash
            if temp_file_path.exists():
                os.remove(temp_file_path)


            sub_process_args = (aimodel, inference_loader, self.__config_id, temp_file_path)

            #Create the subProcess to execute in a separated memory space
            #In order to clean caches between executions
            sub_process_run = Process(target = subRun, args=sub_process_args)

            sub_process_run.start()
            sub_process_run.join()

            if sub_process_run.is_alive():
                 sub_process_run.terminate()
            
            del sub_process_run
            gc.collect()

            if temp_file_path.exists():
                try:
                    with open(temp_file_path, 'r') as f:
                        stats = json.load(f)
                    
                    # Delete the temp file now that we have the data
                    os.remove(temp_file_path)

                    # Extracting data from stats
                    results_list.append({
                        "Model": mod_name,
                        "Optimization": opt_name,
                        "Total_Inference_Time_ms": stats["Total 'kernel' inference time"]

                    })

                    df = DataFrame(results_list)
                    df.to_csv("doe_results_raw.csv", index = False)
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON from worker output.")
            else:
                logger.error("Worker finished but NO output file was found. (Process likely crashed silently)")
                return

        self.__ran=True




    def runAnova(self):
        assert self.__ran, "The DoE should be executed with .run before running ANOVA."

        initialPrint("ANOVA ANALYSIS\n")

        file_path = str(PROJECT_ROOT / "doe_results_raw.csv")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            logger.critical(f"Error: The file {file_path} was not found.")
            exit(0)

        
        pprint(df.head())

        
        formula = 'Total_Inference_Time_ms ~ C(Model) + C(Optimization) + C(Model):C(Optimization)'

        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        initialPrint("RESULTS\n")
        pprint(anova_table)


        plt.figure(figsize=(10, 6))
        interaction_plot(x=df['Optimization'], 
                        trace=df['Model'], 
                        response=df['Total_Inference_Time_ms'], 
                        colors=['red', 'blue'], 
                        markers=['D', '^'], 
                        ms=10)

        plt.title('Interaction Plot: Model vs Optimization')
        plt.xlabel('Optimization Technique')
        plt.ylabel('Inference Time (ms)')
        plt.grid(True)
        plt.show()

        

if __name__ == "__main__":


    config = {
        "models": [
            {
                "module": "torchvision.models",
                "model_name": "mobilenet_v2",
                "native": False,
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
            "batch_size": 15
        },
        "repetitions": 2
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
    doe.runAnova()



    #doe.getString()


    

    







