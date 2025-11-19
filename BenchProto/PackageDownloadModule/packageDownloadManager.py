from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger


from rich.pretty import pprint
from pathlib import Path
from json import load, decoder, dump
from subprocess import check_call
from sys import executable
from importlib.metadata import distributions
from time import sleep

PROJECT_ROOT = Path(__file__).resolve().parent.parent
requirements_file_needed_path = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "needed.txt" )
requirements_file_gpu_path = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "gpu.txt" )
requirements_installed_path= str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / ".installed.json" )


class PackageDownloadManager:

    def __init__(self):
        pass
    
    # def __init__(self):
    #     self.__needed_reqs, self.__gpu_reqs = self.__loadRequirements()
        

    # def __loadRequirements(self) -> (dict, dict):

    #     needed = {}
    #     gpu = {}

    #     path_files = [requirements_file_needed_path, requirements_file_gpu_path]
    #     try:
    #         for idx, path in enumerate(path_files):
    #             with open(path, "r") as file:

    #                 for line in file:
    #                     line = line.strip()
    #                     values = line.split("==")

    #                     if idx==0:
    #                         needed[values[0].lower()] = values[1]
    #                     else:
    #                         gpu[values[0].lower()] = values[1]
                            
    #         return needed, gpu
                
    #     except (FileNotFoundError, Exception) as e:
    #         logger.error(f"Encountered an error retrieving the needed dependencies. Please check {requirements_file_needed_path} file.\nThe specific error is: {e}")
    #         exit(0)



    def __checkAlreadyInstalled(self, there_is_gpu: bool) -> (bool, bool):

        """
        Checks if dependecies are already installed, inspectionating the .installed.json file. 

        Input:
            - there_is_gpu: bool
        
        Output:
            - install_needed: bool
            - install_gpu: bool

        """

        requirementInstalled = {}
        try:
            with open(requirements_installed_path, "r") as installed_requirements:
                requirementInstalled = load(installed_requirements)

            install_needed = not requirementInstalled.get("needed", False)
            install_gpu = there_is_gpu and not requirementInstalled.get("gpu", False)

            return install_needed, install_gpu
            
        except decoder.JSONDecodeError as e:
            logger.error(f"Encountered an error decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
            logger.info(f"Installing only basic dependencies...")
            return True, False

        except (FileNotFoundError,Exception) as e:
            logger.error(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
        
        exit(0)


    def checkDownloadedDependencies(self, there_is_gpu: bool):

        """
        Checks if dependecies are already installed calling __checkingAlreadyInstalled. After, it installs the required dependencies.


        Input:
            - there_is_gpu: bool 
        
        """

        installed_requirements_dict={}

        try:
            with open(requirements_installed_path, "r") as installed_requirements_file:
                installed_requirements_dict = load(installed_requirements_file)


            installNeeded, installGpu = self.__checkAlreadyInstalled(there_is_gpu)
            
            if installNeeded:
                logger.info(f"INSTALLING NEEDED DEPENDENCIES...\n")
                sleep(1)
                check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_needed_path, '--force-reinstall'])
                installed_requirements_dict["needed"] = True
            else:
                logger.info(f"NEEDED DEPENDENCIES ALREADY PRESENT...\n")


            if installGpu:
                logger.info(f"INSTALLING GPU DEPENDENCIES...\n")
                sleep(1)
                check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_gpu_path, '--force-reinstall'])
                installed_requirements_dict["gpu"] = True
            else:
                logger.info(f"GPU DEPENDENCIES ALREADY PRESENT OR NOT NEEDED...\n")

            if installed_requirements_dict and (installNeeded or installGpu):
                with open(requirements_installed_path, "w") as installed_requirements_file:
                    dump(installed_requirements_dict, installed_requirements_file, indent=4)
                
            logger.info("ALL DEPENDENCIES INSTALLED! IF THERE ARE PROBLEMS, WE SUGGEST TO FORCE-REINSTALL THE DEPENDENCIES.\n")


        except decoder.JSONDecodeError as e:
            logger.error(f"Encountered an error Decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
            exit(0)
        except (FileNotFoundError,Exception) as e:
            logger.error(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            exit(0)

        


# if __name__ == "__main__":
#     there_is_gpu = True

#     pdm = PackageDownloadManager()

#     pdm.checkDownloadedDependencies(there_is_gpu)
    