from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger


<<<<<<< HEAD

=======
>>>>>>> main
from rich.pretty import pprint
from pathlib import Path
from json import load, decoder, dump
from subprocess import check_call
from sys import executable
from importlib.metadata import distributions
from time import sleep

PROJECT_ROOT = Path(__file__).resolve().parent.parent
<<<<<<< HEAD
requirements_file_needed_path = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "needed.txt" )
requirements_file_gpu_path = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "gpu.txt" )
requirements_installed_path= str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / ".installed.json" )
=======
requirementsFileNeededPath = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "needed.txt" )
requirementsFileGpuPath = str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / "gpu.txt" )
requirementsInstalledPath= str(PROJECT_ROOT / "PackageDownloadModule" / "requirementsFileDirectory" / ".installed.json" )
>>>>>>> main


class PackageDownloadManager:

    def __init__(self):
        pass
    
    # def __init__(self):
    #     self.__needed_reqs, self.__gpu_reqs = self.__loadRequirements()
        

    # def __loadRequirements(self) -> (dict, dict):

    #     needed = {}
    #     gpu = {}

<<<<<<< HEAD
    #     path_files = [requirements_file_needed_path, requirements_file_gpu_path]
=======
    #     path_files = [requirementsFileNeededPath, requirementsFileGpuPath]
>>>>>>> main
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
<<<<<<< HEAD
    #         logger.error(f"Encountered an error retrieving the needed dependencies. Please check {requirements_file_needed_path} file.\nThe specific error is: {e}")
=======
    #         logger.error(f"Encountered an error retrieving the needed dependencies. Please check {requirementsFileNeededPath} file.\nThe specific error is: {e}")
>>>>>>> main
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
<<<<<<< HEAD
            with open(requirements_installed_path, "r") as installed_requirements:
                requirementInstalled = load(installed_requirements)
=======
            with open(requirementsInstalledPath, "r") as installedRequirements:
                requirementInstalled = load(installedRequirements)
>>>>>>> main

            install_needed = not requirementInstalled.get("needed", False)
            install_gpu = there_is_gpu and not requirementInstalled.get("gpu", False)

            return install_needed, install_gpu
            
        except decoder.JSONDecodeError as e:
<<<<<<< HEAD
            logger.error(f"Encountered an error decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
=======
            logger.error(f"Encountered an error decoding the JSON file at path {requirementsInstalledPath}. It shouldn't be empty!\nThe specific error is: {e}")
>>>>>>> main
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

<<<<<<< HEAD
        installed_requirements_dict={}

        try:
            with open(requirements_installed_path, "r") as installed_requirements_file:
                installed_requirements_dict = load(installed_requirements_file)
=======
        installedRequirementsDict={}

        try:
            with open(requirementsInstalledPath, "r") as installedRequirementsFile:
                installedRequirementsDict = load(installedRequirementsFile)
>>>>>>> main


            installNeeded, installGpu = self.__checkAlreadyInstalled(there_is_gpu)
            
            if installNeeded:
                logger.info(f"INSTALLING NEEDED DEPENDENCIES...\n")
                sleep(1)
<<<<<<< HEAD
                check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_needed_path, '--force-reinstall'])
                installed_requirements_dict["needed"] = True
=======
                check_call([executable, '-m', 'pip', 'install', '-r', requirementsFileNeededPath, '--force-reinstall'])
                installedRequirementsDict["needed"] = True
>>>>>>> main
            else:
                logger.info(f"NEEDED DEPENDENCIES ALREADY PRESENT...\n")


            if installGpu:
                logger.info(f"INSTALLING GPU DEPENDENCIES...\n")
                sleep(1)
<<<<<<< HEAD
                check_call([executable, '-m', 'pip', 'install', '-r', requirements_file_gpu_path, '--force-reinstall'])
                installed_requirements_dict["gpu"] = True
            else:
                logger.info(f"GPU DEPENDENCIES ALREADY PRESENT OR NOT NEEDED...\n")

            if installed_requirements_dict and (installNeeded or installGpu):
                with open(requirements_installed_path, "w") as installed_requirements_file:
                    dump(installed_requirements_dict, installed_requirements_file, indent=4)
=======
                check_call([executable, '-m', 'pip', 'install', '-r', requirementsFileGpuPath, '--force-reinstall'])
                installedRequirementsDict["gpu"] = True
            else:
                logger.info(f"GPU DEPENDENCIES ALREADY PRESENT OR NOT NEEDED...\n")

            if installedRequirementsDict and (installNeeded or installGpu):
                with open(requirementsInstalledPath, "w") as installedRequirementsFile:
                    dump(installedRequirementsDict, installedRequirementsFile, indent=4)
>>>>>>> main
                
            logger.info("ALL DEPENDENCIES INSTALLED! IF THERE ARE PROBLEMS, WE SUGGEST TO FORCE-REINSTALL THE DEPENDENCIES.\n")


        except decoder.JSONDecodeError as e:
<<<<<<< HEAD
            logger.error(f"Encountered an error Decoding the JSON file at path {requirements_installed_path}. It shouldn't be empty!\nThe specific error is: {e}")
=======
            logger.error(f"Encountered an error Decoding the JSON file at path {requirementsInstalledPath}. It shouldn't be empty!\nThe specific error is: {e}")
>>>>>>> main
            exit(0)
        except (FileNotFoundError,Exception) as e:
            logger.error(f"Encountered a generic error checking the already installed dependencies.\nThe specific error is: {e}")
            exit(0)

        


# if __name__ == "__main__":
#     there_is_gpu = True

#     pdm = PackageDownloadManager()

#     pdm.checkDownloadedDependencies(there_is_gpu)
    