from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

class PlatformContext():


    def __init__(self, platform: str):
        """
        Initialize the Device Context of the system, creates the concrete strategies
        """

        self.__packageDownloadModule = None
        self.__runnerModule = None
        self.__configurationModule = None
        self.__statsModule = None


        match platform:

            case "generic":
                
                # --- Generic (ONNX) Imports ---
                from Runner.runner import RunnerModuleGeneric

                #self.__packageDownloadModule = PackageDownloadModuleGeneric()
                #self.__configurationModule = ConfigurationModuleGeneric()
                self.__runnerModule = RunnerModuleGeneric()
                #self.__statsModule = StatsModuleGeneric()

                logger.debug(f"CONTEXT INITIALIZED:")
                logger.debug(f"RUNNER MODULE: GENERIC RUNNER with {self.__runnerModule}")

            case "coral":

                # --- Coral Imports ---

                self.__packageDownloadModule = PackageDownloadModuleCoral()
                self.__configurationModule = ConfigurationModuleCoral()
                self.__runnerModule = RunnerModuleCoral()
                self.__statsModule = StatsModuleCoral() 

            case "fusion":

                # --- Fusion Imports ---

                self.__packageDownloadModule = PackageDownloadModuleFusion()
                self.__configurationModule = ConfigurationModuleFusion()
                self.__runnerModule = RunnerModuleFusion()
                self.__statsModule = StatsModuleFusion()

            
            case _:
                logger.error(f"No Match for platform")
                exit(0)



    def run(self, aimodel, input_data, config_id):
        return self.__runnerModule._runInference(aimodel=aimodel, input_data=input_data, config_id=config_id)
            
                




