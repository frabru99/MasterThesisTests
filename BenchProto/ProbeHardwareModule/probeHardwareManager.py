from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)


from psutil import  cpu_count, cpu_percent, virtual_memory, disk_partitions, disk_usage
from GPUtil import getGPUs
#from amdsmi import init_amd_smi_lib, get_gpu_device_handles
from pyamdgpuinfo import detect_gpus, get_gpu
from platform import uname
from rich.pretty import pprint

from PackageDownloadModule.packageDownloadManager import PackageDownloadManager


#TODO: Take some measurements in order to set the REAL thresholds.
defaultHardwareEmptyMessage="N.A."
defaultMemoryTotalThreshold=4294967296 #Total Memory Required: 4Gb 
defaultMemoryUsageThreshold=2684354560 #Free Memory Required: 2.5Gb
defaultDiskUsageThreshold=2147483648 #Free Disk Required: 2Gb 
defaultDiskTotalThreshold=6442450944 #Total Disk Required: 4Gb
defaultCpuUsageThreshold=70 #CPU Usage Threshold
intervalCpuUsage=2 #interval for CPU Usage Check
VALID_CHOICES = {'y','n'} #Choices for CPU Usage


class ProbeHardwareManager():

    def __init__(self):
        self.__uname = uname()
        
    def __printInformations(self, input: dict, topic: str) -> None:
        """
        Handler function to print System Informations and Usage on terminal.

        Input:
            - input: dict that contains couples key, value to print.
            - topic: the topic to print at the first line

        """
        print(topic + "\n")
        for key, value in input.items():
            if isinstance(value, dict):
                print(f"{key} ")
                for value_keys, value_number in value.items():
                    print(f"\t{value_keys}: {value_number if value_number else defaultHardwareEmptyMessage}")
                continue
            print(f"{key}: {value if value else defaultHardwareEmptyMessage}")

        print("\n")


    def __retrieveSysInfo(self) -> None:
        """
        Retrieves system informations and shows it on terminal.

        """

        uname = self.__uname
        
        sysinfo = {
            "System": uname.system if uname.system else defaultHardwareEmptyMessage,
            "Node name": uname.node if uname.node else defaultHardwareEmptyMessage,
            "Release": uname.release if uname.release else defaultHardwareEmptyMessage,
            "Version": uname.version if uname.version else defaultHardwareEmptyMessage,
            "Machine/Processor": uname.machine if uname.machine else defaultHardwareEmptyMessage,
        }

        self.__printInformations(sysinfo, "SYSTEM INFORMATIONS")



    def __retrieveCpuUsage(self) -> None:
        """
        Retrieves CPU Usage informations and shows it on terminal. If the CPU Usage for the given interval (default 2s)
        is Greater-Equal to defaultCpuUsageThreshold it'll show a warning prompt in order to continue or stop the execution 
        of the tool. 

        """
        
        
        physical_cpus = cpu_count(logical=False)
        cpu_usage = cpu_percent(percpu=False, interval=intervalCpuUsage)

        cpuinfo = {
            "Physical CPUs": physical_cpus if physical_cpus else defaultHardwareEmptyMessage,
            "CPU Usage (%)": f"{cpu_usage}%"
        }

        self.__printInformations(cpuinfo, "CPU INFORMATIONS")

        if cpu_usage >= defaultCpuUsageThreshold:
            while True:
                choice = input(f"{cpu_usage} of CPU Usage detected. Do you want to continue? (y/n): ").lower()

                if choice in VALID_CHOICES:
                    if choice == 'y':
                        break
                    else:
                        logger.info("\nEXITING...")
                        exit(0)
                else:
                    print("Invalid Input. Please enter 'y' or 'n'.")
            print("\n")

                
    def __retrieveMemoryUsage(self) -> None:
        """
        Retrieves Memory Usage informations and shows it on terminal. If the Memory Usage is Lower-Equal to 
        defaultMemoryXThreshold (on Total or Free evaluation), it'll stop the execution.

        """

        vmem = virtual_memory()
        mem_infos = [vmem.total, vmem.available, vmem.used]
        results = [self.__getHumanReadableValue(value) for value in mem_infos]

        memoryinfo = {
            "Total Memory" : results[0],
            "Memory Available": results[1],
            "Memory Used" : results[2],
            "Percent ": vmem.percent
        }

        self.__printInformations(memoryinfo, "MEMORY USAGE INFORMATIONS")
        

        if mem_infos[0] <= defaultMemoryTotalThreshold:
            logger.error(f"The tools requires at least {self.__getHumanReadableValue(defaultMemoryTotalThreshold)} of Total Memory to run smoothly.")
            logger.info("EXITING...")
            exit(0)

        if mem_infos[1] <= defaultMemoryUsageThreshold:
            logger.error(f"The tools requires at least {self.__getHumanReadableValue(defaultMemoryUsageThreshold)} of Free Memory to run smoothly.")
            logger.info("EXITING...")
            exit(0)

    


    def __retrieveDiskUsage(self) -> None:
        """
        Retrieves Disk Usage informations and shows it on terminal. If the Disk Total/Usage is Lower-Equal to 
        defaultDiskXThreshold (on Total or Free evaluation), it'll stop the execution.

        """
        
        partitions = disk_partitions()

        partitions_info = {}
        for partition in partitions:
            if "home" in partition.mountpoint or "/" in partition.mountpoint:
                partitions_info["Mountpoint"] = partition.mountpoint
                partitions_info["Total Disk Space"] = self.__getHumanReadableValue(disk_usage(partition.mountpoint).total)
                partitions_info["Free Disk Space"] = self.__getHumanReadableValue(disk_usage(partition.mountpoint).free)
                break

                # if "home" in partition.mountpoint:
                #     break
    
        self.__printInformations(partitions_info, "DISK USAGE INFORMATIONS")

        if disk_usage(partition.mountpoint).total <= defaultDiskTotalThreshold:
            logger.error(f"The tools requires at least {self.__getHumanReadableValue(defaultDiskTotalThreshold)} of Total Disk.")
            logger.info("EXITING...")
            exit(0)

        if  disk_usage(partition.mountpoint).free <= defaultDiskUsageThreshold:
            logger.error(f"The tools requires at least {self.__getHumanReadableValue(defaultDiskUsageThreshold)} of Free Disk.")
            logger.info("EXITING...")
            exit(0)


    def __checkAMDGpuAvailability(self, gpu_info: dict, there_is_gpu: bool, gpu_type: str) -> (bool,str):
        """
        Checks if there are AMD GPUs in the system. WORKS ONLY ON LINUX!

        Input:
            - gpu_info: dictionary
            - there_is_gpu: bool value
            - gpu_type: str value

        Output: 
            - there_is_gpu: bool
            - gpu_type: str value
        """

        #AMD (not ROCM)
        try:
            amd_gpus = detect_gpus()
            if amd_gpus>0:
                logger.info(f"AT LEAST ONE AMD GPU FOUND.\n")
                for i in range(amd_gpus):
                    gpu = get_gpu(i)
                    vram_usage=self.__getHumanReadableValue(gpu.query_vram_usage())
                    gpu_load = gpu.query_load()*100


                    gpu_info[f"AMD GPU {i} Name"] = gpu.name if gpu.name else f"GPU AMD {i}"
                    gpu_info[f"VRAM Usage AMD {i}"] = vram_usage if vram_usage else defaultHardwareEmptyMessage
                    gpu_info[f"GPU Load AMD {i}"] =  f"{str(gpu_load)}%" if gpu_load>=0 else defaultHardwareEmptyMessage
                    there_is_gpu = True
                    gpu_type="AMD"
        except (ValueError,Exception) as e:
            logger.warning(f"Encountered an error recognising AMD GPU/s. Maybe there aren't AMD device/s or you have some missing dependencies.\nThe specific error is: {e}.")


        return there_is_gpu, gpu_type

    def __checkNVIDIAGpuAvailability(self, gpu_info: dict, there_is_gpu: bool, gpu_type: str) -> (bool,str):
        """
        Checks if there are available NVIDIA GPUs in the system. Requires CUDA drivers.

        Input:
            - gpu_info: dictionary
            - there_is_gpu: bool value
            - gpu_type: str value

        Output: 
            - there_is_gpu: bool
            - gpu_type: str value
        """

        #NVIDIA
        try:

            gpus = getGPUs()

            if len(gpus)>0:
                for i, gpu in enumerate(gpus):
                    gpu_info[f"NVIDIA GPU {i} Name"] = gpu.name if gpu.name else "NVIDIA GPU {i}"
                    gpu_info[f"VRAM Usage NVIDIA {i}"] = gpu.memoryUsed if gpu.memoryUsed else defaultHardwareEmptyMessage
                    gpu_info[f"GPU Load NVIDIA {i}"]= gpu.load*100 if gpu.load>=0 else defaultHardwareEmptyMessage

                there_is_gpu = True
                gpu_type="NVIDIA"
                
        except (ValueError,Exception) as e:
            logger.warning(f"Encountered an error recognising NVIDIA GPU/s. Maybe there aren't NVIDA device/s or you have some missing dependencies.\nThe specific error is: {e}.\n")

        return there_is_gpu, gpu_type

    def __retrieveGpuInfo(self) -> (bool, str):
        """
        Checks for GPU infos, calling the check dedicated functions.

        Input:
            None

        Output: 
            - there_is_gpu: bool
            - gpu_type: str value
        """


        gpu_info = {}
        there_is_gpu = False
        gpu_type = None
        
        there_is_gpu, gpu_type = self.__checkNVIDIAGpuAvailability(gpu_info, there_is_gpu, gpu_type)
        there_is_gpu, gpu_type = self.__checkAMDGpuAvailability(gpu_info, there_is_gpu, gpu_type)

        if gpu_info:
            self.__printInformations(gpu_info, "GPU INFORMATIONS")

        return there_is_gpu, gpu_type



    def __getHumanReadableValue(self, value: bytes, suffix: str="B") -> str:
        """
        Scale bytes to its proper format
        e.g:
            1253656 => '1.20MB'
            1253656678 => '1.17GB'

        Input:
            - value: the value in bytes
            - suffix: the string suffix
        Output: 
            - string: the value in string format

        """
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if value < factor:
                return f"{value:.2f}{unit}{suffix}"
            value /= factor

    

    def checkSystem(self) -> (bool, str):
        """
        Checks the system characteristics, thanks to utility functions, in order to see if the target device has a
        sufficient amount of resources to execute the tool.

        """
        self.__retrieveSysInfo()
        self.__retrieveCpuUsage()
        self.__retrieveMemoryUsage()
        self.__retrieveDiskUsage()
        there_is_gpu, gpu_type = self.__retrieveGpuInfo()


        if not there_is_gpu:
            logger.info("GPU INSTANCES NOT FOUND.\n")
        else:
            logger.info(f"GPU {gpu_type} FOUND!\n")

        return there_is_gpu, gpu_type



if __name__=="__main__":
    logger.info("PROBING HARDWARE RESOURCES...\n")
    probe = ProbeHardwareManager()
    there_is_gpu, gpu_type = probe.checkSystem()

    pdm = PackageDownloadManager()

    pdm.checkDownloadedDependencies(there_is_gpu)





