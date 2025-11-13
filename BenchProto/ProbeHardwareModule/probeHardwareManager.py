from psutil import  cpu_count, cpu_percent, virtual_memory, disk_partitions, disk_usage
from platform import uname
from rich.pretty import pprint

from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG



config.dictConfig(TEST_LOGGING_CONFIG)
logger = getLogger(__name__)

#TODO: Take some measurements in order to set the REAL thresholds.
defaultHardwareEmptyMessage="N.A."
defaultMemoryTotalThreshold=4294967296 #Total Memory Required: 4Gb 
defaultMemoryUsageThreshold=2684354560 #Free Memory Required: 2.5Gb
defaultDiskUsageThreshold=2147483648 #Free Disk Required: 2Gb 
defaultDiskTotalThreshold=6442450944 #Total Disk Required: 4Gb
defaultCpuUsageThreshold=70 #CPU Usage Threshold
VALID_CHOICES = {'y','n'} #Choices for CPU Usage

class ProbeHardwareManager():

    def __init__(self):
        self.result = {}
        self.__uname = uname()
        pass

    def __printInformations(self, input: any, topic: str) -> None:
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
        physical_cpus = cpu_count(logical=False)
        cpu_usage = cpu_percent(percpu=False, interval=2)

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

        partitions = disk_partitions()

        partitions_info = {}
        for partition in partitions:
            if "home" in partition.mountpoint or "/" in partition.mountpoint:
                partitions_info["Mountpoint"] = partition.mountpoint
                partitions_info["Total Disk Space"] = self.__getHumanReadableValue(disk_usage(partition.mountpoint).total)
                partitions_info["Free Disk Space"] = self.__getHumanReadableValue(disk_usage(partition.mountpoint).free)
                if "home" in partition.mountpoint:
                    break
    
        self.__printInformations(partitions_info, "DISK USAGE INFORMATIONS")

        if disk_usage(partition.mountpoint).total <= defaultDiskTotalThreshold:
            logger.error(f"The tools requires at least {self.__getHumanReadableValue(defaultDiskTotalThreshold)} of Total Disk.")
            logger.info("EXITING...")
            exit(0)

        if  disk_usage(partition.mountpoint).free <= defaultDiskUsageThreshold:
            logger.error(f"The tools requires at least {self.__getHumanReadableValue(defaultDiskUsageThreshold)} of Free Disk.")
            logger.info("EXITING...")
            exit(0)



    def __retrieveGpuInfo(self):
        #TODO after: If there is the GPU or not in device, call choose if download gpu packages or not.

        pass


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

    

    def checkSystem(self):
        self.__retrieveSysInfo()
        self.__retrieveCpuUsage()
        self.__retrieveMemoryUsage()
        self.__retrieveDiskUsage()


if __name__=="__main__":
    logger.info("PROBING HARDWARE RESOURCES...\n")
    probe = ProbeHardwareManager()
    probe.checkSystem()



