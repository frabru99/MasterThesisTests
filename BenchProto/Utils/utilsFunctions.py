from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger

import gc
import difflib
import traceback # TRYING
from json import dump
from difflib import SequenceMatcher
from pathlib import Path
from subprocess import run, DEVNULL
from json import dump

PROJECT_ROOT = Path(__file__).resolve().parent.parent

clean_caches_script_bash = "./" / PROJECT_ROOT / "Utils/scripts/cleancache.sh"


def compareModelArchitecture(model1: object, model2: object) -> None:
    """ 
    Utility function to compare two model architectures.

    Input:
        - model1: the first model object (torch model)
        - model2: the second model object (torch model)
    Output:
        - None
    """


    model1_str = str(model1).splitlines()
    model2_str = str(model2).splitlines()

    # Create a differ object
    diff = difflib.ndiff(model1_str, model2_str)

    print(f"Comparing Model 1 vs Model 2:")
    print("-" * 30)
    
    # Print only the differences
    has_diff = False
    for line in diff:
        if line.startswith('+') or line.startswith('-'):
            print(line)
            has_diff = True
    
    if not has_diff:
        print("Architectures are identical.")


def getHumanReadableValue(value: bytes, suffix: str="B") -> str:
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

def getLongestSubString(string1: str, string2: str) -> str:
    """
    Return the longest substring between two strings. 

    Input:
        - string1
        - string2
    Output: 
        - str
    """

    string_match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    longest_substring = string1[string_match.a : string_match.a + string_match.size]

    return longest_substring

def getFilenameList(directory_path: str) -> list:
    """
    Return a list of filename in a given directory_path. 

    Input: 
        - directory_path: the specified path
    Output:
        - file_name_list: list of filenames
    """

    file_name_list = []
    dir_path = Path(directory_path)

    for file_path in dir_path.iterdir():

        # Check if it is a file
        if file_path.is_file():
            file_name_list.append(file_path.name)

    if file_name_list:
        return file_name_list
    else:
        raise FileNotFoundError(f"No files were found in this directory: {directory_path}")
        exit(0)

def cleanCaches() -> None:
    """
    This function writes the '3' in /proc/sys/vm/drop_caches file in order to drop the unused pages, inodes and dentries.
    Visit man proc_sys_vm manpages for more.

    """
    try:

        result = run([str(clean_caches_script_bash)], check=True, stdout=DEVNULL)

        if result.returncode==0:
            logger.info("CACHE CLEANED FOR INDEPENDENT EXPERIMENTS")
    except ChildProcessError as e:
        logger.error(f"Cache not cleaned correctly. The next measurements could be not independent.\nThe error is: {e}")
    except Exception as e:
        logger.error(f"Encountered a generic problem cleaning the caches. The next measurements could be not independent.\nThe error is: {e}")


def subRunQueue(context, aimodel, inference_loader, config_id, queue):
    """
    Worker function to run in the subprocess.
    """
    try:  
        stats = context.run(aimodel=aimodel, input_data=inference_loader, config_id=config_id)

        queue.put({"status": "success", "data": stats})

    except Exception as e:
        logger.error(f"SubProcess CRASHED: {e}")
        logger.error(traceback.format_exc())
        queue.put({"status": "error", "message": str(e)})



def initialPrint(topic: str) -> None:
    """
    Utility function in order to print the main section of the execution in violet. 

    Input:
        - topic: main section title

    Output:
        - None
    """

    print("\n\t\t"+ '\x1b[35m' + topic + '\033[0m')


