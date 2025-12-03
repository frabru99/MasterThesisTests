from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
logger = getLogger(__name__) #logger

import difflib
import gc
from difflib import SequenceMatcher
from pathlib import Path
from subprocess import run, DEVNULL

PROJECT_ROOT = Path(__file__).resolve().parent.parent

clean_caches_script_bash = "./" / PROJECT_ROOT / "Utils/scripts/cleancache.sh"


def compareModelArchitecture(model1, model2):
    # Convert models to string representations
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

def getLongestSubString(string1, string2):
    """
    Return the longest substring between two strings
    """

    string_match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    longest_substring = string1[string_match.a : string_match.a + string_match.size]

    return longest_substring

def getFilenameList(directory_path: str):
    """
    Return a list of filename in a given directory
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

def cleanCaches():
    """
    This function writes the number 3 in /proc/sys/vm/drop_caches file in order to drop the unused pages, inodes and dentries.
    Visit man proc_sys_vm manpages for more.

    """
    try:

        result = run([str(clean_caches_script_bash)z], check=True, stdout=DEVNULL)

        if result.returncode==0:
            logger.info("CACHE CLEANED...")
    except ChildProcessError as e:
        logger.error(f"Cache not cleaned correctly. The next measurements could be not independent.\nThe error is: {e}")
    except Exception as e:
        logger.error(f"Encountered a generic problem cleaning the caches. The next measurements could be not independent.\nThe error is: {e}")

# Sub Function to call a sub-process
def subRun(aimodel, inference_loader, config_id, output_file_path):
    
    try:
        stats = aimodel.runInference(inference_loader, config_id)

        print(f"Returned stats: {stats}\n")

        del aimodel
        del inference_loader

        gc.collect()
        
        with open(output_file_path, 'w') as f:
            json.dump(stats, f, indent=4)
                
        print(f"WORKER: Stats successfully written to {output_file_path}")

    except Exception as e:
        logger.error(f"SubProcess CRASHED")




