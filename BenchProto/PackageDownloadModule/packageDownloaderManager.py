from logging import config, getLogger
from logging_config import TEST_LOGGING_CONFIG
from rich.pretty import pprint
from pathlib import Path



logger = getLogger(__name__) #logger
config.dictConfig(TEST_LOGGING_CONFIG) #logger config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
requirementsFileDirectoryPath = str(PROJECT_ROOT / "PackagDownloadModule" / "requirementsFileDirectory")
#...other paths


class PackageDownloadManager:
    pass

