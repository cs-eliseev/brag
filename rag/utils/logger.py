from functools import lru_cache
from di_autoloader.container_autoloader import ContainerAutoloader
from rag.modules.logs.logger import LoggerWrapper

@lru_cache
def logger() -> LoggerWrapper:
    return ContainerAutoloader.get_instance().log()