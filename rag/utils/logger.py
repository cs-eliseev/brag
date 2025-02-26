from functools import lru_cache
from rag.bootstrap.bootstrap import AppContainer
from rag.modules.logs.logger import LoggerWrapper

@lru_cache
def logger() -> LoggerWrapper:
    return AppContainer.get_instance().log()