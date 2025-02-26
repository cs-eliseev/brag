from loguru import logger
from rag.modules.logs.fake_logger import FakeLogger

class LoggerWrapper:
    def __init__(self, logger_enabled: bool = False) -> None:
        if logger_enabled:
            self.logger = logger
        else:
            self.logger = FakeLogger()

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)