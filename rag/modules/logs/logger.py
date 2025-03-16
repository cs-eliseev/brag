import json
from typing import Any
from loguru import logger
from rag.modules.logs.fake_logger import FakeLogger

class LoggerWrapper:
    def __init__(self, logger_enabled: bool = False) -> None:
        if logger_enabled:
            self.logger = logger
        else:
            self.logger = FakeLogger()

    def debug(self, message: str, **context: Any) -> None:
        self.logger.debug(self._concatinate_args(message, **context))

    def info(self, message: str, **context: Any) -> None:
        self.logger.info(self._concatinate_args(message, **context))

    def warning(self, message: str, **context: Any) -> None:
        self.logger.warning(self._concatinate_args(message, **context))

    def error(self, message: str, **context: Any) -> None:
        self.logger.error(self._concatinate_args(message, **context))

    def critical(self, message: str, **context: Any) -> None:
        self.logger.critical(self._concatinate_args(message, **context))

    def _concatinate_args(self, message: str, **context: Any) -> str:
        if context:
            return f"{message}\n{json.dumps(context, ensure_ascii=False)}"
        return message