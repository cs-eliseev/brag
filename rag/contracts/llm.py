from abc import ABC, abstractmethod

class LLMContract(ABC):
    @abstractmethod
    def generate(self, context: str) -> str:
        """Генерирует ответ."""
        pass