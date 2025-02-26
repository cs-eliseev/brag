from abc import ABC, abstractmethod

class PromptContract(ABC):
    @abstractmethod
    def render(self, **kwargs) -> str:
        """Генерирует промпт"""
        pass