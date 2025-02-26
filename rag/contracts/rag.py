from abc import ABC, abstractmethod

class RagContract(ABC):
    @abstractmethod
    def query(self, question: str) -> str:
        """Обработка запроса"""
        pass
