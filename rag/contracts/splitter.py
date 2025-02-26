from abc import ABC, abstractmethod
from typing import List

class SplitterContract(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Разделение текста."""
        pass