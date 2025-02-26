from abc import ABC, abstractmethod
from pathlib import Path
from rag.entities.document import DocumentCollection

class FileLoaderContract(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> DocumentCollection:
        """Загрузка файлов."""
        pass