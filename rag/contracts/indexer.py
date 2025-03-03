from abc import ABC, abstractmethod
from pathlib import Path
from rag.entities.document import DocumentCollection

class IndexerContract(ABC):
    @abstractmethod
    def index_by_path(self, dataset_path: Path) -> None:
        pass

    @abstractmethod
    def index(self, documents: DocumentCollection) -> None:
        pass