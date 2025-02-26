from abc import ABC, abstractmethod
from pathlib import Path


class IndexerContract(ABC):
    @abstractmethod
    def index(self, dataset_path: Path) -> None:
        pass