from abc import ABC, abstractmethod
from rag.entities.document import DocumentCollection
from rag.entities.vector_store import VectorStoreQueryParams

class RetrieverContract(ABC):
    @abstractmethod
    def search(self, params: VectorStoreQueryParams) -> DocumentCollection:
        """Ищет релевантные документы."""
        pass