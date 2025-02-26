from abc import ABC, abstractmethod
from langchain_core.documents import Document
from rag.entities.vector_store import VectorStoreQueryParams

class VectorStoreContract(ABC):
    @abstractmethod
    def create_db(self, documents: list[Document]):
        """Создание БД."""
        pass

    @abstractmethod
    def search(self, dto: VectorStoreQueryParams) -> list[Document]:
        """Поиск релевантных фрагментов."""
        pass